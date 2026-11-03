from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


# Operational defaults decoded from the parent LabVIEW front panel.  The
# standalone subVI contains a smaller three-row test default, but it is
# overridden by these parent-VI values during normal execution.
LABVIEW_MAIN_ROWS: tuple[int, ...] = (
    437,
    392,
    349,
    339,
    317,
    288,
    241,
    237,
    230,
    215,
    195,
    132,
    99,
    89,
    78,
    69,
    54,
    30,
    3,
)
LABVIEW_MAIN_STEPS = 2
LABVIEW_SUBVI_EXAMPLE_ROWS: tuple[int, ...] = (220, 288, 396)
LABVIEW_SUBVI_EXAMPLE_STEPS = 1

DIRECT_Y = "direct_y"
LABVIEW_MIRRORED = "labview_mirrored"
COORDINATE_MODES = (DIRECT_Y, LABVIEW_MIRRORED)


@dataclass(frozen=True, slots=True)
class RowInterpolationMap:
    """A reproducible set of detector rows requiring interpolation.

    ``rows`` are always stored as zero-based values in the declared coordinate
    convention.  ``direct_y`` means canonical Python ``cube[y, x, t]`` rows.
    ``labview_mirrored`` preserves the original detector-row IDs and resolves
    them as ``height - 1 - row`` (the supplied 512-row VI used ``511-row``).
    """

    rows: tuple[int, ...] = ()
    radius: int = LABVIEW_MAIN_STEPS
    coordinate_mode: str = DIRECT_Y
    name: str = "Custom detector-row map"
    source_path: str = ""
    sha256: str = ""
    declared_index_base: int = 0
    expected_source_height: int | None = None
    algorithm: str = "safe_linear_v1"

    def normalized(self) -> "RowInterpolationMap":
        mode = str(self.coordinate_mode).strip().lower()
        if mode not in COORDINATE_MODES:
            raise ValueError(
                f"Unknown row coordinate mode {self.coordinate_mode!r}; "
                f"expected one of {', '.join(COORDINATE_MODES)}."
            )
        if int(self.declared_index_base) not in (0, 1):
            raise ValueError("Row-map index base must be 0 or 1.")
        radius = int(self.radius)
        if radius < 1:
            raise ValueError("Interpolation donor radius must be at least 1.")
        clean: list[int] = []
        for value in self.rows:
            numeric = float(value)
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"Row identifiers must be finite integers; received {value!r}.")
            row = int(numeric)
            if row < 0:
                raise ValueError(f"Row identifiers must be non-negative; received {row}.")
            clean.append(row)
        expected_height = None if self.expected_source_height in (None, 0) else int(self.expected_source_height)
        if expected_height is not None and expected_height < 1:
            raise ValueError("Expected source height must be positive when specified.")
        return replace(
            self,
            rows=tuple(sorted(set(clean))),
            radius=radius,
            coordinate_mode=mode,
            name=str(self.name).strip() or "Custom detector-row map",
            source_path=str(self.source_path).strip(),
            sha256=str(self.sha256).strip().lower(),
            declared_index_base=int(self.declared_index_base),
            expected_source_height=expected_height,
            algorithm="safe_linear_v1",
        )

    def resolve_rows(self, height: int) -> tuple[int, ...]:
        mapping = self.normalized()
        height = int(height)
        if height < 1:
            raise ValueError("The source image must contain at least one row.")
        if mapping.expected_source_height is not None and height != mapping.expected_source_height:
            raise ValueError(
                f"Row map {mapping.name!r} requires a {mapping.expected_source_height}-row source image; "
                f"the current dataset has {height} rows. The map is not rescaled or remirrored."
            )
        invalid = [row for row in mapping.rows if row >= height]
        if invalid:
            suffix = "" if len(invalid) <= 6 else " ..."
            shown = ", ".join(str(row) for row in invalid[:6])
            raise ValueError(
                f"Row map contains detector row(s) outside 0..{height - 1}: {shown}{suffix}. "
                "Choose a map for this detector geometry; rows are never clipped or wrapped."
            )
        if mapping.coordinate_mode == LABVIEW_MIRRORED:
            return tuple(sorted(height - 1 - row for row in mapping.rows))
        return mapping.rows

    def to_dict(self) -> dict[str, Any]:
        mapping = self.normalized()
        values = asdict(mapping)
        values.update(
            {
                "schema_version": 1,
                "axis": "source_y",
                "rows": list(mapping.rows),
                "coordinate_description": (
                    "canonical cube[y,x,time]"
                    if mapping.coordinate_mode == DIRECT_Y
                    else "original LabVIEW detector row; canonical y = height - 1 - row"
                ),
            }
        )
        if set(mapping.rows) == set(LABVIEW_MAIN_ROWS) and mapping.coordinate_mode == LABVIEW_MIRRORED:
            values["labview_main_control_saved_order"] = list(LABVIEW_MAIN_ROWS)
        return values


def labview_main_preset() -> RowInterpolationMap:
    return RowInterpolationMap(
        rows=LABVIEW_MAIN_ROWS,
        radius=LABVIEW_MAIN_STEPS,
        coordinate_mode=LABVIEW_MIRRORED,
        name="Original LabVIEW main-VI detector map",
        expected_source_height=512,
    ).normalized()


def _tokens_without_comments(text: str) -> list[str]:
    lines = [line.split("#", 1)[0] for line in str(text).splitlines()]
    return [token for token in re.split(r"[\s,;]+", "\n".join(lines).strip()) if token]


def parse_row_spec(text: str, index_base: int = 0) -> tuple[int, ...]:
    """Parse rows such as ``"3, 8, 10-12"`` into zero-based identifiers."""

    index_base = int(index_base)
    if index_base not in (0, 1):
        raise ValueError("Row-map index base must be 0 or 1.")
    values: list[int] = []
    for token in _tokens_without_comments(text):
        match = re.fullmatch(r"([+-]?\d+)(?:-([+-]?\d+))?", token)
        if match is None:
            raise ValueError(
                f"Invalid row-map token {token!r}. Use integers or increasing ranges such as 10-14."
            )
        start = int(match.group(1))
        stop_text = match.group(2)
        stop = start if stop_text is None else int(stop_text)
        if stop < start:
            raise ValueError(f"Descending row range {token!r} is not supported.")
        for entered in range(start, stop + 1):
            row = entered - index_base
            if row < 0:
                raise ValueError(
                    f"Row {entered} is invalid for {index_base}-based indexing; canonical rows cannot be negative."
                )
            values.append(row)
    return tuple(sorted(set(values)))


def format_row_spec(rows: Iterable[int]) -> str:
    """Format zero-based rows compactly, using increasing ranges."""

    clean = sorted(set(int(value) for value in rows))
    if not clean:
        return ""
    ranges: list[str] = []
    start = previous = clean[0]
    for value in clean[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_row_map(
    path: str | Path,
    *,
    default_index_base: int = 0,
    default_coordinate_mode: str = DIRECT_Y,
) -> RowInterpolationMap:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("A row-map JSON file must contain an object.")
        if payload.get("axis", "source_y") != "source_y":
            raise ValueError("This row map does not describe the source-y axis.")
        # Python row-map JSON is always serialized with zero-based numeric rows.
        # `declared_index_base` is provenance about the user's original entry,
        # not an instruction to subtract again while reloading the saved file.
        index_base = int(payload.get("index_base", 0))
        declared_index_base = int(payload.get("declared_index_base", index_base))
        entered_rows = payload.get("rows", ())
        if not isinstance(entered_rows, (list, tuple)):
            raise ValueError("The JSON 'rows' field must be an array of integers.")
        rows = parse_row_spec(" ".join(str(value) for value in entered_rows), index_base=index_base)
        expected_height_value = payload.get("expected_source_height")
        mapping = RowInterpolationMap(
            rows=rows,
            radius=int(payload.get("radius", payload.get("steps", LABVIEW_MAIN_STEPS))),
            coordinate_mode=str(payload.get("coordinate_mode", default_coordinate_mode)),
            name=str(payload.get("name", source.stem)),
            source_path=str(source.resolve()),
            sha256=_file_digest(source),
            declared_index_base=declared_index_base,
            expected_source_height=(
                None if expected_height_value in (None, "", 0, "0") else int(expected_height_value)
            ),
        )
        return mapping.normalized()

    text = source.read_text(encoding="utf-8-sig")
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and "=" in stripped:
            key, value = stripped[1:].split("=", 1)
            metadata[key.strip().lower()] = value.strip()
        else:
            data_lines.append(line)
    text = "\n".join(data_lines)
    if suffix == ".csv":
        rows_from_csv: list[str] = []
        for row in csv.reader(text.splitlines()):
            if not row:
                continue
            value = row[0].strip()
            if not value or value.lower() in {"row", "rows", "source_row", "detector_row"}:
                continue
            rows_from_csv.append(value)
        text = "\n".join(rows_from_csv)
    stored_index_base = int(metadata.get("index_base", default_index_base))
    rows = parse_row_spec(text, index_base=stored_index_base)
    return RowInterpolationMap(
        rows=rows,
        radius=int(metadata.get("radius", LABVIEW_MAIN_STEPS)),
        coordinate_mode=metadata.get("coordinate_mode", default_coordinate_mode),
        name=metadata.get("name", source.stem),
        source_path=str(source.resolve()),
        sha256=_file_digest(source),
        declared_index_base=int(metadata.get("declared_index_base", stored_index_base)),
        expected_source_height=(
            None
            if metadata.get("expected_source_height", "").lower() in {"", "none", "null", "0"}
            else int(metadata["expected_source_height"])
        ),
    ).normalized()


def save_row_map(path: str | Path, mapping: RowInterpolationMap) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = mapping.normalized()
    suffix = destination.suffix.lower()
    if suffix == ".json":
        payload = normalized.to_dict()
        # Values stored by the Python model are already zero-based.
        payload["index_base"] = 0
        payload["declared_index_base"] = normalized.declared_index_base
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    elif suffix == ".csv":
        header = (
            "# flim_studio_row_map_schema=1\n"
            f"# coordinate_mode={normalized.coordinate_mode}\n"
            f"# radius={normalized.radius}\n"
            f"# declared_index_base={normalized.declared_index_base}\n"
            f"# expected_source_height={normalized.expected_source_height or ''}\n"
            f"# name={normalized.name}\n"
            "row\n"
        )
        destination.write_text(header + "\n".join(str(row) for row in normalized.rows) + "\n", encoding="utf-8")
    else:
        header = (
            "# flim_studio_row_map_schema=1\n"
            f"# coordinate_mode={normalized.coordinate_mode}\n"
            f"# radius={normalized.radius}\n"
            f"# declared_index_base={normalized.declared_index_base}\n"
            f"# expected_source_height={normalized.expected_source_height or ''}\n"
            f"# name={normalized.name}\n"
        )
        destination.write_text(header + "\n".join(str(row) for row in normalized.rows) + "\n", encoding="utf-8")
    return destination


def interpolate_mapped_rows(
    cube: np.ndarray,
    mapping: RowInterpolationMap,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace complete mapped row traces from unchanged neighbouring rows.

    The safe algorithm treats each contiguous bad-row run simultaneously.  Up
    to ``radius`` unmarked donor rows are averaged on each side, and synthetic
    rows are linearly blended between those two side estimates.  At an image
    edge the available side is copied.  This avoids the supplied VI's
    order-dependent, asymmetric six-sample stencil while retaining its full-row
    correction semantics.
    """

    source = np.asarray(cube)
    if source.ndim != 3:
        raise ValueError(f"Row interpolation requires a 3-D (y,x,time) cube; received {source.shape}.")
    height, width, time_bins = (int(value) for value in source.shape)
    normalized = mapping.normalized()
    target_rows = normalized.resolve_rows(height)
    if not target_rows:
        raise ValueError("Row interpolation is enabled but the detector-row map is empty.")
    if len(target_rows) >= height:
        raise ValueError("The row map marks every source row; no unmodified donor row remains.")

    original = np.asarray(source, dtype=np.float32)
    output = original.copy()
    bad = np.zeros(height, dtype=bool)
    bad[list(target_rows)] = True
    runs: list[tuple[int, int]] = []
    start = previous = target_rows[0]
    for row in target_rows[1:]:
        if row == previous + 1:
            previous = row
            continue
        runs.append((start, previous))
        start = previous = row
    runs.append((start, previous))

    run_details: list[dict[str, Any]] = []
    radius = normalized.radius
    for run_start, run_stop in runs:
        upper_candidates = [row for row in range(run_start - 1, -1, -1) if not bad[row]][:radius]
        lower_candidates = [row for row in range(run_stop + 1, height) if not bad[row]][:radius]
        upper_rows = tuple(sorted(upper_candidates))
        lower_rows = tuple(lower_candidates)
        upper = np.mean(original[list(upper_rows)], axis=0, dtype=np.float64) if upper_rows else None
        lower = np.mean(original[list(lower_rows)], axis=0, dtype=np.float64) if lower_rows else None
        upper_position = float(np.mean(upper_rows)) if upper_rows else None
        lower_position = float(np.mean(lower_rows)) if lower_rows else None
        if upper is None and lower is None:
            raise ValueError(f"No unmarked donor rows are available for mapped run {run_start}-{run_stop}.")
        mode = "two_sided"
        run_length = run_stop - run_start + 1
        for offset, row in enumerate(range(run_start, run_stop + 1), start=1):
            if upper is None:
                estimate = lower
                mode = "lower_edge_copy"
            elif lower is None:
                estimate = upper
                mode = "upper_edge_copy"
            else:
                fraction = (row - upper_position) / (lower_position - upper_position)
                estimate = (1.0 - fraction) * upper + fraction * lower
            output[row] = np.asarray(estimate, dtype=np.float32)
        run_details.append(
            {
                "target_start_y": int(run_start),
                "target_stop_y_inclusive": int(run_stop),
                "upper_donor_rows_y": list(upper_rows),
                "lower_donor_rows_y": list(lower_rows),
                "mode": mode,
            }
        )

    diagnostics = {
        "algorithm": "safe_linear_v1",
        "legacy_difference": (
            "The original VI used an asymmetric, order-dependent six-sample stencil and ignored "
            "its steps control; this implementation uses simultaneous full-row donor interpolation."
        ),
        "coordinate_mode": normalized.coordinate_mode,
        "entered_rows_zero_based": list(normalized.rows),
        "canonical_target_rows_y": list(target_rows),
        "radius": int(radius),
        "source_shape_yxt": [height, width, time_bins],
        "corrected_rows": len(target_rows),
        "corrected_spatial_pixels": len(target_rows) * width,
        "corrected_traces": len(target_rows) * width,
        "runs": run_details,
        "calculation_dtype": "float32",
    }
    return output, diagnostics


def interpolate_mapped_columns(
    cube: np.ndarray,
    columns: Iterable[int],
    radius: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace complete detector-column traces from unchanged neighbouring columns.

    Columns are canonical, zero-based image ``x`` positions.  Transposing the
    two spatial axes lets the same simultaneous, non-cascading interpolation
    contract used for detector rows handle isolated columns and consecutive
    column runs without duplicating the scientific algorithm.
    """

    source = np.asarray(cube)
    if source.ndim != 3:
        raise ValueError(
            f"Column interpolation requires a 3-D (y,x,time) cube; received {source.shape}."
        )
    clean: list[int] = []
    for value in columns:
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"Interpolation columns must be finite integers; received {value!r}.")
        column = int(numeric)
        if column < 0 or column >= source.shape[1]:
            raise ValueError(
                f"Interpolation column {column} is outside the valid range 0..{source.shape[1] - 1}."
            )
        clean.append(column)
    clean_columns = tuple(sorted(set(clean)))
    if not clean_columns:
        raise ValueError("Column interpolation is enabled but the detector-column list is empty.")
    mapping = RowInterpolationMap(
        rows=clean_columns,
        radius=max(1, int(radius)),
        coordinate_mode=DIRECT_Y,
        name="Defined detector-column map",
    )
    transposed, details = interpolate_mapped_rows(source.transpose(1, 0, 2), mapping)
    renamed = {
        "algorithm": details["algorithm"],
        "coordinate_mode": "direct_x",
        "entered_columns_zero_based": clean_columns,
        "canonical_target_columns_x": details["canonical_target_rows_y"],
        "radius": details["radius"],
        "source_shape_yxt": list(source.shape),
        "corrected_columns": details["corrected_rows"],
        "corrected_spatial_pixels": details["corrected_spatial_pixels"],
        "corrected_traces": details["corrected_traces"],
        "runs": [
            {
                "target_start_x": run["target_start_y"],
                "target_stop_x_inclusive": run["target_stop_y_inclusive"],
                "left_donor_columns_x": run["upper_donor_rows_y"],
                "right_donor_columns_x": run["lower_donor_rows_y"],
                "mode": run["mode"].replace("upper", "left").replace("lower", "right"),
            }
            for run in details["runs"]
        ],
        "calculation_dtype": details["calculation_dtype"],
        "statistical_note": (
            "Interpolated column traces are synthetic floating-point estimates and are not "
            "independent Poisson observations."
        ),
    }
    return transposed.transpose(1, 0, 2), renamed


def column_interpolation_coverage(
    source_height: int,
    source_width: int,
    target_columns_x: Iterable[int],
    block_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-column mask and result-cell corrected-column fractions."""

    source_height = int(source_height)
    source_width = int(source_width)
    factor = max(1, int(block_width))
    mask = np.zeros(source_width, dtype=bool)
    columns = tuple(sorted(set(int(column) for column in target_columns_x)))
    if columns:
        if columns[0] < 0 or columns[-1] >= source_width:
            raise ValueError(f"Interpolation columns must lie in 0..{source_width - 1}.")
        mask[list(columns)] = True
    out_h = source_height // factor
    out_w = source_width // factor
    if out_h < 1 or out_w < 1:
        raise ValueError(f"Spatial bin {factor} is larger than source shape {(source_height, source_width)}.")
    column_fraction = mask[: out_w * factor].reshape(out_w, factor).mean(axis=1, dtype=np.float64)
    coverage = np.broadcast_to(column_fraction[None, :], (out_h, out_w)).astype(np.float32, copy=True)
    return mask, coverage


def interpolation_coverage(
    source_height: int,
    source_width: int,
    target_rows_y: Iterable[int],
    spatial_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a source-row mask and per-result-cell corrected-row fraction."""

    source_height = int(source_height)
    source_width = int(source_width)
    factor = max(1, int(spatial_bin))
    mask = np.zeros(source_height, dtype=bool)
    rows = tuple(sorted(set(int(row) for row in target_rows_y)))
    if rows:
        mask[list(rows)] = True
    out_h = source_height // factor
    out_w = source_width // factor
    if out_h < 1 or out_w < 1:
        raise ValueError(f"Spatial bin {factor} is larger than source shape {(source_height, source_width)}.")
    row_fraction = mask[: out_h * factor].reshape(out_h, factor).mean(axis=1, dtype=np.float64)
    coverage = np.broadcast_to(row_fraction[:, None], (out_h, out_w)).astype(np.float32, copy=True)
    return mask, coverage
