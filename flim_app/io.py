from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence

from .models import FLIMDataset


SUPPORTED_EXTENSIONS = (".ics", ".ids", ".npy", ".npz", ".tif", ".tiff", ".csv", ".txt")


def _canonical_cube(array: np.ndarray, time_axis: int | None = None) -> np.ndarray:
    data = np.asarray(array)
    data = np.squeeze(data)
    if data.ndim == 1:
        return data.reshape(1, 1, -1)
    if data.ndim == 2:
        return data.reshape(1, data.shape[0], data.shape[1])
    if data.ndim != 3:
        raise ValueError(f"Expected a 1-D, 2-D, or 3-D numeric array; received {data.shape}.")
    if time_axis is None:
        candidates = [i for i, size in enumerate(data.shape) if size <= 512]
        if data.shape[0] <= 256 and data.shape[1] == data.shape[2] and data.shape[0] != data.shape[1]:
            time_axis = 0
        elif data.shape[-1] <= 512:
            time_axis = 2
        elif data.shape[0] <= 512:
            time_axis = 0
        elif candidates:
            time_axis = min(candidates, key=lambda i: data.shape[i])
        else:
            time_axis = 2
    data = np.moveaxis(data, int(time_axis), -1)
    return np.asarray(data)


def _split_ics_header(payload: bytes) -> tuple[str, int]:
    match = re.search(rb"(?:^|\r?\n)end[\t ]*(?:\r?\n)", payload, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("The ICS header has no terminating 'end' line.")
    return payload[: match.end()].decode("latin-1", errors="replace"), match.end()


def _parse_ics_lines(header: str) -> tuple[dict[str, list[str]], dict[str, Any]]:
    fields: dict[str, list[str]] = {}
    metadata: dict[str, Any] = {"format": "ICS"}
    for raw_line in header.splitlines():
        line = raw_line.strip("\x00\r\n ")
        if not line or line.lower() == "end":
            continue
        parts = [part for part in re.split(r"\t+", line) if part != ""]
        if len(parts) == 1:
            parts = line.split()
        lower = [part.strip().lower() for part in parts]
        key: str | None = None
        values: list[str] = []
        if lower[:2] == ["layout", "order"]:
            key, values = "layout order", parts[2:]
        elif lower[:2] == ["layout", "sizes"]:
            key, values = "layout sizes", parts[2:]
        elif lower[:2] == ["representation", "format"]:
            key, values = "representation format", parts[2:]
        elif lower[:2] == ["representation", "sign"]:
            key, values = "representation sign", parts[2:]
        elif lower[:2] == ["representation", "compression"]:
            key, values = "representation compression", parts[2:]
        elif lower[:2] == ["representation", "byte_order"]:
            key, values = "representation byte_order", parts[2:]
        elif lower[:2] == ["representation", "byte order"]:
            key, values = "representation byte_order", parts[2:]
        elif lower[:2] == ["parameter", "scale"]:
            key, values = "parameter scale", parts[2:]
        elif lower[:2] == ["parameter", "units"]:
            key, values = "parameter units", parts[2:]
        elif lower[:3] == ["history", "image", "extents"]:
            key, values = "history image extents", parts[3:]
        elif lower[:3] == ["history", "image", "units"]:
            key, values = "history image units", parts[3:]
        elif lower[:3] == ["history", "image", "dimensions"]:
            key, values = "history image dimensions", parts[3:]
        elif lower[:2] == ["history", "dimensions"]:
            key, values = "history dimensions", parts[2:]
        elif lower[:2] == ["history", "labels"]:
            key, values = "history labels", parts[2:]
        elif lower[:2] == ["history", "extents"]:
            # Some exporters (e.g. Leica Interrogator Desktop) drop the
            # "image" token; same meaning as "history image extents".
            key, values = "history image extents", parts[2:]
        elif lower[:2] == ["history", "units"]:
            key, values = "history image units", parts[2:]
        elif lower[:2] == ["history", "type"]:
            key, values = "history type", parts[2:]
        elif lower[:4] == ["history", "laser", "rep", "rate"]:
            key, values = "history laser rep rate", parts[4:]
        elif lower[:5] == ["history", "scan", "no.", "of", "loops"]:
            key, values = "history scan loops", parts[5:]
        elif lower[:5] == ["history", "scan", "number", "of", "loops"]:
            key, values = "history scan loops", parts[5:]
        elif lower[:4] in (
            ["history", "scan", "acq", "time"],
            ["history", "scan", "acquisition", "time"],
        ):
            key, values = "history scan acquisition time", parts[4:]
        elif lower and lower[0] in {
            "history scan acq time",
            "history scan acquisition time",
        }:
            key, values = "history scan acquisition time", parts[1:]
        elif lower[:2] == ["ics_version", "2.0"] or lower[:1] == ["ics_version"]:
            key, values = "ics_version", parts[1:]
        if key is not None:
            # Most exporters tab-separate every value; some (e.g. Leica
            # Interrogator Desktop) tab-separate only the field name from
            # its value list, then space-separate the values themselves
            # within that one tab-delimited chunk. Flatten so callers can
            # always index values positionally either way -- a no-op for
            # already-atomic (no embedded space) tokens.
            fields[key] = [token for chunk in values for token in chunk.split()] or values
        if len(parts) >= 2:
            metadata[" / ".join(parts[:-1])] = parts[-1]
    return fields, metadata


def _ics_dtype(fields: dict[str, list[str]], bit_depth: int) -> np.dtype:
    fmt = " ".join(fields.get("representation format", ["integer"])).lower()
    sign = " ".join(fields.get("representation sign", ["unsigned"])).lower()
    if "real" in fmt or "float" in fmt:
        code = "f4" if bit_depth <= 32 else "f8"
    elif "signed" in sign and "unsigned" not in sign:
        code = f"i{max(1, bit_depth // 8)}"
    else:
        code = f"u{max(1, bit_depth // 8)}"
    byte_order = [int(v) for v in fields.get("representation byte_order", []) if str(v).isdigit()]
    endian = "<" if len(byte_order) < 2 or byte_order[0] < byte_order[-1] else ">"
    return np.dtype(endian + code)


def _unit_to_ns(value: float, unit: str) -> float:
    unit = unit.lower().strip()
    if unit in {"s", "sec", "second", "seconds"}:
        return value * 1e9
    if unit in {"ms", "millisecond", "milliseconds"}:
        return value * 1e6
    if unit in {"us", "µs", "microsecond", "microseconds"}:
        return value * 1e3
    if unit in {"ps", "picosecond", "picoseconds"}:
        return value / 1e3
    return value


def _first_finite_float(fields: dict[str, list[str]], key: str) -> float | None:
    for value in fields.get(key, []):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


_TIME_AXIS_LABELS = {"t", "time", "lifetime"}


def _axes_from_history_labels(
    axes: list[str], sizes: list[int], fields: dict[str, list[str]]
) -> tuple[list[str], bool]:
    """Take each axis's true meaning from the ICS ``history labels`` line.

    The generic ``layout order`` labels can be wrong for a time-resolved
    instrument: the Swarm system, for example, writes ``layout order bits x
    y z`` for what is really time, x, y (it names its time axis "x") and
    tags the file ``history type Time Resolved``, which would send the
    generic fallback to the wrong axis. Its header also carries
    ``history labels t x y`` and ``history dimensions 256 1024 1024``,
    listing the same axes, in the same fastest-varying-first order as
    ``layout sizes``, under their real names.

    The labels are only trusted when they line up with the layout: same
    count, no repeats, a time axis among them, and ``history dimensions``
    present and equal to the layout sizes. Anything less keeps the layout's
    own labels, so a file whose history block is incomplete or inconsistent
    is read exactly as before.
    """

    labels = [value.lower() for value in fields.get("history labels", [])]
    if len(labels) != len(axes) or len(set(labels)) != len(labels):
        return axes, False
    if not any(label in _TIME_AXIS_LABELS for label in labels):
        return axes, False
    dimensions = fields.get("history dimensions") or fields.get("history image dimensions") or []
    try:
        dimension_values = [int(float(value)) for value in dimensions]
    except ValueError:
        return axes, False
    if dimension_values != [int(size) for size in sizes]:
        return axes, False
    return labels, True


def _time_axis_index(axes: list[str], fields: dict[str, list[str]] | None = None) -> int:
    """Locate the time axis within an ICS ``layout order`` axis list.

    Most files label it explicitly (``t``/``time``/``lifetime``). Some
    exporters (e.g. Leica Interrogator Desktop) instead reuse the generic
    ``z`` axis label for their time-gate dimension -- that's only treated
    as time when the header's ``history type`` says so, so a genuine
    spatial z-stack file isn't misread as time-resolved data.
    """

    explicit = next((i for i, axis in enumerate(axes) if axis in {"t", "time", "lifetime"}), None)
    if explicit is not None:
        return explicit
    if fields is not None:
        history_type = " ".join(fields.get("history type", [])).lower()
        if "time" in history_type:
            z_index = next((i for i, axis in enumerate(axes) if axis == "z"), None)
            if z_index is not None:
                return z_index
    return 0


def _ics_temporal_extent_ns(
    fields: dict[str, list[str]], axes: list[str]
) -> float | None:
    """Return the full recorded temporal extent from the ICS history block."""

    time_index = _time_axis_index(axes, fields)
    extents = fields.get("history image extents", [])
    units = fields.get("history image units", [])
    offset = 1 if len(extents) == len(axes) + 1 else 0
    index = time_index + offset
    try:
        extent = float(extents[index])
        unit_index = time_index + (1 if len(units) == len(axes) + 1 else 0)
        unit = units[unit_index] if len(units) > unit_index else "s"
        extent_ns = _unit_to_ns(extent, unit)
    except (IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(extent_ns) or extent_ns <= 0:
        return None
    return float(extent_ns)


def _ics_timing(fields: dict[str, list[str]], axes: list[str], sizes: list[int]) -> tuple[float, float]:
    time_index = _time_axis_index(axes, fields)
    nt = sizes[time_index]
    rep_rate_hz = 80e6
    rep_values = fields.get("history laser rep rate", [])
    if rep_values:
        try:
            rep_rate_hz = float(rep_values[0])
        except ValueError:
            pass

    # The source microscope writes the full temporal extent in the history
    # block. This is more reliable than its occasionally inconsistent unit
    # label on ``parameter scale``.
    temporal_extent_ns = _ics_temporal_extent_ns(fields, axes)
    if temporal_extent_ns is not None:
        dt_ns = temporal_extent_ns / max(nt, 1)
        if 1e-5 <= dt_ns <= 100.0:
            return dt_ns, rep_rate_hz / 1e6

    scales = fields.get("parameter scale", [])
    units = fields.get("parameter units", [])
    # Scale/unit arrays may include a leading entry for the bit axis.
    offset = 1 if len(scales) == len(axes) + 1 else 0
    try:
        scale = float(scales[time_index + offset])
        unit = units[time_index + offset] if len(units) > time_index + offset else "ns"
        dt_ns = _unit_to_ns(scale, unit)
        if unit.lower().startswith("n") and dt_ns < 1e-6:
            dt_ns = scale * 1e9
        if 1e-5 <= dt_ns <= 100.0:
            return dt_ns, rep_rate_hz / 1e6
    except (ValueError, IndexError):
        pass
    return 1e3 / (rep_rate_hz / 1e6) / max(nt, 1), rep_rate_hz / 1e6


def load_ics(path: str | Path) -> FLIMDataset:
    source = Path(path)
    header_path = source.with_suffix(".ics") if source.suffix.lower() == ".ids" else source
    payload = header_path.read_bytes()
    header, data_offset = _split_ics_header(payload)
    fields, metadata = _parse_ics_lines(header)
    raw_axes = [value.lower() for value in fields.get("layout order", [])]
    raw_sizes = [int(float(value)) for value in fields.get("layout sizes", [])]
    if not raw_axes or len(raw_axes) != len(raw_sizes):
        raise ValueError("ICS layout order/sizes are missing or inconsistent.")

    if raw_axes[0] in {"bits", "bit"}:
        bit_depth = raw_sizes[0]
        axes, sizes = raw_axes[1:], raw_sizes[1:]
    else:
        bit_depth = 16
        axes, sizes = raw_axes, raw_sizes
    layout_axes_as_written = list(axes)
    axes, labels_from_history = _axes_from_history_labels(axes, sizes, fields)
    dtype = _ics_dtype(fields, bit_depth)
    expected = int(np.prod(sizes))

    version = " ".join(fields.get("ics_version", ["2.0"]))
    data_path = header_path.with_suffix(".ids")
    embedded_bytes = payload[data_offset:]
    compression = " ".join(fields.get("representation compression", ["uncompressed"])).lower()
    is_compressed = "gzip" in compression or embedded_bytes[:2] == b"\x1f\x8b"

    # ICS v1 always keeps its data in a separate .ids file; so does a v2
    # header with nothing embedded after it. A v2 header *with* embedded
    # bytes needs no companion -- but only compare the embedded byte count
    # against the uncompressed expected size when it's actually
    # uncompressed. Compressed data is legitimately much smaller than its
    # decompressed size, so that comparison would otherwise conclude
    # (wrongly) that a companion file is needed even though the payload is
    # complete right there in the .ics -- this is what a gzip-compressed,
    # fully self-contained export (e.g. Leica Interrogator Desktop) hits.
    needs_companion = version.startswith("1") or len(embedded_bytes) == 0
    if not needs_companion and not is_compressed:
        needs_companion = len(embedded_bytes) < expected * dtype.itemsize

    if needs_companion:
        if not data_path.exists():
            raise FileNotFoundError(f"ICS data companion not found: {data_path}")
        data_bytes = data_path.read_bytes()
    else:
        data_bytes = embedded_bytes
    if "gzip" in compression or data_bytes[:2] == b"\x1f\x8b":
        try:
            data_bytes = gzip.decompress(data_bytes)
        except gzip.BadGzipFile as exc:
            raise ValueError(f"ICS payload is corrupt or truncated gzip data: {exc}") from exc
    needed = expected * dtype.itemsize
    if len(data_bytes) < needed:
        raise ValueError(f"ICS payload is truncated: expected {needed} bytes, found {len(data_bytes)}.")
    values = np.frombuffer(data_bytes[:needed], dtype=dtype, count=expected)
    array = values.reshape(tuple(sizes), order="F")

    time_axis = _time_axis_index(axes, fields)
    array = np.moveaxis(array, time_axis, -1)
    remaining_axes = axes[:time_axis] + axes[time_axis + 1 :]
    if array.ndim > 3:
        # Select the first z/channel/position plane while preserving x, y, t.
        while array.ndim > 3:
            array = array.take(indices=0, axis=0)
            if remaining_axes:
                remaining_axes.pop(0)
    if array.ndim == 2:
        array = array[None, ...]
    if len(remaining_axes) >= 2 and remaining_axes[:2] == ["x", "y"]:
        array = np.swapaxes(array, 0, 1)
    dt_ns, rep_rate_mhz = _ics_timing(fields, axes, sizes)
    time_bins = int(sizes[time_axis])
    temporal_extent_ns = _ics_temporal_extent_ns(fields, axes)
    if temporal_extent_ns is None:
        temporal_extent_ns = float(dt_ns * time_bins)
    scan_loops_value = _first_finite_float(fields, "history scan loops")
    scan_loops = None
    if scan_loops_value is not None and scan_loops_value >= 0 and scan_loops_value.is_integer():
        scan_loops = int(scan_loops_value)
    acquisition_time_s = _first_finite_float(fields, "history scan acquisition time")
    if acquisition_time_s is not None and acquisition_time_s < 0:
        acquisition_time_s = None
    canonical_shape = [int(value) for value in np.asarray(array).shape]
    source_shape = [int(value) for value in sizes]
    metadata.update(
        {
            "ics_version": version,
            "layout_axes": axes,
            "layout_axes_as_written": layout_axes_as_written,
            "axis_labels_from_history": labels_from_history,
            "layout_sizes": sizes,
            "source_axes": list(axes),
            "source_shape": source_shape,
            "source_shape_by_axis": {
                str(axis): int(size) for axis, size in zip(axes, sizes)
            },
            "canonical_axes": ["y", "x", "t"],
            "canonical_shape_yxt": canonical_shape,
            "time_bins": time_bins,
            "temporal_extent_ns": float(temporal_extent_ns),
            "dt_ns": float(dt_ns),
            "rep_rate_mhz": float(rep_rate_mhz),
            "scan_loops": scan_loops,
            "acquisition_time_s": acquisition_time_s,
            "bit_depth": bit_depth,
            "dtype": str(dtype),
            "data_offset": data_offset,
        }
    )
    return FLIMDataset(
        decay=np.asarray(array),
        name=header_path.stem,
        source_path=header_path,
        dt_ns=dt_ns,
        rep_rate_mhz=rep_rate_mhz,
        metadata=metadata,
    )


def load_dataset(path: str | Path) -> FLIMDataset:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".ics", ".ids"}:
        return load_ics(source)
    if suffix == ".npy":
        array = np.load(source, allow_pickle=False)
        return FLIMDataset(_canonical_cube(array), source.stem, source, metadata={"format": "NumPy NPY"})
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            keys = list(archive.keys())
            data_key = next((key for key in ("decay", "data", "cube", "arr_0") if key in archive), keys[0])
            array = np.array(archive[data_key])
            dt_ns = float(np.asarray(archive["dt_ns"]).ravel()[0]) if "dt_ns" in archive else 0.390625
            if "rep_rate_mhz" in archive:
                rep_rate = float(np.asarray(archive["rep_rate_mhz"]).ravel()[0])
            elif "rep_rate_hz" in archive:
                rep_rate = float(np.asarray(archive["rep_rate_hz"]).ravel()[0]) / 1e6
            else:
                rep_rate = 80.0
        return FLIMDataset(
            _canonical_cube(array), source.stem, source, dt_ns, rep_rate, {"format": "NumPy NPZ", "data_key": data_key}
        )
    if suffix in {".tif", ".tiff"}:
        with Image.open(source) as image:
            frames = [np.asarray(frame.copy()) for frame in ImageSequence.Iterator(image)]
        if not frames:
            raise ValueError("The TIFF contains no readable frames.")
        stack = np.stack(frames, axis=0)
        cube = np.moveaxis(stack, 0, -1) if stack.ndim == 3 else _canonical_cube(stack)
        return FLIMDataset(cube, source.stem, source, metadata={"format": "TIFF", "frames": len(frames)})
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.loadtxt(source, delimiter=delimiter)
        return FLIMDataset(_canonical_cube(array), source.stem, source, metadata={"format": suffix[1:].upper()})
    raise ValueError(f"Unsupported input format '{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")


def save_dataset_npz(dataset: FLIMDataset, path: str | Path) -> Path:
    destination = Path(path)
    np.savez_compressed(
        destination,
        decay=dataset.decay,
        dt_ns=np.array(dataset.dt_ns),
        rep_rate_mhz=np.array(dataset.rep_rate_mhz),
        metadata_json=np.array(json.dumps(dataset.metadata, default=str)),
    )
    return destination
