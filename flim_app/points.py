from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import AnalysisResult


POINT_COLOURS = (
    "#56B4E9",
    "#E69F00",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#FFFFFF",
)
MAX_SELECTED_POINTS = len(POINT_COLOURS)


@dataclass(frozen=True, slots=True)
class PointSelection:
    identifier: str
    x: int
    y: int
    colour: str


@dataclass(slots=True)
class PointAnalysis:
    selection: PointSelection
    decay: np.ndarray
    map_tau_ns: float
    valid: bool
    processed_intensity: float
    total_processed_signal: float
    source_bounds_xyxy_nominal: tuple[int, int, int, int]
    lifetime_label: str
    fit_status: int | None = None
    native_diagnostics: dict[str, Any] | None = None
    phasor_g: float = float("nan")
    phasor_s: float = float("nan")
    phasor_valid: bool = False

    def summary(self) -> dict[str, Any]:
        return _json_safe({
            "point_id": self.selection.identifier,
            "colour": self.selection.colour,
            "processed_x": self.selection.x,
            "processed_y": self.selection.y,
            "nominal_source_bounds_xyxy": list(self.source_bounds_xyxy_nominal),
            "lifetime_label": self.lifetime_label,
            "map_tau_ns": self.map_tau_ns,
            "valid": self.valid,
            "fit_status": self.fit_status,
            "processed_intensity": self.processed_intensity,
            "total_processed_signal": self.total_processed_signal,
            "phasor_g": self.phasor_g,
            "phasor_s": self.phasor_s,
            "phasor_valid": self.phasor_valid,
            "native_diagnostics": self.native_diagnostics,
        })


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item())
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def lifetime_label(result: AnalysisResult) -> str:
    method = result.settings.method
    if method == "phasor_phase":
        return "τφ"
    if method == "phasor_modulation":
        return "τM"
    if method == "moment":
        return "τmoment"
    if method == "mono_exponential":
        return "τtail"
    if method == "flimlib_global":
        return "⟨τ⟩A (global model)" if result.settings.model_order > 1 else "τfit (global model)"
    if method == "flimlib_reconvolution":
        return "⟨τ⟩A" if result.settings.model_order > 1 else "τfit"
    return "τ"


def add_point(
    selections: list[PointSelection],
    x: int,
    y: int,
    max_points: int = MAX_SELECTED_POINTS,
) -> tuple[list[PointSelection], bool]:
    """Return an updated point list and whether a new point was added."""

    if any(point.x == int(x) and point.y == int(y) for point in selections):
        return list(selections), False
    if len(selections) >= max_points:
        raise ValueError(f"A maximum of {max_points} point decays can be displayed at once.")
    used = {point.identifier for point in selections}
    index = next(index for index in range(max_points) if f"P{index + 1}" not in used)
    point = PointSelection(f"P{index + 1}", int(x), int(y), POINT_COLOURS[index])
    return [*selections, point], True


def analyse_points(result: AnalysisResult, selections: list[PointSelection]) -> list[PointAnalysis]:
    height, width = result.intensity.shape
    bin_factor = int(result.metadata.get("spatial_transform", {}).get("spatial_bin_width", 1))
    status_map = result.fit_diagnostics.get("fit_status")
    label = lifetime_label(result)
    analysed: list[PointAnalysis] = []
    for point in selections:
        if not (0 <= point.x < width and 0 <= point.y < height):
            raise ValueError(f"{point.identifier} lies outside the current processed image.")
        decay = np.asarray(result.processed_decay[point.y, point.x], dtype=np.float64).copy()
        tau = float(result.lifetime_ns[point.y, point.x])
        valid = bool(result.valid_mask[point.y, point.x] and np.isfinite(tau))
        fit_status = None
        if status_map is not None and np.asarray(status_map).shape == result.intensity.shape:
            fit_status = int(np.asarray(status_map)[point.y, point.x])
        native_diagnostics: dict[str, Any] = {}
        for key, values in result.fit_diagnostics.items():
            array = np.asarray(values)
            if array.ndim >= 2 and array.shape[-2:] == result.intensity.shape:
                native_diagnostics[key] = np.asarray(array[..., point.y, point.x]).copy()
        if result.metadata.get("native_fit"):
            for key in ("global_taus_ns", "global_stats"):
                native_diagnostics[key] = result.metadata["native_fit"].get(key)
        phasor_g = float(result.g[point.y, point.x])
        phasor_s = float(result.s[point.y, point.x])
        phasor_valid = bool(np.isfinite(phasor_g) and np.isfinite(phasor_s))
        analysed.append(
            PointAnalysis(
                selection=point,
                decay=decay,
                map_tau_ns=tau,
                valid=valid,
                processed_intensity=float(result.intensity[point.y, point.x]),
                total_processed_signal=float(np.nansum(decay)),
                source_bounds_xyxy_nominal=(
                    point.x * bin_factor,
                    point.y * bin_factor,
                    (point.x + 1) * bin_factor,
                    (point.y + 1) * bin_factor,
                ),
                lifetime_label=label,
                fit_status=fit_status,
                native_diagnostics=native_diagnostics or None,
                phasor_g=phasor_g,
                phasor_s=phasor_s,
                phasor_valid=phasor_valid,
            )
        )
    return analysed
