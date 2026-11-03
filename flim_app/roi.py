from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import AnalysisResult


BoundsXYXY = tuple[int, int, int, int]
NormalisedBounds = tuple[float, float, float, float]


@dataclass(slots=True)
class ROIAnalysis:
    """Measurements for one rectangular region in processed-image coordinates."""

    bounds_xyxy: BoundsXYXY
    selected_pixels: int
    valid_pixels: int
    summed_decay: np.ndarray
    mean_decay: np.ndarray
    fitted_decay: np.ndarray
    aggregate_tau_ns: float
    native_fitted_decay: np.ndarray | None
    native_fit_label: str
    total_photons: float
    mean_intensity: float
    pixel_tau_mean_ns: float
    pixel_tau_std_ns: float
    pixel_tau_median_ns: float
    pixel_tau_q1_ns: float
    pixel_tau_q3_ns: float
    photon_weighted_tau_ns: float
    guide_fit_start_bin: int
    guide_fit_end_bin: int

    @property
    def width(self) -> int:
        return self.bounds_xyxy[2] - self.bounds_xyxy[0]

    @property
    def height(self) -> int:
        return self.bounds_xyxy[3] - self.bounds_xyxy[1]

    @property
    def valid_fraction(self) -> float:
        return self.valid_pixels / self.selected_pixels if self.selected_pixels else 0.0

    def summary(self) -> dict[str, Any]:
        def finite_or_none(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return {
            "bounds_xyxy_processed": list(self.bounds_xyxy),
            "width_processed_pixels": self.width,
            "height_processed_pixels": self.height,
            "selected_pixels": self.selected_pixels,
            "processed_pixel_count": self.selected_pixels,
            "valid_pixels": self.valid_pixels,
            "valid_fraction": self.valid_fraction,
            "total_processed_signal": self.total_photons,
            "mean_processed_intensity_per_processed_pixel": self.mean_intensity,
            "aggregate_mono_guide_tau_ns": finite_or_none(self.aggregate_tau_ns),
            "aggregate_guide_fit_window_bins": [self.guide_fit_start_bin, self.guide_fit_end_bin],
            "pixel_tau_mean_ns": finite_or_none(self.pixel_tau_mean_ns),
            "pixel_tau_std_ns": finite_or_none(self.pixel_tau_std_ns),
            "pixel_tau_median_ns": finite_or_none(self.pixel_tau_median_ns),
            "pixel_tau_q1_ns": finite_or_none(self.pixel_tau_q1_ns),
            "pixel_tau_q3_ns": finite_or_none(self.pixel_tau_q3_ns),
            "processed_intensity_weighted_pixel_tau_ns": finite_or_none(self.photon_weighted_tau_ns),
            "aggregate_guide": {
                "algorithm": "weighted log-linear mono-exponential tail fit",
                "uses_irf_reconvolution": False,
                "role": "diagnostic guide; separate from the selected lifetime-map method",
            },
            "native_reconvolution_guide": {
                "available": self.native_fitted_decay is not None,
                "label": self.native_fit_label,
                "algorithm": "sum of per-pixel native fitted components, reconvolved with the mean IRF trace",
                "role": "display-only reconstruction of the native FLIMLib fit; not bin-identical to the DLL curve",
            },
        }


def normalised_to_bounds(selection: NormalisedBounds, shape_yx: tuple[int, int]) -> BoundsXYXY:
    """Convert a drag box in 0..1 display coordinates to a non-empty image slice."""

    height, width = (int(shape_yx[0]), int(shape_yx[1]))
    if height < 1 or width < 1:
        raise ValueError("ROI selection requires a non-empty image.")
    x0n, y0n, x1n, y1n = (float(value) for value in selection)
    if not np.all(np.isfinite((x0n, y0n, x1n, y1n))):
        raise ValueError("ROI coordinates must be finite.")
    x_lo, x_hi = sorted((float(np.clip(x0n, 0.0, 1.0)), float(np.clip(x1n, 0.0, 1.0))))
    y_lo, y_hi = sorted((float(np.clip(y0n, 0.0, 1.0)), float(np.clip(y1n, 0.0, 1.0))))

    def snap(value: float) -> float:
        nearest = round(value)
        return float(nearest) if abs(value - nearest) <= 1e-10 else value

    x0 = min(width - 1, max(0, int(np.floor(snap(x_lo * width)))))
    y0 = min(height - 1, max(0, int(np.floor(snap(y_lo * height)))))
    x1 = min(width, max(x0 + 1, int(np.ceil(snap(x_hi * width)))))
    y1 = min(height, max(y0 + 1, int(np.ceil(snap(y_hi * height)))))
    return x0, y0, x1, y1


def normalised_to_pixel(point: tuple[float, float], shape_yx: tuple[int, int]) -> tuple[int, int]:
    height, width = (int(shape_yx[0]), int(shape_yx[1]))
    if height < 1 or width < 1:
        raise ValueError("Point selection requires a non-empty image.")
    nx, ny = (float(value) for value in point)
    if not np.all(np.isfinite((nx, ny))):
        raise ValueError("Point coordinates must be finite.")
    x = min(width - 1, max(0, int(np.floor(np.clip(nx, 0.0, 1.0) * width))))
    y = min(height - 1, max(0, int(np.floor(np.clip(ny, 0.0, 1.0) * height))))
    return x, y


def pixel_to_normalised_center(point_xy: tuple[int, int], shape_yx: tuple[int, int]) -> tuple[float, float]:
    height, width = (int(shape_yx[0]), int(shape_yx[1]))
    x, y = (int(value) for value in point_xy)
    if height < 1 or width < 1 or not (0 <= x < width and 0 <= y < height):
        raise ValueError("Point lies outside the image.")
    return (x + 0.5) / width, (y + 0.5) / height


def bounds_to_normalised(bounds: BoundsXYXY, shape_yx: tuple[int, int]) -> NormalisedBounds:
    height, width = (int(shape_yx[0]), int(shape_yx[1]))
    x0, y0, x1, y1 = clamp_bounds(bounds, shape_yx)
    return x0 / width, y0 / height, x1 / width, y1 / height


def clamp_bounds(bounds: BoundsXYXY | None, shape_yx: tuple[int, int]) -> BoundsXYXY:
    height, width = (int(shape_yx[0]), int(shape_yx[1]))
    if height < 1 or width < 1:
        raise ValueError("ROI analysis requires a non-empty image.")
    if bounds is None:
        return 0, 0, width, height
    x0, y0, x1, y1 = (int(value) for value in bounds)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI bounds do not intersect the image.")
    return x0, y0, x1, y1


def analyse_roi(result: AnalysisResult, bounds: BoundsXYXY | None = None) -> ROIAnalysis:
    """Aggregate every finite processed decay in an ROI and summarize valid map pixels.

    The summed curve deliberately does not exclude pixels whose individual lifetime fit
    failed. Excluding them would bias the region decay. Pixel-lifetime statistics are
    reported separately using ``result.valid_mask``.
    """

    x0, y0, x1, y1 = clamp_bounds(bounds, result.intensity.shape)
    # Use the pre-rolling-bin cube for this region SUM specifically: rolling
    # spatial binning makes neighbouring processed pixels overlap, so summing
    # more than one of them (any ROI/whole-image region wider than 1x1)
    # would double- and triple-count shared photons. The pre-bin cube is
    # genuinely additive per pixel, giving the correct region total
    # regardless of the bin radius used for the per-pixel maps.
    source = result.additive_decay if result.additive_decay is not None else result.processed_decay
    cube = np.asarray(source[y0:y1, x0:x1], dtype=np.float64)
    finite_cube = np.where(np.isfinite(cube), cube, 0.0)
    summed_decay = finite_cube.sum(axis=(0, 1))
    selected_pixels = int((x1 - x0) * (y1 - y0))
    mean_decay = summed_decay / max(selected_pixels, 1)

    # Imported locally to keep ROI geometry independent from the fitting module.
    from .processing import fit_single_decay, native_multiexponential_curve

    fitted_decay, aggregate_tau_ns = fit_single_decay(
        summed_decay,
        result.time_ns,
        result.settings.fit_start_bin,
        result.settings.fit_end_bin,
    )
    finite_fit_bins = np.flatnonzero(np.isfinite(fitted_decay))
    guide_fit_start = int(finite_fit_bins[0]) if finite_fit_bins.size else -1
    guide_fit_end = int(finite_fit_bins[-1] + 1) if finite_fit_bins.size else -1

    native_fitted_decay: np.ndarray | None = None
    native_fit_label = ""
    tau_components = result.fit_diagnostics.get("tau_components_ns")
    amplitude_components = result.fit_diagnostics.get("amplitude_components")
    irf_trace = result.fit_diagnostics.get("native_irf_trace")
    if tau_components is not None and amplitude_components is not None and irf_trace is not None:
        tau_components = np.asarray(tau_components, dtype=np.float64)
        amplitude_components = np.asarray(amplitude_components, dtype=np.float64)
        if tau_components.ndim == 3 and tau_components.shape[1:] == result.intensity.shape:
            order = tau_components.shape[0]
            tau_roi = tau_components[:, y0:y1, x0:x1].reshape(order, -1)
            amplitude_roi = amplitude_components[:, y0:y1, x0:x1].reshape(order, -1)
            status_map = result.fit_diagnostics.get("fit_status")
            background_map = result.fit_diagnostics.get("fit_background")
            status_roi = (
                np.asarray(status_map[y0:y1, x0:x1]).reshape(-1) == 1
                if status_map is not None
                else np.ones(tau_roi.shape[1], dtype=bool)
            )
            background_roi = (
                np.asarray(background_map[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
                if background_map is not None
                else np.zeros(tau_roi.shape[1], dtype=np.float64)
            )
            pixel_ok = status_roi & np.all(np.isfinite(tau_roi) & (tau_roi > 0), axis=0)
            native_fit_count = int(np.count_nonzero(pixel_ok))
            if native_fit_count:
                native_fitted_decay = native_multiexponential_curve(
                    result.time_ns.size,
                    result.settings.native_bin_width_ns,
                    irf_trace,
                    tau_roi[:, pixel_ok],
                    amplitude_roi[:, pixel_ok],
                    background_roi[pixel_ok],
                )
                model_name = {2: "Bi-exponential", 3: "Tri-exponential"}.get(order, "Mono-exponential")
                irf_note = " (mean IRF)" if int(result.settings.fit_irf_mode) == 1 else ""
                native_fit_label = (
                    f"Native {model_name} reconvolution — {native_fit_count:,}/{selected_pixels:,} "
                    f"fitted px{irf_note}"
                )

    lifetime = np.asarray(result.lifetime_ns[y0:y1, x0:x1], dtype=np.float64)
    intensity = np.asarray(result.intensity[y0:y1, x0:x1], dtype=np.float64)
    valid = np.asarray(result.valid_mask[y0:y1, x0:x1], dtype=bool)
    valid &= np.isfinite(lifetime)
    lifetime_values = lifetime[valid]
    intensity_weights = np.clip(np.where(np.isfinite(intensity[valid]), intensity[valid], 0.0), 0.0, None)

    if lifetime_values.size:
        mean_tau = float(np.mean(lifetime_values))
        std_tau = float(np.std(lifetime_values))
        median_tau = float(np.median(lifetime_values))
        q1_tau, q3_tau = (float(value) for value in np.percentile(lifetime_values, [25, 75]))
        weighted_tau = (
            float(np.average(lifetime_values, weights=intensity_weights))
            if float(np.sum(intensity_weights)) > 0
            else float("nan")
        )
    else:
        mean_tau = std_tau = median_tau = q1_tau = q3_tau = weighted_tau = float("nan")

    total_photons = float(np.sum(summed_decay))
    mean_intensity = float(np.mean(np.where(np.isfinite(intensity), intensity, 0.0)))
    return ROIAnalysis(
        bounds_xyxy=(x0, y0, x1, y1),
        selected_pixels=selected_pixels,
        valid_pixels=int(lifetime_values.size),
        summed_decay=summed_decay,
        mean_decay=mean_decay,
        fitted_decay=np.asarray(fitted_decay, dtype=np.float64),
        aggregate_tau_ns=float(aggregate_tau_ns),
        native_fitted_decay=native_fitted_decay,
        native_fit_label=native_fit_label,
        total_photons=total_photons,
        mean_intensity=mean_intensity,
        pixel_tau_mean_ns=mean_tau,
        pixel_tau_std_ns=std_tau,
        pixel_tau_median_ns=median_tau,
        pixel_tau_q1_ns=q1_tau,
        pixel_tau_q3_ns=q3_tau,
        photon_weighted_tau_ns=weighted_tau,
        guide_fit_start_bin=guide_fit_start,
        guide_fit_end_bin=guide_fit_end,
    )
