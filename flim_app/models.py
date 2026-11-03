from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class FLIMDataset:
    """A time-resolved image cube in canonical ``(y, x, time)`` order."""

    decay: np.ndarray
    name: str = "Untitled"
    source_path: Path | None = None
    dt_ns: float = 0.390625
    rep_rate_mhz: float = 80.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cube = np.asarray(self.decay)
        if cube.ndim != 3:
            raise ValueError(f"A FLIM decay cube must be 3-D; received shape {cube.shape}.")
        if min(cube.shape) < 1:
            raise ValueError("A FLIM decay cube cannot contain an empty axis.")
        if cube.shape[-1] < 2:
            raise ValueError("A FLIM decay cube requires at least two time bins.")
        if not np.issubdtype(cube.dtype, np.number):
            raise TypeError("The decay cube must contain numeric values.")
        self.decay = cube
        self.dt_ns = float(self.dt_ns)
        self.rep_rate_mhz = float(self.rep_rate_mhz)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.decay.shape)

    @property
    def y_size(self) -> int:
        return int(self.decay.shape[0])

    @property
    def x_size(self) -> int:
        return int(self.decay.shape[1])

    @property
    def time_bins(self) -> int:
        return int(self.decay.shape[2])


@dataclass(slots=True)
class AnalysisSettings:
    """Settings mirrored from the useful controls in the source LabVIEW VI."""

    method: str = "phasor_phase"
    dt_ns: float = 0.390625
    rep_rate_mhz: float = 80.0
    background_counts: float = 0.0
    background_scale_mode: str = "auto"
    background_multiplier: float = 0.025
    background_auto_fallback: float = 0.025
    # True (default, LabVIEW parity): every time bin is clipped at zero right
    # after background subtraction. That is harmless when the background is a
    # small scaled-down average (e.g. 20/800 loops), but with a background of
    # the SAME exposure as the sample (multiplier ~1) the clip turns the
    # zero-mean noise left over in strongly background-dominated pixels into a
    # large positive bias -- hot detector rows stay visible. False keeps the
    # subtraction signed (negative bins allowed).
    background_clip_negative: bool = True
    min_photons: float = 30.0
    # Negative values preserve the pre-v0.7 behaviour by falling back to
    # ``min_photons``.  The desktop/CLI LabVIEW preset supplies the saved
    # method-specific thresholds (1000 phasor, 2000 fitting).
    phasor_min_counts: float = -1.0
    fit_min_counts: float = -1.0
    fit_start_bin: int = -1
    fit_end_bin: int = -1
    native_fit_bins: int = -1
    # <=0 means auto -- normalized() below resolves it to dt_ns. The
    # sidebar's "Native bin width mode" (Auto/Manual) maps directly onto
    # this: Auto sends -1.0 here regardless of whatever's in the manual
    # entry field, Manual sends that field's value.
    native_bin_width_ns: float = -1.0
    native_irf_baseline_mode: str = "median"
    # >1 splits the per-pixel FLIMLib reconvolution fit ("flimlib_reconvolution"
    # only -- never the global/shared-tau methods, which can't be split this
    # way without changing the answer) into this many row-band chunks, each
    # fitted in its own disposable subprocess concurrently -- see
    # native_backend.fit_pixels_parallel_in_subprocess. 1 preserves the
    # original single-process behaviour exactly.
    native_fit_workers: int = 1
    # Spatial-bin radius level: 0 -> 1x1, 1 -> 3x3, 2 -> 5x5, ...
    spatial_bin: int = 0
    smoothing_passes: int = 0
    timing_offset_bins: float = 0.0
    interpolate_outliers: bool = False
    outlier_z: float = 10.0
    interpolate_rows: bool = False
    interpolation_rows: tuple[int, ...] = ()
    interpolation_radius: int = 2
    interpolation_coordinate_mode: str = "direct_y"
    interpolation_map_name: str = ""
    interpolation_map_source: str = ""
    interpolation_map_sha256: str = ""
    interpolation_expected_source_height: int = 0
    interpolate_columns: bool = False
    interpolation_columns: tuple[int, ...] = ()
    column_interpolation_radius: int = 2
    phasor_heatmap_bins: int = 400
    lifetime_min_ns: float = 0.5
    lifetime_max_ns: float = 4.5
    lifetime_scale_mode: str = "manual"
    histogram_bins: int = 50
    detailed_histogram_bins: int = 1000
    # >0 caps how many pixels the global/shared-tau fit and training use
    # (a performance limit, not a correctness one -- see normalized());
    # <=0 means no cap at all, matching flim_fit_global_pixels_v2 and
    # flim_train_global_model_v1's own native "0/negative = unlimited"
    # convention in native/flimfit_labview.c (`maxGlobalPixels > 0` gates
    # the exclusion check there) -- passed straight through as 0 by
    # normalized() rather than floored up to some nonzero minimum.
    max_global_pixels: int = 20_000
    model_order: int = 1
    # 0 = multi-exponential (model_order discrete species, each a single
    # lifetime). 1 = stretched-exponential, single component
    # (Kohlrausch-Williams-Watts): I(t) = Z + A*exp(-(t/tau)^(1/h)) --
    # represents ONE species as a continuous distribution of decay rates
    # via the stretching exponent h (h=1 is a plain single exponential;
    # h>1 "stretches" it), instead of a single discrete lifetime; forces
    # model_order to 1 (one component, not several). 2 = multi-stretched-
    # exponential: I(t) = Z + sum_i A_i*exp(-(t/tau_i)^(1/h_i)) for
    # i=0..model_order-1 -- up to 3 INDEPENDENTLY distributed components,
    # each with its own (tau_i, h_i); model_order is respected normally
    # (1-3), NOT forced to 1. Only applies to the global/shared-tau fit
    # family (flimlib_global, train_global_model) -- iterative
    # reconvolution and applying a trained model still always use
    # multi-exponential. See native/flimfit_labview.c's
    # FIT_GLOBAL_STRETCHEDEXP/FIT_GLOBAL_MULTI_STRETCHEDEXP support and
    # RECONSTRUCTION_NOTES.md.
    fit_distribution_model: int = 0
    fit_background_mode: int = 0
    fit_background_value: float = 0.0
    fit_noise_model: int = 0
    fit_irf_mode: int = 0
    # Only meaningful when the loaded IRF is a genuine per-pixel (y, x,
    # time) cube (fit_irf_mode == 1 for FLIMLib's native fit methods;
    # phasor decides per-pixel-or-not from the array's own shape,
    # independently of fit_irf_mode). Collapses each column's per-pixel
    # IRF trace to the sum of that column across every row, then
    # broadcasts it back -- see processing.integrate_irf_per_column.
    # Applies uniformly to phasor, iterative reconvolution, global
    # fitting, and applying a trained model, since analyze_dataset
    # applies it once to irf_data itself, upstream of all of them. Off
    # by default so it never silently changes behaviour for a per-pixel
    # IRF that isn't a sweeping-linear-detector acquisition (genuine
    # pixel-to-pixel variation would be destroyed by this).
    irf_integrate_per_column: bool = False
    global_iterations: int = 4
    initial_taus_ns: tuple[float, ...] = ()
    native_dll_path: str = ""
    global_model_path: str = ""
    lifetime_lut_path: str = ""
    lifetime_lut_inverse: bool = False
    lifetime_lut_remove_out_of_range: bool = True
    # "file" uses whatever IRF was loaded (or none); "synthetic_gaussian"
    # ignores it and models the instrument response as a Gaussian pulse
    # instead -- see processing.estimate_synthetic_gaussian_irf.
    irf_source: str = "file"
    # <=0 means auto-estimate the FWHM from the sample's own decay rise
    # time; a positive value pins it instead.
    synthetic_irf_fwhm_ns: float = -1.0

    def normalized(self, time_bins: int) -> "AnalysisSettings":
        values = asdict(self)
        values["method"] = str(self.method).lower().strip()
        values["dt_ns"] = max(float(self.dt_ns), 1e-9)
        values["rep_rate_mhz"] = max(float(self.rep_rate_mhz), 1e-9)
        values["background_counts"] = max(float(self.background_counts), 0.0)
        background_scale_mode = str(self.background_scale_mode).strip().lower()
        if background_scale_mode not in {"auto", "manual"}:
            raise ValueError("Background scale mode must be 'auto' or 'manual'.")
        values["background_scale_mode"] = background_scale_mode
        for field_name in ("background_multiplier", "background_auto_fallback"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field_name.replace('_', ' ').capitalize()} must be finite and non-negative.")
            values[field_name] = value
        minimum_counts = float(self.min_photons)
        if not np.isfinite(minimum_counts):
            raise ValueError("Minimum processed counts must be finite.")
        values["min_photons"] = max(minimum_counts, 0.0)
        for field_name in ("phasor_min_counts", "fit_min_counts"):
            requested = float(getattr(self, field_name))
            if not np.isfinite(requested):
                raise ValueError(f"{field_name.replace('_', ' ').capitalize()} must be finite.")
            values[field_name] = values["min_photons"] if requested < 0 else max(requested, 0.0)
        values["fit_start_bin"] = max(-1, min(int(self.fit_start_bin), time_bins - 1))
        end = int(self.fit_end_bin)
        values["fit_end_bin"] = -1 if end < 0 else max(1, min(end, time_bins))
        requested_native_bins = int(self.native_fit_bins)
        values["native_fit_bins"] = (
            -1 if requested_native_bins <= 0 else max(2, min(requested_native_bins, time_bins))
        )
        native_bin_width = float(self.native_bin_width_ns)
        if not np.isfinite(native_bin_width):
            raise ValueError("Native fit bin width must be finite.")
        values["native_bin_width_ns"] = (
            values["dt_ns"] if native_bin_width <= 0 else max(native_bin_width, 1e-9)
        )
        baseline_mode = str(self.native_irf_baseline_mode).strip().lower()
        if baseline_mode not in {"none", "median", "tail_median"}:
            raise ValueError("Native IRF baseline mode must be none, median, or tail_median.")
        values["native_irf_baseline_mode"] = baseline_mode
        values["native_fit_workers"] = max(1, min(int(self.native_fit_workers), 32))
        values["spatial_bin"] = max(0, min(int(self.spatial_bin), 16))
        values["smoothing_passes"] = max(0, min(int(self.smoothing_passes), 8))
        values["histogram_bins"] = max(10, min(int(self.histogram_bins), 4096))
        values["detailed_histogram_bins"] = max(10, min(int(self.detailed_histogram_bins), 4096))
        requested_global_pixels = int(self.max_global_pixels)
        values["max_global_pixels"] = (
            0 if requested_global_pixels <= 0 else max(100, requested_global_pixels)
        )
        values["fit_distribution_model"] = int(self.fit_distribution_model) if int(self.fit_distribution_model) in (0, 1, 2) else 0
        single_stretched = values["fit_distribution_model"] == 1
        multi_stretched = values["fit_distribution_model"] == 2
        # Single-component stretched-exp is always exactly one distributed
        # component -- forced here regardless of what model_order was set
        # to, mirroring native/flimfit_labview.c's own forcing of this.
        # Multi-stretched-exp (up to 3 INDEPENDENTLY distributed
        # components) respects model_order like plain multiexp does.
        values["model_order"] = 1 if single_stretched else max(1, min(int(self.model_order), 3))
        values["fit_background_mode"] = 1 if int(self.fit_background_mode) == 1 else 0
        values["fit_background_value"] = max(float(self.fit_background_value), 0.0)
        values["fit_noise_model"] = max(0, min(int(self.fit_noise_model), 3))
        values["fit_irf_mode"] = 1 if int(self.fit_irf_mode) == 1 else 0
        values["irf_integrate_per_column"] = bool(self.irf_integrate_per_column)
        # Raised from 20: multi-stretched-exp's shared-parameter vector is
        # up to 2*model_order (6 at model_order=3) -- each outer Newton
        # iteration there costs far more objective evaluations than the
        # plain multiexp/single-stretched case (up to ~4*n + 2*n*(n-1)
        # for gradient+Hessian alone), so the old ceiling could starve a
        # 2-3 component distributed fit of enough iterations to converge.
        # Default (self.global_iterations<=0 -> 4, unchanged) still stays
        # low; this only widens how far the setting can be turned up.
        values["global_iterations"] = max(1, min(int(self.global_iterations), 80))
        # Single-component stretched-exp's shared vector is [tau, h] -- 2
        # slots, even though model_order is 1 (one *component*, not one
        # shared parameter). Multi-stretched-exp's shared vector is
        # [tau0,h0,tau1,h1,...] -- 2*model_order slots.
        if single_stretched:
            shared_slots = 2
        elif multi_stretched:
            shared_slots = 2 * values["model_order"]
        else:
            shared_slots = values["model_order"]
        values["initial_taus_ns"] = tuple(
            float(value) for value in self.initial_taus_ns if np.isfinite(value) and float(value) > 0
        )[:shared_slots]
        values["native_dll_path"] = str(self.native_dll_path).strip()
        values["global_model_path"] = str(self.global_model_path).strip()
        values["lifetime_lut_path"] = str(self.lifetime_lut_path).strip()
        values["lifetime_lut_inverse"] = bool(self.lifetime_lut_inverse)
        values["lifetime_lut_remove_out_of_range"] = bool(self.lifetime_lut_remove_out_of_range)
        irf_source = str(self.irf_source).strip().lower()
        if irf_source not in {"file", "synthetic_gaussian"}:
            raise ValueError("IRF source must be 'file' or 'synthetic_gaussian'.")
        values["irf_source"] = irf_source
        synthetic_fwhm = float(self.synthetic_irf_fwhm_ns)
        if not np.isfinite(synthetic_fwhm):
            raise ValueError("Synthetic IRF FWHM must be finite.")
        values["synthetic_irf_fwhm_ns"] = synthetic_fwhm
        values["outlier_z"] = max(float(self.outlier_z), 1.0)
        clean_rows: list[int] = []
        for value in self.interpolation_rows:
            numeric = float(value)
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"Interpolation rows must be finite integers; received {value!r}.")
            row = int(numeric)
            if row < 0:
                raise ValueError(f"Interpolation rows must be non-negative; received {row}.")
            clean_rows.append(row)
        values["interpolation_rows"] = tuple(sorted(set(clean_rows)))
        values["interpolation_radius"] = max(1, int(self.interpolation_radius))
        coordinate_mode = str(self.interpolation_coordinate_mode).strip().lower()
        if coordinate_mode not in {"direct_y", "labview_mirrored"}:
            raise ValueError(
                "Interpolation coordinate mode must be 'direct_y' or 'labview_mirrored'."
            )
        values["interpolation_coordinate_mode"] = coordinate_mode
        values["interpolation_map_name"] = str(self.interpolation_map_name).strip()
        values["interpolation_map_source"] = str(self.interpolation_map_source).strip()
        values["interpolation_map_sha256"] = str(self.interpolation_map_sha256).strip().lower()
        expected_height = int(self.interpolation_expected_source_height)
        values["interpolation_expected_source_height"] = max(0, expected_height)
        clean_columns: list[int] = []
        for value in self.interpolation_columns:
            numeric = float(value)
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"Interpolation columns must be finite integers; received {value!r}.")
            column = int(numeric)
            if column < 0:
                raise ValueError(f"Interpolation columns must be non-negative; received {column}.")
            clean_columns.append(column)
        values["interpolation_columns"] = tuple(sorted(set(clean_columns)))
        values["column_interpolation_radius"] = max(1, int(self.column_interpolation_radius))
        values["phasor_heatmap_bins"] = max(32, min(int(self.phasor_heatmap_bins), 512))
        lo = float(self.lifetime_min_ns)
        hi = float(self.lifetime_max_ns)
        if hi <= lo:
            hi = lo + 1.0
        values["lifetime_min_ns"] = lo
        values["lifetime_max_ns"] = hi
        scale_mode = str(self.lifetime_scale_mode).strip().lower()
        if scale_mode not in {"auto", "manual"}:
            raise ValueError("Lifetime scale mode must be 'auto' or 'manual'.")
        values["lifetime_scale_mode"] = scale_mode
        return AnalysisSettings(**values)

    @property
    def spatial_bin_width(self) -> int:
        """Actual square block width for the user-facing radius level."""

        return 2 * int(self.spatial_bin) + 1


@dataclass(slots=True)
class AnalysisResult:
    dataset_name: str
    intensity: np.ndarray
    lifetime_ns: np.ndarray
    valid_mask: np.ndarray
    g: np.ndarray
    s: np.ndarray
    tau_phase_ns: np.ndarray
    tau_modulation_ns: np.ndarray
    chi_square: np.ndarray
    mean_decay: np.ndarray
    mean_fit: np.ndarray
    time_ns: np.ndarray
    global_tau_ns: float
    settings: AnalysisSettings
    processed_decay: np.ndarray = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    fit_diagnostics: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    # The same cube as ``processed_decay`` but *before* rolling spatial
    # binning. Rolling binning makes neighbouring pixels overlap (each
    # pixel's value already includes its neighbours' counts), so summing
    # more than one processed pixel together double-counts shared photons.
    # Region-aggregate sums (ROI/whole-image decay, total photon counts --
    # see roi.py) use this instead; per-pixel analysis correctly keeps
    # using ``processed_decay``. Falls back to ``processed_decay`` when
    # unset (e.g. hand-built results in tests), which is exactly correct
    # at bin radius 0, where the two are identical anyway.
    additive_decay: np.ndarray | None = field(default=None, repr=False)

    @property
    def valid_lifetimes(self) -> np.ndarray:
        return self.lifetime_ns[np.isfinite(self.lifetime_ns)]

    def summary(self) -> dict[str, Any]:
        values = self.valid_lifetimes
        return {
            "dataset": self.dataset_name,
            "shape_yxt": [int(v) for v in self.processed_decay.shape],
            "valid_pixels": int(values.size),
            "total_pixels": int(self.lifetime_ns.size),
            "mean_lifetime_ns": float(np.nanmean(values)) if values.size else None,
            "median_lifetime_ns": float(np.nanmedian(values)) if values.size else None,
            "global_tau_ns": float(self.global_tau_ns) if np.isfinite(self.global_tau_ns) else None,
            "max_intensity": float(np.nanmax(self.intensity)) if self.intensity.size else 0.0,
            "settings": asdict(self.settings),
            "metadata": self.metadata,
        }
