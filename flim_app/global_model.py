from __future__ import annotations

"""A shared-lifetime model trained across several images, and its persistence.

Training (``processing.train_global_model``) runs the native
``flim_train_global_model_v1`` DLL entry point across a stack of images to
find one set of shared lifetimes. Applying that model to a new dataset later
(the ``flimlib_apply_global`` lifetime method in ``processing.analyze_dataset``)
holds those lifetimes fixed and only fits amplitude/background per pixel via
``flim_apply_global_model_v1``.

A :class:`TrainedGlobalModel` is the small, portable artifact connecting the
two: the fixed taus, exactly which native-fit settings they were trained
under (so applying reproduces the same fitting conditions), and provenance
describing what was trained on. It is saved/loaded as plain JSON so it can be
inspected, versioned, or handed to a colleague.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


class GlobalModelError(ValueError):
    """A trained-model file could not be built, saved, or loaded."""


@dataclass(slots=True)
class TrainingImageProvenance:
    """One training image's contribution to a trained model, for the record."""

    name: str
    decay_source: str
    irf_source: str
    background_source: str
    shape_yx: tuple[int, int]
    total_pixels: int
    selected_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decay_source": self.decay_source,
            "irf_source": self.irf_source,
            "background_source": self.background_source,
            "shape_yx": list(self.shape_yx),
            "total_pixels": self.total_pixels,
            "selected_pixels": self.selected_pixels,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TrainingImageProvenance":
        return TrainingImageProvenance(
            name=str(data.get("name", "")),
            decay_source=str(data.get("decay_source", "")),
            irf_source=str(data.get("irf_source", "")),
            background_source=str(data.get("background_source", "")),
            shape_yx=tuple(int(v) for v in data.get("shape_yx", (0, 0))),
            total_pixels=int(data.get("total_pixels", 0)),
            selected_pixels=int(data.get("selected_pixels", 0)),
        )


@dataclass(slots=True)
class TrainedGlobalModel:
    """A shared-lifetime model plus the exact fitting conditions it needs to be applied under."""

    name: str
    global_taus_ns: tuple[float, ...]
    model_order: int
    # 0 = multi-exponential (global_taus_ns is model_order discrete
    # lifetimes). 1 = stretched-exponential -- global_taus_ns is exactly
    # [tau, h] instead (see AnalysisSettings.fit_distribution_model).
    # Applying a stretched-exp model isn't wired up yet (flim_apply_global_model_v1
    # only ever does multi-exponential) -- see processing.py's apply
    # branch, which raises rather than silently mistreating h as a
    # second discrete lifetime or dropping it.
    fit_distribution_model: int

    # The native-fit conditions training was run under. Applying the model
    # later reuses these so the fixed-tau fit matches how the taus were
    # found (see AnalysisSettings, which these mirror field-for-field).
    dt_ns: float
    background_mode: int
    background_value: float
    noise_model: int
    fit_min_counts: float
    fit_start_bin: int
    fit_end_bin: int
    native_fit_bins: int
    native_bin_width_ns: float
    native_irf_baseline_mode: str
    irf_mode: int
    irf_integrate_per_column: bool
    spatial_bin: int
    timing_offset_bins: float
    background_scale_mode: str
    background_multiplier: float
    background_auto_fallback: float
    background_counts: float

    # Training-only settings, kept for provenance/reproducibility but not
    # needed to apply the model.
    global_iterations: int
    max_global_pixels: int
    initial_taus_ns: tuple[float, ...]

    # Provenance.
    trained_at: str
    dll_sha256: str
    training_objective: float | None
    training_reduced_objective: float | None
    training_iterations: int
    training_selected_pixels: int
    training_total_pixels: int
    training_images: tuple[TrainingImageProvenance, ...]

    def summary(self) -> str:
        if self.fit_distribution_model == 1 and len(self.global_taus_ns) >= 2:
            # [tau, h], not two lifetimes -- see the field's own docstring.
            taus = f"τ={self.global_taus_ns[0]:.3f} ns, h={self.global_taus_ns[1]:.3f} (stretched exp)"
        elif self.fit_distribution_model == 2 and len(self.global_taus_ns) >= 2:
            # [tau0,h0,tau1,h1,...] -- one (tau,h) pair per independently
            # distributed component, not a flat list of lifetimes.
            pairs = zip(self.global_taus_ns[0::2], self.global_taus_ns[1::2])
            taus = ", ".join(f"τ{i}={t:.3f} ns/h{i}={h:.3f}" for i, (t, h) in enumerate(pairs))
            taus += " (multi-stretched exp)"
        else:
            taus = ", ".join(f"{tau:.3f} ns" for tau in self.global_taus_ns)
        return (
            f"{self.name}: {taus} "
            f"({self.training_selected_pixels:,}/{self.training_total_pixels:,} px, "
            f"{len(self.training_images)} images)"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["training_images"] = [item.to_dict() for item in self.training_images]
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TrainedGlobalModel":
        images = tuple(
            TrainingImageProvenance.from_dict(item) for item in data.get("training_images", ())
        )
        return TrainedGlobalModel(
            name=str(data.get("name", "Untitled global model")),
            global_taus_ns=tuple(float(v) for v in data["global_taus_ns"]),
            model_order=int(data["model_order"]),
            # Absent in a model saved before this field existed -- 0
            # (multi-exponential) matches what training could only ever
            # produce back then.
            fit_distribution_model=int(data.get("fit_distribution_model", 0)),
            dt_ns=float(data.get("dt_ns", 0.390625)),
            background_mode=int(data.get("background_mode", 0)),
            background_value=float(data.get("background_value", 0.0)),
            noise_model=int(data.get("noise_model", 0)),
            fit_min_counts=float(data.get("fit_min_counts", 0.0)),
            fit_start_bin=int(data.get("fit_start_bin", -1)),
            fit_end_bin=int(data.get("fit_end_bin", -1)),
            native_fit_bins=int(data.get("native_fit_bins", -1)),
            native_bin_width_ns=float(data.get("native_bin_width_ns", -1.0)),
            native_irf_baseline_mode=str(data.get("native_irf_baseline_mode", "median")),
            irf_mode=int(data.get("irf_mode", 1)),
            # Absent in a model saved before this field existed -- False
            # (no integration) matches what training actually did back
            # then, so an old model still applies exactly as trained.
            irf_integrate_per_column=bool(data.get("irf_integrate_per_column", False)),
            spatial_bin=int(data.get("spatial_bin", 0)),
            timing_offset_bins=float(data.get("timing_offset_bins", 0.0)),
            background_scale_mode=str(data.get("background_scale_mode", "auto")),
            background_multiplier=float(data.get("background_multiplier", 0.025)),
            background_auto_fallback=float(data.get("background_auto_fallback", 0.025)),
            background_counts=float(data.get("background_counts", 0.0)),
            global_iterations=int(data.get("global_iterations", 8)),
            max_global_pixels=int(data.get("max_global_pixels", 20_000)),
            initial_taus_ns=tuple(float(v) for v in data.get("initial_taus_ns", ())),
            trained_at=str(data.get("trained_at", "")),
            dll_sha256=str(data.get("dll_sha256", "")),
            training_objective=(
                float(data["training_objective"]) if data.get("training_objective") is not None else None
            ),
            training_reduced_objective=(
                float(data["training_reduced_objective"])
                if data.get("training_reduced_objective") is not None
                else None
            ),
            training_iterations=int(data.get("training_iterations", 0)),
            training_selected_pixels=int(data.get("training_selected_pixels", 0)),
            training_total_pixels=int(data.get("training_total_pixels", 0)),
            training_images=images,
        )


def save_global_model(model: TrainedGlobalModel, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "file_format": "flim_studio_global_model",
        "file_format_version": 1,
        **model.to_dict(),
    }
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def load_global_model(path: str | Path) -> TrainedGlobalModel:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GlobalModelError(f"Could not read trained model file {source}: {exc}") from exc
    if data.get("file_format") != "flim_studio_global_model":
        raise GlobalModelError(f"{source} is not a FLIM Studio trained-model file.")
    try:
        return TrainedGlobalModel.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise GlobalModelError(f"{source} is not a valid trained-model file: {exc}") from exc


def new_model_name(model_order: int) -> str:
    label = {2: "bi", 3: "tri"}.get(model_order, "mono")
    # No colon: this name is also offered as a default Windows save filename,
    # and ':' is one of the reserved characters there.
    return f"Global {label}-exponential model - {time.strftime('%Y-%m-%d %H%M')}"


def build_trained_model(
    *,
    name: str,
    global_taus_ns: np.ndarray,
    global_stats: np.ndarray,
    model_order: int,
    fit_distribution_model: int,
    dt_ns: float,
    background_mode: int,
    background_value: float,
    noise_model: int,
    fit_min_counts: float,
    fit_start_bin: int,
    fit_end_bin: int,
    native_fit_bins: int,
    native_bin_width_ns: float,
    native_irf_baseline_mode: str,
    irf_mode: int,
    irf_integrate_per_column: bool,
    spatial_bin: int,
    timing_offset_bins: float,
    background_scale_mode: str,
    background_multiplier: float,
    background_auto_fallback: float,
    background_counts: float,
    global_iterations: int,
    max_global_pixels: int,
    initial_taus_ns: tuple[float, ...],
    dll_sha256: str,
    training_images: tuple[TrainingImageProvenance, ...],
) -> TrainedGlobalModel:
    """Assemble a :class:`TrainedGlobalModel` from a completed training run."""

    taus = tuple(float(value) for value in np.asarray(global_taus_ns, dtype=np.float64))
    stats = np.asarray(global_stats, dtype=np.float64).ravel()

    def stat(index: int) -> float | None:
        if index < stats.size and np.isfinite(stats[index]):
            return float(stats[index])
        return None

    return TrainedGlobalModel(
        name=name,
        global_taus_ns=taus,
        model_order=int(model_order),
        fit_distribution_model=int(fit_distribution_model),
        dt_ns=float(dt_ns),
        background_mode=int(background_mode),
        background_value=float(background_value),
        noise_model=int(noise_model),
        fit_min_counts=float(fit_min_counts),
        fit_start_bin=int(fit_start_bin),
        fit_end_bin=int(fit_end_bin),
        native_fit_bins=int(native_fit_bins),
        native_bin_width_ns=float(native_bin_width_ns),
        native_irf_baseline_mode=str(native_irf_baseline_mode),
        irf_mode=int(irf_mode),
        irf_integrate_per_column=bool(irf_integrate_per_column),
        spatial_bin=int(spatial_bin),
        timing_offset_bins=float(timing_offset_bins),
        background_scale_mode=str(background_scale_mode),
        background_multiplier=float(background_multiplier),
        background_auto_fallback=float(background_auto_fallback),
        background_counts=float(background_counts),
        global_iterations=int(global_iterations),
        max_global_pixels=int(max_global_pixels),
        initial_taus_ns=tuple(float(v) for v in initial_taus_ns),
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        dll_sha256=dll_sha256,
        training_objective=stat(1),
        training_reduced_objective=stat(2),
        training_iterations=int(stat(3) or 0),
        training_selected_pixels=int(stat(4) or 0),
        training_total_pixels=sum(item.total_pixels for item in training_images),
        training_images=training_images,
    )
