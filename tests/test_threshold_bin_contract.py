from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.models import AnalysisSettings, FLIMDataset
from flim_app.native_backend import NativeFitResult
import flim_app.native_backend as native_backend
import flim_app.processing as processing


def _two_level_dataset(time_bins: int = 32) -> FLIMDataset:
    """Return two clean decays with integrated counts of 800 and 1,600."""

    time = (np.arange(time_bins, dtype=np.float64) + 0.5) * 0.390625
    profile = np.exp(-time / 2.2)
    profile /= profile.sum()
    cube = np.stack((800.0 * profile, 1_600.0 * profile), axis=0)
    return FLIMDataset(
        cube.reshape(1, 2, time_bins).astype(np.float32),
        name="threshold-contract",
        dt_ns=0.390625,
        rep_rate_mhz=80.0,
        metadata={"time_bins": time_bins},
    )


def _settings(method: str, **changes: object) -> AnalysisSettings:
    values: dict[str, object] = {
        "method": method,
        "dt_ns": 0.390625,
        "rep_rate_mhz": 80.0,
        "min_photons": 17.0,
        "phasor_min_counts": 500.0,
        "fit_min_counts": 1_200.0,
        "native_fit_bins": -1,
        "lifetime_min_ns": -100.0,
        "lifetime_max_ns": 100.0,
        "histogram_bins": 50,
        "detailed_histogram_bins": 1000,
    }
    values.update(changes)
    return AnalysisSettings(**values)


def test_threshold_sentinels_fall_back_to_legacy_minimum() -> None:
    fallback = AnalysisSettings(
        min_photons=123.0,
        phasor_min_counts=-1.0,
        fit_min_counts=-1.0,
    ).normalized(32)
    assert fallback.phasor_min_counts == 123.0
    assert fallback.fit_min_counts == 123.0

    explicit = AnalysisSettings(
        min_photons=123.0,
        phasor_min_counts=456.0,
        fit_min_counts=789.0,
    ).normalized(32)
    assert explicit.phasor_min_counts == 456.0
    assert explicit.fit_min_counts == 789.0


def test_native_bin_setting_normalizes_to_available_acquisition_bins() -> None:
    assert AnalysisSettings(native_fit_bins=-1).normalized(32).native_fit_bins == -1
    assert AnalysisSettings(native_fit_bins=0).normalized(32).native_fit_bins == -1
    assert AnalysisSettings(native_fit_bins=29).normalized(32).native_fit_bins == 29
    assert AnalysisSettings(native_fit_bins=99).normalized(32).native_fit_bins == 32


def test_estimators_use_their_own_count_thresholds() -> None:
    dataset = _two_level_dataset()

    phasor = processing.analyze_dataset(dataset, _settings("phasor_phase"))
    assert phasor.valid_mask.tolist() == [[True, True]]
    assert np.isfinite(phasor.g).tolist() == [[True, True]]
    assert phasor.metadata["count_thresholds"] == {
        "legacy_min_photons": 17.0,
        "phasor_min_counts": 500.0,
        "fit_min_counts": 1200.0,
        "active_min_counts": 500.0,
        "active_estimator_family": "phasor",
        "preview_signal_basis": "all processed TCSPC bins",
    }

    for method in ("mono_exponential", "moment"):
        fitted = processing.analyze_dataset(dataset, _settings(method))
        assert fitted.valid_mask.tolist() == [[False, True]]
        # Phasor coordinates are still calculated with the phasor-specific
        # threshold even when the displayed lifetime comes from a fit.
        assert np.isfinite(fitted.g).tolist() == [[True, True]]
        thresholds = fitted.metadata["count_thresholds"]
        assert thresholds["active_min_counts"] == 1200.0
        assert thresholds["active_estimator_family"] == "fit"


def test_time_bin_contract_separates_acquisition_native_and_histogram_bins() -> None:
    dataset = _two_level_dataset(time_bins=32)
    result = processing.analyze_dataset(dataset, _settings("phasor_phase", native_fit_bins=29))
    contract = result.metadata["time_bin_contract"]
    assert contract["acquisition_bins"] == 32
    assert contract["processed_bins"] == 32
    assert contract["configured_native_fit_bins"] == 29
    # Merely configuring a native crop must not imply that a phasor analysis
    # actually sent data to the native backend.
    assert contract["native_input_bins"] is None
    assert contract["native_uses_all_bins"] is None
    assert contract["analysis_fit_window_start_inclusive"] == int(
        result.metadata["analysis_fit_window"][0]
    )
    assert contract["analysis_fit_window_stop_exclusive"] == int(
        result.metadata["analysis_fit_window"][1]
    )
    assert contract["overview_histogram_intervals"] == 50
    assert contract["detailed_histogram_intervals"] == 1000


def test_native_backend_receives_the_requested_bin_crop_and_fit_threshold() -> None:
    dataset = _two_level_dataset(time_bins=32)
    irf_trace = np.exp(-0.5 * ((np.arange(32, dtype=np.float32) - 3.0) / 1.0) ** 2)
    irf = np.broadcast_to(irf_trace, dataset.shape).copy()
    captured: dict[str, object] = {}

    def fake_native_fit(mode: str, decay: np.ndarray, irf_data: np.ndarray, **kwargs: object) -> NativeFitResult:
        captured["mode"] = mode
        captured["decay_shape"] = decay.shape
        captured["irf_shape"] = irf_data.shape
        captured["min_photons"] = kwargs["min_photons"]
        captured["bin_width_ns"] = kwargs["bin_width_ns"]
        captured["irf_min"] = float(np.min(irf_data))
        captured["irf_median"] = float(np.max(np.median(irf_data, axis=-1)))
        y_size, x_size, _ = decay.shape
        tau = np.full((y_size, x_size), 2.2, dtype=np.float64)
        status = np.ones((y_size, x_size), dtype=np.int32)
        return NativeFitResult(
            tau_components_ns=tau[None, ...],
            tau_mean_ns=tau,
            amplitude_components=np.ones((1, y_size, x_size), dtype=np.float64),
            background=np.zeros((y_size, x_size), dtype=np.float64),
            chi_square=np.zeros((y_size, x_size), dtype=np.float64),
            fit_status=status,
            photon_counts=decay.sum(axis=-1, dtype=np.float64),
            metadata={"mode": mode},
        )

    # flimlib_reconvolution now calls native_backend.fit_pixels_parallel_in_subprocess
    # (worker_count=1 by default just passes straight through to fit_in_subprocess) --
    # patch fit_in_subprocess where that call actually resolves it (native_backend's
    # own module globals), not the name processing.py imports, since a plain call from
    # inside fit_pixels_parallel_in_subprocess looks it up there at call time regardless
    # of what processing.fit_in_subprocess happens to be bound to.
    original = native_backend.fit_in_subprocess
    native_backend.fit_in_subprocess = fake_native_fit
    try:
        result = processing.analyze_dataset(
            dataset,
            _settings(
                "flimlib_reconvolution",
                fit_min_counts=700.0,
                native_fit_bins=29,
                native_bin_width_ns=0.4,
                native_irf_baseline_mode="median",
            ),
            irf=irf,
        )
    finally:
        native_backend.fit_in_subprocess = original

    assert captured["mode"] == "pixels"
    assert captured["decay_shape"] == (1, 2, 29)
    assert captured["irf_shape"] == (1, 2, 29)
    assert captured["min_photons"] == 700.0
    assert captured["bin_width_ns"] == 0.4
    assert captured["irf_min"] == 0.0
    assert abs(float(captured["irf_median"])) < 1e-6
    assert result.processed_decay.shape == (1, 2, 32)
    contract = result.metadata["time_bin_contract"]
    assert contract["configured_native_fit_bins"] == 29
    assert contract["native_input_bins"] == 29
    assert contract["native_uses_all_bins"] is False
    assert result.metadata["count_thresholds"]["active_min_counts"] == 700.0


def main() -> None:
    tests = [
        test_threshold_sentinels_fall_back_to_legacy_minimum,
        test_native_bin_setting_normalizes_to_available_acquisition_bins,
        test_estimators_use_their_own_count_thresholds,
        test_time_bin_contract_separates_acquisition_native_and_histogram_bins,
        test_native_backend_receives_the_requested_bin_crop_and_fit_threshold,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
