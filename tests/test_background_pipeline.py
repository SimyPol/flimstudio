from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.models import AnalysisSettings, FLIMDataset
from flim_app.processing import analyze_dataset, resolve_background_multiplier


def _dataset(
    values: np.ndarray,
    name: str,
    *,
    dt_ns: float = 0.390625,
    metadata: dict[str, object] | None = None,
) -> FLIMDataset:
    return FLIMDataset(
        np.asarray(values, dtype=np.float32),
        name=name,
        dt_ns=dt_ns,
        rep_rate_mhz=80.0,
        metadata=dict(metadata or {}),
    )


def _settings(**changes: object) -> AnalysisSettings:
    values: dict[str, object] = {
        "method": "mono_exponential",
        "dt_ns": 0.390625,
        "rep_rate_mhz": 80.0,
        "min_photons": 0.0,
        "lifetime_min_ns": -1_000.0,
        "lifetime_max_ns": 1_000.0,
        "background_scale_mode": "auto",
        "background_multiplier": 0.025,
        "background_auto_fallback": 0.025,
    }
    values.update(changes)
    return AnalysisSettings(**values)


def test_auto_multiplier_uses_scan_loop_ratio() -> None:
    sample = _dataset(
        np.full((2, 3, 8), 100.0),
        "sample",
        metadata={"scan_loops": 20, "acquisition_time_s": 5.0},
    )
    background = _dataset(
        np.full((2, 3, 8), 40.0),
        "background",
        metadata={"scan_loops": 800, "acquisition_time_s": 200.0},
    )
    multiplier, details = resolve_background_multiplier(sample, background, _settings())
    assert multiplier == 0.025
    assert details["basis"] == "scan_loops"
    assert details["sample_normalization_value"] == 20
    assert details["background_normalization_value"] == 800

    result = analyze_dataset(sample, _settings(), background=background)
    assert np.allclose(result.processed_decay, 99.0)
    correction = result.metadata["background_correction"]
    assert correction["applied_multiplier"] == 0.025
    assert correction["sample_time_bins"] == 8
    assert correction["background_time_bins"] == 8
    assert correction["spatial_handling"] == "per_pixel_cube"
    assert correction["row_interpolated_separately"] is False


def test_auto_multiplier_supports_legacy_metadata_and_time_fallback() -> None:
    sample = _dataset(
        np.full((1, 1, 8), 10.0),
        "sample",
        metadata={"history scan acq time": "5.000000"},
    )
    background = _dataset(
        np.full((1, 1, 8), 8.0),
        "background",
        metadata={"history scan acq time": "200.000000"},
    )
    multiplier, details = resolve_background_multiplier(sample, background, _settings())
    assert multiplier == 0.025
    assert details["basis"] == "acquisition_time_s"
    assert details["sample_metadata_key"] == "history scan acq time"


def test_manual_multiplier_overrides_acquisition_metadata() -> None:
    sample = _dataset(
        np.full((1, 2, 8), 20.0),
        "sample",
        metadata={"scan_loops": 20},
    )
    background = _dataset(
        np.full((1, 2, 8), 10.0),
        "background",
        metadata={"scan_loops": 800},
    )
    settings = _settings(background_scale_mode="manual", background_multiplier=0.5)
    multiplier, details = resolve_background_multiplier(sample, background, settings)
    assert multiplier == 0.5
    assert details["basis"] == "manual"
    result = analyze_dataset(sample, settings, background=background)
    assert np.allclose(result.processed_decay, 15.0)


def test_automatic_fallback_for_array_background() -> None:
    sample = _dataset(np.full((1, 1, 8), 10.0), "sample")
    background = np.full((1, 1, 8), 40.0, dtype=np.float32)
    settings = _settings(background_auto_fallback=0.1)
    multiplier, details = resolve_background_multiplier(sample, background, settings)
    assert multiplier == 0.1
    assert details["basis"] == "automatic_fallback"
    result = analyze_dataset(sample, settings, background=background)
    assert np.allclose(result.processed_decay, 6.0)


def test_subtraction_and_clipping_precede_timing_shift() -> None:
    sample_values = np.array([[[10, 20, 30, 40, 50, 60, 70, 80]]], dtype=np.float32)
    background_values = np.array([[[1, 2, 3, 4, 5, 6, 7, 100]]], dtype=np.float32)
    sample = _dataset(sample_values, "sample")
    background = _dataset(background_values, "background")
    settings = _settings(
        background_scale_mode="manual",
        background_multiplier=1.0,
        timing_offset_bins=1.0,
    )
    result = analyze_dataset(sample, settings, background=background)
    # First subtract and clip: [9,18,27,36,45,54,63,0].  A +1-bin
    # analysis offset then samples bins 1..7 and zero-fills the final bin.
    assert np.array_equal(
        result.processed_decay[0, 0],
        np.array([18, 27, 36, 45, 54, 63, 0, 0], dtype=np.float32),
    )
    correction = result.metadata["background_correction"]
    assert correction["negative_bins_clipped"] == 1
    assert correction["preprocessing_order"].startswith(
        "background subtraction and zero clipping -> timing shift"
    )


def test_shared_background_trace_and_scalar_counts_are_combined() -> None:
    sample = _dataset(np.full((2, 3, 8), 10.0), "sample")
    trace = np.arange(1, 9, dtype=np.float32)
    settings = _settings(
        background_scale_mode="manual",
        background_multiplier=0.5,
        background_counts=1.0,
    )
    result = analyze_dataset(sample, settings, background=trace)
    expected = 10.0 - 0.5 * trace - 1.0
    assert np.allclose(result.processed_decay, expected[None, None, :])
    assert result.metadata["background_correction"]["spatial_handling"] == "shared_time_trace"


def test_background_time_bin_mismatch_is_rejected() -> None:
    sample = _dataset(np.ones((2, 2, 8)), "sample")
    background = _dataset(np.ones((2, 2, 7)), "background")
    try:
        analyze_dataset(sample, _settings(), background=background)
    except ValueError as exc:
        assert "time-bin count" in str(exc)
    else:
        raise AssertionError("A mismatched background bin count must fail.")


def test_background_time_bin_width_mismatch_is_rejected() -> None:
    sample = _dataset(np.ones((2, 2, 8)), "sample", dt_ns=0.390625)
    background = _dataset(np.ones((2, 2, 8)), "background", dt_ns=0.5)
    try:
        analyze_dataset(sample, _settings(), background=background)
    except ValueError as exc:
        assert "time-bin width" in str(exc)
    else:
        raise AssertionError("A mismatched background time-bin width must fail.")


def test_invalid_background_shape_is_rejected() -> None:
    sample = _dataset(np.ones((2, 2, 8)), "sample")
    background = np.ones((2, 8), dtype=np.float32)
    try:
        analyze_dataset(sample, _settings(), background=background)
    except ValueError as exc:
        assert "scalar" in str(exc) and "three-dimensional" in str(exc)
    else:
        raise AssertionError("An ambiguous 2-D background must fail.")


def test_new_settings_validate_and_preserve_lut_contract() -> None:
    normalized = _settings(
        lifetime_lut_path="  custom.lut  ",
        lifetime_lut_inverse=1,
        lifetime_lut_remove_out_of_range=0,
    ).normalized(8)
    assert normalized.lifetime_lut_path == "custom.lut"
    assert normalized.lifetime_lut_inverse is True
    assert normalized.lifetime_lut_remove_out_of_range is False
    try:
        _settings(background_scale_mode="unknown").normalized(8)
    except ValueError as exc:
        assert "Background scale mode" in str(exc)
    else:
        raise AssertionError("An unknown background scale mode must fail.")


def test_bundled_lut_metadata_is_portable() -> None:
    dataset = _dataset(
        np.full((1, 1, 8), 100.0),
        "portable-lut-metadata",
        metadata={"scan_loops": 20, "acquisition_time_s": 5.0},
    )
    result = analyze_dataset(dataset, _settings(background_scale_mode="manual"))
    provenance = result.metadata["lifetime_lut"]
    assert provenance["source"] == "bundled_labview_default"
    assert provenance["path"] is None
    assert provenance["name"] == "Reverse Rainbow (DM).lut"
    assert len(provenance["sha256"]) == 64


def main() -> None:
    tests = [
        test_auto_multiplier_uses_scan_loop_ratio,
        test_auto_multiplier_supports_legacy_metadata_and_time_fallback,
        test_manual_multiplier_overrides_acquisition_metadata,
        test_automatic_fallback_for_array_background,
        test_subtraction_and_clipping_precede_timing_shift,
        test_shared_background_trace_and_scalar_counts_are_combined,
        test_background_time_bin_mismatch_is_rejected,
        test_background_time_bin_width_mismatch_is_rejected,
        test_invalid_background_shape_is_rejected,
        test_new_settings_validate_and_preserve_lut_contract,
        test_bundled_lut_metadata_is_portable,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} background pipeline tests passed")


if __name__ == "__main__":
    main()
