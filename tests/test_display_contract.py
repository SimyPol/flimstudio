from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.lut import bundled_labview_lut
from flim_app.models import AnalysisResult, AnalysisSettings
from flim_app.rendering import composite_image, lifetime_display_range, lifetime_image


def _result(
    lifetime: np.ndarray,
    intensity: np.ndarray,
    *,
    remove: bool = True,
    scale_mode: str = "manual",
) -> AnalysisResult:
    shape = lifetime.shape
    settings = AnalysisSettings(
        lifetime_min_ns=0.5,
        lifetime_max_ns=4.5,
        lifetime_lut_remove_out_of_range=remove,
        lifetime_scale_mode=scale_mode,
    )
    finite = np.isfinite(lifetime)
    return AnalysisResult(
        dataset_name="display-contract",
        intensity=np.asarray(intensity, dtype=np.float64),
        lifetime_ns=np.asarray(lifetime, dtype=np.float64),
        valid_mask=finite,
        g=np.zeros(shape),
        s=np.zeros(shape),
        tau_phase_ns=np.asarray(lifetime, dtype=np.float64),
        tau_modulation_ns=np.asarray(lifetime, dtype=np.float64),
        chi_square=np.zeros(shape),
        mean_decay=np.ones(4),
        mean_fit=np.ones(4),
        time_ns=np.arange(4, dtype=float),
        global_tau_ns=2.0,
        settings=settings,
        processed_decay=np.ones(shape + (4,)),
    )


def test_exact_lifetime_lut_golden_indices() -> None:
    values = np.asarray([[0.4, 0.5, 1.5, 2.5, 3.5, 4.5, 4.6]])
    image = np.asarray(lifetime_image(_result(values, np.ones_like(values))))
    lut = bundled_labview_lut()
    assert image[0, 0].tolist() == [5, 11, 21]
    assert np.array_equal(image[0, 1:6], lut[[0, 64, 128, 191, 255]])
    assert image[0, 6].tolist() == [5, 11, 21]


def test_display_range_does_not_define_numerical_validity() -> None:
    values = np.asarray([[0.25, 2.0, 6.0]])
    result = _result(values, np.ones_like(values))
    assert result.valid_mask.all()
    assert np.array_equal(result.lifetime_ns, values)
    rendered = np.asarray(lifetime_image(result))
    assert rendered[0, 0].tolist() == [5, 11, 21]
    assert rendered[0, 2].tolist() == [5, 11, 21]


def test_labview_hsv_composite_orientation_and_linear_intensity() -> None:
    lifetimes = np.asarray([[0.5, 4.5], [0.5, 4.5]])
    intensities = np.asarray([[0.0, 0.0], [10.0, 10.0]])
    composite = np.asarray(composite_image(_result(lifetimes, intensities)))
    assert composite[0].max() == 0
    # Short lifetime is hue 0 (red); long lifetime is hue 0.7 (blue/violet).
    assert composite[1, 0, 0] > 245 and composite[1, 0, 1] < 10 and composite[1, 0, 2] < 10
    assert composite[1, 1, 2] > composite[1, 1, 0] > composite[1, 1, 1]


def test_composite_handles_invalid_lifetime_without_crashing() -> None:
    lifetimes = np.asarray([[np.nan, 2.5]])
    result = _result(lifetimes, np.asarray([[5.0, 10.0]]))
    composite = np.asarray(composite_image(result))
    assert composite[0, 0].tolist() == [5, 11, 21]
    assert composite[0, 1].max() > 0


def test_auto_lifetime_scaling_is_robust_and_display_only() -> None:
    values = np.arange(1.0, 101.0).reshape(10, 10)
    result = _result(values, np.ones_like(values), scale_mode="auto")
    lo, hi = lifetime_display_range(result)
    assert np.isclose(lo, np.percentile(values, 1))
    assert np.isclose(hi, np.percentile(values, 99))
    before = result.lifetime_ns.copy()
    rendered = np.asarray(lifetime_image(result))
    # Robust-range outliers saturate in auto mode; they are not hidden as invalid.
    assert rendered[0, 0].tolist() != [5, 11, 21]
    assert rendered[-1, -1].tolist() != [5, 11, 21]
    assert np.array_equal(result.lifetime_ns, before)


def main() -> None:
    tests = [
        test_exact_lifetime_lut_golden_indices,
        test_display_range_does_not_define_numerical_validity,
        test_labview_hsv_composite_orientation_and_linear_intensity,
        test_composite_handles_invalid_lifetime_without_crashing,
        test_auto_lifetime_scaling_is_robust_and_display_only,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} display-contract tests passed")


if __name__ == "__main__":
    main()
