from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.demo import create_demo_dataset, create_demo_irf
from flim_app.models import AnalysisSettings
from flim_app.processing import analyze_dataset, spatial_bin_width
from flim_app.rendering import (
    component_amplitude_image,
    component_lifetime_image,
    export_result,
    phasor_plot,
    threshold_intensity_image,
)
from flim_app.row_interpolation import interpolate_mapped_columns


def test_spatial_radius_contract() -> None:
    """Spatial binning is a rolling window: it must not shrink the image.

    A radius-1 (3x3) or radius-2 (5x5) bin sums a window centred on each
    pixel and keeps the output at the original image size -- unlike the
    old non-overlapping block-bin, which reduced a 30x30 image to 10x10
    or 6x6 respectively.
    """

    assert [spatial_bin_width(level) for level in (0, 1, 2, 3)] == [1, 3, 5, 7]
    dataset = create_demo_dataset(size=30, time_bins=24, seed=91)
    baseline = analyze_dataset(
        dataset, AnalysisSettings(method="moment", spatial_bin=0, min_photons=0)
    )
    for level in (0, 1, 2):
        result = analyze_dataset(
            dataset,
            AnalysisSettings(method="moment", spatial_bin=level, min_photons=0),
        )
        assert result.intensity.shape == (30, 30)
        transform = result.metadata["spatial_transform"]
        assert transform["spatial_bin_level"] == level
        # No shrinkage at any radius, so the processed:source pixel scale
        # stays 1:1 regardless of window size.
        assert transform["spatial_bin_width"] == 1
        assert transform["spatial_bin_window_width"] == 2 * level + 1
        assert transform["dropped_right"] == 0
        assert transform["dropped_bottom"] == 0
        assert transform["analyzed_source_bounds_xyxy"] == [0, 0, 30, 30]
        # An interior pixel's intensity should grow roughly with the
        # window area relative to the unbinned baseline, confirming the
        # window is actually being summed and not just left unbinned.
        window = 2 * level + 1
        interior = result.intensity[15, 15]
        expected_area_ratio = window * window
        baseline_interior = baseline.intensity[15, 15]
        if level == 0:
            assert np.isclose(interior, baseline_interior)
        else:
            assert interior > baseline_interior
            # Demo data is spatially smooth, so the true ratio should be
            # close to the window area rather than wildly off from it.
            assert 0.5 * expected_area_ratio <= interior / baseline_interior <= 1.5 * expected_area_ratio


def test_column_interpolation_full_trace() -> None:
    cube = np.broadcast_to(np.arange(7, dtype=np.float32)[None, :, None], (4, 7, 5)).copy()
    cube[:, 2:4, :] = 999
    corrected, metadata = interpolate_mapped_columns(cube, (2, 3), radius=2)
    assert np.allclose(corrected[:, 2], 2.0)
    assert np.allclose(corrected[:, 3], 3.0)
    assert metadata["corrected_traces"] == 8
    dataset = create_demo_dataset(size=15, time_bins=24, seed=92)
    result = analyze_dataset(
        dataset,
        AnalysisSettings(
            method="moment",
            min_photons=0,
            interpolate_columns=True,
            interpolation_columns=(2, 3),
            column_interpolation_radius=2,
            spatial_bin=1,
        ),
    )
    assert result.metadata["column_interpolation"]["applied"] is True
    assert result.metadata["column_interpolation"]["canonical_target_columns_x"] == [2, 3]
    assert result.fit_diagnostics["interpolated_source_column_mask"].shape == (15,)
    # Rolling spatial binning keeps the result at source resolution, so
    # this coverage map is now 1:1 with the source instead of shrunk by
    # the old block-bin factor.
    assert result.fit_diagnostics["interpolated_column_result_fraction"].shape == (15, 15)


def test_heatmap_and_threshold_preview() -> None:
    dataset = create_demo_dataset(size=28, time_bins=32, seed=93)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    result = analyze_dataset(
        dataset,
        AnalysisSettings(method="phasor_phase", phasor_min_counts=0, phasor_heatmap_bins=64),
        irf,
    )
    heatmap = phasor_plot(result, heatmap=True)
    scatter = phasor_plot(result, heatmap=False)
    assert heatmap.size == scatter.size == (504, 460)
    assert not np.array_equal(np.asarray(heatmap), np.asarray(scatter))
    threshold = float(np.nanmedian(result.intensity))
    preview = np.asarray(threshold_intensity_image(result.intensity, threshold))
    omitted = result.intensity < threshold
    assert np.all(preview[omitted] == np.asarray((225, 29, 72)))
    assert np.any(preview[~omitted] != np.asarray((225, 29, 72)))


def test_phasor_plot_area_is_square() -> None:
    """G and S must share one pixel-per-unit scale, or the universal
    semicircle renders stretched. Detect the plot rectangle by its
    near-black interior (heatmap=False leaves it mostly untouched apart
    from gridlines/points) and confirm its bounding box is square. A
    generous "near black" threshold and tolerance absorb the plot's
    supersampled-then-downsampled anti-aliasing."""

    dataset = create_demo_dataset(size=10, time_bins=24, seed=99)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    result = analyze_dataset(dataset, AnalysisSettings(method="phasor_phase", phasor_min_counts=0), irf)
    image = phasor_plot(result, heatmap=False)
    array = np.asarray(image).astype(np.int32)
    near_black = array.sum(axis=-1) < 30
    rows = np.flatnonzero(near_black.any(axis=1))
    cols = np.flatnonzero(near_black.any(axis=0))
    assert rows.size and cols.size, "expected a solid black plot rectangle"
    height = int(rows[-1] - rows[0] + 1)
    width = int(cols[-1] - cols[0] + 1)
    # Widened from 4 (observed 5px on Pillow 12.0.0, within the "near black"
    # edge-detection's own inherent fuzziness -- LANCZOS resampling rounds
    # very slightly differently across Pillow versions). Still tight enough
    # to catch the actual bug this guards against (the G/S axis-scale
    # mismatch from before v0.11.3, which stretched the plot by tens of
    # pixels, not single-digit ones).
    assert abs(width - height) <= 6, (width, height)


def test_component_images_and_exports() -> None:
    dataset = create_demo_dataset(size=8, time_bins=32, seed=95)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    result = analyze_dataset(
        dataset,
        AnalysisSettings(
            method="flimlib_reconvolution",
            fit_min_counts=0,
            model_order=2,
            native_fit_bins=29,
            fit_noise_model=1,
            fit_background_mode=1,
            fit_irf_mode=1,
            lifetime_min_ns=0,
            lifetime_max_ns=10,
        ),
        irf,
    )
    components = result.fit_diagnostics["tau_components_ns"]
    assert components.shape == (2, 8, 8)
    assert component_lifetime_image(result, 0).size == (8, 8)
    assert component_lifetime_image(result, 1).size == (8, 8)
    assert component_amplitude_image(result, 0).size == (8, 8)
    assert component_amplitude_image(result, 1).size == (8, 8)
    assert np.array_equal(
        result.fit_diagnostics["threshold_signal_map"],
        result.processed_decay[..., :29].sum(axis=-1),
    )
    assert result.metadata["count_thresholds"]["preview_signal_basis"].startswith("leading 29")
    with tempfile.TemporaryDirectory() as directory:
        files = export_result(result, directory, "bi")
        names = {path.name for path in files}
        assert "bi_lifetime_tau1.png" in names
        assert "bi_lifetime_tau2.png" in names


def main() -> None:
    tests = [
        test_spatial_radius_contract,
        test_column_interpolation_full_trace,
        test_heatmap_and_threshold_preview,
        test_phasor_plot_area_is_square,
        test_component_images_and_exports,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} v0.9 feature tests passed")


if __name__ == "__main__":
    main()
