from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.demo import create_demo_dataset, create_demo_irf
from flim_app.models import AnalysisSettings
from flim_app.processing import analyze_dataset
from flim_app.rendering import export_roi_analysis
from flim_app.roi import analyse_roi, bounds_to_normalised, clamp_bounds, normalised_to_bounds


def _result(spatial_bin: int = 0):
    dataset = create_demo_dataset(size=20, time_bins=40, seed=21)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    result = analyze_dataset(
        dataset,
        AnalysisSettings(
            method="phasor_phase",
            dt_ns=dataset.dt_ns,
            rep_rate_mhz=dataset.rep_rate_mhz,
            min_photons=10,
            lifetime_min_ns=0,
            lifetime_max_ns=8,
            spatial_bin=spatial_bin,
        ),
        irf,
    )
    return result


def test_roi_coordinate_conversion() -> None:
    shape = (10, 20)
    assert normalised_to_bounds((0.8, 0.7, 0.2, 0.1), shape) == (4, 1, 16, 7)
    assert normalised_to_bounds((1.0, 1.0, 1.0, 1.0), shape) == (19, 9, 20, 10)
    assert normalised_to_bounds((0.25, 0.3, 0.25, 0.3), shape) == (5, 3, 6, 4)
    bounds = (3, 2, 17, 9)
    assert normalised_to_bounds(bounds_to_normalised(bounds, shape), shape) == bounds
    assert normalised_to_bounds(bounds_to_normalised((15, 0, 16, 1), (1, 22)), (1, 22)) == (15, 0, 16, 1)
    for width in range(1, 80):
        for x0 in range(width):
            bounds = (x0, 0, x0 + 1, 1)
            assert normalised_to_bounds(bounds_to_normalised(bounds, (1, width)), (1, width)) == bounds
    try:
        clamp_bounds((100, 100, 110, 110), (10, 10))
    except ValueError:
        pass
    else:
        raise AssertionError("A wholly outside ROI must be rejected.")
    for invalid in ((np.nan, 0, 1, 1), (0, 0, np.inf, 1)):
        try:
            normalised_to_bounds(invalid, shape)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite ROI coordinates must be rejected.")


def test_roi_sum_and_lifetime_statistics() -> None:
    result = _result()
    bounds = (2, 3, 8, 10)
    roi = analyse_roi(result, bounds)
    expected = result.processed_decay[3:10, 2:8].astype(np.float64).sum(axis=(0, 1))
    assert np.allclose(roi.summed_decay, expected)
    assert np.allclose(roi.mean_decay, expected / 42)
    assert roi.selected_pixels == 42
    assert roi.valid_pixels == int(result.valid_mask[3:10, 2:8].sum())
    assert np.isclose(roi.total_photons, result.intensity[3:10, 2:8].sum())
    assert np.isfinite(roi.aggregate_tau_ns)


def test_invalid_lifetime_still_contributes_to_decay() -> None:
    result = _result()
    result.valid_mask[0, 0] = False
    result.lifetime_ns[0, 0] = np.nan
    roi = analyse_roi(result, (0, 0, 2, 2))
    assert roi.selected_pixels == 4
    assert roi.valid_pixels <= 3
    assert np.allclose(roi.summed_decay, result.processed_decay[:2, :2].sum(axis=(0, 1)))


def test_binned_coordinate_metadata_and_export() -> None:
    result = _result(spatial_bin=1)
    roi = analyse_roi(result, (1, 1, 4, 5))
    transform = result.metadata["spatial_transform"]
    assert transform["spatial_bin_level"] == 1
    # Rolling spatial binning keeps the image at source resolution, so
    # there's no processed:source pixel scale to apply any more -- the
    # window width itself is reported separately.
    assert transform["spatial_bin_width"] == 1
    assert transform["spatial_bin_window_width"] == 3
    assert transform["coordinate_convention"].endswith("exclusive")
    with tempfile.TemporaryDirectory() as directory:
        files = export_roi_analysis(result, roi, directory, "sample")
        assert len(files) == 5
        assert all(path.exists() and path.stat().st_size > 0 for path in files)
        summary = json.loads(
            (Path(directory) / "sample_roi_x1-4_y1-5_summary.json").read_text(encoding="utf-8")
        )
        assert summary["bounds_xyxy_processed"] == [1, 1, 4, 5]
        # bin_factor is now always 1 (no resolution change), so the
        # "nominal source" bounds are identical to the processed bounds.
        assert summary["nominal_source_bounds_xyxy"] == [1, 1, 4, 5]
        assert summary["coordinate_convention"].endswith("exclusive")
        assert summary["export_schema_version"] == 8
        assert summary["application"] == "FLIM Studio 0.12.1"
        assert summary["phasor_coordinates"] == "IRF-corrected analysed first-harmonic G/S"
        expected_phasor_valid = int(
            np.count_nonzero(np.isfinite(result.g[1:5, 1:4]) & np.isfinite(result.s[1:5, 1:4]))
        )
        assert summary["phasor"]["valid_pixels"] == expected_phasor_valid
        assert any(path.name.endswith("_phasor.png") for path in files)
        with np.load(Path(directory) / "sample_roi_x1-4_y1-5_analysis.npz") as archive:
            assert np.array_equal(archive["bounds_xyxy"], [1, 1, 4, 5])
            assert archive["summed_decay"].shape == result.time_ns.shape
            assert archive["chi_square"].shape == (4, 3)
            assert np.array_equal(archive["phasor_g"], result.g[1:5, 1:4])
            assert np.array_equal(archive["phasor_s"], result.s[1:5, 1:4])
            assert np.array_equal(archive["tau_phase_ns"], result.tau_phase_ns[1:5, 1:4])
            assert np.array_equal(archive["tau_modulation_ns"], result.tau_modulation_ns[1:5, 1:4])
        table = np.loadtxt(Path(directory) / "sample_roi_x1-4_y1-5_decay.csv", delimiter=",", skiprows=1)
        assert table.shape == (result.time_ns.size, 4)
        second = analyse_roi(result, (2, 2, 5, 6))
        second_files = export_roi_analysis(result, second, directory, "sample")
        assert not set(files).intersection(second_files)
        assert all(path.exists() for path in files + second_files)


def test_roi_sum_unaffected_by_spatial_bin_radius() -> None:
    """A multi-pixel ROI's true photon total must not depend on bin radius.

    Rolling spatial binning makes neighbouring processed pixels overlap
    (each pixel's value already includes its neighbours' counts), so
    naively summing more than one processed pixel together -- any
    ROI/whole-image aggregate wider than 1x1 -- would double- and
    triple-count shared photons if ``analyse_roi`` used the binned cube
    directly. It must instead use ``AnalysisResult.additive_decay`` (the
    pre-bin cube), so the region total stays identical across bin radii.
    """

    dataset = create_demo_dataset(size=20, time_bins=24, seed=77)
    unbinned = analyze_dataset(
        dataset, AnalysisSettings(method="moment", spatial_bin=0, min_photons=0)
    )
    binned = analyze_dataset(
        dataset, AnalysisSettings(method="moment", spatial_bin=2, min_photons=0)
    )
    # Interior bounds, away from the image edge, where every pixel's 5x5
    # window is unclipped -- the scenario where naive summing would have
    # been off by roughly the window area (25x), not just a little.
    bounds = (5, 5, 15, 15)
    roi_unbinned = analyse_roi(unbinned, bounds)
    roi_binned = analyse_roi(binned, bounds)
    assert np.allclose(roi_binned.summed_decay, roi_unbinned.summed_decay)
    assert np.isclose(roi_binned.total_photons, roi_unbinned.total_photons)
    # The two results' own per-pixel processed_decay must actually differ
    # here (bin radius 2 does change the per-pixel maps) -- otherwise this
    # test would pass trivially without exercising the fix at all.
    assert not np.allclose(
        binned.processed_decay[5:15, 5:15].sum(axis=(0, 1)),
        unbinned.processed_decay[5:15, 5:15].sum(axis=(0, 1)),
    )


def test_native_reconvolution_decay_curve() -> None:
    """A native bi-exponential fit must expose an actual fitted decay curve.

    ``analyse_roi`` previously only ever produced a mono-exponential
    "diagnostic guide" line, so the Decay & fit view never showed a curve
    reflecting a bi/tri-exponential native fit.
    """

    dataset = create_demo_dataset(size=8, time_bins=32, seed=95)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    result = analyze_dataset(
        dataset,
        AnalysisSettings(
            method="flimlib_reconvolution",
            fit_min_counts=0,
            model_order=2,
            native_fit_bins=29,
        ),
        irf,
    )
    roi = analyse_roi(result, (0, 0, dataset.x_size, dataset.y_size))
    assert roi.native_fitted_decay is not None
    assert roi.native_fitted_decay.shape == result.time_ns.shape
    finite = np.isfinite(roi.native_fitted_decay)
    assert finite.any()
    assert not finite.all(), "the native fit window is a strict subset of the full acquisition"
    assert "Bi-exponential" in roi.native_fit_label

    with tempfile.TemporaryDirectory() as directory:
        files = export_roi_analysis(result, roi, directory, "native_curve")
        npz_path = next(path for path in files if path.name.endswith("_analysis.npz"))
        with np.load(npz_path) as archive:
            assert "native_reconvolution_fit_to_sum" in archive
        csv_path = next(path for path in files if path.name.endswith("_decay.csv"))
        table = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        assert table.shape == (result.time_ns.size, 5)


def main() -> None:
    tests = [
        test_roi_coordinate_conversion,
        test_roi_sum_and_lifetime_statistics,
        test_invalid_lifetime_still_contributes_to_decay,
        test_binned_coordinate_metadata_and_export,
        test_roi_sum_unaffected_by_spatial_bin_radius,
        test_native_reconvolution_decay_curve,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} ROI tests passed")


if __name__ == "__main__":
    main()
