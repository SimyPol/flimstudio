from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.demo import create_demo_dataset, create_demo_irf
from flim_app.models import AnalysisSettings
from flim_app.points import PointAnalysis, PointSelection
from flim_app.points import MAX_SELECTED_POINTS, add_point, analyse_points, lifetime_label
from flim_app.processing import analyze_dataset
from flim_app.rendering import export_point_selections, histogram_plot, phasor_plot
from flim_app.roi import normalised_to_pixel, pixel_to_normalised_center


def _result():
    dataset = create_demo_dataset(size=18, time_bins=36, seed=31)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    return analyze_dataset(
        dataset,
        AnalysisSettings(
            method="phasor_phase",
            dt_ns=dataset.dt_ns,
            rep_rate_mhz=dataset.rep_rate_mhz,
            min_photons=10,
            spatial_bin=1,
        ),
        irf,
    )


def test_point_coordinate_helpers() -> None:
    assert normalised_to_pixel((0.0, 0.0), (7, 11)) == (0, 0)
    assert normalised_to_pixel((1.0, 1.0), (7, 11)) == (10, 6)
    assert normalised_to_pixel(pixel_to_normalised_center((5, 3), (7, 11)), (7, 11)) == (5, 3)
    for invalid in ((np.nan, 0.2), (0.2, np.inf)):
        try:
            normalised_to_pixel(invalid, (7, 11))
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite point coordinates must be rejected")


def test_add_deduplicate_and_limit() -> None:
    points = []
    for index in range(MAX_SELECTED_POINTS):
        points, added = add_point(points, index, 0)
        assert added
        assert points[-1].identifier == f"P{index + 1}"
    duplicate, added = add_point(points, 0, 0)
    assert not added and duplicate == points
    try:
        add_point(points, 99, 99)
    except ValueError:
        pass
    else:
        raise AssertionError("The ninth point must be rejected")
    reduced = points[:2] + points[3:]
    restored, added = add_point(reduced, 20, 0)
    assert added and restored[-1].identifier == "P3"


def test_point_analysis_and_invalid_retention() -> None:
    result = _result()
    points = []
    points, _ = add_point(points, 1, 2)
    points, _ = add_point(points, 3, 4)
    result.valid_mask[2, 1] = False
    result.lifetime_ns[2, 1] = np.nan
    analysed = analyse_points(result, points)
    assert [point.selection.identifier for point in analysed] == ["P1", "P2"]
    assert not analysed[0].valid
    assert analysed[0].decay.shape == result.time_ns.shape
    assert np.allclose(analysed[0].decay, result.processed_decay[2, 1])
    # Stale expectation from the pre-rolling-window era fixed here: rolling
    # spatial binning (see processing.py's "spatial_transform" metadata
    # comment) keeps every processed pixel at a 1:1 index correspondence
    # with its source pixel regardless of bin level -- it sums a window
    # *around* each pixel rather than reducing the image into blocks, so
    # there's no processed:source scale factor to apply any more.
    assert analysed[0].source_bounds_xyxy_nominal == (1, 2, 2, 3)
    assert analysed[0].phasor_g == result.g[2, 1]
    assert analysed[0].phasor_s == result.s[2, 1]
    assert analysed[0].phasor_valid
    assert lifetime_label(result) == "τφ"


def test_point_export_and_histogram_rules() -> None:
    result = _result()
    points = []
    for xy in ((1, 2), (3, 4), (5, 5)):
        points, _ = add_point(points, *xy)
    analysed = analyse_points(result, points)
    image = histogram_plot(result, bounds=(0, 0, 5, 5), points=analysed)
    assert image.size == (760, 430)
    assert phasor_plot(result, bounds=(0, 0, 5, 5), points=analysed).size == (504, 460)
    with tempfile.TemporaryDirectory() as directory:
        files = export_point_selections(result, analysed, directory, "sample")
        assert len(files) == 5 and all(path.exists() and path.stat().st_size > 0 for path in files)
        summary_path = next(Path(directory).glob("*_summary.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["export_schema_version"] == 8
        assert summary["application"] == "FLIM Studio 0.12.1"
        assert summary["phasor_coordinates"] == "IRF-corrected analysed first-harmonic G/S"
        assert [point["point_id"] for point in summary["points"]] == ["P1", "P2", "P3"]
        assert np.isclose(summary["points"][0]["phasor_g"], result.g[2, 1])
        assert np.isclose(summary["points"][0]["phasor_s"], result.s[2, 1])
        assert summary["points"][0]["phasor_valid"] is True
        assert any(path.name.endswith("_phasor.png") for path in files)
        with np.load(next(Path(directory).glob("*_decays.npz"))) as archive:
            assert archive["processed_decay"].shape == (3, result.time_ns.size)
            assert archive["xy"].tolist() == [[1, 2], [3, 4], [5, 5]]
            assert archive["nominal_source_bounds_xyxy"].shape == (3, 4)
            assert np.allclose(archive["processed_intensity"], archive["total_processed_signal"])
            assert np.allclose(archive["phasor_g"], [result.g[2, 1], result.g[4, 3], result.g[5, 5]])
            assert np.allclose(archive["phasor_s"], [result.s[2, 1], result.s[4, 3], result.s[5, 5]])
            assert archive["phasor_valid"].all()

        invalid_result = _result()
        invalid_result.g[2, 1] = np.nan
        invalid = analyse_points(invalid_result, [points[0]])
        invalid_files = export_point_selections(invalid_result, invalid, directory, "invalid")
        invalid_summary = json.loads(
            next(path for path in invalid_files if path.name.endswith("_summary.json")).read_text(encoding="utf-8")
        )
        assert invalid_summary["points"][0]["phasor_g"] is None
        assert invalid_summary["points"][0]["phasor_valid"] is False
        with np.load(next(path for path in invalid_files if path.name.endswith("_decays.npz"))) as archive:
            assert np.isnan(archive["phasor_g"][0])
            assert not archive["phasor_valid"][0]

        outside_result = _result()
        outside_result.g[2, 1] = 1.2
        outside = analyse_points(outside_result, [points[0]])
        assert outside[0].phasor_valid
        assert phasor_plot(outside_result, points=outside).size == (504, 460)


def test_native_point_export_scalars_components_and_roi() -> None:
    result = _result()
    result.settings = replace(result.settings, method="flimlib_reconvolution", model_order=2)
    point = PointAnalysis(
        selection=PointSelection("P1", 2, 3, "#56B4E9"),
        decay=result.processed_decay[3, 2].copy(),
        map_tau_ns=2.1,
        valid=True,
        processed_intensity=float(result.intensity[3, 2]),
        total_processed_signal=float(result.intensity[3, 2]),
        source_bounds_xyxy_nominal=(6, 9, 9, 12),
        lifetime_label="⟨τ⟩A",
        fit_status=1,
        native_diagnostics={
            "tau_mean_ns": np.asarray(2.1),
            "tau_components_ns": np.asarray([3.2, 0.8]),
            "amplitude_components": np.asarray([120.0, 45.0]),
            "fit_start_bin": np.asarray(4.0),
            "fit_stop_bin": np.asarray(31.0),
            "global_taus_ns": np.asarray([3.2, 0.8]),
        },
    )
    with tempfile.TemporaryDirectory() as directory:
        files = export_point_selections(
            result,
            [point],
            directory,
            "native",
            active_roi_bounds=(1, 2, 5, 6),
        )
        summary = json.loads(next(Path(directory).glob("*_summary.json")).read_text(encoding="utf-8"))
        assert summary["active_roi_bounds_xyxy"] == [1, 2, 5, 6]
        assert summary["points"][0]["native_diagnostics"]["tau_mean_ns"] == 2.1
        with np.load(next(Path(directory).glob("*_decays.npz"))) as archive:
            assert archive["native_tau_components_ns"].shape == (1, 2)
            assert archive["native_amplitude_components"].shape == (1, 2)
            assert archive["native_fit_start_bin"].shape == (1,)
        assert len(files) == 5


def main() -> None:
    tests = [
        test_point_coordinate_helpers,
        test_add_deduplicate_and_limit,
        test_point_analysis_and_invalid_retention,
        test_point_export_and_histogram_rules,
        test_native_point_export_scalars_components_and_roi,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} point-selection tests passed")


if __name__ == "__main__":
    main()
