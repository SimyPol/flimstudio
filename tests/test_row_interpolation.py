from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.models import AnalysisSettings, FLIMDataset
from flim_app.processing import analyze_dataset
from flim_app.rendering import export_result, interpolation_map_image
from flim_app.row_interpolation import (
    LABVIEW_MAIN_ROWS,
    LABVIEW_MIRRORED,
    RowInterpolationMap,
    format_row_spec,
    interpolate_mapped_rows,
    interpolate_mapped_columns,
    column_interpolation_coverage,
    interpolation_coverage,
    labview_main_preset,
    load_row_map,
    parse_row_spec,
    save_row_map,
)


def test_operational_labview_preset_and_orientation() -> None:
    mapping = labview_main_preset()
    assert mapping.rows == tuple(sorted(LABVIEW_MAIN_ROWS))
    assert mapping.radius == 2
    assert mapping.coordinate_mode == LABVIEW_MIRRORED
    assert mapping.resolve_rows(512) == (
        74, 119, 162, 172, 194, 223, 270, 274, 281, 296,
        316, 379, 412, 422, 433, 442, 457, 481, 508,
    )
    for height in (160, 438, 500, 511, 513, 1024):
        try:
            mapping.resolve_rows(height)
        except ValueError as exc:
            assert "requires a 512-row" in str(exc)
        else:
            raise AssertionError(f"The 512-row preset must not be remirrored onto height {height}.")


def test_parse_format_and_round_trip() -> None:
    assert parse_row_spec("3, 5; 8-10\n# detector note\n5") == (3, 5, 8, 9, 10)
    assert parse_row_spec("1 3-4", index_base=1) == (0, 2, 3)
    assert format_row_spec((10, 8, 9, 3, 3)) == "3, 8-10"
    for invalid in ("2.5", "nan", "5-3", "-1"):
        try:
            parse_row_spec(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid row specification {invalid!r} to fail.")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.json"
        original = RowInterpolationMap((3, 8, 9, 10), 2, LABVIEW_MIRRORED, "Calibration A")
        save_row_map(path, original)
        loaded = load_row_map(path)
        assert loaded.rows == (3, 8, 9, 10)
        assert loaded.coordinate_mode == LABVIEW_MIRRORED
        assert loaded.name == "Calibration A"
        assert len(loaded.sha256) == 64
        assert json.loads(path.read_text())["axis"] == "source_y"
        one_based = RowInterpolationMap((0, 2), declared_index_base=1, name="entered one-based")
        save_row_map(path, one_based)
        loaded_one_based = load_row_map(path)
        assert loaded_one_based.rows == (0, 2)
        assert loaded_one_based.declared_index_base == 1
        preset = labview_main_preset()
        for suffix in (".json", ".csv", ".txt"):
            target = Path(directory) / f"preset{suffix}"
            save_row_map(target, preset)
            reloaded = load_row_map(target)
            assert reloaded.rows == preset.rows
            assert reloaded.coordinate_mode == preset.coordinate_mode
            assert reloaded.radius == preset.radius
            assert reloaded.expected_source_height == 512
            assert reloaded.resolve_rows(512) == preset.resolve_rows(512)


def test_safe_interpolation_is_simultaneous_and_full_trace() -> None:
    y = np.arange(7, dtype=np.float32)[:, None, None]
    x = np.arange(3, dtype=np.float32)[None, :, None] * 0.1
    t = np.arange(4, dtype=np.float32)[None, None, :] * 0.01
    cube = y + x + t
    cube[2:4] = 9999
    mapping = RowInterpolationMap((2, 3), radius=1)
    corrected, details = interpolate_mapped_rows(cube, mapping)
    expected_2 = (2.0 / 3.0) * (1 + x[0] + t[0]) + (1.0 / 3.0) * (4 + x[0] + t[0])
    expected_3 = (1.0 / 3.0) * (1 + x[0] + t[0]) + (2.0 / 3.0) * (4 + x[0] + t[0])
    assert np.allclose(corrected[2], expected_2)
    assert np.allclose(corrected[3], expected_3)
    assert np.array_equal(corrected[[0, 1, 4, 5, 6]], cube[[0, 1, 4, 5, 6]])
    assert details["canonical_target_rows_y"] == [2, 3]
    assert details["corrected_traces"] == 6
    reverse, _ = interpolate_mapped_rows(cube, RowInterpolationMap((3, 2), radius=1))
    assert np.array_equal(corrected, reverse)
    # A wider donor radius must preserve a linear spatial gradient, including
    # through a consecutive bad-row run; blend by donor centroids, not run index.
    linear = np.broadcast_to(np.arange(7, dtype=np.float32)[:, None, None], (7, 2, 3)).copy()
    linear[2:4] = 999
    linear_corrected, _ = interpolate_mapped_rows(linear, RowInterpolationMap((2, 3), radius=2))
    assert np.allclose(linear_corrected[2], 2.0)
    assert np.allclose(linear_corrected[3], 3.0)


def test_edges_radius_and_coverage() -> None:
    cube = np.broadcast_to(np.arange(6, dtype=np.float32)[:, None, None], (6, 2, 3)).copy()
    cube[[0, 5]] = 100
    corrected, details = interpolate_mapped_rows(cube, RowInterpolationMap((0, 5), radius=2))
    assert np.allclose(corrected[0], 1.5)
    assert np.allclose(corrected[5], 3.5)
    assert {run["mode"] for run in details["runs"]} == {"lower_edge_copy", "upper_edge_copy"}
    mask, coverage = interpolation_coverage(5, 7, (1, 2, 4), 2)
    assert mask.tolist() == [False, True, True, False, True]
    assert coverage.shape == (2, 3)
    assert np.allclose(coverage[:, 0], (0.5, 0.5))


def test_column_interpolation_and_coverage() -> None:
    cube = np.broadcast_to(np.arange(7, dtype=np.float32)[None, :, None], (3, 7, 4)).copy()
    cube[:, 2:4] = 999
    corrected, details = interpolate_mapped_columns(cube, (2, 3), radius=2)
    assert np.allclose(corrected[:, 2], 2.0)
    assert np.allclose(corrected[:, 3], 3.0)
    assert np.array_equal(corrected[:, [0, 1, 4, 5, 6]], cube[:, [0, 1, 4, 5, 6]])
    assert details["canonical_target_columns_x"] == [2, 3]
    mask, coverage = column_interpolation_coverage(5, 7, (1, 2, 4), 3)
    assert mask.tolist() == [False, True, True, False, True, False, False]
    assert coverage.shape == (1, 2)
    assert np.allclose(coverage[0], (2 / 3, 1 / 3))


def test_pipeline_metadata_reference_correction_and_export() -> None:
    nt = 16
    time = np.arange(nt, dtype=np.float32)
    cube = np.empty((6, 4, nt), dtype=np.float32)
    for row in range(6):
        cube[row] = 80 * np.exp(-time / (2.0 + 0.1 * row)) + row
    cube[2] = 999
    irf = np.broadcast_to(np.exp(-0.5 * ((time - 2) / 0.7) ** 2), cube.shape).astype(np.float32).copy()
    irf[2] = 888
    background = np.ones_like(cube)
    background[2] = 777
    dataset = FLIMDataset(cube, "mapped", dt_ns=0.2)
    settings = AnalysisSettings(
        method="mono_exponential",
        dt_ns=0.2,
        min_photons=0,
        lifetime_min_ns=0,
        lifetime_max_ns=20,
        spatial_bin=0,
        fit_irf_mode=1,
        interpolate_rows=True,
        interpolation_rows=(2,),
        interpolation_radius=1,
    )
    result = analyze_dataset(dataset, settings, irf, background)
    meta = result.metadata["row_interpolation"]
    assert meta["applied"] is True
    assert meta["canonical_target_rows_y"] == [2]
    assert meta["reference_correction"] == {"background": True, "irf": True}
    assert result.fit_diagnostics["interpolated_source_row_mask"].shape == (6,)
    assert result.fit_diagnostics["interpolated_result_fraction"].shape == result.intensity.shape
    assert np.allclose(result.intensity, result.processed_decay.sum(axis=-1))
    with tempfile.TemporaryDirectory() as directory:
        files = export_result(result, directory, "mapped")
        with np.load(Path(directory) / "mapped_analysis.npz") as archive:
            assert "interpolated_source_row_mask" in archive
            assert "interpolated_result_fraction" in archive
        summary = json.loads((Path(directory) / "mapped_summary.json").read_text())
        assert summary["metadata"]["row_interpolation"]["algorithm"] == "safe_linear_v1"
        assert all(path.exists() for path in files)
    image = interpolation_map_image((6, 4), (2,), base_intensity=cube.sum(axis=-1))
    assert image.size == (720, 500)


def main() -> None:
    tests = [
        test_operational_labview_preset_and_orientation,
        test_parse_format_and_round_trip,
        test_safe_interpolation_is_simultaneous_and_full_trace,
        test_edges_radius_and_coverage,
        test_column_interpolation_and_coverage,
        test_pipeline_metadata_reference_correction_and_export,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} row-interpolation tests passed")


if __name__ == "__main__":
    main()
