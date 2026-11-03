from __future__ import annotations

import gzip
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.io import _parse_ics_lines, load_ics


def _synthetic_header(loops: int = 20, acquisition_time_s: float = 5.0) -> bytes:
    return (
        "\t\n"
        "ics_version\t2.0\n"
        "filename\tcontract\n"
        "layout\tparameters\t4\n"
        "layout\torder\tbits\tt\tx\ty\n"
        "layout\tsizes\t16\t32\t3\t2\n"
        "representation\tformat\tinteger\n"
        "representation\tsign\tunsigned\n"
        "representation\tcompression\tuncompressed\n"
        "representation\tbyte_order\t1\t2\n"
        "parameter\tscale\t1\t0.058629\t1\t1\n"
        "parameter\tunits\ts\tm\tm\n"
        "history\timage\tdimensions\t32\t3\t2\n"
        "history\timage\textents\t1.250000E-8\t3.0E-6\t2.0E-6\n"
        "history\timage\tunits\ts\tm\tm\n"
        "history\tlaser\trep\trate\t8.000000E+7\n"
        f"history scan acq time\t{acquisition_time_s:.6f}\n"
        f"history\tscan\tno.\tof\tloops\t{loops}\n"
        "end\t\n"
    ).encode("ascii")


def test_standardized_ics_metadata() -> None:
    values = np.arange(32 * 3 * 2, dtype="<u2")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "contract.ics"
        path.write_bytes(_synthetic_header() + values.tobytes())
        dataset = load_ics(path)

    metadata = dataset.metadata
    assert dataset.shape == (2, 3, 32)
    assert dataset.dt_ns == 0.390625
    assert dataset.rep_rate_mhz == 80.0
    assert metadata["source_axes"] == ["t", "x", "y"]
    assert metadata["source_shape"] == [32, 3, 2]
    assert metadata["source_shape_by_axis"] == {"t": 32, "x": 3, "y": 2}
    assert metadata["canonical_axes"] == ["y", "x", "t"]
    assert metadata["canonical_shape_yxt"] == [2, 3, 32]
    assert metadata["time_bins"] == 32
    assert metadata["temporal_extent_ns"] == 12.5
    assert metadata["dt_ns"] == 0.390625
    assert metadata["rep_rate_mhz"] == 80.0
    assert metadata["scan_loops"] == 20
    assert metadata["acquisition_time_s"] == 5.0


def test_scan_metadata_spelling_variants() -> None:
    header = (
        "history\tscan\tnumber\tof\tloops\t100\n"
        "history\tscan\tacquisition\ttime\t25.0\n"
        "end\t\n"
    )
    fields, _ = _parse_ics_lines(header)
    assert fields["history scan loops"] == ["100"]
    assert fields["history scan acquisition time"] == ["25.0"]


def test_leica_interrogator_gzip_ics_with_z_labelled_time_axis() -> None:
    """Regression test for a real Leica Interrogator Desktop export.

    That exporter has three quirks none of the other fixtures cover at
    once: (1) ``representation compression gzip`` with the compressed
    payload embedded directly in the ``.ics`` (no ``.ids`` companion,
    and the compressed size is legitimately far smaller than the
    decompressed size); (2) ``history`` fields are only tab-separated
    from their name, with the value list itself space-separated inside
    that one tab-delimited chunk (e.g. ``history\\textents\\t1.0 1.0
    8.0e-9``), rather than every value getting its own tab; and (3) the
    time-gate dimension is labelled ``z`` in ``layout order`` instead of
    ``t``, disambiguated only by ``history type`` saying "Time Resolved".
    """

    sizes = (2, 3, 4)  # y, x, z(=t)
    values = np.arange(int(np.prod(sizes)), dtype="<u2")
    payload = values.tobytes()
    compressed = gzip.compress(payload)
    header = (
        "\t\n"
        "ics_version\t2.0\n"
        "filename\tleica_test\n"
        "layout\tparameters\t4\n"
        "layout\torder\tbits\ty\tx\tz\n"
        "layout\tsizes\t16\t2\t3\t4\n"
        "representation\tformat\tinteger\n"
        "representation\tsign\tunsigned\n"
        "representation\tcompression\tgzip\n"
        "representation\tbyte_order\t1\t2\n"
        "parameter\tscale\t1.000000\t0.100000\t0.100000\t0.100000\n"
        "parameter\tunits\trelative\tm\tm\tm\n"
        "history\ttype\tTime Resolved\n"
        "history\tlabels\tx y t\n"
        "history\textents\t1.0 1.0 8.0e-9\n"
        "history\tunits\tm m s\n"
        "history\tdimensions\t3 2 4\n"
        "end\t\n"
    ).encode("ascii")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "leica.ics"
        path.write_bytes(header + compressed)
        dataset = load_ics(path)  # must not raise "ICS data companion not found"

    metadata = dataset.metadata
    assert dataset.shape == (2, 3, 4)
    assert dataset.decay.dtype == np.dtype("<u2")
    assert np.array_equal(dataset.decay, values.reshape(sizes, order="F"))
    assert dataset.dt_ns == 2.0
    assert dataset.rep_rate_mhz == 80.0
    assert metadata["time_bins"] == 4
    assert metadata["temporal_extent_ns"] == 8.0
    assert metadata["canonical_axes"] == ["y", "x", "t"]
    assert metadata["canonical_shape_yxt"] == [2, 3, 4]


def test_supplied_ics_metadata_when_available() -> None:
    relative_files = {
        Path("Background") / "Background400it_8000Threshold(RACon).ICS": (800, 200.0),
        Path("IRF") / "IRF.ICS": (100, 25.0),
        Path("WGA-CF488A+SPY505") / "WGA-CF488A+SPY505-DNA_FOV_7(RACon).ICS": (20, 5.0),
    }
    # Optional real-data check: set FLIM_STUDIO_TEST_DATA to a folder holding the supplied
    # Background/, IRF/ and WGA-CF488A+SPY505/ acquisitions; skipped silently otherwise.
    configured = os.environ.get("FLIM_STUDIO_TEST_DATA", "")
    root = Path(configured) if configured and Path(configured).is_dir() else None
    if root is None:
        return

    for relative, (expected_loops, expected_time_s) in relative_files.items():
        path = root / relative
        if not path.is_file():
            return
        dataset = load_ics(path)
        metadata = dataset.metadata
        assert dataset.shape == (512, 512, 32)
        assert dataset.decay.dtype == np.dtype("<u2")
        assert dataset.dt_ns == 0.390625
        assert metadata["time_bins"] == 32
        assert metadata["temporal_extent_ns"] == 12.5
        assert metadata["scan_loops"] == expected_loops
        assert metadata["acquisition_time_s"] == expected_time_s
        assert metadata["source_shape"] == [32, 512, 512]
        assert metadata["canonical_shape_yxt"] == [512, 512, 32]


def main() -> None:
    tests = (
        test_standardized_ics_metadata,
        test_scan_metadata_spelling_variants,
        test_leica_interrogator_gzip_ics_with_z_labelled_time_axis,
        test_supplied_ics_metadata_when_available,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
