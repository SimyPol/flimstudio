from __future__ import annotations

import gzip
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.io import load_ics

# Logical cube is always (y, x, t); these are the axis positions in it.
_LOGICAL_INDEX = {"y": 0, "x": 1, "t": 2}
_SIZES = {"y": 4, "x": 6, "t": 8}  # all different, so any axis mix-up shows


def _swarm_style_ics(
    label_order: tuple[str, str, str],
    cube_yxt: np.ndarray,
    *,
    history_dimensions: tuple[int, int, int] | None = None,
) -> bytes:
    """A Swarm-style single-file gzip ICS.

    ``label_order`` is the true axis order, fastest-varying first -- what the
    ``history labels`` line says. Like the real Swarm files, ``layout order``
    deliberately carries the misleading generic names ``x y z`` while
    ``history type`` says ``Time Resolved``.
    """

    sizes = [_SIZES[label] for label in label_order]
    memory = cube_yxt.transpose(
        _LOGICAL_INDEX[label_order[2]], _LOGICAL_INDEX[label_order[1]], _LOGICAL_INDEX[label_order[0]]
    )
    payload = gzip.compress(np.ascontiguousarray(memory, dtype="<u2").tobytes())
    dimensions = history_dimensions if history_dimensions is not None else tuple(sizes)
    extents = ["1.600000E-4"] * 3
    extents[label_order.index("t")] = "1.250000E-8"
    units = ["m"] * 3
    units[label_order.index("t")] = "s"
    header = (
        "\t\n"
        "ics_version\t2.0\n"
        "filename\tswarm\n"
        "layout\tparameters\t4\n"
        "layout\torder\tbits\tx\ty\tz\n"
        f"layout\tsizes\t16\t{sizes[0]}\t{sizes[1]}\t{sizes[2]}\n"
        "representation\tformat\tinteger\n"
        "representation\tsign\tunsigned\n"
        "representation\tcompression\tgzip\n"
        "representation\tbyte_order\t1\t2\n"
        "parameter\tscale\t1.000000\t0.048828\t0.156250\t0.156250\n"
        "parameter\tunits\trelative\tns\tmicrons\tmicrons\n"
        f"history\timage\tdimensions\t{dimensions[0]}\t{dimensions[1]}\t{dimensions[2]}\n"
        f"history\timage\textents\t{extents[0]}\t{extents[1]}\t{extents[2]}\n"
        f"history\timage\tunits\t{units[0]}\t{units[1]}\t{units[2]}\n"
        "history\tlaser\trep\trate\t8.000000E+7\n"
        "history\tauthor\tSWARM-sorter\n"
        "history\ttype\tTime Resolved\n"
        f"history\tlabels\t{' '.join(label_order)}\n"
        f"history\tdimensions\t{dimensions[0]} {dimensions[1]} {dimensions[2]}\n"
        f"history\textents\t{extents[0]} {extents[1]} {extents[2]}\n"
        f"history\tunits\t{' '.join(units)}\n"
        "end\t\n"
    ).encode("ascii")
    return header + payload


def _logical_cube() -> np.ndarray:
    return np.arange(_SIZES["y"] * _SIZES["x"] * _SIZES["t"], dtype=np.uint16).reshape(
        _SIZES["y"], _SIZES["x"], _SIZES["t"]
    )


def _load(data: bytes):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "swarm.ics"
        path.write_bytes(data)
        return load_ics(path)


def test_swarm_header_labels_override_misleading_layout_order() -> None:
    cube = _logical_cube()
    dataset = _load(_swarm_style_ics(("t", "x", "y"), cube))

    # The real Swarm layout: t fastest, then x, then y -> already (y, x, t).
    assert dataset.shape == (_SIZES["y"], _SIZES["x"], _SIZES["t"])
    assert np.array_equal(dataset.decay, cube)
    metadata = dataset.metadata
    assert metadata["layout_axes_as_written"] == ["x", "y", "z"]
    assert metadata["layout_axes"] == ["t", "x", "y"]
    assert metadata["axis_labels_from_history"] is True
    assert metadata["time_bins"] == _SIZES["t"]
    assert metadata["temporal_extent_ns"] == pytest.approx(12.5)
    assert dataset.dt_ns == pytest.approx(12.5 / _SIZES["t"])
    assert dataset.rep_rate_mhz == 80.0


@pytest.mark.parametrize(
    "label_order",
    [("t", "x", "y"), ("t", "y", "x"), ("x", "y", "t"), ("y", "x", "t"), ("x", "t", "y"), ("y", "t", "x")],
)
def test_every_xyt_orientation_in_the_header_loads_to_the_same_yxt_cube(label_order) -> None:
    cube = _logical_cube()
    dataset = _load(_swarm_style_ics(label_order, cube))
    assert dataset.shape == (_SIZES["y"], _SIZES["x"], _SIZES["t"])
    assert np.array_equal(dataset.decay, cube)
    assert dataset.dt_ns == pytest.approx(12.5 / _SIZES["t"])


def test_history_labels_ignored_when_dimensions_disagree_with_layout() -> None:
    # Inconsistent header: keep the layout's own labels, exactly as before
    # this feature existed, rather than trusting a contradictory history block.
    dataset = _load(_swarm_style_ics(("t", "x", "y"), _logical_cube(), history_dimensions=(8, 6, 5)))
    assert dataset.metadata["axis_labels_from_history"] is False
    assert dataset.metadata["layout_axes"] == ["x", "y", "z"]
