from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.lut import (
    DEFAULT_LUT_NAME,
    DEFAULT_LUT_PATH,
    DEFAULT_LUT_SHA256,
    apply_lut,
    bundled_labview_lut,
    load_imagej_lut,
    lut_provenance,
)


def test_bundled_labview_lut_is_exact() -> None:
    payload = DEFAULT_LUT_PATH.read_bytes()
    assert DEFAULT_LUT_NAME == "Reverse Rainbow (DM).lut"
    assert len(payload) == 768
    assert hashlib.sha256(payload).hexdigest() == DEFAULT_LUT_SHA256

    lut = bundled_labview_lut()
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    assert np.array_equal(lut[:, 0], np.frombuffer(payload[:256], dtype=np.uint8))
    assert np.array_equal(lut[:, 1], np.frombuffer(payload[256:512], dtype=np.uint8))
    assert np.array_equal(lut[:, 2], np.frombuffer(payload[512:], dtype=np.uint8))
    assert lut[0].tolist() == [0, 0, 0]
    assert lut[1].tolist() == [255, 3, 0]
    assert lut[128].tolist() == [125, 255, 130]
    assert lut[255].tolist() == [0, 0, 255]


def test_loader_validates_raw_planar_format() -> None:
    planes = np.concatenate(
        (
            np.arange(256, dtype=np.uint8),
            np.arange(255, -1, -1, dtype=np.uint8),
            np.full(256, 17, dtype=np.uint8),
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        valid_path = Path(directory) / "test.lut"
        valid_path.write_bytes(planes.tobytes())
        parsed = load_imagej_lut(valid_path)
        assert parsed[0].tolist() == [0, 255, 17]
        assert parsed[127].tolist() == [127, 128, 17]
        assert parsed[255].tolist() == [255, 0, 17]

        invalid_path = Path(directory) / "short.lut"
        invalid_path.write_bytes(b"\x00" * 767)
        try:
            load_imagej_lut(invalid_path)
        except ValueError as exc:
            assert "exactly 768 bytes" in str(exc)
        else:
            raise AssertionError("A 767-byte LUT was accepted")


def _index_lut() -> np.ndarray:
    index = np.arange(256, dtype=np.uint8)
    return np.column_stack((index, 255 - index, index // 2))


def test_apply_lut_rounding_range_and_inverse_semantics() -> None:
    values = np.asarray(
        [[0.0, 0.25, 0.5, 0.75, 1.0], [-0.1, 1.1, np.nan, np.inf, -np.inf]],
        dtype=np.float64,
    )
    lut = _index_lut()
    missing = (9, 8, 7)

    removed = np.asarray(
        apply_lut(values, lut, 0.0, 1.0, remove_out_of_range=True, nan_colour=missing)
    )
    # LabVIEW round-to-nearest-even: 63.75 -> 64, 127.5 -> 128, 191.25 -> 191.
    assert removed[0, :, 0].tolist() == [0, 64, 128, 191, 255]
    assert np.all(removed[1] == np.asarray(missing, dtype=np.uint8))

    clamped = np.asarray(
        apply_lut(values, lut, 0.0, 1.0, remove_out_of_range=False, nan_colour=missing)
    )
    assert clamped[1, 0].tolist() == lut[0].tolist()
    assert clamped[1, 1].tolist() == lut[255].tolist()
    assert np.all(clamped[1, 2:] == np.asarray(missing, dtype=np.uint8))

    inverse = np.asarray(apply_lut(values[:1], lut, 0.0, 1.0, inverse=True))
    assert inverse[0, :, 0].tolist() == [255, 191, 127, 64, 0]


def test_apply_lut_rejects_ambiguous_inputs() -> None:
    lut = _index_lut()
    for bad_lut in (np.zeros((255, 3)), np.full((256, 3), 256.0)):
        try:
            apply_lut(np.zeros((2, 2)), bad_lut, 0.0, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("An invalid LUT was accepted")
    for limits in ((1.0, 1.0), (2.0, 1.0), (np.nan, 1.0)):
        try:
            apply_lut(np.zeros((2, 2)), lut, *limits)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid LUT limits were accepted: {limits}")


def test_lut_provenance_for_default_and_external_file() -> None:
    default = lut_provenance(None)
    empty = lut_provenance("")
    assert default == empty
    assert default["source"] == "bundled_labview_default"
    assert default["name"] == DEFAULT_LUT_NAME
    assert default["path"] == str(DEFAULT_LUT_PATH.resolve())
    assert default["sha256"] == DEFAULT_LUT_SHA256
    assert default["format"] == "ImageJ raw LUT (768-byte planar R/G/B)"
    assert default["entries"] == 256

    with tempfile.TemporaryDirectory() as directory:
        external_path = Path(directory) / "copy.lut"
        external_path.write_bytes(DEFAULT_LUT_PATH.read_bytes())
        external = lut_provenance(external_path)
        assert external["source"] == "external_file"
        assert external["name"] == "copy.lut"
        assert external["path"] == str(external_path.resolve())
        assert external["sha256"] == DEFAULT_LUT_SHA256


def main() -> None:
    tests = [
        test_bundled_labview_lut_is_exact,
        test_loader_validates_raw_planar_format,
        test_apply_lut_rounding_range_and_inverse_semantics,
        test_apply_lut_rejects_ambiguous_inputs,
        test_lut_provenance_for_default_and_external_file,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} LUT tests passed")


if __name__ == "__main__":
    main()
