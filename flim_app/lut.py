"""LabVIEW-compatible lifetime look-up table support.

The original application uses an ImageJ-style raw LUT: 256 red bytes,
followed by 256 green bytes and 256 blue bytes.  This module deliberately
keeps the file format and value-to-index conversion small and explicit so the
display colours are reproducible independently of plotting libraries.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ._paths import app_root


DEFAULT_LUT_NAME = "Reverse Rainbow (DM).lut"
DEFAULT_LUT_PATH = app_root() / "assets" / DEFAULT_LUT_NAME
DEFAULT_LUT_SHA256 = "5c78c07426172b682743a2d2e7e18d1ae4e27c8ddcf018683ff857838dcb2a87"
DEFAULT_LUT_SOURCE = "bundled_labview_default"

_LUT_ENTRIES = 256
_LUT_BYTES = _LUT_ENTRIES * 3
_LUT_FORMAT = "ImageJ raw LUT (768-byte planar R/G/B)"


def _resolved_lut_path(path: str | Path | None) -> tuple[Path, bool]:
    bundled = path is None or (isinstance(path, str) and not path.strip())
    selected = DEFAULT_LUT_PATH if bundled else Path(path)  # type: ignore[arg-type]
    return selected.expanduser().resolve(), bundled


def load_imagej_lut(path: str | Path) -> np.ndarray:
    """Load a 768-byte planar ImageJ LUT as a ``(256, 3)`` uint8 array.

    The file layout is ``R[0:256], G[0:256], B[0:256]``.  Header-bearing or
    interleaved palette formats are intentionally rejected rather than being
    guessed, because silently accepting one would change scientific figures.
    """

    source = Path(path).expanduser().resolve()
    payload = source.read_bytes()
    if len(payload) != _LUT_BYTES:
        raise ValueError(
            f"ImageJ LUT must contain exactly {_LUT_BYTES} bytes "
            f"(256 red, 256 green, 256 blue); {source} contains {len(payload)}."
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(3, _LUT_ENTRIES).T.copy()


def bundled_labview_lut() -> np.ndarray:
    """Return the exact Reverse Rainbow LUT saved with the LabVIEW workflow."""

    payload = DEFAULT_LUT_PATH.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DEFAULT_LUT_SHA256:
        raise ValueError(
            "Bundled LabVIEW LUT failed its SHA-256 integrity check: "
            f"expected {DEFAULT_LUT_SHA256}, received {digest}."
        )
    return load_imagej_lut(DEFAULT_LUT_PATH)


def lut_provenance(path: str | Path | None = None) -> dict[str, Any]:
    """Describe and validate a bundled or user-selected lifetime LUT."""

    selected, bundled = _resolved_lut_path(path)
    if bundled:
        bundled_labview_lut()
    else:
        load_imagej_lut(selected)
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    return {
        "source": DEFAULT_LUT_SOURCE if bundled else "external_file",
        "name": selected.name,
        "path": str(selected),
        "sha256": digest,
        "format": _LUT_FORMAT,
        "entries": _LUT_ENTRIES,
    }


def _validated_lut(lut: np.ndarray) -> np.ndarray:
    table = np.asarray(lut)
    if table.shape != (_LUT_ENTRIES, 3):
        raise ValueError(f"LUT must have shape (256, 3); received {table.shape}.")
    if not np.issubdtype(table.dtype, np.number):
        raise TypeError("LUT entries must be numeric RGB values.")
    if not np.all(np.isfinite(table)) or np.any(table < 0) or np.any(table > 255):
        raise ValueError("LUT entries must be finite values in the inclusive range 0..255.")
    return table.astype(np.uint8, copy=False)


def apply_lut(
    values: np.ndarray,
    lut: np.ndarray,
    vmin: float,
    vmax: float,
    remove_out_of_range: bool = True,
    inverse: bool = False,
    nan_colour: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Map a 2-D value array through a LabVIEW-compatible 256-entry LUT.

    Values are converted to indices with
    ``round_to_even((value-vmin)/(vmax-vmin) * 255)``.  This mirrors the
    explicit LabVIEW round-to-nearest step before its U8 table index.  The
    endpoints are inclusive.  With ``remove_out_of_range=True``, finite
    values outside the range use ``nan_colour``; otherwise they are clamped to
    the nearest endpoint.  Non-finite values always use ``nan_colour``.
    """

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"LUT input must be a 2-D image; received shape {data.shape}.")
    table = _validated_lut(lut)
    lo = float(vmin)
    hi = float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("LUT limits must be finite and vmax must be greater than vmin.")

    colour = np.asarray(nan_colour)
    if colour.shape != (3,) or not np.issubdtype(colour.dtype, np.number):
        raise ValueError("nan_colour must be a three-element numeric RGB tuple.")
    if not np.all(np.isfinite(colour)) or np.any(colour < 0) or np.any(colour > 255):
        raise ValueError("nan_colour entries must be finite values in the inclusive range 0..255.")
    colour = colour.astype(np.uint8)

    finite = np.isfinite(data)
    clipped = np.clip(data, lo, hi)
    index_values = np.where(finite, clipped, lo)
    # np.rint implements IEEE ties-to-even, matching LabVIEW's round-to-nearest.
    indices = np.rint((index_values - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    if inverse:
        indices = np.uint8(255) - indices
    rgb = table[indices].copy()

    excluded = ~finite
    if remove_out_of_range:
        excluded |= finite & ((data < lo) | (data > hi))
    rgb[excluded] = colour
    return Image.fromarray(rgb, mode="RGB")


__all__ = [
    "DEFAULT_LUT_NAME",
    "DEFAULT_LUT_PATH",
    "DEFAULT_LUT_SHA256",
    "DEFAULT_LUT_SOURCE",
    "apply_lut",
    "bundled_labview_lut",
    "load_imagej_lut",
    "lut_provenance",
]
