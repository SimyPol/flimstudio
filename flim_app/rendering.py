from __future__ import annotations

import colorsys
import json
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .lut import bundled_labview_lut, load_imagej_lut, apply_lut
from .models import AnalysisResult
from .phasor_roi import PhasorROI, boundary_points_gs, combined_owner_mask, contains_array
from .points import PointAnalysis
from .roi import ROIAnalysis


BG = "#0b1220"
PANEL = "#111c2e"
GRID = "#27364d"
TEXT = "#dce7f5"
MUTED = "#8ea3bd"
ACCENT = "#38bdf8"
GREEN = "#34d399"
AMBER = "#f59e0b"
APPLICATION = "FLIM Studio 0.12.1"
EXPORT_SCHEMA_VERSION = 8


def _json_clean(value):
    """Convert NumPy/non-finite values to standards-compliant JSON values."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_clean(value.item())
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


PALETTES: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    "gray": ((0.0, (4, 8, 16)), (1.0, (255, 255, 255))),
    "inferno": (
        (0.0, (0, 0, 4)),
        (0.18, (42, 10, 78)),
        (0.38, (120, 28, 109)),
        (0.58, (194, 55, 84)),
        (0.78, (249, 142, 8)),
        (1.0, (252, 255, 164)),
    ),
    "turbo": (
        (0.0, (48, 18, 59)),
        (0.18, (50, 84, 218)),
        (0.36, (25, 190, 196)),
        (0.54, (100, 226, 74)),
        (0.72, (244, 210, 48)),
        (0.88, (238, 91, 23)),
        (1.0, (122, 4, 3)),
    ),
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _normalise(values: np.ndarray, vmin: float | None, vmax: float | None, log: bool) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.float64)
    working = np.log1p(np.clip(data, 0, None)) if log else data
    finite_values = working[finite]
    lo = float(np.nanpercentile(finite_values, 1)) if vmin is None else float(np.log1p(max(vmin, 0))) if log else float(vmin)
    hi = float(np.nanpercentile(finite_values, 99.5)) if vmax is None else float(np.log1p(max(vmax, 0))) if log else float(vmax)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((working - lo) / (hi - lo), 0.0, 1.0)


def colourise(
    values: np.ndarray,
    palette: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    log: bool = False,
    nan_colour: tuple[int, int, int] = (5, 11, 21),
) -> Image.Image:
    data = np.asarray(values)
    norm = _normalise(data, vmin, vmax, log)
    anchors = PALETTES[palette]
    rgb = np.zeros(data.shape + (3,), dtype=np.float64)
    for (left, c0), (right, c1) in zip(anchors[:-1], anchors[1:]):
        mask = (norm >= left) & (norm <= right)
        fraction = np.clip((norm - left) / max(right - left, 1e-12), 0, 1)
        for channel in range(3):
            rgb[..., channel] = np.where(
                mask, c0[channel] + fraction * (c1[channel] - c0[channel]), rgb[..., channel]
            )
    rgb[norm >= anchors[-1][0]] = anchors[-1][1]
    rgb[~np.isfinite(data)] = nan_colour
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def lifetime_display_range(
    result: AnalysisResult,
    values: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return manual limits or robust automatic limits for a lifetime map."""

    data = np.asarray(result.lifetime_ns if values is None else values, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if result.settings.lifetime_scale_mode == "auto" and finite.size:
        lo, hi = np.nanpercentile(finite, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if hi <= lo:
            padding = max(abs(float(lo)) * 0.05, 0.05)
            lo, hi = lo - padding, hi + padding
        return float(lo), float(hi)
    return float(result.settings.lifetime_min_ns), float(result.settings.lifetime_max_ns)


def lifetime_image(result: AnalysisResult) -> Image.Image:
    """Render lifetimes through the exact configured 256-entry LabVIEW LUT."""

    lut = (
        load_imagej_lut(result.settings.lifetime_lut_path)
        if result.settings.lifetime_lut_path
        else bundled_labview_lut()
    )
    lo, hi = lifetime_display_range(result)
    return apply_lut(
        result.lifetime_ns,
        lut,
        lo,
        hi,
        remove_out_of_range=(
            result.settings.lifetime_lut_remove_out_of_range
            and result.settings.lifetime_scale_mode == "manual"
        ),
        inverse=result.settings.lifetime_lut_inverse,
        nan_colour=(5, 11, 21),
    )


def _hex_to_rgb(colour: str) -> tuple[int, int, int]:
    text = colour.lstrip("#")
    if len(text) != 6:
        return (255, 255, 255)
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


# Flat black for phasor-ROI highlight backgrounds -- everywhere outside a
# drawn ROI (and, with the "flag below-threshold pixels" toggle off, every
# below-threshold pixel too) renders as plain black.
PHASOR_ROI_BACKGROUND = (0, 0, 0)

# A different, warmer flat colour for pixels excluded from the phasor
# calculation entirely (too few photons to pass the sidebar's phasor
# threshold, i.e. G/S is NaN) -- deliberately distinct from
# PHASOR_ROI_BACKGROUND so "no data here" is never visually confused with
# "data present, just not matching any drawn ROI". Picking the two apart is
# exactly what motivated adding this: a dim species (e.g. a thin membrane
# stain) can fail this threshold far more often than a bright one (e.g. a
# dense nuclear stain) over the very same field of view, which looks
# identical to "this species is genuinely absent here" unless the two
# states are told apart.
PHASOR_ROI_INVALID = (64, 24, 24)


def phasor_roi_lifetime_image(
    result: AnalysisResult,
    rois: Sequence[PhasorROI],
    mode: str = "original",
    show_invalid: bool = True,
    additive_overlap: bool = False,
) -> Image.Image:
    """Highlight, on the lifetime map, which pixels' phasor coordinates fall
    inside the user-drawn phasor ROIs (see phasor_roi.py).

    Every pixel falls into exactly one of three states:
    - matched: its (G, S) falls inside a drawn ROI -- rendered per `mode`.
    - unclassified: it has a real (G, S) coordinate, just not inside any
      drawn ROI -- flat PHASOR_ROI_BACKGROUND.
    - invalid: excluded from the phasor calculation entirely (typically
      too few photons to pass the sidebar's phasor threshold, so G/S is
      NaN) -- flat PHASOR_ROI_INVALID, a different, warmer colour so this
      never reads as "unclassified" by mistake, UNLESS ``show_invalid`` is
      False, in which case it renders exactly like "unclassified" instead
      (the pre-this-feature behaviour, for a cleaner view when that
      distinction isn't currently of interest).

    mode="original": a matched pixel keeps its normal lifetime-LUT colour.
    mode="composite": a matched pixel keeps the LabVIEW HSV lifetime x
    intensity composite (see composite_image) instead of the plain
    lifetime colour -- useful when the intensity structure (which
    cells/regions actually have signal) matters alongside the ROI's
    footprint, not just its lifetime colour.
    mode="coloured": a matched pixel is flat-coloured by its OWNING ROI's
    own colour instead (the most-recently-drawn ROI wins where shapes
    overlap -- see phasor_roi.combined_owner_mask) -- UNLESS
    ``additive_overlap`` is True, in which case a pixel inside several
    ROIs at once shows every one of their colours added together
    (clipped to 255) instead of hiding all but the last-drawn one, e.g. a
    green ROI overlapping a red one reads as yellow where they overlap.
    mode="coloured_composite": each ROI's own colour again (summed the
    same way when ``additive_overlap`` is True), but scaled by that
    pixel's own composite brightness (its HSV "V", recovered as
    max(R,G,B) of composite_image's output -- exact for a standard HSV
    encoding) instead of a flat fill, so intensity structure *within* a
    region is still visible through its colour rather than a uniform block.
    """

    owner = combined_owner_mask(list(rois), result.g, result.s)
    invalid = ~(np.isfinite(result.g) & np.isfinite(result.s)) if show_invalid else np.zeros(result.g.shape, dtype=bool)
    matched = owner > 0

    if mode in ("coloured", "coloured_composite"):
        rgb = np.full(result.lifetime_ns.shape + (3,), PHASOR_ROI_BACKGROUND, dtype=np.float64)
        rgb[invalid] = PHASOR_ROI_INVALID
        if mode == "coloured_composite":
            composite = np.asarray(composite_image(result).convert("RGB"), dtype=np.float64)
            value_fraction = composite.max(axis=-1, keepdims=True) / 255.0
        if additive_overlap:
            overlap = np.zeros(result.g.shape + (3,), dtype=np.float64)
            for roi in rois:
                mask = contains_array(roi, result.g, result.s)
                colour = np.asarray(_hex_to_rgb(roi.colour), dtype=np.float64)
                overlap[mask] += colour * value_fraction[mask] if mode == "coloured_composite" else colour
            rgb[matched] = overlap[matched]
        else:
            for index, roi in enumerate(rois, start=1):
                mask = owner == index
                colour = np.asarray(_hex_to_rgb(roi.colour), dtype=np.float64)
                rgb[mask] = colour * value_fraction[mask] if mode == "coloured_composite" else colour
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    base_image = composite_image(result) if mode == "composite" else lifetime_image(result)
    base = np.asarray(base_image.convert("RGB"), dtype=np.uint8)
    rgb_u8 = np.full_like(base, PHASOR_ROI_BACKGROUND)
    rgb_u8[invalid] = PHASOR_ROI_INVALID
    rgb_u8[matched] = base[matched]
    return Image.fromarray(rgb_u8, mode="RGB")


def component_lifetime_image(result: AnalysisResult, component_index: int) -> Image.Image:
    """Render one native bi/tri lifetime component through the lifetime LUT."""

    components = result.fit_diagnostics.get("tau_components_ns")
    if components is None:
        raise ValueError("This analysis does not contain native lifetime components.")
    values = np.asarray(components, dtype=np.float64)
    index = int(component_index)
    if values.ndim != 3 or not (0 <= index < values.shape[0]):
        raise ValueError(f"Lifetime component {index + 1} is unavailable.")
    status = np.asarray(result.fit_diagnostics.get("fit_status", result.valid_mask)) == 1
    component = values[index].copy()
    component[~status | ~np.isfinite(component)] = np.nan
    lut = (
        load_imagej_lut(result.settings.lifetime_lut_path)
        if result.settings.lifetime_lut_path
        else bundled_labview_lut()
    )
    lo, hi = lifetime_display_range(result, component)
    return apply_lut(
        component,
        lut,
        lo,
        hi,
        remove_out_of_range=(
            result.settings.lifetime_lut_remove_out_of_range
            and result.settings.lifetime_scale_mode == "manual"
        ),
        inverse=result.settings.lifetime_lut_inverse,
        nan_colour=(5, 11, 21),
    )


def component_amplitude_image(result: AnalysisResult, component_index: int) -> Image.Image:
    """Render a local native fitted-amplitude component."""

    amplitudes = result.fit_diagnostics.get("amplitude_components")
    if amplitudes is None:
        raise ValueError("This analysis does not contain native amplitude components.")
    values = np.asarray(amplitudes, dtype=np.float64)
    index = int(component_index)
    if values.ndim != 3 or not (0 <= index < values.shape[0]):
        raise ValueError(f"Amplitude component {index + 1} is unavailable.")
    status = np.asarray(result.fit_diagnostics.get("fit_status", result.valid_mask)) == 1
    component = values[index].copy()
    component[~status | ~np.isfinite(component)] = np.nan
    return colourise(component, "inferno", log=True)


def threshold_intensity_image(
    intensity: np.ndarray,
    threshold: float,
    *,
    log: bool = True,
    omitted_colour: tuple[int, int, int] = (225, 29, 72),
) -> Image.Image:
    """Show accepted intensity pixels in grey and omitted pixels in red."""

    values = np.asarray(intensity, dtype=np.float64)
    rgb = np.asarray(colourise(values, "gray", log=log), dtype=np.uint8).copy()
    omitted = (~np.isfinite(values)) | (values < float(threshold))
    rgb[omitted] = np.asarray(omitted_colour, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def chi_square_image(result: AnalysisResult) -> Image.Image:
    """Render the per-pixel reduced chi-square goodness-of-fit map.

    A well-fitted pixel has reduced chi-square near 1; the "turbo" palette
    reads as cool (good fit) to hot (poor fit), which is a more useful
    default here than a plain intensity ramp. The display range is a robust
    95th-percentile of the finite, positive values so a handful of very
    poorly fitted pixels don't wash out the rest of the map.
    """

    values = np.asarray(result.chi_square, dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0)]
    vmax = max(1.5, float(np.nanpercentile(finite, 95))) if finite.size else 2.0
    return colourise(values, "turbo", vmin=0.0, vmax=vmax, log=False)


def chi_square_stats(
    result: AnalysisResult, bounds: tuple[int, int, int, int] | None = None
) -> tuple[float | None, float | None, int]:
    """Return (mean, median, count) reduced chi-square over finite, positive pixels."""

    values = np.asarray(result.chi_square, dtype=np.float64)
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        values = values[y0:y1, x0:x1]
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return None, None, 0
    return float(np.mean(finite)), float(np.median(finite)), int(finite.size)


def composite_image(result: AnalysisResult) -> Image.Image:
    """Reproduce the LabVIEW HSV intensity/lifetime composite.

    The original main VI uses a separate HSV subVI rather than the external
    lifetime LUT: lifetime maps linearly to hue 0..0.7, saturation is one, and
    the saved Auto Scale control maps the actual intensity minimum/maximum to
    value 0..1.
    """

    life_min, life_max = lifetime_display_range(result)
    life = _normalise(result.lifetime_ns, life_min, life_max, False)
    intensity = np.asarray(result.intensity, dtype=np.float64)
    finite_intensity = np.isfinite(intensity)
    if finite_intensity.any():
        intensity_min = float(np.min(intensity[finite_intensity]))
        intensity_max = float(np.max(intensity[finite_intensity]))
        if intensity_max > intensity_min:
            value = np.clip((intensity - intensity_min) / (intensity_max - intensity_min), 0.0, 1.0)
        else:
            value = np.ones_like(intensity)
    else:
        value = np.zeros_like(intensity)
    # ``_normalise`` intentionally leaves invalid lifetimes as NaN.  Replace
    # those only for the colour conversion, then paint them with the explicit
    # invalid colour below.  ``colorsys`` cannot accept a NaN hue.
    invalid = (~np.isfinite(life)) | (~np.asarray(result.valid_mask, dtype=bool))
    hue = np.where(invalid, 0.0, 0.7 * life)
    value = np.where(np.isfinite(value), value, 0.0)
    flat = np.stack((hue, np.ones_like(hue), value), axis=-1).reshape(-1, 3)
    rgb = np.asarray([colorsys.hsv_to_rgb(*pixel) for pixel in flat], dtype=np.float64).reshape(
        life.shape + (3,)
    )
    rgb[invalid] = np.asarray((5, 11, 21), dtype=np.float64) / 255.0
    return Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), mode="RGB")


def interpolation_map_image(
    source_shape_yx: tuple[int, int],
    target_rows_y: Sequence[int],
    *,
    base_intensity: np.ndarray | None = None,
    size: tuple[int, int] = (720, 500),
) -> Image.Image:
    """Render a source-resolution detector-row map with optional intensity context."""

    height, width = (int(value) for value in source_shape_yx)
    if height < 1 or width < 1:
        raise ValueError("Interpolation-map source shape must be positive.")
    if base_intensity is None:
        rgb = np.full((height, width, 3), (10, 20, 35), dtype=np.uint8)
    else:
        base = np.asarray(base_intensity, dtype=np.float64)
        if base.shape != (height, width):
            raise ValueError(
                f"Interpolation-map intensity has shape {base.shape}; expected {(height, width)}."
            )
        rgb = np.asarray(colourise(base, "gray", log=True), dtype=np.uint8)
        rgb = np.clip(rgb.astype(np.float64) * 0.42, 0, 255).astype(np.uint8)
    rows = tuple(sorted(set(int(row) for row in target_rows_y)))
    if any(row < 0 or row >= height for row in rows):
        raise ValueError(f"Interpolation-map rows must lie in 0..{height - 1}.")
    if rows:
        rgb[list(rows), :, :] = np.asarray((244, 114, 182), dtype=np.uint8)
    map_image = Image.fromarray(rgb, mode="RGB")
    canvas = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((10, 10, size[0] - 10, size[1] - 10), radius=14, fill=PANEL)
    available = (size[0] - 90, size[1] - 100)
    scale = min(available[0] / width, available[1] / height)
    rendered = map_image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.NEAREST,
    )
    x = (size[0] - rendered.width) // 2
    y = 54 + max(0, (available[1] - rendered.height) // 2)
    canvas.paste(rendered, (x, y))
    draw.text((22, 18), "Source detector-row map", fill=TEXT, font=_font(17, bold=True))
    draw.text(
        (22, size[1] - 35),
        f"{len(rows)} corrected source rows shown in magenta • image Y increases downward",
        fill=MUTED,
        font=_font(11),
    )
    return canvas


def _nice_range(values: np.ndarray, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def line_plot(
    x: np.ndarray,
    series: Iterable[tuple[str, np.ndarray, str]],
    size: tuple[int, int] = (760, 430),
    x_label: str = "Time (ns)",
    y_label: str = "Photon counts",
    title: str = "Decay and fit",
    x_limits: tuple[float, float] | None = None,
) -> Image.Image:
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 72, 48, size[0] - 24, size[1] - 58
    series = list(series)
    if not series:
        draw.rounded_rectangle((10, 10, size[0] - 10, size[1] - 10), radius=14, fill=PANEL)
        draw.text((22, 17), title, fill=TEXT, font=_font(17, bold=True))
        draw.text((size[0] / 2 - 80, size[1] / 2), "No curves selected", fill=MUTED, font=_font(12))
        return image
    finite_chunks = [np.asarray(values)[np.isfinite(values)] for _, values, _ in series]
    all_y = np.concatenate([chunk for chunk in finite_chunks if chunk.size]) if any(chunk.size for chunk in finite_chunks) else np.array([0.0])
    x_lo, x_hi = x_limits if x_limits is not None else _nice_range(np.asarray(x))
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        raise ValueError("Plot x limits must be finite and increasing.")
    y_lo, y_hi = _nice_range(all_y, (0.0, 1.0))
    y_lo = min(0.0, y_lo)
    draw.rounded_rectangle((10, 10, size[0] - 10, size[1] - 10), radius=14, fill=PANEL)
    for i in range(6):
        px = left + i * (right - left) / 5
        py = bottom - i * (bottom - top) / 5
        draw.line((px, top, px, bottom), fill=GRID, width=1)
        draw.line((left, py, right, py), fill=GRID, width=1)
        draw.text((px - 14, bottom + 9), f"{x_lo + i * (x_hi - x_lo) / 5:.2g}", fill=MUTED, font=_font(11))
        draw.text((12, py - 7), f"{y_lo + i * (y_hi - y_lo) / 5:.2g}", fill=MUTED, font=_font(11))
    draw.line((left, bottom, right, bottom), fill=MUTED, width=2)
    draw.line((left, top, left, bottom), fill=MUTED, width=2)
    for index, (label, values, colour) in enumerate(series):
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(x)
        if valid.sum() >= 2:
            px = left + (np.asarray(x)[valid] - x_lo) / (x_hi - x_lo) * (right - left)
            py = bottom - (values[valid] - y_lo) / (y_hi - y_lo) * (bottom - top)
            draw.line(list(zip(px.tolist(), py.tolist())), fill=colour, width=3)
        lx = right - 160
        ly = top + index * 22
        draw.line((lx, ly + 6, lx + 24, ly + 6), fill=colour, width=3)
        draw.text((lx + 31, ly), label, fill=TEXT, font=_font(12))
    draw.text((22, 17), title, fill=TEXT, font=_font(17, bold=True))
    draw.text(((left + right) / 2 - 30, size[1] - 33), x_label, fill=MUTED, font=_font(12))
    draw.text((left + 6, top + 5), y_label, fill=MUTED, font=_font(11))
    return image


def irf_decay_overlay_plot(result: AnalysisResult, size: tuple[int, int] = (760, 430)) -> Image.Image:
    """Peak-normalised whole-image mean decay and the IRF actually used for
    this run (loaded file or synthetic Gaussian, whichever applies),
    plotted together so their relative timing and shape can be compared
    directly -- a measured decay and an IRF trace have wildly different
    absolute magnitudes, so normalising both to their own peak is what
    makes the overlay actually readable.
    """

    def _peak_normalised(values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        peak = float(finite.max()) if finite.size else 0.0
        return values / peak if peak > 0 else values

    series = [("Decay (normalised, whole image)", _peak_normalised(result.mean_decay), ACCENT)]
    irf_trace = result.fit_diagnostics.get("irf_trace")
    if irf_trace is not None:
        series.append(("IRF (normalised)", _peak_normalised(irf_trace), GREEN))
        title = "IRF vs. decay (normalised)"
    else:
        title = "IRF vs. decay (normalised) — no IRF was used for this analysis"
    return line_plot(result.time_ns, series, size=size, y_label="Normalised signal (peak = 1)", title=title)


def histogram_plot(
    result: AnalysisResult,
    size: tuple[int, int] = (760, 430),
    bounds: tuple[int, int, int, int] | None = None,
    points: Sequence[PointAnalysis] = (),
    bins: int | None = None,
) -> Image.Image:
    estimator = points[0].lifetime_label if points else _lifetime_estimator_label(result)
    if bounds is None:
        values = result.valid_lifetimes
        title = "Lifetime distribution"
    else:
        x0, y0, x1, y1 = bounds
        lifetime = result.lifetime_ns[y0:y1, x0:x1]
        valid = result.valid_mask[y0:y1, x0:x1] & np.isfinite(lifetime)
        values = lifetime[valid]
        title = "ROI lifetime distribution"
    bins = int(result.settings.histogram_bins if bins is None else bins)
    life_min, life_max = lifetime_display_range(result)
    if values.size:
        counts, edges = np.histogram(
            values,
            bins=bins,
            range=(life_min, life_max),
        )
        centres = (edges[:-1] + edges[1:]) / 2
    else:
        edges = np.linspace(life_min, life_max, bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        counts = np.zeros(bins)
    image = line_plot(
        centres,
        [("Pixels", counts.astype(float), ACCENT)],
        size,
        f"{estimator} (ns)",
        "Pixel count",
        title,
        x_limits=(life_min, life_max),
    )
    if points:
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = 72, 48, size[0] - 24, size[1] - 58
        x_lo = float(result.settings.lifetime_min_ns)
        x_hi = float(result.settings.lifetime_max_ns)
        label_level = 0
        for point in points:
            tau = point.map_tau_ns
            if not point.valid or not np.isfinite(tau) or not (x_lo <= tau <= x_hi):
                continue
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                if not (x0 <= point.selection.x < x1 and y0 <= point.selection.y < y1):
                    continue
            px = left + (tau - x_lo) / max(x_hi - x_lo, 1e-12) * (right - left)
            draw.line((px, top, px, bottom), fill=point.selection.colour, width=3)
            draw.text(
                (min(px + 3, right - 22), top + 4 + (label_level % 4) * 15),
                point.selection.identifier,
                fill=point.selection.colour,
                font=_font(11, bold=True),
            )
            label_level += 1
    return image


def _lifetime_estimator_label(result: AnalysisResult) -> str:
    method = result.settings.method
    labels = {
        "phasor_phase": "τφ",
        "phasor_modulation": "τM",
        "moment": "τmoment",
        "mono_exponential": "τtail",
    }
    if method in {"flimlib_reconvolution", "flimlib_global"}:
        return "⟨τ⟩A" if result.settings.model_order > 1 else "τfit"
    return labels.get(method, "τ")


def export_point_selections(
    result: AnalysisResult,
    points: Sequence[PointAnalysis],
    output_dir: str | Path,
    prefix: str | None = None,
    active_roi_bounds: tuple[int, int, int, int] | None = None,
) -> list[Path]:
    if not points:
        return []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = prefix or result.dataset_name.replace(" ", "_")
    identity = {
        "points": [(point.selection.x, point.selection.y) for point in points],
        "method": result.settings.method,
        "settings": asdict(result.settings),
        "roi": active_roi_bounds,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
    point_stem = f"{stem}_points_n{len(points)}_{digest}"
    written: list[Path] = []

    summary_path = destination / f"{point_stem}_summary.json"
    summary_path.write_text(
        json.dumps(
            _json_clean({
                "export_schema_version": EXPORT_SCHEMA_VERSION,
                "dataset": result.dataset_name,
                "application": APPLICATION,
                "analysis_method": result.settings.method,
                "lifetime_estimator": points[0].lifetime_label,
                "phasor_coordinates": (
                    "IRF-corrected analysed first-harmonic G/S"
                    if bool(result.metadata.get("irf_loaded"))
                    else "uncorrected analysed first-harmonic G/S (no IRF)"
                ),
                "analysis_settings": asdict(result.settings),
                "spatial_transform": result.metadata.get("spatial_transform"),
                "active_roi_bounds_xyxy": list(active_roi_bounds) if active_roi_bounds is not None else None,
                "processed_signal_units": "background-corrected/smoothed/binned input-count units per time bin",
                "coordinate_convention": "zero-based processed x/y; nominal source bounds are half-open",
                "native_diagnostic_conventions": {
                    "spatial_component_arrays": "point-major in NPZ; component is the second axis",
                    "fit_start_bin_and_fit_stop_bin": "1-based inclusive values returned by the native wrapper",
                    "fit_status": {
                        "1": "fit succeeded",
                        "0": "masked or unprocessed",
                        "-1": "invalid decay",
                        "-2": "below minimum signal",
                        "-3": "invalid IRF",
                        "-4": "FLIMLib fit failure",
                        "-5": "allocation failure",
                        "-6": "invalid fit window",
                        "-7": "excluded by global-pixel cap",
                    },
                },
                "points": [point.summary() for point in points],
            }),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    written.append(summary_path)

    npz_path = destination / f"{point_stem}_decays.npz"
    native_arrays: dict[str, np.ndarray] = {}
    native_keys = sorted(
        {
            key
            for point in points
            for key in (point.native_diagnostics or {})
        }
    )
    for key in native_keys:
        values = [(point.native_diagnostics or {}).get(key) for point in points]
        present = [np.asarray(value) for value in values if value is not None]
        if not present or any(value is None for value in values):
            continue
        try:
            native_arrays[f"native_{key}"] = np.stack([np.asarray(value) for value in values])
        except ValueError:
            continue
    np.savez_compressed(
        npz_path,
        time_ns=result.time_ns,
        point_ids=np.asarray([point.selection.identifier for point in points]),
        xy=np.asarray([(point.selection.x, point.selection.y) for point in points], dtype=np.int64),
        colours=np.asarray([point.selection.colour for point in points]),
        map_tau_ns=np.asarray([point.map_tau_ns for point in points], dtype=np.float64),
        valid=np.asarray([point.valid for point in points], dtype=bool),
        processed_intensity=np.asarray([point.processed_intensity for point in points], dtype=np.float64),
        total_processed_signal=np.asarray([point.total_processed_signal for point in points], dtype=np.float64),
        phasor_g=np.asarray([point.phasor_g for point in points], dtype=np.float64),
        phasor_s=np.asarray([point.phasor_s for point in points], dtype=np.float64),
        phasor_valid=np.asarray([point.phasor_valid for point in points], dtype=bool),
        nominal_source_bounds_xyxy=np.asarray(
            [point.source_bounds_xyxy_nominal for point in points], dtype=np.int64
        ),
        fit_status=np.asarray(
            [point.fit_status if point.fit_status is not None else 0 for point in points], dtype=np.int32
        ),
        processed_decay=np.stack([point.decay for point in points]),
        **native_arrays,
    )
    written.append(npz_path)

    csv_path = destination / f"{point_stem}_decays.csv"
    columns = [result.time_ns, *[point.decay for point in points]]
    np.savetxt(
        csv_path,
        np.column_stack(columns),
        delimiter=",",
        header="time_ns," + ",".join(point.selection.identifier for point in points),
        comments="",
    )
    written.append(csv_path)

    plot_path = destination / f"{point_stem}_decays.png"
    line_plot(
        result.time_ns,
        [(point.selection.identifier, point.decay, point.selection.colour) for point in points],
        title=f"Selected-point processed decays (N={len(points)})",
        y_label="Processed signal / bin",
    ).save(plot_path)
    written.append(plot_path)

    phasor_path = destination / f"{point_stem}_phasor.png"
    phasor_plot(
        result,
        bounds=active_roi_bounds,
        points=points,
    ).save(phasor_path)
    written.append(phasor_path)
    return written


def _rotated_label(text: str, font: ImageFont.ImageFont, fill: str) -> Image.Image:
    """A vertical (bottom-to-top) text label, for the density key."""

    scratch = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    x0, y0, x1, y1 = scratch.textbbox((0, 0), text, font=font)
    width, height = x1 - x0, y1 - y0
    label = Image.new("RGBA", (width + 4, height + 4), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((2 - x0, 2 - y0), text, font=font, fill=fill)
    return label.rotate(90, expand=True)


def phasor_plot_axes(size: tuple[int, int]) -> tuple[int, int, int, int, float, float, float, float]:
    """The exact pixel<->(G, S) mapping ``phasor_plot`` draws at ``size``.

    Returns ``(left, top, right, bottom, g_min, g_max, s_min, s_max)`` in
    final (unsupersampled) image pixels -- the same square plotting
    rectangle and G/S unit range ``phasor_plot`` itself uses internally.
    Exposed so a rendered phasor plot's pixel coordinates (e.g. a user's
    click on it) can be inverted back to true G/S values with no
    duplicated, driftable copy of these margin constants -- see
    ``phasor_gs_to_pixel``/``phasor_pixel_to_gs``.
    """

    margin_left, margin_top, margin_right, margin_bottom = 74, 22, 68, 58
    available_width = size[0] - margin_left - margin_right
    available_height = size[1] - margin_top - margin_bottom
    plot_side = max(1, min(available_width, available_height))
    left = margin_left + (available_width - plot_side) // 2
    top = margin_top + (available_height - plot_side) // 2
    right = left + plot_side
    bottom = top + plot_side
    return left, top, right, bottom, -0.1, 1.1, -0.1, 1.1


def phasor_gs_to_pixel(g: float, s: float, size: tuple[int, int]) -> tuple[float, float]:
    """(G, S) -> pixel position in a ``phasor_plot(..., size=size)`` image."""

    left, top, right, bottom, g_min, g_max, s_min, s_max = phasor_plot_axes(size)
    return (
        left + (g - g_min) / (g_max - g_min) * (right - left),
        bottom - (s - s_min) / (s_max - s_min) * (bottom - top),
    )


def phasor_pixel_to_gs(px: float, py: float, size: tuple[int, int]) -> tuple[float, float]:
    """Inverse of ``phasor_gs_to_pixel`` -- pixel position -> (G, S)."""

    left, top, right, bottom, g_min, g_max, s_min, s_max = phasor_plot_axes(size)
    return (
        g_min + (px - left) / (right - left) * (g_max - g_min),
        s_min + (bottom - py) / (bottom - top) * (s_max - s_min),
    )


def phasor_plot(
    result: AnalysisResult,
    size: tuple[int, int] = (504, 460),
    max_points: int = 30_000,
    bounds: tuple[int, int, int, int] | None = None,
    points: Sequence[PointAnalysis] = (),
    show_context: bool = True,
    heatmap: bool = True,
    heatmap_bins: int | None = None,
    rois: Sequence[PhasorROI] = (),
    g_s_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> Image.Image:
    """Render the phasor plot.

    ``g_s_override``, when given, is a ``(g, s)`` pair of arrays plotted
    instead of ``result.g``/``result.s`` -- e.g. a pooled cloud built by
    concatenating several datasets' own phasor maps together (see gui.py's
    "accumulate phasor across queue"). ``bounds`` is meaningless against a
    pooled cloud spanning several images and should be left ``None``
    whenever an override is given.

    Drawn supersampled and downsampled with LANCZOS at the end -- PIL's
    ImageDraw has no built-in anti-aliasing, so lines/text/the semicircle
    would otherwise look jagged at this resolution. The density heatmap is
    separately interpolated with BICUBIC when it's built (see below) since
    the closing LANCZOS pass alone isn't enough to turn its coarse
    per-histogram-bin blocks into a smooth gradient. The violet-to-red
    density ramp and the red universal semicircle stay LabVIEW-accurate;
    everything else (panel background, grid, axis text, density key) is
    this app's own design.

    The plotting rectangle is square with matching G/S unit ranges, so one
    G unit and one S unit cover the same number of pixels -- otherwise the
    semicircle renders stretched.
    """

    SS = 3  # supersample factor
    canvas = (size[0] * SS, size[1] * SS)
    image = Image.new("RGB", canvas, PANEL)
    draw = ImageDraw.Draw(image)

    # Same axes phasor_plot_axes() computes (in final-image pixels), just
    # scaled up to this function's own supersampled canvas -- kept as a
    # single source of truth so callers can invert a rendered pixel back
    # to true (G, S) with no separately-maintained copy of these margins.
    axis_left, axis_top, axis_right, axis_bottom, g_min, g_max, s_min, s_max = phasor_plot_axes(size)
    left, top, right, bottom = (v * SS for v in (axis_left, axis_top, axis_right, axis_bottom))

    def plot_xy(g_value: float, s_value: float) -> tuple[float, float]:
        return (
            left + (g_value - g_min) / (g_max - g_min) * (right - left),
            bottom - (s_value - s_min) / (s_max - s_min) * (bottom - top),
        )

    draw.rectangle((left, top, right, bottom), fill="black")

    # Subtle interior gridlines every 0.2 units -- easier to read off an
    # approximate G/S coordinate than bare tick marks at the plot edge.
    grid_line = "#182236"
    for step in range(6):
        value = step * 0.2
        px, _ = plot_xy(value, 0.0)
        draw.line((px, top, px, bottom), fill=grid_line, width=SS)
        _, py = plot_xy(0.0, value)
        draw.line((left, py, right, py), fill=grid_line, width=SS)

    axis_font = _font(10 * SS)
    tick_len = 5 * SS
    for step in range(13):
        value = g_min + step * 0.1
        major = step % 2 == 1  # lines up with 0, 0.2, 0.4 ... 1.0
        px, _ = plot_xy(value, 0.0)
        draw.line((px, bottom, px, bottom + (tick_len if major else tick_len * 0.6)), fill=MUTED)
        if major:
            label = f"{value:.1f}".rstrip("0").rstrip(".")
            box = draw.textbbox((0, 0), label, font=axis_font)
            draw.text((px - (box[2] - box[0]) / 2, bottom + tick_len + 2 * SS), label, fill=MUTED, font=axis_font)
    for step in range(13):
        value = s_min + step * 0.1
        major = step % 2 == 1
        _, py = plot_xy(g_min, value)
        draw.line((left - (tick_len if major else tick_len * 0.6), py, left, py), fill=MUTED)
        if major:
            label = f"{value:.1f}".rstrip("0").rstrip(".")
            box = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (left - tick_len - 4 * SS - (box[2] - box[0]), py - (box[3] - box[1]) / 2),
                label, fill=MUTED, font=axis_font,
            )
    draw.rectangle((left, top, right, bottom), outline=GRID, width=SS)

    if g_s_override is not None:
        g = np.asarray(g_s_override[0], dtype=np.float64).ravel()
        s = np.asarray(g_s_override[1], dtype=np.float64).ravel()
    elif bounds is None:
        g, s = result.g.ravel(), result.s.ravel()
    else:
        x0, y0, x1, y1 = bounds
        g = result.g[y0:y1, x0:x1].ravel()
        s = result.s[y0:y1, x0:x1].ravel()

    valid = (
        np.isfinite(g) & np.isfinite(s)
        & (g >= g_min) & (g <= g_max)
        & (s >= s_min) & (s <= s_max)
    )
    indices = np.flatnonzero(valid)
    if indices.size > max_points:
        rng = np.random.default_rng(1729)
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))

    if heatmap and indices.size:
        bins = int(heatmap_bins or result.settings.phasor_heatmap_bins)
        density, _, _ = np.histogram2d(
            g[indices], s[indices], bins=(bins, bins), range=((g_min, g_max), (s_min, s_max))
        )
        log_density = np.log1p(density)
        peak = float(log_density.max())
        if peak > 0:
            norm = np.power(log_density / peak, 1.35)
            rainbow = np.asarray(
                [colorsys.hsv_to_rgb(0.76 * (1.0 - value), 1.0, 1.0) for value in np.linspace(0, 1, 256)],
                dtype=np.float64,
            )
            rgb = np.clip(rainbow[np.rint(norm * 255).astype(np.int64)] * 255, 0, 255).astype(np.uint8)
            rgb[density == 0] = 0
            # BICUBIC here, not NEAREST: each histogram bin only ends up
            # covering ~1-2 pixels in the final image regardless of the
            # requested plot size (this upscale factor is fixed by
            # heatmap_bins vs. the supersample factor, not by `size`), so
            # the closing LANCZOS downsample never has enough room to turn
            # hard NEAREST blocks into a real gradient -- it still reads as
            # pixelated. Interpolating here instead gives an actually
            # smooth density gradient between bins, which is also the more
            # honest representation: this is a binned density *estimate*,
            # not per-pixel raw data, so there's no real hard edge between
            # adjacent bins to preserve in the first place. The colour ramp
            # itself is untouched.
            heat = Image.fromarray(np.transpose(rgb, (1, 0, 2))[::-1], mode="RGB")
            heat = heat.resize((right - left, bottom - top), Image.Resampling.BICUBIC)
            image.paste(heat, (left, top))
            draw = ImageDraw.Draw(image)

            bar_width = 15 * SS
            bar_height = min(150 * SS, bottom - top)
            bar_left = right + 14 * SS
            bar_top = top
            bar_values = np.linspace(1.0, 0.0, bar_height)
            bar_rgb = np.clip(
                np.asarray(
                    [colorsys.hsv_to_rgb(0.76 * (1.0 - value), 1.0, 1.0) for value in bar_values]
                ) * 255,
                0,
                255,
            ).astype(np.uint8)
            bar = Image.fromarray(np.broadcast_to(bar_rgb[:, None, :], (bar_height, bar_width, 3)), mode="RGB")
            image.paste(bar, (bar_left, bar_top))
            draw.rectangle(
                (bar_left - 1, bar_top - 1, bar_left + bar_width, bar_top + bar_height),
                outline=GRID, width=SS,
            )
            key_font = _font(9 * SS)

            def centred_over_bar(text: str, y: float) -> None:
                box = draw.textbbox((0, 0), text, font=key_font)
                draw.text((bar_left + (bar_width - (box[2] - box[0])) / 2, y), text, fill=MUTED, font=key_font)

            centred_over_bar("High", bar_top - 13 * SS)
            centred_over_bar("Low", bar_top + bar_height + 3 * SS)
            density_label = _rotated_label("Density", _font(10 * SS, bold=True), MUTED)
            image.paste(
                density_label,
                (bar_left + bar_width + 6 * SS, bar_top + (bar_height - density_label.height) // 2),
                density_label,
            )
    else:
        for gx, sy in zip(g[indices], s[indices]):
            px, py = plot_xy(float(gx), float(sy))
            draw.ellipse((px - SS / 2, py - SS / 2, px + SS / 2, py + SS / 2), fill="#8b00ff")

    theta = np.linspace(0, np.pi, 720)
    upoints = [
        plot_xy(float(0.5 + 0.5 * np.cos(value)), float(0.5 * np.sin(value)))
        for value in theta
    ]
    draw.line(upoints, fill="#ff1b13", width=max(1, int(round(2.5 * SS))), joint="curve")

    for point in points:
        gx, sy = float(point.phasor_g), float(point.phasor_s)
        if not point.phasor_valid or not np.isfinite(gx) or not np.isfinite(sy):
            continue
        if not (g_min <= gx <= g_max and s_min <= sy <= s_max):
            continue
        px, py = plot_xy(gx, sy)
        colour = point.selection.colour
        outer, inner = 7 * SS, 4 * SS
        draw.ellipse((px - outer, py - outer, px + outer, py + outer), fill="black", outline="white", width=SS)
        draw.ellipse((px - inner, py - inner, px + inner, py + inner), fill=colour, outline=colour)
        draw.text(
            (px + outer, py - 17 * SS), point.selection.identifier,
            fill=colour, font=_font(11 * SS, bold=True),
        )

    roi_outline_width = max(1, int(round(2.5 * SS)))
    roi_label_font = _font(11 * SS, bold=True)
    for roi in rois:
        # boundary_points_gs gives the true outline in (G, S) space for any
        # shape -- including a rotated rectangle/ellipse, which a plain
        # draw.rectangle/draw.ellipse (axis-aligned only) can't represent --
        # so every shape draws through the same polyline path here.
        vertices = [plot_xy(g_value, s_value) for g_value, s_value in boundary_points_gs(roi)]
        if len(vertices) < 2:
            continue
        draw.line(vertices + [vertices[0]], fill=roi.colour, width=roi_outline_width, joint="curve")
        label_anchor = min(vertices, key=lambda point: point[1])
        label_box = draw.textbbox((0, 0), roi.label, font=roi_label_font)
        draw.text(
            (label_anchor[0] + 4 * SS, label_anchor[1] - (label_box[3] - label_box[1]) - 3 * SS),
            roi.label, fill=roi.colour, font=roi_label_font,
        )

    axis_font_title = _font(12 * SS)
    g_label_box = draw.textbbox((0, 0), "G", font=axis_font_title)
    draw.text(((left + right) / 2 - (g_label_box[2] - g_label_box[0]) / 2, canvas[1] - 22 * SS), "G", fill=TEXT, font=axis_font_title)
    s_label = _rotated_label("S", axis_font_title, TEXT)
    image.paste(s_label, (16 * SS, int((top + bottom) / 2 - s_label.height / 2)), s_label)

    return image.resize(size, Image.Resampling.LANCZOS)


def add_map_frame(image: Image.Image, title: str, subtitle: str, size: tuple[int, int] = (430, 430)) -> Image.Image:
    framed = Image.new("RGB", size, PANEL)
    draw = ImageDraw.Draw(framed)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=14, outline=GRID, width=2)
    draw.text((16, 12), title, fill=TEXT, font=_font(16, bold=True))
    draw.text((16, 35), subtitle, fill=MUTED, font=_font(11))
    available = (size[0] - 24, size[1] - 72)
    scale = min(available[0] / image.width, available[1] / image.height)
    fitted = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS,
    )
    x = (size[0] - fitted.width) // 2
    y = 60 + (available[1] - fitted.height) // 2
    framed.paste(fitted, (x, y))
    return framed


def dashboard_preview(result: AnalysisResult, size: tuple[int, int] = (1500, 920)) -> Image.Image:
    canvas = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size[0], 76), fill="#0f1a2b")
    draw.text((30, 18), "FLIM Studio", fill=TEXT, font=_font(25, bold=True))
    draw.text((30, 49), result.dataset_name, fill=MUTED, font=_font(12))
    summary = result.summary()
    draw.text(
        (size[0] - 520, 28),
        f"Valid pixels  {summary['valid_pixels']:,}    Global τ  {summary['global_tau_ns'] or 0:.3f} ns",
        fill=GREEN,
        font=_font(13, bold=True),
    )
    intensity = add_map_frame(colourise(result.intensity, "inferno", log=True), "Intensity", "Photon-count projection")
    lifetime = add_map_frame(
        lifetime_image(result),
        "Lifetime",
        (
            f"{result.settings.lifetime_min_ns:g}–{result.settings.lifetime_max_ns:g} ns • "
            f"{Path(result.settings.lifetime_lut_path).name if result.settings.lifetime_lut_path else 'Reverse Rainbow (DM)'}"
        ),
    )
    composite = add_map_frame(composite_image(result), "Composite", "LabVIEW HSV lifetime × linear intensity")
    canvas.paste(intensity, (30, 100))
    canvas.paste(lifetime, (465, 100))
    canvas.paste(composite, (900, 100))
    decay = line_plot(
        result.time_ns,
        [("Mean decay", result.mean_decay, ACCENT), ("Mono fit", result.mean_fit, GREEN)],
        (710, 340),
    )
    phasor = phasor_plot(result, (710, 340))
    canvas.paste(decay, (30, 560))
    canvas.paste(phasor, (760, 560))
    return canvas


def export_result(result: AnalysisResult, output_dir: str | Path, prefix: str | None = None) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = prefix or result.dataset_name.replace(" ", "_")
    written: list[Path] = []
    npz_path = destination / f"{stem}_analysis.npz"
    result_arrays = {
        "intensity": result.intensity,
        "lifetime_ns": result.lifetime_ns,
        "valid_mask": result.valid_mask,
        "g": result.g,
        "s": result.s,
        "tau_phase_ns": result.tau_phase_ns,
        "tau_modulation_ns": result.tau_modulation_ns,
        "chi_square": result.chi_square,
        "time_ns": result.time_ns,
        "mean_decay": result.mean_decay,
        "mean_fit": result.mean_fit,
        **result.fit_diagnostics,
    }
    np.savez_compressed(
        npz_path,
        **result_arrays,
    )
    written.append(npz_path)
    summary_path = destination / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(_json_clean(result.summary()), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    written.append(summary_path)
    images = {
        "intensity": colourise(result.intensity, "inferno", log=True),
        "lifetime": lifetime_image(result),
        "composite": composite_image(result),
        "histogram": histogram_plot(result, bins=result.settings.detailed_histogram_bins),
        "phasor": phasor_plot(result),
        "decay_fit": line_plot(
            result.time_ns,
            [("Mean decay", result.mean_decay, ACCENT), ("Mono fit", result.mean_fit, GREEN)],
        ),
        "dashboard": dashboard_preview(result),
    }
    components = result.fit_diagnostics.get("tau_components_ns")
    if components is not None and np.asarray(components).ndim == 3:
        for component_index in range(np.asarray(components).shape[0]):
            images[f"lifetime_tau{component_index + 1}"] = component_lifetime_image(
                result, component_index
            )
    for label, image in images.items():
        path = destination / f"{stem}_{label}.png"
        image.save(path)
        written.append(path)
    values = result.valid_lifetimes
    counts, edges = np.histogram(
        values,
        bins=result.settings.detailed_histogram_bins,
        range=(result.settings.lifetime_min_ns, result.settings.lifetime_max_ns),
    )
    csv_path = destination / f"{stem}_lifetime_histogram.csv"
    centres = (edges[:-1] + edges[1:]) / 2
    np.savetxt(csv_path, np.column_stack((centres, counts)), delimiter=",", header="lifetime_ns,pixel_count", comments="")
    written.append(csv_path)
    return written


def export_roi_analysis(
    result: AnalysisResult,
    roi: ROIAnalysis,
    output_dir: str | Path,
    prefix: str | None = None,
) -> list[Path]:
    """Export the active ROI without changing the normal whole-image export contract."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = prefix or result.dataset_name.replace(" ", "_")
    x0, y0, x1, y1 = roi.bounds_xyxy
    written: list[Path] = []

    coordinate_tag = f"x{x0}-{x1}_y{y0}-{y1}"
    roi_stem = f"{stem}_roi_{coordinate_tag}"
    arrays_path = destination / f"{roi_stem}_analysis.npz"
    native_fit_arrays = (
        {"native_reconvolution_fit_to_sum": roi.native_fitted_decay}
        if roi.native_fitted_decay is not None
        else {}
    )
    np.savez_compressed(
        arrays_path,
        bounds_xyxy=np.asarray(roi.bounds_xyxy, dtype=np.int64),
        time_ns=result.time_ns,
        summed_decay=roi.summed_decay,
        mean_decay_per_processed_pixel=roi.mean_decay,
        mono_guide_fit_to_sum=roi.fitted_decay,
        **native_fit_arrays,
        intensity=result.intensity[y0:y1, x0:x1],
        lifetime_ns=result.lifetime_ns[y0:y1, x0:x1],
        valid_mask=result.valid_mask[y0:y1, x0:x1],
        chi_square=result.chi_square[y0:y1, x0:x1],
        phasor_g=result.g[y0:y1, x0:x1],
        phasor_s=result.s[y0:y1, x0:x1],
        tau_phase_ns=result.tau_phase_ns[y0:y1, x0:x1],
        tau_modulation_ns=result.tau_modulation_ns[y0:y1, x0:x1],
        **{
            f"diagnostic_{key}": (
                value[..., y0:y1, x0:x1]
                if value.ndim >= 2 and value.shape[-2:] == result.intensity.shape
                else value
            )
            for key, value in result.fit_diagnostics.items()
        },
    )
    written.append(arrays_path)

    bin_factor = int(result.metadata.get("spatial_transform", {}).get("spatial_bin_width", 1))
    summary = {
        "dataset": result.dataset_name,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "application": APPLICATION,
        "analysis_method": result.settings.method,
        "phasor_coordinates": (
            "IRF-corrected analysed first-harmonic G/S"
            if bool(result.metadata.get("irf_loaded"))
            else "uncorrected analysed first-harmonic G/S (no IRF)"
        ),
        "analysis_settings": asdict(result.settings),
        "coordinate_convention": "zero-based xyxy; upper bounds exclusive",
        "processed_signal_units": "background-corrected/smoothed/binned input-count units per time bin",
        "spatial_bin_factor": bin_factor,
        "nominal_source_pixel_count": roi.selected_pixels * bin_factor * bin_factor,
        "nominal_source_bounds_xyxy": [x0 * bin_factor, y0 * bin_factor, x1 * bin_factor, y1 * bin_factor],
        "spatial_transform": result.metadata.get("spatial_transform"),
        "phasor": {
            "meaning": "distribution of per-processed-pixel analysed first-harmonic coordinates",
            "valid_pixels": int(
                np.count_nonzero(
                    np.isfinite(result.g[y0:y1, x0:x1])
                    & np.isfinite(result.s[y0:y1, x0:x1])
                )
            ),
        },
        **roi.summary(),
    }
    summary_path = destination / f"{roi_stem}_summary.json"
    summary_path.write_text(
        json.dumps(_json_clean(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    written.append(summary_path)

    csv_path = destination / f"{roi_stem}_decay.csv"
    has_native_fit = roi.native_fitted_decay is not None
    csv_columns = [result.time_ns, roi.summed_decay, roi.mean_decay, roi.fitted_decay]
    csv_header = "time_ns,summed_processed_signal,mean_signal_per_processed_pixel,mono_guide_fit_to_sum"
    if has_native_fit:
        csv_columns.append(roi.native_fitted_decay)
        csv_header += ",native_reconvolution_fit_to_sum"
    np.savetxt(
        csv_path,
        np.column_stack(csv_columns),
        delimiter=",",
        header=csv_header,
        comments="",
    )
    written.append(csv_path)

    plot_path = destination / f"{roi_stem}_decay.png"
    plot_series = [("ROI summed processed signal", roi.summed_decay, ACCENT)]
    if has_native_fit:
        plot_series.append((roi.native_fit_label or "Native reconvolution fit", roi.native_fitted_decay, AMBER))
    else:
        plot_series.append(("Diagnostic mono tail guide", roi.fitted_decay, GREEN))
    line_plot(
        result.time_ns,
        plot_series,
        title=f"ROI decay — x={x0}:{x1}, y={y0}:{y1}",
        y_label="Processed signal / bin",
    ).save(plot_path)
    written.append(plot_path)

    phasor_path = destination / f"{roi_stem}_phasor.png"
    phasor_plot(result, bounds=roi.bounds_xyxy).save(phasor_path)
    written.append(phasor_path)
    return written
