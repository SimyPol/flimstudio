from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Callable, Sequence

import numpy as np

from .global_model import (
    TrainedGlobalModel,
    TrainingImageProvenance,
    build_trained_model,
    load_global_model,
    new_model_name,
)
from .lut import lut_provenance
from .models import AnalysisResult, AnalysisSettings, FLIMDataset
from .native_backend import (
    NativeFitResult,
    apply_global_in_subprocess,
    fit_in_subprocess,
    fit_pixels_parallel_in_subprocess,
    train_global_in_subprocess,
)
from .row_interpolation import (
    RowInterpolationMap,
    column_interpolation_coverage,
    interpolate_mapped_columns,
    interpolate_mapped_rows,
    interpolation_coverage,
)


ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class TrainingImageInput:
    """One row of a global-training set: an image with its own IRF/background."""

    dataset: FLIMDataset
    irf: FLIMDataset | np.ndarray
    background: FLIMDataset | np.ndarray | None = None
    name: str = ""


def _notify(callback: ProgressCallback | None, value: float, message: str) -> None:
    if callback is not None:
        callback(float(value), message)


def shift_time_axis(cube: np.ndarray, offset_bins: float) -> np.ndarray:
    """Shift decays towards earlier bins using linear interpolation and zero fill."""

    offset = float(offset_bins)
    if abs(offset) < 1e-9:
        return cube
    nt = cube.shape[-1]
    source = np.arange(nt, dtype=np.float64) + offset
    lower = np.floor(source).astype(int)
    fraction = source - lower
    valid0 = (lower >= 0) & (lower < nt)
    valid1 = (lower + 1 >= 0) & (lower + 1 < nt)
    output = np.zeros_like(cube, dtype=np.float32)
    if np.any(valid0):
        output[..., valid0] += cube[..., lower[valid0]] * (1.0 - fraction[valid0])
    if np.any(valid1):
        output[..., valid1] += cube[..., (lower + 1)[valid1]] * fraction[valid1]
    return output


def block_bin(cube: np.ndarray, factor: int) -> np.ndarray:
    """Sum non-overlapping spatial blocks, shrinking the output by ``factor``.

    Not used by the analysis pipeline any more -- ``rolling_spatial_bin``
    replaced it there so the "Spatial bin radius" setting no longer
    reduces the output image size. Kept as a general utility.
    """

    factor = max(1, int(factor))
    if factor == 1:
        return cube
    height, width, nt = cube.shape
    out_h = height // factor
    out_w = width // factor
    if out_h < 1 or out_w < 1:
        raise ValueError(f"Spatial bin {factor} is larger than image shape {(height, width)}.")
    cropped = cube[: out_h * factor, : out_w * factor]
    return cropped.reshape(out_h, factor, out_w, factor, nt).sum(axis=(1, 3))


def spatial_bin_width(level: int) -> int:
    """Translate the UI radius level to an odd square window width."""

    return 2 * max(0, int(level)) + 1


def rolling_spatial_bin(cube: np.ndarray, radius: int) -> np.ndarray:
    """Sum photon counts within a sliding square window centred on each
    pixel, keeping the image at its original ``(height, width)`` -- unlike
    ``block_bin``, which reduces the output size by the block factor. A
    radius of 1 sums each pixel's 3x3 neighbourhood into that pixel's own
    output value, a radius of 2 sums 5x5, etc., so the lifetime read-out
    for e.g. a 512x512 acquisition stays 512x512 at every bin radius.

    The window is clipped at the image edges rather than padded, so
    edge/corner pixels sum fewer contributing pixels -- their true
    spatial support -- instead of being biased by fabricated neighbours.
    Implemented as a 2-D integral image (summed-area table) so it stays
    a handful of vectorised operations regardless of radius.
    """

    radius = max(0, int(radius))
    if radius == 0:
        return cube
    height, width, _nt = cube.shape
    # A leading zero row/column turns every window sum into a single
    # inclusive-exclusive rectangle lookup, with no edge-case branching.
    integral = np.cumsum(np.cumsum(cube, axis=0, dtype=np.float64), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0), (0, 0)))
    y0 = np.clip(np.arange(height) - radius, 0, height)
    y1 = np.clip(np.arange(height) + radius + 1, 0, height)
    x0 = np.clip(np.arange(width) - radius, 0, width)
    x1 = np.clip(np.arange(width) + radius + 1, 0, width)
    top_left = integral[y0][:, x0]
    top_right = integral[y0][:, x1]
    bottom_left = integral[y1][:, x0]
    bottom_right = integral[y1][:, x1]
    windowed = bottom_right - top_right - bottom_left + top_left
    return windowed.astype(cube.dtype, copy=False)


def integrate_irf_per_column(irf_cube: np.ndarray) -> np.ndarray:
    """Collapses a per-pixel IRF cube's row axis by summing each column's
    trace across every row, then broadcasts that one high-SNR curve back
    to every row of the column.

    Matches the source LabVIEW workflow for a sweeping linear-array
    detector (e.g. an RA2): each column corresponds to one fixed physical
    detector element, and every row is another pass of the scan across
    it, so a raw per-pixel IRF acquisition is really many noisy single-
    sweep measurements of the *same* per-detector instrument response --
    summing them (not averaging; FLIMLib's own IRF normalisation cares
    about shape, not absolute scale) recovers a much higher-SNR curve
    than any single sweep pass alone, exactly as the original tool did
    ("integrate then repeat across the full sweeping scan").

    This also happens to be exactly what the native fast global-fit path
    needs (see build_irf_groups in native/flimfit_labview.c): every pixel
    in a column becomes byte-identical, so the fit's IRF-grouping
    optimisation collapses the whole image down to one group per column
    instead of effectively one group per pixel -- restoring the fast
    path's actual speedup for hardware that acquires IRF this way. A
    cube that isn't (y, x, time) -- a single shared trace with nothing to
    integrate -- is returned unchanged.
    """

    if irf_cube.ndim != 3:
        return irf_cube
    column_sum = irf_cube.sum(axis=0, keepdims=True, dtype=np.float64)  # (1, x, time)
    return np.broadcast_to(column_sum, irf_cube.shape).astype(irf_cube.dtype, copy=True)


def smooth_spatial(cube: np.ndarray, passes: int) -> np.ndarray:
    """Apply a small separable Gaussian-like spatial filter without SciPy."""

    result = np.asarray(cube, dtype=np.float32)
    for _ in range(max(0, int(passes))):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        result = (
            4.0 * padded[1:-1, 1:-1]
            + 2.0
            * (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            )
            + padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        ) / 16.0
    return result


def interpolate_bad_pixels(cube: np.ndarray, z_limit: float = 10.0) -> np.ndarray:
    """Replace dead/extreme spatial pixels with their local 3x3 median decay."""

    intensity = cube.sum(axis=-1)
    finite = np.isfinite(intensity)
    if not finite.any():
        return cube
    median = float(np.nanmedian(intensity[finite]))
    mad = float(np.nanmedian(np.abs(intensity[finite] - median)))
    scale = max(1.4826 * mad, 1.0)
    bad = (~finite) | (intensity < 0) | (np.abs(intensity - median) > z_limit * scale)
    bad &= intensity != 0
    if not bad.any():
        return cube
    output = cube.copy()
    padded = np.pad(cube, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    for y, x in np.argwhere(bad):
        neighbours = padded[y : y + 3, x : x + 3].reshape(-1, cube.shape[-1])
        output[y, x] = np.nanmedian(neighbours, axis=0)
    return output


def _reference_rows(reference: np.ndarray | None, target: np.ndarray) -> np.ndarray | None:
    if reference is None:
        return None
    ref = np.asarray(reference, dtype=np.float32)
    nt = target.shape[-1]
    if ref.ndim == 1:
        if ref.size != nt:
            raise ValueError("The IRF time-bin count does not match the data.")
        return np.broadcast_to(ref, target.shape).reshape(-1, nt)
    if ref.ndim != 3 or ref.shape[-1] != nt:
        raise ValueError("IRF data must be a 1-D decay or a compatible 3-D cube.")
    if ref.shape[:2] == target.shape[:2]:
        return ref.reshape(-1, nt)
    mean_ref = np.nanmean(ref, axis=(0, 1))
    return np.broadcast_to(mean_ref, target.shape).reshape(-1, nt)


def phasor_lifetimes(
    cube: np.ndarray,
    dt_ns: float,
    rep_rate_mhz: float,
    min_photons: float,
    irf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate first-harmonic phasors, including optional per-pixel IRF correction.

    This follows the equations embedded in the supplied LabVIEW VI: bin-centred
    Fourier components, optional complex division by the IRF phasor, and the
    half-bin correction used by that implementation.
    """

    height, width, nt = cube.shape
    rows = cube.reshape(-1, nt).astype(np.float64, copy=False)
    t_seconds = (np.arange(nt, dtype=np.float64) + 0.5) * float(dt_ns) * 1e-9
    omega = 2.0 * np.pi * float(rep_rate_mhz) * 1e6
    cosine = np.cos(omega * t_seconds)
    sine = np.sin(omega * t_seconds)
    totals = rows.sum(axis=1)
    valid_decay = totals >= float(min_photons)
    safe_totals = np.where(totals > 0, totals, 1.0)
    g_raw = (rows @ cosine) / safe_totals
    s_raw = (rows @ sine) / safe_totals
    z = g_raw + 1j * s_raw

    ref_rows = _reference_rows(irf, cube)
    if ref_rows is not None:
        ref_rows = ref_rows.astype(np.float64, copy=False)
        ref_rows = np.clip(ref_rows - np.median(ref_rows, axis=1, keepdims=True), 0.0, None)
        ref_totals = ref_rows.sum(axis=1)
        valid_ref = ref_totals > 0
        safe_ref = np.where(valid_ref, ref_totals, 1.0)
        z_ref = (ref_rows @ cosine) / safe_ref + 1j * (ref_rows @ sine) / safe_ref
        valid = valid_decay & valid_ref & (np.abs(z_ref) > 1e-12)
        corrected = np.full(z.shape, np.nan + 1j * np.nan, dtype=np.complex128)
        corrected[valid] = z[valid] / z_ref[valid]
        corrected[valid] *= np.exp(1j * omega * float(dt_ns) * 1e-9 / 2.0)
        z = corrected
    else:
        z = np.where(valid_decay, z, np.nan + 1j * np.nan)

    g = np.real(z)
    s = np.imag(z)
    phase = np.arctan2(s, g)
    modulation = np.abs(z)
    tau_phase_ns = np.tan(phase) / omega * 1e9
    tau_mod_ns = np.full_like(modulation, np.nan, dtype=np.float64)
    valid_mod = np.isfinite(modulation) & (modulation > 0) & (modulation <= 1)
    tau_mod_ns[valid_mod] = np.sqrt(1.0 / modulation[valid_mod] ** 2 - 1.0) / omega * 1e9
    shape = (height, width)
    return g.reshape(shape), s.reshape(shape), tau_phase_ns.reshape(shape), tau_mod_ns.reshape(shape)


# --------------------------------------------------------------------------
# Phasor-informed seeding for FIT_GLOBAL_MULTI_STRETCHEDEXP (up to 3
# independently-distributed components -- see native_backend.py and
# RECONSTRUCTION_NOTES.md for the full global-fit story this builds on).
#
# A THIRD seeding mode alongside "unseeded" (multi-start over arbitrary
# formulaic starting points -- native_backend.py's own comments) and
# "manual" (a user-supplied initial_taus_ns, e.g. via the "Set
# manually..." dialog -- [[manual-global-model-dialog]]): derive a
# starting guess automatically from the data's own phasor structure,
# with no user input needed, instead of either extreme.
#
# The idea (the user's own suggestion): phasors combine LINEARLY under
# mixing -- a pixel with amplitude fractions f_i of N components has a
# phasor coordinate equal to the amplitude-weighted average of each
# component's own (pure) phasor coordinate. For N genuinely-distinct
# components with pixels spanning a real range of mixing ratios, the
# whole image's pixel cloud in (g, s) space should span (approximately)
# the convex hull of the N pure-component phasor positions -- finding
# those EXTREME points and inverting each back to a (tau, h) pair gives
# a real, data-grounded starting guess, without needing any single
# "pure" pixel to be labeled in advance.
#
# Two things had to be got right for this to actually work (both
# confirmed by direct, careful validation -- see RECONSTRUCTION_NOTES.md
# for the numbers):
#
# 1. CALIBRATION MUST MATCH THE REAL MEASUREMENT PIPELINE EXACTLY, not
#    an idealized approximation. The standard mono-exponential phasor
#    formulas (tau_phase = tan(phase)/omega, etc.) are only exact for
#    h=1 -- for a genuinely stretched (h>1) component they're
#    systematically biased, and the bias itself depends on h (confirmed
#    directly: a bias factor of ~2.2x for h=1.5 growing to ~4.3x for
#    h=2.8, even for a perfectly pure, unmixed component). An
#    independently-derived theoretical (infinite-time-domain) model for
#    the stretched-exponential's own phasor ALSO turned out to be
#    insufficient on its own: real measurements are on a FINITE
#    acquisition window, which truncates a broad (large-h) component's
#    genuinely long tail, and a finite window's linear (not circular)
#    convolution with the IRF interacts with this app's own IRF-
#    correction formula in a way a clean theoretical model doesn't
#    capture exactly. The fix that actually closes this gap completely:
#    don't approximate at all -- directly SIMULATE the real pipeline
#    noise-free (build the pure decay, convolve with the actual IRF,
#    call this file's own phasor_lifetimes with IRF correction) for each
#    calibration grid point. This guarantees an exact match to what a
#    real, noise-free measurement of that same (tau, h) would show,
#    since it's the identical code path -- confirmed by exact round-trip
#    recovery (tau and h both reproduced to 4+ decimal places) on
#    isolated pure components at every tested (tau, h).
#
# 2. FINDING THE RIGHT EXTREME POINTS is not simply "the 2 or 3 most
#    separated pixels." For N=3, k-means cluster centroids don't work
#    (averaging many still-mixed pixels just gives another mixed point,
#    not a purer one -- phasors are linear, so averaging doesn't help);
#    naive greedy farthest-point selection also doesn't reliably find
#    the true THIRD vertex of the triangle the mixture spans (after 2
#    genuine extremes are found, "farthest from those 2 points" is not
#    the same question as "the third corner of the triangle", and can
#    land on an interior/still-mixed point instead). What worked:
#    perpendicular distance from the LINE through the first two chosen
#    points for the third vertex -- the correct geometric question for
#    finding a genuine third triangle corner. Confirmed directly: this
#    combination recovers all 3 true (tau, h) pairs to within a few
#    percent with zero prior knowledge, and the resulting fit (seeded
#    this way) matches truth-seeded quality almost exactly.
# --------------------------------------------------------------------------


def _stretched_exp_pipeline_phasor(
    tau_ns: float,
    h: float,
    n_bins: int,
    dt_ns: float,
    irf_1d: np.ndarray,
    rep_rate_mhz: float,
) -> tuple[float, float]:
    """Noise-free (g, s) for a pure stretched-exponential component,
    computed by directly running this file's own phasor_lifetimes on a
    simulated, IRF-convolved decay -- see the module comment above for
    why this (not an independently-derived formula) is what makes the
    calibration exact."""

    t_ns = (np.arange(n_bins, dtype=np.float64) + 0.5) * dt_ns
    shape = np.exp(-np.power(np.maximum(t_ns, 1e-9) / tau_ns, 1.0 / h))
    convolved = np.convolve(shape, irf_1d)[:n_bins]
    cube = convolved.reshape(1, 1, n_bins)
    g, s, _tau_phase, _tau_mod = phasor_lifetimes(cube, dt_ns, rep_rate_mhz, min_photons=0.0, irf=irf_1d)
    return float(g[0, 0]), float(s[0, 0])


def _build_phasor_calibration_table(
    n_bins: int,
    dt_ns: float,
    irf_1d: np.ndarray,
    rep_rate_mhz: float,
    tau_max_ns: float,
    h_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds a (tau_grid x h_grid) -> (g, s) lookup table, used only to
    seed the nonlinear inversion below with a good starting point (the
    grid itself doesn't need to be very fine). Cheap enough (a fraction
    of a second for the default resolution) to rebuild once per training
    call rather than caching across calls."""

    tau_grid = np.concatenate(
        [np.linspace(0.05, min(5.0, tau_max_ns), 40), np.linspace(min(5.0, tau_max_ns), tau_max_ns, 25)]
    )
    tau_grid = np.unique(tau_grid)
    h_grid = np.linspace(1.0, h_max, 24)
    g_table = np.zeros((len(tau_grid), len(h_grid)))
    s_table = np.zeros((len(tau_grid), len(h_grid)))
    for i, tau_ns in enumerate(tau_grid):
        for j, h in enumerate(h_grid):
            g_table[i, j], s_table[i, j] = _stretched_exp_pipeline_phasor(
                tau_ns, h, n_bins, dt_ns, irf_1d, rep_rate_mhz
            )
    return tau_grid, h_grid, g_table, s_table


def _invert_phasor_to_tau_h(
    g_obs: float,
    s_obs: float,
    tau_grid: np.ndarray,
    h_grid: np.ndarray,
    g_table: np.ndarray,
    s_table: np.ndarray,
    n_bins: int,
    dt_ns: float,
    irf_1d: np.ndarray,
    rep_rate_mhz: float,
    tau_max_ns: float,
    h_max: float,
) -> tuple[float, float]:
    """Nearest-table-cell lookup, then a small bounded nonlinear refine
    (scipy.optimize.least_squares) against the exact pipeline-simulated
    phasor -- the table alone is coarse; the refine step is what gets
    the essentially-exact recovery this feature depends on."""

    from scipy.optimize import least_squares

    d2 = (g_table - g_obs) ** 2 + (s_table - s_obs) ** 2
    i0, j0 = np.unravel_index(np.argmin(d2), d2.shape)
    tau0, h0 = float(tau_grid[i0]), float(h_grid[j0])

    def residual(params: np.ndarray) -> list[float]:
        tau_ns, h = params
        if tau_ns <= 0.0 or h < 1.0 or h > h_max:
            return [1e3, 1e3]
        g, s = _stretched_exp_pipeline_phasor(tau_ns, h, n_bins, dt_ns, irf_1d, rep_rate_mhz)
        return [g - g_obs, s - s_obs]

    solution = least_squares(
        residual, [tau0, h0], bounds=([1e-3, 1.0], [tau_max_ns, h_max]), diff_step=1e-4
    )
    return float(solution.x[0]), float(solution.x[1])


def _select_phasor_vertices(g: np.ndarray, s: np.ndarray, model_order: int) -> list[int]:
    """Picks model_order pixel indices whose (g, s) positions are good
    proxies for the model_order pure-component phasor locations -- see
    the module comment above for why each case is implemented the way
    it is (plain farthest-point works for 1-2 vertices; 3 needs the
    line-perpendicular-distance step for a genuine third corner)."""

    if model_order <= 1:
        # No real "vertex" question for a single shared component --
        # just use the pixel closest to the cloud's own centroid (a
        # robust "typical" representative, unlike a lone extreme point).
        g_c, s_c = float(np.mean(g)), float(np.mean(s))
        idx = int(np.argmin((g - g_c) ** 2 + (s - s_c) ** 2))
        return [idx]

    idx0 = int(np.argmax(g**2 + s**2))
    d0 = (g - g[idx0]) ** 2 + (s - s[idx0]) ** 2
    idx1 = int(np.argmax(d0))
    if model_order == 2:
        return [idx0, idx1]

    p0 = np.array([g[idx0], s[idx0]])
    p1 = np.array([g[idx1], s[idx1]])
    line = p1 - p0
    line_len = float(np.hypot(*line))
    pts = np.stack([g, s], axis=1) - p0
    cross = np.abs(pts[:, 0] * line[1] - pts[:, 1] * line[0]) / max(line_len, 1e-12)
    cross[idx0] = -1.0
    cross[idx1] = -1.0
    idx2 = int(np.argmax(cross))
    return [idx0, idx1, idx2]


def phasor_informed_seed(
    cube: np.ndarray,
    dt_ns: float,
    rep_rate_mhz: float,
    model_order: int,
    irf: np.ndarray | None = None,
    min_photons: float = 200.0,
) -> tuple[float, ...] | None:
    """Derives an `initial_taus_ns`-shaped seed for FIT_GLOBAL_MULTI_
    STRETCHEDEXP (fit_distribution_model=2) automatically from the
    data's own phasor structure -- see this module's own comment block
    above for the full method and why each design choice was needed.

    Returns a `2*model_order`-length tuple of (tau_ns, h) pairs, sorted
    by tau, suitable for AnalysisSettings.initial_taus_ns directly. NOT
    guaranteed to be a good seed for every dataset (e.g. a truly
    single-component or highly homogeneous image gives the search
    little genuine mixing-ratio information to work with -- see
    RECONSTRUCTION_NOTES.md's account of why that information is what
    this whole approach depends on) -- callers should treat this as a
    strong default, not an infallible one; a caller-supplied explicit
    seed should always be preferred over this when the user has one
    (matching how has_explicit_shared_seed already works natively).
    Returns None if there isn't enough usable per-pixel phasor signal
    to attempt this at all.
    """

    if model_order < 1 or model_order > 3:
        return None

    height, width, n_bins = cube.shape
    if irf is None:
        irf_1d = np.ones(n_bins, dtype=np.float64) / n_bins
    else:
        irf_arr = np.asarray(irf, dtype=np.float64)
        irf_1d = irf_arr if irf_arr.ndim == 1 else np.nanmean(irf_arr, axis=tuple(range(irf_arr.ndim - 1)))
        irf_1d = np.clip(irf_1d - np.median(irf_1d), 0.0, None)
        total = irf_1d.sum()
        irf_1d = irf_1d / total if total > 0 else np.ones(n_bins, dtype=np.float64) / n_bins

    g, s, _tau_phase, _tau_mod = phasor_lifetimes(cube, dt_ns, rep_rate_mhz, min_photons, irf=irf)
    g_flat, s_flat = g.ravel(), s.ravel()
    valid = np.isfinite(g_flat) & np.isfinite(s_flat)
    if int(valid.sum()) < model_order + 1:
        return None
    g_v, s_v = g_flat[valid], s_flat[valid]

    window_ns = n_bins * dt_ns
    tau_max_ns = 1.5 * window_ns
    h_max = 4.5
    tau_grid, h_grid, g_table, s_table = _build_phasor_calibration_table(
        n_bins, dt_ns, irf_1d, rep_rate_mhz, tau_max_ns, h_max
    )

    vertex_indices = _select_phasor_vertices(g_v, s_v, model_order)
    components: list[tuple[float, float]] = []
    for idx in vertex_indices:
        tau_ns, h = _invert_phasor_to_tau_h(
            g_v[idx], s_v[idx], tau_grid, h_grid, g_table, s_table,
            n_bins, dt_ns, irf_1d, rep_rate_mhz, tau_max_ns, h_max,
        )
        components.append((tau_ns, h))
    components.sort()

    seed: list[float] = []
    for tau_ns, h in components:
        seed.append(tau_ns)
        seed.append(h)
    return tuple(seed)


def mono_exponential_fit(
    cube: np.ndarray,
    time_ns: np.ndarray,
    min_photons: float,
    start_bin: int = -1,
    end_bin: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Fast vectorised weighted log-linear mono-exponential fit."""

    height, width, nt = cube.shape
    mean_decay = np.nanmean(cube, axis=(0, 1))
    start = int(np.nanargmax(mean_decay)) if start_bin < 0 else int(start_bin)
    end = nt if end_bin < 0 else min(int(end_bin), nt)
    start = max(0, min(start, nt - 2))
    end = max(start + 2, end)
    y = cube[..., start:end].reshape(-1, end - start).astype(np.float64, copy=False)
    x = np.asarray(time_ns[start:end], dtype=np.float64)
    totals = cube.reshape(-1, nt).sum(axis=1)
    valid = totals >= float(min_photons)

    positive = y > 0
    weights = np.where(positive, y, 0.0)
    log_y = np.log(np.where(positive, y, 1.0))
    sw = weights.sum(axis=1)
    sx = weights @ x
    sy = (weights * log_y).sum(axis=1)
    sxx = weights @ (x * x)
    sxy = (weights * log_y) @ x
    denom = sw * sxx - sx * sx
    slope = np.divide(sw * sxy - sx * sy, denom, out=np.full_like(sw, np.nan), where=np.abs(denom) > 1e-12)
    intercept = np.divide(sy - slope * sx, sw, out=np.full_like(sw, np.nan), where=sw > 0)
    tau = np.divide(-1.0, slope, out=np.full_like(slope, np.nan), where=slope < -1e-12)
    valid &= np.isfinite(tau) & (tau > 0)
    tau[~valid] = np.nan

    predicted = np.exp(intercept[:, None] + slope[:, None] * x[None, :])
    chi = np.nanmean((y - predicted) ** 2 / np.maximum(predicted, 1.0), axis=1)
    chi[~valid] = np.nan
    return tau.reshape(height, width), chi.reshape(height, width), predicted, start, end


def estimate_synthetic_gaussian_irf(
    cube: np.ndarray, time_ns: np.ndarray, fwhm_ns_override: float = -1.0
) -> tuple[np.ndarray, dict]:
    """Approximate the instrument response as a Gaussian pulse when no
    measured IRF is available, for phasor IRF correction and FLIMLib
    reconvolution fitting alike.

    This is a standard simplification in the TCSPC/FLIM literature (e.g.
    Becker, *Advanced Time-Correlated Single Photon Counting Techniques*,
    2005; O'Connor & Phillips, *Time-Correlated Single Photon Counting*,
    1984): for the overwhelming majority of pulsed-diode/fibre-laser
    systems with a fast detector, the combined effect of the laser pulse
    width, detector transit-time spread, and electronics jitter is well
    approximated by a Gaussian. It is *not* a substitute for a measured
    IRF -- it can't capture a genuine detector afterpulsing tail or real
    pulse asymmetry -- but it lets IRF-corrected phasor analysis and
    reconvolution fitting run at all when no real IRF is available,
    rather than refusing to process the data.

    Unless ``fwhm_ns_override`` pins a specific value (>0), the FWHM is
    estimated from the *sample's own* rising edge: for a fluorophore whose
    true excited-state rise is fast relative to the instrument response
    (true for the large majority of one/two-photon FLIM), the recorded
    10-90% rise time of the mean decay is dominated by the IRF, not by the
    fluorescence itself. The standard Gaussian step-response relation
    converts that rise time to a FWHM: convolving a Gaussian impulse
    response with a step gives an error-function edge whose 10-90% rise
    time is ``2*sqrt(2)*erfinv(0.8)*sigma`` (~2.5631*sigma), while
    ``FWHM = 2*sqrt(2*ln(2))*sigma`` (~2.3548*sigma) -- so
    ``FWHM ~= 0.9187 * (10-90% rise time)``.
    """

    # 2*sqrt(2*ln2) / (2*sqrt(2)*erfinv(0.8)); erfinv(0.8) ~= 0.9061938.
    RISE_TIME_TO_FWHM = 0.918722
    FWHM_TO_SIGMA = 2.3548200450309493  # 2*sqrt(2*ln2)

    height, width, nt = cube.shape
    mean_decay = np.nanmean(cube, axis=(0, 1)).astype(np.float64)
    dt_ns = float(time_ns[1] - time_ns[0]) if nt > 1 else 1.0
    # A light 3-point smooth before locating the rise, so a single noisy
    # bin can't shift the threshold crossing -- the output is still a
    # clean analytic Gaussian regardless, this only affects where it's
    # centred and how wide it's estimated to be.
    if nt >= 3:
        padded = np.pad(mean_decay, (1, 1), mode="edge")
        smoothed = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    else:
        smoothed = mean_decay

    peak_index = int(np.nanargmax(smoothed))
    baseline = float(np.nanmin(smoothed[: peak_index + 1]))
    peak_value = float(smoothed[peak_index])
    span = peak_value - baseline

    def crossing(fraction: float) -> float | None:
        """Linearly-interpolated time of the first rising-edge crossing of
        ``baseline + fraction * span``, searching only up to the peak."""

        if span <= 0 or peak_index < 1:
            return None
        target = baseline + fraction * span
        for i in range(1, peak_index + 1):
            y0, y1 = smoothed[i - 1], smoothed[i]
            if y0 <= target <= y1:
                t0, t1 = float(time_ns[i - 1]), float(time_ns[i])
                fraction_between = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
                return t0 + fraction_between * (t1 - t0)
        return None

    t10, t50, t90 = crossing(0.10), crossing(0.50), crossing(0.90)

    if fwhm_ns_override is not None and fwhm_ns_override > 0:
        fwhm_ns = float(fwhm_ns_override)
        estimation_method = "manual_override"
    elif t10 is not None and t90 is not None and t90 > t10:
        fwhm_ns = max((t90 - t10) * RISE_TIME_TO_FWHM, dt_ns * 0.5)
        estimation_method = "rise_time_10_90"
    else:
        # Rising edge too short/flat/noisy to measure reliably -- fall
        # back to a width tied to this acquisition's own time resolution
        # rather than an assumed literature value that may not suit this
        # instrument at all.
        fwhm_ns = 3.0 * dt_ns
        estimation_method = "fallback_bin_width"

    center_ns = t50 if t50 is not None else float(time_ns[peak_index])
    sigma_ns = fwhm_ns / FWHM_TO_SIGMA

    profile = np.exp(-0.5 * ((time_ns.astype(np.float64) - center_ns) / sigma_ns) ** 2)
    total = float(profile.sum())
    if total > 0:
        profile = profile / total  # unit-area, matching a normalised measured IRF trace

    metadata = {
        "method": "synthetic_gaussian",
        "estimation_method": estimation_method,
        "fwhm_ns": float(fwhm_ns),
        "center_ns": float(center_ns),
        "rise_10_90_ns": (float(t90 - t10) if (t10 is not None and t90 is not None) else None),
        "note": (
            "Approximate instrument response modelled as a Gaussian pulse -- "
            "not a substitute for a measured IRF. Standard TCSPC simplification; "
            "see Becker (2005) / O'Connor & Phillips (1984)."
        ),
    }
    return profile.astype(np.float32), metadata


def moment_lifetime(cube: np.ndarray, time_ns: np.ndarray, min_photons: float, irf: np.ndarray | None) -> np.ndarray:
    totals = cube.sum(axis=-1)
    mean_decay = cube.mean(axis=(0, 1))
    zero_bin = int(np.argmax(mean_decay))
    zero_time = float(time_ns[zero_bin])
    if irf is not None:
        ref = np.asarray(irf, dtype=np.float64)
        ref_decay = ref if ref.ndim == 1 else ref.mean(axis=tuple(range(ref.ndim - 1)))
        ref_decay = np.clip(ref_decay - np.median(ref_decay), 0.0, None)
        if ref_decay.sum() > 0:
            zero_time = float(np.sum(ref_decay * time_ns) / np.sum(ref_decay))
    lifetime = np.sum(cube * (time_ns[None, None, :] - zero_time), axis=-1) / np.maximum(totals, 1.0)
    lifetime[(totals < min_photons) | (lifetime <= 0)] = np.nan
    return lifetime


def fit_single_decay(decay: np.ndarray, time_ns: np.ndarray, start_bin: int = -1, end_bin: int = -1) -> tuple[np.ndarray, float]:
    cube = np.asarray(decay, dtype=np.float64).reshape(1, 1, -1)
    tau, _, predicted, start, end = mono_exponential_fit(cube, time_ns, 0.0, start_bin, end_bin)
    fit = np.full(decay.shape, np.nan, dtype=np.float64)
    fit[start:end] = predicted[0]
    return fit, float(tau[0, 0])


def native_multiexponential_curve(
    total_bins: int,
    native_bin_width_ns: float,
    irf_trace: np.ndarray,
    tau_components_ns: np.ndarray,
    amplitude_components: np.ndarray,
    background_values: np.ndarray,
) -> np.ndarray:
    """Reconstruct a display-only native reconvolution curve for one region.

    Sums the per-pixel FLIMLib reconvolution models
    (``background + IRF (x) sum_k A_k * exp(-t / tau_k)``) across the supplied
    pixels' fitted components. This mirrors the model FLIMLib fits
    (``GCI_multiexp_tau``) but is recomputed independently in Python purely
    for plotting; it is not guaranteed to be bin-identical to the DLL's
    internal fitted curve, and per-pixel IRF fits use the mean IRF trace as
    an approximation here (the native per-pixel fit itself still used the
    true per-pixel IRF).

    ``tau_components_ns`` and ``amplitude_components`` are ``(order, npix)``;
    ``background_values`` is ``(npix,)``. Only finite, positive taus
    contribute. Returns a length-``total_bins`` array, NaN beyond the native
    fit window.
    """

    irf_trace = np.asarray(irf_trace, dtype=np.float64)
    n_bins = int(irf_trace.size)
    full = np.full(int(total_bins), np.nan, dtype=np.float64)
    if n_bins == 0 or n_bins > total_bins:
        return full
    tau = np.asarray(tau_components_ns, dtype=np.float64)
    amplitude = np.asarray(amplitude_components, dtype=np.float64)
    native_time = (np.arange(n_bins, dtype=np.float64) + 0.5) * float(native_bin_width_ns)
    if tau.size == 0:
        exp_sum = np.zeros(n_bins, dtype=np.float64)
    else:
        valid = np.isfinite(tau) & (tau > 0) & np.isfinite(amplitude)
        safe_tau = np.where(valid, tau, np.inf)
        safe_amplitude = np.where(valid, amplitude, 0.0)
        with np.errstate(over="ignore"):
            contributions = safe_amplitude[:, :, None] * np.exp(
                -native_time[None, None, :] / safe_tau[:, :, None]
            )
        exp_sum = np.sum(contributions, axis=(0, 1))
    irf_sum = float(np.sum(irf_trace))
    irf_norm = irf_trace / irf_sum if irf_sum > 0 else irf_trace
    convolved = np.convolve(exp_sum, irf_norm, mode="full")[:n_bins]
    finite_background = np.asarray(background_values, dtype=np.float64)
    finite_background = finite_background[np.isfinite(finite_background)]
    curve = convolved + float(np.sum(finite_background))
    full[:n_bins] = curve
    return full


def _positive_metadata_value(
    metadata: dict[str, object],
    standard_key: str,
    legacy_aliases: tuple[str, ...],
) -> tuple[float | None, str | None]:
    """Read a positive acquisition-normalisation value with legacy ICS support."""

    candidates: list[tuple[str, object]] = []
    if standard_key in metadata:
        candidates.append((standard_key, metadata[standard_key]))
    aliases = {re.sub(r"[^a-z0-9]+", " ", value.lower()).strip() for value in legacy_aliases}
    for key, value in metadata.items():
        normalized_key = re.sub(r"[^a-z0-9]+", " ", str(key).lower()).strip()
        if normalized_key in aliases and key != standard_key:
            candidates.append((str(key), value))
    for key, value in candidates:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric > 0:
            return numeric, key
    return None, None


def resolve_background_multiplier(
    dataset: FLIMDataset,
    background: FLIMDataset | np.ndarray | None,
    settings: AnalysisSettings,
) -> tuple[float, dict[str, object]]:
    """Resolve the acquisition background multiplier and explain its provenance.

    Automatic normalisation first uses the sample/background scan-loop ratio,
    then their acquisition-time ratio.  Both ratios express the exposure of the
    sample relative to the background; for the supplied 20-loop sample and
    800-loop background this is ``20 / 800 = 0.025``.  If neither pair is
    available, the explicitly configured automatic fallback is used.
    """

    mode = str(settings.background_scale_mode).strip().lower()
    if mode not in {"auto", "manual"}:
        raise ValueError("Background scale mode must be 'auto' or 'manual'.")
    manual = float(settings.background_multiplier)
    fallback = float(settings.background_auto_fallback)
    for label, value in (("background multiplier", manual), ("background automatic fallback", fallback)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"The {label} must be finite and non-negative.")

    details: dict[str, object] = {
        "requested_mode": mode,
        "manual_multiplier": manual,
        "automatic_fallback": fallback,
        "basis": "manual" if mode == "manual" else "automatic_fallback",
        "sample_normalization_value": None,
        "background_normalization_value": None,
        "sample_metadata_key": None,
        "background_metadata_key": None,
    }
    if background is None:
        details.update({"basis": "no_background", "applied_multiplier": 0.0})
        return 0.0, details
    if mode == "manual":
        details["applied_multiplier"] = manual
        return manual, details

    background_metadata = background.metadata if isinstance(background, FLIMDataset) else {}
    ratio_specs = (
        (
            "scan_loops",
            (
                "scan loops",
                "scan no of loops",
                "history scan no of loops",
                "number of scan loops",
            ),
        ),
        (
            "acquisition_time_s",
            (
                "acquisition time s",
                "scan acquisition time",
                "scan acq time",
                "history scan acq time",
            ),
        ),
    )
    for standard_key, aliases in ratio_specs:
        sample_value, sample_key = _positive_metadata_value(dataset.metadata, standard_key, aliases)
        background_value, background_key = _positive_metadata_value(
            background_metadata, standard_key, aliases
        )
        if sample_value is None or background_value is None:
            continue
        multiplier = sample_value / background_value
        if not np.isfinite(multiplier) or multiplier < 0:
            continue
        details.update(
            {
                "basis": standard_key,
                "sample_normalization_value": sample_value,
                "background_normalization_value": background_value,
                "sample_metadata_key": sample_key,
                "background_metadata_key": background_key,
                "applied_multiplier": multiplier,
            }
        )
        return float(multiplier), details

    details["applied_multiplier"] = fallback
    return fallback, details


def _validate_background_compatibility(
    dataset: FLIMDataset,
    background: FLIMDataset | np.ndarray | None,
) -> dict[str, object]:
    details: dict[str, object] = {
        "sample_time_bins": int(dataset.time_bins),
        "sample_dt_ns": float(dataset.dt_ns),
        "background_time_bins": None,
        "background_dt_ns": None,
        "time_bins_validated": False,
        "time_bin_width_validated": False,
    }
    if background is None:
        return details
    array = background.decay if isinstance(background, FLIMDataset) else np.asarray(background)
    if array.ndim == 0:
        details["background_time_bins"] = "scalar"
    elif array.ndim == 1:
        details["background_time_bins"] = int(array.size)
        if array.size != dataset.time_bins:
            raise ValueError(
                f"Background time-bin count ({array.size}) does not match the sample "
                f"({dataset.time_bins})."
            )
        details["time_bins_validated"] = True
    elif array.ndim == 3:
        details["background_time_bins"] = int(array.shape[-1])
        if array.shape[-1] != dataset.time_bins:
            raise ValueError(
                f"Background time-bin count ({array.shape[-1]}) does not match the sample "
                f"({dataset.time_bins})."
            )
        details["time_bins_validated"] = True
    else:
        raise ValueError(
            "Background data must be a scalar, a one-dimensional time trace, or a three-dimensional cube."
        )
    if isinstance(background, FLIMDataset):
        details["background_dt_ns"] = float(background.dt_ns)
        if not np.isclose(background.dt_ns, dataset.dt_ns, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"Background time-bin width ({background.dt_ns:g} ns) does not match the sample "
                f"({dataset.dt_ns:g} ns)."
            )
        details["time_bin_width_validated"] = True
    return details


def _prepare_background(background: np.ndarray | None, target: np.ndarray) -> np.ndarray | float:
    if background is None:
        return 0.0
    bg = np.asarray(background, dtype=np.float32)
    bg = np.where(np.isfinite(bg), bg, 0.0)
    if bg.ndim == 0:
        return float(bg)
    if bg.shape == target.shape:
        return bg
    if bg.ndim == 3:
        if bg.shape[-1] != target.shape[-1]:
            raise ValueError("The background time-bin count does not match the sample.")
        return np.nanmean(bg, axis=(0, 1))[None, None, :]
    if bg.ndim == 1:
        if bg.size != target.shape[-1]:
            raise ValueError("The background time-bin count does not match the sample.")
        return bg[None, None, :]
    raise ValueError(
        "Background data must be a scalar, a one-dimensional time trace, or a three-dimensional cube."
    )


def _background_spatial_mode(background: np.ndarray | None, target: np.ndarray) -> str:
    if background is None:
        return "none"
    shape = np.asarray(background).shape
    if not shape:
        return "scalar"
    if shape == target.shape:
        return "per_pixel_cube"
    if len(shape) == 1:
        return "shared_time_trace"
    return "spatial_mean_time_trace"


def _preprocess_training_image(
    dataset: FLIMDataset,
    irf: FLIMDataset | np.ndarray,
    background: FLIMDataset | np.ndarray | None,
    cfg: AnalysisSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess one global-training image consistently with ``analyze_dataset``.

    A reduced pipeline for the first version of global training: background
    subtraction, timing shift, and spatial binning are applied, matching the
    corresponding steps ``analyze_dataset`` runs for a single image.
    Per-image detector-row/column interpolation and outlier smoothing are
    not applied here yet.

    Returns the processed ``(rows, columns, bins)`` decay cube and an IRF
    cube of the same spatial shape (a plain trace or an IRF with a different
    frame size is broadcast/averaged up to match), ready to stack alongside
    other training images.
    """

    cube = np.asarray(dataset.decay, dtype=np.float32)
    cube = np.where(np.isfinite(cube), cube, 0.0)
    bg_data = background.decay if isinstance(background, FLIMDataset) else background
    irf_data = irf.decay if isinstance(irf, FLIMDataset) else irf
    if irf_data is None:
        raise ValueError(f"Training image '{dataset.name}' needs an IRF.")

    _validate_background_compatibility(dataset, background)
    background_multiplier, _ = resolve_background_multiplier(dataset, background, cfg)
    prepared_background = _prepare_background(bg_data, cube)
    cube = cube - background_multiplier * prepared_background - cfg.background_counts
    if cfg.background_clip_negative:
        cube = np.clip(cube, 0.0, None)
    cube = cube.astype(np.float32, copy=False)
    cube = shift_time_axis(cube, cfg.timing_offset_bins)

    cube = rolling_spatial_bin(cube, cfg.spatial_bin)

    n_bins = dataset.decay.shape[-1]
    irf_array = np.asarray(irf_data, dtype=np.float32)
    if irf_array.ndim == 1:
        if irf_array.shape[0] != n_bins:
            raise ValueError(f"Training image '{dataset.name}': IRF trace length does not match its decay bins.")
        irf_array = np.broadcast_to(irf_array, dataset.decay.shape).copy()
    elif irf_array.ndim != 3 or irf_array.shape[-1] != n_bins:
        raise ValueError(
            f"Training image '{dataset.name}': IRF must be a 1D trace or an (y, x, time) "
            "cube matching the decay time bins."
        )
    elif irf_array.shape[:2] != dataset.decay.shape[:2]:
        # A calibration IRF acquired at a different frame size than this
        # training image: fall back to its mean trace broadcast across this
        # image's own pixel grid.
        mean_trace = np.nanmean(irf_array, axis=(0, 1))
        irf_array = np.broadcast_to(mean_trace, dataset.decay.shape).copy()
    irf_array = shift_time_axis(irf_array, cfg.timing_offset_bins)
    irf_array = rolling_spatial_bin(irf_array, cfg.spatial_bin)
    if cfg.fit_irf_mode == 1 and cfg.irf_integrate_per_column:
        irf_array = integrate_irf_per_column(irf_array)

    return cube, irf_array


def train_global_model(
    items: Sequence[TrainingImageInput],
    settings: AnalysisSettings,
    progress: ProgressCallback | None = None,
    *,
    name: str | None = None,
    phasor_seed: bool = False,
) -> TrainedGlobalModel:
    """Train one shared-lifetime model across several training images.

    Reuses the same native-fit settings ``analyze_dataset``'s FLIMLib
    methods use (model order, noise model, background mode, IRF mode,
    global iterations, and so on -- everything under "NATIVE FLIMLIB FIT"),
    so a model trained here can be applied later under matching conditions.
    Each image is preprocessed independently (its own background/IRF) via
    ``_preprocess_training_image``, then stacked; every image must end up
    the same processed shape after spatial binning.

    ``phasor_seed``: a THIRD seeding mode for FIT_GLOBAL_MULTI_STRETCHEDEXP
    (fit_distribution_model=2), alongside "unseeded" (the default, no
    initial_taus_ns given -- native multi-start over formulaic starting
    points) and "manual" (settings.initial_taus_ns already populated,
    e.g. via the "Set manually..." dialog). When True AND the caller
    hasn't already supplied an explicit seed, derives one automatically
    from the (first) training image's own phasor structure via
    ``phasor_informed_seed`` -- see that function's own module-level
    comment for the method and why it works. An explicit
    ``initial_taus_ns`` from the caller always takes priority over this
    (matching how the native side's own has_explicit_shared_seed already
    prioritizes an explicit seed over its own multi-start machinery).
    Silently falls back to the caller's own settings (typically unseeded)
    if a phasor-derived seed can't be computed (e.g. too few valid
    pixels, or a fit type/order this doesn't support) -- this is a
    best-effort default, never a hard requirement.
    """

    if not items:
        raise ValueError("Global training requires at least one training image.")
    reference_dt_ns = float(items[0].dataset.dt_ns)
    for item in items[1:]:
        item_dt_ns = float(item.dataset.dt_ns)
        if not np.isclose(item_dt_ns, reference_dt_ns, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "Training images must share the same time-bin width -- "
                f"'{items[0].name or items[0].dataset.name}' is {reference_dt_ns:g} ns, but "
                f"'{item.name or item.dataset.name}' is {item_dt_ns:g} ns."
            )
    # Each training image's *own* (now confirmed-consistent) dt_ns is what
    # actually describes this training set -- not whatever happens to be
    # showing in the main window's "Time-bin width" field at the moment
    # "Train" is clicked, which reflects whichever dataset is loaded in
    # the main window and has no necessary connection to this dialog's
    # training set. Overriding it on a copy (not the caller's own
    # `settings`) before normalizing also correctly carries through to
    # native_bin_width_ns's own <=0-means-auto fallback in .normalized(),
    # rather than that having already resolved to the wrong value.
    settings = replace(settings, dt_ns=reference_dt_ns)
    cfg = settings.normalized(items[0].dataset.time_bins)
    if cfg.model_order < 1:
        raise ValueError("Model order must be at least 1 for global training.")

    _notify(progress, 0.02, "Preparing training images")
    processed_cubes: list[np.ndarray] = []
    processed_irfs: list[np.ndarray] = []
    for index, item in enumerate(items):
        label = item.name or item.dataset.name
        _notify(progress, 0.02 + 0.3 * index / len(items), f"Preprocessing {label}")
        cube, irf_cube = _preprocess_training_image(item.dataset, item.irf, item.background, cfg)
        processed_cubes.append(cube)
        processed_irfs.append(irf_cube)

    reference_shape = processed_cubes[0].shape
    for item, cube in zip(items, processed_cubes):
        if cube.shape != reference_shape:
            raise ValueError(
                "All training images must have the same processed shape after spatial "
                f"binning; '{item.name or item.dataset.name}' is {cube.shape}, expected {reference_shape}."
            )

    native_input_bins = (
        reference_shape[-1] if cfg.native_fit_bins < 0 else min(cfg.native_fit_bins, reference_shape[-1])
    )
    decay_stack = np.stack([cube[..., :native_input_bins] for cube in processed_cubes], axis=0)
    irf_stack = np.stack([irf[..., :native_input_bins] for irf in processed_irfs], axis=0)
    image_count, rows, columns, _n_bins = decay_stack.shape
    mask = np.ones((image_count, rows, columns), dtype=np.float64)
    # The native DLL's own max_global_pixels cap (flim_train_global_model_v1,
    # native/flimfit_labview.c) fills sequentially in (image, row, column)
    # order and simply stops accepting pixels once the cap is hit. With more
    # than one training image and a typical acquisition size -- a single
    # 512x512 image is already 262,144 pixels, versus this app's own 20,000
    # default cap -- that silently means only the *first* training image
    # (or part of it) ever contributes a single pixel to the fit; every
    # later image is entirely excluded (its whole fit_status map comes back
    # -7, "excluded by max pixels cap") without any visible error, and the
    # trained taus reflect only that first image's species. Pre-subsampling
    # each image's own mask down to a fair per-image share here -- before
    # the DLL ever sees it -- guarantees every training image actually
    # contributes pixels regardless of table order or relative size, which
    # is what "shared taus across several images" is supposed to mean.
    # Deterministic (fixed seed), so repeated training runs on the same
    # inputs pick the same pixels rather than drifting between runs.
    per_image_pixels = rows * columns
    if image_count > 1 and cfg.max_global_pixels > 0:
        per_image_budget = max(1, cfg.max_global_pixels // image_count)
        if per_image_budget < per_image_pixels:
            # Draw the per-image sample from pixels that would clear the
            # fit's own minimum-photon-count threshold anyway (the same
            # cfg.fit_min_counts passed to the DLL as minPhotons below),
            # not the raw pixel grid -- background/dim pixels the DLL
            # would immediately reject (fit_status -2) don't deserve a
            # share of an already-scarce budget. This is the same
            # thresholding-as-masking idea analyze_dataset already applies
            # to the single-image global-fit method's own `native_mask`
            # (see the flimlib_global branch below); training's mask
            # simply never had it applied before this.
            eligible_by_counts = decay_stack.sum(axis=-1) >= cfg.fit_min_counts  # (image_count, rows, columns)
            rng = np.random.default_rng(0)
            for index in range(image_count):
                flat_mask = mask[index].reshape(-1)  # a view: mutating it mutates `mask` in place
                eligible = np.flatnonzero(eligible_by_counts[index])
                if eligible.size > per_image_budget:
                    keep = rng.choice(eligible, size=per_image_budget, replace=False)
                    flat_mask[:] = 0.0
                    flat_mask[keep] = 1.0
                elif eligible.size < per_image_pixels:
                    # At or under budget already -- use every eligible
                    # pixel outright rather than randomly sampling a subset
                    # of them (there's no reason to leave any on the table).
                    flat_mask[:] = 0.0
                    flat_mask[eligible] = 1.0
                # else every pixel in this image already clears the
                # threshold and there are still fewer of them than the
                # budget -- leave this image's mask as all-ones.

    native_start = 0 if cfg.fit_start_bin < 0 else min(cfg.fit_start_bin, native_input_bins - 2) + 1
    native_stop = (
        0 if cfg.fit_end_bin < 0 else max(native_start + 1, min(cfg.fit_end_bin, native_input_bins))
    )

    initial_taus_ns = cfg.initial_taus_ns
    if phasor_seed and not initial_taus_ns and cfg.fit_distribution_model == 2:
        # First training image only -- see phasor_informed_seed's own
        # comment for why this is a best-effort default, not guaranteed;
        # a failed/None attempt just leaves initial_taus_ns empty
        # (unseeded), same as if phasor_seed had never been requested.
        derived = phasor_informed_seed(
            decay_stack[0],
            cfg.native_bin_width_ns,
            float(items[0].dataset.rep_rate_mhz),
            cfg.model_order,
            irf=irf_stack[0],
            min_photons=cfg.fit_min_counts,
        )
        if derived is not None:
            initial_taus_ns = derived

    _notify(progress, 0.35, "Training shared lifetime model")
    result = train_global_in_subprocess(
        decay_stack,
        irf_stack,
        mask,
        dll_path=cfg.native_dll_path or None,
        bin_width_ns=cfg.native_bin_width_ns,
        model_order=cfg.model_order,
        fit_type=cfg.fit_distribution_model,
        background_mode=cfg.fit_background_mode,
        background_value=cfg.fit_background_value,
        noise_model=cfg.fit_noise_model,
        min_photons=cfg.fit_min_counts,
        fit_start_bin=native_start,
        fit_stop_bin=native_stop,
        max_global_pixels=cfg.max_global_pixels,
        irf_mode=cfg.fit_irf_mode,
        global_iterations=cfg.global_iterations,
        initial_taus_ns=initial_taus_ns,
    )
    _notify(progress, 0.9, "Recording trained model")

    def source_path(value: FLIMDataset | np.ndarray | None) -> str:
        return str(value.source_path) if isinstance(value, FLIMDataset) and value.source_path else ""

    training_images = tuple(
        TrainingImageProvenance(
            name=item.name or item.dataset.name,
            decay_source=source_path(item.dataset),
            irf_source=source_path(item.irf),
            background_source=source_path(item.background),
            shape_yx=(rows, columns),
            total_pixels=rows * columns,
            selected_pixels=int(np.count_nonzero(result.fit_status[index] == 1)),
        )
        for index, item in enumerate(items)
    )

    model = build_trained_model(
        name=name or new_model_name(cfg.model_order),
        global_taus_ns=result.global_taus_ns,
        global_stats=result.global_stats,
        model_order=cfg.model_order,
        fit_distribution_model=cfg.fit_distribution_model,
        dt_ns=cfg.dt_ns,
        background_mode=cfg.fit_background_mode,
        background_value=cfg.fit_background_value,
        noise_model=cfg.fit_noise_model,
        fit_min_counts=cfg.fit_min_counts,
        fit_start_bin=cfg.fit_start_bin,
        fit_end_bin=cfg.fit_end_bin,
        native_fit_bins=cfg.native_fit_bins,
        native_bin_width_ns=cfg.native_bin_width_ns,
        native_irf_baseline_mode=cfg.native_irf_baseline_mode,
        irf_mode=cfg.fit_irf_mode,
        irf_integrate_per_column=cfg.irf_integrate_per_column,
        spatial_bin=cfg.spatial_bin,
        timing_offset_bins=cfg.timing_offset_bins,
        background_scale_mode=cfg.background_scale_mode,
        background_multiplier=cfg.background_multiplier,
        background_auto_fallback=cfg.background_auto_fallback,
        background_counts=cfg.background_counts,
        global_iterations=cfg.global_iterations,
        max_global_pixels=cfg.max_global_pixels,
        initial_taus_ns=cfg.initial_taus_ns,
        dll_sha256=result.dll_sha256,
        training_images=training_images,
    )
    _notify(progress, 1.0, "Training complete")
    return model


def analyze_dataset(
    dataset: FLIMDataset,
    settings: AnalysisSettings,
    irf: FLIMDataset | np.ndarray | None = None,
    background: FLIMDataset | np.ndarray | None = None,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    """Run the full analysis pipeline used by the desktop interface."""

    cfg = settings.normalized(dataset.time_bins)
    _notify(progress, 0.03, "Preparing decay cube")
    cube = np.asarray(dataset.decay, dtype=np.float32)
    cube = np.where(np.isfinite(cube), cube, 0.0)
    bg_data = background.decay if isinstance(background, FLIMDataset) else background
    irf_data = irf.decay if isinstance(irf, FLIMDataset) else irf
    if cfg.irf_source == "synthetic_gaussian":
        # A synthetic IRF is generated further down, once the decay cube
        # is fully preprocessed (so it reflects this run's own timing
        # exactly rather than whatever a loaded file happens to be) --
        # ignore any loaded/mismatched file entirely rather than letting
        # its now-irrelevant shape/timing-bin checks run against it.
        irf_data = None

    # The saved LabVIEW workflow subtracts its scaled background histogram
    # before any temporal or spatial transformation.  This order matters:
    # shifting first would align the sample but not the independently acquired
    # background, and clipping is non-linear with respect to later operations.
    background_validation = _validate_background_compatibility(dataset, background)
    background_multiplier, background_scale_details = resolve_background_multiplier(
        dataset, background, cfg
    )
    prepared_background = _prepare_background(bg_data, cube)
    background_spatial_mode = _background_spatial_mode(bg_data, cube)
    corrected_unclipped = (
        cube - background_multiplier * prepared_background - cfg.background_counts
    )
    clipped_bins = int(np.count_nonzero(corrected_unclipped < 0))
    total_source_bins = int(corrected_unclipped.size)
    if cfg.background_clip_negative:
        cube = np.clip(corrected_unclipped, 0.0, None).astype(np.float32, copy=False)
    else:
        cube = np.asarray(corrected_unclipped, dtype=np.float32)
    prepared_shape = (
        [int(value) for value in prepared_background.shape]
        if isinstance(prepared_background, np.ndarray)
        else []
    )
    background_correction_metadata: dict[str, object] = {
        "loaded": background is not None,
        **background_scale_details,
        **background_validation,
        "applied_multiplier": float(background_multiplier),
        "sample_shape_yxt": [int(value) for value in dataset.decay.shape],
        "background_shape": (
            [int(value) for value in np.asarray(bg_data).shape] if bg_data is not None else None
        ),
        "prepared_background_shape": prepared_shape,
        "spatial_handling": background_spatial_mode,
        "scalar_counts_per_bin": float(cfg.background_counts),
        "negative_bins_clipped": clipped_bins if cfg.background_clip_negative else 0,
        "negative_bins_kept": 0 if cfg.background_clip_negative else clipped_bins,
        "negative_bin_fraction": clipped_bins / total_source_bins if total_source_bins else 0.0,
        "clip_negative": bool(cfg.background_clip_negative),
        "formula": (
            "clip(sample - applied_multiplier * background - scalar_counts_per_bin, 0, +inf)"
            if cfg.background_clip_negative
            else "sample - applied_multiplier * background - scalar_counts_per_bin (signed, not clipped)"
        ),
        "row_interpolated_separately": False,
        "preprocessing_order": (
            "background subtraction and zero clipping -> timing shift -> source-row interpolation -> "
            "automatic extreme-pixel interpolation -> smoothing -> spatial binning -> phasor/fitting"
        ),
        "source_path": (
            str(background.source_path)
            if isinstance(background, FLIMDataset) and background.source_path
            else None
        ),
    }
    del corrected_unclipped
    cube = shift_time_axis(cube, cfg.timing_offset_bins)

    bin_width = spatial_bin_width(cfg.spatial_bin)
    interpolation_metadata: dict[str, object] = {
        "enabled": bool(cfg.interpolate_rows),
        "applied": False,
    }
    interpolation_diagnostics: dict[str, np.ndarray] = {}
    if cfg.interpolate_rows:
        row_map = RowInterpolationMap(
            rows=cfg.interpolation_rows,
            radius=cfg.interpolation_radius,
            coordinate_mode=cfg.interpolation_coordinate_mode,
            name=cfg.interpolation_map_name or "Defined detector-row map",
            source_path=cfg.interpolation_map_source,
            sha256=cfg.interpolation_map_sha256,
            expected_source_height=(
                cfg.interpolation_expected_source_height
                if cfg.interpolation_expected_source_height > 0
                else None
            ),
        ).normalized()
        # Detector-row correction remains at source resolution, but follows
        # background subtraction so all subsequent analysis sees corrected data.
        cube, row_details = interpolate_mapped_rows(cube, row_map)
        target_rows = tuple(int(value) for value in row_details["canonical_target_rows_y"])
        source_row_mask, result_coverage = interpolation_coverage(
            dataset.y_size,
            dataset.x_size,
            target_rows,
            # The result stays at source resolution -- rolling spatial
            # binning no longer shrinks it -- so this coverage map is
            # computed 1:1 with the source, not reduced by the bin width.
            1,
        )
        interpolation_diagnostics = {
            "interpolated_source_row_mask": source_row_mask,
            "interpolated_result_fraction": result_coverage,
        }
        # The legacy background flag remains true for a per-pixel background to
        # indicate that its spatial structure was incorporated at source
        # resolution.  The background itself is not interpolated separately;
        # see background_correction.row_interpolated_separately.
        reference_flags = {
            "background": bool(
                bg_data is not None
                and np.asarray(bg_data).ndim == 3
                and np.asarray(bg_data).shape == dataset.decay.shape
            ),
            "irf": False,
        }
        if irf_data is not None:
            irf_array = np.asarray(irf_data)
            if irf_array.ndim == 3 and irf_array.shape == dataset.decay.shape:
                irf_data, _ = interpolate_mapped_rows(irf_array, row_map)
                reference_flags["irf"] = True
        interpolation_metadata = {
            "enabled": True,
            "applied": True,
            "map": row_map.to_dict(),
            **row_details,
            "reference_correction": reference_flags,
            "preprocessing_order": (
                "background subtraction -> timing shift -> source-row interpolation -> automatic "
                "extreme-pixel interpolation -> smoothing -> spatial binning"
            ),
            "statistical_note": (
                "Interpolated traces are synthetic floating-point estimates and are not independent "
                "Poisson observations."
            ),
        }

    column_interpolation_metadata: dict[str, object] = {
        "enabled": bool(cfg.interpolate_columns),
        "applied": False,
    }
    if cfg.interpolate_columns:
        cube, column_details = interpolate_mapped_columns(
            cube, cfg.interpolation_columns, cfg.column_interpolation_radius
        )
        target_columns = tuple(
            int(value) for value in column_details["canonical_target_columns_x"]
        )
        source_column_mask, column_result_coverage = column_interpolation_coverage(
            dataset.y_size,
            dataset.x_size,
            target_columns,
            1,  # see the matching comment on the row-interpolation coverage call above
        )
        interpolation_diagnostics.update(
            {
                "interpolated_source_column_mask": source_column_mask,
                "interpolated_column_result_fraction": column_result_coverage,
            }
        )
        reference_irf_corrected = False
        if irf_data is not None:
            irf_array = np.asarray(irf_data)
            if irf_array.ndim == 3 and irf_array.shape == dataset.decay.shape:
                irf_data, _ = interpolate_mapped_columns(
                    irf_array, cfg.interpolation_columns, cfg.column_interpolation_radius
                )
                reference_irf_corrected = True
        column_interpolation_metadata = {
            "enabled": True,
            "applied": True,
            **column_details,
            "reference_correction": {
                "background": bool(
                    bg_data is not None
                    and np.asarray(bg_data).ndim == 3
                    and np.asarray(bg_data).shape == dataset.decay.shape
                ),
                "irf": reference_irf_corrected,
            },
            "preprocessing_order": (
                "background subtraction -> timing shift -> detector-row interpolation -> "
                "detector-column interpolation -> automatic extreme-pixel interpolation -> "
                "smoothing -> spatial binning"
            ),
        }

    _notify(progress, 0.13, "Applying spatial preprocessing")
    if cfg.interpolate_outliers:
        cube = interpolate_bad_pixels(cube, cfg.outlier_z)
    cube = smooth_spatial(cube, cfg.smoothing_passes)
    # Rolling spatial binning makes neighbouring pixels' values overlap (each
    # pixel's value already includes counts from pixels near it), so summing
    # more than one of them together -- as any multi-pixel ROI/whole-image
    # aggregate decay does -- would double- and triple-count shared photons.
    # Keep this pre-bin snapshot around specifically for those region-sum
    # aggregates (see AnalysisResult.additive_decay / roi.py); the per-pixel
    # analysis below still uses the binned cube as intended.
    additive_cube = cube.copy()
    cube = rolling_spatial_bin(cube, cfg.spatial_bin)

    if irf_data is not None:
        irf_data = np.asarray(irf_data, dtype=np.float32)
        irf_data = shift_time_axis(irf_data, cfg.timing_offset_bins)
        if irf_data.ndim == 3 and irf_data.shape[:2] == dataset.decay.shape[:2]:
            irf_data = rolling_spatial_bin(smooth_spatial(irf_data, cfg.smoothing_passes), cfg.spatial_bin)
            if cfg.irf_integrate_per_column:
                # Applied here (once, for this image's genuine per-pixel
                # IRF cube) rather than only inside the FLIMLib-native
                # branch below: phasor's own per-pixel IRF correction
                # (phasor_lifetimes -> _reference_rows) reads this same
                # `irf_data`, independently of fit_irf_mode (phasor
                # decides "per-pixel or not" purely from the array's own
                # shape) -- so doing it here covers phasor, iterative
                # reconvolution, global fit, and apply-model uniformly
                # from one place, instead of duplicating the call in
                # each consumer.
                irf_data = integrate_irf_per_column(irf_data)

    intensity = cube.sum(axis=-1, dtype=np.float64)
    phasor_mask = intensity >= cfg.phasor_min_counts
    fit_mask = intensity >= cfg.fit_min_counts
    active_min_counts = (
        cfg.phasor_min_counts
        if cfg.method in {"phasor_phase", "phasor_modulation"}
        else cfg.fit_min_counts
    )
    valid_mask = phasor_mask if cfg.method in {"phasor_phase", "phasor_modulation"} else fit_mask
    time_ns = (np.arange(cube.shape[-1], dtype=np.float64) + 0.5) * cfg.dt_ns
    synthetic_irf_metadata: dict | None = None
    if cfg.irf_source == "synthetic_gaussian":
        irf_data, synthetic_irf_metadata = estimate_synthetic_gaussian_irf(
            cube, time_ns, cfg.synthetic_irf_fwhm_ns
        )
    _notify(progress, 0.32, "Calculating phasors")
    g, s, tau_phase, tau_mod = phasor_lifetimes(
        cube, cfg.dt_ns, cfg.rep_rate_mhz, cfg.phasor_min_counts, irf_data
    )
    _notify(progress, 0.57, "Fitting fluorescence decays")
    tau_fit, chi, _, fit_start, fit_end = mono_exponential_fit(
        cube, time_ns, cfg.fit_min_counts, cfg.fit_start_bin, cfg.fit_end_bin
    )
    tau_moment = moment_lifetime(cube, time_ns, cfg.fit_min_counts, irf_data)

    native_result: NativeFitResult | None = None
    native_input_bins: int | None = None
    threshold_signal_map = intensity.copy()
    threshold_signal_basis = "all processed TCSPC bins"
    trained_model: TrainedGlobalModel | None = None
    if cfg.method == "flimlib_apply_global":
        if not cfg.global_model_path:
            raise ValueError(
                "Applying a trained global model requires a model file. "
                "Load one in the sidebar's Global model section first."
            )
        trained_model = load_global_model(cfg.global_model_path)
        if trained_model.fit_distribution_model != 0:
            kind = "multi-stretched-exponential" if trained_model.fit_distribution_model == 2 else "stretched-exponential"
            raise ValueError(
                f"'{trained_model.name}' is a {kind} (distribution) model -- "
                "applying one isn't supported yet, only multi-exponential trained models. "
                "flim_apply_global_model_v1 always fits a plain multi-exponential; treating "
                "this model's stored (tau, h) pairs as discrete lifetimes would silently drop "
                "every h and fit the wrong model entirely, so this is refused rather than done "
                "wrong."
            )
        # The fixed taus are only meaningful under the exact fitting
        # conditions they were trained with -- reuse them here rather than
        # whatever happens to be in the sidebar, so "apply" reproduces
        # "train" exactly. Spatial bin and timing offset stay user-controlled
        # since they are per-acquisition preprocessing choices, not part of
        # what the taus mean.
        cfg.model_order = trained_model.model_order
        cfg.fit_background_mode = trained_model.background_mode
        cfg.fit_background_value = trained_model.background_value
        cfg.fit_noise_model = trained_model.noise_model
        cfg.fit_min_counts = trained_model.fit_min_counts
        cfg.fit_start_bin = trained_model.fit_start_bin
        cfg.fit_end_bin = trained_model.fit_end_bin
        cfg.native_fit_bins = trained_model.native_fit_bins
        cfg.native_bin_width_ns = trained_model.native_bin_width_ns
        cfg.native_irf_baseline_mode = trained_model.native_irf_baseline_mode
        cfg.fit_irf_mode = trained_model.irf_mode
        cfg.irf_integrate_per_column = trained_model.irf_integrate_per_column
    if cfg.method in {"flimlib_reconvolution", "flimlib_global", "flimlib_apply_global"}:
        if irf_data is None:
            raise ValueError("FLIMLib reconvolution requires an IRF dataset. Load an IRF and run again.")
        _notify(progress, 0.60, "Running native FLIMLib reconvolution")
        native_input_bins = (
            cube.shape[-1] if cfg.native_fit_bins < 0 else min(cfg.native_fit_bins, cube.shape[-1])
        )
        native_cube = np.ascontiguousarray(cube[..., :native_input_bins])
        native_irf = np.asarray(irf_data, dtype=np.float64)
        if cfg.native_irf_baseline_mode == "median":
            irf_baseline = np.median(native_irf, axis=-1, keepdims=True)
        elif cfg.native_irf_baseline_mode == "tail_median":
            tail_bins = min(8, native_irf.shape[-1])
            irf_baseline = np.median(native_irf[..., -tail_bins:], axis=-1, keepdims=True)
        else:
            irf_baseline = np.zeros(native_irf.shape[:-1] + (1,), dtype=np.float64)
        native_irf = np.clip(native_irf - irf_baseline, 0.0, None)
        native_irf = np.ascontiguousarray(native_irf[..., :native_input_bins])
        # (irf_integrate_per_column, if enabled, was already applied to
        # irf_data above -- once, for every consumer including phasor --
        # so native_irf inherits it here with no separate call needed.)
        # A single representative trace for reconstructing a display-only
        # reconvolution curve later (see ``analyse_roi``). Per-pixel IRF fits
        # use this mean trace as an approximation for that curve only; the
        # native per-pixel fit itself still uses the true per-pixel IRF.
        native_irf_display_trace = (
            np.nanmean(native_irf, axis=tuple(range(native_irf.ndim - 1)))
            if native_irf.ndim > 1
            else np.asarray(native_irf, dtype=np.float64).reshape(-1)
        )
        native_mask = native_cube.sum(axis=-1, dtype=np.float64) >= cfg.fit_min_counts
        threshold_signal_map = native_cube.sum(axis=-1, dtype=np.float64)
        threshold_signal_basis = f"leading {native_input_bins} native-fit TCSPC bins"
        valid_mask = native_mask
        active_min_counts = cfg.fit_min_counts
        native_options = {
            "bin_width_ns": cfg.native_bin_width_ns,
            "model_order": cfg.model_order,
            "background_mode": cfg.fit_background_mode,
            "background_value": cfg.fit_background_value,
            "noise_model": cfg.fit_noise_model,
            "min_photons": cfg.fit_min_counts,
            "irf_mode": cfg.fit_irf_mode,
        }
        native_start = (
            0 if cfg.fit_start_bin < 0 else min(cfg.fit_start_bin, native_input_bins - 2) + 1
        )
        native_stop = (
            0 if cfg.fit_end_bin < 0 else max(native_start + 1, min(cfg.fit_end_bin, native_input_bins))
        )
        if cfg.method == "flimlib_global":
            native_result = fit_in_subprocess(
                "global",
                native_cube,
                native_irf,
                dll_path=cfg.native_dll_path or None,
                mask=native_mask,
                fit_start_bin=native_start,
                fit_stop_bin=native_stop,
                max_global_pixels=cfg.max_global_pixels,
                global_iterations=cfg.global_iterations,
                initial_taus_ns=cfg.initial_taus_ns,
                # Only "global" accepts this -- fit_type isn't in
                # native_options since flimlib_reconvolution/apply_global
                # don't take it (reconvolution has no shared search;
                # apply always uses whatever the loaded model was trained
                # with, guarded above).
                fit_type=cfg.fit_distribution_model,
                **native_options,
            )
        elif cfg.method == "flimlib_apply_global":
            assert trained_model is not None
            stacked_cube = native_cube[None, ...]
            stacked_irf = native_irf[None, ...] if native_irf.ndim == 3 else native_irf
            stacked_mask = np.ones((1,) + native_cube.shape[:2], dtype=np.float64)
            native_result = apply_global_in_subprocess(
                stacked_cube,
                stacked_irf,
                stacked_mask,
                np.asarray(trained_model.global_taus_ns, dtype=np.float64),
                dll_path=cfg.native_dll_path or None,
                fit_start_bin=native_start,
                fit_stop_bin=native_stop,
                **native_options,
            )
        else:
            # flimlib_reconvolution: the only native method where every pixel
            # is fitted fully independently (no shared/joint state across
            # pixels, unlike flimlib_global/flimlib_apply_global above), so
            # it's the only one safe to split into concurrent row-band
            # chunks -- see fit_pixels_parallel_in_subprocess's own
            # docstring. native_fit_workers=1 (the default) is identical to
            # the single fit_in_subprocess call this replaced.
            native_result = fit_pixels_parallel_in_subprocess(
                native_cube,
                native_irf,
                dll_path=cfg.native_dll_path or None,
                worker_count=cfg.native_fit_workers,
                **native_options,
            )

    method_maps = {
        "phasor_phase": tau_phase,
        "phasor_modulation": tau_mod,
        "mono_exponential": tau_fit,
        "moment": tau_moment,
        "flimlib_reconvolution": native_result.tau_mean_ns if native_result is not None else tau_fit,
        "flimlib_global": native_result.tau_mean_ns if native_result is not None else tau_fit,
        "flimlib_apply_global": native_result.tau_mean_ns if native_result is not None else tau_fit,
    }
    if cfg.method not in method_maps:
        raise ValueError(f"Unknown lifetime method: {cfg.method}")
    lifetime = np.asarray(method_maps[cfg.method], dtype=np.float64).copy()
    native_success = native_result.fit_status == 1 if native_result is not None else np.ones_like(valid_mask)
    # Lifetime display limits belong to the LUT/histogram, not the numerical
    # estimator.  The supplied LabVIEW VI's "Remove out of range" input is
    # wired only to image rendering.  Retain every scientifically valid finite
    # lifetime here and apply the display range later in ``lifetime_image``.
    valid_lifetime = valid_mask & native_success & np.isfinite(lifetime)
    lifetime[~valid_lifetime] = np.nan

    _notify(progress, 0.76, "Calculating representative and global fits")
    flattened = cube.reshape(-1, cube.shape[-1])
    eligible = flattened[valid_mask.ravel()]
    if eligible.shape[0] > cfg.max_global_pixels:
        indices = np.linspace(0, eligible.shape[0] - 1, cfg.max_global_pixels, dtype=int)
        eligible = eligible[indices]
    mean_decay = eligible.mean(axis=0) if eligible.size else cube.mean(axis=(0, 1))
    mean_fit, global_tau = fit_single_decay(mean_decay, time_ns, fit_start, fit_end)
    if native_result is not None:
        chi = native_result.chi_square
        if native_result.global_taus_ns is not None and native_result.global_taus_ns.size:
            global_tau = float(native_result.global_taus_ns[0])

    metadata = dict(dataset.metadata)
    lifetime_lut_metadata = {
        **lut_provenance(cfg.lifetime_lut_path or None),
        "range_ns": [float(cfg.lifetime_min_ns), float(cfg.lifetime_max_ns)],
        "scale_mode": str(cfg.lifetime_scale_mode),
        "manual_range_ns": [float(cfg.lifetime_min_ns), float(cfg.lifetime_max_ns)],
        "automatic_range_algorithm": "finite pixel lifetime 1st–99th percentiles",
        "inverse": bool(cfg.lifetime_lut_inverse),
        "remove_out_of_range": bool(cfg.lifetime_lut_remove_out_of_range),
        "display_only": True,
        "numerical_lifetime_effect": (
            "None. The LUT changes display colours only; numerical lifetime values are unchanged."
        ),
    }
    # Package-internal absolute paths are useful while running but are not
    # portable provenance.  The bundled LUT is identified completely by its
    # stable name and SHA-256; omit the developer-machine path from exports.
    if lifetime_lut_metadata.get("source") == "bundled_labview_default":
        lifetime_lut_metadata["path"] = None
    metadata.update(
        {
            "source_path": str(dataset.source_path) if dataset.source_path else None,
            "spatial_transform": {
                "source_shape_yx": [int(dataset.y_size), int(dataset.x_size)],
                "result_shape_yx": [int(cube.shape[0]), int(cube.shape[1])],
                "spatial_bin_level": int(cfg.spatial_bin),
                "spatial_bin_method": "rolling",
                # Rolling spatial binning sums a (2*level+1) square window
                # centred on each pixel but keeps the image at its original
                # size -- there's no processed:source pixel-index scale to
                # apply any more, so this stays 1 regardless of bin level.
                # ``spatial_bin_window_width`` is the actual window size.
                "spatial_bin": 1,
                "spatial_bin_width": 1,
                "spatial_bin_window_width": int(bin_width),
                "analyzed_source_bounds_xyxy": [0, 0, int(cube.shape[1]), int(cube.shape[0])],
                "dropped_right": 0,
                "dropped_bottom": 0,
                "coordinate_convention": "zero-based xyxy; upper bounds exclusive",
            },
            "analysis_fit_window": [int(fit_start), int(fit_end)],
            "time_bin_contract": {
                "acquisition_bins": int(dataset.time_bins),
                "processed_bins": int(cube.shape[-1]),
                "native_input_bins": native_input_bins,
                "native_uses_all_bins": (
                    None if native_input_bins is None else bool(native_input_bins == cube.shape[-1])
                ),
                "configured_native_fit_bins": (
                    "all" if cfg.native_fit_bins < 0 else int(cfg.native_fit_bins)
                ),
                "acquisition_bin_width_ns": float(cfg.dt_ns),
                "native_fit_bin_width_ns": (
                    float(cfg.native_bin_width_ns) if native_input_bins is not None else None
                ),
                "overview_histogram_intervals": int(cfg.histogram_bins),
                "detailed_histogram_intervals": int(cfg.detailed_histogram_bins),
                "analysis_fit_window_start_inclusive": int(fit_start),
                "analysis_fit_window_stop_exclusive": int(fit_end),
                "fit_window_zero_based_half_open": [int(fit_start), int(fit_end)],
                "note": (
                    "Acquisition/native bins are TCSPC time samples; histogram intervals are "
                    "display bins and do not alter the decay cube."
                ),
            },
            "count_thresholds": {
                "legacy_min_photons": float(cfg.min_photons),
                "phasor_min_counts": float(cfg.phasor_min_counts),
                "fit_min_counts": float(cfg.fit_min_counts),
                "active_min_counts": float(active_min_counts),
                "active_estimator_family": (
                    "phasor" if cfg.method in {"phasor_phase", "phasor_modulation"} else "fit"
                ),
                "preview_signal_basis": threshold_signal_basis,
            },
            "irf_loaded": irf_data is not None,
            "irf_source": cfg.irf_source,
            "synthetic_irf": synthetic_irf_metadata,
            "background_loaded": bg_data is not None,
            "background_correction": background_correction_metadata,
            "lifetime_lut": lifetime_lut_metadata,
            "composite_display": {
                "algorithm": "LabVIEW HSV intensity/lifetime composite",
                "lifetime_hue_range": [0.0, 0.7],
                "saturation": 1.0,
                "intensity_scaling": "linear actual minimum/maximum (Auto Scale enabled)",
                "uses_external_lifetime_lut": False,
            },
            "lifetime_validity": {
                "display_range_changes_numerical_values": False,
                "validity_basis": (
                    "method-specific minimum processed counts, estimator/native status, and finite lifetime"
                ),
                "finite_valid_pixels": int(np.count_nonzero(valid_lifetime)),
                "within_display_range_pixels": int(
                    np.count_nonzero(
                        valid_lifetime
                        & (lifetime >= cfg.lifetime_min_ns)
                        & (lifetime <= cfg.lifetime_max_ns)
                    )
                ),
            },
            "row_interpolation": interpolation_metadata,
            "column_interpolation": column_interpolation_metadata,
        }
    )
    fit_diagnostics: dict[str, np.ndarray] = dict(interpolation_diagnostics)
    fit_diagnostics["threshold_signal_map"] = np.asarray(threshold_signal_map, dtype=np.float64)
    if irf_data is not None:
        # A single representative trace (mean over any spatial dimensions)
        # of whatever IRF was actually used for this run -- loaded file or
        # synthetic Gaussian alike -- for the "IRF & decay" display tab.
        # Independent of native_irf_trace below, which only exists for the
        # native-fit methods and serves a different purpose there.
        irf_for_display = np.asarray(irf_data, dtype=np.float64)
        fit_diagnostics["irf_trace"] = (
            irf_for_display if irf_for_display.ndim == 1
            else np.nanmean(irf_for_display, axis=tuple(range(irf_for_display.ndim - 1)))
        )
    if native_result is not None:
        status_values, status_counts = np.unique(native_result.fit_status, return_counts=True)
        metadata["native_fit"] = {
            "backend": "FLIMLib LabVIEW DLL",
            "mode": native_result.metadata["mode"],
            "model_order": cfg.model_order,
            "fit_distribution_model": cfg.fit_distribution_model,
            "noise_model": cfg.fit_noise_model,
            "background_mode": cfg.fit_background_mode,
            "irf_mode": cfg.fit_irf_mode,
            "irf_integrate_per_column": cfg.irf_integrate_per_column,
            "native_input_bins": int(native_input_bins) if native_input_bins is not None else None,
            "native_bin_width_ns": float(cfg.native_bin_width_ns),
            "irf_baseline_mode": cfg.native_irf_baseline_mode,
            "dll_path": str(native_result.dll_path),
            "dll_sha256": native_result.dll_sha256,
            "wrapper_code": native_result.wrapper_code,
            "status_counts": {
                str(int(status)): int(count) for status, count in zip(status_values, status_counts)
            },
            "global_taus_ns": (
                native_result.global_taus_ns.tolist() if native_result.global_taus_ns is not None else None
            ),
            "global_stats": (
                native_result.global_stats.tolist() if native_result.global_stats is not None else None
            ),
        }
        if trained_model is not None:
            metadata["native_fit"]["applied_global_model"] = {
                "name": trained_model.name,
                "path": str(cfg.global_model_path),
                "trained_at": trained_model.trained_at,
                "training_images": len(trained_model.training_images),
                "training_selected_pixels": trained_model.training_selected_pixels,
                "dll_sha256": trained_model.dll_sha256,
            }
        fit_diagnostics.update(native_result.diagnostics())
        fit_diagnostics["native_irf_trace"] = np.asarray(native_irf_display_trace, dtype=np.float64)
    _notify(progress, 1.0, "Analysis complete")
    return AnalysisResult(
        dataset_name=dataset.name,
        intensity=intensity,
        lifetime_ns=lifetime,
        valid_mask=valid_lifetime,
        g=g,
        s=s,
        tau_phase_ns=tau_phase,
        tau_modulation_ns=tau_mod,
        chi_square=chi,
        mean_decay=mean_decay,
        mean_fit=mean_fit,
        time_ns=time_ns,
        global_tau_ns=global_tau,
        settings=cfg,
        processed_decay=cube,
        additive_decay=additive_cube,
        metadata=metadata,
        fit_diagnostics=fit_diagnostics,
    )
