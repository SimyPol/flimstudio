from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.demo import create_demo_dataset, create_demo_irf
from flim_app.io import load_ics
from flim_app.models import AnalysisSettings, FLIMDataset
from flim_app.processing import (
    TrainingImageInput,
    analyze_dataset,
    block_bin,
    estimate_synthetic_gaussian_irf,
    mono_exponential_fit,
    phasor_informed_seed,
    phasor_lifetimes,
    spatial_bin_width,
    train_global_model,
)
from flim_app.rendering import export_result, irf_decay_overlay_plot


def assert_close(actual: float, expected: float, tolerance: float, message: str) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(f"{message}: expected {expected} ± {tolerance}, got {actual}")


def make_decay(tau_ns: float, size: int = 8, nt: int = 64, dt_ns: float = 0.1953125) -> np.ndarray:
    t = (np.arange(nt) + 0.5) * dt_ns
    decay = 10_000 * np.exp(-t / tau_ns) + 2.0
    return np.broadcast_to(decay, (size, size, nt)).astype(np.float32).copy()


def test_block_binning() -> None:
    cube = np.ones((8, 6, 4), dtype=np.float32)
    binned = block_bin(cube, 2)
    assert binned.shape == (4, 3, 4)
    assert np.allclose(binned, 4.0)
    assert [spatial_bin_width(level) for level in range(3)] == [1, 3, 5]


def test_phasor_recovers_mono_lifetime() -> None:
    tau = 2.4
    cube = make_decay(tau)
    clean = np.clip(cube - 2.0, 0, None)
    _, _, phase, modulation = phasor_lifetimes(clean, 0.1953125, 80.0, 10.0)
    # A finite acquisition window truncates the exponential slightly.
    assert_close(float(np.nanmedian(phase)), tau, 0.35, "phase lifetime")
    assert_close(float(np.nanmedian(modulation)), tau, 0.45, "modulation lifetime")


def test_mono_fit_recovers_lifetime() -> None:
    tau = 3.1
    cube = make_decay(tau)
    t = (np.arange(cube.shape[-1]) + 0.5) * 0.1953125
    fitted_tau, chi, _, _, _ = mono_exponential_fit(cube, t, 10.0, 0, -1)
    assert_close(float(np.nanmedian(fitted_tau)), tau, 0.08, "mono exponential lifetime")
    assert np.isfinite(chi).all()


def _ics_header(version: str, embedded: bool) -> bytes:
    return (
        f"\t\nics_version\t{version}\n"
        "filename\ttest\n"
        "layout\tparameters\t4\n"
        "layout\torder\tbits\tt\tx\ty\n"
        "layout\tsizes\t16\t4\t3\t2\n"
        "representation\tformat\tinteger\n"
        "representation\tsign\tunsigned\n"
        "representation\tcompression\tuncompressed\n"
        "representation\tbyte_order\t1\t2\n"
        "parameter\tscale\t1\t0.5\t1\t1\n"
        "parameter\tunits\trelative\tns\tum\tum\n"
        "history\timage\textents\t2.0\t3\t2\n"
        "history\timage\tunits\tns\tum\tum\n"
        "history\tlaser\trep\trate\t80000000\n"
        "end\t\n"
    ).encode("ascii")


def test_ics_v2_and_v1_readers() -> None:
    values = np.arange(24, dtype="<u2")
    expected = values.reshape((4, 3, 2), order="F").transpose(2, 1, 0)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        v2 = folder / "embedded.ics"
        v2.write_bytes(_ics_header("2.0", True) + values.tobytes())
        loaded = load_ics(v2)
        assert loaded.decay.shape == (2, 3, 4)
        assert np.array_equal(loaded.decay, expected)
        assert_close(loaded.dt_ns, 0.5, 1e-9, "ICS dt")

        v1 = folder / "paired.ics"
        v1.write_bytes(_ics_header("1.0", False))
        v1.with_suffix(".ids").write_bytes(values.tobytes())
        paired = load_ics(v1)
        assert np.array_equal(paired.decay, expected)


def test_full_demo_and_export() -> None:
    dataset = create_demo_dataset(size=48, time_bins=48, seed=9)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    settings = AnalysisSettings(
        method="phasor_phase",
        dt_ns=dataset.dt_ns,
        rep_rate_mhz=dataset.rep_rate_mhz,
        min_photons=20,
        lifetime_min_ns=0,
        lifetime_max_ns=8,
        spatial_bin=0,
    )
    result = analyze_dataset(dataset, settings, irf)
    assert result.intensity.shape == (48, 48)
    assert result.valid_lifetimes.size > 20
    with tempfile.TemporaryDirectory() as directory:
        files = export_result(result, directory, "test")
        assert len(files) == 10
        assert all(path.exists() and path.stat().st_size > 0 for path in files)


def test_native_analysis_pipeline() -> None:
    dataset = create_demo_dataset(size=12, time_bins=48, seed=4)
    irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
    settings = AnalysisSettings(
        method="flimlib_reconvolution",
        dt_ns=dataset.dt_ns,
        rep_rate_mhz=dataset.rep_rate_mhz,
        min_photons=20,
        lifetime_min_ns=0,
        lifetime_max_ns=8,
        model_order=1,
        fit_background_mode=0,
        fit_noise_model=0,
        fit_irf_mode=0,
    )
    result = analyze_dataset(dataset, settings, irf)
    assert result.lifetime_ns.shape == (12, 12)
    assert result.valid_lifetimes.size > 10
    assert "tau_components_ns" in result.fit_diagnostics
    assert "fit_status" in result.fit_diagnostics
    # This pin intentionally tracks native/flimfit_labview.dll's identity --
    # update it (with a comment here noting why) whenever that DLL is
    # legitimately rebuilt. Currently the v0.17.0 build that fixes multi-
    # start silently overriding an explicit, correct `initial_taus_ns`
    # seed for FIT_GLOBAL_MULTI_STRETCHEDEXP: `has_explicit_shared_seed()`
    # now makes multi-start run only when the caller did NOT supply a
    # fully specified shared-parameter seed, so an explicit seed always
    # gets exactly one search that respects it as the starting point --
    # FIT_GLOBAL_MULTIEXP/FIT_GLOBAL_STRETCHEDEXP (what this specific
    # test exercises) are deliberately untouched, still exactly one
    # start with the identical seeding formula as always; this pin bump
    # is purely because it's the same DLL binary as everything else in
    # this codebase. See native_backend.py's _PATCHED_DLL_SHA256 comment
    # history and RECONSTRUCTION_NOTES.md for the full account of why
    # multi-start was added (a confirmed real case where the search's
    # own converged answer had a lower objective than the true
    # generating parameters -- genuine model non-identifiability for
    # N>=2 independently-distributed components, not a bug, but multi-
    # start at least removes the search itself as a further source of
    # unreliability for the *default*, unseeded case) and then why it
    # needed this follow-up fix (an explicit seed must never be
    # silently discarded in favor of a statistically "better" but less
    # plausible answer the user didn't ask the search to consider).
    # Rebuild #14 reverts a second same-session attempted fix (a ridge/
    # Tikhonov term in the per-pixel linear solve) that cost measurable
    # N=2 precision drift for no practical N=3 benefit.
    # FIT_GLOBAL_MULTIEXP/FIT_GLOBAL_STRETCHEDEXP, what this specific
    # test exercises, were never touched by either the attempt or the
    # revert; pin bump purely because it's the same DLL binary. See
    # native_backend.py's _PATCHED_DLL_SHA256 comment and
    # RECONSTRUCTION_NOTES.md for the full account.
    assert result.metadata["native_fit"]["dll_sha256"] == (
        "efe017536e9ffaa5b53061966f4a771e5eb6c13bcd5c3a9936cfb18f40d1372c"
    )


def test_synthetic_gaussian_irf_recovers_known_fwhm() -> None:
    """The rise-time-to-FWHM conversion must recover a known Gaussian width.

    Builds a clean error-function rising edge (the standard step response
    of a Gaussian impulse) with a known sigma, so the pre-peak portion of
    the signal is *exactly* the shape estimate_synthetic_gaussian_irf
    assumes it is -- no fluorescence-decay distortion muddying the check.
    A slow exponential decay is added only *after* the peak (via a
    zero-clipped exponent) so argmax finds a clean peak, without touching
    the rise itself.
    """

    from math import erf

    nt = 400
    dt_ns = 0.02
    time_ns = (np.arange(nt) + 0.5) * dt_ns
    sigma_ns = 0.12
    true_fwhm_ns = sigma_ns * 2.3548200450309493
    center_ns = 1.0
    peak_ns = center_ns + 6 * sigma_ns
    tau_decay_ns = 2.0
    baseline = 5.0
    amplitude = 10_000.0

    rise = 0.5 * (1.0 + np.array([erf((t - center_ns) / (sigma_ns * np.sqrt(2.0))) for t in time_ns]))
    decay = np.exp(-np.clip(time_ns - peak_ns, 0.0, None) / tau_decay_ns)
    trace = baseline + amplitude * rise * decay
    cube = np.broadcast_to(trace, (3, 3, nt)).astype(np.float32).copy()

    profile, meta = estimate_synthetic_gaussian_irf(cube, time_ns)
    assert meta["estimation_method"] == "rise_time_10_90"
    assert_close(meta["fwhm_ns"], true_fwhm_ns, 0.05 * true_fwhm_ns, "recovered FWHM")
    assert_close(meta["center_ns"], center_ns, 3 * dt_ns, "recovered centre")
    assert profile.shape == (nt,)
    assert np.isclose(float(profile.sum()), 1.0, atol=1e-4)  # unit-area

    # A manual override must be used verbatim, ignoring the data entirely.
    _, manual_meta = estimate_synthetic_gaussian_irf(cube, time_ns, fwhm_ns_override=0.5)
    assert manual_meta["estimation_method"] == "manual_override"
    assert manual_meta["fwhm_ns"] == 0.5


def test_analyze_dataset_with_synthetic_irf_and_no_loaded_irf() -> None:
    """IRF-corrected phasor analysis must run with no IRF file at all when
    a synthetic Gaussian IRF is requested -- this is the whole point of
    the feature: don't refuse to process data just because no measured
    IRF is available or a loaded one doesn't match this dataset's timing.
    """

    dataset = create_demo_dataset(size=20, time_bins=32, seed=13)
    settings = AnalysisSettings(
        method="phasor_phase",
        dt_ns=dataset.dt_ns,
        rep_rate_mhz=dataset.rep_rate_mhz,
        min_photons=10,
        irf_source="synthetic_gaussian",
    )
    result = analyze_dataset(dataset, settings, irf=None)  # no IRF loaded at all
    assert result.intensity.shape == (20, 20)
    assert result.valid_lifetimes.size > 0
    assert result.metadata["irf_loaded"] is True  # populated by the synthetic profile
    assert result.metadata["irf_source"] == "synthetic_gaussian"
    synthetic = result.metadata["synthetic_irf"]
    assert synthetic is not None and synthetic["fwhm_ns"] > 0
    assert result.fit_diagnostics["irf_trace"].shape == (dataset.time_bins,)

    overlay = irf_decay_overlay_plot(result)
    assert overlay.size == (760, 430)

    # The default ("file") source must be completely unaffected: with no
    # IRF loaded and no synthetic source requested, phasor analysis stays
    # uncorrected exactly as before this feature existed.
    default_settings = AnalysisSettings(
        method="phasor_phase", dt_ns=dataset.dt_ns, rep_rate_mhz=dataset.rep_rate_mhz, min_photons=10
    )
    default_result = analyze_dataset(dataset, default_settings, irf=None)
    assert default_result.metadata["irf_loaded"] is False
    assert default_result.metadata["synthetic_irf"] is None
    assert "irf_trace" not in default_result.fit_diagnostics
    assert irf_decay_overlay_plot(default_result).size == (760, 430)  # renders fine with no IRF too


def _stretched_mix_irf(time_bins: int, dt_ns: float) -> np.ndarray:
    """Unit-area kernel, for convolving synthetic decays -- NOT what a
    raw IRF *dataset* looks like (see _stretched_mix_irf_dataset_profile
    for that; the app's own preprocessing expects realistic count-scale
    data, not an already-normalized kernel)."""
    t = (np.arange(time_bins, dtype=np.float64) + 0.5) * dt_ns
    irf = np.exp(-0.5 * ((t - 2.0) / 0.09) ** 2)
    irf /= irf.sum()
    return irf


def _stretched_mix_irf_dataset_profile(time_bins: int, dt_ns: float) -> np.ndarray:
    """A realistic-magnitude (count-scale, with a small baseline) IRF
    trace, suitable for an actual FLIMDataset fed through the app's own
    preprocessing pipeline -- distinct from the unit-area convolution
    kernel above."""
    t = (np.arange(time_bins, dtype=np.float64) + 0.5) * dt_ns
    return np.exp(-0.5 * ((t - 2.0) / 0.09) ** 2) * 4000.0 + 2.0


def _make_two_component_mix(
    tau_a: float, h_a: float, tau_b: float, h_b: float,
    size: int, time_bins: int, dt_ns: float, seed: int, peak_counts: float, bg: float,
) -> np.ndarray:
    """Two independently-distributed stretched-exponential components,
    mixed in pixel-varying proportions (0.05-0.95 across the grid) -- the
    same "real global analysis needs genuinely varying mixing ratios"
    setup [[multi-stretched-global-fit-n2-unvalidated]] confirmed is
    what makes FIT_GLOBAL_MULTI_STRETCHEDEXP identifiable at all, not a
    spatially-uniform mixture."""
    rng = np.random.default_rng(seed)
    t = (np.arange(time_bins, dtype=np.float64) + 0.5) * dt_ns
    irf = _stretched_mix_irf(time_bins, dt_ns)
    shape_a = np.exp(-np.power(np.maximum(t, 1e-9) / tau_a, 1.0 / h_a))
    shape_b = np.exp(-np.power(np.maximum(t, 1e-9) / tau_b, 1.0 / h_b))
    fractions = np.linspace(0.05, 0.95, size * size).reshape(size, size)
    decay = np.zeros((size, size, time_bins))
    for r in range(size):
        for c in range(size):
            frac = fractions[r, c]
            mixed = frac * shape_a + (1.0 - frac) * shape_b
            convolved = np.convolve(mixed, irf)[:time_bins]
            decay[r, c, :] = peak_counts * convolved / convolved.max() + bg
    return rng.poisson(np.clip(decay, 0, 1e12)).astype(np.float64)


def test_phasor_informed_seed_recovers_two_component_mixture() -> None:
    """phasor_informed_seed derives an initial_taus_ns-shaped guess for
    FIT_GLOBAL_MULTI_STRETCHEDEXP directly from the data's own phasor
    structure, with no prior knowledge of the true components --
    [[multi-stretched-global-fit-n2-unvalidated]] has the full method
    and validation history. Checks the seed itself lands close to truth
    (the core claim this function makes), not just that it runs."""

    size, time_bins, dt_ns = 12, 200, 0.15
    tau_a, h_a, tau_b, h_b = 1.5, 1.8, 5.0, 3.0
    rep_rate_mhz = 20.0
    decay_cube = _make_two_component_mix(
        tau_a, h_a, tau_b, h_b, size, time_bins, dt_ns, seed=9, peak_counts=200_000.0, bg=3.0
    )
    irf_1d = _stretched_mix_irf(time_bins, dt_ns)

    seed = phasor_informed_seed(decay_cube, dt_ns, rep_rate_mhz, model_order=2, irf=irf_1d, min_photons=1000.0)
    assert seed is not None and len(seed) == 4
    (seed_tau0, seed_h0, seed_tau1, seed_h1) = seed
    # Component order isn't semantically meaningful -- sort by tau, same
    # as the true pair, before comparing. The raw seed is only meant to
    # land in the right neighborhood for the search to refine from (see
    # test_train_global_model_phasor_seed_end_to_end for the tighter,
    # practically-meaningful check on the actual trained result) -- a
    # generous tolerance here, not a precision claim about the seed
    # itself.
    assert seed_tau0 < seed_tau1
    assert_close(seed_tau0, tau_a, 0.6 * tau_a, "phasor-derived seed tau (fast component)")
    assert_close(seed_tau1, tau_b, 0.6 * tau_b, "phasor-derived seed tau (slow component)")
    assert 1.0 <= seed_h0 <= 4.5, f"seed h (fast component) out of plausible range: {seed_h0}"
    assert 1.0 <= seed_h1 <= 4.5, f"seed h (slow component) out of plausible range: {seed_h1}"


def test_train_global_model_phasor_seed_end_to_end() -> None:
    """The phasor_seed=True mode wired into train_global_model (a THIRD
    seeding mode alongside unseeded and manual -- see that function's
    own docstring) must actually improve the trained result on a
    realistic two-component mixture, with no initial_taus_ns supplied by
    the caller at all."""

    size, time_bins, dt_ns = 12, 200, 0.15
    tau_a, h_a, tau_b, h_b = 1.5, 1.8, 5.0, 3.0
    rep_rate_mhz = 20.0
    decay_cube = _make_two_component_mix(
        tau_a, h_a, tau_b, h_b, size, time_bins, dt_ns, seed=9, peak_counts=200_000.0, bg=3.0
    )
    irf_dataset_profile = _stretched_mix_irf_dataset_profile(time_bins, dt_ns)
    irf_dataset = FLIMDataset(
        decay=np.broadcast_to(irf_dataset_profile, (size, size, time_bins)).astype(np.float64).copy(),
        name="irf", dt_ns=dt_ns, rep_rate_mhz=rep_rate_mhz,
    )
    decay_dataset = FLIMDataset(decay=decay_cube, name="d", dt_ns=dt_ns, rep_rate_mhz=rep_rate_mhz)
    items = [TrainingImageInput(dataset=decay_dataset, irf=irf_dataset, name="d")]
    settings = AnalysisSettings(
        method="flimlib_reconvolution", dt_ns=dt_ns, model_order=2, fit_distribution_model=2,
        fit_min_counts=50.0, fit_noise_model=1, fit_irf_mode=0,
        fit_background_mode=1, fit_background_value=3.0,
        global_iterations=40, max_global_pixels=0, native_fit_bins=time_bins,
    )
    assert settings.initial_taus_ns == ()  # genuinely unseeded by the caller

    model = train_global_model(items, settings, name="phasor-seeded", phasor_seed=True)
    recovered = sorted(zip(model.global_taus_ns[0::2], model.global_taus_ns[1::2]))
    (rec_tau0, rec_h0), (rec_tau1, rec_h1) = recovered
    assert_close(rec_tau0, tau_a, 0.3 * tau_a, "trained tau (fast component), phasor-seeded")
    assert_close(rec_tau1, tau_b, 0.3 * tau_b, "trained tau (slow component), phasor-seeded")


def main() -> None:
    tests = [
        test_block_binning,
        test_phasor_recovers_mono_lifetime,
        test_mono_fit_recovers_lifetime,
        test_ics_v2_and_v1_readers,
        test_full_demo_and_export,
        test_native_analysis_pipeline,
        test_synthetic_gaussian_irf_recovers_known_fwhm,
        test_analyze_dataset_with_synthetic_irf_and_no_loaded_irf,
        test_phasor_informed_seed_recovers_two_component_mixture,
        test_train_global_model_phasor_seed_end_to_end,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
