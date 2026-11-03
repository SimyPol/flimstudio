from __future__ import annotations

"""Typed bridge to the length-checked FLIMLib LabVIEW DLL.

The DLL is deliberately not loaded on import.  Native execution only occurs
when the user selects one of the FLIMLib lifetime methods.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from ._paths import app_root, is_frozen


class NativeBackendError(RuntimeError):
    """A native FLIM fit could not be started or completed safely."""


@dataclass(slots=True)
class NativeFitResult:
    tau_components_ns: np.ndarray
    tau_mean_ns: np.ndarray
    amplitude_components: np.ndarray
    background: np.ndarray
    chi_square: np.ndarray
    fit_status: np.ndarray
    photon_counts: np.ndarray
    time_shift_ns: np.ndarray | None = None
    fit_start_bin: np.ndarray | None = None
    fit_stop_bin: np.ndarray | None = None
    exit_flag: np.ndarray | None = None
    global_taus_ns: np.ndarray | None = None
    global_stats: np.ndarray | None = None
    dll_path: Path | None = None
    dll_sha256: str = ""
    wrapper_code: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "tau_components_ns": self.tau_components_ns,
            "tau_mean_ns": self.tau_mean_ns,
            "amplitude_components": self.amplitude_components,
            "fit_background": self.background,
            "chi_square_native": self.chi_square,
            "fit_status": self.fit_status,
            "photon_counts_native": self.photon_counts,
        }
        optional = {
            "time_shift_ns": self.time_shift_ns,
            "fit_start_bin": self.fit_start_bin,
            "fit_stop_bin": self.fit_stop_bin,
            "exit_flag": self.exit_flag,
            "global_taus_ns": self.global_taus_ns,
            "global_stats": self.global_stats,
        }
        arrays.update({name: value for name, value in optional.items() if value is not None})
        return arrays


@dataclass(slots=True)
class GlobalTrainResult:
    """Output of ``flim_train_global_model_v1``: one shared model, many images.

    ``fit_status`` and ``photon_counts`` are stacked ``(image_count, rows,
    columns)`` -- one status/count per training pixel across every image in
    the stack, in the same ``image, row, column`` order the DLL uses.
    """

    global_taus_ns: np.ndarray
    global_stats: np.ndarray
    fit_status: np.ndarray
    photon_counts: np.ndarray
    dll_path: Path | None = None
    dll_sha256: str = ""
    wrapper_code: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, np.ndarray]:
        return {
            "global_taus_ns": self.global_taus_ns,
            "global_stats": self.global_stats,
            "fit_status": self.fit_status,
            "photon_counts_native": self.photon_counts,
        }


# 2026-09-02 rebuild #2 (v0.14.0): adds an irf_groups=<n>/<selected>
# diagnostic to globalStatsOut[9] (surfaced by native_worker.py's
# NATIVE_WORKER_TIMING log line) -- reports how many distinct IRF groups
# the 2026-09-02 #1 grouped-fast-path (see below) actually collapsed the
# selected pixels into. Built to diagnose a real-world report of no
# speedup with irfMode=1: if irf_groups comes back close to the
# selected-pixel count instead of the expected handful of detectors, the
# per-pixel IRF rows meant to share a detector's IRF aren't
# byte-identical (build_irf_groups groups by exact content), which
# defeats the grouping. No fitting-numerics change in this rebuild,
# purely additive instrumentation.
#
# 2026-09-02 rebuild #1 (v0.14.0): adds a grouped-by-IRF fast inner solve to
# the shared-tau search (see the block comment above
# score_global_taus_with_irfs in native/flimfit_labview.c). At each trial
# tau, selected pixels are bucketed by IRF *content* (build_irf_groups),
# and FLIMLib's own real global-fit building blocks
# (GCI_marquardt_global_exps_calculate_exps_instr +
# GCI_marquardt_global_exps_do_fit_single -- the exact routines
# GCI_marquardt_global_exps_instr/TRI2 use internally) compute the
# IRF-convolved exponential basis ONCE per group and reuse it for every
# pixel in that group, instead of each pixel independently re-deriving
# the same convolution via the generic single-transient engine. Groups
# are built once per search (outside the outer Marquardt-damped-Newton
# loop added 2026-09-01, still in place) since the grouping doesn't
# depend on tau.
#
# Nothing touched anywhere near the null-sigma-pointer fix this hash pin
# actually guards, so constant weighting is exactly as safe here as in
# every build this hash has previously pointed to. Update this again
# (and keep the old DLL around, e.g.
# flimfit_labview.dll.pre-group-diagnostic-backup) on the next legitimate
# rebuild -- and copy the rebuilt DLL to BOTH native/flimfit_labview.dll
# and flim_app/native/flimfit_labview.dll (see the DLL-resolution gotcha
# in RECONSTRUCTION_NOTES.md; app_root() checks the flim_app/native/ copy
# first when no explicit native_dll_path is given).
#
# 2026-09-02 rebuild #3 (v0.14.0): fixes the irf_groups diagnostic added
# in rebuild #2 above -- it was only wired up in
# flim_fit_global_pixels_v2, not flim_train_global_model_v1 (a separate
# hardcoded globalStatsOut[9] = 0.0 there, missed because its preceding
# comment text differed from the one a find/replace matched). The actual
# grouped-fast-path fitting was never affected by this -- only the
# reported group count for train_global_model calls, which always read
# back as 0 regardless of the real (correctly fast) grouping. If you're
# using train_global_model and seeing irf_groups=0/<n> from a DLL built
# before this rebuild, that 0 is this bug, not a sign grouping failed --
# check timing instead.
#
# 2026-09-03 rebuild (v0.16.0): adds a "distribution model" toggle to
# flim_fit_global_pixels_v2 and flim_train_global_model_v1 -- a new
# `fitType` parameter (FIT_GLOBAL_MULTIEXP, the existing default, or
# FIT_GLOBAL_STRETCHEDEXP) selects between the discrete multi-exponential
# model and FLIMLib's own Kohlrausch-Williams-Watts stretched-exponential
# model (GCI_stretchedexp -- already fully implemented and analytically
# differentiated in the bundled FLIMLib source, just never wired up
# anywhere in this DLL before now). Stretched-exp represents one species'
# lifetime as a distribution (via the stretching exponent h) rather than
# a single discrete value -- h=1 is a plain single exponential, h>1
# "stretches" it. The outer Marquardt-damped-Newton search and the
# IRF-grouped fast inner solve both generalise to the shared parameter
# being [tau, h] instead of [tau] (or [tau1, tau2, tau3]) with no loss of
# speed -- FLIMLib's own GCI_marquardt_global_compute_exps_fn/_final
# already handle both fit types internally. Scoped to global fitting/
# training only this round -- flim_apply_global_model_v1 still always
# uses FIT_GLOBAL_MULTIEXP regardless (applying a stretched-exp model
# isn't wired up yet). See RECONSTRUCTION_NOTES.md for the full design.
# 2026-09-03 rebuild #2 (v0.16.0): fixes a real, pre-existing bug in
# bounded_tau/bounded_h found while validating the distribution-model
# toggle above -- both functions reset ANY negative-but-finite input to
# an arbitrary fixed default (a quarter of the acquisition window, or
# h=1) instead of clamping it to their actual lower bound. This didn't
# just affect out-of-range *inputs*: levenberg_marquardt_min calls these
# same functions to clamp every trial step it proposes, so a normal,
# expected large-but-finite Newton-step overshoot (before damping has
# caught up) got teleported to that one fixed point regardless of how
# far it overshot by. That created a damping-insensitive trap: every
# retry at ever-higher lambda still landed on the exact same reset value
# with the exact same (bad) objective, so the search could get stuck
# bouncing between its current point and that one fixed point forever --
# confirmed via direct evaluation-by-evaluation tracing, reproducing
# reliably for model_order=1 (and, by the same mechanism, likely at
# least occasionally for any model_order/fit_type whenever an early step
# overshoots past zero) on an otherwise completely clean, unambiguous
# single-exponential test case (fixed-tau objective scan confirmed a
# single well-behaved minimum at the true tau, monotonically increasing
# in both directions -- the search should never have missed it). Fixed
# by only resetting genuinely non-finite (NaN/Inf) values to the
# default; everything else, including negative values, now clamps to the
# real lower bound the very next line already computes, preserving the
# monotonic relationship damping needs to actually converge. Pre-existing
# since levenberg_marquardt_min was first written (2026-09-01) --
# discovered, not introduced, by today's distribution-model work; also
# fixes flim_apply_global_model_v1 and flim_fit_global_pixels_v1's own
# uses of bounded_tau for out-of-range fixed/initial taus.
#
# 2026-09-06 rebuild: the clamp-vs-reset fix above was necessary but not
# sufficient -- levenberg_marquardt_min could still get stuck producing the
# exact same (wrong) converged point regardless of retry/iteration budget.
# Root cause: the finite-difference Hessian estimate driving the damped-
# Newton step can come out spuriously small (confirmed via direct file-
# based tracing -- see RECONSTRUCTION_NOTES.md), implying a wildly oversized
# step that increasing lambda alone couldn't shrink into range within the
# retry budget. Fixed by adding a standard trust-region cap: a single step
# may not shrink a shared parameter to less than half or grow it to more
# than double its current value, regardless of what the raw damped-Newton
# formula proposes. This is the DLL that also first wires up the
# stretched-exponential (distribution) global-fit toggle (fitType/
# sharedCount generalization throughout flimfit_labview.c).
# 2026-09-07 rebuild: adds FIT_GLOBAL_MULTI_STRETCHEDEXP (fit_type=2), a
# new app-defined model summing up to 3 INDEPENDENTLY distributed
# (stretched-exponential) components -- I(t) = Z + sum_i A_i*exp(-(t/
# tau_i)^(1/h_i)) -- generalizing the single-component stretched-exp
# toggle above. See flim_app.models.AnalysisSettings.fit_distribution_model
# and RECONSTRUCTION_NOTES.md. Widened MAX_SHARED_PARAMS (3->6) and raised
# the globalIterations ceiling (20->80) to give the larger shared-
# parameter search enough room; both are additive, no behaviour change for
# fit_type 0/1 at their existing defaults. Always uses the per-pixel
# fallback path (score_global_taus_with_irfs), never the fast grouped-IRF
# path -- extending that is deliberately deferred, see below.
#
# 2026-09-07 rebuild #2 (this hash): fixes a real h-seed-degeneracy bug --
# every component's h was seeded to the identical 1.5, so at N>=2 the
# per-pixel amplitude linear sub-problem started badly conditioned
# (near-collinear basis shapes differing only by tau), and most pixels'
# inner fit failed outright the moment ANY parameter moved even slightly
# off that seed, swamping the outer search's gradient with huge failure
# penalties. h is now seeded per component (1.5, 2.1, 2.7, ...).
#
# 2026-09-07 rebuild #3 (this hash): adds the fast IRF-grouped path for
# FIT_GLOBAL_MULTI_STRETCHEDEXP, which previously always used the slow
# per-pixel fallback (a real 32x32/1024px fit measured to exceed a 1800s
# timeout). New fit_linear_multi_stretched_pixel does a small weighted
# linear solve (genuinely linear once tau/h are fixed -- no nonlinear
# search needed for that step at all) against a basis built by calling
# bundled GCI_marquardt_global_exps_calculate_exps_instr once per
# component per IRF group (in ordinary single-component
# FIT_GLOBAL_STRETCHEDEXP mode) -- see RECONSTRUCTION_NOTES.md. Measured:
# a 32x32/1024px N=2 fit that previously timed out now completes in
# ~19s.
#
# This resolved the earlier N=2 "does it move at all" ambiguity: the fast
# path converges deterministically to the same point regardless of
# iteration budget or a 10x photon-count change, confirming it's finding
# a genuine, stable local optimum rather than behaving erratically.
# Investigating that finding explained the *precision* gap too: the
# already-shipped, already-validated multiexp/single-stretched fast path
# shows the identical signature (recovered tau ~3.12-3.13ns vs a true
# 3.0ns, persistent across a 10x SNR change) when tested the same way --
# not a new bug, but FLIMLib's own documented tradeoff. Bundled EcfSingle.c
# has a FIXME explaining it directly: the array-optimized reconvolution
# functions used by any fast/grouped path (stretchedexp_array etc., and
# by extension calculate_exps_instr, which computes the same shape) place
# the model's time-zero at absolute bin 0, whereas the generic per-point
# fitfunc engine the slow fallback path uses places it at fit_start --
# "adverse" by FLIMLib's own authors' description, kept for speed. Every
# fast/grouped path inherits this size-of-bias tradeoff; multi-stretched
# shows a larger version of it than plain multiexp, consistent with (not
# a new problem beyond) the tau/h correlation flagged from when this
# feature was first proposed, now compounded across components. See
# RECONSTRUCTION_NOTES.md for the full account, including where N=2's
# recovered values still sit relative to ground truth on the validation
# scenario.
#
# 2026-09-07 rebuild #4 (this hash): fixes two further, real, confirmed
# issues found while investigating rebuild #3's precision gap further
# (visual diagrams first, per user request -- see RECONSTRUCTION_NOTES.md):
# (1) the fast path's basis anchor was wrong in a *specific* way: an
# intermediate attempt anchored the model's own t=0 at fit_start, which
# stacks two delays on top of each other (fit_start is deliberately past
# the true onset, and the model then gets convolved with an IRF that's
# still at its own natural position) -- confirmed visually to shift the
# observable rise ~13 bins too late. Fixed to anchor at infer_window's
# transientStart concept (decayRise - instrRise, no added margin)
# instead, computed sub-bin (linearly interpolated at each threshold
# crossing, new infer_transient_start_time) rather than snapped to a
# whole bin -- a whole-bin rounding error is a large fraction of a
# narrow IRF's own width. (2) the per-pixel IRLS weighting floored
# variance at 1 count; bundled EcfGlobal.c's GCI_marquardt_global_exps_
# single_step (what the legacy fast path actually calls for the same
# noise models) floors at 15 -- confirmed by direct source comparison.
# Both are real fixes, confirmed via SNR-scaling (no longer erratic
# across a 10x photon-count change) and via bit-identical pytest
# regression on the untouched multiexp/single-stretched fast paths.
#
# 2026-09-07 rebuild #5 (this hash): the transientStart anchor above was
# ITSELF wrong -- confirmed by directly comparing against bin 0 with the
# other two fixes in place (rebuild #4's weighting floor, plus this
# fast path's own from-scratch double-precision convolution): forcing
# bin 0 gave 3.14ns/2.50 against a true 3.0ns/2.5 (matching almost
# exactly), consistently across multiple random seeds, and clearly
# better than the legacy single-component fast path's own 3.66ns/2.27
# on the same data -- while transientStart (even sub-bin-interpolated)
# gave 4.5-4.6ns, worse than doing nothing at all. The reasoning once
# traced through: for ordinary reconvolution fitting (model (*) IRF),
# the pure model's own t=0 is mathematically arbitrary -- the
# convolution with the IRF at its own, correctly-measured position is
# what reproduces the real observable timing, regardless of where the
# model's own zero sits, as long as it's early enough that the model is
# genuinely ~zero before the IRF's own rise (true for bin 0 in any real
# acquisition). bin 0 is not FLIMLib's array-optimized functions taking
# a shortcut at the cost of accuracy (an earlier, wrong reading of a
# bundled FIXME comment) -- it's the physically correct anchor. Fixed
# to always anchor at bin 0; infer_transient_start_time/
# interpolated_crossing_time are kept in source as separately-verified
# but currently-unused building blocks. Seeded near the true (tau, h),
# N=1 now recovers consistently close values (~3.3-3.4ns/~2.5-2.56)
# across several independent random seeds -- a real, validated
# improvement, not just a different bias. The default (far-off) seed
# can still land on a worse local optimum. N=2's gap was investigated
# separately (not fixed by this hash) and confirmed, by direct chi-
# square comparison, to be genuine model non-identifiability rather
# than a further code bug: the search's own converged answer has a
# LOWER (better) objective than the true generating parameters, at
# every SNR tested -- see RECONSTRUCTION_NOTES.md.
#
# 2026-09-08 rebuild #6 (this hash): adds multi-start optimization for
# FIT_GLOBAL_MULTI_STRETCHEDEXP specifically (MULTI_START_COUNT=5 in
# flimfit_labview.c) -- levenberg_marquardt_min is strictly local, and
# the N=2 finding above showed a real, confirmed case where it correctly
# converges to a genuinely-lower-objective point than the true
# parameters; multiple diverse starting points, keeping whichever finds
# the globally-lowest objective, at least removes the search itself as
# a *further* source of unreliability on top of that -- it can't fix
# genuine non-identifiability, but it stops the answer depending on
# which single local optimum an arbitrary default seed happened to
# land near. FIT_GLOBAL_MULTIEXP/FIT_GLOBAL_STRETCHEDEXP are
# deliberately untouched (still exactly one start, identical seeding
# formula) -- both already validated, no reason to add cost or risk
# there. `globalStatsOut[0]` (previously always 0.0) now reports which
# start variant won.
#
# 2026-09-08 rebuild #7 (this hash): fixes a real design flaw found by
# direct test right after rebuild #6 shipped -- multi-start was silently
# overriding a user's own explicit, correct `initial_taus_ns` seed
# whenever some other, wilder starting variant found a lower (but less
# physically plausible) objective. Confirmed concretely: seeding near
# the true (1.5, 1.8, 5.0, 3.0) ns/h values got overridden by a variant
# that converged to tau=73.3ns. Added `has_explicit_shared_seed()`:
# multi-start now only runs when the caller did NOT supply a fully
# specified shared-parameter seed (all `sharedCount` slots > 0); an
# explicit seed always gets exactly one search, respecting it as the
# starting point, matching pre-multi-start behaviour exactly. Re-tested
# after this fix: the same near-truth seed now stays in a sane
# neighborhood (never touches the tau=73 region) and completes in ~2s
# (one search) vs ~7s for the unseeded default case (five searches,
# still finding that same tau=73.3 "better-by-the-numbers" optimum --
# expected, matches the confirmed non-identifiability finding above,
# multi-start was never meant to fix that, only to stop an *arbitrary*
# default seed's local optimum from being the only answer considered).
#
# 2026-09-08 rebuild #8 (this hash): adds soft-ceiling regularization to
# the outer search's objective for FIT_GLOBAL_MULTI_STRETCHEDEXP only
# (evaluate_global_tau_objective; never applied to the reported final
# chi-square). User's framing: "yes lets try a regularisation before
# moving to N=3." Motivation: the multi-start default-seed case above
# was still landing on an unbounded, physically implausible "escape"
# (tau up to ~114ns against a 30ns window, h up to 14) -- a component
# whose shape goes nearly flat over the acquisition window can silently
# absorb residual without being constrained by the data's actual shape.
# Took two iterations to close off both escape routes and find a
# working strength: (1) a penalty on h alone was insufficient -- the
# search just found an equivalent escape via large tau instead; added a
# second penalty on tau relative to the acquisition window
# (TAU_REG_WINDOW_MULTIPLE). (2) the first working lambda (0.25) was
# too weak to matter at the scale of the summed per-pixel objective;
# raised to 5.0. (3) with a loose ceiling (h<=4, tau<=1.5x window) the
# search settled exactly AT the boundary rather than retreating --
# tightening the ceiling itself (h<=3, tau<=1.0x window) is what
# actually helped: the escaping component now parks visibly at that
# boundary (an easy-to-recognize signature) while the *other* component
# genuinely recovers close to its true value (5.0ns/3.0 recovered as
# ~5.5-5.8ns/~2.94, sample-dependent -- within ~10-16%/~2%, across both
# tested noise levels) instead of both being wrong. Does NOT fix the
# near-truth-seeded case's own (different) failure mode -- two
# components collapsing toward each other -- confirmed unaffected by
# this change; that's a separate manifestation of the same underlying
# non-identifiability, not targeted by this regularization. See
# RECONSTRUCTION_NOTES.md for the full iteration-by-iteration account
# and numbers.
# 2026-09-08 rebuild #9 (this hash, identical constants/logic to
# rebuild #8 -- the hash changed anyway from non-deterministic build
# metadata, e.g. the linker's embedded timestamp/PDB GUID, not a
# functional change): recorded here purely because of a real finding
# made while validating rebuild #8's regularization, per the user's
# "no i want n=2 to work". The synthetic test data used throughout this
# whole N=2 investigation had two accidentally-pathological properties
# neither representative of real FLIM data nor of this app's own
# defaults: (1) every pixel was broadcast the *identical* mixing ratio
# between the two components, discarding the actual mechanism that
# makes multi-component global analysis identifiable (real global
# analysis resolves overlapping lifetimes using pixels/images with
# *different* relative amounts of each species -- a spatially uniform
# test image can never provide that, no matter how many photons); (2)
# background was fit as a free per-pixel parameter -- which, paired
# with a component broad/slow enough to look nearly flat over the
# acquisition window, creates a genuine mathematical near-degeneracy
# (a flat curve is redundant with a free additive offset). Retested
# with a corrected synthetic scenario -- pixel-varying mixing fractions
# (0.05-0.95, linearly across the image) plus FIXED background
# (fit_background_mode=1, this app's GUI default) -- and a seed near a
# reasonable prior estimate: recovered both components within roughly
# the same 10-20% band N=1 was already validated to, consistently
# across 3 independent random-data seeds and 2 SNR levels. The
# unregularized default/unseeded case still shows a milder residual
# escape (one component landing on a moderate, plausible-looking but
# wrong tau); the seeded case is genuinely reliable. See
# RECONSTRUCTION_NOTES.md for the full numbers -- this is the resolution
# of the "genuine non-identifiability" diagnosis two entries above:
# that diagnosis was correct for the specific (pathological) test data
# used, not a property of the model or search itself.
# 2026-09-08 rebuild #10 (this hash): improves the *unseeded/default*
# case for FIT_GLOBAL_MULTI_STRETCHEDEXP, which was the one remaining
# weakness noted in rebuild #9's finding. User's framing: "yes lets
# improve". Two additions, both scoped to this fit type only:
#
# 1. `estimate_data_driven_multi_stretched_seed()`: instead of an
#    arbitrary fixed formula, variant 0's tau fallback is now built from
#    the training data's OWN per-pixel apparent (moment) lifetime spread
#    -- sorting selected pixels by a cheap moment-based lifetime estimate
#    and reading off modelOrder evenly-spaced quantiles gives a real,
#    physically-motivated starting tau per component (close to what an
#    analyst manually picking "the purest-looking pixels" to seed a
#    global fit from would do), grounded in the fact that real global
#    analysis identifies overlapping components using pixels with
#    genuinely different relative amounts of each species -- the same
#    mechanism confirmed in rebuild #9's finding.
# 2. Confirmed by direct per-variant trace that step 1 ALONE wasn't
#    enough: the data-driven variant reliably converged close to truth
#    (e.g. 1.27/2.00, 4.35/2.77 against true 1.5/1.8, 5.0/3.0), but a
#    different, purely arbitrary formulaic variant still sometimes
#    converged to a milder, still-plausible-looking "compromise" point
#    (e.g. 18.72/1.83, 5.26/2.67) with a genuinely LOWER raw objective --
#    too mild to trip the h/tau regularization ceiling from rebuild #8.
#    Under the old "keep whichever variant has the lowest objective"
#    rule, that arbitrary-but-lucky variant silently won, discarding the
#    trustworthy data-grounded answer. Added
#    `multi_stretched_result_is_plausible()` and changed the selection
#    rule: the data-driven variant's own converged result is now
#    preferred outright whenever it's available and still looks
#    physically plausible after convergence (same h/tau ceiling used for
#    the search-time regularization), rather than always taking whichever
#    variant reports the lowest number. The diverse formulaic variants
#    remain the fallback for when the data-driven start isn't available
#    or itself converges somewhere implausible.
#
# Result, retested across 4 independent random-data seeds x 2 SNR levels,
# UNSEEDED (no initial_taus_ns at all): at the higher SNR, all 4 seeds
# now recover both components correctly (e.g. 1.27-1.28/2.00,
# 4.2-4.4/2.77-2.79 against true 1.5/1.8, 5.0/3.0) -- unseeded now
# matches seeded quality. At the lower SNR: improved but not perfect (2
# of 4 clean, 1 mild residual escape, 1 component-collapse) -- expected
# graceful degradation with less photon budget, a genuine statistical
# limit, not a further code gap. Seeded behaviour (single-start,
# has_explicit_shared_seed) and every other fit type are untouched and
# reconfirmed unaffected. See RECONSTRUCTION_NOTES.md for the full
# numbers.
# 2026-09-08 rebuild #11 (this hash): fixes a real oversight in rebuild
# #10's "prefer the data-driven variant" override, found while starting
# N=3 (fit_distribution_model=2, model_order=3) validation. The override
# had no safeguard against the data-driven seed itself being bad: with 3
# independently-distributed components, the same per-pixel first-moment
# estimate that separates well for N=2 can fail to separate for N=3 (a
# longer-lived component's tail disproportionately pulls the moment
# estimate upward even in pixels "dominated" by a faster one, so all 3
# quantile seeds can land within ~1ns of each other) -- confirmed by
# direct trace: all 3 seed taus at ~12ns, search stayed collapsed near
# there (objective 1149.9), while an ordinary formulaic variant found a
# genuinely much better point (367.4, 3.1x lower) that
# `multi_stretched_result_is_plausible` alone still accepted (nothing in
# the collapsed answer individually crosses the regularization ceiling).
# Added `DATA_DRIVEN_OBJECTIVE_TOLERANCE` (1.5): the data-driven variant
# is now preferred only when its objective isn't more than 50% worse
# than the best formulaic variant found -- keeps N=2's behaviour
# unchanged (its gap there was only ~7%) while correctly rejecting a
# collapse this large for N=3. N=3 itself is NOT yet validated as
# working -- see RECONSTRUCTION_NOTES.md's N=3 entry for the full,
# honest status (seeded: reasonable but the middle of 3 components is
# consistently weaker; unseeded: not yet reliable).
# 2026-09-08 rebuild #12 (this hash): fixes a real, significant bug
# found while pushing on N=3 (fit_distribution_model=2, model_order=3)
# per the user's "yes lets do this" (cascade seeding request) --
# investigating why a truth-seeded N=3 search still seemed to collapse
# 2 of 3 components together, found the outer search's own evaluation
# budget (`globalIterations * 15`, unchanged since this feature's first
# build) was NEVER SCALED by sharedCount -- for N=2 (sharedCount=4) that
# 15x multiplier is generous (confirmed: search self-terminates at
# ~400 evaluations from a ~600 budget, converges consistently); for N=3
# (sharedCount=6) it is NOT -- a truth-seeded search given the exact
# same budget had not converged at all, needing ~1180 evaluations
# (confirmed directly: 300/800/2000-iteration budgets all gave the
# identical, actually-converged answer, while 40 gave a visibly
# under-converged, apparently-collapsed intermediate snapshot). Added
# `evaluation_budget_multiplier(sharedCount)`: scales the flat 15x up
# by the ratio of each shared-parameter-count's own per-Newton-iteration
# cost (4n+2n(n-1)) to N=2's (n=4), with `max(1.0, ...)` guaranteeing
# every already-validated fit type/order (sharedCount<=4: FIT_GLOBAL_
# MULTIEXP modelOrder 1-3, FIT_GLOBAL_STRETCHEDEXP, FIT_GLOBAL_MULTI_
# STRETCHEDEXP's own N=1/N=2) keeps EXACTLY its previous multiplier,
# reconfirmed byte-identical across 4 seeds. Only sharedCount=6 (N=3)
# gets more (15 -> 31.5). Retested N=3 at the SAME global_iterations=40
# the app already defaults toward using: seeded recovery now reliably
# gets the fast and slow components right across 3 independent random
# seeds (within the usual ~15-20% band) with a stable, systematic
# (not randomly collapsing) ~30-33% overestimate on the middle
# component -- a real, meaningfully better and more honestly
# characterizable result than the apparent "component collapse" this
# same scenario showed before the fix, which turns out to have been
# substantially a search-budget artifact, not (or not solely) a deeper
# identifiability limit. See RECONSTRUCTION_NOTES.md for the full
# account and numbers.
# 2026-09-08 rebuild #13 (this hash): a real regression found and fixed
# within the same session. While tracing N=3's unseeded default case
# (per the user's "yes lets keep going"), found that when two candidate
# component taus land close together the per-pixel linear solve
# (fit_linear_multi_stretched_pixel) goes near-singular, producing a
# wildly inflated objective (confirmed as high as 1.7 MILLION). Tried
# adding a soft minimum-log-separation penalty directly into the outer
# search's own objective (evaluate_global_tau_objective), the same
# pattern as the h/tau ceiling penalties -- and immediately found, by
# direct test, that this REGRESSED rebuild #12's already-validated N=2
# behaviour: N=2's data-driven variant, which reliably converged to a
# genuinely separated (1.27ns, 4.35ns) answer, instead converged to a
# DEGENERATE collapsed pair (1.44ns, 1.39ns) once the new term was added
# -- adding ANY term to the search's own objective reshapes the whole
# gradient/Hessian landscape along the entire convergence path, not just
# near the final point, and can reroute an already-good trajectory
# somewhere new and worse even when the final point itself would never
# have triggered the term. Reverted that specific addition; kept the
# same MIN_LOG_TAU_SEPARATION threshold ONLY as a passive post-hoc
# filter in `multi_stretched_result_is_plausible` (used solely to decide
# whether the data-driven-preference override is trustworthy -- see
# rebuild #10's comment -- never part of what the search itself
# optimizes against), which cannot reroute any trajectory since it's
# evaluated only after a variant has already finished converging.
# Reconfirmed N=2's exact 4-seed unseeded validation byte-identical
# again after this fix, and N=3's rebuild #12 seeded improvement
# (evaluation-budget scaling) unaffected. See RECONSTRUCTION_NOTES.md
# for the full account -- a real lesson (documented there) about adding
# terms to an iterative search's own objective versus filtering its
# output afterward.
# 2026-09-08 rebuild #14 (this hash): a second attempted fix for the
# near-singular-linear-solve finding (rebuild #13's comment) tried and
# reverted, per the user's "no lets keep going. I believe in you" --
# a small ridge (Tikhonov) term added directly to fit_linear_multi_
# stretched_pixel's own per-pixel normal-equations diagonal (scaled to
# the matrix's own mean diagonal magnitude, so intended to be
# negligible for a well-conditioned solve, and -- unlike rebuild #13's
# reverted attempt -- never touching the outer tau/h search's own
# gradient/Hessian at all). Direct before/after testing found it wasn't
# a net win: N=2's unseeded validation drifted measurably (up to ~7% on
# recovered tau -- still the right ballpark, but no longer the exact
# previously-validated numbers) purely from many compounded per-pixel
# solves along a long nonlinear search trajectory each nudged slightly,
# while N=3's unseeded case showed NO corresponding practical
# improvement (still didn't recover the slowest true component,
# comparable final objective to before, and ran noticeably slower).
# Real precision cost for no measured benefit -- reverted. Reconfirmed
# N=2's exact 4-seed unseeded validation byte-identical again. See
# RECONSTRUCTION_NOTES.md for the full numbers.
_PATCHED_DLL_SHA256 = "efe017536e9ffaa5b53061966f4a771e5eb6c13bcd5c3a9936cfb18f40d1372c"
_DLL_LOCK = threading.Lock()


def candidate_dll_paths(explicit_path: str | Path | None = None) -> list[Path]:
    """Return ordered, de-duplicated native-backend candidates."""

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    environment_path = os.environ.get("FLIMFIT_LABVIEW_DLL", "").strip()
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    # app_root() alone already covers the frozen (PyInstaller) case
    # correctly -- see _paths.py -- but the other candidates are kept too,
    # both for source-tree layouts app_root() doesn't match and as a
    # portable-copy fallback (e.g. the DLL dropped next to the exe).
    candidates.append(app_root() / "native" / "flimfit_labview.dll")
    package_root = Path(__file__).resolve().parents[1]
    package_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            package_root / "native" / "flimfit_labview.dll",
            package_dir / "native" / "flimfit_labview.dll",
            package_root / "flimfit_labview.dll",
        ]
    )
    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend(
            [
                exe_dir / "native" / "flimfit_labview.dll",
                exe_dir / "flimfit_labview.dll",
            ]
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve(strict=False)).casefold()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def find_native_dll(explicit_path: str | Path | None = None) -> Path | None:
    for path in candidate_dll_paths(explicit_path):
        if path.is_file():
            return path.resolve()
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_I = ctypes.c_int
_D = ctypes.c_double


class FLIMLibLabViewBackend:
    """ctypes binding for ``flim_fit_*_v2`` from FLIMFitTool."""

    def __init__(self, dll_path: str | Path | None = None) -> None:
        resolved = find_native_dll(dll_path)
        if resolved is None:
            checked = "\n".join(f"  - {path}" for path in candidate_dll_paths(dll_path))
            raise NativeBackendError(
                "The FLIMLib fitting DLL was not found. Choose it in Native fit settings "
                "or set FLIMFIT_LABVIEW_DLL. Checked:\n" + checked
            )
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise NativeBackendError("The supplied FLIM fitting DLL requires 64-bit Python.")
        self.path = resolved
        self.sha256 = _sha256(resolved)
        try:
            self._dll = ctypes.CDLL(str(resolved))
        except OSError as exc:
            raise NativeBackendError(f"Could not load {resolved}: {exc}") from exc
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        try:
            pixel = self._dll.flim_fit_pixels_v2
            global_fit = self._dll.flim_fit_global_pixels_v2
        except AttributeError as exc:
            raise NativeBackendError(
                "This DLL does not export the required length-checked v2 fitting functions."
            ) from exc
        try:
            train_global = self._dll.flim_train_global_model_v1
            apply_global = self._dll.flim_apply_global_model_v1
        except AttributeError:
            train_global = None
            apply_global = None

        pixel.argtypes = [
            _DOUBLE_PTR, _DOUBLE_PTR,
            _I, _I, _I, _I, _I,
            _D, _I, _I, _D, _I, _D, _I,
            _DOUBLE_PTR, _I, _DOUBLE_PTR, _I, _DOUBLE_PTR, _I,
            _DOUBLE_PTR, _I, _DOUBLE_PTR, _I, _DOUBLE_PTR, _I,
            _DOUBLE_PTR, _I, _DOUBLE_PTR, _I, _DOUBLE_PTR, _I,
            _DOUBLE_PTR, _I, _DOUBLE_PTR, _I,
        ]
        pixel.restype = _I

        global_fit.argtypes = [
            _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR,
            _I, _I, _I, _I, _I, _I,
            _D, _I, _I, _I, _D, _I, _D,
            _I, _I, _I, _I, _I,
            _DOUBLE_PTR, _I, _DOUBLE_PTR, _I, _DOUBLE_PTR, _I,
            _DOUBLE_PTR, _I, _DOUBLE_PTR, _I, _DOUBLE_PTR, _I,
            _DOUBLE_PTR, _I, _DOUBLE_PTR, _I, _DOUBLE_PTR, _I,
            _DOUBLE_PTR, _I,
        ]
        global_fit.restype = _I

        if train_global is not None:
            # decay, irf, mask
            # imageCount, rows, columns, nBins
            # decayLength, irfLength, maskLength
            # binWidth, modelOrder, backgroundMode, backgroundValue, noiseModel
            # minPhotons, fitStartBin, fitStopBin, maxGlobalPixels
            # irfMode, globalIterations
            # initialTaus[+Length], globalTauOut[+Length], globalStatsOut[+Length],
            # fitStatusMap[+Length], photonCountMap[+Length]
            train_global.argtypes = [
                _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR,
                _I, _I, _I, _I,
                _I, _I, _I,
                _D, _I, _I, _I, _D, _I,
                _D, _I, _I, _I,
                _I, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
            ]
            train_global.restype = _I

        if apply_global is not None:
            # decay, irf, mask
            # imageCount, rows, columns, nBins
            # decayLength, irfLength, maskLength
            # binWidth, modelOrder, backgroundMode, backgroundValue, noiseModel
            # minPhotons, fitStartBin, fitStopBin, irfMode
            # fixedTaus[+Length], tauMaps[+Length], tauMeanMap[+Length],
            # amplitudeMaps[+Length], backgroundMap[+Length], chiSquareMap[+Length],
            # fitStatusMap[+Length], photonCountMap[+Length]
            apply_global.argtypes = [
                _DOUBLE_PTR, _DOUBLE_PTR, _DOUBLE_PTR,
                _I, _I, _I, _I,
                _I, _I, _I,
                _D, _I, _I, _D, _I,
                _D, _I, _I, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
                _DOUBLE_PTR, _I,
            ]
            apply_global.restype = _I
        self._has_global_model_functions = train_global is not None and apply_global is not None

    @staticmethod
    def _inputs(
        decay: np.ndarray,
        irf: np.ndarray,
        irf_mode: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        cube = np.ascontiguousarray(decay, dtype=np.float64)
        if cube.ndim != 3:
            raise NativeBackendError(f"Native decay input must be (y, x, time), got {cube.shape}.")
        rows, columns, n_bins = (int(value) for value in cube.shape)
        if n_bins <= 3:
            raise NativeBackendError("Native fitting requires at least four time bins.")
        if cube.size > 2_147_483_647:
            raise NativeBackendError("The decay array exceeds the native API's signed 32-bit length limit.")
        if not np.isfinite(cube).all():
            raise NativeBackendError("Native fitting requires finite decay samples.")

        reference = np.asarray(irf, dtype=np.float64)
        if reference.ndim == 1:
            if reference.size != n_bins:
                raise NativeBackendError("The IRF time-bin count does not match the decay data.")
            if int(irf_mode) == 1:
                reference = np.broadcast_to(reference, cube.shape)
        elif reference.ndim == 3 and reference.shape[-1] == n_bins:
            if int(irf_mode) == 0:
                reference = np.nanmean(reference, axis=(0, 1))
            elif reference.shape[:2] != cube.shape[:2]:
                mean_reference = np.nanmean(reference, axis=(0, 1))
                reference = np.broadcast_to(mean_reference, cube.shape)
        else:
            raise NativeBackendError("IRF input must be a trace or a compatible (y, x, time) cube.")

        reference = np.ascontiguousarray(np.where(np.isfinite(reference), reference, 0.0), dtype=np.float64)
        if reference.size > 2_147_483_647:
            raise NativeBackendError("The IRF exceeds the native API's signed 32-bit length limit.")
        if not np.any(reference > 0):
            raise NativeBackendError("The IRF contains no positive finite samples.")
        return cube, reference, rows, columns, n_bins

    @staticmethod
    def _stacked_inputs(
        decay: np.ndarray,
        irf: np.ndarray,
        mask: np.ndarray | None,
        irf_mode: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
        """Validate a multi-image ``(images, y, x, time)`` stack for train/apply.

        Mirrors ``_inputs`` but with a leading image axis, matching the
        ``imagePixel = imageIndex*rows*columns + row*columns + column``
        layout the DLL's ``*_global_model_v1`` functions expect.
        """

        cube = np.ascontiguousarray(decay, dtype=np.float64)
        if cube.ndim != 4:
            raise NativeBackendError(
                f"Stacked decay input must be (images, y, x, time), got {cube.shape}."
            )
        image_count, rows, columns, n_bins = (int(value) for value in cube.shape)
        if n_bins <= 3:
            raise NativeBackendError("Native fitting requires at least four time bins.")
        if cube.size > 2_147_483_647:
            raise NativeBackendError("The stacked decay array exceeds the native API's signed 32-bit length limit.")
        if not np.isfinite(cube).all():
            raise NativeBackendError("Native fitting requires finite decay samples.")

        reference = np.asarray(irf, dtype=np.float64)
        if reference.ndim == 1:
            if reference.size != n_bins:
                raise NativeBackendError("The IRF time-bin count does not match the decay data.")
            if int(irf_mode) == 1:
                reference = np.broadcast_to(reference, cube.shape)
        elif reference.ndim == 4 and reference.shape[-1] == n_bins:
            if int(irf_mode) == 0:
                reference = np.nanmean(reference, axis=(0, 1, 2))
            elif reference.shape[:3] != cube.shape[:3]:
                raise NativeBackendError(
                    "Per-pixel IRF stack shape "
                    f"{reference.shape[:3]} must match the decay stack {cube.shape[:3]}."
                )
        else:
            raise NativeBackendError("IRF input must be a trace or a compatible (images, y, x, time) stack.")

        reference = np.ascontiguousarray(np.where(np.isfinite(reference), reference, 0.0), dtype=np.float64)
        if reference.size > 2_147_483_647:
            raise NativeBackendError("The IRF exceeds the native API's signed 32-bit length limit.")
        if not np.any(reference > 0):
            raise NativeBackendError("The IRF contains no positive finite samples.")

        shape = (image_count, rows, columns)
        if mask is None:
            mask_array = np.ones(shape, dtype=np.float64)
        else:
            raw_mask = np.asarray(mask)
            if raw_mask.shape != shape:
                raise NativeBackendError(f"Mask must have shape {shape}, got {raw_mask.shape}.")
            mask_array = np.ascontiguousarray(np.isfinite(raw_mask) & (raw_mask > 0), dtype=np.float64)

        return cube, reference, mask_array, image_count, rows, columns, n_bins

    @staticmethod
    def _output_arrays(
        rows: int, columns: int, model_order: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = (rows, columns)
        component_shape = (model_order, rows, columns)
        return (
            np.empty(component_shape, dtype=np.float64),
            np.empty(shape, dtype=np.float64),
            np.empty(component_shape, dtype=np.float64),
            np.empty(shape, dtype=np.float64),
            np.empty(shape, dtype=np.float64),
            np.empty(shape, dtype=np.float64),
            np.empty(shape, dtype=np.float64),
        )

    @staticmethod
    def _sort_components(tau: np.ndarray, amplitude: np.ndarray) -> None:
        """Sort successful finite components longest-to-shortest for presentation."""

        if tau.shape[0] < 2:
            return
        order = np.argsort(-np.where(np.isfinite(tau), tau, -np.inf), axis=0)
        tau[...] = np.take_along_axis(tau, order, axis=0)
        amplitude[...] = np.take_along_axis(amplitude, order, axis=0)

    @staticmethod
    def _ptr(array: np.ndarray) -> _DOUBLE_PTR:
        return array.ctypes.data_as(_DOUBLE_PTR)

    @staticmethod
    def _raise_on_wrapper_error(code: int, operation: str) -> None:
        if code == 0:
            return
        meanings = {
            -10: "invalid dimensions or bin width",
            -11: "pixel-count overflow",
            -12: "decay-length overflow",
            -13: "component-length overflow",
            -14: "a required pointer was null",
            -15: "component-length overflow (stacked images)",
            -20: "decay buffer was too short",
            -21: "IRF buffer was too short",
            -22: "mask buffer was too short",
            -23: "global lifetime output was too short",
            -24: "global statistics output was too short",
            -30: "lifetime component output was too short",
            -31: "amplitude component output was too short",
            -32: "mean lifetime output was too short",
            -33: "background output was too short",
            -34: "chi-square output was too short",
            -35: "time-shift output was too short",
            -36: "fit-start output was too short",
            -37: "fit-stop output was too short",
            -38: "exit-flag output was too short",
            -39: "status output was too short",
            -40: "photon-count output was too short",
            -41: "global IRF was invalid",
            -42: "too few selected pixels for the requested model",
            -50: "native memory allocation failed",
            -60: "global lifetime search could not obtain a valid fit",
        }
        detail = meanings.get(code, "native backend error")
        raise NativeBackendError(f"{operation} failed ({code}: {detail}).")

    def fit_pixels(
        self,
        decay: np.ndarray,
        irf: np.ndarray,
        *,
        bin_width_ns: float,
        model_order: int = 1,
        background_mode: int = 0,
        background_value: float = 0.0,
        noise_model: int = 0,
        min_photons: float = 0.0,
        irf_mode: int = 0,
    ) -> NativeFitResult:
        order = max(1, min(int(model_order), 3))
        if not np.isfinite(bin_width_ns) or float(bin_width_ns) <= 0:
            raise NativeBackendError("Time-bin width must be finite and positive.")
        if int(background_mode) not in (0, 1) or int(noise_model) not in (0, 1, 2, 3):
            raise NativeBackendError("Invalid native background or noise mode.")
        if int(irf_mode) not in (0, 1):
            raise NativeBackendError("IRF mode must be global (0) or per-pixel (1).")
        if not np.isfinite(background_value) or not np.isfinite(min_photons) or float(min_photons) < 0:
            raise NativeBackendError("Background and photon threshold must be finite; threshold cannot be negative.")
        if int(noise_model) == 3 and self.sha256 != _PATCHED_DLL_SHA256:
            raise NativeBackendError(
                "Constant weighting is disabled: this DLL build passes a null sigma "
                "buffer to FLIMLib and could terminate Python."
            )
        mode = 1 if int(irf_mode) == 1 else 0
        cube, reference, rows, columns, n_bins = self._inputs(decay, irf, mode)
        tau, tau_mean, amplitude, background, chi, status, photons = self._output_arrays(rows, columns, order)
        time_shift = np.empty((rows, columns), dtype=np.float64)
        fit_start = np.empty((rows, columns), dtype=np.float64)
        fit_stop = np.empty((rows, columns), dtype=np.float64)
        exit_flag = np.empty((rows, columns), dtype=np.float64)
        component_length = int(tau.size)
        map_length = rows * columns

        with _DLL_LOCK:
            code = int(
                self._dll.flim_fit_pixels_v2(
                self._ptr(cube), self._ptr(reference),
                rows, columns, n_bins, int(cube.size), int(reference.size),
                float(bin_width_ns), order, int(background_mode), float(background_value),
                int(noise_model), float(min_photons), mode,
                self._ptr(tau), component_length,
                self._ptr(tau_mean), map_length,
                self._ptr(amplitude), component_length,
                self._ptr(background), map_length,
                self._ptr(chi), map_length,
                self._ptr(time_shift), map_length,
                self._ptr(fit_start), map_length,
                self._ptr(fit_stop), map_length,
                self._ptr(exit_flag), map_length,
                self._ptr(status), map_length,
                self._ptr(photons), map_length,
                )
            )
        self._raise_on_wrapper_error(code, "Per-pixel FLIMLib fit")
        self._sort_components(tau, amplitude)
        return NativeFitResult(
            tau_components_ns=tau,
            tau_mean_ns=tau_mean,
            amplitude_components=amplitude,
            background=background,
            chi_square=chi,
            fit_status=status,
            photon_counts=photons,
            time_shift_ns=time_shift,
            fit_start_bin=fit_start,
            fit_stop_bin=fit_stop,
            exit_flag=exit_flag,
            dll_path=self.path,
            dll_sha256=self.sha256,
            wrapper_code=code,
            metadata={"mode": "per_pixel", "model_order": order, "irf_mode": mode},
        )

    def fit_global(
        self,
        decay: np.ndarray,
        irf: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        bin_width_ns: float,
        model_order: int = 1,
        fit_type: int = 0,
        background_mode: int = 0,
        background_value: float = 0.0,
        noise_model: int = 0,
        min_photons: float = 0.0,
        fit_start_bin: int = 0,
        fit_stop_bin: int = 0,
        max_global_pixels: int = 20_000,
        irf_mode: int = 0,
        global_iterations: int = 4,
        initial_taus_ns: tuple[float, ...] = (),
    ) -> NativeFitResult:
        stretched = int(fit_type) == 1
        multi_stretched = int(fit_type) == 2
        # Single-component stretched-exp is always exactly one distributed
        # component (Z, A, tau, h) -- see native/flimfit_labview.c's own
        # forcing of this, mirrored here so tau/amplitude output maps get
        # allocated the right shape regardless of whatever model_order
        # happens to be. Multi-stretched-exp (up to 3 INDEPENDENTLY
        # distributed components, FIT_GLOBAL_MULTI_STRETCHEDEXP) respects
        # model_order like plain multiexp does.
        order = 1 if stretched else max(1, min(int(model_order), 3))
        shared_count = 2 if stretched else (2 * order if multi_stretched else order)
        if not np.isfinite(bin_width_ns) or float(bin_width_ns) <= 0:
            raise NativeBackendError("Time-bin width must be finite and positive.")
        if int(background_mode) not in (0, 1) or int(noise_model) not in (0, 1, 2, 3):
            raise NativeBackendError("Invalid native background or noise mode.")
        if int(irf_mode) not in (0, 1):
            raise NativeBackendError("IRF mode must be global (0) or per-pixel (1).")
        if not np.isfinite(background_value) or not np.isfinite(min_photons) or float(min_photons) < 0:
            raise NativeBackendError("Background and photon threshold must be finite; threshold cannot be negative.")
        if int(noise_model) == 3 and self.sha256 != _PATCHED_DLL_SHA256:
            raise NativeBackendError(
                "Constant weighting is disabled: this DLL build passes a null sigma "
                "buffer to FLIMLib and could terminate Python."
            )
        mode = 1 if int(irf_mode) == 1 else 0
        cube, reference, rows, columns, n_bins = self._inputs(decay, irf, mode)
        if mask is None:
            mask_array = np.ones((rows, columns), dtype=np.float64)
        else:
            raw_mask = np.asarray(mask)
            if raw_mask.shape != (rows, columns):
                raise NativeBackendError(
                    f"Global-fit mask must have shape {(rows, columns)}, got {raw_mask.shape}."
                )
            mask_array = np.ascontiguousarray(np.isfinite(raw_mask) & (raw_mask > 0), dtype=np.float64)
            if mask_array.shape != (rows, columns):
                raise NativeBackendError(
                    f"Global-fit mask must have shape {(rows, columns)}, got {mask_array.shape}."
                )
        tau, tau_mean, amplitude, background, chi, status, photons = self._output_arrays(rows, columns, order)
        global_taus = np.empty(shared_count, dtype=np.float64)
        global_stats = np.empty(10, dtype=np.float64)
        # Every h slot (single- or multi-stretched) is meaningful to seed
        # too -- only truncate to `order` (one seed per component, no h)
        # for plain multiexp.
        initial = np.ascontiguousarray(
            initial_taus_ns[:shared_count] if (stretched or multi_stretched) else initial_taus_ns[:order],
            dtype=np.float64,
        )
        if initial.size and (not np.isfinite(initial).all() or np.any(initial <= 0)):
            raise NativeBackendError("Initial lifetimes must be finite and positive.")
        start_value, stop_value = int(fit_start_bin), int(fit_stop_bin)
        if (start_value == 0) != (stop_value == 0):
            raise NativeBackendError("Use zero for both global fit bounds (auto), or supply both bounds.")
        if start_value and not (1 <= start_value < stop_value <= n_bins):
            raise NativeBackendError("Global fit bounds must be 1-based, inclusive, and inside the decay.")
        initial_pointer = self._ptr(initial) if initial.size else None
        component_length = int(tau.size)
        map_length = rows * columns

        with _DLL_LOCK:
            code = int(
                self._dll.flim_fit_global_pixels_v2(
                self._ptr(cube), self._ptr(reference), self._ptr(mask_array),
                rows, columns, n_bins, int(cube.size), int(reference.size), int(mask_array.size),
                float(bin_width_ns), order, int(fit_type), int(background_mode), float(background_value),
                int(noise_model), float(min_photons),
                int(fit_start_bin), int(fit_stop_bin), int(max_global_pixels), mode,
                int(global_iterations), initial_pointer, int(initial.size),
                self._ptr(global_taus), int(global_taus.size),
                self._ptr(global_stats), int(global_stats.size),
                self._ptr(tau), component_length,
                self._ptr(tau_mean), map_length,
                self._ptr(amplitude), component_length,
                self._ptr(background), map_length,
                self._ptr(chi), map_length,
                self._ptr(status), map_length,
                self._ptr(photons), map_length,
                )
            )
        self._raise_on_wrapper_error(code, "Global FLIMLib fit")
        self._sort_components(tau, amplitude)
        if stretched or multi_stretched:
            # global_taus interleaves tau/h pairs here -- sorting would
            # nonsensically rank an h value alongside a lifetime;
            # multiexp's descending sort (several genuinely comparable
            # lifetimes, nothing else in the vector) doesn't apply.
            pass
        else:
            global_taus.sort()
            global_taus[:] = global_taus[::-1]
        return NativeFitResult(
            tau_components_ns=tau,
            tau_mean_ns=tau_mean,
            amplitude_components=amplitude,
            background=background,
            chi_square=chi,
            fit_status=status,
            photon_counts=photons,
            global_taus_ns=global_taus,
            global_stats=global_stats,
            dll_path=self.path,
            dll_sha256=self.sha256,
            wrapper_code=code,
            metadata={"mode": "global", "model_order": order, "irf_mode": mode, "fit_type": int(fit_type)},
        )

    def _require_global_model_functions(self, operation: str) -> None:
        if not getattr(self, "_has_global_model_functions", False):
            raise NativeBackendError(
                f"{operation} requires flim_train_global_model_v1/flim_apply_global_model_v1, "
                "which this DLL build does not export. Use a DLL built from the current "
                "native/flimfit_labview.c source."
            )

    def train_global_model(
        self,
        decay: np.ndarray,
        irf: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        bin_width_ns: float,
        model_order: int = 2,
        fit_type: int = 0,
        background_mode: int = 0,
        background_value: float = 0.0,
        noise_model: int = 0,
        min_photons: float = 0.0,
        fit_start_bin: int = 0,
        fit_stop_bin: int = 0,
        max_global_pixels: int = 20_000,
        irf_mode: int = 1,
        global_iterations: int = 8,
        initial_taus_ns: tuple[float, ...] = (),
    ) -> GlobalTrainResult:
        """Train one shared lifetime model across a stack of training images.

        ``decay``/``irf``/``mask`` are ``(image_count, rows, columns[, time])``
        stacks -- see ``_stacked_inputs``. Only ``fit_status`` and
        ``photon_counts`` come back per-pixel; the trained model itself is
        just ``global_taus_ns`` (length ``model_order``, or ``[tau, h]``
        when ``fit_type`` selects the stretched-exponential distribution
        model) plus diagnostic ``global_stats``.
        """

        self._require_global_model_functions("Training a global model")
        stretched = int(fit_type) == 1
        multi_stretched = int(fit_type) == 2
        order = 1 if stretched else max(1, min(int(model_order), 3))
        shared_count = 2 if stretched else (2 * order if multi_stretched else order)
        if not np.isfinite(bin_width_ns) or float(bin_width_ns) <= 0:
            raise NativeBackendError("Time-bin width must be finite and positive.")
        if int(background_mode) not in (0, 1) or int(noise_model) not in (0, 1, 2, 3):
            raise NativeBackendError("Invalid native background or noise mode.")
        if int(irf_mode) not in (0, 1):
            raise NativeBackendError("IRF mode must be global (0) or per-pixel-per-image (1).")
        if not np.isfinite(background_value) or not np.isfinite(min_photons) or float(min_photons) < 0:
            raise NativeBackendError("Background and photon threshold must be finite; threshold cannot be negative.")
        if int(noise_model) == 3 and self.sha256 != _PATCHED_DLL_SHA256:
            raise NativeBackendError(
                "Constant weighting is disabled: this DLL build passes a null sigma "
                "buffer to FLIMLib and could terminate Python."
            )
        mode = 1 if int(irf_mode) == 1 else 0
        cube, reference, mask_array, image_count, rows, columns, n_bins = self._stacked_inputs(
            decay, irf, mask, mode
        )
        total_pixels = image_count * rows * columns
        initial = np.ascontiguousarray(
            initial_taus_ns[:shared_count] if (stretched or multi_stretched) else initial_taus_ns[:order],
            dtype=np.float64,
        )
        if initial.size and (not np.isfinite(initial).all() or np.any(initial <= 0)):
            raise NativeBackendError("Initial lifetimes must be finite and positive.")
        start_value, stop_value = int(fit_start_bin), int(fit_stop_bin)
        if (start_value == 0) != (stop_value == 0):
            raise NativeBackendError("Use zero for both fit bounds (auto), or supply both bounds.")
        if start_value and not (1 <= start_value < stop_value <= n_bins):
            raise NativeBackendError("Fit bounds must be 1-based, inclusive, and inside the decay.")
        initial_pointer = self._ptr(initial) if initial.size else None

        global_taus = np.empty(shared_count, dtype=np.float64)
        global_stats = np.empty(10, dtype=np.float64)
        fit_status = np.empty((image_count, rows, columns), dtype=np.float64)
        photon_counts = np.empty((image_count, rows, columns), dtype=np.float64)

        with _DLL_LOCK:
            code = int(
                self._dll.flim_train_global_model_v1(
                self._ptr(cube), self._ptr(reference), self._ptr(mask_array),
                image_count, rows, columns, n_bins,
                int(cube.size), int(reference.size), int(mask_array.size),
                float(bin_width_ns), order, int(fit_type), int(background_mode), float(background_value),
                int(noise_model), float(min_photons),
                int(fit_start_bin), int(fit_stop_bin), int(max_global_pixels),
                mode, int(global_iterations),
                initial_pointer, int(initial.size),
                self._ptr(global_taus), int(global_taus.size),
                self._ptr(global_stats), int(global_stats.size),
                self._ptr(fit_status), total_pixels,
                self._ptr(photon_counts), total_pixels,
                )
            )
        self._raise_on_wrapper_error(code, "Global model training")
        # Unlike fit_global, the DLL's training routine does not sort its
        # output taus (they start at fixed seed positions and move
        # independently via coordinate-descent, so they can end up in any
        # order). Sort here for the same longest-to-shortest presentation
        # used everywhere else in the app -- except for (multi-)stretched-
        # exp, where global_taus interleaves tau/h pairs and sorting would
        # nonsensically rank an h value alongside a lifetime.
        if not (stretched or multi_stretched):
            global_taus.sort()
            global_taus[:] = global_taus[::-1]
        return GlobalTrainResult(
            global_taus_ns=global_taus,
            global_stats=global_stats,
            fit_status=fit_status,
            photon_counts=photon_counts,
            dll_path=self.path,
            dll_sha256=self.sha256,
            wrapper_code=code,
            metadata={
                "mode": "train_global",
                "model_order": order,
                "irf_mode": mode,
                "image_count": image_count,
                "fit_type": int(fit_type),
            },
        )

    def apply_global_model(
        self,
        decay: np.ndarray,
        irf: np.ndarray,
        fixed_taus_ns: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        bin_width_ns: float,
        model_order: int = 2,
        background_mode: int = 0,
        background_value: float = 0.0,
        noise_model: int = 0,
        min_photons: float = 0.0,
        fit_start_bin: int = 0,
        fit_stop_bin: int = 0,
        irf_mode: int = 1,
    ) -> NativeFitResult:
        """Apply a trained shared-lifetime model to a stack of one or more images.

        Only amplitude and background are free per pixel; ``fixed_taus_ns``
        (length ``model_order``) is held constant. When ``decay`` carries a
        single image (``image_count == 1``) the returned maps are squeezed
        back to plain ``(rows, columns)``/``(order, rows, columns)`` so this
        drops straight into the rest of the single-image pipeline.
        """

        self._require_global_model_functions("Applying a trained global model")
        order = max(1, min(int(model_order), 3))
        if not np.isfinite(bin_width_ns) or float(bin_width_ns) <= 0:
            raise NativeBackendError("Time-bin width must be finite and positive.")
        if int(background_mode) not in (0, 1) or int(noise_model) not in (0, 1, 2, 3):
            raise NativeBackendError("Invalid native background or noise mode.")
        if int(irf_mode) not in (0, 1):
            raise NativeBackendError("IRF mode must be global (0) or per-pixel-per-image (1).")
        if not np.isfinite(background_value) or not np.isfinite(min_photons) or float(min_photons) < 0:
            raise NativeBackendError("Background and photon threshold must be finite; threshold cannot be negative.")
        if int(noise_model) == 3 and self.sha256 != _PATCHED_DLL_SHA256:
            raise NativeBackendError(
                "Constant weighting is disabled: this DLL build passes a null sigma "
                "buffer to FLIMLib and could terminate Python."
            )
        mode = 1 if int(irf_mode) == 1 else 0
        cube, reference, mask_array, image_count, rows, columns, n_bins = self._stacked_inputs(
            decay, irf, mask, mode
        )
        total_pixels = image_count * rows * columns
        fixed = np.ascontiguousarray(fixed_taus_ns, dtype=np.float64)[:order]
        if fixed.size < order or not np.isfinite(fixed).all() or np.any(fixed <= 0):
            raise NativeBackendError("Fixed lifetimes must supply one finite, positive value per model component.")

        tau = np.empty((order, image_count, rows, columns), dtype=np.float64)
        amplitude = np.empty((order, image_count, rows, columns), dtype=np.float64)
        tau_mean = np.empty((image_count, rows, columns), dtype=np.float64)
        background = np.empty((image_count, rows, columns), dtype=np.float64)
        chi = np.empty((image_count, rows, columns), dtype=np.float64)
        status = np.empty((image_count, rows, columns), dtype=np.float64)
        photons = np.empty((image_count, rows, columns), dtype=np.float64)
        component_length = int(tau.size)

        with _DLL_LOCK:
            code = int(
                self._dll.flim_apply_global_model_v1(
                self._ptr(cube), self._ptr(reference), self._ptr(mask_array),
                image_count, rows, columns, n_bins,
                int(cube.size), int(reference.size), int(mask_array.size),
                float(bin_width_ns), order, int(background_mode), float(background_value),
                int(noise_model), float(min_photons),
                int(fit_start_bin), int(fit_stop_bin), mode,
                self._ptr(fixed), int(fixed.size),
                self._ptr(tau), component_length,
                self._ptr(tau_mean), total_pixels,
                self._ptr(amplitude), component_length,
                self._ptr(background), total_pixels,
                self._ptr(chi), total_pixels,
                self._ptr(status), total_pixels,
                self._ptr(photons), total_pixels,
                )
            )
        self._raise_on_wrapper_error(code, "Applying trained global model")
        if image_count == 1:
            tau = tau[:, 0]
            amplitude = amplitude[:, 0]
            tau_mean = tau_mean[0]
            background = background[0]
            chi = chi[0]
            status = status[0]
            photons = photons[0]
        return NativeFitResult(
            tau_components_ns=tau,
            tau_mean_ns=tau_mean,
            amplitude_components=amplitude,
            background=background,
            chi_square=chi,
            fit_status=status,
            photon_counts=photons,
            global_taus_ns=fixed,
            dll_path=self.path,
            dll_sha256=self.sha256,
            wrapper_code=code,
            metadata={
                "mode": "apply_global",
                "model_order": order,
                "irf_mode": mode,
                "image_count": image_count,
            },
        )


def _run_native_subprocess(
    mode: str,
    payload: dict[str, np.ndarray],
    *,
    dll_path: str | Path | None,
    timeout_seconds: float,
) -> tuple[dict[str, np.ndarray], dict, Path]:
    """Write ``payload``, run the native worker subprocess, return its raw output.

    Shared plumbing for every ``*_in_subprocess`` wrapper below; each caller
    only knows how to unpack its own result dataclass from the returned
    ``arrays`` dict. The short-lived NPZ files keep the main GUI process
    insulated from native access violations while preserving exact float64
    array layouts -- this app is validated against LabVIEW/MATLAB to
    double-precision parity in places, so the transport arrays stay float64
    rather than trading any numerical fidelity for speed.
    """

    resolved = find_native_dll(dll_path)
    if resolved is None:
        raise NativeBackendError("The FLIMLib fitting DLL was not found.")
    if is_frozen():
        # Frozen (PyInstaller) build: there is no standalone native_worker.py
        # file on disk to invoke -- sys.executable *is* this app's own exe,
        # which answers to a hidden --native-worker flag instead of
        # relaunching the GUI (see run_flim_studio.py). Both branches end up
        # invoking a fresh, isolated OS process either way.
        command_prefix = [sys.executable, "--native-worker"]
    else:
        worker = Path(__file__).with_name("native_worker.py")
        if not worker.is_file():
            raise NativeBackendError(f"Native helper process is missing: {worker}")
        command_prefix = [sys.executable, str(worker)]

    with tempfile.TemporaryDirectory(prefix="flim-studio-native-") as directory:
        folder = Path(directory)
        input_path = folder / "input.npz"
        output_path = folder / "output.npz"
        # Uncompressed: these files live seconds in a temp directory and are
        # never persisted, so paying zlib's CPU cost buys nothing here. This
        # does not change any value, only how the bytes are packed.
        write_start = time.perf_counter()
        np.savez(input_path, **payload)
        write_elapsed = time.perf_counter() - write_start
        command = command_prefix + [mode, str(resolved), str(input_path), str(output_path)]
        subprocess_start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(float(timeout_seconds), 1.0),
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise NativeBackendError(
                f"Native {mode} fit exceeded {timeout_seconds:g} seconds and was terminated."
            ) from exc
        subprocess_elapsed = time.perf_counter() - subprocess_start
        # The worker prints its own internal timing breakdown to stderr on
        # every run (see native_worker.py). Surface it here regardless of
        # outcome: `subprocess_elapsed` minus the worker's own
        # `worker_total_s` is process-launch/interpreter-startup overhead
        # (this is where AV/EDR real-time scanning shows up, if present) --
        # something LabVIEW's in-process DLL call never pays.
        for line in (completed.stderr or "").splitlines():
            if line.startswith("NATIVE_WORKER_TIMING"):
                print(
                    f"[flim-studio] npz_write_s={write_elapsed:.3f} "
                    f"subprocess_wall_s={subprocess_elapsed:.3f} {line}",
                    file=sys.stderr,
                )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
            raise NativeBackendError(
                f"Native {mode} helper failed with exit code {completed.returncode}: {detail[-2000:]}"
            )
        if not output_path.is_file():
            raise NativeBackendError("Native helper exited without producing a result.")
        read_start = time.perf_counter()
        with np.load(output_path, allow_pickle=False) as archive:
            error = str(archive["error"].item()) if "error" in archive else ""
            if error:
                raise NativeBackendError(error)
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
                if name not in {"metadata_json", "error"}
            }
        read_elapsed = time.perf_counter() - read_start
        print(f"[flim-studio] npz_read_s={read_elapsed:.3f}", file=sys.stderr)
        return arrays, metadata, resolved


def fit_in_subprocess(
    mode: str,
    decay: np.ndarray,
    irf: np.ndarray,
    *,
    dll_path: str | Path | None = None,
    mask: np.ndarray | None = None,
    timeout_seconds: float = 900.0,
    **options: object,
) -> NativeFitResult:
    """Run one native per-pixel/global fit in a disposable helper process."""

    if mode not in {"pixels", "global"}:
        raise NativeBackendError(f"Unknown subprocess fit mode: {mode}")
    payload: dict[str, np.ndarray] = {
        "decay": np.asarray(decay, dtype=np.float64),
        "irf": np.asarray(irf, dtype=np.float64),
        "options_json": np.array(json.dumps(options)),
    }
    if mask is not None:
        payload["mask"] = np.asarray(mask, dtype=np.float64)
    arrays, metadata, resolved = _run_native_subprocess(
        mode, payload, dll_path=dll_path, timeout_seconds=timeout_seconds
    )

    def optional(name: str) -> np.ndarray | None:
        return arrays.get(name)

    return NativeFitResult(
        tau_components_ns=arrays["tau_components_ns"],
        tau_mean_ns=arrays["tau_mean_ns"],
        amplitude_components=arrays["amplitude_components"],
        background=arrays["fit_background"],
        chi_square=arrays["chi_square_native"],
        fit_status=arrays["fit_status"],
        photon_counts=arrays["photon_counts_native"],
        time_shift_ns=optional("time_shift_ns"),
        fit_start_bin=optional("fit_start_bin"),
        fit_stop_bin=optional("fit_stop_bin"),
        exit_flag=optional("exit_flag"),
        global_taus_ns=optional("global_taus_ns"),
        global_stats=optional("global_stats"),
        dll_path=resolved,
        dll_sha256=str(metadata["dll_sha256"]),
        wrapper_code=int(metadata["wrapper_code"]),
        metadata=dict(metadata["fit_metadata"]),
    )


def fit_pixels_parallel_in_subprocess(
    decay: np.ndarray,
    irf: np.ndarray,
    *,
    dll_path: str | Path | None = None,
    worker_count: int = 1,
    timeout_seconds: float = 900.0,
    **options: object,
) -> NativeFitResult:
    """Like ``fit_in_subprocess("pixels", ...)``, split across ``worker_count``
    row-band chunks, each fitted in its own disposable subprocess concurrently.

    This is exact, not an approximation: ``flim_fit_pixels_v2`` fits every
    pixel independently with no cross-pixel state -- unlike the *global*
    fit modes (one tau shared/jointly fitted across every selected pixel),
    which this function deliberately does not handle. Chunking a global fit
    the same way would silently compute a different tau per chunk instead
    of the one true whole-image joint fit, so callers must keep using plain
    ``fit_in_subprocess("global", ...)``/``train_global_in_subprocess`` for
    those.

    Each chunk still runs in its own disposable OS process with its own
    independent DLL load -- exactly like a single unchunked fit already
    does (see native_worker.py) -- this only changes how many run at once,
    not the isolation model. That also sidesteps ``_DLL_LOCK`` entirely:
    that lock only ever matters for concurrent calls into *one* loaded DLL
    instance inside a single process, and every chunk here gets a
    completely separate process and DLL instance, so there is nothing
    shared between them to protect. Threads (not more subprocesses) drive
    the fan-out below purely because each one only ever sits blocked on its
    own ``subprocess.run()`` -- CPython releases the GIL for that wait, so
    this genuinely runs ``worker_count`` fits in parallel across CPU cores
    despite being "just" a thread pool.
    """

    cube = np.asarray(decay, dtype=np.float64)
    rows = int(cube.shape[0]) if cube.ndim == 3 else 0
    workers = max(1, min(int(worker_count), rows)) if rows else 1
    if workers <= 1:
        return fit_in_subprocess(
            "pixels", decay, irf, dll_path=dll_path, timeout_seconds=timeout_seconds, **options
        )

    reference = np.asarray(irf, dtype=np.float64)
    # A per-pixel IRF cube must be sliced in lockstep with the decay data;
    # a shared trace (1-D) or a mismatched-shape cube is instead handed to
    # every chunk unchanged, letting each chunk's own fit_pixels()/_inputs()
    # reduce/broadcast it exactly as the single, unchunked call already
    # does today -- same behaviour, just repeated per chunk.
    per_pixel_irf = reference.ndim == 3 and reference.shape[:2] == cube.shape[:2]
    # Plain integer arithmetic (not np.linspace + float truncation) so the
    # boundaries are exactly correct with no floating-point risk of two
    # adjacent boundaries colliding into an empty chunk: rows is split as
    # evenly as possible, with the first `extra` chunks getting one extra
    # row each. workers <= rows (clamped above) guarantees every chunk gets
    # at least `base` >= 1 row.
    base, extra = divmod(rows, workers)
    boundaries = [0]
    for index in range(workers):
        boundaries.append(boundaries[-1] + base + (1 if index < extra else 0))

    def fit_chunk(index: int) -> NativeFitResult:
        start, stop = boundaries[index], boundaries[index + 1]
        chunk_irf = reference[start:stop] if per_pixel_irf else reference
        return fit_in_subprocess(
            "pixels", cube[start:stop], chunk_irf,
            dll_path=dll_path, timeout_seconds=timeout_seconds, **options,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunk_results = list(pool.map(fit_chunk, range(workers)))

    def stack(field_name: str, axis: int) -> np.ndarray | None:
        values = [getattr(result, field_name) for result in chunk_results]
        if any(value is None for value in values):
            return None
        return np.concatenate(values, axis=axis)

    first = chunk_results[0]
    merged_metadata = dict(first.metadata)
    merged_metadata["chunked_fit"] = {"workers": workers, "row_boundaries": boundaries}
    return NativeFitResult(
        tau_components_ns=stack("tau_components_ns", axis=1),
        tau_mean_ns=stack("tau_mean_ns", axis=0),
        amplitude_components=stack("amplitude_components", axis=1),
        background=stack("background", axis=0),
        chi_square=stack("chi_square", axis=0),
        fit_status=stack("fit_status", axis=0),
        photon_counts=stack("photon_counts", axis=0),
        time_shift_ns=stack("time_shift_ns", axis=0),
        fit_start_bin=stack("fit_start_bin", axis=0),
        fit_stop_bin=stack("fit_stop_bin", axis=0),
        exit_flag=stack("exit_flag", axis=0),
        dll_path=first.dll_path,
        dll_sha256=first.dll_sha256,
        wrapper_code=first.wrapper_code,
        metadata=merged_metadata,
    )


def train_global_in_subprocess(
    decay: np.ndarray,
    irf: np.ndarray,
    mask: np.ndarray,
    *,
    dll_path: str | Path | None = None,
    timeout_seconds: float = 1800.0,
    **options: object,
) -> GlobalTrainResult:
    """Run ``flim_train_global_model_v1`` on a stacked training set in a helper process.

    ``decay``/``irf`` are ``(image_count, rows, columns[, time])`` stacks and
    ``mask`` is ``(image_count, rows, columns)`` -- see
    ``FLIMLibLabViewBackend._stacked_inputs``. Training a global model over
    many large images can run much longer than a single-image fit, hence the
    longer default timeout.
    """

    payload: dict[str, np.ndarray] = {
        "decay": np.asarray(decay, dtype=np.float64),
        "irf": np.asarray(irf, dtype=np.float64),
        "mask": np.asarray(mask, dtype=np.float64),
        "options_json": np.array(json.dumps(options)),
    }
    arrays, metadata, resolved = _run_native_subprocess(
        "train_global", payload, dll_path=dll_path, timeout_seconds=timeout_seconds
    )
    return GlobalTrainResult(
        global_taus_ns=arrays["global_taus_ns"],
        global_stats=arrays["global_stats"],
        fit_status=arrays["fit_status"],
        photon_counts=arrays["photon_counts_native"],
        dll_path=resolved,
        dll_sha256=str(metadata["dll_sha256"]),
        wrapper_code=int(metadata["wrapper_code"]),
        metadata=dict(metadata["fit_metadata"]),
    )


def apply_global_in_subprocess(
    decay: np.ndarray,
    irf: np.ndarray,
    mask: np.ndarray,
    fixed_taus_ns: np.ndarray,
    *,
    dll_path: str | Path | None = None,
    timeout_seconds: float = 900.0,
    **options: object,
) -> NativeFitResult:
    """Run ``flim_apply_global_model_v1`` on a stacked image set in a helper process."""

    payload: dict[str, np.ndarray] = {
        "decay": np.asarray(decay, dtype=np.float64),
        "irf": np.asarray(irf, dtype=np.float64),
        "mask": np.asarray(mask, dtype=np.float64),
        "fixed_taus": np.asarray(fixed_taus_ns, dtype=np.float64),
        "options_json": np.array(json.dumps(options)),
    }
    arrays, metadata, resolved = _run_native_subprocess(
        "apply_global", payload, dll_path=dll_path, timeout_seconds=timeout_seconds
    )
    return NativeFitResult(
        tau_components_ns=arrays["tau_components_ns"],
        tau_mean_ns=arrays["tau_mean_ns"],
        amplitude_components=arrays["amplitude_components"],
        background=arrays["fit_background"],
        chi_square=arrays["chi_square_native"],
        fit_status=arrays["fit_status"],
        photon_counts=arrays["photon_counts_native"],
        global_taus_ns=arrays.get("global_taus_ns"),
        dll_path=resolved,
        dll_sha256=str(metadata["dll_sha256"]),
        wrapper_code=int(metadata["wrapper_code"]),
        metadata=dict(metadata["fit_metadata"]),
    )
