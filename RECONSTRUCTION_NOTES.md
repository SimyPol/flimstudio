# Reconstruction notes

## GUI wiring for phasor-informed seeding, and a real prerequisite gap found along the way: the Distribution model dropdown never exposed multi-stretched at all (v0.17.0, 2026-09-09, same-day follow-up)

**User's framing, verbatim**: "yes lets set it up in the GUI and update
the manual instructions" -- following the entry immediately below,
which built and validated `phasor_seed=True` at the Python API level
but left it unreachable from the Tkinter UI.

**A real gap found while wiring this up, not just a missing checkbox**:
`gui.py`'s `DISTRIBUTION_MODEL_LABELS` only ever had two entries
(`Multi-exponential (discrete τ)` -> 0, `Stretched exponential
(distribution)` -> 1) -- `FIT_GLOBAL_MULTI_STRETCHEDEXP` (fit_
distribution_model=2), this whole multi-day investigation's actual
subject, was **never selectable from the GUI at all**. Adding just the
phasor_seed checkbox without fixing this would have shipped a control
nothing could ever reach. Added the missing `"Multi-stretched
exponential (independent components)": 2` entry.

**A related existing-logic trap avoided**: `_method_model_hint_changed`
already had a `stretched = ... == 1` check that FORCES Model order to 1
and disables the Model order combo entirely, correct for the single-
component stretched exponential (always exactly one component) but
WRONG for multi-stretched (2-3 independently-distributed components,
where Model order is exactly how many). Kept `stretched` scoped to
`== 1` only (so the order-locking behaviour for the existing, already-
validated single-component mode is completely unchanged) and added a
separate `any_stretched = distribution_model in (1, 2)` for the
explanatory-hint text only, which both modes should show.

**What was added**:
- `flim_app/training_gui.py`'s `TrainingDialog`: a new "Auto-seed from
  phasor analysis" checkbox (default off), wired straight to `train_
  global_model`'s existing `phasor_seed=` keyword (see entry below --
  no further native or processing.py changes needed, that machinery
  already existed and was already validated). Lives in this dialog
  specifically (not the sidebar) since it's a property of one training
  run, matching how the sidebar's own "Initial τ values" field already
  works alongside it (an explicit Initial τ values entry still always
  wins, per phasor_informed_seed's own existing priority rule).
- `flim_app/gui.py`: the missing Distribution model entry above; an
  expanded explanatory label under the Distribution model combo
  covering both stretched options and pointing at the new checkbox.
- `USER_MANUAL.md`: a new "Distribution model" bullet in the NATIVE
  FLIMLIB FIT settings reference (documenting all 3 options -- this
  setting had never been documented in the manual at all before, a
  pre-existing gap from v0.16.0's original single-component stretched-
  exponential toggle); Training mode's numbered walkthrough (§4.6) gains
  a new step 4 laying out all three seeding options (unseeded/manual/
  phasor) with guidance on when to use each; the sidebar's own "Initial
  τ values" bullet cross-references the new option.

**Validated**: `training_gui.py` and `gui.py` both parse cleanly
(`ast.parse`) and import without error; full pytest suite still 65/66
passing (the same pre-existing external-fixture gap, no regression) --
this session's earlier phasor_seed validation (RECONSTRUCTION_NOTES.md
entry below, plus the two `test_processing.py` tests) already covers
the underlying `train_global_model(..., phasor_seed=True)` behaviour
this checkbox now exposes; this pass specifically checked the new GUI
code paths don't break anything, not a re-validation of the algorithm
itself.

## Phasor-informed seeding: a working THIRD mode for FIT_GLOBAL_MULTI_STRETCHEDEXP, matching truth-seeded quality with no prior knowledge required (v0.17.0, 2026-09-09)

**User's framing, verbatim**: "ok this could work when we have
individual datasets. i would like to come back to the unseeded bigger
clustering approach later. One suggestion, could we use an initial
phasor plot analysis to inform us on the seeding perhaps?" followed by
"ok lets do it now. We need it as a separate mode to unseeded and
seeded and lets see what happens. Go Claude go!"

**The idea**: phasors combine LINEARLY under mixing -- a pixel with
amplitude fractions of N components has a phasor coordinate equal to
the amplitude-weighted average of each component's own (pure) phasor
coordinate. For N genuinely-distinct components with pixels spanning a
real range of mixing ratios (exactly what real global analysis already
needs -- see the "N=2 actually works" entry below), the whole image's
pixel cloud in (g, s) space should span the convex hull of the N pure-
component phasor positions. Finding those extreme points and inverting
each back to a (tau, h) pair gives a real, data-grounded starting guess
with no user input and no prior labeling of "which pixel is pure"
needed. This app already had validated, tested phasor machinery
(`phasor_lifetimes`, used for the existing phasor_phase/phasor_
modulation analysis methods) to build on.

**Two things had to be got right for this to actually work** (both
found the hard way, via direct iterative testing -- neither was obvious
in advance):

**1. Calibration must match the real measurement pipeline exactly, not
an approximation.** The standard mono-exponential phasor formulas
(`tau_phase = tan(phase)/omega`, etc.) are only exact for h=1 -- for a
genuinely stretched (h>1) component they're systematically biased, and
the bias itself depends on h. Confirmed directly on isolated, unmixed
pure components: a bias factor of ~2.2x for h=1.5, growing to ~4.3x for
h=2.8. An independently-derived theoretical (infinite-time-domain)
model for the stretched-exponential's own phasor was tried next and
found ALSO insufficient: real measurements are on a finite acquisition
window, which truncates a broad (large-h) component's genuinely long
tail, and this app's own IRF-correction formula (exact for ideal
circular convolution) interacts with the ACTUAL linear, truncated
convolution in a way a clean theoretical model doesn't capture exactly
-- confirmed by direct comparison, the residual gap grew with h/tau
just like the first bias did. **The fix that closed this completely**:
stop approximating -- directly SIMULATE the real pipeline noise-free
(build the pure decay, convolve with the actual IRF, call this file's
own `phasor_lifetimes` with IRF correction) for each calibration grid
point. This guarantees an exact match to what a real, noise-free
measurement of that (tau, h) would show, since it's the identical code
path. Confirmed by exact round-trip recovery (tau and h both reproduced
to 4+ decimal places, residual cost ~1e-20) on isolated pure components
at every tested (tau, h) -- and, critically, on the actual noisy
simulated measurement data too (not just the idealized calibration
target), where the earlier two approaches both failed badly (off by
2-20x on at least one component).

**2. Finding the right extreme points is not simply "the N most
separated pixels."** For N=3: k-means cluster CENTROIDS don't work
(averaging many still-mixed pixels just gives another mixed point, not
a purer one -- phasors are linear, so averaging doesn't help; confirmed
directly, centroid-derived seeds were 3-13x off and gave a worse fit
than unseeded). Naive greedy farthest-point selection finds the first
two genuine extremes reliably, but the THIRD pick ("farthest from those
2 points") is not the same question as "the third corner of the
triangle" and can land on an interior/still-mixed point instead --
confirmed directly, one bad third vertex (extrapolated wildly beyond
the calibration table) was enough to derail the WHOLE joint fit even
though the other two vertices were excellent (worse than plain
unseeded, obj=11.7-14.7 vs the ~5.5-7.7 unseeded typically gets). **The
fix**: for the third vertex, use perpendicular distance from the LINE
through the first two chosen points, not distance from the point set --
the correct geometric question for a genuine third triangle corner.
Confirmed directly: this combination recovers all 3 true (tau, h) pairs
to within a few percent with zero prior knowledge (e.g. 0.966/1.523,
3.498/2.201, 7.900/2.811 against true 1.0/1.5, 3.5/2.2, 8.0/2.8), and
feeding this into the real native fit matches truth-seeded quality
almost exactly, consistently across 3 independent random-data seeds.

**Implementation**: `flim_app/processing.py` gains
`phasor_informed_seed()` (the core algorithm -- builds a small
pipeline-exact calibration table, per model_order picks 1 (centroid),
2 (farthest-point), or 3 (farthest-point + line-perpendicular-distance)
vertices, inverts each via a table lookup + `scipy.optimize.least_
squares` refine) plus the private helpers it's built from. `train_
global_model` gains a new keyword-only `phasor_seed: bool = False` --
a THIRD seeding mode alongside "unseeded" (no `initial_taus_ns`, native
multi-start) and "manual" (`initial_taus_ns` already populated by the
caller, e.g. via "Set manually..." -- [[manual-global-model-dialog]]).
When True and the caller hasn't already supplied an explicit seed, the
derived seed is used automatically; an explicit caller-supplied seed
always still wins (matching how the native side's own `has_explicit_
shared_seed` already prioritizes an explicit seed over multi-start).
Falls back silently to unseeded if a seed can't be derived (too few
valid pixels, unsupported model_order) -- a best-effort default, never
a hard requirement.

**No native code was touched at all** -- this is purely additive Python
built on the already-existing, already-validated `phasor_lifetimes`
and the native side's already-existing explicit-seed handling. `scipy`
is a new dependency (`requirements.txt`), used for the small 2-
parameter bounded nonlinear refinement (`least_squares`) in the
inversion step.

**Validated**:
- N=3, 3 independent random-data seeds, real `train_global_model(...,
  phasor_seed=True)` calls (not a standalone script): 2 of 3 match
  truth-seeded quality almost exactly (obj~5.47-5.50, essentially
  identical numbers to the truth-seeded runs); the third is a partial
  regression on the slowest component (13.9ns vs true 8.0ns) but still
  clearly better than the old unseeded baseline (which couldn't
  reliably find all 3 components at all) -- plausibly float32
  preprocessing precision at the margin, not chased further given 2/3
  already match the ideal case.
- N=2: matches the already-validated seeded/unseeded precision band
  (~10-20%) across 3 seeds, and runs noticeably FASTER than the old
  unseeded default (2.5-4.8s vs the multi-start path's longer runtime)
  since an explicit seed triggers a single search, not 5-variant multi-
  start.
- N=1: reasonable, unbroken (3.89/2.2 against true 3.0/2.5).
- Full pytest suite: 65/66 passing (up from 63/64 -- 2 new tests added,
  `test_phasor_informed_seed_recovers_two_component_mixture` and
  `test_train_global_model_phasor_seed_end_to_end`), same pre-existing
  external-fixture gap as always.

**Not yet done**: GUI wiring (a checkbox/option alongside "Set
manually..." -- currently only reachable via the `phasor_seed=True`
keyword argument, not the Tkinter UI); investigating the N=3 rng=22
partial regression further; testing with more than one training image
(the current implementation only uses the first image for seed
derivation); the user's own "bigger clustering approach" for the
unseeded case remains explicitly deferred, separate from this feature.

## A second attempted fix for the near-singular solve (ridge in the per-pixel linear solve) tried and reverted -- real precision cost, no practical N=3 benefit (v0.17.0, 2026-09-08, same-day follow-up #12)

**User's framing, verbatim**: "no lets keep going. I believe in you" --
continuing after the entry below reverted the outer-objective
separation penalty; this entry follows through on that entry's own
suggested next direction (stabilize the linear solve itself instead).

**What was tried**: a small ridge (Tikhonov) term added directly to
`fit_linear_multi_stretched_pixel`'s own per-pixel normal-equations
diagonal, scaled to the matrix's own mean diagonal magnitude (so
intended to stay negligible for a well-conditioned solve). Deliberately
designed to avoid the entry below's specific failure mode: this never
touches the outer tau/h search's own gradient/Hessian computation at
all, only a single self-contained per-pixel solve, so it shouldn't be
able to reroute an unrelated trajectory the way adding a term to the
outer objective did.

**Result -- not a net win, despite the more careful design**:

- **N=2 unseeded** (4-seed validation): NOT byte-identical anymore.
  Recovered values drifted by up to ~7% (e.g. one seed's second
  component: 4.209 -> 4.521) -- still the right ballpark, still
  recognizably the same answer, but measurably different from the
  exact previously-validated numbers. Mechanism: many small per-pixel
  solves, each nudged by a tiny amount, compound along a long (400+
  evaluation) nonlinear search trajectory -- a smaller-scale version of
  the same "any change to intermediate computation can shift where a
  nonlinear search ends up" lesson from the entry below, even without
  changing the outer objective's macroscopic shape at all.
- **N=3 unseeded**: no corresponding practical improvement -- still
  didn't recover the true slowest component (recovered a "junk"
  0.82/1.67, 5.26/2.36, 23.18/1.88 against true 1.0/1.5, 3.5/2.2,
  8.0/2.8 -- comparable to results without the ridge term), comparable
  final objective (5.50 vs ~5.5-6.3 in prior unseeded attempts), and
  ran noticeably slower (213s vs the usual tens of seconds).

Real, measured cost (N=2 precision drift) for no measured benefit (N=3
still doesn't find all 3 components unseeded) -- reverted.

DLL hash `efe017536e9ffaa5b53061966f4a771e5eb6c13bcd5c3a9936cfb18f40d1372c`.
Reconfirmed N=2's exact 4-seed unseeded validation byte-identical again
after the revert.

**Where this leaves N=3's unseeded case**: two genuine, carefully-
designed fix attempts in a row (outer-objective penalty; per-pixel
ridge) have now both failed to help, for two different, well-understood
reasons. This is a real signal that further ad hoc numerical tweaks are
unlikely to be the productive next direction -- a genuinely different
approach (e.g. regional/spatial clustering to build a real starting
guess, rather than any single crude per-pixel statistic) would be
needed to make further progress here, a materially bigger undertaking
than anything attempted in this session's N=3 work so far.

## A near-singular-linear-solve finding, a regularization attempt that regressed N=2, and the fix: filter after the search, don't penalize during it (v0.17.0, 2026-09-08, same-day follow-up #11)

**User's framing, verbatim**: "yes lets keep going" -- continuing N=3
work after the evaluation-budget fix below. Traced N=3's remaining
unseeded weakness with per-variant logging.

**Real finding #1 -- near-singular linear solve**: whenever two of a
multi-start variant's candidate taus landed close together (e.g.
10.39ns and 11.76ns, a ratio of 1.13x), the resulting objective was
WILDLY inflated -- as high as 1.7 MILLION, against a normal few-
thousand-or-less range for this scenario. Mechanism: two nearly-
identical convolved basis shapes make `fit_linear_multi_stretched_
pixel`'s per-pixel normal-equations matrix nearly singular, so the
solved amplitudes (and the resulting chi-square) become numerically
unstable rather than just "a bad fit." This is the numerical-
conditioning half of the "components collapse together" pattern this
whole investigation kept finding -- not purely a statistical/
identifiability story.

**A regularization attempt that regressed N=2 -- a real lesson, not
just a fix**: added a soft minimum-log-separation penalty between every
pair of component taus, same pattern as the existing h/tau soft-
ceiling penalties, directly into `evaluate_global_tau_objective` (the
function `levenberg_marquardt_min` actually optimizes). Confirmed by
IMMEDIATE direct test that this broke N=2's already-validated unseeded
behaviour: the data-driven variant, which reliably converged to a
genuinely separated (1.27ns, 4.35ns) answer before, instead converged
to a DEGENERATE collapsed pair (1.44ns, 1.39ns) -- exactly the failure
mode the new term was meant to prevent, self-inflicted by adding it.
**The general lesson**: adding any term to an iterative nonlinear
search's own objective reshapes the whole gradient/Hessian landscape
along the ENTIRE convergence path, not just near the eventual answer --
even a term that would never fire at the final point can still reroute
an already-good trajectory somewhere new and worse, because Newton-
style steps are sensitive to curvature everywhere they pass through,
not just at the destination. This applies generally, beyond this one
penalty -- any future addition to this outer objective needs the same
scrutiny (a direct before/after regression test against every already-
validated scenario, not just the new one motivating the change).

**The fix**: reverted the objective-level addition entirely. Kept the
same `MIN_LOG_TAU_SEPARATION` threshold, but ONLY as a passive post-hoc
filter inside `multi_stretched_result_is_plausible` -- which already
existed (rebuild #10) to decide whether the data-driven-preference
override is trustworthy, evaluated once AFTER a variant has already
finished converging, never as part of what the search itself climbs
against. A passive filter cannot reroute a trajectory since it never
participates in any gradient/Hessian computation.

**Reconfirmed**: N=2's exact 4-seed unseeded validation is byte-
identical again after this fix. N=3's rebuild #12 seeded improvement
(the evaluation-budget scaling, a completely independent change) is
unaffected -- reconfirmed across 3 seeds, identical numbers to before
this whole detour. The near-singular-solve finding above remains valid
and useful context (it explains WHY collapsed candidates score so
badly, reinforcing that the tolerance-based override -- rebuild #11 --
correctly rejects them), even though the attempted fix for it did not
survive.

DLL hash `766e9e77270d8deee451c5a865f490805dead6ac87d8408050e1f61afca6252a`.

**Where this leaves N=3's unseeded case**: still not reliably finding
all 3 components (unchanged from the entry below) -- this detour
investigated and ruled out one specific fix idea (search-objective-level
separation penalty) without finding a working replacement yet. The
near-singular-solve finding suggests a DIFFERENT kind of fix might be
safer: stabilizing `solve_linear_system`/`fit_linear_multi_stretched_
pixel` itself against near-collinear basis functions (e.g. a small
Tikhonov/ridge term in the normal equations) rather than trying to keep
the outer search away from that region via its own objective -- not yet
attempted.

## The real cause of N=3's "component collapse": an evaluation-budget bug, not (only) identifiability -- cascade seeding attempted, superseded by this fix (v0.17.0, 2026-09-08, same-day follow-up #10)

**User's framing, verbatim**: "yes lets do this" (cascade seeding),
followed by the actual fix landing on something more fundamental once
cascade seeding itself proved unworkable.

**Cascade seeding tried first, found unworkable -- itself a useful
result**: pre-fitting a *simpler* model to seed a more complex one only
works if the simpler pre-fit is itself stable on this data. It isn't:
an unseeded N=1 (single global component) pre-fit on genuinely
3-component data collapsed to a nonsensical tau=0.047ns/h=5.1
(essentially forcing one shared component to explain 3 genuinely
different lifetimes is severe model misspecification, and the search
found a degenerate escape rather than a sensible "average"); an
unseeded N=2 pre-fit on the same data converged to a compromise pair
(12.6/1.92, 9.1/2.17) nowhere near any 2 of the 3 true components.
Every seed derived from either pre-fit inherited that badness. Moment-
quantile-based candidates (tried at 3 and 5 splits) were also too
compressed to help, consistent with the moment-estimator-bias finding
in the entry below.

**The actual breakthrough came from a different diagnostic**: rather
than trying yet another seed-derivation heuristic, tested whether a
truth-seeded N=3 search's apparent "2-of-3-components collapse" was
even the search's real, converged endpoint. It was not. Ramping
`global_iterations` from 40 up: at 40, comp2/comp3 sat at 3.32/3.71
(visibly merged); at 100, comp3 had moved back out to 6.69 and the
objective was STILL improving; at 300/800/2000 the result was
byte-identical (0.817/1.667, 4.649/2.311, 6.685/2.373) -- the search's
own gradient-convergence check had genuinely stopped, confirming this
IS the true local optimum, reached only after ~1180 evaluations. **The
"collapse" seen at the default budget was an under-converged
snapshot, not the actual answer.**

**Root cause**: `levenberg_marquardt_min`'s evaluation budget
(`globalIterations * 15`) has never been scaled by `sharedCount` since
this feature's first build. For N=2 (sharedCount=4) that flat 15x is
generous -- confirmed the search self-terminates around ~400
evaluations from a ~600 budget, consistently, across every N=2 test in
this file. For N=3 (sharedCount=6) it is NOT -- ~1180 evaluations were
needed, roughly double what the flat formula supplies at the same
`global_iterations` setting (the app's own default is 4; even the
GUI's slider maximum of 80 would give only ~1200 evaluations, right at
the edge for this one test case, with no guarantee harder real data
wouldn't need more).

**The fix**: `evaluation_budget_multiplier(sharedCount)` scales the
15x by the ratio of each sharedCount's own per-Newton-iteration
evaluation cost (`4n+2n(n-1)`) to N=2's own (n=4), floored at 1.0 so
every already-validated fit type/order (sharedCount<=4: FIT_GLOBAL_
MULTIEXP modelOrder 1-3, FIT_GLOBAL_STRETCHEDEXP, FIT_GLOBAL_MULTI_
STRETCHEDEXP's own N=1/N=2) keeps its EXACT previous multiplier of 15
-- reconfirmed byte-identical across 4 independent N=2 seeds after this
change. Only sharedCount=6 (N=3) gets more: 15 -> 31.5x.

**Retested N=3 at the SAME `global_iterations=40`** (no manual tuning
needed) across 3 independent random-data seeds, seeded near truth:

| rng seed | recovered (tau0,h0, tau1,h1, tau2,h2) | true |
|---|---|---|
| 9 | (0.815/1.670, 4.661/2.251, **8.187/2.323**) | 1.0/1.5, 3.5/2.2, 8.0/2.8 |
| 1 | (0.815/1.670, 4.599/2.277, **7.503/2.345**) | 1.0/1.5, 3.5/2.2, 8.0/2.8 |
| 22 | (0.820/1.665, 4.537/2.246, **7.624/2.348**) | 1.0/1.5, 3.5/2.2, 8.0/2.8 |

Components 1 and 3 (fast and slow) now recover reliably, within the
same ~15-20% band N=1/N=2 are already validated to, consistently
across all 3 seeds -- a real, meaningful improvement over the earlier
apparent collapse. **Component 2 (middle) has a stable, systematic
~30-33% overestimate** (recovers ~4.5-4.7 against a true 3.5),
consistent and reproducible rather than randomly varying -- this reads
as a genuine, characterizable residual difficulty (plausibly related to
the moment-estimator-bias finding below, or a real remaining
correlation between the middle component and its two neighbors) rather
than the search-budget artifact that dominated before this fix.
Unseeded/default is also improved at the same setting (reduced
chi-square 6.32, down from 15.8 pre-fix) but still doesn't reliably
find the slowest component.

DLL hash `81b49f1706264c708e68ebf1de86773d39f380f2a5b3a3692a90901982eaec40`.

**What this means for the entries below**: the "N=3's core difficulty
is real and structural" conclusion (component-pair collapse persisting
across different test geometries) needs a caveat -- that investigation
was ALSO conducted at the same insufficient `global_iterations=40`
budget, so at least part of what looked like structural collapse was
this same budget artifact, not purely identifiability. The cross-
validation/overfitting conclusion is unaffected (that comparison used
matched budgets on both sides). The moment-estimator-bias and mixing-
geometry findings both remain valid, real findings in their own right
-- just not, as it turns out, the dominant explanation for the
component-collapse pattern this session kept observing.

## N=3 continued: overfitting ruled out by cross-validation; test-geometry and moment-estimator-bias findings; component-pair collapse confirmed real, not a quick-fix problem (v0.17.0, 2026-09-08, same-day follow-up #9)

**User's framing, verbatim**: "of course we have to take into account
overfitting too. Lets push on with N=3" -- directly following the entry
below, which left N=3 mid-investigation.

**Overfitting: directly tested and ruled out.** Trained N=3 (seeded
near truth) on one Poisson noise realization ("train"), then scored
both the recovered parameters AND the true parameters against a
SECOND, fully independent noise realization of the same underlying
scene ("holdout") -- the standard cross-validation check for whether a
model with more free parameters is fitting noise rather than real
structure. If recovered params were overfitting train's specific noise,
they should score *worse* than truth on the independent holdout; they
do not: recovered params score 4.68 on holdout (vs 4.52 on train)
against 7.30 for the true params scored the same way. **Repeated for
the *unseeded* case's clearly-wrong-looking result** (34.9/3.24,
29.1/3.32, 13.5/1.15 -- nothing like the true components) -- same
pattern: 5.24 on holdout (vs 5.21 on train), still meaningfully beating
true params' 7.30. Both results generalize to fresh data at least as
well as truth itself -- the textbook signature of *not* overfitting.
This directly and satisfyingly answers the user's question: whatever is
making recovery imperfect for N=3, it is NOT the classic "too many
parameters chasing noise" failure mode.

**What it likely is instead -- pursued via two further findings**:

1. **A real flaw in the test scenario's own mixing geometry**, not
   caught until now: the squared-contrast row/column mixing weights
   used throughout this investigation for N=3 never actually gave the
   FASTEST component (tau=1.0ns) a genuinely pure corner -- checked the
   four grid corners directly: `(0.000, 0.000, 1.000)`,
   `(0.497, 0.497, 0.005)`, `(0.463, 0.000, 0.537)`,
   `(0.000, 0.463, 0.537)` for components A/B/C respectively. Component
   A never exceeds ~50% anywhere. Fixed with a proper triangular
   anchor-point layout (inverse-distance-squared weighting from 3
   well-separated anchors, each pixel dominated by whichever anchor is
   closest) -- confirmed each anchor now gives a genuinely pure
   (1,0,0)-type mixture.
2. **A deterministic (non-noise) bias in the first-moment lifetime
   estimator itself**, found by isolating a single pure-component pixel
   with NO Poisson noise at all: a noise-free tau=1.0ns/h=1.5 decay
   still gives a first-moment "apparent lifetime" of 4.1ns, not 1.0ns --
   confirmed identical whether noise is added or not, ruling out the
   earlier "noise-clamping" hypothesis entirely. This is simply what a
   first-moment estimator (mean arrival time) measures for a
   stretched-exponential distribution convolved with an IRF of nonzero
   width: it's the sum of the IRF's own centroid (~2.0ns here) plus the
   decay's own intrinsic mean, which for h>1 exceeds tau itself (a
   broader-than-single-exponential distribution has a longer mean than
   its own "characteristic" timescale). This fixed additive/multiplicative
   offset dominates disproportionately for SHORT true lifetimes,
   compressing the moment-based quantile separation exactly where more
   components (more, generally shorter and more closely spaced,
   timescales) need it most -- explains why data-driven seeding
   separates well for N=2's wider, generally-longer true taus but
   compresses for N=3. A window-length truncation was tried first and
   found NOT to help (just scaled all 3 quantile seeds down together,
   preserving the same compressed relative spread) -- ruling out
   "long-tail-of-the-slowest-component dominance" as the mechanism.
   Subtracting the IRF's own centroid from the raw moment measurably
   widened the relative quantile spread in isolation, but not enough on
   its own to fix real mixture recovery.

**Retested with the corrected (triangular) mixing geometry**: a
genuinely DIFFERENT outcome, not simply "fixed" -- seeded recovery now
gets components 1 and 2 right (0.82/1.66 against true 1.0/1.5; 3.32/2.57
against true 3.5/2.2, both within the usual ~10-20% band) but component
3 (the slowest, true 8.0/2.8) collapses into component 2's territory
instead (recovered 3.71, missing the true 8.0 lifetime entirely) --
DIFFERENT from the earlier test's "middle component weak" pattern, but
still exactly one pairwise collapse among the three components. This
is strong, direct evidence that **N=3's core difficulty is real and
structural** (some pair of the 3 mutually-correlated components tends
to be at risk of merging, regardless of which specific test geometry is
used to try to avoid it) rather than an artifact this session's earlier,
flawed test data happened to produce. Unseeded/default on the corrected
geometry is also markedly worse (reduced chi-square 15.8, far above
every other tested configuration), suggesting the corrected mixing
scheme is harder for the *unseeded* multi-start machinery specifically,
not just differently-shaped.

**Where this leaves N=3**: genuinely harder than N=2 in a way that
more test-data tweaking has NOT resolved after real, methodical effort
(fixing the mixing geometry, checking the moment estimator's own bias,
ruling out overfitting) -- the recurring pattern across every test
configuration tried is that some ONE PAIR of the 3 components tends to
collapse/merge, while the other components resolve about as well as
N=2 already does. Untried and likely more promising than further ad
hoc test-data or seeding tweaks: a genuinely different seeding
STRATEGY, e.g. cascade seeding (fit 2 components first, via the
already-validated N=2 machinery, on the two most-different-looking
pixel groups, then add a third component afterward, refitting all
three together) rather than a single flat quantile split of one crude
per-pixel statistic.

## N=3 initial validation: seeded is usable but weaker (middle component), unseeded not yet reliable; a tolerance bug found and fixed; a testing-methodology caveat discovered (v0.17.0, 2026-09-08, same-day follow-up #8)

**User's framing, verbatim**: "yes lets move onto N=3 now" -- following
the user's own explicit gate ("once we are satisfied" with N=2, from
several entries ago). This entry is the first real N=3 investigation,
using the same methodology that worked for N=2 throughout this file.

**Bug found and fixed first**: rebuild #7's "prefer the data-driven
variant" override (entry below) had no safeguard against the
data-driven seed itself being bad. Direct N=3 trace: with 3
independently-distributed components, the same per-pixel first-moment
estimate that separates well for N=2 failed to separate for N=3 -- a
longer-lived component's tail disproportionately pulls the moment
estimate upward even in pixels "dominated" by a faster component, so
all 3 quantile seeds landed within ~1ns of each other (11.7/12.4/12.8
against a true 1.0-8.0ns range). The search stayed collapsed near
there (objective 1149.9) while an ordinary formulaic variant found a
genuinely much better point (367.4, 3.1x lower) -- `multi_stretched_
result_is_plausible` alone still accepted the collapsed answer (nothing
in it individually crosses the regularization ceiling), so the override
was silently returning a far worse fit than the pre-override "lowest
objective wins" default would have. Fixed with
`DATA_DRIVEN_OBJECTIVE_TOLERANCE=1.5`: the data-driven variant now only
wins when its objective isn't more than 50% worse than the best
formulaic variant found. Reconfirmed N=2's 4-seed unseeded validation
byte-identical to before this fix (its own gap there was only ~7%, well
inside the new tolerance). DLL hash
`ff7753437b55e3a0227498e22971f6ad9ad657ca1dea82dcce5c26142d67bd1b`.

**N=3 test scenario**: 3 components (1.0ns/h=1.5, 3.5ns/h=2.2,
8.0ns/h=2.8), fixed background, pixel-varying mixing fractions across a
3-way simplex (squared-contrast weights per pixel for sharper
near-pure corners than a linear gradient gives), 20x20 pixels,
peak_counts up to 400000 -- deliberately generous conditions, following
the same "give the model every advantage" principle that revealed N=2's
real identifiability once the pathological test data was fixed.

**Seeded result** (near-truth `initial_taus_ns`): recovered
(0.789/1.645, 5.017/2.285, 7.289/2.483) against true (1.0/1.5, 3.5/2.2,
8.0/2.8). Component 1: ~21%/10% error. Component 3: ~9%/11% error --
both roughly in N=1/N=2's established ~10-20% precision band.
**Component 2 (the middle lifetime) is meaningfully worse: ~43% tau
error.** This matches a well-known, structurally-expected difficulty in
multi-exponential decomposition -- a middle timescale is the hardest to
disentangle since it's correlated with (sandwiched between) both
neighbors, not just one. Not obviously a code bug; plausibly a genuine,
harder-but-real identifiability cost of going from 2 to 3 independently
distributed components (4 to 6 mutually correlated free parameters).

**Unseeded/default result**: NOT yet reliable -- (34.9/3.24, 29.1/3.32,
13.5/1.15), none matching any true component, with 2 of 3 h values
above the H_REG_SOFT_CEILING regularization was tuned to enforce for
N=2 (the ceiling is a soft penalty, not a hard bound -- for N=3's
larger 6-parameter escape space, the post-regularization objective
apparently still favours crossing it in this scenario). The N=2-tuned
regularization constants and data-driven-seeding quantile scheme are
confirmed NOT to transfer cleanly to N=3 as-is.

**Important testing-methodology caveat discovered while investigating
further** (affects how to *read* the "objective at truth vs. converged"
diagnostic used throughout this file, not the shipped fitting code):
tried scaling photon count up 100x (peak_counts 400k/4M/40M) to see
whether the seeded case's remaining gap was a shrinking, noise-driven
effect (as N=2's original investigation found) -- instead, the ratio
between "objective at true params" and "converged objective" stayed
roughly constant (~1.6-2.0x) rather than closing, AND the *reduced*
chi-square at the true parameters itself grew roughly linearly with
photon count (7.0 -> 62.8 -> 607.8 for a 10x/10x/10x count ramp) --
which should stay near a constant O(1) for a correctly-specified model
regardless of photon count. Checked whether this is N=3-specific: it is
NOT -- the SAME linear-growth pattern appears for the already-validated
N=2 case (reduced chi-square at truth: 18.6 -> 175.4 -> 1749.3 for the
same 10x/10x ramp, near-perfectly linear). This is the signature of a
small, fixed *proportional* mismatch between this repo's own hand-
written Python synthetic-data generators (used throughout this
session's validation testing, not shipped code) and the native fitter's
own exact internal convolution/discretization -- a mismatch that
doesn't shrink with more photons, so it inflates reduced chi-square
without bound as noise shrinks below its own fixed size. Tried the
obvious specific hypothesis (bin-start vs bin-center time sampling for
the synthetic pure-decay shape, matching `stretched_component_basis`'s
own `x = i*binWidth` convention) -- this made recovery WORSE, not
better, so the exact source of the mismatch is not that specific offset
and was not chased further given the practical validation approach
below already accounts for it correctly.

**What this means for every non-identifiability finding in this file,
including N=2's**: the "compare objective at true params vs. the
search's own converged answer" diagnostic (used to conclude N=2's
original data was non-identifiable, several entries below) is
confounded by this same effect and should be trusted with more caution
than it was -- a nonzero, even large, gap between these two objective
values is *expected* from generator/fitter discretization differences
alone, not proof by itself of real parameter non-identifiability. This
does NOT undermine this file's actual validation conclusions, which
relied primarily on the *practical* metric instead (does the recovered
parameter set match the true parameters within a reasonable margin) --
that comparison is unaffected by this confound, since it never compares
objective values, only recovered vs. true tau/h directly. It does mean
future non-identifiability claims in this codebase should lean on
parameter-recovery comparisons first, and treat the objective-value
comparison as at most corroborating evidence, not primary evidence.

**Where this leaves N=3**: real progress (a genuine bug fixed,
generously-conditioned seeded recovery works for 2 of 3 components, the
middle component's extra difficulty is characterized and plausibly
expected rather than a bug), but **not yet validated as reliably
working** the way N=1 and N=2 now are -- unseeded is currently
unusable, and even seeded has a real, unresolved weak point. Not yet
attempted: N=3-specific regularization/data-driven-seeding tuning
(everything reused from N=2 as-is); more/sharper training data
specifically targeting the middle component's identifiability (e.g.
deliberately near-pure single-component reference regions rather than
smooth gradients); a cascade-seeding approach (fit N=2 on the two most
different-looking pixel groups first, then add the third component).

## Improving N=2's unseeded/default case: data-driven multi-start seeding, and preferring it over "lowest objective wins" (v0.17.0, 2026-09-08, same-day follow-up #7)

**User's framing, verbatim**: "yes lets improve" -- in response to the
entry below reporting N=2 as genuinely working GIVEN a reasonable seed,
with the *unseeded* default case still weaker. This entry closes most
of that remaining gap.

**Step 1 -- data-driven seeding**: added
`estimate_data_driven_multi_stretched_seed()`. Instead of an arbitrary
fixed formula, variant 0's tau fallback is now built from the training
data's own per-pixel apparent (moment) lifetime spread: each selected
pixel gets a cheap moment-based lifetime estimate (`estimate_tau_
moment`, already used elsewhere in this file for ordinary
single-component seeding), pixels are sorted by that estimate, and
modelOrder evenly-spaced quantiles are read off as starting taus --
physically motivated by the same mechanism confirmed in the entry
below (pixels with different relative amounts of each species carry
the real information that resolves overlapping components). h keeps
the existing fixed-formula fallback (not reliably estimable the same
cheap way). An explicit `initialTaus` entry for any slot still always
wins over both.

**Direct per-variant trace revealed step 1 alone was insufficient**:
the data-driven variant (variant 0) reliably converged close to truth
(1.27/2.00, 4.35/2.77 against true 1.5/1.8, 5.0/3.0, objective 707.66)
-- but a *different*, purely arbitrary formulaic variant (variant 2, no
relation to the real data) still converged to a milder, still-
plausible-looking "compromise" point (18.72/1.83, 5.26/2.67, objective
660.68) with a genuinely LOWER raw objective -- too mild a deviation to
trip the h/tau regularization ceiling from two entries below (h=1.83 <
3.0, tau=18.72 < 30ns window). Under the existing "keep whichever
variant has the lowest objective" rule, that arbitrary-but-lucky
variant silently won every time, discarding the trustworthy,
data-grounded answer -- exactly the failure this session had already
seen in a more extreme form.

**Step 2 -- prefer the data-driven result outright**: added
`multi_stretched_result_is_plausible()` (reusing the same h/tau ceiling
already used for search-time regularization) and changed the selection
rule: the data-driven variant's own converged result is now preferred
directly whenever it's available and still looks plausible after
convergence, rather than always taking whichever variant reports the
lowest number. The diverse formulaic variants remain the fallback for
when the data-driven start isn't available (too few valid pixels) or
itself converges somewhere implausible.

**Retested, unseeded, 4 independent random-data seeds x 2 SNR levels**
(fixed background, pixel-varying mixing ratios 0.05-0.95 -- same
corrected setup as the entry below):

| rng seed | peak=200000 (normal SNR) | peak=20000 (low SNR) |
|---|---|---|
| 9 | (1.271, 2.001, 4.350, 2.770) ✓ | (3.984, 2.818, 1.297, 1.982) ✓ |
| 1 | (1.276, 1.998, 4.209, 2.786) ✓ | (0.710, 3.000, 0.715, 2.956) ✗ collapse |
| 22 | (1.277, 1.996, 4.364, 2.771) ✓ | (2.885, 2.974, 1.317, 1.972) ✓ (tau_b low) |
| 555 | (4.062, 2.808, 1.264, 2.006) ✓ | (22.45, 1.663, 4.189, 2.789) partial escape |

true: tau_a=1.5/h_a=1.8, tau_b=5.0/h_b=3.0 (component order not
semantically meaningful -- both orderings appear above).

**At normal SNR: all 4 seeds now recover correctly, unseeded** --
matching seeded-quality precision (~10-20% band) with no seed supplied
at all. **At low SNR: improved but not perfect** -- 2 of 4 clean, one
partial residual escape (555), one component-collapse (1). This is
expected graceful degradation as photon budget shrinks (a genuine
statistical limit on identifiability, not a further code gap) rather
than a regression -- the pre-improvement default case would have shown
escapes/collapses at BOTH SNR levels, not just the lower one.

**Reconfirmed unaffected**: the explicit-seed case (single-start,
`has_explicit_shared_seed`) -- same near-truth seed still converges to
exactly (1.269, 2.002, 4.523, 2.752), byte-identical to before this
change. Every other fit type (FIT_GLOBAL_MULTIEXP, single-component
FIT_GLOBAL_STRETCHEDEXP) is untouched -- the new logic is gated to
FIT_GLOBAL_MULTI_STRETCHEDEXP throughout.

DLL hash `39c85056924eda15737fa7dde2c828e7707f4ce53c2fcbeae336c78e5703de99`.

## N=2 actually works -- the "genuine non-identifiability" diagnosis was correct for pathological test data, not the model itself (v0.17.0, 2026-09-08, same-day follow-up #6)

**User's framing, verbatim**: "no i want n=2 to work" -- a flat rejection
of stopping at "N=2 has two characterized-but-unfixed failure modes."
Correctly so: the right response to that pushback was not another
regularization tweak, but re-examining whether the earlier
non-identifiability *diagnosis itself* was as fundamental as it looked.

**The re-examination**: every synthetic test in this entire N=2
investigation (multi-start, both regularization entries below) used
data with two properties that, looked at again, are actually the
worst possible case for this specific model, not a representative one:

1. **Every pixel got the identical 50/50 mixing ratio** between the
   two components (a single decay curve broadcast to all 144 pixels).
   Real global analysis resolves overlapping-timescale components
   using pixels/images with *different* relative amounts of each
   species -- amplitude ratios varying is the actual mechanism that
   breaks the ambiguity between two decay shapes. A spatially uniform
   test image structurally cannot provide that information, regardless
   of how many total photons or how many identical copies of that one
   pixel are pooled.
2. **Background was fit as a free per-pixel parameter.** A component
   whose shape goes broad/slow enough to look nearly flat over the
   acquisition window is then mathematically close to redundant with a
   free additive offset -- creating a real, structural near-degeneracy
   on top of the amplitude-ratio problem above, not just noise-driven
   ambiguity.

**Direct test, isolating each factor (2x2, seeded near truth, same
underlying tau_a=1.5/h_a=1.8, tau_b=5.0/h_b=3.0 scenario)**:

| mixing ratio | background | recovered | verdict |
|---|---|---|---|
| uniform | free | (1.42, 2.59, 1.61, 2.47) | components collapse together |
| uniform | fixed | (4.27, 0.77, 0.75, 3.45) | garbled |
| varying | free | (1.77, 2.39, **137.0**, 4.77) | one ok, one runaway escape |
| **varying** | **fixed** | **(1.27, 2.00, 4.51, 2.75)** | **both roughly right** |

Only the combination -- varying mixing ratios *and* fixed background --
gives a usable answer. Neither alone is sufficient; this matches
standard global-analysis theory (amplitude-ratio diversity is *the*
mechanism that makes shared-lifetime fitting identifiable at all) and
is not a new or unusual requirement -- it's how real global analysis
has always been expected to work, and (per gui.py's `FIT_BACKGROUND_
LABELS` default) this app's GUI already defaults to fixed background.

**Validated robust** with the corrected test setup (varying per-pixel
mixing fractions 0.05-0.95, fixed background, seeded near a reasonable
prior estimate) across 3 independent random-data seeds and 2 SNR
levels:

| rng seed / SNR | recovered (tau0,h0, tau1,h1) | true |
|---|---|---|
| seed=9, peak=200000 | (1.269, 2.002, 4.523, 2.752) | 1.5/1.8, 5.0/3.0 |
| seed=9, peak=20000 (low SNR) | (1.292, 1.986, 4.327, 2.777) | 1.5/1.8, 5.0/3.0 |
| seed=1, peak=200000 | (1.267, 2.004, 4.824, 2.719) | 1.5/1.8, 5.0/3.0 |
| seed=22, peak=200000 | (1.271, 2.001, 4.761, 2.726) | 1.5/1.8, 5.0/3.0 |
| seed=555, peak=200000 | (1.262, 2.007, 4.156, 2.796) | 1.5/1.8, 5.0/3.0 |

Consistent, bounded errors (~10-20%, similar band to N=1's already-
validated precision) across every run, not a one-off. This is a
genuine positive result, not a mitigation of a limitation.

**What's still weaker**: the *unseeded/default* case, even with fixed
background and varying ratios, still lands on a milder residual escape
(one component wandering to tau~19-21ns, h~1.7-1.9 -- plausible-
looking, not the earlier extreme tau=114-137ns/h=7-14 runaway, but
still not the true tau_a=1.5/h_a=1.8) while the other component still
finds something close to true tau_b/h_b. The regularization from the
entry below barely engages here (this milder escape sits under both
soft ceilings) and doesn't meaningfully change the outcome. **Practical
implication: N=2 needs a reasonable seed to be reliable** -- ordinary,
expected advice for any correlated multi-parameter nonlinear fit, not
a special weakness unique to this feature. This app already has the
tools for that (the "Set manually..." global model dialog --
[[manual-global-model-dialog]] -- or seeding from an independent
single-component estimate).

**Revises the framing of the two entries below**: the "genuine model
non-identifiability, confirmed by direct chi-square comparison" finding
was a correct diagnosis *of the specific test data used* (uniform
mixing ratio, free background) -- not a property of the multi-
stretched-exponential model or the search algorithm themselves. Under
realistic conditions (spatially varying species mixtures, which is
inherent to any real heterogeneous biological sample; fixed
background, this app's own GUI default; a reasonable seed), N=2
recovers both components to essentially the same precision N=1 was
already validated to. The regularization work in the entry immediately
below is not wasted -- it remains a genuine, still-active safety net
against the worst unbounded runaway escapes (tau>100ns, h>10) for
whatever configurations *do* still hit them -- but it was never the
real fix; better test data (matching how this feature is actually meant
to be used) was.

DLL hash `0880d6c4ed4976f3aa0c9e430077e32ef46037b3c9d8b405257348b3baca1856`
(identical constants/logic to the entry below -- the hash changed from
non-deterministic build metadata only, recorded here because of this
finding, made while validating that build).

## Soft-ceiling regularization on h/tau, closing off N=2's worst runaway escapes (v0.17.0, 2026-09-08, same-day follow-up #5)

**User's framing, verbatim**: "yes lets try a regularisation before
moving to N=3." Direct continuation of the multi-start work below --
even with multi-start's search-reliability fixes in place, the default
(unseeded) case could still converge to a physically implausible
"escape" (tau up to ~114ns against a 30ns acquisition window, h up to
14) that's a genuinely lower objective but not a real photophysical
answer.

**The mechanism**: both observed escapes share a signature -- a
component whose h (spread) and/or tau (centre) get large enough that
the stretched-exponential shape goes nearly flat over the acquisition
window. A shape that flat can silently "absorb" residual anywhere
without being meaningfully constrained by the data's actual curvature,
which is exactly the kind of extra, unconstrained flexibility that
produces a lower chi-square without being a real answer.

**Iteration 1 -- h-only penalty, found insufficient**: added a soft
quadratic penalty above h=4.0 to the *outer search's* objective only
(never to the reported final chi-square -- `evaluate_global_tau_
objective` wraps `score_global_taus_with_irfs`, which is what
`levenberg_marquardt_min` actually minimizes; the plain, unregularized
chi-square is still what's reported as `training_objective`/
`training_reduced_objective` for provenance and diagnostic comparison).
Retested the same default-seed scenario: the search found a
*different* escape instead -- tau pushed to 114ns (h parked just above
the new ceiling) -- confirming large-tau-alone is an equally effective
route to the same "nearly flat" degenerate shape. h regularization
alone cannot close this.

**Iteration 2 -- added a tau penalty too, still insufficient**: added a
second soft ceiling, on tau relative to the acquisition window
(tau <= 1.5x window). With lambda=0.25 for both, the escape barely
moved (tau=114 -> 49.6). Diagnosis: at the scale of the objective (a
straight sum of per-pixel reduced chi-square across ~144 pixels, so
O(150-700)), a lambda=0.25 penalty only contributed ~18 objective units
near the boundary -- far too weak to compete with the chi-square gain
from escaping.

**Iteration 3 -- lambda=5.0 (20x), h<=4/tau<=1.5x window**: the search
now converged to *exactly* the boundary (tau~45ns, h~4.0) regardless of
SNR level (both a "normal" and a "low" peak-count run landed within a
few percent of the same boundary point). Progress -- no more open-ended
runaway -- but landing exactly at whichever wall was drawn suggested
the "absorb residual" benefit of a flat component saturates near where
these ceilings were set, so the search was settling at the marginal
break-even point rather than actually being deterred.

**Iteration 4 -- tightened ceilings (h<=3.0, tau<=1.0x window), same
lambda=5.0 -- this is what shipped**: a qualitatively different,
better outcome. The escaping component still parks at the (now
tighter) boundary -- an easily recognized signature (tau near the full
acquisition window, h near 3) -- but the *other* component now
genuinely recovers close to its true value:

| scenario | recovered (tau0,h0, tau1,h1) | true tau_b/h_b | component-1 error |
|---|---|---|---|
| peak=200000 | (30.5, 3.06, **5.52, 2.94**) | 5.0 / 3.0 | ~10% / ~2% |
| peak=20000 (lower SNR) | (30.1, 3.00, **5.80, 2.94**) | 5.0 / 3.0 | ~16% / ~2% |

Component 0 in both rows is the recognizable "junk absorber" (tau at
the window, h at the ceiling) -- not a real answer, but visibly flagged
as one rather than looking like a plausible-but-wrong lifetime the way
tau=73-114ns/h=7-14 did before.

**What this does NOT fix**: retested the near-truth explicit-seed case
(`initial_taus_ns=(1.5, 1.8, 5.0, 3.0)`) with the final regularization
in place -- essentially unchanged (converges to two components near
each other, ~1.8-2.1ns, same as before regularization was added). This
is a *different* failure mode -- component collapse/merging toward a
compromise, not runaway flatness -- and this regularization was never
targeted at it; h and tau both stay comfortably inside both ceilings
throughout that search, so the new penalty term is never even active.
Confirms the fix is doing exactly what it was built for and nothing
more.

**N=1 sanity-checked unaffected** (single-component fit, ceilings never
approached: recovered tau=3.71ns against true 3.0ns, h=2.37 against
2.5 -- consistent with N=1's already-documented ~10-14%/~2-3% recovery
band, no regression).

DLL hash `c2d5cce0afb61308eb05118959d083ba7ac51e6e32d44b240d500eebcb1b73c2`.
Constants: `H_REG_SOFT_CEILING=3.0`, `H_REG_LAMBDA=5.0`,
`TAU_REG_WINDOW_MULTIPLE=1.0`, `TAU_REG_LAMBDA=5.0` (see their own doc
comments in flimfit_labview.c for the full rationale) -- considered
engineering defaults reached by this iteration sequence, not
independently swept/optimized.

## Multi-start optimization for N=2, and a follow-up fix so it never discards an explicit seed (v0.17.0, 2026-09-08, same-day follow-up #4)

**User's framing, verbatim**: "yes lets optimise N=2 first then we can
move onto n=3 once we are satisfied" -- following directly from the
non-identifiability diagnosis below (entry immediately below this one).
This entry is the concrete "optimise N=2" work that instruction asked
for.

**What was tried**: `levenberg_marquardt_min` is strictly local --
given the confirmed non-identifiability, a single arbitrary default
seed's local optimum is not a trustworthy answer on its own. Added
`MULTI_START_COUNT=5` diverse starting (tau, h) combinations
(`seed_multi_start_taus`, tauScale=0.3+0.5*variant per component,
h=1.2+0.7*component+0.5*variant) tried whenever
`fitType==FIT_GLOBAL_MULTI_STRETCHEDEXP`, keeping whichever converges
to the globally-lowest training objective, in both
`flim_fit_global_pixels_v2` and `flim_train_global_model_v1`.
`globalStatsOut[0]` (previously always hardcoded 0.0) now reports
which start variant won.

**Confirmed by direct test**: multi-start does **not** fix the
non-identifiability (expected -- it was never meant to) -- for the same
two-component scenario used in the entry below, it finds an *even
lower, even more extreme* "better" objective than the single-start
search had (tau=73.3ns/h=7.3 for one component, reduced chi-square
1.46). What it **does** fix is seed-dependence for the *default*
(unseeded) case: instead of one arbitrary local optimum, the answer is
now the best of 5 diverse attempts, every time.

**A real design flaw found immediately after, by direct test**: with a
user-supplied `initial_taus_ns` seed placed *near the true values*
(1.5, 1.8, 5.0, 3.0), the result was **still silently overridden** by
one of the five multi-start variants finding a lower-but-wilder
objective (that same tau=73.3ns answer) -- discarding a correct,
informed user seed exactly like an uninformed default one. This was
not something the user flagged; found proactively while validating the
feature and judged a real usability problem worth fixing immediately,
since a user seeding at a value they trust (e.g. from a separate
single-component fit) should get a refinement of that seed, not a
wholesale replacement by a search they didn't ask to run.

**The fix**: `has_explicit_shared_seed(sharedCount, initialTaus,
initialTausLength)` -- true only when the caller supplied a fully
specified shared-parameter seed (every one of `sharedCount` slots
`> 0.0`). `startCount` is now `MULTI_START_COUNT` only when
`fitType==FIT_GLOBAL_MULTI_STRETCHEDEXP && !has_explicit_shared_seed(...)`,
else always `1` -- an explicit seed gets exactly one search, respecting
it as the starting point, identical to pre-multi-start behaviour.

**Re-validated after the fix** (same scenario: `tau_a=1.5/h_a=1.8`,
`tau_b=5.0/h_b=3.0`, peak_counts=200000, seed=9):

| seed given | elapsed | result | reduced chi-square |
|---|---|---|---|
| explicit (1.5, 1.8, 5.0, 3.0) | 2.05s | (1.42, 2.59, 1.61, 2.47) | 4.03 |
| none (default) | 7.03s | (73.33, 7.33, 2.01, 2.52) | 1.46 |

The explicit-seed run no longer touches the tau=73 region at all, and
completes in roughly 1/3.4 the time of the 5-variant default run --
consistent with one search vs five. The default run still finds the
same extreme-but-lower-objective optimum multi-start found before the
fix, confirming the fix didn't accidentally disable multi-start for
the case it's meant to help.

**Where this leaves N=2**: multi-start makes the *unseeded* case
reliably find its own best local answer (still not truth, per the
confirmed non-identifiability), and an *explicit* seed is now always
honoured as a genuine starting point rather than being at the mercy of
a wilder alternative. Neither closes the underlying identifiability
gap -- that isn't fixable by more search, only by more information
(more photons, more images, or a prior/regularization discouraging the
two components from being mutually redundant, none of which have been
attempted). DLL hash `df4942a6f3ec03b634bf2ea47733514ca0449ada072cfeeffe373e024efcfdc0`.

## N=2's remaining gap is genuine model non-identifiability, confirmed by direct chi-square comparison, not a further code bug (v0.17.0, 2026-09-07, same-day follow-up #3)

**User's framing, verbatim**: "lets do it" -- after N=1 was validated
(entry below) but N=2 still showed a real, uninvestigated gap. This
entry closes that investigation with a direct, quantitative answer
rather than more guessing.

**What was found**: N=2 (two independently-stretched components) does
**not** converge with increasing SNR the way N=1 does -- tested
peak_counts 20000/200000/2,000,000 on the same two-component scenario
(`tau_a=1.5/h_a=1.8`, `tau_b=5.0/h_b=3.0`) and the recovered values
stayed erratic, including one run where `h_b` hit 14 (a physically
implausible, extremely broad distribution) rather than settling near
the true 3.0.

**The decisive test**: rather than treating this as "still probably a
bug," directly compared the training objective (`training_objective`/
`training_reduced_objective`, already returned by `train_global_model`)
at the true generating parameters against the objective at the search's
own converged answer, on the same data, using the exact seed-with-tiny-
budget technique that worked for N=1's earlier investigation:

| scenario | point | reduced chi-square |
|---|---|---|
| peak=200000 | near true (1.13, 2.21, 5.35, 2.26) | **4.61** |
| peak=200000 | search's answer (1.42, 2.59, 1.61, 2.47) | **4.03** (lower = better fit) |
| peak=20000 | near true (1.13, 2.19, 5.34, 2.28) | **1.34** |
| peak=20000 | search's answer (1.83, 2.35, 6.55, 14.06) | **1.01** (lower = better fit) |

**At both tested noise levels, the search's converged answer is a
genuinely better (lower chi-square) fit to the actual data than the
true generating parameters are.** The optimizer isn't malfunctioning --
it's correctly finding the true minimum of a real, well-defined
objective; that minimum simply isn't at the parameters used to generate
the data. This is the textbook signature of model non-identifiability:
two independently-stretched components is enough flexibility (4
correlated free parameters, each pair internally correlated the same
way the single-component case already showed) that noisy data can
genuinely be explained *better* by a "compromise" pair of components
than by the true, well-separated ones.

**This is not a new phenomenon in this project.** It matches, closely,
an already-documented finding from much earlier work
([[global-fit-real-marquardt-unused]]/`global-fit-real-marquardt-
unused.md`): 3 pure reference images, jointly tri-exponential-trained,
don't recover their known lifetimes even seeded exactly at the truth --
concluded then as "a modelling/identifiability limitation, not a
search-algorithm bug." The same reasoning applies here, now for
independently-distributed components specifically, which have twice
the free parameters per component (tau *and* h, both mutually
correlated) that the plain multiexp case does -- so a *more* severe
version of the same well-known limitation, not a different problem.

**What this does and doesn't mean going forward**: N=2/N=3 multi-
stretched fitting is the right tool when the underlying photophysics
genuinely mixes multiple distributed populations within every pixel
and there's enough independent information (photon counts, well-
separated timescales, multiple images/conditions) to constrain all the
free parameters -- same caveat that's always applied to global fitting
in this project. It is not a tool that will reliably recover 2+
components from a single, modest-photon-count, closely-spaced-lifetime
synthetic test case, and that's an expected property of the underlying
statistics, not something further code changes here can fix. Possible
mitigations if this is revisited (not attempted, listed for later):
more images/pixels/photons; multi-start optimization (try several
seeds, keep the globally-best objective) since the current search is
strictly local; a soft regularization prior discouraging components
from collapsing toward each other; reducing to N=1-per-pixel-class
approaches when components are known to be well-separated in space
rather than mixed within pixels.

**Files changed**: none (this entry is a diagnosis, not a fix -- no
DLL rebuild). Uses `TrainedGlobalModel.training_objective`/
`training_reduced_objective` (already-existing fields) as the
comparison metric.

## The anchor "fix" below was itself wrong; bin 0 is correct -- N=1 now validated (v0.17.0, 2026-09-07, same-day follow-up #2)

**User's framing, verbatim**: "keep going - this is something whic hcan
be very useful. We can do it. I have faith in you" -- after the entry
below's two fixes still left a large, systematic gap (my new anchor+
weighting: tau~4.5-4.6ns; legacy: tau~3.66ns; true: 3.0ns). That
persistence turned out to be the right call: a third real issue
remained, and finding it required questioning the entry below's own
central conclusion, not just adding another fix on top of it.

**What was checked, in order, to find it:**

1. Confirmed the outer search wasn't buggy: seeded from legacy's answer,
   from the true (tau, h), and from its own converged answer, with only
   1 outer iteration of budget -- it moved strongly toward the same
   point (~4.4-4.5ns) regardless of starting point. That's a genuine
   minimum of the objective, not a search artifact.
2. Confirmed the IRLS per-pixel solve was fully converged (8 vs 50
   iterations gave bit-identical results to 9+ significant figures).
3. Confirmed the gap was systematic, not noise: tested 3 independent
   random seeds, ratio between my answer and legacy's stayed at
   ~1.15-1.17x every time.
4. Directly isolated the anchor's own contribution by forcing it to
   bin 0 (with the entry-below fixes -- corrected weighting floor, and
   this fast path's own clean double-precision convolution -- still in
   place): **recovered 3.14ns/2.50 against a true 3.0ns/2.5** -- close
   to exact, and clearly better than both the entry-below's
   transientStart anchor (4.5-4.6ns) *and* legacy's own bin-0-anchored
   answer (3.66ns).

**The reasoning, once traced through**: the entry below's central claim
-- that bundled EcfSingle.c's FIXME comment means bin-0 anchoring is a
deliberate accuracy-for-speed tradeoff -- was a misreading. For ordinary
reconvolution fitting (`model (*) IRF`), the pure model's own t=0 is
mathematically arbitrary: the convolution with the IRF at *its own*,
correctly-measured position is what reproduces the true observable
timing, regardless of where the model's own zero sits, **as long as
it's early enough that the model is genuinely ~zero before the IRF's
own rise** -- true for bin 0 in any real acquisition, which always
starts recording before the excitation pulse arrives. Anchoring
anywhere *after* that point (`fit_start`, or the imprecisely-estimated
`transientStart` heuristic used below) doesn't fix a bias -- it
introduces one, by feeding the model a "pure decay" curve that's
zero for longer than the true physics, right at the model's own
input to the convolution.

`infer_transient_start_time`/`interpolated_crossing_time` (added below)
are kept in source as separately-verified, possibly-useful building
blocks for a case not tested here (a real, non-synthetic dataset where
the true excitation genuinely happens meaningfully after bin 0) -- just
not used by the fast path's anchor choice, which is now a fixed `0.0`.

**Validated result** (N=1, seeded near the true tau/h -- the same
caveat as before: this is a local search on a model with genuine
tau/h correlation, so the *far-off default seed* can still land on a
worse local optimum, same as it always could):

| random seed | recovered tau | recovered h | true |
|---|---|---|---|
| 2 | 3.37-3.68 (SNR-dependent) | 2.38-2.53 | 3.0 / 2.5 |
| 7 | 3.35 | 2.53 | 3.0 / 2.5 |
| 11 | 3.41 | 2.50 | 3.0 / 2.5 |
| 42 | 3.28 | 2.56 | 3.0 / 2.5 |

Consistently within ~10-14% of true tau and within ~2-3% of true h
across independent noise realizations -- a real, validated improvement,
not just a different bias. This is genuinely better than the legacy
single-component fast path's own answer (~3.66ns/2.27) on the same
data, not merely "no longer erratic."

**N=2 is improved but not as cleanly resolved**: re-tested on the
32x32/1024px two-component scenario (`tau_a=1.5/h_a=1.8`,
`tau_b=5.0/h_b=3.0`, seeded at truth) -- no more wild outliers (the
earlier tau=10.5 anomaly is gone), but the two recovered (tau, h) pairs
still don't cleanly separate into the two true components; results
vary meaningfully between random seeds (one seed landed near
(2.9,2.5)/(3.3,2.5), a plausible "averaged, not separated" outcome).
Consistent with (not necessarily beyond) the already-flagged tau/h
correlation risk compounding across two components -- not
distinguished from a further, not-yet-found code issue, since time
ran out on this investigation before that could be checked as
rigorously as N=1 was.

**Files changed**: `native/flimfit_labview.c` (the fast path's
`groupAnchorTime` is now a fixed `0.0`, not `infer_transient_start_
time`'s output; that function and `interpolated_crossing_time` remain
in source, unused), `flim_app/native_backend.py` (hash pin
`41b3042a51b08a4122afaec27089d95137053c91c621d278c9238221c5638bb7`),
`tests/test_processing.py` (matching pin).

## Two real, confirmed anchor/weighting bugs in the multi-stretched fast path, found via visual diagnosis (v0.17.0, 2026-09-07, same-day follow-up)

**User's framing, verbatim, since it shaped how this was investigated**:
"lets see if it helps. I don't want to just stop if there is a real issue
here" -- after the fast path (entry below) showed a persistent, ~50%
tau bias even after correctly anchoring at `fit_start`. That instinct was
right: two real bugs remained, found by generating and inspecting actual
diagnostic plots (at the user's request) rather than continuing to guess
from numbers alone.

### Bug 1: `fit_start` is the wrong anchor -- confirmed visually, fixed with `transient_start`

The previous entry's fix anchored the fast path's pure-decay model at
`fit_start`. A diagnostic figure (three candidate anchors' pure and
IRF-convolved curves, overlaid on the real synthetic data, computed with
a Python port of `infer_window`'s exact logic) showed why that was still
wrong: `fit_start` is deliberately placed *past* the decay's true onset
(specifically to dodge IRF-shape artifacts in the summed fit region).
Anchoring the model there, then convolving with an IRF that's still
sitting at its own natural, unshifted position, stacks two delays on
top of each other -- the observable rise lands roughly
`fit_start + instr_peak` bins late, clearly visible in the figure as a
~13-bin-late spurious step. The correct anchor is `infer_window`'s own
internal `transientStart` (`decayRise - instrRise`, no added margin) --
visually, its convolved curve tracked the real data's rise just as well
as the original bin-0 anchor does, confirming it as the fix.

Implemented as a new `infer_transient_start_time`, computed **sub-bin**
(linear interpolation at each threshold crossing) rather than snapped to
a whole bin, since `infer_window`'s own whole-bin `first_crossing` would
otherwise waste most of the fix -- a whole-bin rounding error is a large
fraction of a narrow IRF's own width (the validation IRF here is under 1
bin FWHM). `stretched_component_basis`'s `anchor` parameter changed from
an `int` bin index to a `double` time, used directly in the pure-decay
formula (`x = i*binWidth - anchorTime`) with no rounding.

### Bug 2: IRLS weighting floored variance at 1 count, not 15

Found while investigating why the anchor fix alone didn't close the gap
against the legacy path. Direct comparison against bundled
`EcfGlobal.c`'s `GCI_marquardt_global_exps_single_step` -- the actual
function the legacy fast path calls for `NOISE_POISSON_FIT`/
`NOISE_POISSON_DATA` -- shows it floors variance at 15 counts
(`weight = yfit > 15 ? 1/yfit : 1/15`), not 1. `fit_linear_multi_
stretched_pixel`'s IRLS weighting used a floor of 1 throughout, giving
any bin with fewer than 15 predicted counts up to 15x more relative
weight than FLIMLib's own convention -- exactly the long, low-count tail
of a decay, which is also where a stretched exponential's `h` differs
most from an ordinary exponential's. Fixed to match (floor=15 for the
two Poisson-weighted models; `NOISE_MLE`, noiseModel=0, is a further,
separate case -- FLIMLib uses a genuine Poisson deviance there, not a
weighted chi-square at all, so this IRLS scheme is a reasonable proxy
toward the same fit but won't reproduce FLIMLib's exact MLE objective
value; kept at floor=1 there since that's what NOISE_MLE's own bundled
formula uses).

**Confirmed real but did not move this specific test's number**: the
validation scenario uses `h=2.5`, a heavily-stretched decay -- directly
checked the actual decay values and only 11 of 200 bins ever drop below
15 counts, all in the pre-transient background, none in the tail
(`exp(-(t/tau)^(1/h))` with `h>1` decays far slower than an ordinary
exponential; over the tested window it never gets there). The floor fix
is real, confirmed correct against FLIMLib's own source, and will matter
for shorter-tau or less-stretched (lower h) decays whose tails do drop
below 15 counts within the fit window -- just not exercised by this
particular test case, a fact discovered by checking rather than assumed.

### Net result

SNR-scaling test (N=1, seeded near truth, `initial_taus_ns=(3.0, 2.5)`,
true tau=3.0/h=2.5), before and after both fixes in this entry:

| | tau | h |
|---|---|---|
| fit_start anchor (previous entry, buggy) | 3.07 -> 67.6 (blew up) | 0.66 -> 0.05 (sign-flipped) |
| transient_start anchor (whole-bin) | 4.69 -> 4.75 | 2.08 -> 2.06 |
| + sub-bin interpolation | 4.47 -> 4.61 | 2.10 -> 2.17 |
| + variance floor fix (this test unaffected, see above) | 4.47 -> 4.61 | 2.10 -> 2.17 |

Legacy (bin-0 anchor, already-shipped, already-validated) on the same
data: tau=3.66, h=2.27.

**Both fixes are real, confirmed correct (one visually, one by direct
source comparison), and stable (SNR-independent, matching pytest's
bit-identical regression check on the untouched multiexp/single-
stretched fast paths)**. A residual gap vs both ground truth and the
legacy path remains on this specific N=1 scenario, not yet closed.
Candidates not yet checked: whether `fit_linear_multi_stretched_pixel`'s
IRLS genuinely reaches the same converged (Z, amplitude) point FLIMLib's
own nonlinear Marquardt does for the *other* noise models tested here
(NOISE_POISSON_FIT specifically, where iteration count/convergence
tolerance could differ in ways not yet compared); whether the remaining
gap is a genuine tau/h identifiability characteristic of this
correlated 2-parameter model rather than a further code issue (plausible
given the model's well-documented correlation, not yet distinguished
from a bug); N=2/N=3 have not been re-validated against these two fixes.

**Files changed**: `native/flimfit_labview.c` (`infer_transient_start_
time`, `interpolated_crossing_time`, `stretched_component_basis`'s
anchor parameter, `fit_linear_multi_stretched_pixel`'s weighting floor),
`flim_app/native_backend.py` (hash pin `b8f8c79dd4f52b8fcca16102236f16
c3e9c6781bfb27d71f76d15736d2868c8b`), `tests/test_processing.py`
(matching pin). Diagnostic figure generated at
`anchor_diagram.py`/`anchor_comparison.png` in this session's scratchpad
(not committed to the project -- regenerate if needed, it's a
standalone Python port of `infer_window`'s logic plus matplotlib plots).

## Fast IRF-grouped path for FIT_GLOBAL_MULTI_STRETCHEDEXP, and what it revealed about the fast/fallback precision gap (v0.17.0, 2026-09-07 follow-up)

**What was built**: the fast path deliberately deferred in the entry
below. `score_global_taus_with_irfs` gained a third branch, parallel to
the existing multiexp/single-stretched one: for each IRF group, calls
bundled `GCI_marquardt_global_exps_calculate_exps_instr` once *per
component* (in ordinary single-component `FIT_GLOBAL_STRETCHEDEXP` mode,
with placeholder amplitude/background -- that branch only ever reads
tau/h) to get each component's IRF-convolved basis shape, then fits
every pixel in the group against the combined basis via a new
`fit_linear_multi_stretched_pixel`. That function is a small, hand-
written weighted linear solve (reusing `solve_linear_system`), *not*
another call into bundled Marquardt machinery -- with tau/h held fixed,
`y(t) = Z + sum_i A_i*basis_i(t)` is genuinely linear in the remaining
free parameters, so no nonlinear search is needed for that step at all.
Poisson-flavoured noise models get a handful of IRLS reweighting passes
(standard Fisher-scoring-style practice); constant/data-weighted models
need only one pass since their weights don't depend on the fit.

**Measured speedup**: a 32x32/1024px N=2 fit that previously exceeded a
1800s subprocess timeout entirely now completes in **~19 seconds**. This
was the single highest-leverage fix recommended in the entry below, and
it delivered -- every validation cycle that used to cost minutes now
costs seconds, which is what made the rest of this entry possible to
investigate at all.

**This also resolved the earlier N=2 ambiguity, and explained the
precision gap -- read this before trusting any N>=2 result**:

1. *Is the search actually working at N=2?* Yes, confirmed conclusively
   now that iteration is cheap: the fast path converges to the exact
   same (tau, h) values regardless of iteration budget (40 vs 80) or a
   10x photon-count change. That's the signature of a genuine, stable
   local optimum, not erratic or unconverged behaviour -- the earlier
   "does it move at all" question is settled.

2. *Why doesn't it match ground truth more closely, then?* Investigating
   this turned up something more interesting than a bug in the new code.
   The **already-shipped, already-validated multiexp global fast path**
   (fit_distribution_model=0, nothing changed this session) was tested
   the identical way -- seeded at the true tau, checked across a 10x
   photon-count increase -- and shows the **same signature**: recovered
   tau sits at ~3.12-3.13ns against a true 3.0ns, persistent regardless
   of SNR. Not a new bug. Bundled `EcfSingle.c` has a FIXME comment
   describing exactly this: FLIMLib's array-optimized reconvolution
   functions (`stretchedexp_array` etc., recognized by name inside
   `GCI_marquardt_compute_fn_instr`; `GCI_marquardt_global_exps_
   calculate_exps_instr` computes the same shape) place the model's
   time-zero at absolute bin 0, computing the whole curve from t=0 once
   and convolving with the IRF at its own absolute position -- a
   deliberate speed shortcut. The generic per-point fitfunc engine the
   *slow fallback path* uses instead evaluates the model at `x = xincr*
   (i - fit_start)`, i.e. time-zero at `fit_start` -- which FLIMLib's own
   authors' comment says is where t=0 *should* be aligned, calling the
   array-optimized shortcut's placement "adverse" by comparison. Every
   fast/grouped path -- multiexp, single-stretched, and now multi-
   stretched -- inherits this same tradeoff; it was already being paid,
   silently, by the multiexp/single-stretched fast paths validated in
   earlier sessions, at a magnitude nobody had previously isolated this
   precisely.

3. *Why is multi-stretched's gap bigger than multiexp's ~4%?* Confirmed
   on the real N=2 scenario (two well-separated stretched components,
   32x32px, seeded at truth): one component recovers reasonably close
   (tau 1.5->1.35, ~10% off), the other lands far off (tau 5.0->10.5,
   more than 2x, h 3.0->1.6) -- stable across iteration budget and a
   10x SNR change, so a real, converged answer, not noise. This is
   consistent with, not contradicting, the tau/h correlation flagged as
   a real risk when this feature was first proposed: the array-
   optimized shortcut's time-alignment tradeoff compounds with the
   model's own known parameter correlation, and does so unevenly across
   components rather than as a single, predictable percentage.

**Where this leaves fit_type=2**: the fast path is correctly implemented
and delivers the speedup it was built for. It inherits a real, now well-
understood precision tradeoff that pre-existed this session's work
(shared by every fast/grouped path in this codebase), amplified here by
genuine model correlation. The slow fallback path remains available
(automatically used whenever `nBins` exceeds the fast path's scratch
limit) and is more accurate but far slower -- exactly the same speed/
precision choice multiexp and single-stretched already offer, just not
yet exposed as a user-facing choice for this fit type. Not yet done:
quantifying the fast-path bias's dependence on component separation
(are well-separated components less affected than close ones?), and
deciding whether to expose a "prefer accuracy over speed" toggle now
that both paths exist and work.

**Files changed**: `native/flimfit_labview.c` (`fit_linear_multi_
stretched_pixel`, the new fast-path branch in `score_global_taus_with_
irfs`, a forward declaration for `solve_linear_system`), `flim_app/
native_backend.py` (hash pin `f454cff325b9905b5ec6e131f58fbe9e0b72cf7
cbca60198af3459e840703ab0`), `tests/test_processing.py` (matching pin).

## Up to 3 independently-distributed global-fit components (FIT_GLOBAL_MULTI_STRETCHEDEXP, v0.17.0, 2026-09-07) -- superseded by the fast-path entry above, read that first

**Superseded 2026-09-07, same day**: the fast-path entry immediately
above this one resolved the "does N=2 even move" question this entry
left open (yes, it does, confirmed conclusively once fast iteration
made that cheap to check) and explained the precision gap below as a
real, pre-existing tradeoff shared by every fast/grouped path in this
codebase, not a bug isolated to this fit type. This entry is kept for
its accurate account of the concept, the h-seed-degeneracy bug, and the
debugging methodology -- just read the entry above first for the
current, fuller picture.

**Concept** (user: "Can we develop something new which allows up to three
distinct lifetime distributions in a data set?"): generalizes v0.16.0's
single-component stretched-exponential toggle to up to 3 components,
**each independently carrying its own (tau, h)**, not just one shared
distribution mixed with discrete lifetimes:

```
I(t) = Z + sum_i A_i * exp(-(t/tau_i)^(1/h_i))     i = 0..modelOrder-1
```

New app-defined `FIT_GLOBAL_MULTI_STRETCHEDEXP` (=2, never passed to any
bundled FLIMLib function -- see its own doc comment in
`native/flimfit_labview.c`) selected via `AnalysisSettings.
fit_distribution_model = 2` (0 = multiexp, 1 = legacy single-component
stretched, 2 = this). Unlike the legacy toggle, selecting 2 does **not**
force `model_order` to 1 -- it's respected normally (1-3), same as
multiexp.

**Implementation** (all in `native/flimfit_labview.c`, no bundled
FLIMLib source touched):
- New `multi_stretchedexp` fitfunc (matches `GCI_marquardt_fitting_
  engine`'s generic `void(float x, float param[], float *y, float
  dy_dparam[], int nparam)` contract, same as the bundled `GCI_
  stretchedexp`/`GCI_multiexp_tau`) -- computes the model and analytic
  derivatives for N summed stretched-exponential components. Hand-
  derived, then verified to reduce to bundled `GCI_stretchedexp`'s own
  formula bit-for-bit at N=1 (same intermediate `ex` reuse pattern).
  Parameter layout: `param[0]`=Z, then per component i: amplitude at
  `1+3i`, tau at `2+3i`, h at `3+3i`.
- `shared_param_index`/`bound_shared_param`/a new `is_shared_slot_h`
  helper generalized to a third fit type: shared slot `2i` is component
  i's tau, slot `2i+1` is its h (`sharedCount = 2*modelOrder`).
- `fit_fixed_tau_pixel` (the per-pixel "shared params fixed, fit Z +
  amplitudes" building block) gained a third `multiStretched` branch,
  parallel to its existing `stretched`/plain-multiexp ones.
- **Phase 1 deliberately never uses the fast IRF-grouped path**
  (`score_global_taus_with_irfs`'s branch that reuses bundled
  `GCI_marquardt_global_exps_calculate_exps_instr`/`_do_fit_single`) --
  those bundled functions only understand `FIT_GLOBAL_MULTIEXP`/
  `FIT_GLOBAL_STRETCHEDEXP` internally, so this fit type is unconditionally
  routed through the (already-existing) per-pixel fallback loop instead.
  This was a deliberate correctness-first sequencing choice, but its
  **practical cost turned out to dominate this entire validation
  session**: with no basis reused across pixels, every objective
  evaluation costs a full independent Marquardt fit per pixel, and a
  real-scale test (32x32/1024px, global_iterations=40) measured to
  **exceed a 1800s subprocess timeout without finishing**. All
  validation in this entry had to be done at deliberately tiny image
  sizes (6x6 to 12x12) to keep iteration cycles under a few minutes.
  Building the fast path for this fit type (call the bundled single-
  stretched-component routine once per component, accumulating into a
  combined basis -- outlined but not attempted) is very likely the right
  next step before further validation, not just a later speed polish.
- `MAX_SHARED_PARAMS` raised 3->6 (sharedCount can now be up to 6 at
  modelOrder=3) and every fixed-size-3 array in the outer search
  (`levenberg_marquardt_min`, `solve_linear_system`,
  `evaluate_global_tau_objective`) widened to match. `globalIterations`
  ceiling raised 20->80 (default still 4, unchanged) since each outer
  Newton iteration costs far more evaluations at higher `sharedCount`.

**A real, confirmed, fixed bug**: every component's h was seeded to the
identical `1.5` (`flim_fit_global_pixels_v2`/`flim_train_global_model_v1`'s
seeding loop). At N=1 this is harmless, but at N>=2 it makes every
component's convolved basis shape differ *only* by tau -- a badly
conditioned (near-collinear) linear sub-problem for the per-pixel
amplitude fit. Confirmed via file-based tracing (see the debugging-
methodology note in [[stretched-exponential-distribution-toggle]] --
`getenv`/stderr still don't work from inside this ctypes-loaded DLL,
file-based logging is still the only reliable method found): most
pixels' inner fit failed outright (`ret<0`) the moment ANY shared
parameter moved even slightly off that degenerate seed, and the failed-
pixel penalty (`objective += 1e30` each) swamped the outer search's
finite-difference gradient, making it appear falsely converged exactly
at the seed. Fixed by seeding each component's h differently
(`1.5 + 0.6*component`).

**Validation status: N=1 confirmed working; N=2/N=3 NOT yet confirmed.**
This is the honest, current stopping point -- do not treat N>=2 as
validated based on this entry.

- **N=1** (one distributed component, `modelOrder=1, fit_distribution_
  model=2`): confirmed via the same diagnostic that validated the
  single-component work in v0.16.0 -- recovered (tau, h) moved
  measurably *closer* to the known synthetic ground truth as photon
  counts increased 10x (peak_counts 8000 -> 80000: tau 3.38->3.07ns,
  h 2.65->2.58, true tau=3.0/h=2.5), the signature of a genuinely
  working, SNR-limited estimator rather than a biased/broken one. This
  required seeding reasonably close to the truth first -- from the
  default far-off seed (tau~10.5, h=1.5), the search can land on a
  distant, much-worse local optimum instead (expected for a local,
  single-start method on a model with known tau/h correlation, not
  itself evidence of a bug).
- **N=2** (two independently distributed components): inconclusive.
  Direct iteration-by-iteration tracing (`levenberg_marquardt_min`
  instrumented to log gradient/Hessian/every accept-reject decision, see
  method above) on a degenerate seed showed the search DOES do genuine,
  correct-looking work -- real trust-region-capped Newton steps, correct
  accept/reject based on actual evaluated objective values, halving the
  objective over several iterations. But a separate test, seeded
  slightly away from the true (tau, h) pair at high SNR (a case designed
  to have an easy, nearby true optimum), showed **zero movement at all**
  -- the recovered values were bit-identical to the perturbed seed. This
  could mean the search correctly recognised it was already at (or
  extremely near) a true local minimum for that specific noise
  realization -- plausible, not ruled out -- or it could mean a residual
  bug distinct from the h-seed-degeneracy one already fixed. Time
  constraints (each test cycle costing minutes, per the fast-path gap
  above) stopped the investigation before this was disambiguated.
- **N=3**: not attempted at all this session -- N=2 needs resolving
  first, per the user's own stated sequencing ("validate first for N=2
  then move onto N=3").

**Recommended next steps, in order**: (1) build the fast IRF-grouped path
for this fit type so validation cycles stop costing minutes each --
almost certainly the highest-leverage next step, since it was the
practical bottleneck throughout this entire session; (2) with fast
iteration available, directly instrument and disambiguate the N=2
"zero movement" case (re-add the `levenberg_marquardt_min`/`evaluate_
global_tau_objective` file-based tracing used above, now cheap to run
repeatedly); (3) once N=2 is confirmed (or fixed), extend the same
checks to N=3; (4) only then move on to the GUI toggle, `TrainedGlobalModel`
data-model changes for a variable-length component list, and
`flim_apply_global_model_v1` support -- none of that has been started.

**Files changed**: `native/flimfit_labview.c` (all of the above),
`flim_app/models.py` (`fit_distribution_model` now 0/1/2), `flim_app/
native_backend.py` (`fit_type=2` handling in `fit_global`/`train_global_
model`; hash pin `1c8d0bff6b424f6d3c3afc960611be5b624e9be31f163e8eaacd
3841cdd1af1d`), `flim_app/processing.py` (apply-model guard extended to
refuse `fit_distribution_model != 0`, not just `== 1`), `flim_app/
global_model.py` (`summary()` handles the `[tau0,h0,tau1,h1,...]`
layout), `tests/test_processing.py` (matching hash pin). GUI toggle:
**not started** (v0.16.0's toggle stays multiexp/single-stretched only;
extending it to offer "N components, each distributed" is future work).

## Stretched-exponential (distribution) global-fit toggle, and a real two-part bug it exposed in the shared-tau search (v0.16.0, 2026-09-06)

**Feature**: global fitting can now represent one species as a
*distribution* of lifetimes (FLIMLib's bundled `GCI_stretchedexp`,
Kohlrausch-Williams-Watts: `I(t) = Z + A·exp(-(t/τ)^(1/h))`) instead of
always assuming each species has one exact lifetime. A new "Distribution
model" sidebar combo (Multi-exponential / Stretched exponential) toggles
between the two -- forcing and disabling "Model order" to 1 when
stretched is selected, since a stretched exponential is always one
component. `AnalysisSettings.fit_distribution_model` (0/1) threads through
`train_global_model`, `analyze_dataset`'s `flimlib_global` method, and
`TrainedGlobalModel` (which stores `[tau, h]` instead of discrete taus
when stretched). Applying a stretched-trained model via
`flimlib_apply_global` is deliberately refused with a clear error --
`flim_apply_global_model_v1` was not generalized this round, so treating
`h` as a second discrete lifetime would silently fit the wrong model.

**A genuine, pre-existing, two-part bug in the shared-tau outer search**
(`levenberg_marquardt_min` in `native/flimfit_labview.c`, present since
its creation on 2026-09-01, *discovered* rather than introduced by this
work) was found and fixed while validating the toggle:

1. `bounded_tau`/`bounded_h` were resetting any negative-but-finite
   proposed value to an arbitrary fixed default (`binWidth*nBins/4` /
   `1.0`) instead of clamping it to the actual lower bound. Since this
   same function clamps every damped-Newton step the search proposes, an
   oversized (but otherwise normal) negative overshoot got teleported to
   one fixed point regardless of how far it overshot -- a
   damping-insensitive trap: raising `lambda` up to 5^7-fold within the
   8-retry budget still produced the identical (wrong) destination and
   identical (rejected) objective every time. Fixed: only a genuinely
   non-finite (NaN/Inf) value gets reset; everything else, including
   negative values, clamps to the nearest real bound instead.
2. That fix alone was **not sufficient** -- the search still converged to
   the same wrong point, confirmed via direct evaluation-by-evaluation
   file-based tracing (neither `getenv()` nor `fprintf(stderr, ...)` from
   inside the ctypes-loaded DLL produced any visible output under a
   piped subprocess, for CRT/stdio-isolation reasons; writing debug lines
   directly to a file via `fopen`/`fprintf`/`fclose` was the only method
   that worked). Root cause: the finite-difference Hessian estimate
   driving the damped-Newton step can come out spuriously small (from
   fine-scale noise in the *inner* per-pixel fit's own convergence
   tolerance leaking into the second-difference formula's
   subtraction-heavy cancellation), implying a wildly oversized step that
   no amount of retry-limited damping alone could shrink into range.
   Fixed with a standard trust-region cap: a single step may not shrink a
   shared parameter to less than half, or grow it to more than double,
   its current value, regardless of what the raw formula proposes.

Both fixes together turned a search that could get permanently stuck at
more than 2x the true value (confirmed via a direct fixed-tau chi-square
landscape scan showing a single, clean minimum at the true value being
missed entirely) into one that reliably finds the actual minimum of its
own objective, independent of iteration budget or window choice.

**A residual, smaller (~1-4% relative) discrepancy remains** between the
shared-tau search's answer and a dataset's true generating tau on
synthetic single-realization test data, confirmed to be neither
statistical noise (stable across a 1000x photon-count range once a
`uint16`-overflow artifact in the test's own data generator was found and
fixed) nor further optimizer misbehaviour (an independent fixed-tau
chi-square landscape probe confirms the search lands on the objective's
genuine minimum). It traces to the fixed-tau profile fit (holding the
shared parameters fixed and only re-optimizing per-pixel amplitude/
background) landing at a slightly different optimum than the fully-free
joint fit under the same explicit fit window -- a small, real precision
characteristic of the profile-search technique itself, not a further bug
found before time ran out on this investigation. For the 2-parameter
stretched-exponential case specifically, a substantially larger
seed-dependent spread was also found between entry points (training vs.
single-image apply), consistent with tau and h's well-known, genuine
correlation for this model (a shallow ridge in the objective, not a
separate defect) -- exactly the caveat given when this feature was first
proposed. `verify_stretched_exp.py` in the scratchpad documents all of
this with realistic, explained tolerances rather than either hiding it or
overclaiming precision.

**Files changed**: `native/flimfit_labview.c` (fitType/sharedCount
generalization throughout the outer search and fast inner solve, both
bug fixes above), `flim_app/models.py`, `flim_app/global_model.py`,
`flim_app/processing.py`, `flim_app/native_backend.py` (hash pin
`86737ce6a719122c33c76fca786462f28ac2bce5a254b60f6fd0a7a498cd9f7c`),
`flim_app/gui.py` (new "Distribution model" combo + force/disable logic
for Model order), `tests/test_processing.py` (matching hash pin).

## Three fixes: mouse-wheel-changes-dropdown-values, settings presets can now remember loaded files, background-subtracted pre-analysis threshold preview (2026-09-03)

### 1. Scrolling over a dropdown was silently changing its value

**Root cause**: `ttk.Combobox` has value-cycling on mouse-wheel-hover
baked into its own default Tk widget-class bindings -- not something
this app added. Scrolling the sidebar (or anywhere else) while the
cursor happened to be over any dropdown changed that dropdown's
selection, with no click and no keyboard focus involved.

**Fix**: `FLIMStudioApp._disable_combobox_wheel(combo, canvas=None)` --
binds `<MouseWheel>`/`<Button-4>`/`<Button-5>` at the *instance* level
(which Tk processes before the widget's class-level defaults) and
returns `"break"`, stopping the class default from running at all. When
the combo lives inside the sidebar's scrollable canvas, the handler
manually replicates that canvas's own scroll first -- so scrolling over
a dropdown still scrolls the panel, it just never changes the dropdown's
value; outside any scrollable area (`canvas=None`), the value-cycling is
simply suppressed. Applied to every `ttk.Combobox` in the app: both
sidebar helper methods (`_combo`/`_compact_combo`), the two standalone
sidebar/main-window combos (IRF source, the SECOND MAP mode selector),
and the Leica export dialog's two combos (own small local copy of the
same fix, `leica_export_gui._disable_combobox_wheel`, since that dialog
has nothing scrollable to redirect to anyway).

**Follow-up, same day**: the sidebar's own scroll handler had a second,
separate bug -- `canvas.bind_all("<MouseWheel>", ...)` fires for *every*
widget in the whole application, not just the sidebar, so scrolling
anywhere else in the window (an image view, a dialog, ...) also scrolled
the sidebar underneath it. `bind_all` is still needed (a plain
`canvas.bind` only fires for the canvas's own background, not its child
widgets), so the fix keeps it but adds a bounds check inside the handler
itself: compare `event.x_root`/`event.y_root` against
`canvas.winfo_rootx()`/`winfo_rooty()` + `winfo_width()`/`winfo_height()`,
and only scroll if the cursor is actually within the sidebar's own
on-screen rectangle. Verified against a real, properly-sized
`FLIMStudioApp` window (a point inside the sidebar's real screen bounds
passes the check, a point outside it doesn't).

### 2. Settings presets can now remember which files were loaded

**What changed**: `settings_ini.save_settings_ini` gained an optional
`file_paths: LoadedFilePaths | None` keyword (a new small dataclass:
`decay_path`/`irf_path`/`background_path`) that writes those into a
*separate* `[flim_studio.loaded_files]` section, and a new
`load_file_paths_ini()` reads it back (returning all-empty paths, not an
error, when that section is absent -- an older preset, or one saved
without it). `AnalysisSettings` itself stays untouched -- deliberately
kept as a pure "conditions" object with no notion of specific loaded
files (see the module's own docstring for why: things like
`ManualModelDialog`/`.normalized()` build/consume it with no dataset in
scope at all, and mixing the two concerns would complicate both).

`gui.py`'s four preset call sites (`_save_settings_ini_dialog`,
`_save_settings_as_default`, `_load_settings_ini_dialog`,
`_load_default_ini_at_startup`) now pass/read this alongside the regular
settings. Loading restores decay first, then IRF/background (which
validate their time-bin count/width against the now-loaded decay) --
reusing the exact same loading/validation `open_irf`/`open_background`
already used, factored out into `_load_irf_path`/`_load_background_path`
so there's exactly one implementation of "load this IRF file and check
it" rather than two. A stale or moved file is skipped with a warning,
not a hard failure -- restoring what's still possible beats aborting the
whole preset load over one missing path.

**Verified**: a standalone round-trip test (save with `file_paths=`,
reload, `load_file_paths_ini` gives back exactly what was saved; save
without it, `load_file_paths_ini` correctly returns all-empty) plus a
real `FLIMStudioApp` instantiation exercising `_show_raw_intensity`
end-to-end (see check 3 below, same verification script covers both new
features' actual GUI code paths, not just the underlying pure
functions).

### 3. Pre-analysis "Intensity with threshold" preview now reflects background subtraction

**Why**: previously `_show_raw_intensity` (which feeds both the
pre-analysis Intensity view and the "Intensity"/"Intensity with
threshold" secondary map) used the *raw* decay sum, with no background
subtraction at all -- so a threshold picked against that preview could
be meaningfully wrong once "Analyze" actually ran its own
background-subtracted fit, since the same nominal count value means
something different before vs. after subtraction.

**Fix**: `_show_raw_intensity` now reuses `resolve_background_multiplier`
+ `_prepare_background` (the exact same building blocks
`analyze_dataset` itself uses) to subtract the currently-loaded
background from the *full* decay cube -- per time bin, clip to >=0, then
sum over time -- before spatial binning, live as the user tunes
background scale mode/multiplier/scalar counts-per-bin (new trace
callbacks on those three sidebar variables, and a call after loading a
new background file). Per-bin subtract-then-clip, not a subtraction of
already-time-summed totals, matches `analyze_dataset`'s own order
exactly -- clipping is non-linear with respect to that order, so doing
it the other way would give a different (wrong) answer whenever
background exceeds signal in some bins but not overall. The image
caption now reads "Background-subtracted (×N) photon-count projection"
when a background is loaded, "Raw photon-count projection" otherwise, so
it's visible at a glance which one you're looking at. Falls back to the
raw cube (with a status note, not a crash) if background resolution
fails for any reason (mismatched shapes, bad metadata) -- the real
"Analyze" run still surfaces that properly.

**Verified** (`verify_bg_subtracted_preview.py`, a real `FLIMStudioApp`
instance, not a mock): background-subtracted preview matches a
hand-computed expectation using the same `resolve_background_multiplier`/
`_prepare_background` calls exactly; with no background loaded, falls
back to the plain raw sum exactly as before this change.

Full test suite: 63/64 (same pre-existing external-fixture gap), no
regressions, across all three fixes. Same `FLIM_Studio_Python_v0.15.0`;
`v0.10.0`/`v0.13.0`/`v0.14.0` untouched.

## "Set global model manually" -- fix global tau values with no search at all (2026-09-03)

**Why**: user asked how to set the global tau values directly. Two
different things were already possible but easy to conflate:
`initial_taus_ns` *seeds* the Marquardt search (still moves from there);
only "FLIMLib — apply trained global model" (`flimlib_apply_global`)
truly holds taus fixed, fitting just amplitude/background per pixel
against them -- but the only way to get taus into that mode was to train
a model first (which searches) or hand-write a JSON model file. No GUI
path let you just type numbers and go straight to "fixed, no search."

**What was added**: a new dialog (`manual_model_gui.ManualModelDialog`,
opened via a new "Set manually…" button next to "Load model…"/"Train
new model…" in the sidebar's GLOBAL MODEL section). Type 1-3
comma-separated lifetimes, click Create -- builds a `TrainedGlobalModel`
directly via the existing `build_trained_model` (same one training
itself calls), with the typed taus used verbatim as `global_taus_ns`, no
native/DLL call made at all. Every other field (noise model, background
mode, IRF mode, `irf_integrate_per_column`, ...) is pulled from the
current sidebar settings, exactly like the Train dialog already does --
no duplicate control set. `global_stats` is filled with NaN throughout
(no fit ran, so no real objective/iteration-count to report --
`build_trained_model`'s own `stat()` helper already turns a non-finite
entry into `None` rather than a fabricated number) and `training_images`
is empty (nothing was actually trained on). Save/Use-now buttons reuse
the exact same `save_global_model`/`app._use_trained_model` calls the
Train dialog uses, so a manually-set model behaves identically to a
trained one everywhere downstream (saving, reloading, applying, the
sidebar mirroring its conditions when loaded).

**Verified** (`verify_manual_model.py`): built a model with taus typed
as `(4.0, 1.0)` ns, saved it to disk, reloaded it, applied it
(`flimlib_apply_global`) to a synthetic bi-exponential dataset. Every
selected pixel's `tau_components_ns` came back *exactly* `(4.0, 1.0)` --
no drift, no search, confirming the fixed-tau apply path genuinely never
touches the taus. Recovered amplitude ratio 2.064 vs the true 2.000 (a
photon-noise-scale difference, not a bug). Full test suite: 63/64, same
pre-existing gap, no regressions -- confirmed both via `ast.parse` and a
real import of `gui.py` (which now imports the new dialog module) after
adding it.

## "Integrate IRF per column" extended to phasor (2026-09-03, same day follow-up)

**Why**: the user asked whether the fix below also covers phasor and
iterative reconvolution. Answer worked out to be mode-dependent in a way
worth recording:

- **The *speedup* (IRF grouping, see the v0.14.0 entry further below)
  only ever applies to the global/shared-tau fit family** (Global
  training, "FLIMLib — global", apply-trained-model) -- it works by
  reusing one convolution across many pixels that share the *same trial
  tau* during the outer search. Iterative reconvolution
  (`flimlib_reconvolution`) genuinely fits tau freely per pixel, so
  there's no shared trial value to reuse a convolution across; this
  isn't a gap, it's just a different problem shape.
- **The *SNR benefit* (a cleaner, integrated IRF) is a separate,
  independent thing**, and applies anywhere a per-pixel IRF is used for
  anything -- including phasor's own per-pixel IRF correction
  (`phasor_lifetimes` -> `_reference_rows`, which divides each pixel's
  phasor by its own reference IRF's phasor). That path reads `irf_data`
  directly and doesn't consult `fit_irf_mode` at all -- phasor decides
  per-pixel-or-not purely from the array's own shape -- so it was
  initially missed: the integration call was only wired into the
  FLIMLib-native branch of `analyze_dataset`.

**Fix**: moved the `integrate_irf_per_column` call to apply once, to
`irf_data` itself, right after `analyze_dataset`'s own shift/spatial-bin
step and before *any* consumer (phasor, then later the FLIMLib-native
branch) reads it -- rather than duplicating the call per consumer. The
FLIMLib branch's own (now-redundant) call was removed; `native_irf`
simply inherits the already-integrated `irf_data`.

**Verified**: `verify_phasor_integration.py`, same synthetic per-column-
true-timing / per-pixel-noise scenario as the training check below, run
through `phasor_phase` instead. Within-column standard deviation of the
corrected phasor lifetime (the noise this feature targets -- pixels
sharing a detector/column should agree; scatter between them is
measurement noise) drops from **0.134 ns to 0.021 ns** (~6.3x), mean
lifetime essentially unchanged (3.17 -> 3.21 ns, both close to the true
3.0 ns). Training verification re-run unchanged (still `800/800` ->
`40/800`, same recovered tau) -- confirms the refactor didn't disturb the
training path. Full suite: 63/64, same pre-existing gap.

## "Integrate IRF per column" (v0.15.0): restores the v0.14.0 grouping speedup for real sweeping-linear-detector data, and matches the source LabVIEW tool's own IRF handling (2026-09-02/03)

**Why this exists**: after v0.14.0 shipped, the user reported training was
still very slow on real data despite the IRF-grouped fast path below.
Three real, separate bugs/gaps surfaced while chasing this down (all
fixed here, all worth knowing about if this ever recurs):

1. **`default.ini` pinned `native_dll_path` to `v0.10.0`'s DLL, an
   absolute path.** This one line silently overrode every rebuild all
   session -- the app was never actually running any of the v0.13.0/
   v0.14.0 work when launched normally. Fixed by blanking that field so
   it falls back to auto-detection.
2. **`app_root()`'s DLL candidate (`flim_app/native/flimfit_labview.dll`,
   present for PyInstaller packaging) is checked before the project-root
   `native/flimfit_labview.dll` this session's rebuilds actually wrote
   to**, and had gone stale. Fixed by keeping both copies in sync;
   **remember to copy any future DLL rebuild to both locations**.
3. **A diagnostic bug**: the `irf_groups=<n>/<total>` counter added to
   `native_worker.py`'s log line to debug this was only wired up in
   `flim_fit_global_pixels_v2`, not `flim_train_global_model_v1` --
   always read back 0 for `train_global_model` regardless of the real
   (correctly working) group count. Fixed.

With those out of the way, the real diagnostic came back
`irf_groups=1996/1998` on the user's actual RA2 dataset -- essentially
no sharing at all. The reason: `build_irf_groups` (see the entry below)
groups pixels by *exact* IRF content, but the per-pixel IRF cube
`flim_app` was handed was raw, per-pixel, per-sweep-pass acquisition
data -- genuinely noisy, no two pixels byte-identical, even within one
physical detector's column. The user's original LabVIEW tool never fed
the fitter this raw data directly: "I simply integrated these then
repeated the same irf across the full sweeping scan to provide more
signal" -- i.e. for a sweeping linear-array sensor (512 detectors), each
column is one fixed detector and every row is another sweep pass across
it, so summing a column's raw acquired trace across every row recovers
one much-higher-SNR curve per detector, which is what should represent
that detector's IRF. This preprocessing step was simply never ported to
this Python codebase.

**What changed**: `processing.integrate_irf_per_column(irf_cube)` sums a
`(y, x, time)` per-pixel IRF cube's row axis per column, then broadcasts
that one curve back across every row of the column (exact byte
duplication within a column, by construction). Gated behind a new
`AnalysisSettings.irf_integrate_per_column` field (default `False` --
never silently changes behaviour for a per-pixel IRF that's genuinely
supposed to vary pixel-to-pixel), applied only when `fit_irf_mode == 1`,
right before the per-pixel IRF cube is handed to the native layer -- both
in `_preprocess_training_image` (global training) and `analyze_dataset`
(single-image analysis, including applying a trained model). Wired
through: the sidebar (a checkbox + explanatory label right under "IRF
mode"), `.ini` settings save/load (fully generic already -- no extra code
needed there), the CLI (`--irf-integrate-per-column`), and
`TrainedGlobalModel` (a new field, so applying a model later reuses
whatever setting it was actually *trained* with, not whatever the live
sidebar happens to show -- the same pattern `irf_mode` itself already
followed). No native/DLL changes at all; this is a pure Python
preprocessing step.

**Verified** (`verify_irf_per_column.py`, a synthetic sweeping-detector
scenario -- per-column true IRF timing, independent Poisson noise per
pixel, no exact duplication anywhere, reproducing the real 1996/1998
failure mode exactly):

| | `irf_groups` | fit_call time (800 px, 40 columns) | recovered tau |
|---|---|---|---|
| toggle off | `800/800` (matches the real report) | 4.67s | 3.5735 ns |
| toggle on | `40/800` (exact column count) | 0.74s | 3.5709 ns |

Grouping collapses to *exactly* the true column count, ~6x faster fit
call at this small scale (20 rows/column -- real 512-row images would
see a proportionally much larger win), and the recovered lifetime is
essentially unchanged between the two (this is a speed/SNR improvement,
not a fit-corrupting one). A separate unit check confirms every row
within a column is byte-identical post-integration. Full test suite:
63/64 passed (same pre-existing external-fixture gap), no regressions.

`FLIM_Studio_Python_v0.15.0` is a new isolated copy from `v0.14.0`
(itself untouched, as is `v0.13.0`/`v0.10.0`); the native DLL is
unchanged from v0.14.0's (same hash), just carried forward with both
locations already in sync and `default.ini`'s `native_dll_path` already
blanked.

## IRF-grouped fast inner solve (v0.14.0): ~150x faster than the Marquardt-only search on top of it, by reusing FLIMLib's own real convolve-once machinery per unique IRF (2026-09-02)

**Why this exists**: after the Marquardt-damped-Newton outer search below
shipped, the user asked how to speed things up further, and specifically
whether new code could perform the global fit while still taking
individual IRFs per decay/detector. Investigating FLIMLib's own
`GCI_marquardt_global_exps_instr` (`EcfGlobal.c`) turned up the actual
remaining inefficiency: at each trial shared-tau, the outer search
scored every selected pixel by calling `fit_fixed_tau_pixel`
independently -- and that goes through the *generic* single-transient
reconvolution engine (`GCI_marquardt_fitting_engine`), which redoes the
IRF convolution from scratch on every Marquardt inner iteration of every
single pixel, even when many (or all) pixels share the same IRF. The
real global routine avoids exactly this: `GCI_marquardt_global_exps_calculate_exps_instr`
convolves the exponential basis for the current tau's **once**, and
`GCI_marquardt_global_exps_do_fit_single` then fits each transient's
Z/amplitudes against that fixed basis with no further convolution at
all. Both are file-scope (non-`static`) functions in `EcfGlobal.c`, not
exposed via `EcfGlobal.h`, so they're forward-declared directly in
`flimfit_labview.c` and called with external linkage -- no reimplementing
FLIMLib's numerics, just reusing the real building blocks.

**What changed**: `score_global_taus_with_irfs` now buckets selected
pixels by IRF *content* (`build_irf_groups` -- FNV-1a hash, memcmp on
collision, CSR-style member lists for O(1)-amortised per-pixel group
walking) before scoring each trial tau. A single shared IRF (`irfMode=0`,
the common case) collapses to one group covering every pixel --
matching the real global routine's own convolve-once behaviour exactly.
Per-pixel/per-detector IRF (`irfMode=1`) with a handful of distinct
per-detector IRFs collapses to a handful of groups. Genuinely
all-distinct per-pixel IRFs degrade gracefully to one group per pixel,
i.e. the old per-pixel cost -- never worse. Groups are built once, before
the outer Marquardt-damped-Newton search starts (the grouping doesn't
depend on tau), and reused across every trial-tau evaluation via a new
`groups` field threaded through `global_tau_objective_context`. A
per-pixel fallback (the original independent `fit_fixed_tau_pixel` loop)
stays in place for nBins exceeding FLIMLib's internal `MAXBINS` (2048)
scratch-array limit -- a defensive guard, not expected to trigger in
practice.

**Measured result** (same synthetic 3-image tri-exponential scenario as
below, default seed, three DLL builds compared with explicit paths to
rule out any DLL-resolution ambiguity -- see the gotcha below):

| search | wall time | recovered taus (ns) |
|---|---|---|
| 1. Nelder-Mead (original) | 143.0s | `[3.46, 1.32, 1.22]` |
| 2. Marquardt-damped-Newton only (no grouping) | 125.4s | `[5.25, 2.22, 0.49]` |
| 3. Marquardt + IRF-grouped fast inner solve | **0.80s** | `[3.34, 2.23, 1.07]` |

Roughly 150x faster than the Marquardt-only search, ~180x faster than
the original Nelder-Mead, on this test (which collapses to a single IRF
group despite `irfMode=1`, since the synthetic per-pixel IRF happens to
be uniform across pixels -- representative of both the plain
single-shared-IRF case and a single-detector's-worth of pixels in a
multi-detector dataset). A real multi-detector dataset would see a
smaller but still large speedup, roughly proportional to pixels-per-
detector. Full test suite: 63/64 passed (same pre-existing external-
fixture gap), no regressions -- see the DLL-hash-pin gotcha below for the
one test that needed updating and why.

**The three variants recover three different (still not fully
separated) tau sets**, each internally deterministic and reproducible
(verified via repeated runs). This is expected, not a red flag: the
inner per-pixel solve differs numerically between variants (different
convergence criteria between `GCI_marquardt_fitting_engine` and
`GCI_marquardt_global_exps_do_fit_single`, even though both solve the
same fixed-tau linear sub-problem), and on the near-degenerate landscape
this specific "3 pure reference images" scenario has (see the
modelling-limitation conclusion further below), tiny objective
differences are enough to walk the outer search into genuinely different
points of a shallow valley. Three unrelated numerical paths all landing
in different-but-similarly-unseparated points is, if anything, further
evidence for that conclusion, not against it.

**Gotcha found and fixed along the way -- a real, pre-existing DLL-
resolution footgun, not something introduced by this work**: when no
explicit `native_dll_path` is given, `native_backend.candidate_dll_paths`
checks `app_root()/native/flimfit_labview.dll` *before*
`<project_root>/native/flimfit_labview.dll`. `app_root()` resolves to
`flim_app/`'s own directory when running from source, so the actual
first candidate is `flim_app/native/flimfit_labview.dll` -- a *second*,
separate copy of the DLL (present for PyInstaller packaging, where
`Path(__file__)`-based paths can't locate bundled data files) that
happens to also sit in the source tree. This copy was never touched by
any of this session's rebuilds (which only ever wrote to the top-level
`native/`), so it silently still held the *original*, pre-any-of-this-
session DLL (hash `067f7e338d78...4327ffb`, dated 12/08/2026) -- meaning
every default-path fit call (this project's own verification scripts
included, whenever they omitted an explicit `native_dll_path`) was
silently exercising the wrong DLL the whole time, while explicit-path
calls correctly used whichever DLL had actually been rebuilt. This is
inherited from v0.10.0's own layout (present since project creation, not
something the Nelder-Mead/Marquardt work introduced) and would affect
`v0.13.0` too if it's ever run from source without an explicit path.
Fixed here by copying the current `native/flimfit_labview.dll` to
`flim_app/native/flimfit_labview.dll` too, and confirmed via a
default-path (no explicit `native_dll_path`) run matching the explicit-
path timing/result exactly post-fix. **Whenever the DLL is rebuilt again,
copy it to both locations**, or better, always pass an explicit
`native_dll_path` in verification scripts to sidestep this entirely.
`tests/test_processing.py::test_native_analysis_pipeline` hardcodes the
default-resolved DLL's hash as a sanity pin; it needed updating to the
new hash for the same reason `native_backend.py`'s `_PATCHED_DLL_SHA256`
does on every legitimate rebuild.

## Marquardt-damped Newton shared-tau search (v0.14.0): 3x-plus faster than Nelder-Mead, and independently confirms the modelling-limitation conclusion below (2026-09-01/02)

**Why this exists**: after the Nelder-Mead work below shipped, the user
asked directly why TRI2 (and this app's own single-IRF-only `v1` path,
`GCI_marquardt_global_exps_instr`) can run global fits across multiple
images/all pixels much faster than the `v2`/`train_global_model_v1`
custom outer search. Answer: `v1` is a real joint Levenberg-Marquardt
solve with an analytic Jacobian -- it *knows* which direction to step.
Nelder-Mead (and coordinate descent before it) is derivative-free: every
trial point only reveals its own objective value, so the search has to
blindly probe multiple points per iteration just to infer a direction,
and each probe re-runs a full per-pixel FLIMLib fit across the whole
pooled dataset. That's the entire cost gap -- not implementation quality.

**What changed**: in `native/flimfit_labview.c`, `nelder_mead_min` was
replaced by `levenberg_marquardt_min` -- a genuine Marquardt-style damped
Newton search over the shared tau's: build a finite-difference gradient
and Hessian of the pooled-chi-square objective (`score_global_taus_with_irfs`,
unchanged) at each iterate, solve `(H + lambda*diag(H))*delta = -g`,
accept/shrink lambda if the step improves, reject/grow lambda otherwise --
Marquardt's original 1963 trust-region logic, just with numerical
derivatives instead of an analytic Jacobian over raw residuals (an
analytic per-timepoint Jacobian would need a much bigger rewrite; this
gets the same curvature-following behaviour far more cheaply). This still
deliberately does not call `GCI_marquardt_global_exps_instr` directly --
that routine only accepts one IRF per call, so it still can't support
per-pixel/per-detector IRF (`irfMode=1`), which is the entire reason `v2`
exists. Built in a fresh isolated copy, `FLIM_Studio_Python_v0.14.0`
(copied from `v0.13.0`; `v0.10.0` and `v0.13.0` both untouched). DLL
rebuilt via the same confirmed MSVC+CMake toolchain; old (Nelder-Mead)
DLL kept as `native/flimfit_labview.dll.pre-levenberg-marquardt-backup`
(SHA-256 `4cc8cb91d7e681464db51fe6fde736e75d57344e55524662c0c0b357ec11337b`);
new DLL SHA-256 `0550ec7ef54cbd1babc88ba385e1e27f8600d5098e5ccbc46eb923636633eee9`,
pinned into `flim_app/native_backend.py`'s `_PATCHED_DLL_SHA256`.

**Measured result** (same synthetic 3-image tri-exponential scenario as
below, same v0.14.0 Python layer, only the DLL swapped via
`native_dll_path` to isolate the search-algorithm effect):

| scenario | Nelder-Mead (old) | Marquardt-damped Newton (new) |
|---|---|---|
| default fallback seed | 177.7s, recovered `[3.46, 1.32, 1.22]` | **57.9s**, recovered `[3.08, 1.32, 1.31]` |
| seeded exactly at ground truth | **timed out after 1800s, had to be killed -- never converged** | **78.9s**, recovered `[3.36, 1.43, 1.41]` |

Roughly 3x faster in the easy case, and the old search's pathological
case (never converging at all, even given the literal right answer as
its starting point) simply doesn't happen with the new search -- it
converges in comparable time regardless of seed, because it follows
curvature instead of needing the simplex to accidentally wander into a
good region.

**This also independently strengthens the modelling-limitation
conclusion below**, rather than reopening it: two structurally very
different search algorithms (blind simplex vs damped Newton with real
second-derivative information), started from the literal true answer in
one case, both converge to essentially the same "2-components-merged"
point (`[3.36, 1.43, 1.41]` vs `[3.38, 1.43, 1.41]` from the seeded
Nelder-Mead run recorded below). Two unrelated algorithms agreeing on
where the minimum is, from the true answer as a starting point, is
strong evidence that's genuinely where the minimum sits for this data --
not an artifact of either search's own quirks. The practical recommendation
(3 separate mono-exponential fits for 3 already-pure reference images,
not one joint tri-exponential training run) stands unchanged; this
search-algorithm swap is a real, independent win on speed for the cases
global/shared-tau training is actually designed for (species genuinely
mixing within pixels), where it isn't fighting an identifiability problem
in the first place.

Full test suite: 63/64 passed against v0.14.0, same pre-existing gap
(missing external `validation_data/single_decay_mono.npz`) as every
prior run -- no regressions from the search-algorithm swap.

## Nelder-Mead shared-tau search: built, verified correct -- the "3 species" problem turned out to be a modelling limitation, not a search-algorithm one (2026-09-01, conclusion)

**Update superseding everything below in this section.** The amplitude-
non-negativity hypothesis was checked directly against the FLIMLib source
and is **ruled out**: `EcfUtil.c`'s `check_ecf_params` (the
`ECF_RESTRAIN_DEFAULT` restraint `fit_fixed_tau_pixel` actually uses) has
`#define MIN_A 0` -- amplitudes are already correctly constrained
non-negative for every per-pixel fit, exactly as they should be.

The synthetic verification run that first showed a "3.38/3.36 ns
collapse" also turned out to have its own bug -- `AnalysisSettings.dt_ns`
defaults to 0.390625 and is **not** auto-derived from a dataset's own
`dt_ns` (that's the caller's job; the GUI does it via `_set_dataset()`
populating `dt_var`, a bare script calling `train_global_model` directly
does not get that for free) -- the verification script never set it,
producing a ~2.6x systematic scaling error that made *every* earlier
result in this investigation meaningless, including the "does it stay at
the seeded ground truth" diagnostic. Fixed and re-run properly, with a
sanity check first: **3 separate single-image mono-exponential fits
(the existing, unmodified `flim_fit_pixels_v2` per-pixel path) recover
4.0/2.0/0.5 ns to within ~0.01 ns** -- confirming the underlying fitting
machinery, IRF, and synthetic data are all sound in isolation.

With that same (corrected) data, the joint tri-exponential *global*
training -- Nelder-Mead outer search, real per-pixel Marquardt inner
solve, non-negative amplitudes, everything working as designed -- still
does not recover the 3 species distinctly, **even when seeded exactly at
the true values**: it moves away to a "2 of 3 components merged" answer.
Nelder-Mead can only ever report a point *at least as good as* every
point it visited, including the seed -- so this isn't the search missing
a better answer, it's the search correctly finding one: for this noisy,
finite-photon-count pooled dataset, a compromise set of taus genuinely
achieves a **lower** summed chi-square than the true separated values do.
That's a classic symptom of an over-parameterised model (3 free shared
lifetimes, fully free per-pixel amplitudes) being under-identified by
this kind of data, not a bug in either search algorithm -- a better
*optimizer* (this one, or a real Marquardt-over-shared-tau one) would
find the same minimum, because it's genuinely the minimum of this
objective for this data.

**Practical conclusion for "3 separate pure single-species reference
images"**: global/shared-tau training is the right tool when species
genuinely *mix within individual pixels* (e.g. free/bound NAD(P)H
co-existing everywhere) and no single pixel can resolve them alone --
recovering one shared set of components across many mixed pixels is
exactly what it's built for. It is very likely the *wrong* tool for 3
already-separate, already-pure reference images, where each image
already contains everything needed for its own independent answer, and
pooling them into one joint estimation just adds cross-image ambiguity
that a noisy, finite dataset can't always resolve. **Three separate
mono-exponential reconvolution fits (one per image, already fully
working)** is the better-conditioned, already-validated way to get those
3 numbers for that specific scenario.

The Nelder-Mead change itself stays in place -- it's still strictly
better than the coordinate descent it replaced (fixes the confirmed
pixel-starvation bug, no regressions, genuinely finds better/equal
optima given the same objective) -- this conclusion is about what global
training can identify from this *kind* of data at all, not a flaw
specific to this search algorithm.

## Nelder-Mead shared-tau search: built, verified not to regress, does NOT yet fix the reported accuracy issue (2026-09-01, in progress)

Follow-up to the section directly below. Implemented in this
(`FLIM_Studio_Python_v0.13.0`, an isolated copy -- `v0.10.0` untouched)
copy: `flim_fit_global_pixels_v2` and `flim_train_global_model_v1`'s
coordinate-descent shared-tau search replaced with a Nelder-Mead simplex
search over the same objective (`score_global_taus_with_irfs`, itself
unchanged -- still a real per-pixel FLIMLib Marquardt fit via
`fit_fixed_tau_pixel` with tau held fixed, still supports per-pixel/
per-detector IRF). Built with the MSVC toolchain confirmed available on
this machine (`native/build/`, Release config); the rebuilt DLL is now at
`native/flimfit_labview.dll` (old one backed up alongside as
`flimfit_labview.dll.pre-nelder-mead-backup`); `native_backend.py`'s
`_PATCHED_DLL_SHA256` updated to match (the null-sigma-pointer fix that
hash guards is untouched by this change).

**Verified, confirmed working:**
- Full existing test suite: 63/64 passed against the new DLL (same single
  failure as before the change -- a missing external `validation_data`
  fixture, unrelated). No regressions.
- The *pixel-starvation* bug this was originally chasing (a later
  training image silently getting zero pixels because
  `max_global_pixels`'s sequential fill-until-cap exhausted the whole
  budget on an earlier image -- see the 0.12.8-0.12.10 Python-side fixes)
  stays fixed: a fresh 3-image synthetic training run shows all three
  images contributing a real share of pixels (e.g. 66-84/100 each, not
  0/100 for two of them).

**NOT yet confirmed working -- the actual thing that was asked for:**
A targeted reproduction of the reported scenario (3 clean single-species
mono-exponential "detector" images at 4.0/2.0/0.5 ns, tri-exponential
global training, `max_global_pixels=0`) still does **not** recover the 3
known lifetimes with useful accuracy. Diagnostic run seeding
`initial_taus_ns` at the *exact* ground truth `(4.0, 2.0, 0.5)` --
specifically to test whether that point is even a stable local optimum
of the objective, independent of which search algorithm is used --
still drifted away to `(3.38, 3.36, 0.5)`: the 0.5 ns component held
still, but the 4.0 and 2.0 ns components collapsed toward each other
rather than staying separated. The *old* coordinate-descent DLL, run on
the exact same synthetic data with no seed, landed at values in the same
ballpark of wrongness.

**Working hypothesis, not yet confirmed**: this points to the objective
itself having a genuine degeneracy for this kind of data -- pooling
several distinct single-species sub-populations into one tri-exponential
fit with *unconstrained per-pixel amplitudes* (`Ecf.h`'s
`ECF_RESTRAIN_DEFAULT` is documented only as "very wide sensible limits",
not explicitly amplitude non-negativity) may let two components trade
off / partially cancel against each other while still fitting the pooled
data about as well as the true separated solution -- meaning a smarter
*search* alone (Nelder-Mead vs. coordinate descent) doesn't fix it, since
both would need to escape or avoid that degeneracy, not just search more
efficiently around it. Not confirmed: haven't yet checked whether
amplitude signs are actually unconstrained, or tested on real (not
synthetic-uniform) detector data, where genuine per-pixel variation might
condition the problem differently than this deliberately idealised
(spatially uniform per image, noise only) synthetic test.

**Left in place, not reverted**: the DLL swap and algorithm change are
still an improvement (fixes the confirmed pixel-starvation bug, no
regressions) even though the deeper accuracy question remains open --
kept in `v0.13.0` for continued investigation rather than rolled back.

## Global fitting: the real FLIMLib/TRI2 algorithm is bundled but unused (2026-08-31)

FLIMLib is not an independent reimplementation -- its own site
(flimlib.github.io) states plainly: "The core fitting routines from TRI2
are now open source in the FLIMLib library," and "FLIMLib is used for
FLIM functionality in the Time Resolved Imaging (TRI2) software." The
copy of that source this project links (`native/flimlib_source/src/main/c/`)
includes `EcfGlobal.c`'s `GCI_marquardt_global_exps_instr` -- a genuine
joint Levenberg-Marquardt global-analysis fit (TRI2's own algorithm, not
a reimplementation), operating on a flat list of per-pixel decay traces
that all share one set of nonlinear lifetimes while their linear
amplitude/background parameters are solved per pixel.

`native/flimfit_labview.c` actually wires this real routine up --
in **`flim_fit_global_pixels_v1`** (single-image, no `imageCount`
parameter, and no per-pixel IRF: `GCI_marquardt_global_exps_instr`'s own
signature in `EcfGlobal.h` takes exactly one `float instr[]` array for
the whole call, no per-trace variant exists anywhere in FLIMLib). But
FLIM Studio's Python layer (`native_backend.py`) never calls it --
`_configure_signatures()` only binds `flim_fit_global_pixels_v2` and
`flim_train_global_model_v1`, and **both of those instead implement
their own custom coordinate-descent search over the *shared τ values
only*** (`score_global_taus_with_irfs`: a per-tau ±step exponential
trial with shrinking step size on stagnation).

**Important correction/refinement**, found while tracing this further:
the *per-pixel* step inside that search (`fit_fixed_tau_pixel`) is
**not** a custom approximation -- for each trial τ set it calls the real
`GCI_marquardt_fitting_engine` (genuine FLIMLib Marquardt fitting) with
the τ parameters' `paramfree` flags set to 0 (fixed), so it's a properly
correct fixed-τ linear (amplitude/background) solve, and it already
takes a **per-pixel** IRF pointer (`irfPixel`) -- `irfMode=1` ("Per-pixel
IRF" in the sidebar) already works correctly today, real Marquardt fit
and all. Only the *outer* search over the 2-3 shared τ values themselves
is the slow, custom part: every one of its trial steps re-scores the
*entire* selected pixel pool by calling `fit_fixed_tau_pixel` once per
pixel, rather than using gradient information (the way
`GCI_marquardt_global_exps_instr`/a proper VARPRO-Marquardt over just
the shared τ's would) to reach the optimum in far fewer trials. That's
why this project's global fit/training needed a `max_global_pixels` cap
for performance at all (see the 0.12.8-0.12.10 release notes), and
presumably why TRI2 itself, built directly on the same core routines,
doesn't appear to need one.

**Practical upshot (2026-08-31, in response to a per-detector-IRF
question for a linear-array/RA2-style sensor)**: per-detector IRF
support genuinely doesn't need any DLL work at all -- give every pixel
belonging to one detector's row/channel that detector's own IRF trace
(replicated across the row) in the loaded IRF file, matching this app's
existing "Per-pixel IRF" mode and its already-existing per-row detector
convention (the same one the detector-row interpolation map feature
already uses). `v1` couldn't do this even if it were revived (one IRF
per call, full stop) -- it's specifically `v2`/training's per-pixel path
that makes it possible, and it's already real Marquardt fitting under
the hood for the linear step.

Why `v1` was set aside for `v2` in the first place isn't documented
anywhere found in this repo -- the README only explains `v2` as adding
*length-checked* buffer safety and fixing two specific defects (a null
sigma pointer for constant weighting, an inclusive/exclusive fit-window
off-by-one), which is a different concern from the algorithm/IRF-mode
change. Given `GCI_marquardt_global_exps_instr`'s one-IRF-per-call
limit, adding per-pixel IRF support at all *required* moving off it,
regardless of any speed considerations.

**What's still an open, larger project**: only the *outer* shared-τ
search is slow, and speeding *that* part up while preserving per-pixel/
per-detector IRF support is a genuinely bigger algorithmic undertaking
than "just call `GCI_marquardt_global_exps_instr` like `v1` does" --
that routine can't take a per-pixel IRF at all, so a real fix would mean
a *new* gradient-based (Levenberg-Marquardt-style) search over just the
shared τ's, where each gradient/objective evaluation still calls the
existing (already-correct) `fit_fixed_tau_pixel` per pixel, just far
fewer times per run than a coordinate-descent grid search needs. **Not
attempted here**: this would mean modifying and rebuilding the compiled,
GPL-linked native DLL, with no C toolchain or way to numerically verify
the result available in this environment (see `BUILDING.md`'s existing
caveat, which already covers the DLL-rebuild history for the
length-checked v2 audit). Worth pursuing in a session/environment with a
real C build+test setup.

## FLIMFitTool follow-up

The subsequently supplied `FLIMFitTool` resolved the custom fitting ABI. Its
length-checked v2 API accepts row-major `(y,x,time)` doubles and provides
component lifetime/amplitude maps, amplitude-weighted mean lifetime,
background, reduced chi-square, time shift, fit bounds, exit flags, statuses,
photon counts, and global-fit statistics. The Python app now binds those v2
calls. The supplied C source was audited and rebuilt after correcting its
constant-weighting sigma pointer and inclusive/exclusive stop-index bugs.

The MATLAB example fixtures contain inputs and truth values but no saved golden
fit structs. Synthetic native regression tests therefore establish ABI, layout,
status, and basic numerical behavior; quantitative equivalence should still be
checked on representative microscope acquisitions saved from LabVIEW.

## What was recovered from the LabVIEW archive

`ImageProcessingwt_IRF_looping1-6c.vi` is a LabVIEW 2020 run-to-completion
application. Its block diagram uses a file loop and a 17-frame processing
sequence rather than an event-driven interface. The front panel contains about
160 data objects across many wide, nested diagnostic tabs.

The high-confidence execution sequence is:

1. Read the ICS header and binary payload; reshape and transpose x/y.
2. Load and preprocess an IRF acquisition.
3. Create a background-subtracted 3-D histogram.
4. Load/display background and secondary inputs.
5. Interpolate selected pixels and optionally apply INL correction.
6. Run a legacy mono-exponential fit.
7. Optionally calculate lifetime and phasor maps.
8. Route results to displays.
9. Run per-pixel FLIMLib-style fitting.
10. Run shared-lifetime/global fitting.
11. Reshape maps, apply LUTs, create histograms and composites.
12. Write PNG, TIFF, spreadsheet/RAW data and optional gzip output.

The original panel includes single/multiple/bulk modes; data, background and
IRF paths; timing and photon thresholds; phasor limits; lifetime range and LUT;
per-pixel and global fit controls; intensity/lifetime/composite maps; lifetime
histograms; decay/fit graphs; and advanced calibration diagnostics.

FLIM Studio turns these into a task-oriented sidebar plus Overview, Decay &
fit, Phasor, Histogram and Metadata tabs. "Run current" corresponds to the
single/processing modes; the queue and "Process & export queue" correspond to
the multiple/bulk modes. Hidden byte arrays, error clusters and intermediate
graphs were intentionally not copied into the primary interface.

The Overview keeps intensity and the toggleable lifetime/composite map on its
upper row. The lower inspector places selected-point/ROI decays beside an
ROI-scoped phasor cloud, with a switch to the compact lifetime histogram.
Selected phasors use the stored IRF-corrected G/S coordinates and the same
P1–P8 colours as their map markers and decay curves.

## Implemented parity

- ICS header and U16 payload parsing by named axes, dimensions, representation,
  sign, compression and endianness
- defensive time-bin and laser repetition-rate extraction
- background acquisition/scalar subtraction
- fractional timing shift, spatial count binning and smoothing
- automatic extreme-pixel interpolation
- explicit source detector-row interpolation, including the parent VI's exact
  19-value operational preset and mirrored 512-row coordinate convention
- first-harmonic phasor calculation and phase/modulation lifetimes
- global or per-pixel IRF phasor correction, including the source VI's
  half-bin phase correction
- fast independent mono-exponential fit, chi-square map and representative
  global decay fit
- first-moment lifetime
- lifetime masking, LUT-like colour rendering and lifetime × intensity HSV
  composite
- phasor plot/universal semicircle, lifetime histogram and interactive pixel
  decay/fit inspection
- multi-file queue and reproducible numeric/image/tabular exports

## Known gaps

Exact fitting parity is not possible from the archive alone. These referenced
dependencies are missing:

- `flimfit_labview.dll::flim_fit_pixels_v2`
- `flimfit_labview.dll::flim_fit_global_pixels_v2`
- `FLIMlibTest_Clean.dll::{flim_fit_mono,flim_fit_bi,flim_fit_tri}`
- external MATLAB `fitFlimLabview2D` code from a hard-coded developer path

The source fit panels expose mono/bi/tri models, four noise models, fixed/free
background, global/per-pixel IRF, component amplitudes/lifetimes, shared global
taus and fit-status maps. The Python application exposes the usable independent
mono fit and marks its representative mean-decay fit as "global τ"; it does not
claim this is the missing DLL's shared-parameter global optimiser.

The INL correction VI exposes nearest, linear, spline, cubic-Hermite and
Lagrange modes, but the exact detector correction equation and calibration file
are not present. It has therefore not been guessed. The original selected-pixel
interpolation was recovered: the main VI uses 19 detector-row IDs, mirrors them
with `y=511-row`, and replaces complete histogram rows. Its saved subVI uses an
asymmetric six-neighbour mean, is fixed to 32 time bins, and does not actually
connect the displayed steps value to the live mean. FLIM Studio retains the
mapping and row semantics but uses a documented, simultaneous safe-linear
reconstruction across arbitrary time-bin counts rather than reproduce that
likely wiring bug. The eight orientation-transform codes still require a golden
microscope dataset for full validation.

## Validation needed with laboratory data

Before using the application for quantitative work, compare one complete set of
sample, background and IRF ICS files against LabVIEW at these checkpoints:

1. payload shape, dtype and total photon sum;
2. x/y orientation and summed decay;
3. background- and IRF-corrected decays;
4. per-pixel G/S phasors and lifetime maps;
5. representative pixel fits and lifetime standards;
6. histogram/statistics and exported orientation.

Once a representative data set, INL table, bad-pixel list and (ideally) the
native fitting DLL/source are available, these remaining adapters can be added
without changing the interface or core data model.
