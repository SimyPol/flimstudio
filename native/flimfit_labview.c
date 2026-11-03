#include "flimfit_labview.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Ecf.h"
#include "EcfGlobal.h"

#ifndef FLIMFIT_NAN
#define FLIMFIT_NAN ((double)NAN)
#endif

/* FLIMLib's own internal MAXBINS (EcfInternal.h) -- some of its global-fit
 * helper routines below use a fixed-size MAXBINS-row scratch array
 * internally, so nBins must never exceed it when calling them. Mirrored
 * here (rather than including EcfInternal.h) to avoid pulling in that
 * header's other internal-only declarations. */
#define FLIMFIT_MAX_GLOBAL_FAST_BINS 2048

/* App-defined fit type, NOT part of FLIMLib's own `fit_type` enum
 * (Ecf.h: FIT_GLOBAL_MULTIEXP=0, FIT_GLOBAL_STRETCHEDEXP=1) -- this value
 * is never passed into any bundled FLIMLib function, only used for our
 * own dispatch below. Represents up to 3 independently-distributed
 * (stretched-exponential) components summed together:
 *   I(t) = Z + sum_i A_i * exp(-(t/tau_i)^(1/h_i)),  i = 0..modelOrder-1
 * -- the generalization of FIT_GLOBAL_STRETCHEDEXP (which is always
 * exactly one component) to modelOrder components, each with its own
 * (tau, h) pair, vs. FIT_GLOBAL_MULTIEXP where every component is a
 * single discrete lifetime (h implicitly 1, not fit). See
 * multi_stretchedexp below for the model+derivatives, and
 * shared_param_index/bound_shared_param for how each component's (tau,h)
 * pair maps into the outer search's shared-parameter vector. */
#define FIT_GLOBAL_MULTI_STRETCHEDEXP 2

/* Largest shared-parameter vector any global fit type needs: up to 3
 * components, 2 shared values (tau, h) each, for FIT_GLOBAL_
 * MULTI_STRETCHEDEXP. FIT_GLOBAL_MULTIEXP needs at most 3 (one tau per
 * component), FIT_GLOBAL_STRETCHEDEXP always exactly 2. levenberg_
 * marquardt_min/solve_linear_system/evaluate_global_tau_objective size
 * their working arrays to this bound rather than a plain "3" now that a
 * fit type can need more than 3. */
#define MAX_SHARED_PARAMS 6

/* How many different starting points flim_fit_global_pixels_v2/
 * flim_train_global_model_v1 try for FIT_GLOBAL_MULTI_STRETCHEDEXP
 * specifically, when the caller hasn't already given an explicit shared
 * seed (see has_explicit_shared_seed, seed_multi_start_taus, and both
 * entry points' own comments) -- levenberg_marquardt_min is a strictly
 * local search, and direct chi-square comparison confirmed a real case
 * where it correctly converges to a genuinely lower-objective point
 * than the true generating parameters (RECONSTRUCTION_NOTES.md),
 * consistent with real model non-identifiability for N>=2
 * independently-distributed components -- multiple diverse starts,
 * keeping whichever finds the globally-lowest objective, at least makes
 * the search itself no longer the source of unreliability when the
 * caller has no prior to seed from, even though it can't fix genuine
 * non-identifiability in the underlying statistics. Confirmed by direct
 * experiment that it's NOT always an improvement to run regardless: a
 * caller-supplied seed already near the true parameters can still lose
 * to a genuinely-lower-objective, less physically plausible point found
 * by another variant -- silently overriding an informed guess with a
 * "statistically better" one is worse than respecting it, so an
 * explicit seed skips multi-start entirely. Not applied to
 * FIT_GLOBAL_MULTIEXP/FIT_GLOBAL_STRETCHEDEXP -- both already validated
 * with a single start, no reason to add cost or risk changing their
 * behaviour. */
#define MULTI_START_COUNT 5

/* Soft ceiling regularization on h, applied only to the OUTER search's
 * objective for FIT_GLOBAL_MULTI_STRETCHEDEXP (see
 * evaluate_global_tau_objective) -- never to the final reported
 * objective/chi-square (score_global_taus_with_irfs's return value is
 * used unmodified for that, both for provenance and for the diagnostic
 * comparisons in RECONSTRUCTION_NOTES.md, which must stay pure
 * goodness-of-fit numbers), and never to any other fit type.
 *
 * Why: both confirmed non-identifiability escape routes (tau=73.3ns/
 * h=7.3; h=14 alone) share a mechanism -- a component whose h (spread)
 * gets large enough that the stretched-exponential shape goes nearly
 * flat over the acquisition window, letting it silently "absorb"
 * residual anywhere without being meaningfully constrained by the
 * data's actual shape. That's not a real photophysical population (real
 * stretched-exponential fits in FLIM practice essentially never need
 * h above ~3-4; h=14 isn't a broader distribution, it's the search
 * finding a degenerate near-constant offset dressed up as a decay).
 * A soft quadratic penalty above H_REG_SOFT_CEILING pushes the search
 * back from that specific failure mode while leaving legitimate,
 * moderately broad distributions (h up to the ceiling) completely
 * untouched -- zero penalty until h actually crosses into implausible
 * territory. H_REG_LAMBDA sets how strongly that preference outweighs
 * a small chi-square improvement; both values are a considered
 * engineering default, not independently tuned/validated against a
 * sweep -- see RECONSTRUCTION_NOTES.md for the before/after test this
 * was validated against.
 *
 * H_REG_SOFT_CEILING alone was tried first and found INSUFFICIENT by
 * direct test: penalizing large h just made the search find a
 * different route to the same degenerate "nearly flat over the
 * acquisition window" component -- tau pushed out to ~3.8x the window
 * (114ns against a 30ns window) with h parked just above the ceiling,
 * achieving the same absorb-anything shape a different way. A second,
 * analogous soft ceiling on tau itself (relative to the acquisition
 * window, since "does this decay actually decay within the window" is
 * what determines whether its shape carries real information) closes
 * that second route -- see TAU_REG_WINDOW_MULTIPLE below and
 * RECONSTRUCTION_NOTES.md for the two-step account. */
#define H_REG_SOFT_CEILING 3.0
#define H_REG_LAMBDA 5.0

/* Soft ceiling on tau, expressed as a multiple of the acquisition
 * window (nBins*binWidth) rather than an absolute ns value, so it
 * scales correctly across datasets with different time ranges. 1.5x
 * the window: a component that hasn't meaningfully decayed within 1.5x
 * the measured range is barely constrained by the data's shape at all
 * -- true tau_a/tau_b in every validation scenario so far (1.5-5.0ns
 * against a 30ns window) sit far below this, so legitimate components
 * are untouched; only the specific degenerate-escape regime is
 * penalized. See H_REG_SOFT_CEILING's comment for why this exists
 * alongside the h ceiling rather than instead of it -- each closes a
 * different route to the same failure mode. */
#define TAU_REG_WINDOW_MULTIPLE 1.0
#define TAU_REG_LAMBDA 5.0

/* Soft floor on how close together (on a log scale) any two of
 * FIT_GLOBAL_MULTI_STRETCHEDEXP's component taus are allowed to get --
 * used ONLY as a passive post-hoc filter, in multi_stretched_result_is_
 * plausible, NOT as a term in the outer search's own objective (a
 * version that added it there was tried and reverted -- see evaluate_
 * global_tau_objective's own comment for why: adding any term to the
 * search's objective reshapes the whole gradient/Hessian landscape
 * along the entire convergence path, not just near the final point, and
 * measurably rerouted an already-good N=2 trajectory into a WORSE,
 * degenerate one). Found necessary by direct N=3 trace: whenever two of
 * the multi-start search's candidate taus ended up close together (e.g.
 * 10.39ns and 11.76ns -- a ratio of just 1.13x), the resulting per-pixel
 * linear solve (fit_linear_multi_stretched_pixel) produced a WILDLY
 * inflated objective (as high as 1.7 MILLION, against a normal few-
 * thousand-or-less range) -- two nearly-identical basis shapes make the
 * per-pixel normal-equations matrix nearly singular, so the solved
 * amplitudes (and hence the fitted curve and its chi-square) become
 * numerically unstable rather than merely "a bad fit." This is exactly
 * the "components collapse toward each other" failure mode this whole
 * investigation kept finding, now understood as also a genuine
 * numerical-conditioning problem, not solely a statistical one -- this
 * threshold at least stops the DATA-DRIVEN-preference override (see
 * DATA_DRIVEN_OBJECTIVE_TOLERANCE below) from ever picking a collapsed
 * result, without touching the search itself.
 * MIN_LOG_TAU_SEPARATION = log(1.5) -- a 1.5x ratio -- is deliberately
 * loose: true components in every scenario validated so far (N=2's
 * 1.5/5.0ns = 3.3x; N=3's 1.0/3.5/8.0ns = 3.5x and 2.3x) sit far above
 * this threshold, so genuinely close-but-distinct real lifetimes are
 * not what this penalizes -- only the much tighter (typically <1.2x)
 * clustering this session's traces showed causing near-singular
 * behaviour. */
#define MIN_LOG_TAU_SEPARATION 0.405465108  /* log(1.5) */

/* A second attempted fix for the near-singular-solve finding above,
 * ALSO tried and reverted: a small ridge (Tikhonov) term added directly
 * to fit_linear_multi_stretched_pixel's own per-pixel normal-equations
 * diagonal (scaled to the matrix's own mean diagonal magnitude, so
 * negligible for a well-conditioned solve). This one doesn't carry the
 * outer-objective regression's specific risk (it never touches the
 * tau/h search's own gradient/Hessian) -- but direct before/after
 * testing found it wasn't a net win either: N=2's unseeded validation
 * drifted measurably (up to ~7% on recovered tau, still in the right
 * ballpark but no longer the exact previously-validated numbers) purely
 * from many compounded per-pixel solves along a long nonlinear search
 * trajectory each nudged slightly, while N=3's unseeded case showed NO
 * corresponding practical improvement (still didn't recover the
 * slowest true component, comparable final objective to before, and
 * ran noticeably slower). Real precision cost for no measured benefit
 * -- reverted. Left undocumented as a #define since nothing references
 * it; see RECONSTRUCTION_NOTES.md for the full numbers if this idea is
 * ever revisited (e.g. with a much smaller epsilon, or applied only
 * when a conditioning check actually trips, rather than unconditionally
 * on every solve). */

/* How much worse the data-driven multi-start variant's own converged
 * objective is allowed to be than the best of the other (formulaic)
 * variants' and still be preferred outright -- see the override
 * decision's own extensive comment for the mechanism this whole
 * preference exists to correct. Found NECESSARY by direct N=3 test,
 * not just a theoretical safeguard: with 3 independently-distributed
 * components, the same per-pixel first-moment estimate that works well
 * for N=2 can itself fail to separate (a longer-lived component's tail
 * disproportionately pulls the moment upward even in pixels
 * "dominated" by a faster one, compressing the seed's own spread) --
 * confirmed by direct trace, all 3 seed taus landed within ~1ns of each
 * other (11.7/12.4/12.8) and the search stayed collapsed near there
 * (objective 1149.9), while an ordinary formulaic variant found a
 * genuinely much better point (objective 367.4, 3.1x lower). Without
 * this tolerance check, `multi_stretched_result_is_plausible` alone
 * accepted that collapsed answer anyway (nothing in it is individually
 * implausible -- each tau/h stays under the regularization ceiling),
 * silently returning a far worse fit than simply keeping the "lowest
 * objective wins" default would have. 1.5 (50% worse) keeps the N=2
 * case working (its data-driven objective was only ~7% worse than the
 * winning formulaic variant there) while rejecting a collapse this
 * large. */
#define DATA_DRIVEN_OBJECTIVE_TOLERANCE 1.5

/* Not exposed via EcfGlobal.h, but defined with external linkage in
 * EcfGlobal.c (see its own file-local forward-declaration block) --
 * these are the real per-transient building blocks
 * GCI_marquardt_global_exps_instr (and hence TRI2) itself uses: compute
 * the IRF-convolved exponential basis for a fixed set of shared taus
 * ONCE, then fit each transient's linear (Z/amplitude) parameters
 * against that fixed basis with no further convolution. Declared here so
 * the grouped-by-IRF fast global scoring below can reuse them directly
 * instead of reimplementing the same numerics. */
int GCI_marquardt_global_exps_calculate_exps_instr(
    float xincr, int ndata, float instr[], int ninstr,
    int ftype, float param[], int nparam,
    float exp_pure[], float *exp_conv[]);
int GCI_marquardt_global_exps_do_fit_single(
    float xincr, float y[], int ndata, int fit_start, int fit_end,
    noise_type noise, float sig[], int ftype,
    float param[], int paramfree[], int nparam, restrain_type restrain, float chisq_delta,
    float *exp_conv[], float *fitted, float *residuals,
    float **covar, float **alpha, float *chisq_trans);

static double total_counts(const double *y, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; ++i) {
        if (!isfinite(y[i])) return FLIMFIT_NAN;
        s += y[i];
    }
    return s;
}

static int all_zero_or_bad(const double *y, int n)
{
    int i;
    int nonzero = 0;
    for (i = 0; i < n; ++i) {
        if (!isfinite(y[i])) return 1;
        if (y[i] != 0.0) nonzero = 1;
    }
    return !nonzero;
}

static void copy_normalized_irf(const double *src, float *dst, int n)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < n; ++i) {
        if (isfinite(src[i]) && src[i] > 0.0) sum += src[i];
    }
    if (sum <= 0.0) sum = 1.0;
    for (i = 0; i < n; ++i) {
        double v = isfinite(src[i]) && src[i] > 0.0 ? src[i] : 0.0;
        dst[i] = (float)(v / sum);
    }
}

static int first_crossing(const float *y, int stop, float threshold)
{
    int i;
    for (i = 0; i <= stop; ++i) {
        if (y[i] >= threshold) return i;
    }
    return stop;
}

static int fwhm_bins(const float *y, int n)
{
    int i, first = -1, last = -1;
    float maxv = 0.0f;
    float half;
    for (i = 0; i < n; ++i) {
        if (y[i] > maxv) maxv = y[i];
    }
    half = 0.5f * maxv;
    for (i = 0; i < n; ++i) {
        if (y[i] >= half) {
            if (first < 0) first = i;
            last = i;
        }
    }
    if (first < 0) return 1;
    return last - first + 1;
}

static void infer_window(
    const float *decay,
    const float *instr,
    int nBins,
    double binWidth,
    int *fitStart,
    int *fitStop,
    double *timeShift)
{
    int i;
    int decayPeak = 0;
    int instrPeak = 0;
    int decayRise, instrRise, transientStart, width;
    float decayMax = decay[0];
    float instrMax = instr[0];
    float baseline = 0.0f;
    float threshold;
    int baselineBins = nBins / 10;
    if (baselineBins < 3) baselineBins = 3;
    if (baselineBins > nBins) baselineBins = nBins;

    for (i = 0; i < baselineBins; ++i) baseline += decay[i];
    baseline /= (float)baselineBins;

    for (i = 0; i < nBins; ++i) {
        if (decay[i] > decayMax) {
            decayMax = decay[i];
            decayPeak = i;
        }
        if (instr[i] > instrMax) {
            instrMax = instr[i];
            instrPeak = i;
        }
    }

    threshold = baseline + 0.10f * fmaxf(decayMax - baseline, 1.0f);
    decayRise = first_crossing(decay, decayPeak, threshold);
    instrRise = first_crossing(instr, instrPeak, 0.10f * instrMax);
    width = fwhm_bins(instr, nBins);

    transientStart = decayRise - instrRise;
    if (transientStart < 0) transientStart = 0;

    *fitStart = decayPeak > transientStart ? decayPeak : transientStart;
    *fitStart += (width + 1) / 2;
    if (*fitStart < 0) *fitStart = 0;
    if (*fitStart > nBins - 6) *fitStart = nBins > 6 ? nBins - 6 : 0;
    *fitStop = nBins - 1;
    *timeShift = transientStart * binWidth;
}

/* Linearly interpolates the sub-bin time at which `y` first crosses
 * `threshold`, given first_crossing already found the whole bin `at`
 * where it does (at<=0: no room to interpolate below it, returns that
 * bin's own time exactly). */
static double interpolated_crossing_time(const float *y, int at, float threshold, double binWidth)
{
    if (at <= 0) return (double)at * binWidth;
    {
        float y0 = y[at - 1];
        float y1 = y[at];
        double frac = (y1 > y0) ? (double)(threshold - y0) / (double)(y1 - y0) : 0.0;
        if (frac < 0.0) frac = 0.0;
        if (frac > 1.0) frac = 1.0;
        return ((double)(at - 1) + frac) * binWidth;
    }
}

/* Sub-bin-resolution version of infer_window's own transientStart
 * (decayRise - instrRise), for FIT_GLOBAL_MULTI_STRETCHEDEXP's fast-
 * path basis anchor specifically (see stretched_component_basis's own
 * comment for why that anchor -- not fitStart -- is the right one).
 * infer_window itself only finds the first whole bin at/above each
 * threshold (first_crossing) -- fine for choosing a safe integer-bin
 * summation window, but a whole-bin rounding error is a large fraction
 * of a narrow IRF's own width, which could plausibly explain a
 * meaningful share of the fast path's residual precision gap by
 * itself (untested until this function existed to check). This
 * linearly interpolates each threshold crossing instead of snapping to
 * the first bin that clears it, returning a continuous time (not a bin
 * count). Same baseline/threshold/peak logic as infer_window, kept as
 * a separate function rather than changed there -- several other call
 * sites use infer_window for the (unrelated) summation-window question
 * and don't need this. */
static double infer_transient_start_time(const float *decay, const float *instr, int nBins, double binWidth)
{
    int i;
    int decayPeak = 0;
    int instrPeak = 0;
    int decayRise, instrRise;
    float decayMax = decay[0];
    float instrMax = instr[0];
    float baseline = 0.0f;
    float threshold;
    int baselineBins = nBins / 10;
    double decayRiseTime, instrRiseTime, transientTime;
    if (baselineBins < 3) baselineBins = 3;
    if (baselineBins > nBins) baselineBins = nBins;

    for (i = 0; i < baselineBins; ++i) baseline += decay[i];
    baseline /= (float)baselineBins;

    for (i = 0; i < nBins; ++i) {
        if (decay[i] > decayMax) { decayMax = decay[i]; decayPeak = i; }
        if (instr[i] > instrMax) { instrMax = instr[i]; instrPeak = i; }
    }

    threshold = baseline + 0.10f * fmaxf(decayMax - baseline, 1.0f);
    decayRise = first_crossing(decay, decayPeak, threshold);
    instrRise = first_crossing(instr, instrPeak, 0.10f * instrMax);

    decayRiseTime = interpolated_crossing_time(decay, decayRise, threshold, binWidth);
    instrRiseTime = interpolated_crossing_time(instr, instrRise, 0.10f * instrMax, binWidth);

    transientTime = decayRiseTime - instrRiseTime;
    if (transientTime < 0.0) transientTime = 0.0;
    return transientTime;
}

static double estimate_background(const float *decay, int nBins)
{
    int i;
    int nTail = nBins / 5;
    double s = 0.0;
    if (nTail < 3) nTail = 3;
    if (nTail > nBins) nTail = nBins;
    for (i = nBins - nTail; i < nBins; ++i) s += decay[i];
    return s / nTail;
}

static double estimate_tau_moment(const float *decay, int fitStart, int fitStop, double bg, double binWidth)
{
    int i;
    double s = 0.0;
    double st = 0.0;
    for (i = fitStart; i <= fitStop; ++i) {
        double v = decay[i] - bg;
        double t = (i - fitStart) * binWidth;
        if (v < 0.0) v = 0.0;
        s += v;
        st += t * v;
    }
    if (s <= 0.0) return binWidth * (fitStop - fitStart + 1) / 4.0;
    return st / s;
}

static noise_type map_noise(int noiseModel)
{
    switch (noiseModel) {
        case 1: return NOISE_POISSON_FIT;
        case 2: return NOISE_POISSON_DATA;
        case 3: return NOISE_CONST;
        case 0:
        default: return NOISE_MLE;
    }
}

static double bounded_tau(double tau, double binWidth, int nBins)
{
    double lo = fmax(1e-6, binWidth * 0.05);
    double hi = fmax(lo * 10.0, binWidth * nBins * 10.0);
    /* Only a genuinely invalid (non-finite) value gets reset to this
     * arbitrary "quarter of the window" default -- a merely out-of-range
     * value, including a negative one, clamps to the nearest bound
     * instead (the very next two lines). This matters a lot more than it
     * looks: levenberg_marquardt_min calls this same function to clamp
     * every trial step it proposes, and a large-but-finite overshoot
     * (an entirely normal thing for a damped Newton step to produce
     * before damping has caught up) used to get teleported to this fixed
     * default point regardless of how far it overshot by -- creating a
     * damping-insensitive trap: every retry at higher lambda still
     * landed on the exact same reset value with the exact same (bad)
     * objective, so the search could get stuck bouncing between its
     * current point and that one fixed point forever, no matter how much
     * further damping it tried. Clamping to `lo` instead preserves the
     * monotonic relationship damping depends on to actually converge. */
    if (!isfinite(tau)) tau = binWidth * nBins / 4.0;
    if (tau < lo) tau = lo;
    if (tau > hi) tau = hi;
    return tau;
}

/* GCI_stretchedexp models I(t) = Z + A*exp(-(t/tau)^(1/h)) (Ecf.h/EcfUtil.c)
 * -- h=1 is a plain single exponential; h>1 "stretches" it, the standard
 * proxy for an underlying continuous distribution of decay rates (a
 * Kohlrausch-Williams-Watts/KWW stretched exponential). h is dimensionless
 * and appears as 1/h in that exponent, so it must stay safely away from 0
 * (blows up) and from implausibly large values (approaches a step
 * function, and the fit gains nothing from exploring further); this range
 * comfortably covers both stretched (h>1) and compressed (h<1) cases
 * without letting a pathological search step run away numerically. */
static double bounded_h(double h)
{
    double lo = 0.05;
    double hi = 25.0;
    /* See the matching comment in bounded_tau -- same reasoning, same
     * fix: only a non-finite value gets reset to the arbitrary default,
     * everything else (including negative) clamps to a real bound so an
     * iterative search's own damping can actually converge. */
    if (!isfinite(h)) h = 1.0;
    if (h < lo) h = lo;
    if (h > hi) h = hi;
    return h;
}

/* I(t) = Z + sum_i A_i * exp(-(t/tau_i)^(1/h_i)) -- FIT_GLOBAL_
 * MULTI_STRETCHEDEXP's model, generalizing GCI_stretchedexp (EcfUtil.c,
 * exactly one component) to modelOrder independently-distributed
 * components summed together, the same way GCI_multiexp_tau sums
 * modelOrder plain exponentials. Matches GCI_marquardt_fitting_engine's
 * generic fitfunc contract exactly (see GCI_stretchedexp/GCI_multiexp_tau
 * above for the same pattern): called once per time bin with the bin's
 * actual time value `x` (not a fixed increment -- IRF convolution and
 * background are both handled generically by the caller, outside any
 * fitfunc), writes the *uncolvolved* model value into `*y` and every
 * *free* parameter's derivative into dy_dparam[]; index 0 (background) is
 * never touched here, same convention as the two bundled fitfuncs.
 *
 * Parameter layout: param[0]=Z, then for component i (i=0..N-1, N =
 * (nparam-1)/3): amplitude at 1+3i, tau at 2+3i, h at 3+3i.
 *
 * Derivatives (matches EcfGlobal.c's own single-component stretched-exp
 * derivation exactly, see GCI_marquardt_global_exps_calculate_exps_instr's
 * FIT_GLOBAL_STRETCHEDEXP branch -- independently re-derived here since
 * this is a new, app-level model, then cross-checked against it):
 *   let x_i = t/tau_i, u_i = x_i^(1/h_i), E_i = exp(-u_i)
 *   dI/dA_i   = E_i
 *   dI/dtau_i = A_i * E_i * u_i / (h_i * tau_i)
 *   dI/dh_i   = A_i * E_i * u_i * ln(x_i) / h_i^2
 */
static void multi_stretchedexp(float x, float param[], float *y, float dy_dparam[], int nparam)
{
    int componentCount = (nparam - 1) / 3;
    int i;

    *y = 0.0f;
    for (i = 0; i < componentCount; ++i) {
        int ampIdx = 1 + 3 * i;
        int tauIdx = 2 + 3 * i;
        int hIdx = 3 + 3 * i;
        float amp = param[ampIdx];
        float tau = param[tauIdx];
        float h = param[hIdx];

        if (x > 0) {
            float xa = x / tau;
            float lxa = logf(xa);
            float xah = expf(lxa / h);   /* u_i = (x/tau_i)^(1/h_i) */
            float e = expf(-xah);        /* E_i */
            float ex;
            dy_dparam[ampIdx] = e;
            *y += amp * e;
            ex = amp * e * xah / h;      /* A_i*E_i*u_i/h_i, shared by both remaining derivatives */
            dy_dparam[tauIdx] = ex / tau;
            dy_dparam[hIdx] = ex * lxa / h;
        } else if (x > -1e-10f) {  /* almost zero */
            dy_dparam[ampIdx] = 1.0f;
            *y += amp;
            dy_dparam[tauIdx] = 0.0f;
            dy_dparam[hIdx] = 0.0f;
        }
        /* x<0 shouldn't happen (GCI_stretchedexp itself treats it as an
         * error); silently contribute nothing rather than spam stderr
         * from what can be a very hot per-bin/per-pixel/per-trial loop. */
    }
}

/* True if shared slot k (0-indexed into the outer search's own
 * sharedCount-length vector) represents an h (spread), not a tau
 * (centre), for this fit type. */
static int is_shared_slot_h(int fitType, int k)
{
    if (fitType == FIT_GLOBAL_STRETCHEDEXP) return k == 1;
    if (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP) return (k % 2) == 1;
    return 0;
}

/* Maps "shared parameter slot k" (what the outer search actually
 * explores -- sharedCount values) to its index in the underlying
 * nparam-sized FLIMLib-style parameter array, which differs by fit type:
 *  - FIT_GLOBAL_MULTIEXP:          Z, A0, tau0, A1, tau1, A2, tau2 -- taus
 *    at every other slot starting at 2, interleaved with each
 *    component's own amplitude. sharedCount == modelOrder.
 *  - FIT_GLOBAL_STRETCHEDEXP:      Z, A, tau, h -- always exactly 4
 *    params, one component, tau and h adjacent at indices 2 and 3.
 *    sharedCount == 2.
 *  - FIT_GLOBAL_MULTI_STRETCHEDEXP: Z, (A0,tau0,h0), (A1,tau1,h1), ... --
 *    see multi_stretchedexp. Shared slot k is tau at index 2+3*(k/2) when
 *    k is even, h at index 3+3*(k/2) when k is odd. sharedCount ==
 *    2*modelOrder.
 */
static int shared_param_index(int fitType, int k)
{
    if (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP) {
        int component = k / 2;
        return is_shared_slot_h(fitType, k) ? (3 + 3 * component) : (2 + 3 * component);
    }
    return fitType == FIT_GLOBAL_STRETCHEDEXP ? (2 + k) : (2 + 2 * k);
}

/* Bounds a shared-search value for slot k according to what it actually
 * represents for this fit type -- see is_shared_slot_h. */
static double bound_shared_param(int fitType, int k, double value, double binWidth, int nBins)
{
    if (is_shared_slot_h(fitType, k)) return bounded_h(value);
    return bounded_tau(value, binWidth, nBins);
}

/* Extracts component k's tau (not h) from a shared-parameter vector laid
 * out per shared_param_index -- needed anywhere code wants "this
 * component's centre lifetime" specifically, since for
 * FIT_GLOBAL_MULTI_STRETCHEDEXP the shared vector interleaves tau/h pairs
 * (component k's tau is at slot 2k, not slot k). For FIT_GLOBAL_MULTIEXP
 * and the single-component FIT_GLOBAL_STRETCHEDEXP, slot k already *is*
 * component k's tau directly. */
static double shared_tau_for_component(int fitType, const double *sharedParams, int k)
{
    return sharedParams[fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP ? 2 * k : k];
}

/* True if the caller supplied a real, usable seed for every shared slot
 * (not just some of them) -- used to decide whether multi-start runs at
 * all (see has_explicit_shared_seed's own call sites): confirmed by
 * direct experiment that when a caller-supplied seed already sits near
 * the true parameters, other multi-start variants can still find a
 * genuinely lower-objective point that's further from them (the same
 * real, confirmed non-identifiability MULTI_START_COUNT's own comment
 * describes) -- silently overriding an informed guess with a
 * statistically "better" but less physically plausible one is worse
 * than just respecting it, so an explicit seed skips multi-start
 * entirely rather than competing against it. */
static int has_explicit_shared_seed(int sharedCount, const double *initialTaus, int initialTausLength)
{
    int k;
    if (!initialTaus || initialTausLength < sharedCount) return 0;
    for (k = 0; k < sharedCount; ++k) {
        if (!(initialTaus[k] > 0.0)) return 0;
    }
    return 1;
}

/* Computes one multi-start variant's starting shared-parameter vector
 * for FIT_GLOBAL_MULTI_STRETCHEDEXP (see MULTI_START_COUNT's own
 * comment for why this exists). Only called at all when
 * has_explicit_shared_seed is false (see call sites) -- an explicit
 * seed bypasses multi-start entirely rather than competing against it.
 * `variant` 0..MULTI_START_COUNT-1 spans a deliberately wide, diverse
 * range of tau/h combinations -- tau seeds scale from a narrow (0.3x)
 * to a broad (2.3x) fraction of the acquisition window across variants,
 * h seeds shift by 0.5 per variant on top of the existing per-component
 * spread -- rather than clustering many similar starts near one guess.
 * `dataDrivenTaus` (NULL, or modelOrder-length, see
 * estimate_data_driven_multi_stretched_seed) supplies variant 0's tau
 * fallback specifically -- a real, data-informed guess instead of the
 * arbitrary fixed formula -- for whichever tau slots the caller didn't
 * already give an explicit seed for; h keeps the fixed formula either
 * way (not reliably estimable the same cheap way -- see that function's
 * own comment), and an explicit initialTaus entry for a slot always
 * wins over both. */
static void seed_multi_start_taus(
    int fitType, int sharedCount, double binWidth, int nBins,
    const double *initialTaus, int initialTausLength, int variant,
    const double *dataDrivenTaus, double *out)
{
    int k;
    for (k = 0; k < sharedCount; ++k) {
        int isH = is_shared_slot_h(fitType, k);
        int component = fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP ? (k / 2) : k;
        double fallback;
        if (isH) {
            fallback = 1.2 + 0.7 * component + 0.5 * variant;
        } else if (variant == 0 && dataDrivenTaus && dataDrivenTaus[component] > 0.0) {
            fallback = dataDrivenTaus[component];
        } else {
            double tauScale = 0.3 + 0.5 * variant;
            fallback = binWidth * nBins * (tauScale / (component + 1));
        }
        if (variant == 0 && initialTaus && initialTausLength > k && initialTaus[k] > 0.0) {
            fallback = initialTaus[k];
        }
        out[k] = bound_shared_param(fitType, k, fallback, binWidth, nBins);
    }
}

/* Builds a data-driven starting guess for FIT_GLOBAL_MULTI_STRETCHEDEXP's
 * shared tau parameters (modelOrder of them; h is left to seed_multi_
 * start_taus's own fixed formula -- see below) directly from the
 * training data's own per-pixel apparent-lifetime spread, instead of an
 * arbitrary fixed formula. Confirmed by direct experiment
 * (RECONSTRUCTION_NOTES.md's "N=2 actually works" entry) that this
 * model is identifiable given data with pixels/images that have
 * genuinely different relative amounts of the two-or-more components --
 * exactly the situation real global analysis relies on. Each pixel's
 * own crude moment-based apparent lifetime (estimate_tau_moment,
 * already used elsewhere in this file to seed ordinary single-component
 * fits) is a cheap, real proxy for "how much of the fast vs slow
 * component this pixel is dominated by" -- a pixel with a short
 * apparent lifetime is close to a pure fast-component pixel, a pixel
 * with a long one is close to a pure slow-component pixel, and pixels
 * in between reflect a mixture. Sorting selected pixels by that
 * estimate and reading off modelOrder evenly-spaced quantiles (not
 * literal min/max, which a single noisy pixel could distort) gives a
 * real, physically-motivated starting tau for each component -- close
 * to what an analyst manually picking "the purest-looking pixels" to
 * seed a global fit from would do.
 *
 * Returns 1 and fills `out` (modelOrder doubles) on success; 0 (out
 * left untouched) when there isn't enough usable per-pixel signal to
 * trust this (fewer than modelOrder+1 pixels produced a finite,
 * positive moment estimate, or an allocation failed) -- the caller
 * (seed_multi_start_taus) then falls back to its own fixed-formula
 * guess for variant 0 instead, exactly as if this function had never
 * been added. Uses the *full* [fitStartBin, fitStopBin] window (or all
 * of [0, nBins-1] when neither is given) rather than infer_window's
 * per-pixel IRF-relative windowing -- this is a coarse seed for the
 * search to refine, not a fit in its own right, so the small extra
 * bias from not excluding the IRF's own rise doesn't matter here the
 * way it would for score_global_taus_with_irfs's actual fit. */
static int estimate_data_driven_multi_stretched_seed(
    const double *decay, const int *sourcePixel, int selectedCount,
    int nBins, double binWidth, int modelOrder,
    int backgroundMode, double backgroundValue,
    int fitStartBin, int fitStopBin, double *out)
{
    double *moments;
    float *scratch;
    int nValid = 0;
    int i, k;
    int fitStart = fitStartBin > 0 ? fitStartBin - 1 : 0;
    int fitStop = fitStopBin > 0 ? fitStopBin - 1 : nBins - 1;

    if (fitStart < 0) fitStart = 0;
    if (fitStop > nBins - 1) fitStop = nBins - 1;
    if (fitStop <= fitStart || selectedCount <= 0) return 0;

    moments = (double *)malloc((size_t)selectedCount * sizeof(double));
    scratch = (float *)malloc((size_t)nBins * sizeof(float));
    if (!moments || !scratch) { free(moments); free(scratch); return 0; }

    for (i = 0; i < selectedCount; ++i) {
        const double *decayPixel = decay + (size_t)sourcePixel[i] * nBins;
        double bg, m;
        int b;
        for (b = 0; b < nBins; ++b) scratch[b] = (float)decayPixel[b];
        bg = backgroundMode == 1 ? backgroundValue : estimate_background(scratch, nBins);
        m = estimate_tau_moment(scratch, fitStart, fitStop, bg, binWidth);
        if (isfinite(m) && m > 0.0) moments[nValid++] = m;
    }
    free(scratch);

    if (nValid < modelOrder + 1) { free(moments); return 0; }

    /* Simple insertion sort -- runs once per training call (not once
     * per objective evaluation the way everything in the outer search
     * itself does), so O(n^2) is fine for realistic pixel counts; not
     * worth pulling in qsort's comparator overhead for this. */
    for (i = 1; i < nValid; ++i) {
        double v = moments[i];
        int j = i - 1;
        while (j >= 0 && moments[j] > v) { moments[j + 1] = moments[j]; --j; }
        moments[j + 1] = v;
    }

    for (k = 0; k < modelOrder; ++k) {
        double q = (k + 0.5) / (double)modelOrder;
        int idx = (int)(q * (nValid - 1) + 0.5);
        if (idx < 0) idx = 0;
        if (idx >= nValid) idx = nValid - 1;
        out[k] = moments[idx];
    }

    free(moments);
    return 1;
}

/* True if a converged FIT_GLOBAL_MULTI_STRETCHEDEXP shared-parameter
 * vector looks physically plausible -- every component's h within
 * H_REG_SOFT_CEILING and tau within TAU_REG_WINDOW_MULTIPLE*window,
 * the SAME thresholds the outer search's own regularization already
 * uses (see H_REG_SOFT_CEILING's comment), reused here rather than a
 * second, independent set of numbers to keep in sync. Always true for
 * every other fit type (nothing here is specific to them).
 *
 * Used for exactly one decision -- see its call sites' own comment --
 * deciding whether the data-driven multi-start variant's own converged
 * answer (estimate_data_driven_multi_stretched_seed) is trustworthy
 * enough to prefer directly, rather than whichever variant happens to
 * report the lowest raw objective. */
static int multi_stretched_result_is_plausible(int fitType, int modelOrder, const double *taus, double window)
{
    int k, j;
    if (fitType != FIT_GLOBAL_MULTI_STRETCHEDEXP) return 1;
    for (k = 0; k < modelOrder; ++k) {
        double tau = taus[2 * k];
        double h = taus[2 * k + 1];
        if (!isfinite(tau) || !isfinite(h) || tau <= 0.0 || h <= 0.0) return 0;
        if (h > H_REG_SOFT_CEILING) return 0;
        if (window > 0.0 && tau > TAU_REG_WINDOW_MULTIPLE * window) return 0;
        for (j = k + 1; j < modelOrder; ++j) {
            /* Also reject any two components collapsed too close
             * together (see MIN_LOG_TAU_SEPARATION's own comment) --
             * exactly the near-singular-linear-solve regime that made
             * the data-driven variant's own objective spuriously huge
             * in the trace that motivated this whole penalty; without
             * this check here too, a collapsed data-driven result could
             * still pass the tolerance comparison below by pure luck on
             * a dataset where the formulaic variants happen to fare
             * worse. */
            double logRatio = fabs(log(fmax(tau, 1e-6)) - log(fmax(taus[2 * j], 1e-6)));
            if (logRatio < MIN_LOG_TAU_SEPARATION) return 0;
        }
    }
    return 1;
}

/* How many objective evaluations one Marquardt-damped Newton iteration
 * costs a sharedCount=n outer search -- a full finite-difference
 * gradient+Hessian pass needs 2n (gradient) + 4*C(n,2) = 2n+2n(n-1)
 * (Hessian) evaluations... this file's own long-standing comment next
 * to the *15 multiplier below simplifies that to "4n+2n(n-1)", used
 * here verbatim for consistency with it. */
static int evals_per_newton_iteration(int n)
{
    return 4 * n + 2 * n * (n - 1);
}

/* Scales the flat "*15" per-globalIterations-unit evaluation multiplier
 * up for outer searches with more shared parameters than
 * FIT_GLOBAL_MULTI_STRETCHEDEXP's N=2 case (sharedCount=4) -- found
 * NECESSARY by direct N=3 test, not a theoretical nicety: with
 * sharedCount=6 (3 components), a truth-seeded search given the
 * previously-flat "globalIterations*15" budget (600 evaluations at the
 * GUI's default global_iterations=40) had NOT converged -- its own
 * gradient-convergence check only fired at ~1180 evaluations, and the
 * intermediate, under-converged snapshots looked like a genuine
 * 2-of-3-components collapse that fully converging resolved into a
 * meaningfully better-separated (though still imperfect) answer. Since
 * one Newton iteration's own cost scales as ~n^2 (evals_per_newton_
 * iteration above), and empirically more OUTER iterations were also
 * needed for the larger, more elongated n=6 landscape (not just each
 * iteration costing more), this scales by that n^2-ish ratio directly
 * -- `max(1.0, ...)` so sharedCount<=4 (every already-validated fit
 * type/order: FIT_GLOBAL_MULTIEXP with modelOrder 1-3 gives sharedCount
 * 1-3; FIT_GLOBAL_STRETCHEDEXP and FIT_GLOBAL_MULTI_STRETCHEDEXP's own
 * N=1 both give sharedCount 2; N=2 gives exactly 4) keeps the exact,
 * unchanged multiplier of 15 it already had -- only sharedCount=6 (N=3)
 * gets more. */
static double evaluation_budget_multiplier(int sharedCount)
{
    double ratio = (double)evals_per_newton_iteration(sharedCount) /
        (double)evals_per_newton_iteration(4);
    if (ratio < 1.0) ratio = 1.0;
    return 15.0 * ratio;
}

/* Fits a single pixel's Z/amplitude(s) with the shared parameter(s) held
 * fixed -- tau[0..modelOrder-1] for FIT_GLOBAL_MULTIEXP; [tau, h] for
 * FIT_GLOBAL_STRETCHEDEXP; (tau_i, h_i) per component for FIT_GLOBAL_
 * MULTI_STRETCHEDEXP (see shared_param_index/shared_tau_for_component).
 * This is the per-pixel building block both the grouped-fast-path's
 * fallback and flim_apply_global_model_v1 use; modelOrder is ignored for
 * single-component stretched-exp (always exactly one component, nparam=4)
 * but still governs amplitudeOut's/tauMeanOut's shape for multiexp and
 * multi-stretched-exp as before. */
static int fit_fixed_tau_pixel(
    const double *decayPixel,
    const double *irfPixel,
    int nBins,
    double binWidth,
    int fitType,
    int modelOrder,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    int fitStartBin,
    int fitStopBin,
    const double *sharedParams,
    double *amplitudeOut,
    double *backgroundOut,
    double *reducedChiOut,
    double *tauMeanOut)
{
    int b, k;
    int stretched = fitType == FIT_GLOBAL_STRETCHEDEXP;
    int multiStretched = fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP;
    int componentCount = stretched ? 1 : modelOrder;
    int nparam = stretched ? 4 : (multiStretched ? (1 + 3 * modelOrder) : (2 * modelOrder + 1));
    int fitStart = fitStartBin > 0 ? fitStartBin - 1 : 0;
    int fitStop = fitStopBin > 0 ? fitStopBin - 1 : nBins - 1;
    int freeCount = 0;
    int ret;
    double bg;
    double ampSeed = 0.0;
    double ampSum = 0.0;
    double tauWeighted = 0.0;
    double shift = 0.0;
    float chisq = 0.0f;
    float *y = NULL;
    float *instr = NULL;
    float *fitted = NULL;
    float *residuals = NULL;
    float *params = NULL;
    int *paramfree = NULL;
    float **covar = NULL;
    float **alpha = NULL;
    float **erraxes = NULL;
    noise_type noise = map_noise(noiseModel);
    float sigma[1] = {1.0f};
    void (*fitfunc)(float, float[], float *, float[], int) =
        stretched ? GCI_stretchedexp : (multiStretched ? multi_stretchedexp : GCI_multiexp_tau);

    if (all_zero_or_bad(decayPixel, nBins)) return -1;
    if (all_zero_or_bad(irfPixel, nBins)) return -3;

    y = (float *)calloc((size_t)nBins, sizeof(float));
    instr = (float *)calloc((size_t)nBins, sizeof(float));
    fitted = (float *)calloc((size_t)nBins, sizeof(float));
    residuals = (float *)calloc((size_t)nBins, sizeof(float));
    params = (float *)calloc((size_t)nparam, sizeof(float));
    paramfree = (int *)calloc((size_t)nparam, sizeof(int));
    covar = GCI_ecf_matrix(nparam, nparam);
    alpha = GCI_ecf_matrix(nparam, nparam);
    erraxes = GCI_ecf_matrix(nparam, nparam);
    if (!y || !instr || !fitted || !residuals || !params || !paramfree ||
        !covar || !alpha || !erraxes) {
        ret = -5;
        goto cleanup;
    }

    for (b = 0; b < nBins; ++b) {
        y[b] = (float)decayPixel[b];
    }
    copy_normalized_irf(irfPixel, instr, nBins);

    if (fitStartBin <= 0 || fitStopBin <= 0) {
        infer_window(y, instr, nBins, binWidth, &fitStart, &fitStop, &shift);
    }
    if (fitStart < 0) fitStart = 0;
    if (fitStop > nBins - 1) fitStop = nBins - 1;
    if (fitStop <= fitStart + componentCount + 2) {
        ret = -6;
        goto cleanup;
    }

    bg = backgroundMode == 1 ? backgroundValue : estimate_background(y, nBins);
    params[0] = (float)bg;
    paramfree[0] = backgroundMode == 1 ? 0 : 1;
    for (b = fitStart; b <= fitStop; ++b) {
        double v = y[b] - bg;
        if (v > ampSeed) ampSeed = v;
    }
    if (ampSeed <= 0.0) ampSeed = total_counts(decayPixel, nBins) / nBins;

    if (stretched) {
        params[1] = (float)ampSeed;
        params[2] = (float)bound_shared_param(fitType, 0, sharedParams[0], binWidth, nBins);
        params[3] = (float)bound_shared_param(fitType, 1, sharedParams[1], binWidth, nBins);
        paramfree[1] = 1;
        paramfree[2] = 0;
        paramfree[3] = 0;
    } else if (multiStretched) {
        for (k = 0; k < modelOrder; ++k) {
            int ampIdx = 1 + 3 * k, tauIdx = 2 + 3 * k, hIdx = 3 + 3 * k;
            int tauSlot = 2 * k, hSlot = 2 * k + 1;
            params[ampIdx] = (float)(ampSeed / modelOrder);
            params[tauIdx] = (float)bound_shared_param(fitType, tauSlot, sharedParams[tauSlot], binWidth, nBins);
            params[hIdx] = (float)bound_shared_param(fitType, hSlot, sharedParams[hSlot], binWidth, nBins);
            paramfree[ampIdx] = 1;
            paramfree[tauIdx] = 0;
            paramfree[hIdx] = 0;
        }
    } else {
        for (k = 0; k < modelOrder; ++k) {
            params[1 + 2 * k] = (float)(ampSeed / modelOrder);
            params[2 + 2 * k] = (float)bound_shared_param(fitType, k, sharedParams[k], binWidth, nBins);
            paramfree[1 + 2 * k] = 1;
            paramfree[2 + 2 * k] = 0;
        }
    }
    for (k = 0; k < nparam; ++k) {
        if (paramfree[k]) ++freeCount;
    }

    ret = GCI_marquardt_fitting_engine(
        (float)binWidth, y, nBins, fitStart, fitStop + 1,
        instr, nBins, noise, noise == NOISE_CONST ? sigma : NULL, params, paramfree, nparam,
        ECF_RESTRAIN_DEFAULT, fitfunc, fitted, residuals,
        &chisq, covar, alpha, erraxes,
        1.1f * (float)fmax(1, fitStop - fitStart + 1 - freeCount),
        0.001f, 60);

    if (ret < 0) {
        ret = -4;
        goto cleanup;
    }

    if (backgroundOut) *backgroundOut = params[0];
    if (reducedChiOut) {
        *reducedChiOut = chisq / fmax(1.0, (double)(fitStop - fitStart + 1 - freeCount));
    }
    if (stretched) {
        /* Single component -- tauMeanOut is just that one tau, not an
         * amplitude-weighted mean across several. */
        if (amplitudeOut) amplitudeOut[0] = params[1];
        if (tauMeanOut) *tauMeanOut = params[2];
    } else if (multiStretched) {
        for (k = 0; k < modelOrder; ++k) {
            double amp = params[1 + 3 * k];
            double tau = params[2 + 3 * k];
            if (amplitudeOut) amplitudeOut[k] = amp;
            ampSum += amp;
            tauWeighted += amp * tau;
        }
        if (tauMeanOut) *tauMeanOut = ampSum > 0.0 ? tauWeighted / ampSum : FLIMFIT_NAN;
    } else {
        for (k = 0; k < modelOrder; ++k) {
            double amp = params[1 + 2 * k];
            double tau = params[2 + 2 * k];
            if (amplitudeOut) amplitudeOut[k] = amp;
            ampSum += amp;
            tauWeighted += amp * tau;
        }
        if (tauMeanOut) *tauMeanOut = ampSum > 0.0 ? tauWeighted / ampSum : FLIMFIT_NAN;
    }
    ret = 1;

cleanup:
    if (covar) GCI_ecf_free_matrix(covar);
    if (alpha) GCI_ecf_free_matrix(alpha);
    if (erraxes) GCI_ecf_free_matrix(erraxes);
    free(y);
    free(instr);
    free(fitted);
    free(residuals);
    free(params);
    free(paramfree);
    return ret;
}

/* ---------------------------------------------------------------------
 * IRF grouping for the fast path below: buckets selected pixels by IRF
 * *content* (a cheap running hash, verified with memcmp on collision) so
 * every pixel sharing an IRF can reuse one convolved exponential basis
 * computed only once, instead of each pixel independently re-deriving
 * it. irfMode=0 (one shared IRF for the whole dataset) short-circuits to
 * a single group covering every selected pixel; irfMode=1 with a
 * handful of distinct per-detector IRFs collapses to a handful of
 * groups; genuinely all-distinct per-pixel IRFs degrade gracefully to
 * one group per pixel -- the fast path then costs the same as fitting
 * every pixel independently, never worse.
 * ------------------------------------------------------------------- */

typedef struct {
    int numGroups;
    int *pixelGroup;    /* length selectedCount: selIndex -> group id */
    int *groupIrfPixel; /* length numGroups: representative absolute pixel id per group */
    int *groupStart;    /* length numGroups+1: CSR offsets into memberOrder */
    int *memberOrder;   /* length selectedCount: selIndex values, grouped contiguously by group id */
} irf_grouping;

typedef struct {
    unsigned long long hash;
    int selIndex;
} irf_hash_entry;

static int compare_irf_hash_entry(const void *a, const void *b)
{
    unsigned long long ha = ((const irf_hash_entry *)a)->hash;
    unsigned long long hb = ((const irf_hash_entry *)b)->hash;
    if (ha < hb) return -1;
    if (ha > hb) return 1;
    return 0;
}

static unsigned long long hash_bytes(const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    unsigned long long h = 1469598103934665603ULL; /* FNV-1a offset basis */
    size_t i;
    for (i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL; /* FNV-1a prime */
    }
    return h;
}

static void free_irf_groups(irf_grouping *g)
{
    free(g->pixelGroup); g->pixelGroup = NULL;
    free(g->groupIrfPixel); g->groupIrfPixel = NULL;
    free(g->groupStart); g->groupStart = NULL;
    free(g->memberOrder); g->memberOrder = NULL;
    g->numGroups = 0;
}

/* Fills *out (caller must free_irf_groups it). Returns 0 on success. */
static int build_irf_groups(
    const double *irf, const int *sourcePixel, int selectedCount,
    int nBins, int irfMode, irf_grouping *out)
{
    int i;

    out->numGroups = 0;
    out->pixelGroup = NULL;
    out->groupIrfPixel = NULL;
    out->groupStart = NULL;
    out->memberOrder = NULL;

    if (selectedCount <= 0) return 0;

    out->pixelGroup = (int *)calloc((size_t)selectedCount, sizeof(int));
    if (!out->pixelGroup) return -1;

    if (irfMode == 0 || selectedCount == 1) {
        out->groupIrfPixel = (int *)malloc(sizeof(int));
        if (!out->groupIrfPixel) { free_irf_groups(out); return -1; }
        out->groupIrfPixel[0] = sourcePixel[0];
        out->numGroups = 1;
        /* out->pixelGroup already all zero from calloc */
    } else {
        irf_hash_entry *entries = (irf_hash_entry *)malloc((size_t)selectedCount * sizeof(irf_hash_entry));
        int *groupIrfPixel = (int *)malloc((size_t)selectedCount * sizeof(int));
        unsigned char *consumed = (unsigned char *)calloc((size_t)selectedCount, 1);
        int numGroups = 0;
        size_t rowBytes = (size_t)nBins * sizeof(double);

        if (!entries || !groupIrfPixel || !consumed) {
            free(entries); free(groupIrfPixel); free(consumed);
            free_irf_groups(out);
            return -1;
        }

        for (i = 0; i < selectedCount; ++i) {
            const double *row = irf + (size_t)sourcePixel[i] * nBins;
            entries[i].hash = hash_bytes(row, rowBytes);
            entries[i].selIndex = i;
        }
        qsort(entries, (size_t)selectedCount, sizeof(irf_hash_entry), compare_irf_hash_entry);

        i = 0;
        while (i < selectedCount) {
            int runEnd = i + 1;
            int m;
            while (runEnd < selectedCount && entries[runEnd].hash == entries[i].hash) ++runEnd;

            for (m = i; m < runEnd; ++m) {
                int selM = entries[m].selIndex;
                const double *repRow;
                int gid, n;
                if (consumed[selM]) continue;

                repRow = irf + (size_t)sourcePixel[selM] * nBins;
                gid = numGroups++;
                groupIrfPixel[gid] = sourcePixel[selM];
                out->pixelGroup[selM] = gid;
                consumed[selM] = 1;

                for (n = m + 1; n < runEnd; ++n) {
                    int selN = entries[n].selIndex;
                    const double *row;
                    if (consumed[selN]) continue;
                    row = irf + (size_t)sourcePixel[selN] * nBins;
                    if (memcmp(row, repRow, rowBytes) == 0) {
                        out->pixelGroup[selN] = gid;
                        consumed[selN] = 1;
                    }
                }
            }
            i = runEnd;
        }

        free(entries);
        free(consumed);
        out->numGroups = numGroups;
        out->groupIrfPixel = groupIrfPixel;
    }

    /* CSR-style member lists so the fast-path fitter below can walk each
     * group's pixels in O(1) amortised per pixel instead of rescanning
     * every selected pixel once per group (which would cost
     * O(selectedCount * numGroups) -- fine for irfMode=0's single group,
     * but quadratic in the all-distinct-per-pixel-IRF worst case). */
    out->groupStart = (int *)calloc((size_t)out->numGroups + 1, sizeof(int));
    out->memberOrder = (int *)malloc((size_t)selectedCount * sizeof(int));
    if (!out->groupStart || !out->memberOrder) { free_irf_groups(out); return -1; }
    for (i = 0; i < selectedCount; ++i) out->groupStart[out->pixelGroup[i] + 1]++;
    for (i = 0; i < out->numGroups; ++i) out->groupStart[i + 1] += out->groupStart[i];
    {
        int *cursor = (int *)malloc((size_t)out->numGroups * sizeof(int));
        if (!cursor) { free_irf_groups(out); return -1; }
        memcpy(cursor, out->groupStart, (size_t)out->numGroups * sizeof(int));
        for (i = 0; i < selectedCount; ++i) {
            int gid = out->pixelGroup[i];
            out->memberOrder[cursor[gid]++] = i;
        }
        free(cursor);
    }

    return 0;
}

/* Forward declaration -- defined below, alongside levenberg_marquardt_min
 * (the outer search's own use of it); fit_linear_multi_stretched_pixel
 * needs it earlier in the file. */
static int solve_linear_system(int n, double A[MAX_SHARED_PARAMS][MAX_SHARED_PARAMS], const double *b, double *x);

/* Fits Z (if free) + modelOrder amplitudes against modelOrder
 * precomputed, IRF-convolved basis shapes -- the per-pixel building
 * block of FIT_GLOBAL_MULTI_STRETCHEDEXP's fast path below. With tau/h
 * held fixed at every trial point, y(t) = Z + sum_i A_i*basis_i(t) is
 * genuinely LINEAR in the remaining free parameters, so unlike
 * fit_fixed_tau_pixel this needs no nonlinear Marquardt search at all --
 * no bundled FLIMLib fit_type understands "N independently stretched
 * components summed together" (see the fast path's own comment), so
 * there's nothing to call into for the fit itself; this is a small,
 * hand-written weighted linear solve instead, reusing solve_linear_system
 * (already sized up to MAX_SHARED_PARAMS, comfortably more than
 * modelOrder+1<=4). Poisson-flavoured noise models (MLE, fit-weighted)
 * iterate a handful of passes, reweighting from the current fit each
 * time -- standard iteratively-reweighted-least-squares / Fisher-scoring
 * practice for approximating a Poisson fit via repeated weighted-linear
 * solves; constant and data-weighted models need only one pass since
 * their weights don't depend on the fit. Returns 1 on success, 0 if the
 * weighted normal equations are singular or the result isn't finite.
 * `weightScratch` is caller-owned scratch, length >= nBins (reused across
 * every pixel in a group to avoid per-pixel malloc/free). */
/* Computes one component's IRF-convolved basis shape for the FIT_GLOBAL_
 * MULTI_STRETCHEDEXP fast path, anchoring the *pure* (pre-convolution)
 * decay's own t=0 at bin `anchor` instead of bin 0. This matters:
 * bundled EcfSingle.c has its own FIXME acknowledging the bin-0 anchor
 * its array-optimized functions use (and that GCI_marquardt_global_exps_
 * calculate_exps_instr -- this fast path's original basis source --
 * computes the identical shape via) is "adverse" -- t=0 should be
 * aligned with where the decay's own transient actually starts. See the
 * fast path's own call site for exactly what `anchor` needs to be
 * (infer_window's transientStart, NOT fitStart -- an earlier version of
 * this code anchored at fitStart instead and made things measurably
 * worse; RECONSTRUCTION_NOTES.md has the full account of both the
 * original bin-0 bias and that intermediate mistake, confirmed visually
 * before being fixed).
 *
 * Does its own discrete convolution (the same causal, truncated-to-
 * nBins-samples formula FLIMLib's own array functions use --
 * conv[j] = sum_{k=0}^{j} pure[j-k]*instr[k], since instr here is always
 * exactly nBins long) rather than reusing calculate_exps_instr, so it
 * can choose its own anchor freely; O(nBins) per bin summed over nBins
 * bins, same asymptotic cost as the function it replaces, and -- like
 * the basis it replaces -- computed once per (IRF group, component),
 * never once per pixel, so the fast path's actual speed win is
 * unaffected. `pureScratch` is caller-owned scratch, length >= nBins. */
static void stretched_component_basis(
    double binWidth,
    int nBins,
    const float *instr,
    double tau,
    double h,
    double anchorTime,
    float *pureScratch,
    float *basisOut)
{
    int i, j;

    /* anchorTime is a continuous time (from infer_transient_start_time),
     * not a whole-bin index -- deliberately not snapped to the nearest
     * bin, since a narrow IRF's own width can be a small fraction of one
     * bin, making whole-bin rounding here a large relative timing error. */
    for (i = 0; i < nBins; ++i) {
        double x = (double)i * binWidth - anchorTime;
        if (x <= 0.0) {
            pureScratch[i] = 0.0f;
        } else {
            double xa = x / tau;
            double u = pow(xa, 1.0 / h);
            pureScratch[i] = (float)exp(-u);
        }
    }
    for (j = 0; j < nBins; ++j) {
        double sum = 0.0;
        int k;
        for (k = 0; k <= j; ++k) sum += (double)pureScratch[j - k] * (double)instr[k];
        basisOut[j] = (float)sum;
    }
}

static int fit_linear_multi_stretched_pixel(
    const float *y,
    int nBins,
    int fitStart,
    int fitStop,
    int backgroundMode,
    double backgroundValue,
    int modelOrder,
    const float *basisBlock,
    noise_type noise,
    double *amplitudeOut,
    double *backgroundOut,
    double *reducedChiOut,
    double *weightScratch)
{
    int freeBg = backgroundMode != 1;
    int nfree = freeBg ? (modelOrder + 1) : modelOrder;
    int maxIter = (noise == NOISE_CONST || noise == NOISE_POISSON_DATA) ? 1 : 8;
    double bg = freeBg ? 0.0 : backgroundValue;
    double x[MAX_SHARED_PARAMS];
    int b, k, iter;

    for (k = 0; k < modelOrder; ++k) amplitudeOut[k] = 0.0;

    /* Variance floored at 15 counts, not 1 -- matches FLIMLib's own
     * Marquardt-family weighting exactly (confirmed in bundled
     * EcfGlobal.c's GCI_marquardt_global_exps_single_step, the function
     * the legacy fast path calls for this same NOISE_POISSON_FIT/DATA
     * pair: "weight = (yfit > 15 ? 1.0f/yfit : 1.0f/15)"). Using a
     * floor of 1 here (an earlier version of this code did) let the
     * long, low-count tail of a decay -- exactly where a stretched
     * exponential's h is most different from an ordinary one's -- carry
     * up to 15x more relative weight than FLIMLib's own convention
     * gives it, measurably biasing the fit. NOISE_MLE (noiseModel=0)
     * is a further, separate case: FLIMLib uses a genuine Poisson
     * deviance there, not a weighted-least-squares chi-square at all
     * (2*(yfit-y) - 2*y*log(yfit/y)); this IRLS weighting is a
     * reasonable Fisher-scoring-style proxy toward the same fit, and
     * floored at 1 there to match, but the *objective value* reported
     * below won't exactly equal FLIMLib's own MLE deviance the way it
     * does for the two Poisson-weighted cases. */
    for (b = fitStart; b <= fitStop; ++b) {
        double floor = noise == NOISE_MLE ? 1.0 : 15.0;
        weightScratch[b] = (noise == NOISE_CONST) ? 1.0 : 1.0 / fmax(floor, (double)y[b]);
    }

    for (iter = 0; iter < maxIter; ++iter) {
        double A[MAX_SHARED_PARAMS][MAX_SHARED_PARAMS];
        double rhs[MAX_SHARED_PARAMS];
        int j;
        for (k = 0; k < nfree; ++k) {
            rhs[k] = 0.0;
            for (j = 0; j < nfree; ++j) A[k][j] = 0.0;
        }
        for (b = fitStart; b <= fitStop; ++b) {
            double w = weightScratch[b];
            double basisRow[MAX_SHARED_PARAMS];
            int r = 0;
            if (freeBg) basisRow[r++] = 1.0;
            for (k = 0; k < modelOrder; ++k) basisRow[r++] = (double)basisBlock[(size_t)k * nBins + b];
            for (k = 0; k < nfree; ++k) {
                rhs[k] += w * basisRow[k] * (double)y[b];
                for (j = 0; j < nfree; ++j) A[k][j] += w * basisRow[k] * basisRow[j];
            }
        }
        if (!solve_linear_system(nfree, A, rhs, x)) return 0;

        {
            int r = 0;
            if (freeBg) bg = x[r++];
            for (k = 0; k < modelOrder; ++k) amplitudeOut[k] = x[r++];
        }

        if (iter + 1 < maxIter) {
            double floor = noise == NOISE_MLE ? 1.0 : 15.0;
            for (b = fitStart; b <= fitStop; ++b) {
                double fitted = bg;
                for (k = 0; k < modelOrder; ++k) {
                    fitted += amplitudeOut[k] * (double)basisBlock[(size_t)k * nBins + b];
                }
                weightScratch[b] = 1.0 / fmax(floor, fitted);
            }
        }
    }

    {
        double chi = 0.0;
        double floor = noise == NOISE_MLE ? 1.0 : 15.0;
        for (b = fitStart; b <= fitStop; ++b) {
            double fitted = bg;
            double resid, w;
            for (k = 0; k < modelOrder; ++k) {
                fitted += amplitudeOut[k] * (double)basisBlock[(size_t)k * nBins + b];
            }
            resid = (double)y[b] - fitted;
            w = (noise == NOISE_CONST) ? 1.0 : 1.0 / fmax(floor, fitted);
            chi += w * resid * resid;
        }
        *reducedChiOut = chi / fmax(1.0, (double)(fitStop - fitStart + 1 - nfree));
    }
    if (backgroundOut) *backgroundOut = bg;
    if (!isfinite(bg) || !isfinite(*reducedChiOut)) return 0;
    for (k = 0; k < modelOrder; ++k) {
        if (!isfinite(amplitudeOut[k])) return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------------
 * Scores (and optionally records outputs for) a trial shared-tau vector
 * across every selected pixel. Three implementations:
 *
 * Fast path, FIT_GLOBAL_MULTIEXP/FIT_GLOBAL_STRETCHEDEXP (groups != NULL
 * and nBins fits FLIMLib's internal MAXBINS scratch-array limit): reuses
 * FLIMLib's own real global-fit building blocks -- the exact routines
 * GCI_marquardt_global_exps_instr (and hence TRI2) uses internally --
 * instead of independently calling the generic single-transient
 * reconvolution engine (fit_fixed_tau_pixel/GCI_marquardt_fitting_engine)
 * once per pixel. That generic engine redoes the IRF convolution from
 * scratch on every Marquardt inner iteration of every single pixel, even
 * though the convolution only actually depends on the (fixed, shared)
 * taus and the IRF -- identical for every pixel that shares one. Per IRF
 * group, this instead calls GCI_marquardt_global_exps_calculate_exps_
 * instr ONCE to get a small basis of precomputed convolved curves, then
 * GCI_marquardt_global_exps_do_fit_single fits each pixel's Z/amplitudes
 * against that fixed basis with no further convolution at all.
 *
 * Fast path, FIT_GLOBAL_MULTI_STRETCHEDEXP (same eligibility): no bundled
 * FLIMLib fit_type understands "N independently distributed components
 * summed together", so this can't reuse GCI_marquardt_global_exps_
 * do_fit_single -- instead it calls GCI_marquardt_global_exps_calculate_
 * exps_instr once *per component* per IRF group (still just once per
 * group, the same actual speed win), then fits each pixel via
 * fit_linear_multi_stretched_pixel's own small weighted linear solve
 * (genuinely linear once tau/h are fixed, so it needs no nonlinear
 * search of its own at all) instead of a bundled Marquardt routine.
 *
 * Fallback path, all fit types (groups == NULL, or nBins too large for
 * the scratch limit above): the original independent per-pixel
 * fit_fixed_tau_pixel loop. All three paths compute the exact same
 * objective definition (summed per-pixel reduced chi-square) and
 * populate the same output maps, so callers -- including the outer
 * Marquardt-damped-Newton search -- see no behavioural difference beyond
 * speed.
 * ------------------------------------------------------------------- */
static double score_global_taus_with_irfs(
    const double *decay,
    const double *irf,
    const int *sourcePixel,
    int selectedCount,
    int nBins,
    double binWidth,
    int fitType,
    int modelOrder,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    int fitStartBin,
    int fitStopBin,
    int irfMode,
    const double *taus,
    int mapLength,
    double *tauMaps,
    double *tauMeanMap,
    double *amplitudeMaps,
    double *backgroundMap,
    double *chiSquareMap,
    double *fitStatusMap,
    int storeOutputs,
    const irf_grouping *groups)
{
    int failed = 0;
    double objective = 0.0;
    int stretched = fitType == FIT_GLOBAL_STRETCHEDEXP;
    /* How many (tau, amplitude) components actually get a per-pixel map
     * entry -- 1 for stretched-exp (one distributed component; its
     * second shared parameter, h, isn't per-pixel and is returned once,
     * at the dataset level, by the caller instead), modelOrder for
     * multiexp exactly as before. */
    int componentCount = stretched ? 1 : modelOrder;

    if (!groups || groups->numGroups <= 0 || nBins > FLIMFIT_MAX_GLOBAL_FAST_BINS) {
        int i, k;
        double amplitudes[3] = {FLIMFIT_NAN, FLIMFIT_NAN, FLIMFIT_NAN};

        for (i = 0; i < selectedCount; ++i) {
            int pixel = sourcePixel[i];
            const double *decayPixel = decay + (size_t)pixel * nBins;
            const double *irfPixel = irfMode == 0 ? irf : irf + (size_t)pixel * nBins;
            double bg = FLIMFIT_NAN;
            double chi = FLIMFIT_NAN;
            double tauMean = FLIMFIT_NAN;
            int status = fit_fixed_tau_pixel(decayPixel, irfPixel, nBins, binWidth, fitType,
                modelOrder, backgroundMode, backgroundValue, noiseModel,
                fitStartBin, fitStopBin, taus, amplitudes, &bg, &chi, &tauMean);
            if (status != 1 || !isfinite(chi)) {
                ++failed;
                if (storeOutputs) fitStatusMap[pixel] = (double)status;
                objective += 1e30;
                continue;
            }
            objective += chi;
            if (fitStatusMap) fitStatusMap[pixel] = 1.0;
            if (storeOutputs) {
                backgroundMap[pixel] = bg;
                chiSquareMap[pixel] = chi;
                tauMeanMap[pixel] = tauMean;
                for (k = 0; k < componentCount; ++k) {
                    /* Not simply taus[k] -- for FIT_GLOBAL_MULTI_STRETCHEDEXP
                     * the shared vector interleaves each component's
                     * (tau, h) pair, so component k's tau lives at slot 2k,
                     * not slot k. See shared_tau_for_component. */
                    tauMaps[k * mapLength + pixel] = shared_tau_for_component(fitType, taus, k);
                    amplitudeMaps[k * mapLength + pixel] = amplitudes[k];
                }
            }
        }

        if (failed >= selectedCount) return DBL_MAX / 4.0;
        return objective;
    }

    if (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP) {
        /* Fast path for up to 3 independently-distributed components.
         * No bundled FLIMLib fit_type understands this combined model
         * (their own `ftype` switch has no case for an app-defined
         * value), so this doesn't reuse GCI_marquardt_global_exps_
         * do_fit_single the way the multiexp/single-stretched branch
         * below does -- it computes each component's IRF-convolved basis
         * shape itself, once per component per IRF group (the actual
         * speed win, identical in spirit to the multiexp/stretched fast
         * path's one-convolution-per-group reuse), via
         * stretched_component_basis -- see that function's own comment
         * for why this fast path writes its own convolution rather than
         * reusing GCI_marquardt_global_exps_calculate_exps_instr the way
         * an earlier version of this code did (that bundled function's
         * hardwired bin-0 time anchor, shared with every other fast/
         * grouped path in this codebase, is a real, measured source of
         * bias -- RECONSTRUCTION_NOTES.md has the full account). Each
         * pixel's Z/amplitudes are then fit against the combined basis
         * via fit_linear_multi_stretched_pixel's small weighted linear
         * solve (a real nonlinear Marquardt search isn't needed for that
         * step at all, since tau/h are fixed here -- see that function's
         * own comment). */
        float *basisBlock = (float *)malloc((size_t)modelOrder * nBins * sizeof(float));
        float *instrGroup = (float *)calloc((size_t)nBins, sizeof(float));
        float *y = (float *)calloc((size_t)nBins, sizeof(float));
        float *pureScratch = (float *)calloc((size_t)nBins, sizeof(float));
        double *weightScratch = (double *)malloc((size_t)nBins * sizeof(double));
        noise_type noise = map_noise(noiseModel);
        int gi, k;

        if (!basisBlock || !instrGroup || !y || !pureScratch || !weightScratch) {
            free(basisBlock); free(instrGroup); free(y); free(pureScratch);
            free(weightScratch);
            return DBL_MAX / 4.0;
        }

        for (gi = 0; gi < groups->numGroups; ++gi) {
            const double *irfRow = irf + (size_t)groups->groupIrfPixel[gi] * nBins;
            int idx;
            double groupAnchorTime;

            copy_normalized_irf(irfRow, instrGroup, nBins);

            /* The basis's time anchor is bin 0 -- deliberately, and
             * confirmed correct (not just "matches FLIMLib's own
             * shortcut") by direct experiment: see RECONSTRUCTION_NOTES.md
             * for the full path that led here, including two earlier,
             * wrong attempts (fit_start, then infer_window's
             * transientStart heuristic) that each measurably made things
             * worse. The underlying reasoning, once traced through: for
             * ordinary reconvolution fitting (model (*) IRF), the pure
             * model's own t=0 is mathematically arbitrary -- the
             * convolution with the IRF at *its own*, correctly-measured
             * position is what reproduces the real observable timing,
             * regardless of where the model's own zero sits, as long as
             * it's early enough that the model is genuinely ~zero before
             * the IRF's own rise (true for bin 0 in any real acquisition,
             * which always starts recording before the excitation pulse
             * arrives). Anchoring anywhere *after* that point (fit_start,
             * or an imprecisely-estimated transient-start heuristic)
             * introduces a real, avoidable error instead of removing one.
             * bin 0 is not a shortcut FLIMLib's own array-optimized
             * functions take at the cost of accuracy -- it's the
             * physically correct choice, confirmed empirically: recovers
             * true tau=3.0ns/h=2.5 as 3.14ns/2.50 on the validation
             * scenario, closer than either intermediate anchor attempt
             * and closer than the legacy single-component fast path's
             * own answer. infer_transient_start_time (above) is kept in
             * source as a separately-verified, possibly-useful building
             * block, just not used here. */
            groupAnchorTime = 0.0;

            for (k = 0; k < modelOrder; ++k) {
                double tau = shared_tau_for_component(fitType, taus, k);
                double h = bound_shared_param(fitType, 2 * k + 1, taus[2 * k + 1], binWidth, nBins);
                stretched_component_basis(binWidth, nBins, instrGroup, tau, h, groupAnchorTime,
                    pureScratch, basisBlock + (size_t)k * nBins);
            }

            for (idx = groups->groupStart[gi]; idx < groups->groupStart[gi + 1]; ++idx) {
                int i = groups->memberOrder[idx];
                int pixel = sourcePixel[i];
                const double *decayPixel = decay + (size_t)pixel * nBins;
                int pixelFitStart, pixelFitStop, b, fitOk;
                double amplitudeOut[3];
                double bgOut = 0.0, reducedChiOut = 0.0;

                for (b = 0; b < nBins; ++b) y[b] = (float)decayPixel[b];

                pixelFitStart = fitStartBin > 0 ? fitStartBin - 1 : 0;
                pixelFitStop = fitStopBin > 0 ? fitStopBin - 1 : nBins - 1;
                if (fitStartBin <= 0 || fitStopBin <= 0) {
                    double shift = 0.0;
                    infer_window(y, instrGroup, nBins, binWidth, &pixelFitStart, &pixelFitStop, &shift);
                }
                if (pixelFitStart < 0) pixelFitStart = 0;
                if (pixelFitStop > nBins - 1) pixelFitStop = nBins - 1;
                if (pixelFitStop <= pixelFitStart + modelOrder + 2) {
                    ++failed;
                    if (storeOutputs) fitStatusMap[pixel] = -6.0;
                    objective += 1e30;
                    continue;
                }

                fitOk = fit_linear_multi_stretched_pixel(y, nBins, pixelFitStart, pixelFitStop,
                    backgroundMode, backgroundValue, modelOrder, basisBlock, noise,
                    amplitudeOut, &bgOut, &reducedChiOut, weightScratch);

                if (!fitOk) {
                    ++failed;
                    if (storeOutputs) fitStatusMap[pixel] = -4.0;
                    objective += 1e30;
                    continue;
                }

                objective += reducedChiOut;
                if (fitStatusMap) fitStatusMap[pixel] = 1.0;
                if (storeOutputs) {
                    double ampSum = 0.0, tauWeighted = 0.0;
                    backgroundMap[pixel] = bgOut;
                    chiSquareMap[pixel] = reducedChiOut;
                    for (k = 0; k < modelOrder; ++k) {
                        double tau = shared_tau_for_component(fitType, taus, k);
                        tauMaps[k * mapLength + pixel] = tau;
                        amplitudeMaps[k * mapLength + pixel] = amplitudeOut[k];
                        ampSum += amplitudeOut[k];
                        tauWeighted += amplitudeOut[k] * tau;
                    }
                    tauMeanMap[pixel] = ampSum > 0.0 ? tauWeighted / ampSum : FLIMFIT_NAN;
                }
            }
        }

        free(basisBlock); free(instrGroup); free(y); free(pureScratch);
        free(weightScratch);

        if (failed >= selectedCount) return DBL_MAX / 4.0;
        return objective;
    }

    {
        int nparam = stretched ? 4 : (2 * modelOrder + 1);
        noise_type noise = map_noise(noiseModel);
        float sigma[1] = {1.0f};
        float *expBlock = (float *)malloc((size_t)nparam * nBins * sizeof(float));
        float *instrGroup = (float *)calloc((size_t)nBins, sizeof(float));
        float *y = (float *)calloc((size_t)nBins, sizeof(float));
        float *fitted = (float *)calloc((size_t)nBins, sizeof(float));
        float *residuals = (float *)calloc((size_t)nBins, sizeof(float));
        float *params = (float *)calloc((size_t)nparam, sizeof(float));
        int *paramfree = (int *)calloc((size_t)nparam, sizeof(int));
        float **covar = GCI_ecf_matrix(nparam, nparam);
        float **alpha = GCI_ecf_matrix(nparam, nparam);
        float *exp_conv[MAXFIT];
        float paramTemplate[MAXFIT];
        int k, gi, freeCount;
        int sharedCount = stretched ? 2 : modelOrder;

        if (!expBlock || !instrGroup || !y || !fitted || !residuals ||
            !params || !paramfree || !covar || !alpha) {
            free(expBlock); free(instrGroup); free(y); free(fitted); free(residuals);
            free(params); free(paramfree);
            if (covar) GCI_ecf_free_matrix(covar);
            if (alpha) GCI_ecf_free_matrix(alpha);
            return DBL_MAX / 4.0;
        }

        for (k = 0; k < nparam; ++k) {
            exp_conv[k] = expBlock + (size_t)k * nBins;
            paramTemplate[k] = 0.0f;
        }
        for (k = 0; k < sharedCount; ++k) {
            paramTemplate[shared_param_index(fitType, k)] = (float)bound_shared_param(fitType, k, taus[k], binWidth, nBins);
        }

        paramfree[0] = backgroundMode == 1 ? 0 : 1;
        if (stretched) {
            paramfree[1] = 1;  /* amplitude free */
            paramfree[2] = 0;  /* tau fixed */
            paramfree[3] = 0;  /* h fixed */
        } else {
            for (k = 0; k < modelOrder; ++k) {
                paramfree[1 + 2 * k] = 1;
                paramfree[2 + 2 * k] = 0;
            }
        }
        freeCount = 0;
        for (k = 0; k < nparam; ++k) if (paramfree[k]) ++freeCount;

        for (gi = 0; gi < groups->numGroups; ++gi) {
            const double *irfRow = irf + (size_t)groups->groupIrfPixel[gi] * nBins;
            int convRet;
            int idx;

            copy_normalized_irf(irfRow, instrGroup, nBins);
            /* `y` is reused purely as calculate_exps_instr's exp_pure
             * scratch buffer here -- it's fully overwritten with each
             * pixel's own decay before being used for anything else
             * below. */
            convRet = GCI_marquardt_global_exps_calculate_exps_instr(
                (float)binWidth, nBins, instrGroup, nBins,
                fitType, paramTemplate, nparam, y, exp_conv);

            if (convRet != 0) {
                for (idx = groups->groupStart[gi]; idx < groups->groupStart[gi + 1]; ++idx) {
                    int i = groups->memberOrder[idx];
                    ++failed;
                    if (storeOutputs) fitStatusMap[sourcePixel[i]] = -8.0;
                    objective += 1e30;
                }
                continue;
            }

            for (idx = groups->groupStart[gi]; idx < groups->groupStart[gi + 1]; ++idx) {
                int i = groups->memberOrder[idx];
                int pixel = sourcePixel[i];
                const double *decayPixel = decay + (size_t)pixel * nBins;
                int pixelFitStart, pixelFitStop, ret, b;
                double bg, ampSeed;
                float chisqRaw;

                for (b = 0; b < nBins; ++b) y[b] = (float)decayPixel[b];

                pixelFitStart = fitStartBin > 0 ? fitStartBin - 1 : 0;
                pixelFitStop = fitStopBin > 0 ? fitStopBin - 1 : nBins - 1;
                if (fitStartBin <= 0 || fitStopBin <= 0) {
                    double shift = 0.0;
                    infer_window(y, instrGroup, nBins, binWidth, &pixelFitStart, &pixelFitStop, &shift);
                }
                if (pixelFitStart < 0) pixelFitStart = 0;
                if (pixelFitStop > nBins - 1) pixelFitStop = nBins - 1;
                if (pixelFitStop <= pixelFitStart + componentCount + 2) {
                    ++failed;
                    if (storeOutputs) fitStatusMap[pixel] = -6.0;
                    objective += 1e30;
                    continue;
                }

                bg = backgroundMode == 1 ? backgroundValue : estimate_background(y, nBins);
                params[0] = (float)bg;
                ampSeed = 0.0;
                for (b = pixelFitStart; b <= pixelFitStop; ++b) {
                    double v = y[b] - bg;
                    if (v > ampSeed) ampSeed = v;
                }
                if (ampSeed <= 0.0) ampSeed = total_counts(decayPixel, nBins) / nBins;
                if (stretched) {
                    params[1] = (float)ampSeed;
                    params[2] = paramTemplate[2];
                    params[3] = paramTemplate[3];
                } else {
                    for (k = 0; k < modelOrder; ++k) {
                        params[1 + 2 * k] = (float)(ampSeed / modelOrder);
                        params[2 + 2 * k] = paramTemplate[2 + 2 * k];
                    }
                }

                ret = GCI_marquardt_global_exps_do_fit_single(
                    (float)binWidth, y, nBins, pixelFitStart, pixelFitStop + 1,
                    noise, noise == NOISE_CONST ? sigma : NULL, fitType,
                    params, paramfree, nparam, ECF_RESTRAIN_DEFAULT, 0.001f,
                    exp_conv, fitted, residuals, covar, alpha, &chisqRaw);

                if (ret < 0) {
                    ++failed;
                    if (storeOutputs) fitStatusMap[pixel] = (double)ret;
                    objective += 1e30;
                    continue;
                }

                {
                    double reducedChi = (double)chisqRaw /
                        fmax(1.0, (double)(pixelFitStop - pixelFitStart + 1 - freeCount));
                    objective += reducedChi;
                    if (fitStatusMap) fitStatusMap[pixel] = 1.0;
                    if (storeOutputs) {
                        double ampSum = 0.0, tauWeighted = 0.0;
                        backgroundMap[pixel] = (double)params[0];
                        chiSquareMap[pixel] = reducedChi;
                        for (k = 0; k < componentCount; ++k) {
                            int ampIdx = stretched ? 1 : (1 + 2 * k);
                            int tauIdx = shared_param_index(fitType, k);
                            double amp = (double)params[ampIdx];
                            double tau = (double)params[tauIdx];
                            tauMaps[k * mapLength + pixel] = tau;
                            amplitudeMaps[k * mapLength + pixel] = amp;
                            ampSum += amp;
                            tauWeighted += amp * tau;
                        }
                        tauMeanMap[pixel] = ampSum > 0.0 ? tauWeighted / ampSum : FLIMFIT_NAN;
                    }
                }
            }
        }

        free(expBlock); free(instrGroup); free(y); free(fitted); free(residuals);
        free(params); free(paramfree);
        GCI_ecf_free_matrix(covar);
        GCI_ecf_free_matrix(alpha);
    }

    if (failed >= selectedCount) return DBL_MAX / 4.0;
    return objective;
}

/* ---------------------------------------------------------------------
 * Shared-tau search via a Marquardt-style damped Newton step, replacing
 * the Nelder-Mead simplex search this file used previously (which itself
 * replaced an even older per-coordinate +-step grid/coordinate-descent
 * search). Both earlier searches were derivative-free: each trial point
 * only ever asked "what is the pooled chi-square here", with no memory
 * of curvature between trials -- Nelder-Mead alone needed on the order
 * of 15 evaluations per outer iteration just to *infer* a good direction
 * for 2-3 parameters, which is expensive because every single evaluation
 * already does a correct, real FLIMLib Marquardt fit per selected pixel
 * (fit_fixed_tau_pixel, via score_global_taus_with_irfs).
 *
 * This version instead builds a finite-difference gradient and Hessian
 * of that same scalar objective at each iterate, then takes a genuinely
 * Marquardt-damped Newton step: solve (H + lambda*diag(H)) * delta = -g,
 * accept it (shrink lambda, trust the Newton step more) if it improves
 * the objective, or reject it (grow lambda, fall back toward damped
 * steepest descent) otherwise -- Marquardt's original 1963 trust-region
 * logic, applied to this pooled-chi-square objective via numerical
 * derivatives instead of to raw per-timepoint residuals via an analytic
 * Jacobian. A full gradient+Hessian costs 2n + 4*C(n,2) evaluations
 * (18 for n=3) -- typically far fewer per outer iteration than
 * Nelder-Mead needed, and each accepted step moves toward the minimum
 * using real curvature rather than a blind reflection/expansion guess.
 *
 * This intentionally still does NOT call GCI_marquardt_global_exps_instr
 * (the upstream FLIMLib global routine used by the older, single-IRF-only
 * flim_fit_global_pixels_v1): that routine's signature only accepts one
 * IRF trace for the whole call, so it can't be reused here without
 * dropping per-pixel/per-detector IRF support (irfMode=1). The per-pixel
 * inner fit is still the same genuine FLIMLib Marquardt fit as always --
 * only the *outer* search over the shared taus changes here.
 * ------------------------------------------------------------------- */

typedef struct {
    const double *decay;
    const double *irf;
    const int *sourcePixel;
    int selectedCount;
    int nBins;
    double binWidth;
    int fitType;
    int modelOrder;
    int sharedCount; /* modelOrder for FIT_GLOBAL_MULTIEXP; always 2 ([tau, h])
                       * for FIT_GLOBAL_STRETCHEDEXP; 2*modelOrder
                       * ([tau0,h0,tau1,h1,...]) for FIT_GLOBAL_
                       * MULTI_STRETCHEDEXP -- see shared_param_index. */
    int backgroundMode;
    double backgroundValue;
    int noiseModel;
    int fitStartBin;
    int fitStopBin;
    int irfMode;
    int mapLength;
    const irf_grouping *groups; /* built once by the caller, reused across
                                  * every trial-tau evaluation -- the
                                  * grouping itself doesn't depend on tau,
                                  * only on which pixels were selected and
                                  * their IRFs, so recomputing it per
                                  * evaluation would waste the fast path's
                                  * whole advantage. */
} global_tau_objective_context;

static double evaluate_global_tau_objective(const global_tau_objective_context *ctx, const double *taus)
{
    double bounded[MAX_SHARED_PARAMS];
    int k;
    double objective;
    for (k = 0; k < ctx->sharedCount; ++k) {
        bounded[k] = bound_shared_param(ctx->fitType, k, taus[k], ctx->binWidth, ctx->nBins);
    }
    objective = score_global_taus_with_irfs(ctx->decay, ctx->irf, ctx->sourcePixel,
        ctx->selectedCount, ctx->nBins, ctx->binWidth, ctx->fitType, ctx->modelOrder,
        ctx->backgroundMode, ctx->backgroundValue, ctx->noiseModel,
        ctx->fitStartBin, ctx->fitStopBin, ctx->irfMode,
        bounded, ctx->mapLength, NULL, NULL, NULL, NULL, NULL, NULL, 0, ctx->groups);

    /* h/tau soft-ceiling regularization -- outer search only, see
     * H_REG_SOFT_CEILING's and TAU_REG_WINDOW_MULTIPLE's own comments
     * for the full rationale (two separate penalties: an earlier direct
     * test found h-only insufficient, since the search just found an
     * equally-degenerate escape via large tau instead). Skipped when
     * the raw fit already failed outright (score_global_taus_with_irfs's
     * own "unusable" sentinel is 1e30 per failed pixel, or DBL_MAX/4 if
     * every pixel failed) -- no point steering the search away from a
     * point that's already rejected. Scaled by selectedCount so each
     * penalty's strength relative to the summed per-pixel chi-square
     * objective doesn't silently change with image size (objective is
     * a straight sum over pixels, not a mean). */
    if (ctx->fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP && objective < 1e29) {
        double window = ctx->nBins * ctx->binWidth;
        double tauCeiling = TAU_REG_WINDOW_MULTIPLE * window;
        double hPenalty = 0.0, tauPenalty = 0.0;
        /* NOTE: a minimum-log-separation term was tried here too (same
         * shape as the h/tau ceilings above) and reverted -- confirmed
         * by direct test that adding ANY term to the search's own
         * objective reshapes the whole gradient/Hessian landscape along
         * the ENTIRE convergence path, not just near the final point,
         * and can reroute an already-good trajectory somewhere new and
         * worse even when the final point itself would never have
         * triggered the term. Concretely: N=2's already-validated
         * data-driven variant, which reliably converged to a genuinely
         * separated (1.27ns, 4.35ns) answer before, instead converged to
         * a DEGENERATE collapsed pair (1.44ns, 1.39ns) once this term
         * was added to the objective -- exactly the failure mode the
         * term was meant to prevent, self-inflicted by adding it. See
         * MIN_LOG_TAU_SEPARATION's own comment and
         * multi_stretched_result_is_plausible (which DOES still use this
         * threshold, safely, as a passive post-hoc filter rather than
         * as part of what the search itself optimizes against) for where
         * this idea lives now instead. */
        for (k = 0; k < ctx->modelOrder; ++k) {
            double h = bounded[2 * k + 1];
            double hOver = h - H_REG_SOFT_CEILING;
            if (hOver > 0.0) hPenalty += hOver * hOver;

            if (tauCeiling > 0.0) {
                double tau = bounded[2 * k];
                double tauOver = tau - tauCeiling;
                if (tauOver > 0.0) {
                    /* Normalize by the window before squaring, same
                     * reasoning as H_REG_SOFT_CEILING's own units --
                     * keeps the penalty dimensionless/comparable across
                     * datasets with different time ranges rather than
                     * scaling with binWidth^2. */
                    double tauOverNorm = tauOver / window;
                    tauPenalty += tauOverNorm * tauOverNorm;
                }
            }
        }
        objective += (double)ctx->selectedCount *
            (H_REG_LAMBDA * hPenalty + TAU_REG_LAMBDA * tauPenalty);
    }
    return objective;
}

/* Solves the n x n system A*x = b (n <= MAX_SHARED_PARAMS) via Gaussian elimination with
 * partial pivoting. Returns 1 on success, 0 if A is (numerically)
 * singular -- the caller treats that the same as a rejected step (raise
 * lambda, retry), same as Marquardt's method degrading gracefully toward
 * steepest descent under heavy damping. */
static int solve_linear_system(int n, double A[MAX_SHARED_PARAMS][MAX_SHARED_PARAMS], const double *b, double *x)
{
    double M[MAX_SHARED_PARAMS][MAX_SHARED_PARAMS + 1];
    int i, j, k, p;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) M[i][j] = A[i][j];
        M[i][n] = b[i];
    }

    for (k = 0; k < n; ++k) {
        double best = fabs(M[k][k]);
        p = k;
        for (i = k + 1; i < n; ++i) {
            if (fabs(M[i][k]) > best) { best = fabs(M[i][k]); p = i; }
        }
        if (best < 1e-300) return 0;
        if (p != k) {
            for (j = 0; j <= n; ++j) { double t = M[k][j]; M[k][j] = M[p][j]; M[p][j] = t; }
        }
        for (i = k + 1; i < n; ++i) {
            double f = M[i][k] / M[k][k];
            for (j = k; j <= n; ++j) M[i][j] -= f * M[k][j];
        }
    }

    for (i = n - 1; i >= 0; --i) {
        double sum = M[i][n];
        for (j = i + 1; j < n; ++j) sum -= M[i][j] * x[j];
        x[i] = sum / M[i][i];
    }
    return 1;
}

/* Minimizes evaluate_global_tau_objective over ctx->sharedCount (1 to
 * MAX_SHARED_PARAMS) shared parameters via a Marquardt-damped Newton search with
 * finite-difference gradient/Hessian. `bestTaus` is the initial guess on
 * entry and the best point found on return; `*bestObjective` is that
 * point's objective value. Returns the number of objective evaluations
 * actually used, for the same diagnostic slot the previous search's
 * iteration count used to fill. Bounded by maxEvaluations regardless of
 * convergence, so this can never run away on a pathological objective. */
static int levenberg_marquardt_min(
    const global_tau_objective_context *ctx,
    double *bestTaus,
    double *bestObjective,
    int maxEvaluations)
{
    int n = ctx->sharedCount;
    double current[MAX_SHARED_PARAMS], trial[MAX_SHARED_PARAMS];
    double fCurrent;
    double lambda = 1e-2;
    int evaluations = 0;
    int k;

    if (n < 1) n = 1;
    if (n > MAX_SHARED_PARAMS) n = MAX_SHARED_PARAMS;

    for (k = 0; k < n; ++k) current[k] = bound_shared_param(ctx->fitType, k, bestTaus[k], ctx->binWidth, ctx->nBins);
    fCurrent = evaluate_global_tau_objective(ctx, current);
    ++evaluations;

    while (evaluations + 4 * n < maxEvaluations) {
        double h[MAX_SHARED_PARAMS];
        double fPlus[MAX_SHARED_PARAMS], fMinus[MAX_SHARED_PARAMS];
        double grad[MAX_SHARED_PARAMS];
        double hess[MAX_SHARED_PARAMS][MAX_SHARED_PARAMS];
        double delta[MAX_SHARED_PARAMS];
        int j, retry, accepted;
        double gnorm;

        for (k = 0; k < n; ++k) {
            h[k] = fmax(1e-4, 0.01 * fabs(current[k]));
        }

        /* Diagonal Hessian terms reuse the same +-h evaluations as the
         * gradient -- no extra cost over a plain gradient computation. */
        for (k = 0; k < n; ++k) {
            for (j = 0; j < n; ++j) trial[j] = current[j];
            trial[k] = current[k] + h[k];
            fPlus[k] = evaluate_global_tau_objective(ctx, trial);
            ++evaluations;
            trial[k] = current[k] - h[k];
            fMinus[k] = evaluate_global_tau_objective(ctx, trial);
            ++evaluations;

            grad[k] = (fPlus[k] - fMinus[k]) / (2.0 * h[k]);
            hess[k][k] = (fPlus[k] - 2.0 * fCurrent + fMinus[k]) / (h[k] * h[k]);
            /* Non-convex/degenerate curvature: fall back to a
             * gradient-scaled diagonal so the damped solve below still
             * behaves like a sane (if conservative) descent step. */
            if (!(hess[k][k] > 1e-12)) hess[k][k] = fabs(grad[k]) + 1e-6;
        }

        for (k = 0; k < n && evaluations + 4 <= maxEvaluations; ++k) {
            for (j = k + 1; j < n && evaluations + 4 <= maxEvaluations; ++j) {
                double fpp, fpm, fmp, fmm;
                int m;
                for (m = 0; m < n; ++m) trial[m] = current[m];
                trial[k] = current[k] + h[k]; trial[j] = current[j] + h[j];
                fpp = evaluate_global_tau_objective(ctx, trial); ++evaluations;
                trial[k] = current[k] + h[k]; trial[j] = current[j] - h[j];
                fpm = evaluate_global_tau_objective(ctx, trial); ++evaluations;
                trial[k] = current[k] - h[k]; trial[j] = current[j] + h[j];
                fmp = evaluate_global_tau_objective(ctx, trial); ++evaluations;
                trial[k] = current[k] - h[k]; trial[j] = current[j] - h[j];
                fmm = evaluate_global_tau_objective(ctx, trial); ++evaluations;

                hess[k][j] = hess[j][k] = (fpp - fpm - fmp + fmm) / (4.0 * h[k] * h[j]);
            }
        }

        /* Converged: gradient is flat relative to the objective's own
         * scale. Stops early on an easy problem instead of always
         * burning the full evaluation budget. */
        gnorm = 0.0;
        for (k = 0; k < n; ++k) gnorm += grad[k] * grad[k];
        if (sqrt(gnorm) <= 1e-9 * (fabs(fCurrent) + 1e-12)) break;

        accepted = 0;
        for (retry = 0; retry < 8 && evaluations < maxEvaluations; ++retry) {
            double A[MAX_SHARED_PARAMS][MAX_SHARED_PARAMS];
            double negGrad[MAX_SHARED_PARAMS];
            double fTrial;

            for (k = 0; k < n; ++k) {
                for (j = 0; j < n; ++j) A[k][j] = hess[k][j];
                A[k][k] += lambda * hess[k][k];
                negGrad[k] = -grad[k];
            }

            if (!solve_linear_system(n, A, negGrad, delta)) {
                lambda *= 10.0;
                continue;
            }

            /* Trust-region cap: never let one step move a parameter to
             * less than half or more than double its current value,
             * regardless of what the raw damped-Newton formula suggests.
             * The finite-difference Hessian estimate this solves against
             * can occasionally come out spuriously small -- e.g. genuine
             * fine-scale noise from the *inner* per-pixel fit's own
             * convergence tolerance (fit_fixed_tau_pixel's chisq_delta)
             * showing up in the second-difference formula's subtraction,
             * which involves far more cancellation than the gradient's
             * simpler first-difference does -- which then implies a
             * wildly oversized step. Without a hard cap, that single bad
             * step can overshoot the true minimum by orders of
             * magnitude, and increasing lambda alone doesn't reliably
             * fix it (a step that's ~20x too large needs ~20x more
             * damping to shrink into range, quickly exceeding this
             * search's retry budget) -- confirmed by direct tracing: the
             * search got stuck retrying ever-higher lambda while the
             * clamped destination barely moved, because the raw
             * (uncapped) delta was still enormous even after 5^7-fold
             * damping. Capping the step directly, a standard trust-region
             * safeguard, instead forces a smaller-but-real move every
             * time, so a spurious single-step Hessian estimate costs an
             * extra outer iteration or two rather than derailing the
             * whole search. */
            for (k = 0; k < n; ++k) {
                double capped = delta[k];
                double lower_step = -0.5 * current[k];
                double upper_step = 1.0 * current[k];
                if (capped < lower_step) capped = lower_step;
                if (capped > upper_step) capped = upper_step;
                trial[k] = bound_shared_param(ctx->fitType, k, current[k] + capped, ctx->binWidth, ctx->nBins);
            }
            fTrial = evaluate_global_tau_objective(ctx, trial);
            ++evaluations;

            if (fTrial < fCurrent) {
                for (k = 0; k < n; ++k) current[k] = trial[k];
                fCurrent = fTrial;
                lambda = fmax(lambda / 3.0, 1e-8);
                accepted = 1;
                break;
            }
            lambda = fmin(lambda * 5.0, 1e8);
        }

        if (!accepted) break;
    }

    for (k = 0; k < n; ++k) bestTaus[k] = current[k];
    *bestObjective = fCurrent;
    return evaluations;
}

static int fit_pixels_impl(
    const double *decay,
    const double *irf,
    int rows,
    int columns,
    int nBins,
    double binWidth,
    int modelOrder,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    double minPhotons,
    int irfMode,
    double *tauMaps,
    double *tauMeanMap,
    double *amplitudeMaps,
    double *backgroundMap,
    double *chiSquareMap,
    double *timeShiftMap,
    double *fitStartMap,
    double *fitStopMap,
    double *exitflagMap,
    double *fitStatusMap,
    double *photonCountMap)
{
    int nPixels;
    int p, b, k;
    int mapSize;
    int nparam;
    noise_type noise = map_noise(noiseModel);
    float sigma[1] = {1.0f};

    if (!decay || !irf || !tauMaps || !tauMeanMap || !amplitudeMaps ||
        !backgroundMap || !chiSquareMap || !timeShiftMap || !fitStartMap ||
        !fitStopMap || !exitflagMap || !fitStatusMap || !photonCountMap ||
        rows <= 0 || columns <= 0 || nBins <= 3 || binWidth <= 0.0) return -1;
    if (rows > INT_MAX / columns) return -1;
    nPixels = rows * columns;
    mapSize = nPixels;
    if (modelOrder < 1) modelOrder = 1;
    if (modelOrder > 3) modelOrder = 3;
    nparam = 2 * modelOrder + 1;

    for (p = 0; p < nPixels; ++p) {
        tauMeanMap[p] = FLIMFIT_NAN;
        backgroundMap[p] = FLIMFIT_NAN;
        chiSquareMap[p] = FLIMFIT_NAN;
        timeShiftMap[p] = FLIMFIT_NAN;
        fitStartMap[p] = FLIMFIT_NAN;
        fitStopMap[p] = FLIMFIT_NAN;
        exitflagMap[p] = FLIMFIT_NAN;
        fitStatusMap[p] = 0.0;
        photonCountMap[p] = FLIMFIT_NAN;
        for (k = 0; k < modelOrder; ++k) {
            tauMaps[k * mapSize + p] = FLIMFIT_NAN;
            amplitudeMaps[k * mapSize + p] = FLIMFIT_NAN;
        }
    }

    for (p = 0; p < nPixels; ++p) {
        const double *decayPixel = decay + (size_t)p * nBins;
        const double *irfPixel = irfMode == 0 ? irf : irf + (size_t)p * nBins;
        double photons = total_counts(decayPixel, nBins);
        float *y = NULL;
        float *instr = NULL;
        float *fitted = NULL;
        float *residuals = NULL;
        float *params = NULL;
        int *paramfree = NULL;
        float **covar = NULL;
        float **alpha = NULL;
        float **erraxes = NULL;
        float chisq = 0.0f;
        int fitStart = 0, fitStop = nBins - 1;
        double shift = 0.0;
        double bg, tauSeed, ampSeed;
        int ret;
        int freeCount = 0;
        double ampSum = 0.0;
        double tauWeighted = 0.0;

        photonCountMap[p] = photons;
        if (!isfinite(photons) || all_zero_or_bad(decayPixel, nBins)) {
            fitStatusMap[p] = -1.0;
            continue;
        }
        if (photons < minPhotons) {
            fitStatusMap[p] = -2.0;
            continue;
        }
        if (all_zero_or_bad(irfPixel, nBins)) {
            fitStatusMap[p] = -3.0;
            continue;
        }

        y = (float *)calloc((size_t)nBins, sizeof(float));
        instr = (float *)calloc((size_t)nBins, sizeof(float));
        fitted = (float *)calloc((size_t)nBins, sizeof(float));
        residuals = (float *)calloc((size_t)nBins, sizeof(float));
        params = (float *)calloc((size_t)nparam, sizeof(float));
        paramfree = (int *)calloc((size_t)nparam, sizeof(int));
        covar = GCI_ecf_matrix(nparam, nparam);
        alpha = GCI_ecf_matrix(nparam, nparam);
        erraxes = GCI_ecf_matrix(nparam, nparam);
        if (!y || !instr || !fitted || !residuals || !params || !paramfree || !covar || !alpha || !erraxes) {
            fitStatusMap[p] = -5.0;
            goto cleanup;
        }

        for (b = 0; b < nBins; ++b) {
            y[b] = (float)decayPixel[b];
        }
        copy_normalized_irf(irfPixel, instr, nBins);

        infer_window(y, instr, nBins, binWidth, &fitStart, &fitStop, &shift);
        bg = backgroundMode == 1 ? backgroundValue : estimate_background(y, nBins);
        tauSeed = estimate_tau_moment(y, fitStart, fitStop, bg, binWidth);
        ampSeed = 0.0;
        for (b = fitStart; b <= fitStop; ++b) {
            double v = y[b] - bg;
            if (v > ampSeed) ampSeed = v;
        }
        if (ampSeed <= 0.0) ampSeed = photons / nBins;
        if (tauSeed <= 0.0 || !isfinite(tauSeed)) tauSeed = binWidth * nBins / 4.0;

        params[0] = (float)bg;
        paramfree[0] = backgroundMode == 1 ? 0 : 1;
        for (k = 0; k < modelOrder; ++k) {
            double scale = modelOrder == 1 ? 1.0 : (1.4 - (0.95 * k / (modelOrder - 1)));
            params[1 + 2 * k] = (float)(ampSeed / modelOrder);
            params[2 + 2 * k] = (float)fmax(1e-6, tauSeed * scale);
            paramfree[1 + 2 * k] = 1;
            paramfree[2 + 2 * k] = 1;
        }
        for (k = 0; k < nparam; ++k) {
            if (paramfree[k]) ++freeCount;
        }
        if (fitStop - fitStart + 1 <= freeCount) {
            fitStatusMap[p] = -6.0;
            goto cleanup;
        }

        ret = GCI_marquardt_fitting_engine(
            (float)binWidth, y, nBins, fitStart, fitStop + 1,
            instr, nBins, noise, noise == NOISE_CONST ? sigma : NULL, params, paramfree, nparam,
            ECF_RESTRAIN_DEFAULT, GCI_multiexp_tau, fitted, residuals,
            &chisq, covar, alpha, erraxes,
            1.1f * (float)fmax(1, fitStop - fitStart + 1 - freeCount),
            0.001f, 95);

        if (ret < 0) {
            fitStatusMap[p] = -4.0;
            exitflagMap[p] = (double)ret;
            goto cleanup;
        }

        backgroundMap[p] = params[0];
        chiSquareMap[p] = chisq / fmax(1.0, (double)(fitStop - fitStart + 1 - freeCount));
        timeShiftMap[p] = shift;
        fitStartMap[p] = (double)(fitStart + 1);
        fitStopMap[p] = (double)(fitStop + 1);
        exitflagMap[p] = (double)ret;
        fitStatusMap[p] = 1.0;
        for (k = 0; k < modelOrder; ++k) {
            double amp = params[1 + 2 * k];
            double tau = params[2 + 2 * k];
            amplitudeMaps[k * mapSize + p] = amp;
            tauMaps[k * mapSize + p] = tau;
            ampSum += amp;
            tauWeighted += amp * tau;
        }
        tauMeanMap[p] = ampSum > 0.0 ? tauWeighted / ampSum : FLIMFIT_NAN;

cleanup:
        if (covar) GCI_ecf_free_matrix(covar);
        if (alpha) GCI_ecf_free_matrix(alpha);
        if (erraxes) GCI_ecf_free_matrix(erraxes);
        free(y);
        free(instr);
        free(fitted);
        free(residuals);
        free(params);
        free(paramfree);
    }

    GCI_marquardt_cleanup();
    return 0;
}

FLIMFIT_API int FLIMFIT_CALL flim_fit_pixels(
    const double *decay,
    const double *irf,
    int rows,
    int columns,
    int nBins,
    double binWidth,
    int modelOrder,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    double minPhotons,
    double *tauMaps,
    double *tauMeanMap,
    double *amplitudeMaps,
    double *backgroundMap,
    double *chiSquareMap,
    double *timeShiftMap,
    double *fitStartMap,
    double *fitStopMap,
    double *exitflagMap,
    double *fitStatusMap,
    double *photonCountMap)
{
    return fit_pixels_impl(decay, irf, rows, columns, nBins, binWidth,
        modelOrder, backgroundMode, backgroundValue, noiseModel, minPhotons,
        1, tauMaps, tauMeanMap, amplitudeMaps, backgroundMap, chiSquareMap,
        timeShiftMap, fitStartMap, fitStopMap, exitflagMap, fitStatusMap,
        photonCountMap);
}

FLIMFIT_API int FLIMFIT_CALL flim_fit_pixels_v2(
    const double *decay,
    const double *irf,
    int rows,
    int columns,
    int nBins,
    int decayLength,
    int irfLength,
    double binWidth,
    int modelOrder,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    double minPhotons,
    int irfMode,
    double *tauMaps,
    int tauMapsLength,
    double *tauMeanMap,
    int tauMeanMapLength,
    double *amplitudeMaps,
    int amplitudeMapsLength,
    double *backgroundMap,
    int backgroundMapLength,
    double *chiSquareMap,
    int chiSquareMapLength,
    double *timeShiftMap,
    int timeShiftMapLength,
    double *fitStartMap,
    int fitStartMapLength,
    double *fitStopMap,
    int fitStopMapLength,
    double *exitflagMap,
    int exitflagMapLength,
    double *fitStatusMap,
    int fitStatusMapLength,
    double *photonCountMap,
    int photonCountMapLength)
{
    int nPixels;
    int mapLength;
    int componentLength;
    int expectedIrfLength;

    if (rows <= 0 || columns <= 0 || nBins <= 3) return -10;
    if (rows > INT_MAX / columns) return -11;
    nPixels = rows * columns;
    if (nPixels > INT_MAX / nBins) return -12;
    mapLength = nPixels;
    if (modelOrder < 1) modelOrder = 1;
    if (modelOrder > 3) modelOrder = 3;
    if (mapLength > INT_MAX / modelOrder) return -13;
    componentLength = modelOrder * mapLength;
    expectedIrfLength = irfMode == 0 ? nBins : mapLength * nBins;

    if (decayLength < mapLength * nBins) return -20;
    if (irfLength < expectedIrfLength) return -21;
    if (tauMapsLength < componentLength) return -30;
    if (amplitudeMapsLength < componentLength) return -31;
    if (tauMeanMapLength < mapLength) return -32;
    if (backgroundMapLength < mapLength) return -33;
    if (chiSquareMapLength < mapLength) return -34;
    if (timeShiftMapLength < mapLength) return -35;
    if (fitStartMapLength < mapLength) return -36;
    if (fitStopMapLength < mapLength) return -37;
    if (exitflagMapLength < mapLength) return -38;
    if (fitStatusMapLength < mapLength) return -39;
    if (photonCountMapLength < mapLength) return -40;

    return fit_pixels_impl(decay, irf, rows, columns, nBins, binWidth,
        modelOrder, backgroundMode, backgroundValue, noiseModel, minPhotons,
        irfMode == 0 ? 0 : 1, tauMaps, tauMeanMap, amplitudeMaps,
        backgroundMap, chiSquareMap, timeShiftMap, fitStartMap, fitStopMap,
        exitflagMap, fitStatusMap, photonCountMap);
}

FLIMFIT_API int FLIMFIT_CALL flim_fit_global_pixels_v1(
    const double *decay,
    const double *irf,
    const double *mask,
    int rows,
    int columns,
    int nBins,
    int decayLength,
    int irfLength,
    int maskLength,
    double binWidth,
    int modelOrder,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    double minPhotons,
    int fitStartBin,
    int fitStopBin,
    int maxGlobalPixels,
    const double *initialTaus,
    int initialTausLength,
    double *globalTauOut,
    int globalTauOutLength,
    double *globalStatsOut,
    int globalStatsOutLength,
    double *tauMaps,
    int tauMapsLength,
    double *tauMeanMap,
    int tauMeanMapLength,
    double *amplitudeMaps,
    int amplitudeMapsLength,
    double *backgroundMap,
    int backgroundMapLength,
    double *chiSquareMap,
    int chiSquareMapLength,
    double *fitStatusMap,
    int fitStatusMapLength,
    double *photonCountMap,
    int photonCountMapLength)
{
    int nPixels;
    int mapLength;
    int componentLength;
    int nparam;
    int p, b, k, i;
    int selectedCount = 0;
    int fitStart;
    int fitStop;
    int df = 0;
    int ret = 0;
    int localDof;
    float chisqGlobal = 0.0f;
    float *instr = NULL;
    float **trans = NULL;
    float **param = NULL;
    float **fitted = NULL;
    float **residuals = NULL;
    float *chisqTrans = NULL;
    int *sourcePixel = NULL;
    int *paramfree = NULL;
    noise_type noise = map_noise(noiseModel);
    float sigma[1] = {1.0f};

    if (rows <= 0 || columns <= 0 || nBins <= 3 || binWidth <= 0.0) return -10;
    if (rows > INT_MAX / columns) return -11;
    nPixels = rows * columns;
    if (nPixels > INT_MAX / nBins) return -12;
    mapLength = nPixels;
    if (modelOrder < 1) modelOrder = 1;
    if (modelOrder > 3) modelOrder = 3;
    if (mapLength > INT_MAX / modelOrder) return -13;
    componentLength = modelOrder * mapLength;
    nparam = 2 * modelOrder + 1;

    if (!decay || !irf || !mask || !globalTauOut || !globalStatsOut ||
        !tauMaps || !tauMeanMap || !amplitudeMaps || !backgroundMap ||
        !chiSquareMap || !fitStatusMap || !photonCountMap) return -14;
    if (decayLength < mapLength * nBins) return -20;
    if (irfLength < nBins) return -21;
    if (maskLength < mapLength) return -22;
    if (globalTauOutLength < modelOrder) return -23;
    if (globalStatsOutLength < 8) return -24;
    if (tauMapsLength < componentLength) return -30;
    if (amplitudeMapsLength < componentLength) return -31;
    if (tauMeanMapLength < mapLength) return -32;
    if (backgroundMapLength < mapLength) return -33;
    if (chiSquareMapLength < mapLength) return -34;
    if (fitStatusMapLength < mapLength) return -39;
    if (photonCountMapLength < mapLength) return -40;
    if (all_zero_or_bad(irf, nBins)) return -41;

    for (k = 0; k < modelOrder; ++k) globalTauOut[k] = FLIMFIT_NAN;
    for (i = 0; i < globalStatsOutLength; ++i) globalStatsOut[i] = FLIMFIT_NAN;
    for (p = 0; p < mapLength; ++p) {
        tauMeanMap[p] = FLIMFIT_NAN;
        backgroundMap[p] = FLIMFIT_NAN;
        chiSquareMap[p] = FLIMFIT_NAN;
        fitStatusMap[p] = 0.0;
        photonCountMap[p] = FLIMFIT_NAN;
        for (k = 0; k < modelOrder; ++k) {
            tauMaps[k * mapLength + p] = FLIMFIT_NAN;
            amplitudeMaps[k * mapLength + p] = FLIMFIT_NAN;
        }
    }

    for (p = 0; p < mapLength; ++p) {
        const double *decayPixel = decay + (size_t)p * nBins;
        double photons = total_counts(decayPixel, nBins);
        photonCountMap[p] = photons;
        if (mask[p] <= 0.0) continue;
        if (!isfinite(photons) || all_zero_or_bad(decayPixel, nBins)) {
            fitStatusMap[p] = -1.0;
            continue;
        }
        if (photons < minPhotons) {
            fitStatusMap[p] = -2.0;
            continue;
        }
        if (maxGlobalPixels > 0 && selectedCount >= maxGlobalPixels) {
            fitStatusMap[p] = -7.0;
            continue;
        }
        ++selectedCount;
    }

    globalStatsOut[4] = (double)selectedCount;
    if (selectedCount < modelOrder + 1) return -42;

    instr = (float *)calloc((size_t)nBins, sizeof(float));
    sourcePixel = (int *)calloc((size_t)selectedCount, sizeof(int));
    chisqTrans = (float *)calloc((size_t)selectedCount, sizeof(float));
    paramfree = (int *)calloc((size_t)nparam, sizeof(int));
    trans = GCI_ecf_matrix(selectedCount, nBins);
    param = GCI_ecf_matrix(selectedCount, nparam);
    fitted = GCI_ecf_matrix(selectedCount, nBins);
    residuals = GCI_ecf_matrix(selectedCount, nBins);
    if (!instr || !sourcePixel || !chisqTrans || !paramfree ||
        !trans || !param || !fitted || !residuals) {
        ret = -50;
        goto cleanup;
    }

    copy_normalized_irf(irf, instr, nBins);

    i = 0;
    for (p = 0; p < mapLength; ++p) {
        const double *decayPixel = decay + (size_t)p * nBins;
        double photons = photonCountMap[p];
        if (mask[p] <= 0.0 || fitStatusMap[p] != 0.0 ||
            !isfinite(photons) || photons < minPhotons ||
            all_zero_or_bad(decayPixel, nBins)) {
            continue;
        }
        if (i >= selectedCount) break;
        sourcePixel[i] = p;
        for (b = 0; b < nBins; ++b) trans[i][b] = (float)decayPixel[b];
        ++i;
    }

    fitStart = fitStartBin > 0 ? fitStartBin - 1 : 0;
    fitStop = fitStopBin > 0 ? fitStopBin - 1 : nBins - 1;
    if (fitStartBin <= 0 || fitStopBin <= 0) {
        double shift = 0.0;
        infer_window(trans[0], instr, nBins, binWidth, &fitStart, &fitStop, &shift);
    }
    if (fitStart < 0) fitStart = 0;
    if (fitStop > nBins - 1) fitStop = nBins - 1;
    if (fitStop <= fitStart + modelOrder + 2) {
        ret = -43;
        goto cleanup;
    }

    paramfree[0] = backgroundMode == 1 ? 0 : 1;
    for (k = 0; k < modelOrder; ++k) {
        paramfree[1 + 2 * k] = 1;
        paramfree[2 + 2 * k] = 1;
    }

    for (i = 0; i < selectedCount; ++i) {
        double bg = backgroundMode == 1 ? backgroundValue : estimate_background(trans[i], nBins);
        double tauSeed = estimate_tau_moment(trans[i], fitStart, fitStop, bg, binWidth);
        double ampSeed = 0.0;
        param[i][0] = (float)bg;
        for (b = fitStart; b <= fitStop; ++b) {
            double v = trans[i][b] - bg;
            if (v > ampSeed) ampSeed = v;
        }
        if (ampSeed <= 0.0) ampSeed = photonCountMap[sourcePixel[i]] / nBins;
        if (tauSeed <= 0.0 || !isfinite(tauSeed)) tauSeed = binWidth * nBins / 4.0;
        for (k = 0; k < modelOrder; ++k) {
            double scale = modelOrder == 1 ? 1.0 : (1.4 - (0.95 * k / (modelOrder - 1)));
            double tau = tauSeed * scale;
            if (initialTaus && initialTausLength > k && initialTaus[k] > 0.0) {
                tau = initialTaus[k];
            }
            param[i][1 + 2 * k] = (float)(ampSeed / modelOrder);
            param[i][2 + 2 * k] = (float)fmax(1e-6, tau);
        }
    }

    ret = GCI_marquardt_global_exps_instr(
        (float)binWidth, trans, nBins, selectedCount, fitStart, fitStop + 1,
        instr, nBins, noise, noise == NOISE_CONST ? sigma : NULL, FIT_GLOBAL_MULTIEXP, param, paramfree,
        nparam, ECF_RESTRAIN_DEFAULT, 0.001f, fitted, residuals,
        chisqTrans, &chisqGlobal, &df, 1);

    globalStatsOut[0] = (double)ret;
    globalStatsOut[1] = (double)chisqGlobal;
    globalStatsOut[2] = df > 0 ? (double)chisqGlobal / df : FLIMFIT_NAN;
    globalStatsOut[3] = (double)df;
    globalStatsOut[4] = (double)selectedCount;
    globalStatsOut[5] = (double)(fitStart + 1);
    globalStatsOut[6] = (double)(fitStop + 1);
    globalStatsOut[7] = (double)modelOrder;

    if (ret < 0) {
        for (i = 0; i < selectedCount; ++i) fitStatusMap[sourcePixel[i]] = -4.0;
        ret = -60 + ret;
        goto cleanup;
    }

    localDof = fitStop - fitStart + 1 - modelOrder - (backgroundMode == 1 ? 0 : 1);
    if (localDof < 1) localDof = 1;
    for (k = 0; k < modelOrder; ++k) {
        globalTauOut[k] = param[0][2 + 2 * k];
    }
    for (i = 0; i < selectedCount; ++i) {
        int pixel = sourcePixel[i];
        double ampSum = 0.0;
        double tauWeighted = 0.0;
        backgroundMap[pixel] = param[i][0];
        chiSquareMap[pixel] = chisqTrans[i] / localDof;
        fitStatusMap[pixel] = 1.0;
        for (k = 0; k < modelOrder; ++k) {
            double amp = param[i][1 + 2 * k];
            double tau = param[i][2 + 2 * k];
            amplitudeMaps[k * mapLength + pixel] = amp;
            tauMaps[k * mapLength + pixel] = tau;
            ampSum += amp;
            tauWeighted += amp * tau;
        }
        tauMeanMap[pixel] = ampSum > 0.0 ? tauWeighted / ampSum : FLIMFIT_NAN;
    }
    ret = 0;

cleanup:
    if (trans) GCI_ecf_free_matrix(trans);
    if (param) GCI_ecf_free_matrix(param);
    if (fitted) GCI_ecf_free_matrix(fitted);
    if (residuals) GCI_ecf_free_matrix(residuals);
    free(instr);
    free(sourcePixel);
    free(chisqTrans);
    free(paramfree);
    GCI_marquardt_cleanup();
    return ret;
}

FLIMFIT_API int FLIMFIT_CALL flim_fit_global_pixels_v2(
    const double *decay,
    const double *irf,
    const double *mask,
    int rows,
    int columns,
    int nBins,
    int decayLength,
    int irfLength,
    int maskLength,
    double binWidth,
    int modelOrder,
    int fitType,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    double minPhotons,
    int fitStartBin,
    int fitStopBin,
    int maxGlobalPixels,
    int irfMode,
    int globalIterations,
    const double *initialTaus,
    int initialTausLength,
    double *globalTauOut,
    int globalTauOutLength,
    double *globalStatsOut,
    int globalStatsOutLength,
    double *tauMaps,
    int tauMapsLength,
    double *tauMeanMap,
    int tauMeanMapLength,
    double *amplitudeMaps,
    int amplitudeMapsLength,
    double *backgroundMap,
    int backgroundMapLength,
    double *chiSquareMap,
    int chiSquareMapLength,
    double *fitStatusMap,
    int fitStatusMapLength,
    double *photonCountMap,
    int photonCountMapLength)
{
    int nPixels;
    int mapLength;
    int componentLength;
    int expectedIrfLength;
    int selectedCount = 0;
    int p, k, i;
    int *sourcePixel = NULL;
    int iterationsUsed = 0;
    int irfGroupCount = 0;
    int sharedCount;
    double bestTaus[MAX_SHARED_PARAMS] = {1.0, 0.5, 0.25, 0.0, 0.0, 0.0};
    double bestObjective;
    double finalObjective;

    if (rows <= 0 || columns <= 0 || nBins <= 3 || binWidth <= 0.0) return -10;
    if (rows > INT_MAX / columns) return -11;
    nPixels = rows * columns;
    if (nPixels > INT_MAX / nBins) return -12;
    mapLength = nPixels;
    if (modelOrder < 1) modelOrder = 1;
    if (modelOrder > 3) modelOrder = 3;
    if (fitType != FIT_GLOBAL_STRETCHEDEXP && fitType != FIT_GLOBAL_MULTI_STRETCHEDEXP) fitType = FIT_GLOBAL_MULTIEXP;
    /* Stretched-exp is always exactly one distributed component (Z, A,
     * tau, h -- see fit_fixed_tau_pixel/score_global_taus_with_irfs), so
     * modelOrder is forced here regardless of what the caller passed --
     * defensive, in case the Python layer's own model_order=1 forcing
     * (see models.py) is ever bypassed. */
    if (fitType == FIT_GLOBAL_STRETCHEDEXP) modelOrder = 1;
    sharedCount = fitType == FIT_GLOBAL_STRETCHEDEXP ? 2 : (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP ? 2 * modelOrder : modelOrder);
    if (mapLength > INT_MAX / modelOrder) return -13;
    componentLength = modelOrder * mapLength;
    irfMode = irfMode == 0 ? 0 : 1;
    expectedIrfLength = irfMode == 0 ? nBins : mapLength * nBins;
    if (globalIterations <= 0) globalIterations = 4;
    if (globalIterations > 80) globalIterations = 80;

    if (!decay || !irf || !mask || !globalTauOut || !globalStatsOut ||
        !tauMaps || !tauMeanMap || !amplitudeMaps || !backgroundMap ||
        !chiSquareMap || !fitStatusMap || !photonCountMap) return -14;
    if (decayLength < mapLength * nBins) return -20;
    if (irfLength < expectedIrfLength) return -21;
    if (maskLength < mapLength) return -22;
    if (globalTauOutLength < sharedCount) return -23;
    if (globalStatsOutLength < 10) return -24;
    if (tauMapsLength < componentLength) return -30;
    if (amplitudeMapsLength < componentLength) return -31;
    if (tauMeanMapLength < mapLength) return -32;
    if (backgroundMapLength < mapLength) return -33;
    if (chiSquareMapLength < mapLength) return -34;
    if (fitStatusMapLength < mapLength) return -39;
    if (photonCountMapLength < mapLength) return -40;
    if (irfMode == 0 && all_zero_or_bad(irf, nBins)) return -41;

    for (k = 0; k < sharedCount; ++k) globalTauOut[k] = FLIMFIT_NAN;
    for (i = 0; i < globalStatsOutLength; ++i) globalStatsOut[i] = FLIMFIT_NAN;
    for (p = 0; p < mapLength; ++p) {
        tauMeanMap[p] = FLIMFIT_NAN;
        backgroundMap[p] = FLIMFIT_NAN;
        chiSquareMap[p] = FLIMFIT_NAN;
        fitStatusMap[p] = 0.0;
        photonCountMap[p] = FLIMFIT_NAN;
        for (k = 0; k < modelOrder; ++k) {
            tauMaps[k * mapLength + p] = FLIMFIT_NAN;
            amplitudeMaps[k * mapLength + p] = FLIMFIT_NAN;
        }
    }

    for (p = 0; p < mapLength; ++p) {
        const double *decayPixel = decay + (size_t)p * nBins;
        const double *irfPixel = irfMode == 0 ? irf : irf + (size_t)p * nBins;
        double photons = total_counts(decayPixel, nBins);
        photonCountMap[p] = photons;
        if (mask[p] <= 0.0) continue;
        if (!isfinite(photons) || all_zero_or_bad(decayPixel, nBins)) {
            fitStatusMap[p] = -1.0;
            continue;
        }
        if (photons < minPhotons) {
            fitStatusMap[p] = -2.0;
            continue;
        }
        if (all_zero_or_bad(irfPixel, nBins)) {
            fitStatusMap[p] = -3.0;
            continue;
        }
        if (maxGlobalPixels > 0 && selectedCount >= maxGlobalPixels) {
            fitStatusMap[p] = -7.0;
            continue;
        }
        ++selectedCount;
    }

    globalStatsOut[4] = (double)selectedCount;
    if (selectedCount < modelOrder + 1) return -42;

    sourcePixel = (int *)calloc((size_t)selectedCount, sizeof(int));
    if (!sourcePixel) return -50;
    i = 0;
    for (p = 0; p < mapLength; ++p) {
        if (mask[p] > 0.0 && fitStatusMap[p] == 0.0) {
            if (i >= selectedCount) break;
            sourcePixel[i++] = p;
        }
    }

    {
        irf_grouping groups;
        /* Built once, before the search (and before/outside any multi-
         * start loop below) -- the grouping only depends on which pixels
         * were selected and their IRFs, never on the trial tau, so it's
         * reused unchanged across every evaluation and every start. On
         * allocation failure build_irf_groups leaves groups in its empty
         * (numGroups=0) state, which score_global_taus_with_irfs already
         * treats as "use the per-pixel fallback" -- safe either way. */
        int startCount = (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP &&
            !has_explicit_shared_seed(sharedCount, initialTaus, initialTausLength)) ? MULTI_START_COUNT : 1;
        int variant, bestVariant = -1;
        double dataDrivenTaus[MAX_SHARED_PARAMS / 2];
        int haveDataDrivenTaus = 0;
        double dataDrivenResultTaus[MAX_SHARED_PARAMS];
        double dataDrivenResultObjective = DBL_MAX;
        int dataDrivenResultIterations = 0;
        int dataDrivenResultCaptured = 0;

        build_irf_groups(irf, sourcePixel, selectedCount, nBins, irfMode, &groups);
        irfGroupCount = groups.numGroups;
        bestObjective = DBL_MAX;

        /* Computed once, before the multi-start loop (doesn't depend on
         * variant) -- see estimate_data_driven_multi_stretched_seed's
         * own comment. Only worth attempting when multi-start is even
         * going to run (startCount>1); harmless either way since
         * seed_multi_start_taus only consults this for variant 0. */
        if (startCount > 1) {
            haveDataDrivenTaus = estimate_data_driven_multi_stretched_seed(
                decay, sourcePixel, selectedCount, nBins, binWidth, modelOrder,
                backgroundMode, backgroundValue, fitStartBin, fitStopBin, dataDrivenTaus);
        }

        /* MULTI_START_COUNT diverse starts for FIT_GLOBAL_MULTI_
         * STRETCHEDEXP when the caller hasn't already given an explicit
         * seed (see has_explicit_shared_seed's own comment for why);
         * exactly one start, with the original single-seed formula
         * unchanged, for FIT_GLOBAL_MULTIEXP/FIT_GLOBAL_STRETCHEDEXP --
         * both already validated this way, no reason to add cost or
         * risk changing their behaviour. */
        for (variant = 0; variant < startCount; ++variant) {
            double trialTaus[MAX_SHARED_PARAMS];
            double trialObjective;
            int trialIterations;
            global_tau_objective_context ctx;

            if (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP) {
                seed_multi_start_taus(fitType, sharedCount, binWidth, nBins,
                    initialTaus, initialTausLength, variant,
                    haveDataDrivenTaus ? dataDrivenTaus : NULL, trialTaus);
            } else {
                for (k = 0; k < sharedCount; ++k) {
                    /* h (FIT_GLOBAL_STRETCHEDEXP's single slot 1) has
                     * nothing analogous to "a fraction of the
                     * acquisition window" to fall back on -- 1.5 is a
                     * mild, deliberately-not-exactly-1 initial stretch,
                     * far enough from the pure-single-exponential case
                     * (h=1) that the search has a real gradient signal
                     * in h to work with from the very first step. */
                    int isH = is_shared_slot_h(fitType, k);
                    double fallback = isH ? 1.5 : binWidth * nBins * (0.35 / (k + 1));
                    if (initialTaus && initialTausLength > k && initialTaus[k] > 0.0) {
                        fallback = initialTaus[k];
                    }
                    trialTaus[k] = bound_shared_param(fitType, k, fallback, binWidth, nBins);
                }
            }

            ctx.decay = decay;
            ctx.irf = irf;
            ctx.sourcePixel = sourcePixel;
            ctx.selectedCount = selectedCount;
            ctx.nBins = nBins;
            ctx.binWidth = binWidth;
            ctx.fitType = fitType;
            ctx.modelOrder = modelOrder;
            ctx.sharedCount = sharedCount;
            ctx.backgroundMode = backgroundMode;
            ctx.backgroundValue = backgroundValue;
            ctx.noiseModel = noiseModel;
            ctx.fitStartBin = fitStartBin;
            ctx.fitStopBin = fitStopBin;
            ctx.irfMode = irfMode;
            ctx.mapLength = mapLength;
            ctx.groups = &groups;
            /* globalIterations (4-80, sidebar "Global iterations") previously
             * meant "coordinate-descent sweeps", then "Nelder-Mead evaluation
             * budget / 15"; it still just scales the total objective-
             * evaluation budget handed to the outer search *per start* --
             * 15x (scaled up further for sharedCount>4 -- see
             * evaluation_budget_multiplier's own comment, found necessary
             * by direct N=3 test) keeps headroom generous (a full
             * gradient+Hessian costs up to 4n+2n(n-1) evaluations per
             * Newton iteration), while the search's own gradient-
             * convergence check (see levenberg_marquardt_min) means an
             * easy problem still stops early rather than spending the
             * whole budget. */
            trialIterations = levenberg_marquardt_min(&ctx, trialTaus, &trialObjective,
                (int)(globalIterations * evaluation_budget_multiplier(sharedCount)));

            /* Captured separately from the "lowest objective wins"
             * comparison just below -- see the post-loop override's own
             * comment for why variant 0's converged result gets tracked
             * on its own here, not just folded into bestObjective. */
            if (haveDataDrivenTaus && variant == 0) {
                dataDrivenResultObjective = trialObjective;
                dataDrivenResultIterations = trialIterations;
                dataDrivenResultCaptured = 1;
                for (k = 0; k < sharedCount; ++k) dataDrivenResultTaus[k] = trialTaus[k];
            }

            if (isfinite(trialObjective) && trialObjective < bestObjective) {
                bestObjective = trialObjective;
                bestVariant = variant;
                iterationsUsed = trialIterations;
                for (k = 0; k < sharedCount; ++k) bestTaus[k] = trialTaus[k];
            }
        }

        /* Prefer the data-driven variant's OWN converged answer over
         * whichever variant happens to report the lowest raw objective,
         * when it's available and still looks physically plausible after
         * convergence -- confirmed by direct experiment (RECONSTRUCTION_
         * NOTES.md's "improving the unseeded default case" entry) that
         * this matters: the data-driven start (grounded in this specific
         * dataset's own per-pixel apparent-lifetime spread -- see
         * estimate_data_driven_multi_stretched_seed) reliably converges
         * close to the true parameters, but one of the arbitrary,
         * data-*blind* formulaic variants can still converge to a
         * milder, still-plausible-looking "compromise" point with a
         * genuinely lower raw objective -- the same kind of noise-driven
         * non-identifiability documented elsewhere in this file, just
         * too mild to trip the h/tau regularization ceiling. Letting
         * "lowest objective wins" decide in that situation silently
         * discards the trustworthy, data-grounded answer in favour of a
         * blind guess that got lucky on this noise realization -- worse
         * than just trusting the data-driven start outright, so it wins
         * whenever it's available and plausible; the diverse formulaic
         * variants remain the fallback for when the data-driven start
         * *isn't* available or itself converges somewhere implausible
         * (e.g. too little real amplitude-ratio diversity in this
         * dataset for the estimate to mean anything). */
        if (dataDrivenResultCaptured && isfinite(dataDrivenResultObjective) &&
            multi_stretched_result_is_plausible(fitType, modelOrder, dataDrivenResultTaus, (double)nBins * binWidth) &&
            dataDrivenResultObjective <= DATA_DRIVEN_OBJECTIVE_TOLERANCE * bestObjective) {
            bestObjective = dataDrivenResultObjective;
            bestVariant = 0;
            iterationsUsed = dataDrivenResultIterations;
            for (k = 0; k < sharedCount; ++k) bestTaus[k] = dataDrivenResultTaus[k];
        }

        if (bestVariant < 0 || !isfinite(bestObjective) || bestObjective >= DBL_MAX / 8.0) {
            for (i = 0; i < selectedCount; ++i) fitStatusMap[sourcePixel[i]] = -4.0;
            free_irf_groups(&groups);
            free(sourcePixel);
            GCI_marquardt_cleanup();
            return -60;
        }

        finalObjective = score_global_taus_with_irfs(decay, irf, sourcePixel,
            selectedCount, nBins, binWidth, fitType, modelOrder, backgroundMode,
            backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
            bestTaus, mapLength, tauMaps, tauMeanMap, amplitudeMaps, backgroundMap,
            chiSquareMap, fitStatusMap, 1, &groups);

        free_irf_groups(&groups);

        /* globalTauOut[0]=tau always; for stretched-exp, globalTauOut[1]=h
         * too (see shared_param_index) -- for multiexp with modelOrder>1 this
         * naturally writes every component's tau, same as before. For multi-
         * stretched-exp it writes [tau0,h0,tau1,h1,...] -- one (tau,h) pair
         * per component, per shared_param_index's interleaved layout. */
        for (k = 0; k < sharedCount; ++k) globalTauOut[k] = bestTaus[k];
        /* Which multi-start variant won (0 for the single-start fit
         * types, always variant 0 there) -- a genuinely useful
         * diagnostic for a slot that was previously always 0.0. */
        globalStatsOut[0] = (double)bestVariant;
    }
    globalStatsOut[1] = finalObjective;
    globalStatsOut[2] = selectedCount > 0 ? finalObjective / selectedCount : FLIMFIT_NAN;
    globalStatsOut[3] = (double)iterationsUsed;
    globalStatsOut[4] = (double)selectedCount;
    globalStatsOut[5] = (double)fitStartBin;
    globalStatsOut[6] = (double)fitStopBin;
    globalStatsOut[7] = (double)modelOrder;
    globalStatsOut[8] = (double)irfMode;
    /* Number of distinct IRF groups the fast path collapsed the selected
     * pixels into (see build_irf_groups) -- 1 means every pixel shared
     * one IRF (irfMode=0, or irfMode=1 with byte-identical per-pixel
     * rows), selectedCount means no sharing was found at all (every
     * pixel got its own group, same cost as the old per-pixel path).
     * The single most useful speed diagnostic for irfMode=1: if this is
     * unexpectedly close to selectedCount, the per-pixel IRF data isn't
     * byte-identical across pixels meant to share a detector's IRF (e.g.
     * float rounding from upstream processing), which defeats the
     * grouping entirely. Previously the final coordinate-step size (a
     * diagnostic specific to the coordinate-descent search predating
     * Nelder-Mead); repurposed since neither Nelder-Mead nor this search
     * has an equivalent scalar of its own. */
    globalStatsOut[9] = (double)irfGroupCount;

    free(sourcePixel);
    GCI_marquardt_cleanup();
    return 0;
}

FLIMFIT_API int FLIMFIT_CALL flim_train_global_model_v1(
    const double *decay,
    const double *irf,
    const double *mask,
    int imageCount,
    int rows,
    int columns,
    int nBins,
    int decayLength,
    int irfLength,
    int maskLength,
    double binWidth,
    int modelOrder,
    int fitType,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    double minPhotons,
    int fitStartBin,
    int fitStopBin,
    int maxGlobalPixels,
    int irfMode,
    int globalIterations,
    const double *initialTaus,
    int initialTausLength,
    double *globalTauOut,
    int globalTauOutLength,
    double *globalStatsOut,
    int globalStatsOutLength,
    double *fitStatusMap,
    int fitStatusMapLength,
    double *photonCountMap,
    int photonCountMapLength)
{
    int pixelsPerImage;
    int totalPixels;
    int expectedIrfLength;
    int selectedCount = 0;
    int p, k, i;
    int *sourcePixel = NULL;
    int iterationsUsed = 0;
    int irfGroupCount = 0;
    int sharedCount;
    double bestTaus[MAX_SHARED_PARAMS] = {1.0, 0.5, 0.25, 0.0, 0.0, 0.0};
    double bestObjective;
    double finalObjective;

    if (imageCount <= 0 || rows <= 0 || columns <= 0 || nBins <= 3 || binWidth <= 0.0) return -10;
    if (rows > INT_MAX / columns) return -11;
    pixelsPerImage = rows * columns;
    if (imageCount > INT_MAX / pixelsPerImage) return -12;
    totalPixels = imageCount * pixelsPerImage;
    if (totalPixels > INT_MAX / nBins) return -13;
    if (modelOrder < 1) modelOrder = 1;
    if (modelOrder > 3) modelOrder = 3;
    if (fitType != FIT_GLOBAL_STRETCHEDEXP && fitType != FIT_GLOBAL_MULTI_STRETCHEDEXP) fitType = FIT_GLOBAL_MULTIEXP;
    if (fitType == FIT_GLOBAL_STRETCHEDEXP) modelOrder = 1;
    sharedCount = fitType == FIT_GLOBAL_STRETCHEDEXP ? 2 : (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP ? 2 * modelOrder : modelOrder);
    irfMode = irfMode == 0 ? 0 : 1;
    expectedIrfLength = irfMode == 0 ? nBins : totalPixels * nBins;
    if (globalIterations <= 0) globalIterations = 4;
    if (globalIterations > 80) globalIterations = 80;

    if (!decay || !irf || !mask || !globalTauOut || !globalStatsOut ||
        !fitStatusMap || !photonCountMap) return -14;
    if (decayLength < totalPixels * nBins) return -20;
    if (irfLength < expectedIrfLength) return -21;
    if (maskLength < totalPixels) return -22;
    if (globalTauOutLength < sharedCount) return -23;
    if (globalStatsOutLength < 10) return -24;
    if (fitStatusMapLength < totalPixels) return -39;
    if (photonCountMapLength < totalPixels) return -40;
    if (irfMode == 0 && all_zero_or_bad(irf, nBins)) return -41;

    for (k = 0; k < sharedCount; ++k) globalTauOut[k] = FLIMFIT_NAN;
    for (i = 0; i < globalStatsOutLength; ++i) globalStatsOut[i] = FLIMFIT_NAN;
    for (p = 0; p < totalPixels; ++p) {
        const double *decayPixel = decay + (size_t)p * nBins;
        const double *irfPixel = irfMode == 0 ? irf : irf + (size_t)p * nBins;
        double photons = total_counts(decayPixel, nBins);
        fitStatusMap[p] = 0.0;
        photonCountMap[p] = photons;
        if (mask[p] <= 0.0) continue;
        if (!isfinite(photons) || all_zero_or_bad(decayPixel, nBins)) {
            fitStatusMap[p] = -1.0;
            continue;
        }
        if (photons < minPhotons) {
            fitStatusMap[p] = -2.0;
            continue;
        }
        if (all_zero_or_bad(irfPixel, nBins)) {
            fitStatusMap[p] = -3.0;
            continue;
        }
        if (maxGlobalPixels > 0 && selectedCount >= maxGlobalPixels) {
            fitStatusMap[p] = -7.0;
            continue;
        }
        ++selectedCount;
    }

    globalStatsOut[4] = (double)selectedCount;
    if (selectedCount < modelOrder + 1) return -42;

    sourcePixel = (int *)calloc((size_t)selectedCount, sizeof(int));
    if (!sourcePixel) return -50;
    i = 0;
    for (p = 0; p < totalPixels; ++p) {
        if (mask[p] > 0.0 && fitStatusMap[p] == 0.0) {
            if (i >= selectedCount) break;
            sourcePixel[i++] = p;
        }
    }

    {
        irf_grouping groups;
        /* See the matching comment in flim_fit_global_pixels_v2 -- built
         * once, before the search (and before/outside the multi-start
         * loop below), and reused unchanged across every trial-tau
         * evaluation and every start. */
        int startCount = (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP &&
            !has_explicit_shared_seed(sharedCount, initialTaus, initialTausLength)) ? MULTI_START_COUNT : 1;
        int variant, bestVariant = -1;
        double dataDrivenTaus[MAX_SHARED_PARAMS / 2];
        int haveDataDrivenTaus = 0;
        double dataDrivenResultTaus[MAX_SHARED_PARAMS];
        double dataDrivenResultObjective = DBL_MAX;
        int dataDrivenResultIterations = 0;
        int dataDrivenResultCaptured = 0;

        build_irf_groups(irf, sourcePixel, selectedCount, nBins, irfMode, &groups);
        irfGroupCount = groups.numGroups;
        bestObjective = DBL_MAX;

        /* Computed once, before the multi-start loop (doesn't depend on
         * variant) -- see estimate_data_driven_multi_stretched_seed's
         * own comment. Only worth attempting when multi-start is even
         * going to run (startCount>1); harmless either way since
         * seed_multi_start_taus only consults this for variant 0. */
        if (startCount > 1) {
            haveDataDrivenTaus = estimate_data_driven_multi_stretched_seed(
                decay, sourcePixel, selectedCount, nBins, binWidth, modelOrder,
                backgroundMode, backgroundValue, fitStartBin, fitStopBin, dataDrivenTaus);
        }

        /* See the matching comment in flim_fit_global_pixels_v2 for why
         * this is a loop over MULTI_START_COUNT starts for FIT_GLOBAL_
         * MULTI_STRETCHEDEXP and exactly one, unchanged, start otherwise. */
        for (variant = 0; variant < startCount; ++variant) {
            double trialTaus[MAX_SHARED_PARAMS];
            double trialObjective;
            int trialIterations;
            global_tau_objective_context ctx;

            if (fitType == FIT_GLOBAL_MULTI_STRETCHEDEXP) {
                seed_multi_start_taus(fitType, sharedCount, binWidth, nBins,
                    initialTaus, initialTausLength, variant,
                    haveDataDrivenTaus ? dataDrivenTaus : NULL, trialTaus);
            } else {
                for (k = 0; k < sharedCount; ++k) {
                    int isH = is_shared_slot_h(fitType, k);
                    double fallback = isH ? 1.5 : binWidth * nBins * (0.35 / (k + 1));
                    if (initialTaus && initialTausLength > k && initialTaus[k] > 0.0) {
                        fallback = initialTaus[k];
                    }
                    trialTaus[k] = bound_shared_param(fitType, k, fallback, binWidth, nBins);
                }
            }

            ctx.decay = decay;
            ctx.irf = irf;
            ctx.sourcePixel = sourcePixel;
            ctx.selectedCount = selectedCount;
            ctx.nBins = nBins;
            ctx.binWidth = binWidth;
            ctx.fitType = fitType;
            ctx.modelOrder = modelOrder;
            ctx.sharedCount = sharedCount;
            ctx.backgroundMode = backgroundMode;
            ctx.backgroundValue = backgroundValue;
            ctx.noiseModel = noiseModel;
            ctx.fitStartBin = fitStartBin;
            ctx.fitStopBin = fitStopBin;
            ctx.irfMode = irfMode;
            ctx.mapLength = totalPixels;
            ctx.groups = &groups;
            /* See the matching comment in flim_fit_global_pixels_v2 -- same
             * reasoning, same per-start evaluation-budget scaling. */
            trialIterations = levenberg_marquardt_min(&ctx, trialTaus, &trialObjective,
                (int)(globalIterations * evaluation_budget_multiplier(sharedCount)));

            /* Captured separately from the "lowest objective wins"
             * comparison just below -- see the post-loop override's own
             * comment for why variant 0's converged result gets tracked
             * on its own here, not just folded into bestObjective. */
            if (haveDataDrivenTaus && variant == 0) {
                dataDrivenResultObjective = trialObjective;
                dataDrivenResultIterations = trialIterations;
                dataDrivenResultCaptured = 1;
                for (k = 0; k < sharedCount; ++k) dataDrivenResultTaus[k] = trialTaus[k];
            }

            if (isfinite(trialObjective) && trialObjective < bestObjective) {
                bestObjective = trialObjective;
                bestVariant = variant;
                iterationsUsed = trialIterations;
                for (k = 0; k < sharedCount; ++k) bestTaus[k] = trialTaus[k];
            }
        }

        /* Prefer the data-driven variant's OWN converged answer over
         * whichever variant happens to report the lowest raw objective,
         * when it's available and still looks physically plausible after
         * convergence -- confirmed by direct experiment (RECONSTRUCTION_
         * NOTES.md's "improving the unseeded default case" entry) that
         * this matters: the data-driven start (grounded in this specific
         * dataset's own per-pixel apparent-lifetime spread -- see
         * estimate_data_driven_multi_stretched_seed) reliably converges
         * close to the true parameters, but one of the arbitrary,
         * data-*blind* formulaic variants can still converge to a
         * milder, still-plausible-looking "compromise" point with a
         * genuinely lower raw objective -- the same kind of noise-driven
         * non-identifiability documented elsewhere in this file, just
         * too mild to trip the h/tau regularization ceiling. Letting
         * "lowest objective wins" decide in that situation silently
         * discards the trustworthy, data-grounded answer in favour of a
         * blind guess that got lucky on this noise realization -- worse
         * than just trusting the data-driven start outright, so it wins
         * whenever it's available and plausible; the diverse formulaic
         * variants remain the fallback for when the data-driven start
         * *isn't* available or itself converges somewhere implausible
         * (e.g. too little real amplitude-ratio diversity in this
         * dataset for the estimate to mean anything). */
        if (dataDrivenResultCaptured && isfinite(dataDrivenResultObjective) &&
            multi_stretched_result_is_plausible(fitType, modelOrder, dataDrivenResultTaus, (double)nBins * binWidth) &&
            dataDrivenResultObjective <= DATA_DRIVEN_OBJECTIVE_TOLERANCE * bestObjective) {
            bestObjective = dataDrivenResultObjective;
            bestVariant = 0;
            iterationsUsed = dataDrivenResultIterations;
            for (k = 0; k < sharedCount; ++k) bestTaus[k] = dataDrivenResultTaus[k];
        }

        if (bestVariant < 0 || !isfinite(bestObjective) || bestObjective >= DBL_MAX / 8.0) {
            for (i = 0; i < selectedCount; ++i) fitStatusMap[sourcePixel[i]] = -4.0;
            free_irf_groups(&groups);
            free(sourcePixel);
            GCI_marquardt_cleanup();
            return -60;
        }

        finalObjective = score_global_taus_with_irfs(decay, irf, sourcePixel,
            selectedCount, nBins, binWidth, fitType, modelOrder, backgroundMode,
            backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
            bestTaus, totalPixels, NULL, NULL, NULL, NULL, NULL, fitStatusMap, 0, &groups);

        free_irf_groups(&groups);

        for (k = 0; k < sharedCount; ++k) globalTauOut[k] = bestTaus[k];
        globalStatsOut[0] = (double)bestVariant;
    }
    globalStatsOut[1] = finalObjective;
    globalStatsOut[2] = selectedCount > 0 ? finalObjective / selectedCount : FLIMFIT_NAN;
    globalStatsOut[3] = (double)iterationsUsed;
    globalStatsOut[4] = (double)selectedCount;
    globalStatsOut[5] = (double)fitStartBin;
    globalStatsOut[6] = (double)fitStopBin;
    globalStatsOut[7] = (double)modelOrder;
    globalStatsOut[8] = (double)irfMode;
    /* See the matching comment in flim_fit_global_pixels_v2 for what this
     * slot means (IRF group count from build_irf_groups). */
    globalStatsOut[9] = (double)irfGroupCount;

    free(sourcePixel);
    GCI_marquardt_cleanup();
    return 0;
}

FLIMFIT_API int FLIMFIT_CALL flim_apply_global_model_v1(
    const double *decay,
    const double *irf,
    const double *mask,
    int imageCount,
    int rows,
    int columns,
    int nBins,
    int decayLength,
    int irfLength,
    int maskLength,
    double binWidth,
    int modelOrder,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    double minPhotons,
    int fitStartBin,
    int fitStopBin,
    int irfMode,
    const double *fixedTaus,
    int fixedTausLength,
    double *tauMaps,
    int tauMapsLength,
    double *tauMeanMap,
    int tauMeanMapLength,
    double *amplitudeMaps,
    int amplitudeMapsLength,
    double *backgroundMap,
    int backgroundMapLength,
    double *chiSquareMap,
    int chiSquareMapLength,
    double *fitStatusMap,
    int fitStatusMapLength,
    double *photonCountMap,
    int photonCountMapLength)
{
    int pixelsPerImage;
    int totalPixels;
    int componentLength;
    int expectedIrfLength;
    int p, k;
    double taus[3] = {1.0, 0.5, 0.25};
    double amplitudes[3] = {FLIMFIT_NAN, FLIMFIT_NAN, FLIMFIT_NAN};

    if (imageCount <= 0 || rows <= 0 || columns <= 0 || nBins <= 3 || binWidth <= 0.0) return -10;
    if (rows > INT_MAX / columns) return -11;
    pixelsPerImage = rows * columns;
    if (imageCount > INT_MAX / pixelsPerImage) return -12;
    totalPixels = imageCount * pixelsPerImage;
    if (totalPixels > INT_MAX / nBins) return -13;
    if (modelOrder < 1) modelOrder = 1;
    if (modelOrder > 3) modelOrder = 3;
    if (totalPixels > INT_MAX / modelOrder) return -15;
    componentLength = totalPixels * modelOrder;
    irfMode = irfMode == 0 ? 0 : 1;
    expectedIrfLength = irfMode == 0 ? nBins : totalPixels * nBins;

    if (!decay || !irf || !mask || !fixedTaus || !tauMaps || !tauMeanMap ||
        !amplitudeMaps || !backgroundMap || !chiSquareMap ||
        !fitStatusMap || !photonCountMap) return -14;
    if (decayLength < totalPixels * nBins) return -20;
    if (irfLength < expectedIrfLength) return -21;
    if (maskLength < totalPixels) return -22;
    if (fixedTausLength < modelOrder) return -23;
    if (tauMapsLength < componentLength) return -30;
    if (amplitudeMapsLength < componentLength) return -31;
    if (tauMeanMapLength < totalPixels) return -32;
    if (backgroundMapLength < totalPixels) return -33;
    if (chiSquareMapLength < totalPixels) return -34;
    if (fitStatusMapLength < totalPixels) return -39;
    if (photonCountMapLength < totalPixels) return -40;
    if (irfMode == 0 && all_zero_or_bad(irf, nBins)) return -41;

    for (k = 0; k < modelOrder; ++k) {
        taus[k] = bounded_tau(fixedTaus[k], binWidth, nBins);
    }

    for (p = 0; p < totalPixels; ++p) {
        const double *decayPixel = decay + (size_t)p * nBins;
        const double *irfPixel = irfMode == 0 ? irf : irf + (size_t)p * nBins;
        double photons = total_counts(decayPixel, nBins);
        double bg = FLIMFIT_NAN;
        double chi = FLIMFIT_NAN;
        double tauMean = FLIMFIT_NAN;
        int status;

        tauMeanMap[p] = FLIMFIT_NAN;
        backgroundMap[p] = FLIMFIT_NAN;
        chiSquareMap[p] = FLIMFIT_NAN;
        fitStatusMap[p] = 0.0;
        photonCountMap[p] = photons;
        for (k = 0; k < modelOrder; ++k) {
            tauMaps[k * totalPixels + p] = FLIMFIT_NAN;
            amplitudeMaps[k * totalPixels + p] = FLIMFIT_NAN;
        }

        if (mask[p] <= 0.0) continue;
        if (!isfinite(photons) || all_zero_or_bad(decayPixel, nBins)) {
            fitStatusMap[p] = -1.0;
            continue;
        }
        if (photons < minPhotons) {
            fitStatusMap[p] = -2.0;
            continue;
        }
        if (all_zero_or_bad(irfPixel, nBins)) {
            fitStatusMap[p] = -3.0;
            continue;
        }

        /* Always FIT_GLOBAL_MULTIEXP for now -- applying a stretched-exp
         * trained model isn't wired up yet (this round scoped the
         * distribution toggle to global fitting/training only; see
         * RECONSTRUCTION_NOTES.md). */
        status = fit_fixed_tau_pixel(decayPixel, irfPixel, nBins, binWidth, FIT_GLOBAL_MULTIEXP,
            modelOrder, backgroundMode, backgroundValue, noiseModel,
            fitStartBin, fitStopBin, taus, amplitudes, &bg, &chi, &tauMean);
        if (status != 1 || !isfinite(chi)) {
            fitStatusMap[p] = (double)status;
            continue;
        }
        backgroundMap[p] = bg;
        chiSquareMap[p] = chi;
        tauMeanMap[p] = tauMean;
        fitStatusMap[p] = 1.0;
        for (k = 0; k < modelOrder; ++k) {
            tauMaps[k * totalPixels + p] = taus[k];
            amplitudeMaps[k * totalPixels + p] = amplitudes[k];
        }
    }

    GCI_marquardt_cleanup();
    return 0;
}
