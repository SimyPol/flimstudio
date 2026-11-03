#include "flimfit_labview.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "Ecf.h"
#include "EcfGlobal.h"

#ifndef FLIMFIT_NAN
#define FLIMFIT_NAN ((double)NAN)
#endif

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
    if (!isfinite(tau) || tau <= 0.0) tau = binWidth * nBins / 4.0;
    if (tau < lo) tau = lo;
    if (tau > hi) tau = hi;
    return tau;
}

static int fit_fixed_tau_pixel(
    const double *decayPixel,
    const double *irfPixel,
    int nBins,
    double binWidth,
    int modelOrder,
    int backgroundMode,
    double backgroundValue,
    int noiseModel,
    int fitStartBin,
    int fitStopBin,
    const double *taus,
    double *amplitudeOut,
    double *backgroundOut,
    double *reducedChiOut,
    double *tauMeanOut)
{
    int b, k;
    int nparam = 2 * modelOrder + 1;
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
    if (fitStop <= fitStart + modelOrder + 2) {
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

    for (k = 0; k < modelOrder; ++k) {
        params[1 + 2 * k] = (float)(ampSeed / modelOrder);
        params[2 + 2 * k] = (float)bounded_tau(taus[k], binWidth, nBins);
        paramfree[1 + 2 * k] = 1;
        paramfree[2 + 2 * k] = 0;
    }
    for (k = 0; k < nparam; ++k) {
        if (paramfree[k]) ++freeCount;
    }

    ret = GCI_marquardt_fitting_engine(
        (float)binWidth, y, nBins, fitStart, fitStop + 1,
        instr, nBins, noise, noise == NOISE_CONST ? sigma : NULL, params, paramfree, nparam,
        ECF_RESTRAIN_DEFAULT, GCI_multiexp_tau, fitted, residuals,
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
    for (k = 0; k < modelOrder; ++k) {
        double amp = params[1 + 2 * k];
        double tau = params[2 + 2 * k];
        if (amplitudeOut) amplitudeOut[k] = amp;
        ampSum += amp;
        tauWeighted += amp * tau;
    }
    if (tauMeanOut) *tauMeanOut = ampSum > 0.0 ? tauWeighted / ampSum : FLIMFIT_NAN;
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

static double score_global_taus_with_irfs(
    const double *decay,
    const double *irf,
    const int *sourcePixel,
    int selectedCount,
    int nBins,
    double binWidth,
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
    int storeOutputs)
{
    int i, k;
    int failed = 0;
    double objective = 0.0;
    double amplitudes[3] = {FLIMFIT_NAN, FLIMFIT_NAN, FLIMFIT_NAN};

    for (i = 0; i < selectedCount; ++i) {
        int pixel = sourcePixel[i];
        const double *decayPixel = decay + (size_t)pixel * nBins;
        const double *irfPixel = irfMode == 0 ? irf : irf + (size_t)pixel * nBins;
        double bg = FLIMFIT_NAN;
        double chi = FLIMFIT_NAN;
        double tauMean = FLIMFIT_NAN;
        int status = fit_fixed_tau_pixel(decayPixel, irfPixel, nBins, binWidth,
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
            for (k = 0; k < modelOrder; ++k) {
                tauMaps[k * mapLength + pixel] = taus[k];
                amplitudeMaps[k * mapLength + pixel] = amplitudes[k];
            }
        }
    }

    if (failed >= selectedCount) return DBL_MAX / 4.0;
    return objective;
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
    int p, k, i, iter;
    int *sourcePixel = NULL;
    int iterationsUsed = 0;
    double bestTaus[3] = {1.0, 0.5, 0.25};
    double trialTaus[3] = {1.0, 0.5, 0.25};
    double bestObjective;
    double step = 0.35;
    double finalObjective;

    if (rows <= 0 || columns <= 0 || nBins <= 3 || binWidth <= 0.0) return -10;
    if (rows > INT_MAX / columns) return -11;
    nPixels = rows * columns;
    if (nPixels > INT_MAX / nBins) return -12;
    mapLength = nPixels;
    if (modelOrder < 1) modelOrder = 1;
    if (modelOrder > 3) modelOrder = 3;
    if (mapLength > INT_MAX / modelOrder) return -13;
    componentLength = modelOrder * mapLength;
    irfMode = irfMode == 0 ? 0 : 1;
    expectedIrfLength = irfMode == 0 ? nBins : mapLength * nBins;
    if (globalIterations <= 0) globalIterations = 4;
    if (globalIterations > 20) globalIterations = 20;

    if (!decay || !irf || !mask || !globalTauOut || !globalStatsOut ||
        !tauMaps || !tauMeanMap || !amplitudeMaps || !backgroundMap ||
        !chiSquareMap || !fitStatusMap || !photonCountMap) return -14;
    if (decayLength < mapLength * nBins) return -20;
    if (irfLength < expectedIrfLength) return -21;
    if (maskLength < mapLength) return -22;
    if (globalTauOutLength < modelOrder) return -23;
    if (globalStatsOutLength < 10) return -24;
    if (tauMapsLength < componentLength) return -30;
    if (amplitudeMapsLength < componentLength) return -31;
    if (tauMeanMapLength < mapLength) return -32;
    if (backgroundMapLength < mapLength) return -33;
    if (chiSquareMapLength < mapLength) return -34;
    if (fitStatusMapLength < mapLength) return -39;
    if (photonCountMapLength < mapLength) return -40;
    if (irfMode == 0 && all_zero_or_bad(irf, nBins)) return -41;

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

    for (k = 0; k < modelOrder; ++k) {
        double fallback = binWidth * nBins * (0.35 / (k + 1));
        if (initialTaus && initialTausLength > k && initialTaus[k] > 0.0) {
            fallback = initialTaus[k];
        }
        bestTaus[k] = bounded_tau(fallback, binWidth, nBins);
    }

    bestObjective = score_global_taus_with_irfs(decay, irf, sourcePixel,
        selectedCount, nBins, binWidth, modelOrder, backgroundMode,
        backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
        bestTaus, mapLength, NULL, NULL, NULL, NULL, NULL, NULL, 0);
    if (!isfinite(bestObjective) || bestObjective >= DBL_MAX / 8.0) {
        for (i = 0; i < selectedCount; ++i) fitStatusMap[sourcePixel[i]] = -4.0;
        free(sourcePixel);
        GCI_marquardt_cleanup();
        return -60;
    }

    for (iter = 0; iter < globalIterations; ++iter) {
        int improved = 0;
        for (k = 0; k < modelOrder; ++k) {
            double candidate;
            double objective;
            memcpy(trialTaus, bestTaus, sizeof(bestTaus));
            trialTaus[k] = bounded_tau(bestTaus[k] * exp(step), binWidth, nBins);
            objective = score_global_taus_with_irfs(decay, irf, sourcePixel,
                selectedCount, nBins, binWidth, modelOrder, backgroundMode,
                backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
                trialTaus, mapLength, NULL, NULL, NULL, NULL, NULL, NULL, 0);
            if (objective < bestObjective) {
                bestObjective = objective;
                memcpy(bestTaus, trialTaus, sizeof(bestTaus));
                improved = 1;
            }

            candidate = bestTaus[k] * exp(-step);
            memcpy(trialTaus, bestTaus, sizeof(bestTaus));
            trialTaus[k] = bounded_tau(candidate, binWidth, nBins);
            objective = score_global_taus_with_irfs(decay, irf, sourcePixel,
                selectedCount, nBins, binWidth, modelOrder, backgroundMode,
                backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
                trialTaus, mapLength, NULL, NULL, NULL, NULL, NULL, NULL, 0);
            if (objective < bestObjective) {
                bestObjective = objective;
                memcpy(bestTaus, trialTaus, sizeof(bestTaus));
                improved = 1;
            }
        }
        ++iterationsUsed;
        if (!improved) step *= 0.5;
        if (step < 0.02) break;
    }

    finalObjective = score_global_taus_with_irfs(decay, irf, sourcePixel,
        selectedCount, nBins, binWidth, modelOrder, backgroundMode,
        backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
        bestTaus, mapLength, tauMaps, tauMeanMap, amplitudeMaps, backgroundMap,
        chiSquareMap, fitStatusMap, 1);

    for (k = 0; k < modelOrder; ++k) globalTauOut[k] = bestTaus[k];
    globalStatsOut[0] = 0.0;
    globalStatsOut[1] = finalObjective;
    globalStatsOut[2] = selectedCount > 0 ? finalObjective / selectedCount : FLIMFIT_NAN;
    globalStatsOut[3] = (double)iterationsUsed;
    globalStatsOut[4] = (double)selectedCount;
    globalStatsOut[5] = (double)fitStartBin;
    globalStatsOut[6] = (double)fitStopBin;
    globalStatsOut[7] = (double)modelOrder;
    globalStatsOut[8] = (double)irfMode;
    globalStatsOut[9] = step;

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
    int p, k, i, iter;
    int *sourcePixel = NULL;
    int iterationsUsed = 0;
    double bestTaus[3] = {1.0, 0.5, 0.25};
    double trialTaus[3] = {1.0, 0.5, 0.25};
    double bestObjective;
    double step = 0.35;
    double finalObjective;

    if (imageCount <= 0 || rows <= 0 || columns <= 0 || nBins <= 3 || binWidth <= 0.0) return -10;
    if (rows > INT_MAX / columns) return -11;
    pixelsPerImage = rows * columns;
    if (imageCount > INT_MAX / pixelsPerImage) return -12;
    totalPixels = imageCount * pixelsPerImage;
    if (totalPixels > INT_MAX / nBins) return -13;
    if (modelOrder < 1) modelOrder = 1;
    if (modelOrder > 3) modelOrder = 3;
    irfMode = irfMode == 0 ? 0 : 1;
    expectedIrfLength = irfMode == 0 ? nBins : totalPixels * nBins;
    if (globalIterations <= 0) globalIterations = 4;
    if (globalIterations > 20) globalIterations = 20;

    if (!decay || !irf || !mask || !globalTauOut || !globalStatsOut ||
        !fitStatusMap || !photonCountMap) return -14;
    if (decayLength < totalPixels * nBins) return -20;
    if (irfLength < expectedIrfLength) return -21;
    if (maskLength < totalPixels) return -22;
    if (globalTauOutLength < modelOrder) return -23;
    if (globalStatsOutLength < 10) return -24;
    if (fitStatusMapLength < totalPixels) return -39;
    if (photonCountMapLength < totalPixels) return -40;
    if (irfMode == 0 && all_zero_or_bad(irf, nBins)) return -41;

    for (k = 0; k < modelOrder; ++k) globalTauOut[k] = FLIMFIT_NAN;
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

    for (k = 0; k < modelOrder; ++k) {
        double fallback = binWidth * nBins * (0.35 / (k + 1));
        if (initialTaus && initialTausLength > k && initialTaus[k] > 0.0) {
            fallback = initialTaus[k];
        }
        bestTaus[k] = bounded_tau(fallback, binWidth, nBins);
    }

    bestObjective = score_global_taus_with_irfs(decay, irf, sourcePixel,
        selectedCount, nBins, binWidth, modelOrder, backgroundMode,
        backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
        bestTaus, totalPixels, NULL, NULL, NULL, NULL, NULL, NULL, 0);
    if (!isfinite(bestObjective) || bestObjective >= DBL_MAX / 8.0) {
        for (i = 0; i < selectedCount; ++i) fitStatusMap[sourcePixel[i]] = -4.0;
        free(sourcePixel);
        GCI_marquardt_cleanup();
        return -60;
    }

    for (iter = 0; iter < globalIterations; ++iter) {
        int improved = 0;
        for (k = 0; k < modelOrder; ++k) {
            double objective;
            memcpy(trialTaus, bestTaus, sizeof(bestTaus));
            trialTaus[k] = bounded_tau(bestTaus[k] * exp(step), binWidth, nBins);
            objective = score_global_taus_with_irfs(decay, irf, sourcePixel,
                selectedCount, nBins, binWidth, modelOrder, backgroundMode,
                backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
                trialTaus, totalPixels, NULL, NULL, NULL, NULL, NULL, NULL, 0);
            if (objective < bestObjective) {
                bestObjective = objective;
                memcpy(bestTaus, trialTaus, sizeof(bestTaus));
                improved = 1;
            }

            memcpy(trialTaus, bestTaus, sizeof(bestTaus));
            trialTaus[k] = bounded_tau(bestTaus[k] * exp(-step), binWidth, nBins);
            objective = score_global_taus_with_irfs(decay, irf, sourcePixel,
                selectedCount, nBins, binWidth, modelOrder, backgroundMode,
                backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
                trialTaus, totalPixels, NULL, NULL, NULL, NULL, NULL, NULL, 0);
            if (objective < bestObjective) {
                bestObjective = objective;
                memcpy(bestTaus, trialTaus, sizeof(bestTaus));
                improved = 1;
            }
        }
        ++iterationsUsed;
        if (!improved) step *= 0.5;
        if (step < 0.02) break;
    }

    finalObjective = score_global_taus_with_irfs(decay, irf, sourcePixel,
        selectedCount, nBins, binWidth, modelOrder, backgroundMode,
        backgroundValue, noiseModel, fitStartBin, fitStopBin, irfMode,
        bestTaus, totalPixels, NULL, NULL, NULL, NULL, NULL, fitStatusMap, 0);

    for (k = 0; k < modelOrder; ++k) globalTauOut[k] = bestTaus[k];
    globalStatsOut[0] = 0.0;
    globalStatsOut[1] = finalObjective;
    globalStatsOut[2] = selectedCount > 0 ? finalObjective / selectedCount : FLIMFIT_NAN;
    globalStatsOut[3] = (double)iterationsUsed;
    globalStatsOut[4] = (double)selectedCount;
    globalStatsOut[5] = (double)fitStartBin;
    globalStatsOut[6] = (double)fitStopBin;
    globalStatsOut[7] = (double)modelOrder;
    globalStatsOut[8] = (double)irfMode;
    globalStatsOut[9] = step;

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

        status = fit_fixed_tau_pixel(decayPixel, irfPixel, nBins, binWidth,
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
