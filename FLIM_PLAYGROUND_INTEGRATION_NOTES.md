# FLIM Playground: capability inventory and integration notes

Planning document, not a spec — surveyed 2026-08-27 from
[skalalab.github.io/flim_playground_doc](https://skalalab.github.io/flim_playground_doc/).
Nothing here is implemented. Written to support a future "Analysis…"
window in FLIM Studio (see the closing section) — a per-topic capability
list, each with a verdict on how (or whether) it maps onto this app's
existing architecture.

## What FLIM Playground is

A CellProfiler-style companion tool, not a competing fitting engine:
**raw FLIM data → per-cell feature extraction (from an externally supplied
segmentation mask — it does not segment itself) → a per-cell feature
table → statistical/ML analysis** across cells, fields of view (FOVs),
and experiments. Explicitly modelled on CellProfiler/CellProfiler
Analyst, aimed at the fragmentation between tools that each only handle
one data level (pixel, cell, FOV, experiment).

A key architectural point for everything below: Playground fits/analyses
**per-cell summed decays** ("summing up all the decays belonging to the
same cell ROI"), not per-pixel maps aggregated afterward — the opposite
end of the pipeline from where FLIM Studio currently lives. FLIM Studio
*is*, in Playground's own terms, most of its "pixel-level Data
Extraction" step, already done to LabVIEW-validated precision; Playground's
whole value-add is the layer above the pixel level, which FLIM Studio has
none of today.

Supported input formats (for context — separate from the analysis
question): Becker & Hickl `.sdt`, PicoQuant `.ptu`, generic CSV (2D
decay), SPCImage `.asc` (pixel *pre-fit* lifetime features), plus
`.tiff`/`.tif` masks/intensity images/lifetime standards and `.txt` IRFs.
Neither `.sdt` nor `.ptu` are things FLIM Studio's own `io.py` reads today
— a separate, unrelated gap from everything below, noted here only for
completeness.

---

## 1. Data Extraction (Playground's pixel/cell layer)

### 1.1 FOV Metadata
Per-field-of-view bookkeeping: paths to each channel's mask/decay/IRF
file (matched by FOV name + configured suffix), channel names/modality,
decay duration/bins/laser rep rate (from file metadata or inferred),
which feature extractor is assigned to each channel, and lifetime-standard
calibration info if used. Exportable as CSV. Explicitly "the first step,"
feeding numeric extraction.

**Verdict — mostly not applicable.** FLIM Studio already tracks nearly
all of this per-dataset (`FLIMDataset.metadata`, the Metadata tab, JSON
provenance export) at the single-file level it operates on; it just has
no *FOV-batch* bookkeeping table across many files/channels at once,
because it has no multi-channel-per-FOV concept. Only relevant if/when
a batch feature table (§4) is built — that table's row-construction step
would need something equivalent (which file paths went into which row).

### 1.2 Numerical Feature Extraction — Lifetime Features (Fit)
Multi-exponential decay model `I(t) = Σ Aᵢ·e^(−t/τᵢ)`, 1–3 components:
- τᵢ (per component, ps)
- Aᵢ (amplitude) and αᵢ = Aᵢ / ΣA (fraction, %)
- τ_mean = Σ αᵢτᵢ (amplitude-weighted mean)
- τ_m,iw = (Σ αᵢτᵢ²) / (Σ αᵢτᵢ) (**intensity**-weighted mean — biases
  toward longer-lived components, distinct from τ_mean)
- reduced χ² = (1/(N−p)) Σ (yₙ−ŷₙ)²/ŷₙ
- constant offset (fitted baseline)

**Verdict — mostly already present, two small gaps.** FLIM Studio's
native FLIMLib reconvolution already produces τᵢ, Aᵢ, χ², background, and
its Lifetime map for multi-exponential results already *is* the
amplitude-weighted τ_mean (per `NativeFitResult`/`processing.py`). Gaps:
- **τ_m,iw (intensity-weighted mean)** isn't computed anywhere — cheap
  addition: one more array from the existing `tau_components_ns`/
  `amplitude_components` outputs, no new fit needed.
- **αᵢ as an explicit exported percentage** — the raw amplitudes are
  already there; a normalized-% version is a one-line derived field, not
  currently surfaced.

### 1.3 Numerical Feature Extraction — Lifetime Features (Fit-Free / Phasor)
First **and second** harmonic phasor coordinates:
- g, s (1st harmonic, raw) — Fourier transform of the decay
- g_2nd, s_2nd (2nd harmonic, raw)
- calibrated g/s and g_2nd/s_2nd (IRF-division or lifetime-standard
  calibration)
- τ_phase = tan(φ)/ω, τ_mod = (1/ω)·√(1/mod²−1)

**Verdict — 1st harmonic fully present, 2nd harmonic is the gap.**
`processing.phasor_lifetimes()` already computes calibrated (IRF-divided)
1st-harmonic G/S and both τ_phase/τ_mod, per-pixel, exactly matching this
list. The 2nd harmonic is the same integral at `2ω` instead of `ω` —
cheap to add as a second computed G/S pair, its own semicircle plot, and
useful specifically for resolving lifetime combinations that are
degenerate at the 1st harmonic but separate at the 2nd.

### 1.4 Numerical Feature Extraction — Morphological Features
Per-cell, via `skimage.measure.regionprops` on a mask: area, perimeter,
solidity, eccentricity, major/minor axis length, circularity (4πA/P²).

**Verdict — needs a segmentation/mask concept FLIM Studio doesn't have.**
These are all defined *on a cell mask*; FLIM Studio's only region concept
today is a plain rectangular ROI (trivially "solidity=1, circularity=1"
for any rectangle — not meaningful). Only becomes relevant alongside a
real segmentation feature (see §5).

### 1.5 Numerical Feature Extraction — Textural Features
Per-cell: granularity (n=1,3,5,7 px, via morphological opening),
radial distribution (4 concentric rings' intensity fraction), mass
displacement (intensity-weighted vs. geometric centroid distance),
intensity sum (total photon count).

**Verdict — intensity sum is trivial, the rest need real cell shape.**
"Intensity sum" is exactly `result.intensity` region-summed — FLIM Studio
already computes and exports total photon counts per ROI. Granularity/
radial distribution/mass displacement are all meaningfully defined only
against an actual cell boundary (their whole point is relating
sub-cellular signal distribution to cell shape) — a coarse version against
the existing rectangular ROI would technically run but wouldn't mean much
(radial rings of a rectangle, mass displacement from a rectangle's centre)
until real segmentation exists (§5).

### 1.6 Numerical Feature Extraction — Derived Features
User-defined combinations of other features via templates (A+B, A−B,
A×B, A÷B) or arbitrary formulas. One named example given: the optical
redox ratio, NAD(P)H / (NAD(P)H + FAD).

**Verdict — the redox ratio specifically is directly buildable now.**
FLIM Studio already loads multiple `FLIMDataset` objects (sample, IRF,
background) side by side; a second *sample* channel (e.g. a FAD
acquisition aligned to the NADH one) with its own `intensity` computed
the normal way gives a one-line ratiometric map from two existing
per-pixel intensity arrays — no new fitting, no segmentation, just a new
"second channel" load slot and one derived output map. The general
"build any formula from any two features" template engine is a much
bigger, more open-ended feature and only really pays off once there's a
feature *table* to run it against (§4) — not worth building in isolation.

### 1.7 Categorical Feature Extraction
Parses FOV filenames by a delimiter into categorical metadata columns
(e.g. `Panc1_Cyanide_dish1` → cell line / treatment / replicate),
attached to every cell record from that FOV.

**Verdict — cheap, useful for batch/queue exports specifically.** FLIM
Studio's queue/batch processing (`run_batch`, `process_queue_cached`)
already iterates many files; parsing configurable delimiter-separated
metadata out of each filename and folding it into the existing JSON
provenance export is a small, self-contained addition, independent of
everything else here.

---

## 2. Data Analysis — shared infrastructure

Feature-table-driven throughout: pick feature(s) via a side panel
(1 for univariate, 2 for bivariate, N for multivariate), apply
set-based (not sequential) categorical/numerical filters, encode
color/opacity/shape by categorical columns, hover to identify rows,
export a reproducible Python script. Accepts either its own Data
Extraction output or an arbitrary external CSV (needs: a unique row ID,
optional FOV ID, numeric columns auto-detected via pandas dtype with
≤1% non-numeric tolerance, zero-or-more categorical columns). Supports
saved/switchable "analysis profiles" as reusable configs.

**Verdict — this is the part that doesn't exist in FLIM Studio at all,
and everything below depends on it.** There is currently no concept of
"a table with one row per {thing}" anywhere in the app — every view is
either one whole-image map or one ROI's stats. Building §3's ideas
requires deciding what a *row* is for FLIM Studio specifically, since
there's no cell segmentation to make "one row per cell" the obvious
choice yet. The two realistic candidates, usable with zero new
architecture: **one row per queued/batch-processed file** (a "compare my
whole queue" table — mean/median τ, valid-pixel count, χ², etc. per
file, already all computed today) and **one row per saved point/ROI**
within a session (compare P1..P8 / multiple drawn ROIs against each
other). Both reuse numbers FLIM Studio already computes; neither needs
segmentation.

---

## 3. Data Analysis — the statistics/plots themselves

### 3.1 Univariate — Feature Comparison
Sina plot (points by kernel density) per group, optional box overlay
(median/mean/IQR/Tukey whiskers) and group-mean trend line; all pairwise
group comparisons via independent/Welch's t-test; Glass's Δ and |Cohen's
d| effect sizes; significance-asterisk annotations.

**Verdict — directly portable onto the "one row per file/ROI" table.**
Once such a table exists (§2), this is a fairly standard stats/plotting
job (`scipy.stats.ttest_ind` + a sina/box render) with no other
dependency. Good second-phase candidate after the table itself exists.

### 3.2 Univariate — Feature Histogram (+ GMM)
Per-group histograms with skewness/shape readout, plus a Gaussian
Mixture Model mode decomposing a distribution into components
(subpopulations), reporting an entropy-based "H-index" heterogeneity
score and gating pixels/cells by component.

**Verdict — directly usable *today*, no table needed at all.** FLIM
Studio's existing per-pixel Lifetime histogram (whole-image or
ROI-scoped, already rendered on the Overview/Histogram tabs) is itself
a distribution GMM decomposition can run against right now — fit a
`sklearn.mixture.GaussianMixture` to the histogram's underlying
`lifetime_ns` values, report component means/weights/H-index, and
optionally paint a new per-pixel "which GMM component" map. This is
probably the cheapest genuinely new capability on this whole list: it
needs no segmentation, no feature table, no second channel — just the
data already sitting in `AnalysisResult.lifetime_ns`.

### 3.3 Univariate — FOV Comparison
Per-FOV boxplots of one numeric feature, for spotting FOV-level outliers
across an experiment/session.

**Verdict — directly portable onto the "one row per file" table** (§2) —
essentially the same plot as 3.1 but grouped by file instead of by an
arbitrary category. Same dependency, same phase.

### 3.4 Bivariate — Feature Distribution
Scatter of two numeric features with marginal KDE/box/violin
distributions, optional log axes, 2D-GMM fitting with 95% confidence
ellipses, linear regression + R² per group, hard-assignment
GMM-cluster labelling exported as a new categorical column.

**Verdict — portable onto the table, one step past 3.1/3.3.** Same
prerequisite (§2); the 2D-GMM/regression pieces are a direct reuse of
whatever stats library backs 3.2's 1D GMM.

### 3.5 Bivariate — Phasor Analysis
1st/2nd-harmonic phasor cloud plotted against the universal semicircle
with lifetime reference markers; K-means clustering into K
subpopulations with convex hulls and centroid markers; exportable
cluster-ID column. Notably: **no manual gating** described — clustering
is automatic (K-means only), not user-drawn regions.

**Verdict — directly usable *today*, and probably the single
highest-value item overall.** FLIM Studio already renders exactly this
cloud per-pixel (`rendering.phasor_plot`, `processing.phasor_lifetimes`'s
G/S output) for the whole image or an ROI. Running K-means (or a 2D GMM,
which would also give soft boundaries/ellipses, arguably better than
Playground's own hard K-means) directly on the existing per-pixel (G,S)
array, drawing convex hulls on the existing Phasor tab, and emitting a
new per-pixel cluster-ID map is buildable with no new inputs at all —
same "already have the data" situation as 3.2, just phasor space instead
of the lifetime histogram. Real, common FLIM use case: separating
overlapping populations (e.g. free vs. protein-bound NAD(P)H) that
overlap in a scalar lifetime but separate in phasor space.

### 3.6 Multivariate — Dimension Reduction
PCA, UMAP, t-SNE over N standardized numeric features (from the table),
configurable hyperparameters (UMAP: n_neighbors/min_dist; t-SNE:
perplexity/early_exaggeration), coloured/shaped 2D projection output.

**Verdict — needs the table (§2) *and* genuinely many feature columns
to be worthwhile.** With only a handful of per-file/per-ROI scalar
metrics (τ_mean, χ², valid pixels, …), PCA/UMAP/t-SNE add little over
just looking at the numbers directly — this only starts paying off once
there's a richer feature set per row, which in practice means either
real per-cell features (§1.4/1.5, needing segmentation) or a much larger
scalar feature set than FLIM Studio computes today. Lowest-priority item
on this list for that reason, not because it's hard to wire up
technically (scikit-learn's PCA/UMAP/TSNE are drop-in).

### 3.7 Multivariate — Classification
Random Forest / Gradient Boosting / SVM / Logistic Regression (all
scikit-learn, fixed seed), z-score normalization for SVM/LogReg only,
classes from combinations of categorical columns, confusion matrix +
accuracy/precision/recall/specificity/Youden's index/F1 + ROC/AUC,
feature importance (tree methods only), class-imbalance handling
(sampling / class weights / threshold tuning).

**Verdict — same prerequisite and same caveat as 3.6.** Needs the table,
needs categorical labels to classify against (which for FLIM Studio would
mean the filename-parsed metadata of §1.7, or manually-assigned group
labels on saved ROIs/files), and needs enough rows to be statistically
meaningful. Realistic only as a later phase once §2's table and a
reasonable N of processed files/ROIs exist. Technically straightforward
(scikit-learn) once those exist.

---

## Priority summary (effort vs. value, given FLIM Studio's *current* architecture)

**No new architecture needed — pure additions to existing per-pixel output:**
1. Phasor-cloud clustering (K-means/GMM on existing G/S) — §3.5
2. Lifetime-histogram GMM decomposition — §3.2
3. 2nd-harmonic phasor (g₂/s₂) — §1.3
4. Intensity-weighted mean lifetime τ_m,iw — §1.2
5. Amplitude fractions as % — §1.2
6. Redox-ratio-style 2-channel derived map — §1.6 (needs a "second
   channel" load slot, but no fitting/segmentation change)
7. Filename-parsed categorical metadata for batch exports — §1.7

**Needs a new "one row per X" feature table (§2), but no segmentation:**
8. Row = queued/batch file, or row = saved ROI/point
9. On top of that table: univariate comparison stats (§3.1), FOV
   comparison (§3.3), bivariate scatter/GMM (§3.4)

**Needs real cell segmentation (biggest, separate undertaking):**
10. Morphology (§1.4), texture (§1.5), and the full per-cell pipeline
    these enable

**Needs the table *and* a rich multi-feature set to be worthwhile:**
11. PCA/UMAP/t-SNE (§3.6), classification (§3.7)

## Proposed integration shape (for when this is picked up)

A separate **"Analysis…"** window, following the existing `Toplevel`
dialog pattern already used for `TrainingDialog`
([flim_app/training_gui.py](flim_app/training_gui.py)) and
`LeicaExportDialog`
([flim_app/leica_export_gui.py](flim_app/leica_export_gui.py)) rather
than a new main-window tab — keeps the Overview/sidebar uncluttered and
matches how every other secondary workflow in this app is already
structured. Natural build order given the priority list above: start
with items 1–5 (they need nothing beyond what a single `AnalysisResult`
already contains, so they could even ship as new Overview-tab
views/toggles before the separate window exists at all), then stand up
the "one row per file/ROI" table (item 8) inside the new window once
there's more than one thing worth comparing, then layer 3.1/3.3/3.4 on
top of that table.
