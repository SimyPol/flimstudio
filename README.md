# FLIM Studio

FLIM Studio is a Python desktop interface reconstructed from the supplied
`ImageProcessingwt_IRF_looping1-6c.vi` and its subVIs. It keeps the useful
workflow—decay-cube loading, background/IRF correction, lifetime and phasor
analysis, image composites, histograms, interactive region inspection, multi-file looping,
and export—while replacing the very wide LabVIEW front panel with a compact
tabbed application.

## Quick start on Windows

1. Install Python 3.10 or newer from <https://python.org>.
2. Open a terminal in this folder and run:

   ```powershell
   py -3 -m pip install -r requirements.txt
   py -3 run_flim_studio.py
   ```

   You can also double-click `run_flim_studio.bat` after installing the two
   dependencies. The application opens with synthetic cell-like FLIM data so
   every view can be explored before loading microscope data.

   To build a standalone `FLIM Studio.exe` that runs on a machine with no
   Python installed, see [BUILDING.md](BUILDING.md).

   For a full walkthrough of every control, panel, and mode of operation,
   see [USER_MANUAL.md](USER_MANUAL.md), or the same content as a Word
   document: [FLIM_Studio_User_Manual.docx](FLIM_Studio_User_Manual.docx).

## Supported data

- ICS v1 (`.ics` + `.ids`) and ICS v2 (`.ics`, embedded payload), including
  uncompressed and gzip-compressed payloads
- NumPy `.npy` and `.npz` cubes
- multi-page `.tif` / `.tiff`
- decay matrices in `.csv` / `.txt`
- Leica `.lif` raw acquisitions, via Leica Interrogator Desktop's own
  converter (not included in this repository) — see
  [Leica `.lif` integration](#leica-lif-integration) below

The internal cube order is always `(y, x, time)`. For NPZ input, the preferred
keys are `decay`, `data`, `cube`, or `arr_0`; optional scalars are `dt_ns` and
`rep_rate_mhz`.

The supplied microscope header example has `layout order bits t x y`, unsigned
16-bit samples, 32 time bins, an 80 MHz laser repetition rate, and a time extent
of 12.5 ns. The reader uses the image time extent to derive the bin width when
available, because some source headers contain inconsistent `parameter scale`
labels.

## Analysis controls

- every sidebar setting can be saved to / loaded from a plain `.ini`
  settings-preset file; **Save as default** writes `default.ini` next to
  the app, which loads automatically on every startup — share a `.ini`
  with a colleague, or drop it next to their copy of the exe, to hand off
  a whole acquisition setup at once instead of re-entering it by hand
- an intensity thresholding preview, usable *before* running analysis,
  not only after: switch the second map to `Intensity with threshold` as
  soon as a dataset loads to preview which pixels a given minimum-count
  threshold would omit, using raw counts spatially binned exactly as the
  current spatial-bin-radius setting (below) — the preview updates live
  as that radius changes
- scaled background-acquisition subtraction as the first processing step;
  automatic loop/time normalization (the supplied sample resolves to
  `20/800 = 0.025`), editable manual multiplier, and optional additional scalar
  counts per time bin
- timing offset and independently labelled phasor/fit count gates (the supplied
  mono-fit panel uses 1000 total processed counts), radius-labelled spatial count binning
  (`0 = 1x1`, `1 = 3x3`, `2 = 5x5`), smoothing, and
  optional automatic extreme-pixel interpolation
- spatial binning is a *rolling* window centred on every pixel, not
  block-binning: the processed image stays at the acquisition's full
  resolution at any radius (unlike the original VI's block reduction),
  with edge pixels correctly summing only their true spatial support.
  ROI/whole-image aggregate sums are computed independently of the
  per-pixel bin radius and stay accurate regardless of it
- an optional **synthetic Gaussian IRF**, for phasor correction and
  native reconvolution/global fitting alike, when no measured IRF is
  available or a loaded one doesn't match the current dataset's timing —
  a standard TCSPC approximation (Becker 2005; O'Connor & Phillips 1984)
  auto-parameterised from the sample decay's own 10–90% rise time, with
  a manual FWHM override; a dedicated **IRF & Decay** tab overlays the
  peak-normalised IRF against the peak-normalised decay to check it
- an explicit detector-row interpolation map: enter row IDs/ranges, load or
  save JSON/CSV/TXT maps, preview them against the source intensity image, and
  show the corrected rows as synchronized magenta overlays on both maps
- an independent zero-based detector-column list with donor-radius correction
  and synchronized purple overlays
- phase or modulation phasor lifetime, first-moment lifetime, or fast
  mono-exponential log-linear fit
- native FLIMLib iterative reconvolution with mono-, bi-, or tri-exponential
  models, shared or per-pixel IRFs, free or fixed counts/bin background, and
  all four weighting models used by the LabVIEW application
- the reconvolution fit (only — not the global/shared-τ methods, which
  can't be split this way without changing the answer) can run across
  multiple worker processes, splitting the image into row-band chunks each
  fitted concurrently in its own isolated subprocess for real wall-clock
  speedup on multi-core machines; identical results to a single process,
  since every pixel is already fitted fully independently
- automatic robust (1st–99th percentile) or manual lifetime display scaling;
  changes redraw the lifetime, composite, histogram and component maps live and
  never alter the fitted numerical lifetimes
- a dedicated **Multi-exponential** tab with paired tau1/A1 through tau3/A3
  maps for native bi/tri-exponential results; the main lifetime image remains
  the amplitude-weighted mean, and component maps are also exported as PNGs
- global fitting with shared lifetimes and local amplitudes/backgrounds
- per-pixel or averaged IRF phasor correction
- read-only acquisition TCSPC bin count/width/period, a zero-based half-open fit
  window, a lifetime display range, 50-bin Overview histogram, and independent
  1000-bin detailed histogram
- the exact original 256-colour `Reverse Rainbow (DM).lut`, with optional LUT
  import/inversion and display-only out-of-range removal
- representative whole-image decay and mono-exponential guide
- a four-panel Overview with intensity, a Lifetime/Composite/component/threshold-mask toggle,
  selected-point decay comparison, and an ROI/point-scoped phasor panel
- a Phasor/Histogram selector on the Overview, retaining both compact views
  alongside their larger dedicated tabs
- a log-density phasor heatmap with the universal semicircle, ROI scoping, and
  colour-matched selected-point markers
- an on-the-fly **Intensity / Intensity with threshold** choice: omitted pixels
  are painted red without rerunning analysis. Native previews sum the same leading
  input bins as FLIMLib (29 in the supplied preset), while phasor previews use all
  processed acquisition bins. **Use for fit** applies the value to the active
  estimator family for the next analysis run
- click-to-add comparison points with up to eight persistent, colour-matched
  map markers, active-method lifetimes, and decay curves
- synchronized rectangular ROI selection on the intensity and active secondary map
- summed ROI decay on the Overview page, with pixel-lifetime mean, standard
  deviation, median, IQR, valid-pixel count, and aggregate mono-exponential guide
- batch processing for all files in the queue

### Detector-row interpolation map

The original LabVIEW front panel stored a one-dimensional set of detector-row
IDs rather than a 2-D mask. Its operational preset contained 19 zero-based IDs:
`437, 392, 349, 339, 317, 288, 241, 237, 230, 215, 195, 132, 99, 89, 78, 69,
54, 30, 3`. For that 512-row acquisition the VI mirrored them before indexing,
using `image_y = 511 - detector_row`. Choose **VI preset** to load this exact
configuration; it remains disabled until explicitly enabled and is rejected,
not clipped, if the current detector height is incompatible.
The preset remains locked to exactly 512 source rows. Editing any row, radius,
or coordinate convention converts it to a custom height-relative map and clears
the preset/file provenance so stale calibration metadata cannot be exported.

Custom maps can use either direct zero-based image-Y rows or the original
mirrored LabVIEW detector-row convention. The Python correction replaces each
complete `(x,time)` row after scaled background subtraction and before smoothing,
spatial binning, phasor analysis, or fitting. Consecutive bad rows are reconstructed
simultaneously from unchanged neighbouring rows, with one-sided edge handling.
This deliberately repairs limitations in the saved subVI: its six-sample
stencil was asymmetric/order-dependent, fixed to 32 time bins, and its visible
"steps" control was disconnected from the live mean. Python calls the active
control **donor radius** and records the algorithm, entered IDs, resolved image
rows, donors, reference correction, coverage, and preprocessing order in JSON
and NPZ exports. Interpolated traces are synthetic estimates rather than
independent Poisson photon observations; treat native Poisson fit statistics
accordingly.

The phasor implementation follows the bin-centred equations embedded in the
source VI, including complex IRF division and its half-bin correction. The
lifetime image uses the byte-exact LabVIEW Reverse Rainbow table over the saved
0.5–4.5 ns display range. The composite follows the original separate HSV path:
lifetime maps to hue 0–0.7, saturation is 1, and intensity is linearly auto-scaled
to its actual minimum and maximum. The LUT/display limits never alter numerical
lifetime values or scientific validity.

### Which “bins” setting is which?

The original front panel uses the word *bins* for four independent quantities:

| Control | Supplied/saved value | Meaning |
|---|---:|---|
| Acquisition TCSPC bins | 32 | Samples in every pixel decay (0.390625 ns/bin; 12.5 ns total) |
| Native fit input bins | 29 | Leading decay bins sent to the FLIMLib fit; it does not change the 32-bin acquisition |
| Overview lifetime intervals | 50 | Compact lifetime histogram resolution |
| Detailed lifetime bins | 1000 | Detailed/export lifetime histogram resolution |

The histogram settings do not crop the decay and are unrelated to the 256-entry
colour LUT. The GUI shows the acquisition value read-only and labels the other
three controls explicitly. `-1` in the native input field uses all acquisition
bins instead of the saved 29-bin LabVIEW preset.

Native reconvolution also has its own explicit bin-width control. The supplied
LabVIEW mono-fit panel used 0.4 ns/bin even though the ICS acquisition header
encodes 0.390625 ns/bin. The original native-fit match therefore defaults to
0.4 ns/bin, while phasor and acquisition-time displays retain the precise ICS
value. Before native reconvolution, the IRF prompt has its per-pixel temporal
median pedestal removed and is clipped to non-negative values; this matches the
prepared-prompt behaviour required by the supplied IRF and is recorded in
result metadata. `None` and tail-median alternatives remain available.

The older LabVIEW mono-fit controls also saved a peak/start value of 5, an end
index of 24, and an IRF-range value of 7. FLIM Studio exposes the zero-based,
half-open fit start/stop fields and shows the 5:24 preset beside them, but keeps
automatic timing as the safe default because the peak can move between
acquisitions. The file-derived 0.390625 ns/bin is retained instead of rounding
it to the old front-panel display value of 0.4 ns.

## Exported files

Each result exports:

- numeric maps and decay curves in compressed NPZ
- JSON settings, metadata, and summary statistics
- intensity, lifetime, composite, phasor, histogram, decay-fit, and dashboard PNGs
- lifetime histogram CSV

When a rectangular ROI is active, **Export current result** also writes a
coordinate-tagged ROI NPZ, JSON summary, decay CSV, and decay PNG, so exporting
different regions does not overwrite earlier ones. ROI coordinates use zero-based
`(x0, y0, x1, y1)` bounds with exclusive upper limits and are reported both on
the processed map and, when spatial binning is active, in approximate source
pixels. The ROI trace sums every selected processed pixel; invalid per-pixel
lifetime estimates remain in the photon trace but are excluded from the
separately reported map-lifetime statistics.

Point and ROI selection are independent: click either map to add P1–P8, or drag
to define the histogram/ROI scope. **Undo** and **Clear points** affect only the
comparison points; **Clear ROI** restores the whole-image histogram/phasor scope
without removing the point curves. Point-set exports include JSON metadata, NPZ arrays,
CSV decay curves, and a comparison PNG with stable point colours and processed
plus nominal source coordinates. For native bi/tri-exponential results, the
point export also retains component lifetimes, amplitudes, background, fit
quality/status, native fit-window diagnostics, and shared global lifetimes.
Point exports additionally retain exact analysed G/S coordinates and a scoped
phasor PNG. ROI exports include the cropped G/S and phase/modulation lifetime
maps plus their ROI phasor PNG.

Dragging an ROI highlights that region's phasor cloud against muted whole-image
context. Selected P1–P8 remain independent overlays, using the same colours as
their decay curves even when a point lies outside the active ROI.

## Command line

The same pipeline can run without the GUI:

```powershell
py -3 -m flim_app.cli --demo --output demo_results
py -3 -m flim_app.cli sample.ics --irf irf.ics --method phasor_phase --output results
py -3 -m flim_app.cli sample.ics --background background.ics --background-scale-mode auto --output results
```

Run `py -3 -m flim_app.cli --help` for all options.

## Native fitting backend

The later-supplied `FLIMFitTool` contained the missing wrapper source, 64-bit
DLL, MATLAB reference implementation, and example data. This package now ships
a rebuilt `native/flimfit_labview.dll` and exposes only the length-checked v2
entry points. The rebuild corrects two source defects found during audit: a
null sigma pointer in constant weighting and an inclusive/exclusive fit-window
off-by-one. It also fixes successful training status reporting and rejects
underdetermined independent fit windows.

Native fitting is opt-in: select **FLIMLib — reconvolution** or **FLIMLib —
global shared τ**, then load an IRF. The portable phasor, moment, and fast
mono-exponential paths do not load the DLL. Native calls are serialized because
the linked FLIMLib code contains shared state — but only *within one loaded
DLL instance*; the reconvolution method (only, not the global/shared-τ
methods) can still run across multiple worker processes, each with its own
independent DLL load and therefore nothing shared to serialize against
(see [Analysis controls](#analysis-controls) above). Fit components, amplitudes,
background, reduced chi-square, status, photons, time shift, fit window, exit
flags, and global statistics are included in the exported NPZ/JSON output.

The rebuilt DLL SHA-256 is
`067f7e338d780cdd1e078de6544a15791ec0321159d6cc6d9c6f02eab4327ffb`.
Its wrapper source, exact linked FLIMLib source subset, and GPL text are under
`native/` for reproducibility and license compliance. The Python-only portions
remain MIT licensed; the native FLIMLib component is GPL-3.0-or-later. Consult
your organisation's licensing process before redistribution.

## Leica `.lif` integration

FLIM Studio doesn't decode Leica's raw `.lif` format itself — it calls a
`leica_flim_converter` package to scan and export `.lif` acquisitions to
`.ics`, run as an isolated subprocess (the same crash-isolation approach
used for native FLIMLib fitting, so a decode failure on an unusual file
can't take down the main GUI).

**This repository does not include the converter.** The vendored core copy
that packaged builds can carry (`flim_app/leica_bundled/`) is deliberately
left out, so `.lif` scanning/export needs the sidebar's **Leica
Interrogator…** control pointed at a separately installed **Leica
Interrogator Desktop** folder (which also enables Interrogator's own
**Full GUI…** for manual browsing/preview). With neither a configured
folder nor a vendored copy, `.lif` files can't be opened; every other
input format is unaffected. A build made from a checkout that does contain
`flim_app/leica_bundled/` needs no separate install and uses it whenever no
external folder is set.

Just queue a `.lif` file the same way as any other decay file. Clicking it
scans the file (a single `.lif` commonly contains many datasets — multiple
series, timepoints, channels/detectors), opens a picker to choose which
ones to export, and adds the resulting `.ics` files straight to the queue
automatically — no separate GUI window or manual re-import needed for the
common case. A polling **Watch for exports…** folder option is also
available, for picking up `.ics` files exported by Interrogator's own full
application.

## Scientific validation

Synthetic tests verify shape, masking, defined detector-row reconstruction and
map persistence/coverage, phasor lifetime recovery, IRF correction,
mono-exponential fitting, ICS v1/v2 reading, spatial binning, export, native
reconvolution under all four noise models, shared/per-pixel IRF equivalence,
and global fitting. Before
using results for publication or clinical decisions, validate the pipeline with
your laboratory's lifetime standards and compare representative microscope
files against the original LabVIEW application.

The supplied `FLIMFitTool` fixtures are also checked directly: its 2.4 ns
single-decay example fits within 0.08 ns, and both 18×24 reference images
(shared and per-pixel IRFs) complete with every pixel successful and mean
absolute lifetime errors below 0.11 ns against their stored truth maps.

## Provenance

The included Instant-FLIM VIs originate from the Apache-2.0 licensed
[Instant-FLIM-Control](https://github.com/yzhang34/Instant-FLIM-Control/tree/885b82dfcb84b1b41756640395fe04a3ea2d0da5)
project. This implementation is newly written Python code based on the supplied
interface/metadata and published algorithms. The optional native folder contains
the user's wrapper plus the exact GPL-licensed FLIMLib source used to rebuild it.
