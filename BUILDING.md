# Building a standalone FLIM Studio.exe

This packages FLIM Studio into a folder that runs on a machine with no
Python installed. It's built with [PyInstaller](https://pyinstaller.org/).

## Prerequisites

- A working Python 3.10+ install (the one you've been running
  `run_flim_studio.bat` with is fine).
- The project's own dependencies, plus PyInstaller:

```bash
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

## Build

```bash
build_exe.bat
```

or directly:

```bash
pyinstaller --noconfirm flim_studio.spec
```

Output: `dist\FLIM Studio\`. That whole folder is the app -- the DLL, LUT
file, and the bundled Leica `.lif` reader (see below) get bundled inside
it automatically (see `flim_studio.spec`). **Distribute the whole folder**
(zip it up), not just the `.exe` inside it; it won't run on its own
without the rest of the folder alongside it.

Run it via `dist\FLIM Studio\FLIM Studio.exe`.

## Why this needed code changes, not just running PyInstaller

Two things about this app don't survive freezing unmodified, both already
fixed in the source (see `flim_app/_paths.py`, `flim_app/native_backend.py`,
`run_flim_studio.py`):

1. **Every native fit runs in a subprocess** (crash isolation from a DLL
   access violation -- see `native_backend.py`). From source, that
   subprocess is `python native_worker.py ...`. Frozen, there's no
   standalone `native_worker.py` file on disk to run -- `sys.executable`
   *is* the frozen exe itself, so it now relaunches itself with a hidden
   `--native-worker` flag instead, which `run_flim_studio.py` intercepts
   before doing anything else (no GUI import, no window).
2. **Bundled data files** (the DLL, the default LUT) can't be found via
   `Path(__file__).resolve().parent` once frozen, because source files get
   compiled into an archive rather than existing as loose files at that
   path. `flim_app/_paths.py` uses PyInstaller's own `sys._MEIPASS`
   mechanism instead, which is set consistently for this purpose.

If you add other bundled data files later, add them to `datas` in
`flim_studio.spec` under `flim_app/...` to match what `app_root()` in
`_paths.py` expects.

## The bundled Leica `.lif` reader

> **Not part of this repository.** `flim_app/leica_bundled/` is git-ignored
> and not published here, so an exe built from a plain checkout has no
> built-in `.lif` reader; `.lif` support then needs a separately installed
> Leica Interrogator Desktop folder set in the sidebar. The rest of this
> section describes a build from a local tree that does contain that folder.

The packaged exe includes a vendored, core-only copy of Leica
Interrogator's `leica_flim_converter` (scan + export to `.ics`, no
PySide6/GUI) at `flim_app/leica_bundled/` -- see that folder's
`README_VENDORED.md`. This means `.lif` files can be scanned and exported
straight from the exe with **no separate Interrogator install needed**;
setting a "Leica Interrogator…" folder in the sidebar is only required for
the "Full GUI…" button (Interrogator's own full preview/export window),
which stays a genuinely separate, externally-installed program.

Like the native-fit DLL worker, this reader also runs as an isolated
subprocess rather than inside FLIM Studio's own process (crash isolation
from the low-level `.lif` parsing and its native decoder DLL). Frozen,
there's no standalone `leica_worker.py` file for *this exe's own bundled
copy* to run under a plain `python.exe` -- it self-relaunches with a hidden
`--leica-worker` flag instead, exactly like `--native-worker` above (see
`run_flim_studio.py`). An *externally* configured Interrogator install
still runs `leica_worker.py` as a real loose `.py` file under that
install's own venv `python.exe`, same as always -- which is also why
`flim_studio.spec` ships `flim_app/leica_worker.py` itself as a loose data
file rather than only leaving it compiled into the app's own archive: an
external interpreter needs a real file path to point at.

If Leica Interrogator's own converter changes, re-sync
`flim_app/leica_bundled/` by hand -- see that folder's `README_VENDORED.md`
for exactly which files matter and why.

## Troubleshooting

**"The FLIMLib fitting DLL was not found."** -- the build didn't bundle
`native/flimfit_labview.dll` correctly. Confirm it exists at
`dist\FLIM Studio\_internal\flim_app\native\flimfit_labview.dll` (exact
subfolder depends on your PyInstaller version) and that
`native/flimfit_labview.dll` exists in the source tree before building.

**A cryptic `OSError`/`WinError` when a DLL loads** -- both native DLLs
this app uses (`flimfit_labview.dll` for fitting, and the bundled Leica
reader's `LeicaFlimDecoder.dll`) link the MSVC runtime dynamically, so the
machine running the exe needs the
[Microsoft Visual C++ Redistributable (x64)](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)
installed. Most Windows 10/11 machines already have it; if not, it's a
small, standard Microsoft download.

**Antivirus/EDR flags the exe or its native-fit relaunches** -- unsigned,
freshly-built executables are common false-positive bait, and this app
already relaunches itself once per native fit (see the AV/EDR discussion
earlier in this project's history) -- that's now a second executable
launching itself repeatedly, which can look suspicious to some scanners.
An exclusion for the app's install folder is the standard fix, same as was
needed when running from source.

**"Leica .lif reader not available"** -- the build's `flim_app/leica_bundled/`
folder didn't get bundled correctly (or was deleted from the source tree
before building). Confirm `dist\FLIM Studio\_internal\flim_app\leica_bundled\leica_flim_converter\__init__.py`
exists (exact subfolder depends on your PyInstaller version), and that
`flim_app/leica_bundled/` exists in the source tree before building --
either re-vendor it per its `README_VENDORED.md`, or configure an external
"Leica Interrogator Desktop" folder in the sidebar as a workaround.

**Something else fails to import** -- rerun with a console build (the
default; `console=True` in the spec) so you can see the traceback, and
share it -- PyInstaller occasionally needs an explicit `--collect-all` for
a package's data files (Pillow's image-format plugins are a common one)
that its automatic dependency scan misses.

## Status

Confirmed working end to end (2026-08-18): built via `build_exe.bat`
from an Anaconda Python environment, launched, and ran a full analysis
on the bundled demo data — native fitting, the phasor plot, lifetime
histogram, multi-exponential tab, and metadata provenance all rendered
correctly from the frozen exe.

One environment-specific snag along the way, not a bug in the app
itself: `build_exe.bat`'s original interpreter auto-detection could
pick a different Python than the one PyInstaller was actually installed
into on machines with more than one Python on PATH (e.g. Anaconda's
`python` plus a separate `py` launcher). Fixed by testing each
candidate interpreter for a working PyInstaller import instead of just
checking whether it exists on PATH.
