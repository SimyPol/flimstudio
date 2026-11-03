# PyInstaller build spec for FLIM Studio.
#
# Build with:
#     pyinstaller --noconfirm flim_studio.spec
#
# Output lands in dist/FLIM Studio/ -- that whole folder is the standalone
# app; distribute it as a folder (zip it up), not just the .exe inside it.
# See BUILDING.md for prerequisites and troubleshooting.
#
# --onedir (via COLLECT below), not --onefile: this app loads a native DLL
# via ctypes and relaunches itself as a subprocess for every native fit
# (see native_backend.py) -- onedir avoids onefile's per-run temp-extraction
# step, which would otherwise re-extract the DLL and interpreter on every
# single launch and every native fit subprocess.

from pathlib import Path

project_root = Path(SPECPATH)

datas = [
    (str(project_root / "native" / "flimfit_labview.dll"), "flim_app/native"),
    (str(project_root / "flim_app" / "assets" / "Reverse Rainbow (DM).lut"), "flim_app/assets"),
    # Shipped as a loose .py file (not compiled into flim_app's own PYZ
    # archive), so it stays a real, runnable file on disk for an *external*
    # Interrogator venv's python.exe to invoke directly -- see
    # leica_export_gui.py's WORKER_SCRIPT / app_root() comment. The bundled
    # copy below only ever runs it via this same exe's own --leica-worker
    # self-relaunch, which imports it as a normal (frozen) module instead
    # and doesn't need this loose copy at all -- it's kept regardless since
    # both paths share one spec entry and an external install is always
    # still supported.
    (str(project_root / "flim_app" / "leica_worker.py"), "flim_app"),
]

# Vendored core of Leica Interrogator's leica_flim_converter (see
# flim_app/leica_bundled/README_VENDORED.md) -- lets a packaged build read
# .lif files with no separate Interrogator install. Walked file-by-file
# (rather than one directory-level `datas` tuple) so the native decoder DLL
# and every source file land at the same relative path under
# flim_app/leica_bundled/ as they have on disk, which is what
# leica_flim_converter/native_decoder.py's own __file__-relative DLL lookup
# and lif_flim_forensics.py's sibling-of-the-package import both need.
_leica_bundled_dir = project_root / "flim_app" / "leica_bundled"
if _leica_bundled_dir.is_dir():
    for _f in _leica_bundled_dir.rglob("*"):
        if _f.is_file():
            _rel_parent = _f.relative_to(project_root / "flim_app").parent.as_posix()
            datas.append((str(_f), f"flim_app/{_rel_parent}"))

a = Analysis(
    ["run_flim_studio.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FLIM Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Console left on for the first build: it lets you see Python
    # tracebacks (and the [flim-studio] native-fit timing lines from
    # earlier work) if anything goes wrong. Switch to console=False once
    # you've confirmed a real analysis run works end to end.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FLIM Studio",
)
