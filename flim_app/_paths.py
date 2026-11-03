from __future__ import annotations

"""Resolve paths to this package's bundled resources (DLL, LUT), whether
running from source or from a frozen (PyInstaller) build.

Source files get compiled into an archive when frozen rather than existing
as loose files on disk, so ``Path(__file__)``-based paths are unreliable
for locating bundled *data* (non-code) files there. PyInstaller's own
documented mechanism is ``sys._MEIPASS``, set consistently for both
``--onedir`` and ``--onefile`` builds; this module is the single place
that distinction is handled so the rest of the app doesn't need to know
whether it's frozen.
"""

import sys
from pathlib import Path


def app_root() -> Path:
    """The directory that mirrors this package's own layout (``flim_app/``).

    From source, this is simply where this file lives. Frozen, it is
    wherever PyInstaller extracted/placed the bundle's data files --
    which, by convention in this project's ``.spec`` file, is laid out
    the same way (a ``flim_app/`` directory containing ``native/`` and
    ``assets/``), so callers can use the same relative paths either way.
    """

    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidate = Path(frozen_root) / "flim_app"
        if candidate.is_dir():
            return candidate
        return Path(frozen_root)
    return Path(__file__).resolve().parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def main_folder() -> Path:
    """The folder a user would call "the app's main folder" -- next to
    ``FLIM Studio.exe`` when frozen, or the project root (the one holding
    ``run_flim_studio.py``/``run_flim_studio.bat``, one level up from this
    ``flim_app/`` package) when running from source.

    Distinct from :func:`app_root`, which is FLIM Studio's own *internal*
    resource folder (mirroring this package's own layout -- the DLL, LUT,
    bundled Leica reader) and, frozen, lives inside PyInstaller's
    ``_internal``-style subfolder rather than next to the exe. This is for
    the opposite case: something a user is meant to find and edit by hand
    next to the executable itself -- currently just ``default.ini`` (see
    ``settings_ini.py``).
    """

    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def bundled_leica_root() -> Path | None:
    """Directory holding the vendored copy of Leica Interrogator's
    ``leica_flim_converter`` package (see ``flim_app/leica_bundled/`` and
    its ``README_VENDORED.md``), if this build includes it.

    ``None`` when the vendored copy isn't present -- e.g. an older build,
    or a source checkout where that folder was deliberately removed. Callers
    (``leica_export_gui.resolve_leica_backend``) fall back to requiring an
    externally configured Interrogator install in that case, same as before
    this bundled copy existed at all.
    """

    candidate = app_root() / "leica_bundled"
    if (candidate / "leica_flim_converter" / "__init__.py").is_file():
        return candidate
    return None
