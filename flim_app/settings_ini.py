from __future__ import annotations

"""Save/load :class:`AnalysisSettings` to/from a plain ``.ini`` file.

This is a *settings preset*, not a session snapshot: every field of
``AnalysisSettings`` round-trips (every sidebar control that feeds into an
actual analysis run -- method, timing, thresholds, spatial binning, native
fit conditions, display ranges, IRF source, ...). The queue is still
deliberately excluded (a preset is about repeatable *conditions*, not a
specific work-in-progress batch), but which decay/IRF/background files
were loaded *can* optionally round-trip too, via ``file_paths=``/
:func:`load_file_paths_ini` -- see those for why this is a separate,
optional section rather than folded into ``AnalysisSettings`` itself.

``gui.py`` is the only caller: ``_load_settings_ini_dialog``/
``_save_settings_ini_dialog``/``_save_settings_as_default`` for the
sidebar buttons, and a startup check for ``default.ini`` in the app's main
folder (see ``_paths.main_folder``) -- present, it's loaded automatically
so the whole sidebar starts from a user's own saved defaults instead of
this module's hardcoded ones; absent, startup is unaffected.
"""

import configparser
import dataclasses
from dataclasses import dataclass
from pathlib import Path

from .models import AnalysisSettings

# A distinctive section name (rather than plain "settings" or "DEFAULT")
# so a sensible error surfaces if this is ever pointed at an unrelated
# .ini file instead of silently misreading it.
SECTION = "flim_studio.analysis_settings"
# Separate from SECTION and always optional: AnalysisSettings itself
# stays a pure "conditions" object with no notion of which specific
# files were open (kept that way deliberately -- see this module's
# docstring, and e.g. AnalysisSettings.normalized()/ManualModelDialog,
# neither of which needs or wants to think about loaded files at all).
# A preset written without file_paths= simply has no section named this
# and load_file_paths_ini returns all-empty paths for it, same as an
# older preset saved before this feature existed.
FILES_SECTION = "flim_studio.loaded_files"

DEFAULT_INI_NAME = "default.ini"


@dataclass(slots=True)
class LoadedFilePaths:
    """Which decay/IRF/background files were open when a preset was
    saved, if any -- see save_settings_ini's file_paths argument and
    load_file_paths_ini. All-empty means "not recorded" (an older
    preset, or one saved with no dataset loaded at all), not "load
    nothing on purpose" -- gui.py's restore step already treats an empty
    path as "leave whatever's currently loaded alone" either way, so the
    distinction doesn't currently matter to any caller, but keep it in
    mind before changing that.
    """

    decay_path: str = ""
    irf_path: str = ""
    background_path: str = ""

# dataclasses.fields()' .type is the raw annotation *text* here (this
# module, like models.py, has `from __future__ import annotations`), e.g.
# "bool" or "tuple[int, ...]" -- not an evaluated type object. Matching on
# that text is simpler and just as reliable as resolving it via
# typing.get_type_hints() for the handful of shapes AnalysisSettings
# actually uses (str, bool, int, float, tuple[int, ...], tuple[float, ...]).
_FIELD_TYPES: dict[str, str] = {f.name: f.type for f in dataclasses.fields(AnalysisSettings)}


def _serialize(name: str, value: object) -> str:
    type_text = _FIELD_TYPES[name]
    if type_text.startswith("tuple["):
        return ",".join(str(item) for item in value)  # type: ignore[union-attr]
    if type_text == "bool":
        return "true" if value else "false"
    return str(value)


def _deserialize(name: str, raw: str) -> object:
    type_text = _FIELD_TYPES[name]
    raw = raw.strip()
    if type_text.startswith("tuple[int"):
        return tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if type_text.startswith("tuple[float"):
        return tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if type_text == "bool":
        return raw.lower() in {"1", "true", "yes", "on"}
    if type_text == "int":
        return int(raw)
    if type_text == "float":
        return float(raw)
    return raw  # str


def save_settings_ini(
    settings: AnalysisSettings,
    path: str | Path,
    *,
    file_paths: "LoadedFilePaths | None" = None,
) -> None:
    """Write every ``AnalysisSettings`` field to ``path`` as a plain .ini.

    Overwrites unconditionally -- callers (the "Save as…"/"Save as
    default" sidebar actions) are themselves the user's confirmation that
    they want this exact file written, matching how this app's other
    single-click save actions (e.g. "Clear queue") already work.

    ``file_paths``, when given, additionally records which decay/IRF/
    background files were loaded, in a separate ``[flim_studio.loaded_files]``
    section -- read back with :func:`load_file_paths_ini`. Omit it (the
    default) to write a file with no notion of loaded files at all, same
    as before this parameter existed.
    """

    # interpolation=None: a couple of these fields are free-text file paths
    # (native_dll_path, lifetime_lut_path, global_model_path), and
    # ConfigParser's default interpolation treats a bare "%" in a value as
    # the start of a substitution -- disabling it avoids that misreading a
    # path that happens to contain one.
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # field names are already lowercase; keep them exact regardless
    parser[SECTION] = {name: _serialize(name, getattr(settings, name)) for name in _FIELD_TYPES}
    if file_paths is not None:
        parser[FILES_SECTION] = {
            "decay_path": file_paths.decay_path,
            "irf_path": file_paths.irf_path,
            "background_path": file_paths.background_path,
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)


def load_settings_ini(path: str | Path) -> AnalysisSettings:
    """Read an ``AnalysisSettings`` back from an .ini written by
    :func:`save_settings_ini` (or a compatible hand-edited one).

    A field missing from the file (an older save from before that field
    existed, or one deliberately removed by hand) falls back to
    ``AnalysisSettings``'s own default for it rather than failing the
    whole load -- forward/backward compatible with this dataclass gaining
    or losing fields across versions. An unparsable *present* value raises
    ``ValueError`` naming the offending field, rather than silently
    substituting a default that might not be what the file's author
    intended.
    """

    path = Path(path)
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"Settings file not found: {path}")
    if SECTION not in parser:
        raise ValueError(
            f"{path} has no [{SECTION}] section -- this doesn't look like a FLIM Studio settings file."
        )
    section = parser[SECTION]
    values: dict[str, object] = {}
    for name in _FIELD_TYPES:
        if name not in section:
            continue
        try:
            values[name] = _deserialize(name, section[name])
        except ValueError as exc:
            raise ValueError(f"Invalid value for '{name}' in {path.name}: {section[name]!r} ({exc})") from exc
    return AnalysisSettings(**values)


def load_file_paths_ini(path: str | Path) -> LoadedFilePaths:
    """Reads the optional loaded-file-paths section :func:`save_settings_ini`
    writes when given ``file_paths=``.

    Returns all-empty paths (rather than raising) when that section is
    simply absent -- an older preset saved before this feature existed,
    or one saved with ``file_paths=None`` deliberately. Unlike
    :func:`load_settings_ini`, a missing *file* still raises
    (``FileNotFoundError``): callers already call ``load_settings_ini`` on
    the same path first and would have hit that error already, so this
    only reruns the same check for a caller using this function alone.
    """

    path = Path(path)
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"Settings file not found: {path}")
    if FILES_SECTION not in parser:
        return LoadedFilePaths()
    section = parser[FILES_SECTION]
    return LoadedFilePaths(
        decay_path=section.get("decay_path", "").strip(),
        irf_path=section.get("irf_path", "").strip(),
        background_path=section.get("background_path", "").strip(),
    )
