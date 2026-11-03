from __future__ import annotations

"""Headless helper that runs, as a separate process, in whichever Python
environment holds a usable ``leica_flim_converter`` package -- keeping .lif
decoding entirely out of FLIM Studio's own process either way. There are
two such environments (see ``leica_export_gui.resolve_leica_backend``):

- A separately installed "Leica Interrogator Desktop" folder's own venv,
  when the user has configured one (sidebar: "Leica Interrogator..."). Its
  own venv sidesteps any dependency mismatch, since that's the exact
  environment its decoder is built/tested against.
- The copy vendored inside FLIM Studio itself (``flim_app/leica_bundled/``,
  core-only, no PySide6 -- see its ``README_VENDORED.md``), used when no
  external folder is configured. There, this script runs under FLIM
  Studio's *own* interpreter -- ``sys.executable`` directly when running
  from source, or this same frozen exe relaunched with a hidden
  ``--leica-worker`` flag when packaged (see run_flim_studio.py), since a
  packaged build has no standalone copy of this file on disk to invoke.

Either way this stays a separate OS process rather than running in-process,
mirroring ``native_worker.py``'s reasoning for the FLIMLib fitting DLL:
the decoder does low-level mmap/struct binary parsing and an optional
native (compiled) fast path, either of which could misbehave on an
unusual or corrupt .lif -- running it as a separate process means that
stays contained instead of taking the whole GUI down with it.

Usage (either invoked directly as a script, or via the --leica-worker
self-relaunch above -- both end up calling main() with the same argv):
    python leica_worker.py <interrogator_or_bundled_root> scan   <lif_path>
    python leica_worker.py <interrogator_or_bundled_root> export <lif_path> <output_dir> <keys_json_or_"all"> [bins]

The root folder (containing the leica_flim_converter package -- either an
external Interrogator install or FLIM Studio's own bundled copy) is passed
explicitly and prepended to sys.path here, rather than relied on implicitly
via the subprocess's cwd: plain ``python script.py`` invocations only add
*that script's own* directory to sys.path[0], not the current working
directory (only ``python -m package`` does that) -- so without this,
importing leica_flim_converter would fail even with cwd set correctly by
the caller.

Prints exactly one JSON object to stdout -- the caller must not depend on
anything else being written there. A non-zero exit code pairs with a
``{"error": "..."}`` object rather than a bare traceback, wherever the
failure is anticipated (import failures, bad arguments, decode errors);
anything unanticipated will still surface as a Python traceback on
stderr for the caller to show verbatim.
"""

import json
import sys
from pathlib import Path


def _json_default(value):
    # numpy scalar types (e.g. np.uint64 photon counts) aren't natively
    # JSON-serialisable; .item() converts them to the equivalent plain
    # Python type. Anything else unexpected just becomes its str().
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def _emit(payload: dict) -> None:
    # flush=True matters here: stdout piped to a subprocess is normally
    # block-buffered (not line-buffered like a terminal), so without it
    # every progress line would sit in a buffer and only actually reach
    # the caller in one lump at exit -- indistinguishable from a hang for
    # however long the export takes. (Also launched with `-u` for the
    # same reason, in case anything else in the import chain prints.)
    print(json.dumps(payload, default=_json_default), flush=True)


def _dataset_summary(dataset) -> dict:
    return {
        "key": dataset.key,
        "display_name": dataset.display_name,
        "width": dataset.width,
        "height": dataset.height,
        "plane_count": dataset.plane_count,
        "channel_name": dataset.channel_name,
        "detector_name": dataset.detector_name,
        "acquisition_name": dataset.acquisition_name,
        "series_name": dataset.series_name,
        "timepoint_name": dataset.timepoint_name,
        "z_name": dataset.z_name,
        "total_photons": dataset.total_photons,
        "dimension_status": dataset.dimension_status,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        _emit({"error": "usage: leica_worker.py <interrogator_root> <scan|export> <lif_path> [...]"})
        return 2
    interrogator_root, mode, lif_path = argv[0], argv[1], argv[2]
    # Drop only interrogator_root, so argv[0] is mode again -- everything
    # below (including the export branch's argv[2:]) was written assuming
    # the pre-interrogator_root indexing (argv[0]=mode, argv[1]=lif_path,
    # argv[2]=output_dir, ...). Dropping 2 elements here instead of 1 was
    # the actual bug: it silently shifted every later argument one slot
    # left (output_dir became the keys JSON, the keys argument became the
    # bins string, etc.), which is exactly why `set(json.loads(keys_arg))`
    # blew up trying to iterate a plain int.
    argv = argv[1:]
    sys.path.insert(0, str(Path(interrogator_root)))

    try:
        from leica_flim_converter import scan_lif
    except Exception as exc:
        _emit({"error": f"Could not import leica_flim_converter: {exc}"})
        return 1

    try:
        datasets = scan_lif(lif_path)
    except Exception as exc:
        _emit({"error": f"Could not scan {lif_path}: {exc}"})
        return 1

    if mode == "scan":
        _emit({"datasets": [_dataset_summary(dataset) for dataset in datasets]})
        return 0

    if mode == "export":
        if len(argv) < 3:
            _emit({"error": "export requires an output directory"})
            return 2
        output_dir = argv[2]
        keys_arg = argv[3] if len(argv) > 3 else "all"
        bins = int(argv[4]) if len(argv) > 4 else 128
        gzip_level = int(argv[5]) if len(argv) > 5 else None
        try:
            from leica_flim_converter import IcsExportOptions, export_ics_blocks
            # Not re-exported from the package's own __init__.py -- only
            # from its .ics submodule directly.
            from leica_flim_converter.ics import DEFAULT_GZIP_LEVEL
        except Exception as exc:
            _emit({"error": f"Could not import export functions: {exc}"})
            return 1
        if keys_arg == "all":
            selected = datasets
        else:
            try:
                parsed_keys = json.loads(keys_arg)
                if not isinstance(parsed_keys, list) or not all(isinstance(item, str) for item in parsed_keys):
                    raise TypeError(f"expected a JSON array of strings, got {parsed_keys!r}")
                wanted = set(parsed_keys)
            except (json.JSONDecodeError, TypeError) as exc:
                _emit({"error": f"Invalid dataset key selection: {exc}"})
                return 2
            selected = [dataset for dataset in datasets if dataset.key in wanted]
        if not selected:
            _emit({"error": "No matching datasets selected for export"})
            return 2

        def report(message: str, completed: float, total: float) -> None:
            # One JSON line per update -- this is what actually lets the
            # GUI show real progress instead of a static "Exporting..."
            # label for however long a large multi-plane series takes.
            _emit({"progress": {"message": message, "completed": completed, "total": total}})

        try:
            results = export_ics_blocks(
                selected, output_dir,
                IcsExportOptions(bins=bins, gzip_level=gzip_level if gzip_level else DEFAULT_GZIP_LEVEL),
                progress=report,
            )
        except Exception as exc:
            _emit({"error": f"Export failed: {exc}"})
            return 1
        _emit({"results": results})
        return 0

    _emit({"error": f"Unknown mode {mode!r}"})
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
