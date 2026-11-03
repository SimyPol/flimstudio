from __future__ import annotations

"""Private subprocess entry point for isolated native FLIM fitting."""

import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flim_app.native_backend import FLIMLibLabViewBackend
else:
    from .native_backend import FLIMLibLabViewBackend


def main(argv: list[str] | None = None) -> int:
    process_start = time.perf_counter()
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4:
        print("Usage: native_worker.py MODE DLL INPUT_NPZ OUTPUT_NPZ", file=sys.stderr)
        return 2
    mode, dll_path, input_name, output_name = args
    input_path, output_path = Path(input_name), Path(output_name)
    try:
        load_start = time.perf_counter()
        with np.load(input_path, allow_pickle=False) as archive:
            decay = np.asarray(archive["decay"])
            irf = np.asarray(archive["irf"])
            mask = np.asarray(archive["mask"]) if "mask" in archive else None
            fixed_taus = np.asarray(archive["fixed_taus"]) if "fixed_taus" in archive else None
            options = json.loads(str(archive["options_json"].item()))
        load_elapsed = time.perf_counter() - load_start

        dll_load_start = time.perf_counter()
        backend = FLIMLibLabViewBackend(dll_path)
        dll_load_elapsed = time.perf_counter() - dll_load_start

        fit_start = time.perf_counter()
        if mode == "pixels":
            result = backend.fit_pixels(decay, irf, **options)
        elif mode == "global":
            if "initial_taus_ns" in options:
                options["initial_taus_ns"] = tuple(options["initial_taus_ns"])
            result = backend.fit_global(decay, irf, mask=mask, **options)
        elif mode == "train_global":
            if "initial_taus_ns" in options:
                options["initial_taus_ns"] = tuple(options["initial_taus_ns"])
            result = backend.train_global_model(decay, irf, mask=mask, **options)
        elif mode == "apply_global":
            if fixed_taus is None:
                raise ValueError("apply_global requires a fixed_taus array in the input payload.")
            result = backend.apply_global_model(decay, irf, fixed_taus, mask=mask, **options)
        else:
            raise ValueError(f"Unknown fit mode: {mode}")
        fit_elapsed = time.perf_counter() - fit_start

        save_start = time.perf_counter()
        arrays = result.diagnostics()
        arrays["metadata_json"] = np.array(
            json.dumps(
                {
                    "dll_sha256": result.dll_sha256,
                    "wrapper_code": result.wrapper_code,
                    "fit_metadata": result.metadata,
                }
            )
        )
        # Uncompressed: this file is read back by the parent process within
        # milliseconds and then discarded, so zlib's CPU cost is pure loss.
        np.savez(output_path, **arrays)
        save_elapsed = time.perf_counter() - save_start

        # Printed on success so the parent can show where native-fit
        # wall-clock time goes: reading the transfer file, loading the DLL,
        # the actual DLL fit call, and writing results back. Compare this
        # worker_total_s against the parent's own subprocess_wall_s to see
        # interpreter-startup/process-launch overhead outside of all this.
        total_elapsed = time.perf_counter() - process_start
        # irf_groups=<groups>/<selected pixels>: how many distinct IRF
        # groups the fast path (see build_irf_groups in
        # flimfit_labview.c) collapsed the selected pixels into, for
        # "global"/"train_global" modes only (global_stats[9]/[4]).
        # irf_groups==selected means grouping found no sharing at all --
        # for irfMode=1 (per-pixel IRF) that usually means the per-pixel
        # IRF rows meant to share a detector's IRF aren't byte-identical
        # (e.g. float rounding from upstream processing), which defeats
        # the speedup entirely; irf_groups==1 is the best case (one
        # shared IRF, or every per-pixel row byte-identical).
        irf_groups_note = ""
        global_stats = getattr(result, "global_stats", None)
        if global_stats is not None and len(global_stats) > 9:
            irf_groups_note = (
                f" irf_groups={int(global_stats[9])}/{int(global_stats[4])}"
            )
        print(
            "NATIVE_WORKER_TIMING "
            f"mode={mode} decay_shape={tuple(decay.shape)} "
            f"load_input_s={load_elapsed:.3f} dll_load_s={dll_load_elapsed:.3f} "
            f"fit_call_s={fit_elapsed:.3f} save_output_s={save_elapsed:.3f} "
            f"worker_total_s={total_elapsed:.3f}{irf_groups_note}",
            file=sys.stderr,
        )
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
