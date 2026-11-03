from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.native_backend import FLIMLibLabViewBackend


def synthetic_cube(tau_ns: float, rows: int = 4, columns: int = 4, bins: int = 64, dt_ns: float = 0.125):
    time = np.arange(bins, dtype=np.float64) * dt_ns
    irf = np.exp(-0.5 * ((np.arange(bins) - 3.0) / 0.85) ** 2)
    irf /= irf.sum()
    pure = np.exp(-time / tau_ns)
    reconvolved = np.convolve(irf, pure, mode="full")[:bins]
    reconvolved *= 1500.0 / reconvolved.max()
    cube = np.empty((rows, columns, bins), dtype=np.float64)
    for y in range(rows):
        for x in range(columns):
            cube[y, x] = reconvolved * (0.8 + 0.04 * (y * columns + x)) + 2.0
    return cube, irf


def main() -> None:
    dll = Path(sys.argv[1]).resolve()
    backend = FLIMLibLabViewBackend(dll)
    cube, irf = synthetic_cube(2.4)
    results: dict[str, object] = {"dll": str(dll), "sha256": backend.sha256}
    for noise in range(4):
        fit = backend.fit_pixels(
            cube,
            irf,
            bin_width_ns=0.125,
            model_order=1,
            background_mode=1,
            background_value=2.0,
            noise_model=noise,
            min_photons=100.0,
            irf_mode=0,
        )
        valid = fit.fit_status == 1
        if not valid.all():
            raise AssertionError(f"Noise {noise}: unexpected statuses {np.unique(fit.fit_status)}")
        if not np.allclose(fit.photon_counts, cube.sum(axis=-1)):
            raise AssertionError("Photon-count map does not match input sums")
        results[f"noise_{noise}"] = {
            "median_tau_ns": float(np.nanmedian(fit.tau_mean_ns)),
            "median_chi": float(np.nanmedian(fit.chi_square)),
            "status": sorted(set(fit.fit_status.ravel().tolist())),
        }

    per_pixel = backend.fit_pixels(
        cube,
        np.broadcast_to(irf, cube.shape),
        bin_width_ns=0.125,
        model_order=1,
        background_mode=1,
        background_value=2.0,
        noise_model=0,
        min_photons=100.0,
        irf_mode=1,
    )
    shared = backend.fit_pixels(
        cube,
        irf,
        bin_width_ns=0.125,
        model_order=1,
        background_mode=1,
        background_value=2.0,
        noise_model=0,
        min_photons=100.0,
        irf_mode=0,
    )
    if not np.allclose(per_pixel.tau_mean_ns, shared.tau_mean_ns, rtol=1e-6, atol=1e-6):
        raise AssertionError("Shared and identical tiled IRFs produced different lifetimes")

    global_fit = backend.fit_global(
        cube,
        irf,
        bin_width_ns=0.125,
        model_order=1,
        background_mode=1,
        background_value=2.0,
        noise_model=0,
        min_photons=100.0,
        max_global_pixels=16,
        irf_mode=0,
        global_iterations=3,
        initial_taus_ns=(2.2,),
    )
    if not np.isfinite(global_fit.global_taus_ns).all():
        raise AssertionError("Global fit returned no finite lifetime")
    results["global"] = {
        "taus_ns": global_fit.global_taus_ns.tolist(),
        "stats": global_fit.global_stats.tolist(),
        "status": sorted(set(global_fit.fit_status.ravel().tolist())),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
