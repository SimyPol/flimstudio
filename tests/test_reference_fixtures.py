from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.native_backend import fit_in_subprocess


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation_data"


def fit_fixture(name: str, per_pixel_irf: bool = False):
    with np.load(FIXTURES / name, allow_pickle=False) as source:
        cube = np.transpose(source["data"], (1, 2, 0))
        irf = source["irfCounts"]
        irf = np.transpose(irf, (1, 2, 0)) if per_pixel_irf else irf.ravel()
        truth = source["tauTruth"].copy()
        dt = float(source["binWidth"].item())
    result = fit_in_subprocess(
        "pixels",
        cube,
        irf,
        bin_width_ns=dt,
        model_order=1,
        background_mode=0,
        background_value=0.0,
        noise_model=0,
        min_photons=0.0,
        irf_mode=1 if per_pixel_irf else 0,
        timeout_seconds=120,
    )
    return result, truth


def test_single_decay_mono() -> None:
    with np.load(FIXTURES / "single_decay_mono.npz", allow_pickle=False) as source:
        result = fit_in_subprocess(
            "pixels",
            source["decayCounts"].reshape(1, 1, -1),
            source["irfCounts"].ravel(),
            bin_width_ns=float(source["binWidth"].item()),
            model_order=1,
            background_mode=0,
            background_value=0.0,
            noise_model=0,
            min_photons=0.0,
            irf_mode=0,
        )
    assert result.fit_status.item() == 1
    assert abs(result.tau_mean_ns.item() - 2.4) < 0.08
    assert result.fit_start_bin.item() == 31
    assert abs(result.time_shift_ns.item() - 0.9765625) < 1e-9


def _assert_map_reference(name: str, per_pixel_irf: bool, mean_target: float, mae_limit: float) -> None:
    result, truth = fit_fixture(name, per_pixel_irf)
    assert np.all(result.fit_status == 1)
    fitted = result.tau_mean_ns
    assert abs(float(np.mean(fitted)) - mean_target) < 0.04
    assert float(np.mean(np.abs(fitted - truth))) < mae_limit


def main() -> None:
    test_single_decay_mono()
    print("PASS supplied single-decay mono fixture")
    _assert_map_reference("image_shared_irf.npz", False, 2.3474, 0.11)
    print("PASS supplied shared-IRF image fixture")
    _assert_map_reference("image_per_pixel_irf.npz", True, 2.0365, 0.10)
    print("PASS supplied per-pixel-IRF image fixture")


if __name__ == "__main__":
    main()
