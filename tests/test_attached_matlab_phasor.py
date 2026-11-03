from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.processing import phasor_lifetimes


def test_attached_embedded_matlab_equations_exactly() -> None:
    rng = np.random.default_rng(311)
    decay = rng.poisson(80, size=(5, 7, 32)).astype(np.float64)
    irf = rng.poisson(15, size=(5, 7, 32)).astype(np.float64)
    irf[..., 5:8] += 500
    dt_ns = 0.390625
    rep_rate_mhz = 80.0
    threshold = 1000.0
    g, s, tau_phase, tau_mod = phasor_lifetimes(
        decay, dt_ns, rep_rate_mhz, threshold, irf
    )

    decay_rows = decay.reshape(-1, 32)
    irf_rows = irf.reshape(-1, 32)
    irf_clean = np.clip(irf_rows - np.median(irf_rows, axis=1, keepdims=True), 0, None)
    time = (np.arange(32) + 0.5) * dt_ns * 1e-9
    omega = 2 * np.pi * rep_rate_mhz * 1e6
    cosine, sine = np.cos(omega * time), np.sin(omega * time)
    decay_total, irf_total = decay_rows.sum(1), irf_clean.sum(1)
    z_raw = (decay_rows @ cosine + 1j * (decay_rows @ sine)) / decay_total
    z_irf = (irf_clean @ cosine + 1j * (irf_clean @ sine)) / irf_total
    valid = (decay_total >= threshold) & (irf_total > 0) & (np.abs(z_irf) > 0)
    corrected = np.full(z_raw.shape, np.nan + 1j * np.nan)
    corrected[valid] = z_raw[valid] / z_irf[valid]
    corrected[valid] *= np.exp(1j * omega * dt_ns * 1e-9 / 2)
    assert np.allclose(g.ravel(), corrected.real, equal_nan=True, atol=1e-14)
    assert np.allclose(s.ravel(), corrected.imag, equal_nan=True, atol=1e-14)
    expected_phase = np.tan(np.angle(corrected)) / omega * 1e9
    assert np.allclose(tau_phase.ravel(), expected_phase, equal_nan=True, atol=1e-12)
    modulation = np.abs(corrected)
    expected_mod = np.full(modulation.shape, np.nan)
    valid_mod = valid & (modulation > 0) & (modulation <= 1)
    expected_mod[valid_mod] = np.sqrt(1 / modulation[valid_mod] ** 2 - 1) / omega * 1e9
    assert np.allclose(tau_mod.ravel(), expected_mod, equal_nan=True, atol=1e-12)


if __name__ == "__main__":
    test_attached_embedded_matlab_equations_exactly()
    print("PASS attached embedded MATLAB phasor equations")
