from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import pytest
from tkinter import messagebox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flim_app.gui as gui_module
from flim_app.gui import FLIMStudioApp
from flim_app.models import FLIMDataset
from flim_app.processing import analyze_dataset, phasor_lifetimes

DT_NS = 12.5 / 128
REP_MHZ = 80.0
TAU_NS = 2.7


def _bead_dataset(background_level: float = 0.0):
    """Small noise-free cube: Gaussian-broadened excitation at ~2 ns then a 2.7 ns decay,
    on top of an optional flat per-pixel background acquisition."""

    t = (np.arange(128) + 0.5) * DT_NS
    fine = np.linspace(0.0, 12.5, 6000, endpoint=False)
    sigma = 0.12
    pulse = np.exp(-0.5 * ((fine - 2.0) / sigma) ** 2)
    kernel = np.where(fine >= 0, np.exp(-fine / TAU_NS), 0.0)
    decay = np.convolve(pulse, kernel)[: fine.size]
    decay = np.interp(t, fine, decay)
    decay = decay / decay.sum() * 5000.0
    sample = np.broadcast_to(decay + background_level / 128.0, (6, 6, 128)).astype(np.float32).copy()
    background = np.full((6, 6, 128), background_level / 128.0, dtype=np.float32)
    mk = lambda cube, name: FLIMDataset(cube, name=name, dt_ns=DT_NS, rep_rate_mhz=REP_MHZ, metadata={"scan_loops": 1})
    return mk(sample, "bead"), mk(background, "bg")


@pytest.fixture()
def app(monkeypatch, tmp_path):
    for name in ("showinfo", "showwarning", "showerror", "askyesno", "askokcancel"):
        monkeypatch.setattr(messagebox, name, lambda *a, **k: None)
    # Don't let a user's real default.ini (file paths, column interpolation, ...) leak into the test.
    monkeypatch.setattr(gui_module, "main_folder", lambda: tmp_path)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    window = FLIMStudioApp(root)
    window.method_var.set(next(k for k, v in window.METHOD_LABELS.items() if v == "phasor_phase"))
    window.phasor_threshold_var.set("100")
    yield window
    root.destroy()


def test_live_synthetic_irf_matches_the_committed_analysis_irf(app) -> None:
    sample, background = _bead_dataset(background_level=2000.0)
    app._set_dataset(sample)
    app.irf = None
    app.background = background
    app.irf_source_var.set("Synthetic (Gaussian)")
    settings = app._settings()

    live = app._live_fit_generic_irf_trace(settings)
    committed = analyze_dataset(sample, settings, None, background).fit_diagnostics["irf_trace"]

    assert app.result is None  # no analysis has been run -- this is the pre-analysis path
    assert live is not None and live.shape == (128,)
    np.testing.assert_allclose(live, committed, rtol=1e-3, atol=1e-6)


def test_live_phasor_preview_uses_the_synthetic_irf_before_any_analysis(app) -> None:
    sample, _ = _bead_dataset()
    app._set_dataset(sample)
    app.irf = None
    settings_file = None

    app.irf_source_var.set("Loaded file")
    app._run_single_point_live_fit((0, 0, 6, 6))
    uncorrected = app.roi_summary_var.get()
    app.irf_source_var.set("Synthetic (Gaussian)")
    app._run_single_point_live_fit((0, 0, 6, 6))
    corrected = app.roi_summary_var.get()

    def tau_of(text: str) -> float:
        return float(text.split("tau=")[1].split(" ns")[0])

    # The synthetic Gaussian is an approximation (its width comes from the decay's own rising edge),
    # so this only checks the correction lands near the truth; exactness against the committed
    # analysis is covered by test_live_synthetic_irf_matches_the_committed_analysis_irf.
    assert abs(tau_of(corrected) - TAU_NS) < 0.5
    assert abs(tau_of(uncorrected) - TAU_NS) > 1.0
    del settings_file


def test_committed_irf_is_only_reused_when_it_matches_the_sidebar(app) -> None:
    sample, _ = _bead_dataset()
    app._set_dataset(sample)
    app.irf = None
    app.irf_source_var.set("Synthetic (Gaussian)")
    result = analyze_dataset(sample, app._settings(), None)

    assert app._result_irf_matches(result, app._settings()) is True
    app.irf_source_var.set("Loaded file")
    assert app._result_irf_matches(result, app._settings()) is False
    # A different pinned FWHM invalidates the committed synthetic IRF too.
    app.irf_source_var.set("Synthetic (Gaussian)")
    app.synthetic_irf_fwhm_var.set("0.5")
    assert app._result_irf_matches(result, app._settings()) is False
