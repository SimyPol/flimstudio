from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.models import AnalysisSettings, FLIMDataset
from flim_app.processing import analyze_dataset
from flim_app.settings_ini import load_settings_ini, save_settings_ini


def _noisy_pair(seed: int = 0):
    """A 'hot' pixel that is pure dark counts: sample and background are two
    independent Poisson draws from the same flat rate (the Swarm situation)."""

    rng = np.random.default_rng(seed)
    shape = (6, 6, 64)
    background = rng.poisson(5.0, size=shape).astype(np.float32)
    sample = rng.poisson(5.0, size=shape).astype(np.float32)
    return sample, background


def _dataset(values: np.ndarray, name: str) -> FLIMDataset:
    return FLIMDataset(values, name=name, dt_ns=0.0488, rep_rate_mhz=80.0, metadata={"scan_loops": 1})


def _settings(clip: bool) -> AnalysisSettings:
    return AnalysisSettings(
        method="phasor_phase", dt_ns=0.0488, rep_rate_mhz=80.0, min_photons=0.0, phasor_min_counts=0.0,
        background_scale_mode="auto", background_clip_negative=clip,
    )


def _processed_total(sample: np.ndarray, background: np.ndarray, clip: bool) -> np.ndarray:
    result = analyze_dataset(_dataset(sample, "s"), _settings(clip), None, _dataset(background, "bg"))
    assert result.metadata["background_correction"]["applied_multiplier"] == 1.0
    return np.asarray(result.intensity, dtype=np.float64)


def test_default_still_clips_every_bin_like_labview() -> None:
    assert AnalysisSettings().background_clip_negative is True


def test_clipping_leaves_a_positive_bias_on_pure_background_pixels() -> None:
    sample, background = _noisy_pair()
    clipped = _processed_total(sample, background, clip=True)
    # The true net signal is zero, but rectifying every bin keeps the positive half of the noise.
    assert clipped.mean() > 40.0


def test_signed_subtraction_removes_pure_background_without_bias() -> None:
    sample, background = _noisy_pair()
    signed = _processed_total(sample, background, clip=False)
    assert abs(signed.mean()) < 15.0
    # ...while a real signal sitting on the same background is preserved.
    boosted = sample.copy()
    boosted[..., 10:20] += 30.0
    recovered = _processed_total(boosted, background, clip=False)
    assert abs(recovered.mean() - signed.mean() - 300.0) < 1.0


def test_metadata_records_whether_bins_were_clipped_or_kept() -> None:
    sample, background = _noisy_pair()
    for clip in (True, False):
        result = analyze_dataset(_dataset(sample, "s"), _settings(clip), None, _dataset(background, "bg"))
        meta = result.metadata["background_correction"]
        assert meta["clip_negative"] is clip
        assert (meta["negative_bins_clipped"] > 0) is clip
        assert (meta["negative_bins_kept"] > 0) is (not clip)


def test_clip_option_round_trips_through_the_settings_ini() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "preset.ini"
        save_settings_ini(AnalysisSettings(background_clip_negative=False), path)
        assert load_settings_ini(path).background_clip_negative is False
