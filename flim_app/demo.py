from __future__ import annotations

import numpy as np

from .models import FLIMDataset


def create_demo_dataset(size: int = 160, time_bins: int = 64, seed: int = 7) -> FLIMDataset:
    """Create a cell-like synthetic FLIM field for exploration and testing."""

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    cell_1 = np.exp(-(((x + 0.28) / 0.48) ** 2 + ((y + 0.08) / 0.62) ** 2) * 3.2)
    cell_2 = 0.75 * np.exp(-(((x - 0.38) / 0.34) ** 2 + ((y - 0.22) / 0.45) ** 2) * 3.0)
    nucleus = np.exp(-(((x + 0.18) / 0.22) ** 2 + ((y + 0.03) / 0.27) ** 2) * 3.0)
    puncta = np.zeros_like(x)
    for _ in range(22):
        px, py = rng.uniform(-0.75, 0.75, size=2)
        radius = rng.uniform(0.018, 0.055)
        puncta += rng.uniform(0.3, 1.1) * np.exp(-((x - px) ** 2 + (y - py) ** 2) / radius**2)
    envelope = np.clip(cell_1 + cell_2 - 0.45 * nucleus, 0.0, None)
    amplitude = 7.0 + 950.0 * envelope + 310.0 * puncta
    lifetime = 1.35 + 1.25 * (x + 1.0) / 2.0 + 0.75 * nucleus + 0.25 * np.sin(7 * x) * envelope
    lifetime = np.clip(lifetime, 0.7, 4.5)

    dt_ns = 0.1953125
    t = (np.arange(time_bins, dtype=np.float64) + 0.5) * dt_ns
    irf_centre = 1.2
    onset = 1.0 / (1.0 + np.exp(-(t - irf_centre) / 0.10))
    decay = amplitude[..., None] * onset[None, None, :] * np.exp(
        -np.maximum(t[None, None, :] - irf_centre, 0.0) / lifetime[..., None]
    )
    decay += 1.8
    noisy = rng.poisson(np.clip(decay, 0, 1e8)).astype(np.uint16)
    return FLIMDataset(
        decay=noisy,
        name="Synthetic FLIM cells",
        dt_ns=dt_ns,
        rep_rate_mhz=80.0,
        metadata={
            "format": "Synthetic demo",
            "description": "Two cell-like objects with a spatial lifetime gradient and bright puncta.",
            "ground_truth_tau_range_ns": [float(lifetime.min()), float(lifetime.max())],
        },
    )


def create_demo_irf(size: int, time_bins: int, dt_ns: float = 0.1953125) -> FLIMDataset:
    t = (np.arange(time_bins, dtype=np.float64) + 0.5) * dt_ns
    trace = 2500.0 * np.exp(-0.5 * ((t - 1.2) / 0.11) ** 2) + 2.0
    cube = np.broadcast_to(trace.astype(np.float32), (size, size, time_bins)).copy()
    return FLIMDataset(cube, "Synthetic IRF", dt_ns=dt_ns, rep_rate_mhz=80.0, metadata={"format": "Synthetic IRF"})
