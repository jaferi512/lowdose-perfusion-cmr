"""Simulate a reduced contrast dose on measured signal-time curves.

A dose reduction by a factor f scales the contrast enhancement by ~f while the image
noise stays the same. For ratio indices (myocardium / blood pool) this is equivalent
to keeping the enhancement and raising the noise to sigma / f. The measured curves
already contain noise sigma, so we add extra Gaussian noise with SD
sigma * sqrt(1/f^2 - 1).

Assumptions (stated in the README): signal is linear in concentration - roughly true
at low dose, less so in the blood pool at full dose - and bolus timing is taken from
the full-dose curve, so the experiment isolates the effect of noise on the upslope.
"""
from __future__ import annotations

import numpy as np

from .curves import N_SECTORS, semi_quant
from .io_utils import Series


def extra_noise_factor(f: float) -> float:
    return float(np.sqrt(max(1.0 / f**2 - 1.0, 0.0)))


def simulate_dose_reduction(s: Series, fractions, n_rep: int, rng) -> list[dict]:
    c, m, t = s.curves, s.metrics, s.times
    timing = (m["n_base"], m["onset"], m["peak_blood_idx"], m["window_end"])
    rows = []
    for f in fractions:
        k = extra_noise_factor(f)
        whole, sector = [], []
        for _ in range(n_rep):
            b = c["blood"] + rng.normal(0, k * m["noise_blood"], len(t))
            y = c["myo"] + rng.normal(0, k * m["noise_myo"], len(t))
            whole.append(semi_quant(b, y, t, timing)["upslope_ratio"])
            j = rng.integers(N_SECTORS)  # one random sector per repetition keeps this cheap
            ref = m["sector_upslope_ratio"][j]
            if np.isfinite(m["sector_noise"][j]) and ref > 0:
                ys = c["sectors"][:, j] + rng.normal(0, k * m["sector_noise"][j], len(t))
                sector.append(semi_quant(b, ys, t, timing)["upslope_ratio"] / ref - 1)
        whole = np.asarray(whole)
        rows.append({
            "series": s.key, "patient": s.patient, "condition": s.condition, "slice": s.slice_name,
            "dose_fraction": f,
            "whole_slice_rel_sd": float(np.nanstd(whole) / abs(m["upslope_ratio"])),
            "whole_slice_rel_bias": float(np.nanmean(whole) / m["upslope_ratio"] - 1),
            "sector_rel_sd": float(np.nanstd(sector)) if sector else np.nan,
        })
    return rows
