"""Physics experiments on simulated data with known ground truth.

1. Repeated fits of healthy and ischaemic tissue curves at decreasing dose.
2. A digital reference object (DRO): a 12 x 12 grid of (Fp, PS) combinations, each a
   small block of voxels with independent noise, fitted voxel-wise at full and low dose.
   This mirrors the DRO used to evaluate PINNs in Tsepas et al. (STACOM 2026), in a
   reduced form that runs on a CPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kinetics import (ACQ_IDX, HEALTHY, ISCHAEMIC, PARAM_NAMES, T_ACQ, T_FINE, fit_2cxm,
                       gamma_variate_aif, noise_sd, tissue_curve)

FP_VALUES = np.round(np.geomspace(0.5, 4.0, 12), 2)   # ml/min/ml
PS_VALUES = np.round(np.geomspace(0.2, 2.0, 12), 2)
DRO_VP, DRO_VE = 0.08, 0.22


def repeated_fits(doses, n_rep: int, rng) -> pd.DataFrame:
    sd = noise_sd()
    rows = []
    for dose in doses:
        aif = gamma_variate_aif(T_FINE, dose)
        for name, p in (("healthy", HEALTHY), ("ischaemic", ISCHAEMIC)):
            clean = tissue_curve(p, aif)[ACQ_IDX]
            for r in range(n_rep):
                est, est_sd = fit_2cxm(clean + rng.normal(0, sd, len(T_ACQ)), aif)
                rows.append({"dose": dose, "tissue": name, "rep": r,
                             **{f"{k}_true": v for k, v in zip(PARAM_NAMES, p)},
                             **{f"{k}_est": v for k, v in zip(PARAM_NAMES, est)},
                             **{f"{k}_sd": v for k, v in zip(PARAM_NAMES, est_sd)}})
    return pd.DataFrame(rows)


def make_dro(doses, block: int, rng) -> dict:
    """Fit the DRO voxel-wise. Returns true Fp map and (estimate, SD) maps per dose."""
    sd = noise_sd()
    n = len(FP_VALUES) * block
    true_fp = np.repeat(np.repeat(FP_VALUES[:, None], len(PS_VALUES), 1), block, 0).repeat(block, 1)
    out = {"true_fp": true_fp, "fp_values": FP_VALUES, "ps_values": PS_VALUES, "maps": {}}
    for dose in doses:
        aif = gamma_variate_aif(T_FINE, dose)
        est = np.full((n, n), np.nan)
        est_sd = np.full((n, n), np.nan)
        for i, fp in enumerate(FP_VALUES):
            for j, ps in enumerate(PS_VALUES):
                clean = tissue_curve((fp, ps, DRO_VP, DRO_VE), aif)[ACQ_IDX]
                for bi in range(block):
                    for bj in range(block):
                        e, s = fit_2cxm(clean + rng.normal(0, sd, len(T_ACQ)), aif)
                        est[i * block + bi, j * block + bj] = e[0]
                        est_sd[i * block + bi, j * block + bj] = s[0]
        out["maps"][dose] = (est, est_sd)
    return out


def _calib_row(source: str, dose: float, z: np.ndarray) -> dict:
    z = z[np.isfinite(z)]
    return {"source": source, "dose": dose, "n_fits": len(z),
            "coverage_95": float(np.mean(np.abs(z) < 1.96)) if len(z) else np.nan,
            "z_sd": float(np.std(z)) if len(z) else np.nan,
            "z_median": float(np.median(z)) if len(z) else np.nan}


def calibration_table(fits: pd.DataFrame, dro: dict) -> pd.DataFrame:
    """How often does the NLLS 95% interval (estimate +/- 1.96 SD) contain the true Fp? Ideal: 95%."""
    rows = []
    for (tissue, dose), sub in fits.groupby(["tissue", "dose"]):
        rows.append(_calib_row(tissue, dose, ((sub.Fp_est - sub.Fp_true) / sub.Fp_sd).values))
    for dose, (est, sd) in dro["maps"].items():
        rows.append(_calib_row("DRO", dose, ((est - dro["true_fp"]) / sd).ravel()))
    return pd.DataFrame(rows)
