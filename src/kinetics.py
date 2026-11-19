"""Two-compartment exchange model (2CXM) for myocardial perfusion, and an NLLS fitter.

The 2CXM describes contrast agent in a plasma compartment (volume vp) exchanging with
the interstitium (volume ve):

    vp dCp/dt = Fp (Ca - Cp) + PS (Ce - Cp)
    ve dCe/dt = PS (Cp - Ce)
    C_tissue  = vp Cp + ve Ce

Fp = plasma flow, PS = permeability-surface area product (both ml/min/ml).
The system is linear, so its impulse response is a sum of two exponentials whose rates
and weights come from the eigen-decomposition of the 2x2 system matrix. The tissue curve
is that impulse response convolved with the arterial input function (AIF), computed on
a fine time grid with a recursive filter. ``tests/test_kinetics.py`` checks this against
a direct ODE solution.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lfilter

DT_FINE = 0.1                       # s, internal simulation grid
T_ACQ = np.linspace(0, 120, 100)    # 2 min, 100 time points (as in Tsepas et al., 2026)
T_FINE = np.arange(0, T_ACQ[-1] + DT_FINE, DT_FINE)
ACQ_IDX = np.round(T_ACQ / DT_FINE).astype(int)

SNR_FULL = 17.5                     # noise level used in the paper, defined at full dose
PARAM_NAMES = ["Fp", "PS", "vp", "ve"]
BOUNDS = ([0.05, 0.01, 0.01, 0.02], [8.0, 4.0, 0.5, 0.8])
HEALTHY = np.array([2.5, 1.2, 0.10, 0.20])      # illustrative stress values
ISCHAEMIC = np.array([0.9, 0.6, 0.06, 0.25])


def gamma_variate_aif(t: np.ndarray, dose: float = 1.0, t0: float = 8.0, alpha: float = 3.0,
                      beta: float = 1.6, peak: float = 6.0, recirculation: float = 0.12) -> np.ndarray:
    """Gamma-variate first pass (peak in mM at full dose) + a small recirculation tail."""
    tt = np.clip(t - t0, 0, None)
    first = (tt / (alpha * beta)) ** alpha * np.exp(alpha - tt / beta)
    tail = recirculation * (1 - np.exp(-tt / 6.0)) * np.exp(-tt / 250.0)
    return dose * peak * (first + tail)


def impulse_terms(p):
    """Eigenvalues (1/s) and weights of the 2CXM impulse response h(t) = sum w_i exp(lam_i t)."""
    Fp, PS, vp, ve = p[0] / 60.0, p[1] / 60.0, p[2], p[3]
    A = np.array([[-(Fp + PS) / vp, PS / vp], [PS / ve, -PS / ve]])
    b = np.array([Fp / vp, 0.0])
    c = np.array([vp, ve])
    lam, V = np.linalg.eig(A)
    w = (c @ V) * np.linalg.solve(V, b)
    return lam.real, w.real


def tissue_curve(p, aif_fine: np.ndarray) -> np.ndarray:
    """Tissue concentration on the fine grid for parameters p = (Fp, PS, vp, ve)."""
    lam, w = impulse_terms(p)
    y = np.zeros_like(aif_fine)
    for l_i, w_i in zip(lam, w):
        y += w_i * lfilter([DT_FINE], [1.0, -np.exp(l_i * DT_FINE)], aif_fine)
    return y


def fit_2cxm(y: np.ndarray, aif_fine: np.ndarray):
    """Non-linear least squares from two starting points.

    Returns (estimates, 1-sigma SDs from the linearised covariance). NaNs if all fits fail.
    """
    def f(_, *p):
        return tissue_curve(p, aif_fine)[ACQ_IDX]

    best = None
    for p0 in ([1.5, 0.8, 0.08, 0.2], [0.8, 0.4, 0.05, 0.3]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(f, T_ACQ, y, p0=p0, bounds=BOUNDS, method="trf", max_nfev=300)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        sse = float(np.sum((f(None, *popt) - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    if best is None:
        return np.full(4, np.nan), np.full(4, np.nan)
    sd = np.sqrt(np.clip(np.diag(best[2]), 0, None))
    return best[1], np.where(np.isfinite(sd), sd, np.nan)


def noise_sd() -> float:
    """Absolute noise SD: healthy full-dose tissue peak / 17.5. Kept fixed when dose drops."""
    return float(tissue_curve(HEALTHY, gamma_variate_aif(T_FINE))[ACQ_IDX].max() / SNR_FULL)
