"""The 2CXM implementation must match a direct ODE solution and recover clean parameters."""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kinetics import (ACQ_IDX, HEALTHY, ISCHAEMIC, T_ACQ, T_FINE, fit_2cxm,  # noqa: E402
                          gamma_variate_aif, tissue_curve)


def ode_reference(p, aif):
    Fp, PS, vp, ve = p[0] / 60, p[1] / 60, p[2], p[3]

    def rhs(t, x):
        ca = np.interp(t, T_FINE, aif)
        return [(Fp * (ca - x[0]) + PS * (x[1] - x[0])) / vp, PS * (x[0] - x[1]) / ve]

    sol = solve_ivp(rhs, (0, T_ACQ[-1]), [0, 0], t_eval=T_ACQ, max_step=0.05)
    return vp * sol.y[0] + ve * sol.y[1]


def test_matches_ode():
    aif = gamma_variate_aif(T_FINE)
    for p in (HEALTHY, ISCHAEMIC):
        ours = tissue_curve(p, aif)[ACQ_IDX]
        ref = ode_reference(p, aif)
        assert np.max(np.abs(ours - ref)) / ref.max() < 0.03


def test_recovers_noise_free_parameters():
    aif = gamma_variate_aif(T_FINE)
    est, _ = fit_2cxm(tissue_curve(ISCHAEMIC, aif)[ACQ_IDX], aif)
    assert np.allclose(est, ISCHAEMIC, rtol=1e-3)
