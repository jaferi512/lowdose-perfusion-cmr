"""Signal-time curves from per-frame contours and semi-quantitative perfusion indices.

Blood pool  = pixels inside the endocardial contour of each frame (arterial input proxy).
Myocardium  = inside epicardium and outside endocardium, again per frame, so breathing
              motion is followed by the manual contours.
Sectors     = myocardium split into 6 angular sectors starting at the RV insertion point.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.signal import savgol_filter

from .io_utils import Series

N_SECTORS = 6


def _mask(poly, shape, bbox):
    y0, y1, x0, x1 = bbox
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = MplPath(poly).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(yy.shape)
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = inside
    return m


def frame_masks(s: Series, t: int):
    """(blood_mask, myo_mask) for frame t, or (None, None) if the frame has no contours."""
    epi, endo = s.epi[t], s.endo[t]
    if epi is None or endo is None:
        return None, None
    H, W = s.images.shape[1:]
    x0, y0 = np.floor(epi.min(0)).astype(int) - 1
    x1, y1 = np.ceil(epi.max(0)).astype(int) + 2
    bbox = (max(y0, 0), min(y1, H), max(x0, 0), min(x1, W))
    m_epi, m_endo = _mask(epi, (H, W), bbox), _mask(endo, (H, W), bbox)
    return m_endo, m_epi & ~m_endo


def borrow_missing_endocardium(s: Series, min_fraction: float = 0.5) -> int:
    """Frames that have an epicardium but no endocardium get the endocardium of the nearest
    fully annotated frame, shifted by the displacement of the epicardium between the two
    frames. The dataset paper notes that some pre-contrast frames have no clear boundaries.
    Returns the number of frames filled; does nothing if fewer than half the frames are complete."""
    full = [t for t in range(len(s.epi)) if s.epi[t] is not None and s.endo[t] is not None]
    todo = [t for t in range(len(s.epi)) if s.epi[t] is not None and s.endo[t] is None]
    if not todo or len(full) < min_fraction * len(s.epi):
        return 0
    full_arr = np.asarray(full)
    for t in todo:
        n = int(full_arr[np.argmin(np.abs(full_arr - t))])
        shift = s.epi[t].mean(0) - s.epi[n].mean(0)
        s.endo[t] = s.endo[n] + shift
    return len(todo)


def extract_curves(s: Series) -> None:
    n_borrowed = borrow_missing_endocardium(s)
    T = len(s.images)
    blood, myo, myo_sd = (np.full(T, np.nan) for _ in range(3))
    sectors = np.full((T, N_SECTORS), np.nan)
    centroid = np.full((T, 2), np.nan)
    H, W = s.images.shape[1:]
    reasons = {"no epicardium": 0, "no endocardium": 0, "contour outside image": 0, "region too small": 0}
    for t in range(T):
        if s.epi[t] is None:
            reasons["no epicardium"] += 1
            continue
        if s.endo[t] is None:
            reasons["no endocardium"] += 1
            continue
        lo, hi = s.epi[t].min(0), s.epi[t].max(0)
        if hi[0] < 0 or hi[1] < 0 or lo[0] >= W or lo[1] >= H:
            reasons["contour outside image"] += 1
            continue
        m_blood, m_myo = frame_masks(s, t)
        if m_blood is None or m_myo.sum() < 10 or m_blood.sum() < 5:
            reasons["region too small"] += 1
            continue
        img = s.images[t]
        blood[t], myo[t], myo_sd[t] = img[m_blood].mean(), img[m_myo].mean(), img[m_myo].std()
        centroid[t] = s.endo[t].mean(0)

        c = s.centre[t] if s.centre[t] is not None else s.endo[t].mean(0)
        a0 = np.arctan2(s.rv_dir[t][1], s.rv_dir[t][0]) if s.rv_dir[t] is not None else 0.0
        yy, xx = np.nonzero(m_myo)
        ang = np.mod(np.arctan2(yy - c[1], xx - c[0]) - a0, 2 * np.pi)
        sec = np.minimum((ang / (2 * np.pi / N_SECTORS)).astype(int), N_SECTORS - 1)
        vals = img[yy, xx]
        for k in range(N_SECTORS):
            sel = sec == k
            if sel.sum() >= 3:
                sectors[t, k] = vals[sel].mean()

    idx = np.arange(T)

    def fill(y):  # interpolate the odd frame without contours
        ok = np.isfinite(y)
        return np.interp(idx, idx[ok], y[ok]) if ok.sum() >= 2 else y

    s.curves = {
        "blood": fill(blood), "myo": fill(myo), "myo_sd": fill(myo_sd),
        "sectors": np.stack([fill(sectors[:, k]) for k in range(N_SECTORS)], 1),
        "centroid": np.stack([fill(centroid[:, 0]), fill(centroid[:, 1])], 1),
        "has_rv": any(r is not None for r in s.rv_dir),
        "n_valid": int(np.isfinite(blood).sum()),
        "n_borrowed_endo": n_borrowed,
        "fail_reasons": {k: v for k, v in reasons.items() if v},
    }


def curves_ok(s: Series, min_fraction: float = 0.8) -> tuple[bool, str]:
    """A series is usable if most frames gave a blood-pool and myocardium value."""
    c = s.curves
    T = len(s.images)
    if c["n_valid"] >= min_fraction * T and np.all(np.isfinite(c["blood"])) and np.all(np.isfinite(c["myo"])):
        return True, ""
    why = ", ".join(f"{k}: {v} frames" for k, v in c["fail_reasons"].items()) or "unknown"
    return False, f"usable contours in {c['n_valid']}/{T} frames ({why})"


# ------------------------------------------------------------ indices ---
def smooth(y: np.ndarray) -> np.ndarray:
    return savgol_filter(y, 5, 2) if len(y) >= 7 else y


def robust_noise(y: np.ndarray) -> float:
    """White-noise SD estimated from first differences (robust to the slow bolus signal)."""
    d = np.diff(y)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def bolus_timing(blood: np.ndarray, window: int = 15):
    """(n_baseline_frames, onset, blood_peak, end_of_first_pass_window) in frame indices."""
    b0 = np.median(blood[:3])
    peak_i = int(np.argmax(smooth(blood)))
    thr = b0 + 0.1 * (blood[peak_i] - b0)
    above = np.nonzero(blood[: peak_i + 1] > thr)[0]
    onset = int(above[0]) if len(above) else max(peak_i - 3, 0)
    n_base = max(2, onset - 1)
    end = min(len(blood), peak_i + window)
    return n_base, onset, peak_i, end


def upslope(y: np.ndarray, t: np.ndarray, lo: int, hi: int) -> float:
    g = np.gradient(smooth(y), t)
    return float(np.nanmax(g[max(lo - 1, 0): max(hi + 1, lo + 1)]))


def semi_quant(blood, myo, t, timing) -> dict:
    """Upslope and peak-enhancement ratios (myocardium / blood pool), time to peak.

    Ratios are invariant to a linear intensity scaling, so they are valid for the
    per-series 8-bit PNGs as well as the 16-bit DICOMs.
    """
    n_base, onset, peak_b, end = timing
    db = blood - blood[:n_base].mean()
    dm = myo - myo[:n_base].mean()
    sdb, sdm = smooth(db), smooth(dm)
    peak_m = onset + int(np.argmax(sdm[onset:end]))
    up_b, up_m = upslope(db, t, onset, peak_b), upslope(dm, t, onset, peak_m)
    return {
        "upslope_blood": up_b, "upslope_myo": up_m,
        "upslope_ratio": up_m / up_b if up_b > 0 else np.nan,
        "pe_blood": float(sdb[peak_b]), "pe_myo": float(sdm[peak_m]),
        "pe_ratio": float(sdm[peak_m] / sdb[peak_b]) if sdb[peak_b] > 0 else np.nan,
        "ttp_blood": float(t[peak_b] - t[onset]), "ttp_myo": float(t[peak_m] - t[onset]),
        "peak_myo_idx": peak_m,
    }


def compute_metrics(s: Series) -> None:
    c, t = s.curves, s.times
    timing = bolus_timing(c["blood"])
    n_base, onset, peak_b, end = timing
    m = semi_quant(c["blood"], c["myo"], t, timing)

    # Contrast-to-noise: peak myocardial enhancement / spatial SD of the myocardium at baseline.
    m["cnr_myo"] = m["pe_myo"] / max(float(np.mean(c["myo_sd"][:n_base])), 1e-6)
    m["noise_blood"], m["noise_myo"] = robust_noise(c["blood"]), robust_noise(c["myo"])

    # Respiratory motion: LV centroid displacement along its main direction, in mm.
    # If the contours never move (one contour copied to every frame), motion cannot be
    # measured from them: report NaN rather than a misleading 0 mm.
    xy = (c["centroid"] - np.nanmedian(c["centroid"], 0)) * s.spacing_mm
    m["static_contours"] = bool(np.nanmax(np.abs(xy)) < 1e-3)
    if m["static_contours"]:
        m["motion_p2p_mm"], m["motion_trace_mm"] = np.nan, np.full(len(t), np.nan)
    else:
        _, _, vt = np.linalg.svd(np.nan_to_num(xy), full_matrices=False)
        proj = xy @ vt[0]
        m["motion_p2p_mm"] = float(np.nanpercentile(proj, 95) - np.nanpercentile(proj, 5))
        m["motion_trace_mm"] = proj

    # Per-sector upslope ratio, blood-pool upslope as the arterial reference.
    db = c["blood"] - c["blood"][:n_base].mean()
    up_b = upslope(db, t, onset, peak_b)
    ratios, noises = [], []
    for k in range(N_SECTORS):
        y = c["sectors"][:, k]
        if not np.all(np.isfinite(y)) or up_b <= 0:
            ratios.append(np.nan)
            noises.append(np.nan)
            continue
        dy = y - y[:n_base].mean()
        pk = onset + int(np.argmax(smooth(dy)[onset:end]))
        ratios.append(upslope(dy, t, onset, pk) / up_b)
        noises.append(robust_noise(y))
    m["sector_upslope_ratio"] = np.asarray(ratios)
    m["sector_noise"] = np.asarray(noises)
    m.update(n_base=n_base, onset=onset, peak_blood_idx=peak_b, window_end=end)
    s.metrics = m
