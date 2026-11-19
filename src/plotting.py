"""All figures. One categorical order (rest = blue, stress = orange), a one-hue sequential
ramp for magnitudes, a blue-gray-orange diverging ramp for signed errors, no dual axes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .curves import N_SECTORS, smooth  # noqa: E402
from .io_utils import SLICE_NAMES, Series  # noqa: E402
from .kinetics import (ACQ_IDX, HEALTHY, ISCHAEMIC, T_ACQ, T_FINE, fit_2cxm,  # noqa: E402
                       gamma_variate_aif, noise_sd, tissue_curve)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND_COLOR = {"Rest": BLUE, "Stress": ORANGE}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#7fb0ea", BLUE, "#0d2f5c"])
DIV = LinearSegmentedColormap.from_list("div_blue_orange", [BLUE, "#d9d8d4", ORANGE])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, out_dir: Path, name: str, log=print) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name)
    plt.close(fig)
    log(f"  [fig] {name}")


def _time_label(series) -> str:
    unit = "s" if all(s.time_unit == "s" for s in series) else "frames"
    return f"time since bolus arrival ({unit})"


# ================================================================ real data ===
def fig_overview(series: list[Series], out: Path) -> None:
    patients = sorted({s.patient for s in series})
    cols = [(c, k) for c in ("Rest", "Stress") for k in (2, 1, 0)]
    grid = np.full((len(patients), len(cols)), np.nan)
    for s in series:
        if (s.condition, s.slice_idx) in cols:
            grid[patients.index(s.patient), cols.index((s.condition, s.slice_idx))] = len(s.times)
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(patients) + 1.7))
    vmax = np.nanmax(grid) if np.isfinite(grid).any() else 1
    ax.imshow(grid, cmap=SEQ, vmin=0, vmax=vmax * 1.15, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, "missing" if np.isnan(v) else f"{int(v)}", ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(v) and v > 0.55 * vmax else INK)
    ax.set_xticks(range(len(cols)), [f"{c}\n{SLICE_NAMES[k]}" for c, k in cols])
    ax.set_yticks(range(len(patients)), [f"patient {p}" for p in patients])
    ax.grid(False)
    ax.axvline(2.5, color="white", lw=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    pd_removed = sorted({s.n_pd_skipped for s in series})
    fmt = ", ".join(sorted({s.image_format for s in series}))
    ax.set_title(f"Dynamic frames per slice series ({len(series)} series, {fmt}; "
                 f"proton-density frames removed: {pd_removed})", loc="left")
    save(fig, out, "01_dataset_overview.png")


def _crop_box(s: Series, pad: int = 28):
    pts = np.concatenate([e for e in s.epi if e is not None])
    x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - pad, 0)
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    return y0, min(y1, s.images.shape[1]), x0, min(x1, s.images.shape[2])


def fig_montage(s: Series, out: Path) -> None:
    m = s.metrics
    idx = {"baseline": max(m["onset"] - 2, 0),
           "early bolus": m["onset"] + max((m["peak_blood_idx"] - m["onset"]) // 3, 0),
           "blood-pool peak": m["peak_blood_idx"],
           "myocardial peak": m["peak_myo_idx"],
           "late": len(s.times) - 1}
    y0, y1, x0, x1 = _crop_box(s)
    crop = s.images[:, y0:y1, x0:x1]
    vmin, vmax = np.percentile(crop, [1, 99.5])
    fig, axes = plt.subplots(1, len(idx), figsize=(2.3 * len(idx), 2.8))
    unit = "s" if s.time_unit == "s" else "fr"
    for ax, (name, t) in zip(axes, idx.items()):
        ax.imshow(crop[t], cmap="gray", vmin=vmin, vmax=vmax)
        for poly, col in ((s.epi[t], AQUA), (s.endo[t], ORANGE)):
            if poly is not None:
                p = np.vstack([poly, poly[:1]])
                ax.plot(p[:, 0] - x0, p[:, 1] - y0, color=col, lw=1.2)
        ax.set_title(f"{name}\nt = {s.times[t]:.0f} {unit}", fontsize=8.5)
        ax.axis("off")
    axes[0].plot([], [], color=AQUA, label="epicardium")
    axes[0].plot([], [], color=ORANGE, label="endocardium")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"First pass of the contrast bolus: patient {s.patient}, {s.condition.lower()}, "
                 f"{s.slice_name} slice (per-frame manual contours)", x=0.01, ha="left")
    save(fig, out, "02_perfusion_montage.png")


def _aligned(s: Series):
    t = s.times - s.times[s.metrics["onset"]]
    n = s.metrics["n_base"]
    return t, s.curves["blood"] - s.curves["blood"][:n].mean(), s.curves["myo"] - s.curves["myo"][:n].mean()


def fig_curves_patient(series: list[Series], patient: int, out: Path) -> None:
    sub = [s for s in series if s.patient == patient]
    slices = sorted({s.slice_idx for s in sub}, reverse=True)
    fig, axes = plt.subplots(2, len(slices), figsize=(3.3 * len(slices), 5.2), sharex=True, squeeze=False)
    for j, k in enumerate(slices):
        for s in (s for s in sub if s.slice_idx == k):
            t, db, dm = _aligned(s)
            axes[0, j].plot(t, db, color=COND_COLOR[s.condition], label=s.condition)
            axes[1, j].plot(t, dm, color=COND_COLOR[s.condition], label=s.condition)
        axes[0, j].set_title(f"{SLICE_NAMES[k]} slice")
    axes[0, 0].set_ylabel("LV blood pool\nsignal change (a.u.)")
    axes[1, 0].set_ylabel("myocardium\nsignal change (a.u.)")
    for ax in axes[1]:
        ax.set_xlabel(_time_label(sub))
    axes[0, -1].legend(loc="upper right")
    fig.suptitle(f"Signal-time curves, patient {patient}: blood pool (arterial input) vs. myocardium "
                 "- note the different y-scales", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "03_signal_time_curves.png")


def fig_rest_stress_all(series: list[Series], out: Path, slice_idx: int = 1) -> None:
    patients = sorted({s.patient for s in series})
    ncol = min(5, len(patients))
    nrow = int(np.ceil(len(patients) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    for ax, p in zip(axes.ravel(), patients):
        for s in series:
            if s.patient == p and s.slice_idx == slice_idx:
                t, db, dm = _aligned(s)
                ax.plot(t, dm / np.nanmax(smooth(db)), color=COND_COLOR[s.condition], label=s.condition)
        ax.set_title(f"patient {p}", fontsize=9)
    for ax in axes.ravel()[len(patients):]:
        ax.axis("off")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.supxlabel(_time_label(series), fontsize=9)
    fig.supylabel("myocardial enhancement / blood-pool peak", fontsize=9)
    fig.tight_layout()
    fig.suptitle(f"Myocardial first-pass curves, {SLICE_NAMES.get(slice_idx, slice_idx)} slice, "
                 "normalised to each scan's arterial peak", x=0.01, y=1.04, ha="left")
    save(fig, out, "04_rest_vs_stress_all_patients.png")


def fig_semiquant(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    per_patient = df.groupby(["patient", "condition"])[["upslope_ratio", "pe_ratio"]].mean().unstack()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    for ax, metric, title in ((axes[0], "upslope_ratio", "Upslope ratio (myo / blood)"),
                              (axes[1], "pe_ratio", "Peak-enhancement ratio (myo / blood)")):
        d = per_patient[metric]
        if {"Rest", "Stress"} <= set(d.columns):
            for _, row in d.dropna().iterrows():
                ax.plot([0, 1], [row["Rest"], row["Stress"]], color=GRID, lw=1.2, zorder=1)
        for x, cond in enumerate(("Rest", "Stress")):
            if cond in d.columns:
                v = d[cond].dropna()
                ax.scatter(np.full(len(v), x), v, s=40, color=COND_COLOR[cond],
                           edgecolor="white", linewidth=1.5, zorder=2)
        ax.set_xticks([0, 1], ["Rest", "Stress"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(title, loc="left")
    reserve = pd.Series(dtype=float)
    if {"Rest", "Stress"} <= set(per_patient["upslope_ratio"].columns):
        reserve = (per_patient["upslope_ratio"]["Stress"] / per_patient["upslope_ratio"]["Rest"]).dropna()
        axes[2].bar([str(p) for p in reserve.index], reserve.values, color=BLUE, width=0.6)
        axes[2].axhline(1.0, color=INK2, lw=1, ls="--")
        axes[2].set_xlabel("patient")
    axes[2].set_title("Perfusion reserve index (stress / rest)", loc="left")
    fig.suptitle("Semi-quantitative perfusion indices, mean of three slices "
                 "(all patients in this dataset have a clinical stress perfusion defect)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "05_semi_quantitative_indices.png")
    return reserve.rename("MPRI").to_frame()


def fig_bullseye(series: list[Series], out: Path) -> None:
    stress = [s for s in series if s.condition == "Stress"]
    patients = sorted({s.patient for s in stress})
    if not patients:
        return
    ncol = min(5, len(patients))
    nrow = int(np.ceil(len(patients) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow + 0.5),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    width = 2 * np.pi / N_SECTORS
    for ax, p in zip(axes.ravel(), patients):
        vals = {s.slice_idx: s.metrics["sector_upslope_ratio"] for s in stress if s.patient == p}
        allv = np.concatenate(list(vals.values()))
        ref = np.nanmax(allv) if np.isfinite(allv).any() else 1.0
        for ring, k in enumerate((0, 1, 2)):  # apical inside, basal outside
            for j, v in enumerate(vals.get(k, [])):
                col = SEQ(float(np.clip(v / ref, 0, 1))) if np.isfinite(v) else "#f2f2f0"
                ax.bar(j * width + width / 2, 1, width=width, bottom=ring + 0.3, color=col,
                       edgecolor="white", linewidth=1.5)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        ax.set_title(f"patient {p}", fontsize=9, pad=2)
    for ax in axes.ravel()[len(patients):]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cb.set_label("stress upslope ratio\n(relative to patient's best sector)")
    fig.suptitle("Stress perfusion by sector. Rings: apical (inner) to basal (outer); 6 sectors per slice, "
                 "starting at the RV insertion point, clockwise as seen in the image", x=0.01, ha="left",
                 fontsize=9.5)
    save(fig, out, "06_stress_sector_map.png")


def fig_motion_cnr(series: list[Series], df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    moving = [s for s in series if np.isfinite(s.metrics["motion_p2p_mm"])] or series
    ex = max(moving, key=lambda s: np.nan_to_num(s.metrics["motion_p2p_mm"]))
    axes[0].plot(ex.times, ex.metrics["motion_trace_mm"], color=COND_COLOR[ex.condition])
    axes[0].set_xlabel("time (s)" if ex.time_unit == "s" else "time (frames)")
    axes[0].set_ylabel("LV displacement (mm)")
    axes[0].set_title(f"Breathing motion, largest case ({ex.key})", loc="left")
    jit = np.random.default_rng(1)
    for ax, metric, label in ((axes[1], "motion_p2p_mm", "Peak-to-peak LV motion (mm)"),
                              (axes[2], "cnr_myo", "Myocardial contrast-to-noise ratio")):
        for x, cond in enumerate(("Rest", "Stress")):
            v = df.loc[df.condition == cond, metric].dropna().values
            if len(v) == 0:
                continue
            ax.boxplot([v], positions=[x], widths=0.45, showfliers=False,
                       medianprops={"color": INK, "lw": 1.5}, boxprops={"color": INK2},
                       whiskerprops={"color": INK2}, capprops={"color": INK2})
            ax.scatter(x + jit.uniform(-0.12, 0.12, len(v)), v, s=14, color=COND_COLOR[cond],
                       alpha=0.85, zorder=3)
        ax.set_xticks([0, 1], ["Rest", "Stress"])
        ax.set_title(label, loc="left")
    fig.suptitle("Acquisition challenges: free-breathing motion and limited contrast-to-noise "
                 "(each dot = one slice series)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "07_motion_and_cnr.png")


def fig_dose_real(dose_df: pd.DataFrame, out: Path) -> None:
    fr = np.array(sorted(dose_df.dose_fraction.unique()))
    g = dose_df.groupby("dose_fraction")
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for col, name, color in (("whole_slice_rel_sd", "whole-slice myocardium", BLUE),
                             ("sector_rel_sd", "single sector (1/6 of a slice)", ORANGE)):
        med = g[col].median().reindex(fr) * 100
        q1, q3 = g[col].quantile(0.25).reindex(fr) * 100, g[col].quantile(0.75).reindex(fr) * 100
        ax.fill_between(fr * 100, q1, q3, color=color, alpha=0.18, lw=0)
        ax.plot(fr * 100, med, "o-", color=color, label=f"{name} (median, IQR band)", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.invert_xaxis()
    ax.set_xlabel("simulated dose (% of the acquired dose, 0.05 mmol/kg)")
    ax.set_ylabel("variability of upslope ratio (%)")
    ax.set_title("Simulated dose reduction on the real curves", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, "08_real_data_dose_reduction.png")


# ================================================================== physics ===
def fig_forward_model(out: Path) -> None:
    aif = gamma_variate_aif(T_FINE)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    axes[0].plot(T_FINE, aif, color=INK2)
    axes[0].set_title("Arterial input function (full dose)", loc="left")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("concentration (mM)")
    for p, name, col in ((HEALTHY, "healthy", BLUE), (ISCHAEMIC, "ischaemic", ORANGE)):
        axes[1].plot(T_FINE, tissue_curve(p, aif), color=col, label=f"{name}: Fp = {p[0]:.1f} ml/min/ml")
    y = tissue_curve(ISCHAEMIC, aif)[ACQ_IDX] + np.random.default_rng(0).normal(0, noise_sd(), len(T_ACQ))
    axes[1].scatter(T_ACQ, y, s=10, color=ORANGE, alpha=0.6, label="ischaemic + noise (SNR 17.5)")
    axes[1].set_title("Myocardial tissue curves from the 2CXM", loc="left")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("concentration (mM)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Physics forward model: tissue curve = AIF convolved with the two-compartment exchange "
                 "impulse response", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "09_physics_forward_model.png")


def fig_low_dose(fits: pd.DataFrame, out: Path) -> None:
    doses = sorted(fits.dose.unique(), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7))
    for x_off, (name, col) in zip((-0.17, 0.17), (("healthy", BLUE), ("ischaemic", ORANGE))):
        data = []
        for d in doses:
            sub = fits[(fits.tissue == name) & (fits.dose == d)]
            data.append(((sub.Fp_est - sub.Fp_true) / sub.Fp_true * 100).dropna().values)
        bp = axes[0].boxplot(data, positions=np.arange(len(doses)) + x_off, widths=0.3, showfliers=False,
                             patch_artist=True, medianprops={"color": INK, "lw": 1.5})
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.75, edgecolor="white")
        axes[0].plot([], [], color=col, lw=6, label=name)
    axes[0].axhline(0, color=INK2, lw=1)
    axes[0].set_xticks(range(len(doses)), [f"{d:.0%}" for d in doses])
    axes[0].set_xlabel("contrast dose (fraction of full dose)")
    axes[0].set_ylabel("error in fitted Fp (%)")
    axes[0].set_title("Blood-flow error grows as the dose drops", loc="left")
    axes[0].legend(loc="upper left")

    # Show a *typical* fit: of 15 noise draws, the one with the median Fp error.
    rng = np.random.default_rng(3)
    shades = {doses[0]: "#0d2f5c", doses[-1]: "#7fb0ea"}
    for d in (doses[0], doses[-1]):
        aif = gamma_variate_aif(T_FINE, d)
        clean = tissue_curve(ISCHAEMIC, aif)[ACQ_IDX]
        draws = []
        for _ in range(15):
            y = clean + rng.normal(0, noise_sd(), len(T_ACQ))
            est, _ = fit_2cxm(y, aif)
            if np.isfinite(est[0]):
                draws.append((abs(est[0] - ISCHAEMIC[0]), y, est))
        draws.sort(key=lambda r: r[0])
        _, y, est = draws[len(draws) // 2]
        axes[1].scatter(T_ACQ, y, s=9, color=shades[d], alpha=0.55)
        axes[1].plot(T_FINE, tissue_curve(est, aif), color=shades[d],
                     label=f"{d:.0%} dose: fitted Fp = {est[0]:.2f} (true {ISCHAEMIC[0]:.2f})")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("concentration (mM)")
    axes[1].set_title("Same noise, less signal: typical (median-error) fit, ischaemic tissue", loc="left")
    axes[1].legend(fontsize=8)
    fig.suptitle("Two-compartment exchange model fitted by non-linear least squares (NLLS) under dose reduction",
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "10_low_dose_parameter_error.png")


def fig_dro(dro: dict, out: Path) -> None:
    true = dro["true_fp"]
    doses = sorted(dro["maps"], reverse=True)
    lo = doses[-1]
    est_lo, sd_lo = dro["maps"][lo]
    panels = [(true, "true Fp", SEQ, (0, 4.2))]
    panels += [(dro["maps"][d][0], f"NLLS Fp, {d:.0%} dose", SEQ, (0, 4.2)) for d in doses]
    panels.append(((est_lo - true) / true * 100, f"error at {lo:.0%} dose (%)", DIV, (-60, 60)))
    panels.append((sd_lo / true * 100, f"fitted SD at {lo:.0%} dose (%, capped)", SEQ, (0, 150)))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.3))
    for ax, (img, title, cmap, (vmin, vmax)) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel(f"Fp {dro['fp_values'][0]} to {dro['fp_values'][-1]} ml/min/ml")
    axes[0].set_xlabel(f"PS {dro['ps_values'][0]} to {dro['ps_values'][-1]}")
    fig.suptitle("Digital reference object (12 x 12 parameter combinations): Fp maps degrade at low dose",
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "11_dro_parameter_maps.png")


def fig_calibration(dro: dict, cal: pd.DataFrame, out: Path) -> None:
    doses = sorted(dro["maps"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    zz = np.linspace(-5, 5, 300)
    axes[0].plot(zz, np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi), color=INK2, lw=1.5, ls="--", label="ideal N(0, 1)")
    for d, col in zip(doses, [SEQ(v) for v in np.linspace(0.95, 0.5, len(doses))]):
        est, sd = dro["maps"][d]
        z = ((est - dro["true_fp"]) / sd).ravel()
        z = z[np.isfinite(z)]
        axes[0].hist(np.clip(z, -5, 5), bins=40, density=True, histtype="step", lw=2, color=col,
                     label=f"{d:.0%} dose")
    axes[0].set_xlabel("standardised error = (estimate - truth) / fitted SD")
    axes[0].set_ylabel("density")
    axes[0].set_title("Errors relative to the fit's own uncertainty", loc="left")
    axes[0].legend(fontsize=8)

    for source, col in (("healthy", BLUE), ("ischaemic", ORANGE), ("DRO", AQUA)):
        c = cal[cal.source == source].sort_values("dose")
        if len(c):
            axes[1].plot(c.dose * 100, c.coverage_95 * 100, "o-", color=col, markersize=8,
                         markeredgecolor="white", markeredgewidth=1.5, label=source)
    axes[1].axhline(95, color=INK2, lw=1, ls="--", label="nominal 95%")
    axes[1].invert_xaxis()
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set_xlabel("contrast dose (% of full)")
    axes[1].set_ylabel("true Fp inside the 95% interval (%)")
    axes[1].set_title("Coverage of the NLLS confidence interval", loc="left")
    fig.suptitle("Is the uncertainty trustworthy? Linearised NLLS error bars vs. actual errors",
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "12_uncertainty_calibration.png")
