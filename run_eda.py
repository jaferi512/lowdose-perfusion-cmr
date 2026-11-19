"""Exploratory analysis of first-pass myocardial perfusion MRI and a physics-based
low-dose simulation. Runs on a CPU.

    python run_eda.py               # real data + physics (a few minutes)
    python run_eda.py --quick       # fewer noise repetitions (~1 min)
    python run_eda.py --sim-only    # physics only, no dataset needed

Writes figures/01..12, results/*.csv, results/summary.md and the results section of
README.md. The data folder is only read, never modified.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import plotting as P  # noqa: E402
from src.curves import N_SECTORS, compute_metrics, curves_ok, extract_curves  # noqa: E402
from src.dose_sim import simulate_dose_reduction  # noqa: E402
from src.dro_phantom import calibration_table, make_dro, repeated_fits  # noqa: E402
from src.io_utils import describe_first_frame, load_dataset, pydicom  # noqa: E402
from src.report import build_summary, update_readme, write_summary  # noqa: E402

SCALARS = ["upslope_blood", "upslope_myo", "upslope_ratio", "pe_blood", "pe_myo", "pe_ratio",
           "ttp_blood", "ttp_myo", "cnr_myo", "noise_blood", "noise_myo", "motion_p2p_mm"]


def run_real(args, rng) -> dict | None:
    data = args.data_dir.resolve()
    if not data.exists():
        print(f"\n[real data] {data} not found - run `python download_data.py` first. Skipping.")
        return None
    print(f"\n[real data] reading {data}")
    series = load_dataset(data, args.skip_pd)
    if not series:
        print("  no usable series found (expects Study*_Rest / Study*_Stress folders with segment*.set files)")
        return None
    print(f"  {len(series)} slice series; proton-density frames removed: "
          f"{sorted({s.n_pd_skipped for s in series})}; time axis: {sorted({s.time_unit for s in series})}")
    usable, described = [], False
    for s in series:
        extract_curves(s)
        ok, why = curves_ok(s)
        if not ok:
            print(f"  ! {s.key} ({s.source}): {why} - skipped")
            if not described and s.source_path:  # show the file structure once, to diagnose
                print("    structure of the first <frame> in this file:")
                print(describe_first_frame(Path(s.source_path)))
                described = True
            continue
        try:
            compute_metrics(s)
        except (ValueError, IndexError) as exc:
            print(f"  ! {s.key} ({s.source}): metrics failed ({exc}) - skipped")
            continue
        if not np.isfinite(s.metrics.get("upslope_ratio", np.nan)):
            print(f"  ! {s.key} ({s.source}): no valid upslope - skipped")
            continue
        usable.append(s)
    print(f"  {len(usable)} of {len(series)} series usable for analysis")
    borrowed = [f"{s.key} ({s.curves['n_borrowed_endo']} frames)" for s in usable if s.curves["n_borrowed_endo"]]
    if borrowed:
        print("  note: endocardium copied from the nearest annotated frame for " + ", ".join(borrowed))
    static = [s.key for s in usable if s.metrics.get("static_contours")]
    if static:
        print("  note: contours identical in every frame (motion not measurable): " + ", ".join(static))
    if not usable:
        return None
    series = usable

    out, res = args.figures, args.results
    df = pd.DataFrame([{"series": s.key, "patient": s.patient, "condition": s.condition,
                        "slice": s.slice_name, "n_frames": len(s.times), "time_unit": s.time_unit,
                        "spacing_mm": s.spacing_mm, "format": s.image_format, "source": s.source,
                        "endo_frames_copied": s.curves["n_borrowed_endo"],
                        "static_contours": s.metrics["static_contours"],
                        **{k: s.metrics[k] for k in SCALARS}} for s in series])
    df.to_csv(res / "series_metrics.csv", index=False)
    pd.DataFrame([{"series": s.key, "patient": s.patient, "condition": s.condition, "slice": s.slice_name,
                   "sector": k + 1, "upslope_ratio": s.metrics["sector_upslope_ratio"][k]}
                  for s in series for k in range(N_SECTORS)]).to_csv(res / "sector_metrics.csv", index=False)

    counts = df.groupby("patient").size()
    example_patient = int(counts.idxmax())
    pool = [s for s in series if s.patient == example_patient]
    example = next((s for s in pool if s.condition == "Stress" and s.slice_idx == 1), pool[0])
    mid = 1 if any(s.slice_idx == 1 for s in series) else series[0].slice_idx

    P.fig_overview(series, out)
    P.fig_montage(example, out)
    P.fig_curves_patient(series, example.patient, out)
    P.fig_rest_stress_all(series, out, slice_idx=mid)
    mpri = P.fig_semiquant(df, out)
    mpri.to_csv(res / "perfusion_reserve.csv")
    if any(s.curves["has_rv"] for s in series):
        P.fig_bullseye(series, out)
    else:
        print("  (no RV insertion points in the segmentations - sector map skipped)")
    P.fig_motion_cnr(series, df, out)

    n_rep = 40 if args.quick else 150
    print(f"  dose-reduction simulation on the real curves ({n_rep} noise draws per dose)...")
    rows, t0 = [], time.time()
    for i, s in enumerate(series, 1):
        try:
            rows += simulate_dose_reduction(s, [1.0, 0.75, 0.5, 0.25, 0.1], n_rep, rng)
        except Exception as exc:  # one odd series should not stop the whole run
            print(f"  ! dose simulation failed for {s.key}: {type(exc).__name__}: {exc}")
        if i % 10 == 0 or i == len(series):
            print(f"    {i}/{len(series)} series ({time.time() - t0:.0f} s)")
    dose = pd.DataFrame(rows)
    dose.to_csv(res / "dose_reduction_real.csv", index=False)
    P.fig_dose_real(dose, out)
    return {"df": df, "mpri": mpri, "dose": dose}


def run_physics(args, rng) -> dict:
    print("\n[physics] two-compartment exchange model")
    out, res = args.figures, args.results
    P.fig_forward_model(out)
    t0 = time.time()
    n_rep = args.fit_reps or (25 if args.quick else 100)
    fits = repeated_fits([1.0, 0.5, 0.25, 0.1], n_rep=n_rep, rng=rng)
    fits.to_csv(res / "physics_repeated_fits.csv", index=False)
    print(f"  {len(fits)} repeated fits in {time.time() - t0:.0f} s")
    P.fig_low_dose(fits, out)
    t0 = time.time()
    dro = make_dro([1.0, 0.25], block=args.dro_block or (2 if args.quick else 3), rng=rng)
    print(f"  digital reference object fitted in {time.time() - t0:.0f} s")
    P.fig_dro(dro, out)
    cal = calibration_table(fits, dro)
    cal.to_csv(res / "physics_uncertainty_calibration.csv", index=False)
    P.fig_calibration(dro, cal, out)
    return {"fits": fits, "calibration": cal}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data",
                    help="folder containing myoperfdata/ (or the Study* folders directly)")
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--skip-pd", default="auto", help="proton-density frames to drop per series, or 'auto'")
    ap.add_argument("--quick", action="store_true", help="fewer repetitions (faster, noisier statistics)")
    ap.add_argument("--sim-only", action="store_true", help="physics part only")
    ap.add_argument("--fit-reps", type=int, default=0, help="override: repeated fits per dose and tissue")
    ap.add_argument("--dro-block", type=int, default=0, help="override: voxels per DRO block edge")
    ap.add_argument("--no-readme-update", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P.setup_style()
    rng_real = np.random.default_rng(args.seed)       # separate streams: physics results do not
    rng_phys = np.random.default_rng(args.seed + 1)   # change when the real-data part changes
    args.figures.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    if pydicom is None:
        print("note: pydicom not installed - PNG images are used and time is counted in frames.")

    t0 = time.time()
    real = None if args.sim_only else run_real(args, rng_real)
    phys = run_physics(args, rng_phys)
    text = build_summary(real, phys)
    write_summary(text, args.results)
    if not args.no_readme_update:
        update_readme(text, ROOT / "README.md")
    print(f"\nDone in {time.time() - t0:.0f} s -> {args.figures}, {args.results}")


if __name__ == "__main__":
    main()
