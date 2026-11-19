"""End-to-end run on a synthetic dataset laid out like the real one."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.make_synthetic_dataset import main as make_data  # noqa: E402
from src.io_utils import load_dataset  # noqa: E402


def test_loader_separates_slices_and_drops_pd(tmp_path):
    make_data(tmp_path)
    series = load_dataset(tmp_path)
    assert [(s.patient, s.condition) for s in series][::3] == [
        (1, "Rest"), (1, "Stress"), (2, "Rest"), (2, "Stress"), (10, "Rest"), (10, "Stress")]
    assert len(series) == 18                      # 3 patients x 2 conditions x 3 slices
    assert {s.n_pd_skipped for s in series} == {2}
    assert {len(s.times) for s in series} == {45, 60}
    assert all(s.rv_dir[0] is not None and s.endo[0] is not None for s in series)


def test_full_run(tmp_path):
    make_data(tmp_path)
    fig, res = tmp_path / "figures", tmp_path / "results"
    subprocess.run([sys.executable, str(ROOT / "run_eda.py"), "--data-dir", str(tmp_path),
                    "--figures", str(fig), "--results", str(res), "--quick", "--fit-reps", "5",
                    "--dro-block", "1", "--no-readme-update"], check=True)
    assert len(list(fig.glob("*.png"))) == 12
    df = pd.read_csv(res / "series_metrics.csv")
    by_cond = df.groupby("condition").upslope_ratio.median()
    assert by_cond["Stress"] > by_cond["Rest"]     # synthetic stress flow is 2x rest
    sec = pd.read_csv(res / "sector_metrics.csv")
    stress = sec[sec.condition == "Stress"].groupby("sector").upslope_ratio.median()
    assert stress.idxmin() == 1                     # the simulated defect sits in sector 1


def test_bad_series_are_skipped_not_fatal(tmp_path):
    make_data(tmp_path)
    root = tmp_path / "myoperfdata"
    # 1) a study whose segmentation files point to images that do not exist
    for f in (root / "Study2_Stress").glob("data*.png"):
        f.rename(f.with_name("img" + f.name))
    # 2) a series whose contours have no endocardium (only one closed section per frame)
    for seg in root.glob("Study10_Rest/segment1-*.set"):
        txt = seg.read_text()
        parts = txt.split('<section color="red"')
        txt = parts[0] + "".join("<!--" + p.split("</section>", 1)[0] + "-->" + p.split("</section>", 1)[1]
                                 for p in parts[1:])
        seg.write_text(txt)
    fig, res = tmp_path / "figures", tmp_path / "results"
    out = subprocess.run([sys.executable, str(ROOT / "run_eda.py"), "--data-dir", str(tmp_path),
                          "--figures", str(fig), "--results", str(res), "--quick", "--fit-reps", "3",
                          "--dro-block", "1", "--no-readme-update"],
                         check=True, capture_output=True, text=True).stdout
    assert "Study2_Stress/segment0-dcm.set: only 0 images found" in out
    assert "P10-Rest-mid" in out and "no endocardium" in out
    assert len(list(fig.glob("*.png"))) == 12
    assert len(pd.read_csv(res / "series_metrics.csv")) == 14   # 18 - 3 missing - 1 without endo


def _rewrite_sets(root, fn):
    for f in root.rglob("segment*.set"):
        f.write_text(fn(f.read_text()))


def test_contour_layout_variants_and_4_digit_names(tmp_path):
    """Contours nested in a wrapper, inside the star, or with upper-case tags; images
    numbered data0001 while the segmentation says data001."""
    from src.io_utils import parse_workset
    variants = {
        "wrapped": lambda t: t.replace("<section", "<contours><section").replace("</section>", "</section></contours>"),
        "in_star": lambda t: t.replace("    </star>\n", "").replace("  </frame>", "    </star>\n  </frame>"),
        "upper": lambda t: t.replace("<section", "<Section").replace("</section>", "</Section>")
                            .replace("<point", "<Point"),
    }
    for name, fn in variants.items():
        d = tmp_path / name
        make_data(d)
        _rewrite_sets(d, fn)
        frames = parse_workset(d / "myoperfdata" / "Study1_Rest" / "segment1-dcm.set")
        assert all(f["epi"] is not None and f["endo"] is not None for f in frames), name
        assert frames[0]["rv_dir"] is not None, name

    d = tmp_path / "digits"
    make_data(d)
    for f in (d / "myoperfdata" / "Study2_Stress").glob("data*.png"):
        f.rename(f.with_name(f"data{int(f.stem[4:]):04d}.png"))
    series = load_dataset(d)
    assert len(series) == 18


def test_empty_template_is_not_chosen(tmp_path):
    """segmentN.set without contours must lose against the annotated segmentN-dcm.set."""
    make_data(tmp_path)
    series = load_dataset(tmp_path)
    assert {Path(s.source).name.split("-")[-1] for s in series} == {"dcm.set"}
    assert all(s.epi[0] is not None for s in series)


def test_slice_without_any_contours_is_reported(tmp_path, capsys):
    make_data(tmp_path)
    for f in (tmp_path / "myoperfdata" / "Study1_Rest").glob("segment2-*.set"):
        f.unlink()
    series = load_dataset(tmp_path)
    assert len(series) == 17
    assert "Study1_Rest slice 2: no manual contours" in capsys.readouterr().out


def test_partial_endocardium_is_filled_and_static_contours_flagged(tmp_path):
    import re
    from src.curves import compute_metrics, extract_curves
    make_data(tmp_path)
    root = tmp_path / "myoperfdata"
    # Study2_Stress slice 0: drop the endocardium in frames 3-14 (as in the real Study9_Stress)
    for seg in root.glob("Study2_Stress/segment0-*.set"):
        frames = seg.read_text().split("  <frame ")
        for i in range(3, 15):
            frames[i + 1] = re.sub(r'<section color="red".*?</section>', "", frames[i + 1], flags=re.S)
        seg.write_text("  <frame ".join(frames))
    # Study1_Rest slice 1: the same contour copied to every frame (as in the real Study6_Rest)
    for seg in root.glob("Study1_Rest/segment1-*.set"):
        frames = seg.read_text().split("  <frame ")
        inner = frames[1].split(">", 1)[1].split("</frame>", 1)[0]
        frames[1:] = [f.split(">", 1)[0] + ">" + inner + "</frame>" + f.split("</frame>", 1)[1]
                      for f in frames[1:]]
        seg.write_text("  <frame ".join(frames))
    series = {s.key: s for s in load_dataset(tmp_path)}
    s = series["P02-Stress-apical"]
    extract_curves(s)
    assert s.curves["n_borrowed_endo"] == 12 and s.curves["n_valid"] == len(s.times)
    st = series["P01-Rest-mid"]
    extract_curves(st)
    compute_metrics(st)
    assert st.metrics["static_contours"] and np.isnan(st.metrics["motion_p2p_mm"])
