"""Create a small synthetic dataset with the same layout as the real one, for testing.

    python tests/make_synthetic_dataset.py /tmp/synthetic

Mimics: interleaved slices (data001.png = slice 0 / t 0, data002.png = slice 1 / t 0, ...),
an empty segmentN.set template per slice (frame list, no contours) next to annotated
segmentN-dcm.set (references .dcm names; the loader must fall back to .png) and
segmentN-png.set files, two bright proton-density frames per slice, a "star"
with unit direction vectors, breathing motion, and a perfusion defect in one sector
under stress. Study10 is included to check numeric (not alphabetical) ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

H, W = 192, 256
N_PD = 2


def gamma(t, t0, a=3.0, b=1.6):
    tt = np.clip(t - t0, 0, None)
    return (tt / (a * b)) ** a * np.exp(a - tt / b)


def make_study(folder: Path, n_t: int, stress: bool, rng) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_t, dtype=float)
    aif = gamma(t, 8) + 0.1 * (1 - np.exp(-np.clip(t - 8, 0, None) / 6))
    flow = 2.0 if stress else 1.0
    myo = np.convolve(aif, flow * 0.12 * np.exp(-t / 12.0))[:n_t]
    yy, xx = np.mgrid[0:H, 0:W]
    centres = {0: (130.0, 96.0), 1: (128.0, 100.0), 2: (126.0, 104.0)}
    r_endo, r_epi = {0: 10, 1: 13, 2: 15}, {0: 17, 1: 21, 2: 23}
    rv_dir = np.array([-0.8, -0.6])
    sets = {k: [] for k in range(3)}
    frames_total = N_PD + n_t
    for f in range(frames_total):
        for k in range(3):
            dy = 6.0 * np.sin(2 * np.pi * f / 5.3)  # breathing, ~12 mm peak-to-peak
            cx, cy = centres[k][0], centres[k][1] + dy
            r = np.hypot(xx - cx, yy - cy)
            ang = np.mod(np.arctan2(yy - cy, xx - cx) - np.arctan2(rv_dir[1], rv_dir[0]), 2 * np.pi)
            img = np.full((H, W), 25.0)
            img[:, :20] = 160.0  # bright chest wall, the old code's "blood pool"
            if f < N_PD:
                img *= 3.5
                img[r < r_epi[k]] = 200
            else:
                ti = f - N_PD
                img[r < r_endo[k]] = 40 + 160 * aif[ti]
                ring = (r >= r_endo[k]) & (r < r_epi[k])
                defect = ring & (ang < 2 * np.pi / 6) & stress  # sector 1 under-perfused at stress
                img[ring] = 35 + 160 * myo[ti]
                img[defect] = 35 + 160 * myo[ti] * 0.4
            img += rng.normal(0, 4, img.shape)
            n = f * 3 + k + 1
            Image.fromarray(np.clip(img * 0.45, 0, 255).astype(np.uint8)).save(folder / f"data{n:03d}.png")
            th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            sets[k].append((f"data{n:03d}.dcm", cx, cy, r_endo[k], r_epi[k], th))
    for k, frames in sets.items():
        xml = ['<?xml version="1.0"?>', '<workset version="2">',
               '  <description><RVpeak value="12"/><LVpeak value="14"/></description>']
        for name, cx, cy, re_, rp, th in frames:
            xml.append(f'  <frame image="{name}">')
            xml.append(f'    <star x="{cx:.2f}" y="{cy:.2f}" r="{rp + 4}">')
            for a in (0, 2.0944, 4.1888):
                v = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ rv_dir
                xml.append(f'      <point x="{v[0]:.4f}" y="{v[1]:.4f}"/>')
            xml.append('    </star>')
            for rad, color in ((rp, "white"), (re_, "red")):
                xml.append(f'    <section color="{color}" open="0">')
                for a in th:
                    xml.append(f'      <point x="{cx + rad * np.cos(a):.2f}" y="{cy + rad * np.sin(a):.2f}"/>')
                xml.append('    </section>')
            xml.append('  </frame>')
        xml.append('</workset>')
        (folder / f"segment{k}-dcm.set").write_text("\n".join(xml))
        (folder / f"segment{k}-png.set").write_text("\n".join(xml).replace(".dcm", ".png"))
        # as in the real data: an empty template that only lists the frames, no contours
        empty = ['<?xml version="1.0" encoding="UTF-8"?>', "<workset>", "  <description>",
                 '    <RVpeak value="-1"/>', '    <LVpeak value="-1"/>', "  </description>"]
        empty += [f'  <frame image="{name.replace(".dcm", ".png")}"/>' for name, *_ in frames]
        (folder / f"segment{k}.set").write_text("\n".join(empty + ["</workset>"]))


def main(out: Path) -> None:
    rng = np.random.default_rng(0)
    root = out / "myoperfdata"
    for p, n_t in ((1, 45), (2, 60), (10, 60)):
        for cond in ("Rest", "Stress"):
            make_study(root / f"Study{p}_{cond}", n_t, cond == "Stress", rng)
    print(f"synthetic dataset written to {root}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data"))
