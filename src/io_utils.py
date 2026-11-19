"""Load the Wollny & Kellman perfusion dataset slice by slice.

Layout (as distributed):
    data/myoperfdata/StudyN_Rest|Stress/dataNNN.dcm | .png   (all 3 slices, interleaved)
    data/myoperfdata/StudyN_Rest|Stress/segmentS[-dcm|-png].set

Each ``segmentS*.set`` file is a MIA/tabseg ``<workset>`` XML that lists, frame by
frame, the image belonging to slice S plus manual endo-/epicardial contours. We
therefore load exactly the images a segmentation file names - never "all images
in the folder", which would interleave the three slices.

This module only reads files; it never writes to the data folder.
"""
from __future__ import annotations

import re
from collections import Counter
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

try:
    import pydicom
except ImportError:  # PNG fallback still works
    pydicom = None

SLICE_NAMES = {0: "apical", 1: "mid", 2: "basal"}  # numbering used by the dataset
STUDY_RE = re.compile(r"^study[_\- ]?(\d+)[_\- ]?(rest|stress)$", re.I)
# Pixel spacing reported in the dataset paper; used when no DICOM header is available.
SPACING_FALLBACK_MM = {6: 1.6, 7: 1.6, 8: 1.6}
DEFAULT_SPACING_MM = 1.4


@dataclass
class Series:
    """One slice of one study: a (T, H, W) image series with per-frame contours."""

    patient: int
    condition: str            # "Rest" | "Stress"
    slice_idx: int            # 0 apical, 1 mid, 2 basal
    images: np.ndarray        # (T, H, W) float32
    times: np.ndarray         # (T,) seconds, or frame index if no timing available
    time_unit: str            # "s" | "frame"
    spacing_mm: float
    endo: list                # per frame: (N, 2) array of (x, y), or None
    epi: list
    centre: list              # per frame: (x, y) of the segmentation "star", or None
    rv_dir: list              # per frame: unit vector towards the RV insertion point, or None
    image_format: str
    source: str               # segmentation file it was loaded from (study/file)
    source_path: str = ""     # full path, for diagnostics
    n_pd_skipped: int = 0
    curves: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def slice_name(self) -> str:
        return SLICE_NAMES.get(self.slice_idx, str(self.slice_idx))

    @property
    def key(self) -> str:
        return f"P{self.patient:02d}-{self.condition}-{self.slice_name}"


# ------------------------------------------------------------------ contours ---
def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def split_contours(polys: list[np.ndarray]):
    """Largest closed contour = epicardium; the largest one inside it = endocardium."""
    if not polys:
        return None, None
    areas = [abs(polygon_area(p)) for p in polys]
    order = np.argsort(areas)[::-1]
    epi = polys[order[0]]
    epi_path = MplPath(epi)
    for i in order[1:]:
        if areas[i] < 0.95 * areas[order[0]] and epi_path.contains_point(polys[i].mean(0)):
            return polys[i], epi
    return None, epi


def _xy(el) -> tuple[float, float] | None:
    """(x, y) of a point element: x/y attributes (any case) or <x>/<y> child elements."""
    a = {k.lower(): v for k, v in el.attrib.items()}
    x, y = a.get("x"), a.get("y")
    if x is None or y is None:
        kids = {c.tag.lower(): (c.text or "").strip() for c in el}
        x, y = kids.get("x"), kids.get("y")
    try:
        return (float(x), float(y)) if x is not None and y is not None else None
    except ValueError:
        return None


def _star(fr):
    for el in fr.iter():
        if el.tag.lower() == "star":
            return el
    return None


def _contours(fr, skip: set) -> list[np.ndarray]:
    """Every element (any tag, any depth) whose children are >= 3 points is one contour."""
    polys = []
    for el in fr.iter():
        if el is fr or el in skip:
            continue
        pts = [p for p in (_xy(c) for c in el) if p is not None]
        if len(pts) >= 3:
            polys.append(np.asarray(pts))
    return polys


def parse_workset(path: Path) -> list[dict]:
    """Parse a ``<workset>`` segmentation file into a list of per-frame dicts.

    Each ``<frame image=...>`` holds a ``<star x y r>`` (LV centre; its first point is a
    direction vector towards the RV insertion) and closed contours (endo-/epicardium).
    Contours are found by structure rather than tag name: any element inside the frame,
    other than the star, whose children are three or more x/y points. This keeps the star
    vectors out of the contours and tolerates differences between versions of the files.
    """
    root = ET.parse(path).getroot()
    frames = []
    for fr in (el for el in root.iter() if el.tag.lower() == "frame"):
        centre = rv = None
        star = _star(fr)
        skip = {star} if star is not None else set()  # the star's own points are not a contour
        if star is not None and _xy(star) is not None:
            centre = np.array(_xy(star))
            pts = [p for p in (_xy(c) for c in star) if p is not None]
            if pts:
                p = np.array(pts[0])
                v = p if np.linalg.norm(p) <= 1.5 else p - centre  # direction or absolute point
                rv = v / (np.linalg.norm(v) + 1e-9)
        endo, epi = split_contours(_contours(fr, skip))
        image = next((v for k, v in fr.attrib.items() if k.lower() == "image"), None)
        frames.append({"image": image, "centre": centre, "rv_dir": rv, "endo": endo, "epi": epi})
    return frames


def describe_first_frame(path: Path, max_lines: int = 12) -> str:
    """Compact tag outline of the first <frame>, printed when no contours could be read."""
    root = ET.parse(path).getroot()
    fr = next((el for el in root.iter() if el.tag.lower() == "frame"), None)
    if fr is None:
        return "no <frame> element found"
    lines = []

    def walk(el, depth):
        if len(lines) >= max_lines:
            return
        kids = list(el)
        attrs = " ".join(f'{k}="{v[:12]}"' for k, v in list(el.attrib.items())[:4])
        same = f" x{len(kids)} children <{kids[0].tag}>" if kids else ""
        lines.append("      " + "  " * depth + f"<{el.tag} {attrs}>{same}".replace(" >", ">"))
        seen = set()
        for c in kids:
            if c.tag not in seen:
                seen.add(c.tag)
                walk(c, depth + 1)

    walk(fr, 0)
    return "\n".join(lines)


# -------------------------------------------------------------------- images ---
class ImageResolver:
    """Finds the image a segmentation frame refers to (DICOM preferred, PNG fallback)."""

    def __init__(self, study_dir: Path):
        self.study_dir = study_dir
        self.exts = ([".dcm"] if pydicom else []) + [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        self._index = None

    @staticmethod
    def _key(stem: str):
        """'data001' and 'data0001' share the key ('data', 1)."""
        m = re.match(r"^(.*?)(\d+)$", stem.lower())
        return (m.group(1), int(m.group(2))) if m else (stem.lower(), None)

    def _build_index(self):
        idx: dict[tuple, list[Path]] = {}
        for p in self.study_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.exts:
                idx.setdefault(self._key(p.stem), []).append(p)
        return idx

    def resolve(self, ref: str, set_dir: Path) -> Path | None:
        ref_p = Path(ref.replace("\\", "/"))
        base = set_dir / ref_p.parent
        for ext in self.exts:
            for e in (ext, ext.upper()):
                cand = base / (ref_p.stem + e)
                if cand.is_file():
                    return cand
        if self._index is None:
            self._index = self._build_index()
        hits = self._index.get(self._key(ref_p.stem), [])
        if not hits:
            return None
        pref = {e: i for i, e in enumerate(self.exts)}
        hits = sorted(hits, key=lambda p: (0 if set_dir in p.parents else 1,
                                           pref.get(p.suffix.lower(), 99), len(p.parts)))
        return hits[0]


def dicom_seconds(ds) -> float | None:
    for key in ("AcquisitionTime", "ContentTime"):
        v = getattr(ds, key, None)
        if v:
            s = str(v)
            try:
                return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + (float(s[4:]) if len(s) > 4 else 0.0)
            except ValueError:
                continue
    return None


def read_image(path: Path):
    """Return (image, acquisition time in s or None, pixel spacing in mm or None)."""
    if path.suffix.lower() == ".dcm":
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:  # not expected for this dataset, but never mix slices silently
            raise ValueError(f"{path.name}: multi-frame DICOM is not supported")
        arr = arr * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        sp = getattr(ds, "PixelSpacing", None)
        return arr, dicom_seconds(ds), (float(sp[0]) if sp else None)
    im = Image.open(path)
    if im.mode not in ("L", "I", "I;16", "F"):
        im = im.convert("L")
    return np.asarray(im, dtype=np.float32), None, None


def detect_pd_frames(images: np.ndarray, max_pd: int = 3) -> int:
    """Count the leading proton-density reference frames (much brighter than the rest)."""
    m = images.reshape(len(images), -1).mean(1)
    if len(m) < max_pd + 3:
        return 0
    ref = np.median(m[max_pd:max_pd + 5])
    n = 0
    while n < max_pd and m[n] > 1.5 * ref:
        n += 1
    return n


# ----------------------------------------------------------------- discovery ---
def find_data_root(data_dir: Path) -> Path:
    """Accept either data/ or data/myoperfdata/ (or any parent of the Study* folders)."""
    for cand in (data_dir / "myoperfdata", data_dir):
        if cand.is_dir() and any(STUDY_RE.match(p.name) for p in cand.iterdir() if p.is_dir()):
            return cand
    return data_dir


def discover_studies(root: Path):
    """Return [(patient, condition, folder)] sorted numerically (1, 2, ..., 10)."""
    found = {}
    for d in sorted(root.rglob("*")):
        if d.is_dir() and ".git" not in d.parts:
            m = STUDY_RE.match(d.name)
            if m:
                key = (int(m.group(1)), m.group(2).capitalize())
                if key not in found or len(d.parts) < len(found[key].parts):
                    found[key] = d
    return [(p, c, d) for (p, c), d in sorted(found.items())]


def find_worksets(study_dir: Path) -> dict[int, list[Path]]:
    """Map slice index -> all segmentation files for that slice.

    The dataset ships several ``<workset>`` files per slice (e.g. ``segment1.set``,
    ``segment1-dcm.set``, ``segment1-png.set``); some are empty frame lists without
    contours, so the choice is made later by ``choose_workset``.
    """
    sets = []
    for p in sorted(study_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".set", ".xml"):
            try:
                if b"<workset" in p.read_bytes()[:4096]:
                    sets.append(p)
            except OSError:
                continue
    by_slice: dict[int, list[Path]] = {}
    for i, p in enumerate(sorted(sets, key=lambda q: (len(q.name), q.name))):
        digits = re.findall(r"\d+", p.stem)
        by_slice.setdefault(int(digits[0]) if digits else i, []).append(p)
    return dict(sorted(by_slice.items()))


def choose_workset(candidates: list[Path]):
    """Pick the file with the most annotated frames. Returns (path, frames, {name: n_annotated})."""
    best, counts = None, {}
    for p in candidates:
        try:
            frames = parse_workset(p)
        except ET.ParseError:
            counts[p.name] = "unreadable"
            continue
        n = sum(f["epi"] is not None for f in frames)
        counts[p.name] = n
        if best is None or n > best[2]:
            best = (p, frames, n)
    if best is None:
        return None, [], counts
    return best[0], best[1], counts


def load_series(set_path: Path, resolver: ImageResolver, patient: int, condition: str,
                slice_idx: int, skip_pd: str = "auto", log=print, frames: list | None = None) -> Series | None:
    frames = parse_workset(set_path) if frames is None else frames
    imgs, times, keep, spacing, fmts, missing = [], [], [], None, set(), 0
    for fr in frames:
        if not fr["image"]:
            continue
        p = resolver.resolve(fr["image"], set_path.parent)
        if p is None:
            missing += 1
            continue
        arr, t, sp = read_image(p)
        if imgs and arr.shape != imgs[0].shape:
            missing += 1
            continue
        imgs.append(arr)
        times.append(t)
        keep.append(fr)
        fmts.add(p.suffix.lower().lstrip("."))
        spacing = spacing or sp
    name = f"{set_path.parent.name}/{set_path.name}"
    if len(imgs) < 10:
        refs = [f["image"] for f in frames if f["image"]][:2]
        present = sorted(p.name for p in set_path.parent.iterdir() if p.is_file())
        imgs_present = [n for n in present if Path(n).suffix.lower() in (".dcm", ".png", ".jpg")]
        log(f"  ! {name}: only {len(imgs)} images found ({missing} missing) - skipped")
        log(f"      segmentation refers to e.g. {refs}; folder holds {len(imgs_present)} image files, "
            f"e.g. {imgs_present[:3]}")
        return None
    if missing:
        log(f"  ! {name}: {missing} referenced images missing or mismatched")
    images = np.stack(imgs)

    if all(t is not None for t in times) and np.all(np.diff(times) > 0):
        t_arr, unit = np.asarray(times, float) - times[0], "s"
    else:
        t_arr, unit = np.arange(len(images), dtype=float), "frame"

    n_pd = detect_pd_frames(images) if skip_pd == "auto" else int(skip_pd)
    images, t_arr, keep = images[n_pd:], t_arr[n_pd:] - t_arr[n_pd], keep[n_pd:]

    return Series(
        patient=patient, condition=condition, slice_idx=slice_idx, images=images, times=t_arr,
        time_unit=unit, spacing_mm=spacing or SPACING_FALLBACK_MM.get(patient, DEFAULT_SPACING_MM),
        endo=[f["endo"] for f in keep], epi=[f["epi"] for f in keep],
        centre=[f["centre"] for f in keep], rv_dir=[f["rv_dir"] for f in keep],
        image_format="/".join(sorted(fmts)), source=name, source_path=str(set_path), n_pd_skipped=n_pd,
    )


def load_dataset(data_dir: Path, skip_pd: str = "auto", log=print) -> list[Series]:
    root = find_data_root(data_dir)
    series = []
    for patient, condition, d in discover_studies(root):
        resolver = ImageResolver(d)
        sets = find_worksets(d)
        if not sets:
            log(f"  ! {d.name}: no segmentation (<workset>) files found")
        for slice_idx, candidates in sets.items():
            set_path, frames, counts = choose_workset(candidates)
            if set_path is None or counts.get(set_path.name, 0) == 0:
                log(f"  ! {d.name} slice {slice_idx}: no manual contours in any segmentation file "
                    f"({', '.join(f'{k}: {v} annotated frames' for k, v in counts.items())}) - skipped")
                continue
            s = load_series(set_path, resolver, patient, condition, slice_idx, skip_pd, log, frames)
            if s is not None:
                series.append(s)
    used = Counter(re.sub(r"\d+", "N", Path(s.source).name) for s in series)
    if used:
        log("  segmentation files used: " + ", ".join(f"{k} x{v}" for k, v in used.most_common()))
    return series
