"""Download (and unpack) the Wollny & Kellman myocardial perfusion dataset into ./data/.

    python download_data.py              # download if missing, then unpack if not unpacked yet
    python download_data.py --no-extract # download only

Safe by design: an existing archive is never re-downloaded or overwritten, nothing is
deleted, and the archive is only unpacked if no Study* folders exist yet. Archive
entries that would land outside data/ (absolute paths, "..", links) are refused.

Dataset: G. Wollny, P. Kellman. Free breathing myocardial perfusion data sets for
performance analysis of motion compensation algorithms. GigaScience 3:23, 2014.
https://doi.org/10.1186/2047-217X-3-23   (data: https://doi.org/10.5524/100106)
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent / "data"
ARCHIVE = DATA_DIR / "wollny_perfusion.zip"
WOLLNY_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/"
    "100106/gigascience-myperfdata.zip"
)
FALLBACKS = ("https://doi.org/10.5524/100106 (GigaDB)",
             "git clone https://git.code.sf.net/p/tabseg/myoperfdata data/myoperfdata")


def has_studies(folder: Path) -> bool:
    return folder.is_dir() and any(p.is_dir() and p.name.lower().startswith("study") for p in folder.rglob("Study*"))


def download(url: str, dest: Path, expected_min_mb: float = 100.0) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="bytes") as bar:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)  # only a complete download gets the final name
    mb = dest.stat().st_size / 1e6
    print(f"[{'warn' if mb < expected_min_mb else 'ok'}] downloaded {mb:.1f} MB")


def safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            dest = (target / info.filename).resolve()
            if target != dest and target not in dest.parents:
                raise RuntimeError(f"refusing entry outside {target}: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"refusing symbolic link in archive: {info.filename}")
        for info in tqdm(zf.infolist(), desc="unpacking", unit="file"):
            if not (target / info.filename).exists():  # never overwrite
                zf.extract(info, target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-extract", action="store_true", help="download only")
    args = ap.parse_args()

    if has_studies(DATA_DIR):
        print(f"[ok] Study* folders already present under {DATA_DIR} - nothing to do.")
        return
    try:
        download(WOLLNY_URL, ARCHIVE)
    except Exception as e:
        print(f"[error] download failed: {e}", file=sys.stderr)
        print("Download manually from one of:\n  " + "\n  ".join(FALLBACKS) +
              f"\nand put the archive at {ARCHIVE} (or the Study* folders under {DATA_DIR}).", file=sys.stderr)
        sys.exit(1)
    if args.no_extract:
        return
    if not zipfile.is_zipfile(ARCHIVE):
        print(f"[error] {ARCHIVE.name} is not a zip archive - unpack it manually into {DATA_DIR}", file=sys.stderr)
        sys.exit(1)
    safe_extract(ARCHIVE, DATA_DIR)
    print(f"[ok] unpacked; Study* folders found: {has_studies(DATA_DIR)}. Next: python run_eda.py")


if __name__ == "__main__":
    main()
