"""Download SkyFinder images to data/images/<CamId>/<Filename>.

For each (CamId, Filename) in complete_table_with_mcr.csv, fetches
  https://cs.valdosta.edu/~rpmihail/skyfinder/images/<CamId>/<Filename>
into data/images/<CamId>/<Filename>. Resumable: files already on disk are
skipped. Writes via .part-then-rename so a Ctrl-C never leaves a partial JPEG.

Run from project root:  python data/download_images.py
Pilot on one camera:    set CAMS = [10066] below.
"""
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "complete_table_with_mcr.csv"
IMG_DIR = DATA_DIR / "images"
BASE_URL = "https://cs.valdosta.edu/~rpmihail/skyfinder/images"

CAMS = None        # None = all 53 cameras; or e.g. [10066] for a pilot
WORKERS = 4        # host appears to cap concurrent connections per IP
TIMEOUT = 30       # seconds; release a stalled connection


def fetch_one(cam: int, fname: str) -> tuple[str, str]:
    """Download one image atomically. Returns (fname, status)."""
    out = IMG_DIR / str(cam) / fname
    if out.exists():
        return fname, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    url = f"{BASE_URL}/{cam}/{fname}"
    try:
        with urlopen(url, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(out)
        return fname, "ok"
    except (HTTPError, URLError, TimeoutError) as e:
        tmp.unlink(missing_ok=True)
        if isinstance(e, HTTPError):
            return fname, f"http {e.code}"
        if isinstance(e, TimeoutError):
            return fname, "timeout"
        return fname, f"url {e.reason}"


def download_camera(cam: int, fnames: list[str]) -> None:
    ok = skip = 0
    errs: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(fetch_one, cam, fn) for fn in fnames]
        bar = tqdm(as_completed(futures), total=len(fnames), desc=f"cam {cam}", unit="img")
        for fut in bar:
            fname, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                errs.append((fname, status))
            bar.set_postfix(ok=ok, skip=skip, err=len(errs))
    for fname, status in errs[:5]:
        print(f"    [err] {fname}  {status}")
    if len(errs) > 5:
        print(f"    ...and {len(errs) - 5} more errors")


def main() -> None:
    df = pd.read_csv(CSV_PATH, usecols=["CamId", "Filename"])
    cams = sorted(df["CamId"].unique()) if CAMS is None else CAMS
    print(f"[plan] {len(cams)} cameras, {len(df):,} total rows")
    for cam in cams:
        fnames = df.loc[df["CamId"] == cam, "Filename"].tolist()
        print(f"[cam {cam}] {len(fnames)} rows")
        download_camera(cam, fnames)


if __name__ == "__main__":
    main()
