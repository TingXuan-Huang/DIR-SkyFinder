"""Download per-camera sky masks for D1 ablation.

URL pattern (verified): https://cs.valdosta.edu/~rpmihail/skyfinder/Masks/<CamId>.png
                        (capital M in "Masks")

By default downloads only the cameras needed for fold 0 of `loco_5fold.json`'s
val split -- that's the set D1's inference touches. To grab masks for all
cameras instead, pass `--all`.

Run from project root:
    python data/download_masks.py            # fold 0 val cameras
    python data/download_masks.py --fold 2   # fold 2
    python data/download_masks.py --all      # every CamId in labels_with_images.csv

Resumable: files already on disk are skipped. Atomic .part-then-rename.
"""
import argparse
import json
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("data")
LABELS = DATA_DIR / "labels_with_images.csv"
SPLITS = DATA_DIR / "splits" / "loco_5fold.json"
MASK_DIR = DATA_DIR / "masks"
BASE_URL = "https://cs.valdosta.edu/~rpmihail/skyfinder/Masks"
TIMEOUT = 30


def fetch_one(cam) -> tuple[str, str]:
    """Download mask for one camera. Returns (cam_id_str, status)."""
    out = MASK_DIR / f"{cam}.png"
    if out.exists():
        return str(cam), "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    url = f"{BASE_URL}/{cam}.png"
    try:
        with urlopen(url, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(out)
        return str(cam), "ok"
    except (HTTPError, URLError, TimeoutError) as e:
        tmp.unlink(missing_ok=True)
        if isinstance(e, HTTPError):
            return str(cam), f"http {e.code}"
        if isinstance(e, TimeoutError):
            return str(cam), "timeout"
        return str(cam), f"url {e.reason}"


def _cams_for_fold(fold: int) -> list:
    df = pd.read_csv(LABELS, usecols=["CamId"])
    splits = json.loads(SPLITS.read_text())
    val_idx = splits[fold]["val"]
    return sorted(df.iloc[val_idx]["CamId"].unique())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fold", type=int, default=0, help="LOCO fold (default 0); val cameras used")
    ap.add_argument("--all", action="store_true", help="ignore --fold, download every CamId in labels_with_images.csv")
    a = ap.parse_args()

    if a.all:
        cams = sorted(pd.read_csv(LABELS, usecols=["CamId"])["CamId"].unique())
        scope = "all cams"
    else:
        cams = _cams_for_fold(a.fold)
        scope = f"fold {a.fold} val cams"
    print(f"[plan] {len(cams)} cameras ({scope}) -> {MASK_DIR}/")

    ok = skip = 0
    errs: list[tuple[str, str]] = []
    bar = tqdm(cams, desc="masks", unit="cam")
    for cam in bar:
        _, status = fetch_one(cam)
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            errs.append((str(cam), status))
        bar.set_postfix(ok=ok, skip=skip, err=len(errs))
    print(f"[done] ok={ok} skip={skip} err={len(errs)}")
    for cam, status in errs:
        print(f"    [err] cam {cam}  {status}")


if __name__ == "__main__":
    main()
