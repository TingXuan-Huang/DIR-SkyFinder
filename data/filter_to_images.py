"""Filter labels.parquet to rows whose image is on disk.

Scans data/images/<CamId>/*.jpg, intersects with data/labels.parquet
(the cleaned-TempM table), and writes data/labels_with_images.csv —
the trainable subset.

Run from project root:  python data/filter_to_images.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
LABELS_IN = DATA_DIR / "labels.parquet"
IMG_DIR = DATA_DIR / "images"
LABELS_OUT = DATA_DIR / "labels_with_images.csv"


def main() -> None:
    df = pd.read_parquet(LABELS_IN)
    before = len(df)
    print(f"[in]   {before:,} rows from {LABELS_IN.name}")

    existing: set[tuple[int, str]] = set()
    for cam_dir in IMG_DIR.iterdir():
        if not cam_dir.is_dir():
            continue
        cam = int(cam_dir.name)
        for jpg in cam_dir.glob("*.jpg"):
            existing.add((cam, jpg.name))
    print(f"[scan] found {len(existing):,} images in {IMG_DIR}/")

    keys = list(zip(df["CamId"].tolist(), df["Filename"].tolist()))
    mask = [k in existing for k in keys]
    df = df.loc[mask].reset_index(drop=True)
    after = len(df)
    dropped = before - after
    print(f"[out]  {before:,} -> {after:,} rows  ({dropped:,} dropped, {dropped/before*100:.1f}%)")
    print(f"[cams] {df['CamId'].nunique()} unique remaining")

    df.to_csv(LABELS_OUT, index=False)
    print(f"[saved] {LABELS_OUT}  ({LABELS_OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
