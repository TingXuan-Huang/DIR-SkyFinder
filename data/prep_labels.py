"""Drop rows with invalid TempM and save a clean label table.

Reads data/complete_table_with_mcr.csv, drops rows where TempM is the
sentinel -9999, -999, or NaN, keeps the columns we'll consume downstream,
and writes data/labels.parquet.

Run from project root:  python data/prep_labels.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "complete_table_with_mcr.csv"
OUT_PATH = DATA_DIR / "labels.parquet"

KEEP = ["Filename", "CamId", "TempM", "Month", "Hour", "Latitude", "Longitude"]
SENTINELS = {-9999.0, -999.0}


def main() -> None:
    df = pd.read_csv(CSV_PATH, usecols=KEEP)
    before = len(df)
    df = df[~df["TempM"].isin(SENTINELS) & df["TempM"].notna()]
    df = df.reset_index(drop=True)
    after = len(df)
    dropped = before - after
    print(f"[drop] {before:,} -> {after:,} rows  ({dropped} dropped, {dropped/before*100:.2f}%)")
    print(f"[TempM] min={df['TempM'].min():.1f}  max={df['TempM'].max():.1f}  mean={df['TempM'].mean():.2f}")
    print(f"[cams] {df['CamId'].nunique()} unique")
    df.to_parquet(OUT_PATH, index=False)
    print(f"[saved] {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
