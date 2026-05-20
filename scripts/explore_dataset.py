"""Download SkyFinder metadata CSV and preview its contents.

Run: python explore_dataset.py
Outputs: prints schema/stats; saves data/temperature_hist.png if a temp column is found.
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
CSV_URL = "https://cs.valdosta.edu/~rpmihail/skyfinder/analysis/complete_table_with_mcr.csv"
CSV_PATH = DATA_DIR / "complete_table_with_mcr.csv"


def download_metadata() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if CSV_PATH.exists():
        print(f"[skip] {CSV_PATH} exists ({CSV_PATH.stat().st_size:,} bytes)")
        return
    print(f"[download] {CSV_URL}")
    urlretrieve(CSV_URL, CSV_PATH)
    print(f"[done] {CSV_PATH} ({CSV_PATH.stat().st_size:,} bytes)")


def preview(df: pd.DataFrame) -> None:
    print(df.head())


def plot_temperature_hist(df: pd.DataFrame) -> None:
    # Sentinels: -9999 in TempM/HeatIndexM, -999 in WindChillM when not computed.
    series = {
        "TempM (actual)":       df["TempM"][df["TempM"] > -500],
        "WindChillM (cold)":    df["WindChillM"][df["WindChillM"] > -500],
        "HeatIndexM (hot)":     df["HeatIndexM"][df["HeatIndexM"] > -500],
    }
    colors = {"TempM (actual)": "tab:gray", "WindChillM (cold)": "tab:blue", "HeatIndexM (hot)": "tab:red"}

    plt.figure(figsize=(9, 4.5))
    bins = range(-50, 55, 1)
    for label, s in series.items():
        plt.hist(s, bins=bins, alpha=0.55, color=colors[label], label=f"{label}  n={len(s):,}")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("count")
    plt.title("SkyFinder: actual temp vs. wind chill (low) vs. heat index (high)")
    plt.legend()
    plt.tight_layout()
    out = DATA_DIR / "temperature_hist.png"
    plt.savefig(out, dpi=120)
    print(f"[saved] {out}")


SENTINELS = {-9999.0, -999.0}
# Identifier / redundant / high-cardinality columns to skip in pairwise plots.
SKIP_NUMERIC = {"TempM", "TempI", "CamId", "Year", "Date"}


def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace SkyFinder missing-value sentinels with NaN in numeric columns."""
    df = df.copy()
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].mask(df[c].isin(SENTINELS))
    return df


def plot_pairs_with_temp(df: pd.DataFrame, target: str = "TempM", cols_per_row: int = 6) -> None:
    """One panel per numeric column vs target: boxplot if low-cardinality, scatter otherwise."""
    cols = [c for c in df.select_dtypes(include="number").columns if c not in SKIP_NUMERIC | {target}]
    n = len(cols)
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(3.0 * cols_per_row, 2.4 * rows))
    axes = axes.flatten()

    y = df[target]
    for ax, c in zip(axes, cols):
        x = df[c]
        mask = x.notna() & y.notna()
        r = x[mask].corr(y[mask])
        uniq = x.dropna().unique()
        if len(uniq) <= 5:
            vals = sorted(uniq)
            ax.boxplot([y[(x == v) & y.notna()] for v in vals],
                       tick_labels=[str(int(v)) if float(v).is_integer() else f"{v:.2g}" for v in vals],
                       showfliers=False)
        else:
            ax.scatter(x[mask], y[mask], s=2, alpha=0.1)
        ax.set_title(f"{c}  r={r:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.axis("off")
    fig.supylabel(target)
    fig.tight_layout()
    out = DATA_DIR / "pairs_with_temp.png"
    fig.savefig(out, dpi=110)
    print(f"[saved] {out}  ({n} variables)")


def plot_temp_by_category(df: pd.DataFrame, col: str, target: str = "TempM", top_n: int = 20) -> None:
    """Boxplot of target by the top-N most-frequent categories of `col`, ordered by median."""
    top = df[col].value_counts().head(top_n).index
    sub = df[df[col].isin(top)]
    order = sub.groupby(col)[target].median().sort_values().index
    data = [sub.loc[sub[col] == v, target].dropna() for v in order]

    plt.figure(figsize=(max(8, 0.55 * len(order)), 4.8))
    plt.boxplot(data, tick_labels=list(order), showfliers=False)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(target)
    plt.title(f"{target} by {col} (top {len(order)} categories, ordered by median)")
    plt.tight_layout()
    out = DATA_DIR / f"temp_by_{col}.png"
    plt.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    download_metadata()
    df = pd.read_csv(CSV_PATH)
    preview(df)
    df = clean_sentinels(df)
    plot_temperature_hist(df)
    plot_pairs_with_temp(df)
    plot_temp_by_category(df, "Conds")
