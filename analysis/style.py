"""Nature-journal figure style helpers.

Convention used everywhere in `analysis/figures.py`:

  fig, ax = plt.subplots(figsize=figsize("single"))
  ax.plot(...)
  save_fig(fig, "fig_main_sweep")

`apply_nature_style()` is called at module import in figures.py so individual
figure functions never touch rcParams.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Column widths from Nature's figure guidelines (89 / 120 / 183 mm).
WIDTHS_MM = {"single": 89.0, "onehalf": 120.0, "double": 183.0}
MM_PER_INCH = 25.4

# Categorical palette for the 4 DIR configs. Order matches per_bin_mae's
# (many, medium, few) when needed, but primarily indexed by config_kind.
PALETTE = {
    "baseline": "#4C72B0",   # blue
    "lds":      "#DD8452",   # orange
    "fds":      "#55A868",   # green
    "lds_fds":  "#C44E52",   # red
}
MARKERS = {"baseline": "o", "lds": "s", "fds": "^", "lds_fds": "D"}

# For reference baselines drawn as horizontal lines. Picked from tab10 indices
# that don't collide with PALETTE (blue/orange/green/red): purple, cyan, olive.
REF_COLORS = {"c1": "#9467BD", "c2": "#17BECF", "d1": "#BCBD22"}


def figsize(width: str = "single", height_ratio: float = 0.62) -> tuple[float, float]:
    """Return (w, h) in inches for a Nature column width.

    `width` in {"single", "onehalf", "double"}; height defaults to ~golden ratio.
    """
    w_in = WIDTHS_MM[width] / MM_PER_INCH
    return (w_in, w_in * height_ratio)


def apply_nature_style() -> None:
    """Set matplotlib rcParams for Nature-style output. Idempotent."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # TrueType in PDF (editable in Illustrator)
        "ps.fonttype": 42,
    })


def save_fig(fig, name: str, out_dir: str | Path = "figures") -> Path:
    """Save `fig` as both PDF (vector) and PNG (300 dpi) under `out_dir`. Returns the PDF path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {pdf_path}")
    return pdf_path
