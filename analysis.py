"""Top-level CLI for the analysis pipeline. Each subcommand maps to one of the
`analysis/` modules. C2 lives in `ablation.py` (it predates this file and didn't
need to move).

Examples:
    python analysis.py c1                         # constant predictors (no GPU)
    python analysis.py d1                         # sky-mask inference (4 ckpts x fold 0)
    python analysis.py embeddings                 # extract 2048-d features (4 ckpts x fold 0)
    python analysis.py aggregate                  # build ablations/results/aggregate.csv
    python analysis.py figures                    # render all PDFs/PNGs into figures/
    python analysis.py all                        # c1 -> aggregate -> figures (skips d1/embeddings)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_c1(a):
    from analysis.c1 import run_c1
    run_c1(a.labels, a.splits, a.out)


def _cmd_d1(a):
    from analysis.d1 import run_d1
    run_d1(a.out, mask_dir=a.masks, fold=a.fold,
           batch_size=a.batch_size, num_workers=a.num_workers)


def _cmd_embeddings(a):
    from analysis.embeddings import run_embeddings
    run_embeddings(a.out_dir, fold=a.fold,
                   batch_size=a.batch_size, num_workers=a.num_workers)


def _cmd_aggregate(_a):
    from analysis.aggregate import main as agg_main
    agg_main()


def _cmd_figures(_a):
    from analysis.figures import main as fig_main
    fig_main()


def _cmd_all(_a):
    # Lightweight pipeline: things that don't need GPU/checkpoints.
    from analysis.c1 import run_c1
    from analysis.aggregate import main as agg_main
    from analysis.figures import main as fig_main
    run_c1(Path("data/labels_with_images.csv"),
           Path("data/splits/loco_5fold.json"),
           Path("ablations/results/c1_constants.json"))
    agg_main()
    fig_main()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("c1", help="constant predictors (CPU, ~10s)")
    p.add_argument("--labels", type=Path, default=Path("data/labels_with_images.csv"))
    p.add_argument("--splits", type=Path, default=Path("data/splits/loco_5fold.json"))
    p.add_argument("--out",    type=Path, default=Path("ablations/results/c1_constants.json"))
    p.set_defaults(func=_cmd_c1)

    p = sub.add_parser("d1", help="sky-mask inference, 4 ckpts x fold 0 (GPU)")
    p.add_argument("--masks", type=Path, default=Path("data/masks"))
    p.add_argument("--fold",  type=int,  default=0)
    p.add_argument("--out",   type=Path, default=Path("ablations/results/d1_skymask.json"))
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.set_defaults(func=_cmd_d1)

    p = sub.add_parser("embeddings", help="extract penultimate features, 4 ckpts x fold 0 (GPU)")
    p.add_argument("--out-dir", type=Path, default=Path("ablations/results/embeddings"))
    p.add_argument("--fold",    type=int,  default=0)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.set_defaults(func=_cmd_embeddings)

    sub.add_parser("aggregate", help="build aggregate.csv from all result JSONs").set_defaults(func=_cmd_aggregate)
    sub.add_parser("figures",   help="render all nature-style figures").set_defaults(func=_cmd_figures)
    sub.add_parser("all",       help="c1 -> aggregate -> figures (skips d1/embeddings)").set_defaults(func=_cmd_all)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
