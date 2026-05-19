"""Top-level CLI for the analysis pipeline.

Every subcommand resolves paths from `analysis_config.yaml` (override with
`--config PATH`). C2 lives in `ablation.py` (its CLI predates this file).

Examples:
    python analysis.py c1                              # constant predictors (no GPU)
    python analysis.py d1                              # sky-mask inference (4 ckpts x fold 0)
    python analysis.py embeddings                      # extract 2048-d features (val + test)
    python analysis.py trajectory --with-figures       # per-epoch features + plots
    python analysis.py aggregate                       # build aggregate.csv
    python analysis.py figures                         # render all PDFs/PNGs
    python analysis.py all                             # c1 -> aggregate -> figures
    python analysis.py --config other.yaml figures     # use a different config
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis._config import load_config


def _cmd_c1(cfg, a):
    from analysis.c1 import run_c1
    run_c1(cfg, out_path=a.out)


def _cmd_c2(cfg, a):
    from analysis.c2 import run_c2
    run_c2(cfg, out_path=a.out, seed=a.seed)


def _cmd_d1(cfg, a):
    from analysis.d1 import DEFAULT_RUNS, run_d1
    runs = tuple(a.runs) if a.runs else DEFAULT_RUNS
    run_d1(cfg, out_path=a.out, fold=a.fold, runs=runs,
           batch_size=a.batch_size, num_workers=a.num_workers)


def _cmd_linear_probe(cfg, a):
    from analysis.linear_probe import run_linear_probe_summary
    run_linear_probe_summary(cfg, out_path=a.out, fold=a.fold)


def _cmd_embeddings(cfg, a):
    from analysis.embeddings import DEFAULT_RUNS, run_embeddings
    runs = tuple(a.runs) if a.runs else DEFAULT_RUNS
    run_embeddings(cfg, out_dir=a.out_dir, fold=a.fold, runs=runs,
                   batch_size=a.batch_size, num_workers=a.num_workers)


def _cmd_trajectory(cfg, a):
    from analysis.trajectory import DEFAULT_RUNS, run_trajectory
    runs = tuple(a.runs) if a.runs else DEFAULT_RUNS
    splits = tuple(a.splits) if a.splits else ("train", "val", "test")
    train_subsample = None if a.no_subsample else a.train_subsample
    run_trajectory(cfg, out_dir=a.out_dir, fold=a.fold, runs=runs, splits=splits,
                   train_subsample=train_subsample,
                   batch_size=a.batch_size, num_workers=a.num_workers)
    if a.with_figures:
        from analysis.figures import make_trajectory
        make_trajectory(cfg, runs=runs, fold=a.fold)


def _cmd_sample_traj(cfg, a):
    from analysis.figures_sample_trajectory import DEFAULT_RUNS, make_sample_trajectory
    runs    = tuple(a.runs) if a.runs else DEFAULT_RUNS
    splits  = tuple(a.splits) if a.splits else ("val", "test")
    schemes = tuple(a.bin_schemes) if a.bin_schemes else ("freq", "temp")
    make_sample_trajectory(cfg, runs=runs, fold=a.fold, splits=splits,
                           bin_schemes=schemes, seed=a.seed)


def _cmd_closest_train(cfg, a):
    from analysis.closest_train_inspect import DEFAULT_RUNS, run_closest_train_inspect
    runs = tuple(a.runs) if a.runs else DEFAULT_RUNS
    run_closest_train_inspect(cfg, runs=runs, fold=a.fold, seed=a.seed)


def _cmd_bin_traj(cfg, a):
    from analysis.figures_sample_trajectory import DEFAULT_RUNS, make_bin_distribution_traj
    runs    = tuple(a.runs) if a.runs else DEFAULT_RUNS
    splits  = tuple(a.splits) if a.splits else ("val", "test")
    schemes = tuple(a.bin_schemes) if a.bin_schemes else ("freq", "temp")
    make_bin_distribution_traj(cfg, runs=runs, fold=a.fold, splits=splits,
                               bin_schemes=schemes, every=a.every,
                               subsample=a.subsample, seed=a.seed)


def _cmd_aggregate(cfg, _a):
    from analysis.aggregate import main as agg_main
    agg_main(cfg)


def _cmd_figures(cfg, _a):
    from analysis.figures import main as fig_main
    fig_main(cfg)


def _cmd_all(cfg, _a):
    from analysis.c1 import run_c1
    from analysis.aggregate import main as agg_main
    from analysis.figures import main as fig_main
    run_c1(cfg)
    agg_main(cfg)
    fig_main(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("analysis_config.yaml"),
                    help="path to analysis config YAML (default: analysis_config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("c1", help="constant predictors (CPU, ~10s)")
    p.add_argument("--out", type=Path, default=None, help="override config['c1_results_path']")
    p.set_defaults(func=_cmd_c1)

    p = sub.add_parser("c2", help="metadata-only HistGBR baseline (CPU, ~5min)")
    p.add_argument("--out", type=Path, default=None, help="override config['c2_results_path']")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_c2)

    p = sub.add_parser("linear_probe", help="D4: compute linprobe vs full-FT delta")
    p.add_argument("--out",  type=Path, default=None,
                   help="override default: <ablation_results_dir>/d4_linprobe_summary.json")
    p.add_argument("--fold", type=int, default=0)
    p.set_defaults(func=_cmd_linear_probe)

    p = sub.add_parser("d1", help="sky-mask inference, 4 ckpts x fold (GPU)")
    p.add_argument("--fold",  type=int,  default=0)
    p.add_argument("--out",   type=Path, default=None, help="override config['d1_results_path']")
    p.add_argument("--runs",  nargs="+", default=None,
                   help="run names (default: 4 resnet50 configs in d1.DEFAULT_RUNS)")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.set_defaults(func=_cmd_d1)

    p = sub.add_parser("embeddings", help="extract penultimate features (val + test, fold 0)")
    p.add_argument("--out-dir", type=Path, default=None, help="override config['embeddings_dir']")
    p.add_argument("--fold",    type=int,  default=0)
    p.add_argument("--runs",    nargs="+", default=None,
                   help="run names (default: 4 resnet50 configs in embeddings.DEFAULT_RUNS)")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.set_defaults(func=_cmd_embeddings)

    p = sub.add_parser("trajectory",
                       help="embedding trajectory across snapshot epochs (requires snapshot_every>0)")
    p.add_argument("--out-dir", type=Path, default=None, help="override config['trajectory_dir']")
    p.add_argument("--fold",    type=int,  default=0)
    p.add_argument("--runs",    nargs="+", default=None,
                   help="run names (default: 4 configs x 2 archs)")
    p.add_argument("--splits",  nargs="+", default=None,
                   help="which splits to extract (default: train val test)")
    p.add_argument("--train-subsample", type=int, default=5000,
                   help="random subsample size for train (default 5000)")
    p.add_argument("--no-subsample", action="store_true",
                   help="disable train subsampling (full ~65k images per snapshot)")
    p.add_argument("--with-figures", action="store_true",
                   help="render PCA/CKA/knn-mae figures right after extraction")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.set_defaults(func=_cmd_trajectory)

    p = sub.add_parser("sample_traj",
                       help="per-sample trajectory plot (needs snapshot_every>0 runs + trajectory npz)")
    p.add_argument("--fold",        type=int,  default=0)
    p.add_argument("--seed",        type=int,  default=0)
    p.add_argument("--runs",        nargs="+", default=None,
                   help="run names (default: 4 resnet50 configs)")
    p.add_argument("--splits",      nargs="+", default=None,
                   help="default: val test")
    p.add_argument("--bin-schemes", nargs="+", default=None, choices=["freq", "temp"],
                   help="default: both freq and temp")
    p.set_defaults(func=_cmd_sample_traj)

    p = sub.add_parser("closest_train",
                       help="median-residual test image vs feature-NN and geo-NN train neighbours")
    p.add_argument("--fold", type=int,  default=0)
    p.add_argument("--seed", type=int,  default=0)
    p.add_argument("--runs", nargs="+", default=None,
                   help="run names (default: 4 resnet50 configs)")
    p.set_defaults(func=_cmd_closest_train)

    p = sub.add_parser("bin_traj",
                       help="bin-colored distribution filmstrip across snapshot epochs")
    p.add_argument("--fold",          type=int,  default=0)
    p.add_argument("--seed",          type=int,  default=0)
    p.add_argument("--every",         type=int,  default=5,
                   help="keep epochs where ep %% every == 0 (plus first and last). default 5")
    p.add_argument("--subsample",     type=int,  default=None,
                   help="optional: random subsample per panel; same indices across panels. "
                        "default: plot all points")
    p.add_argument("--runs",          nargs="+", default=None,
                   help="run names (default: 4 resnet50 configs)")
    p.add_argument("--splits",        nargs="+", default=None,
                   help="default: val test")
    p.add_argument("--bin-schemes",   nargs="+", default=None, choices=["freq", "temp"],
                   help="default: both freq and temp")
    p.set_defaults(func=_cmd_bin_traj)

    sub.add_parser("aggregate", help="build aggregate.csv from all result JSONs").set_defaults(func=_cmd_aggregate)
    sub.add_parser("figures",   help="render all nature-style figures").set_defaults(func=_cmd_figures)
    sub.add_parser("all",       help="c1 -> aggregate -> figures (skips d1/embeddings)").set_defaults(func=_cmd_all)

    a = ap.parse_args()
    cfg = load_config(a.config)
    a.func(cfg, a)


if __name__ == "__main__":
    main()
