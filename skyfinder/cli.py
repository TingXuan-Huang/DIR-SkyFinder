"""SkyFinder console script. Lazy-imports keep `skyfinder --help` fast.

Subcommands:
    skyfinder train      — run experiments (by family or by name)
    skyfinder inference  — test-set MAE for every saved checkpoint
    skyfinder analyze    — c1 / c2 / d1 / embeddings / trajectory / linear_probe / aggregate / figures / all
    skyfinder figures    — convenience: same as `analyze figures` plus extra one-off figures
    skyfinder dist       — distribution-shift analysis
    skyfinder data-prep  — utility (currently: --migrate-results)

All subcommands accept `--help` for full flag listings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ============================================================
# train
# ============================================================

def _cmd_train(args: argparse.Namespace) -> int:
    """Mirror the old `python run.py --family ... --experiment ...` UX."""
    # Discovery modes — no data load.
    if args.list is not None:
        from skyfinder.training.families import print_list
        print_list(None if args.list == "__ALL__" else args.list, args.names_only)
        return 0
    if args.count is not None:
        from skyfinder.training.families import print_count
        print_count(args.count)
        return 0

    from skyfinder.training import families
    from skyfinder.training.config import SPLITS
    from skyfinder.training.engine import get_device

    # --- Resolve config + filter ---
    pattern = None
    config_path = args.config
    if args.family is not None:
        if args.family not in families.FAMILIES:
            print(f"unknown family: {args.family!r}", file=sys.stderr)
            print(f"known families: {', '.join(families.FAMILIES)}", file=sys.stderr)
            return 2
        entry = families.FAMILIES[args.family]
        config_path = config_path or Path(entry["config"])
        pattern = entry.get("pattern")
    if config_path is None:
        print("either --family or --config is required", file=sys.stderr)
        return 2

    yaml_cfg = families.load_yaml(config_path)
    families.bind_paths(yaml_cfg)
    n_splits = families.split_count()
    print(f"[env] device={get_device()}  splits={n_splits}")

    save = bool(yaml_cfg.get("save", True))
    skip_existing = bool(yaml_cfg.get("skip_existing", True))

    smoke = yaml_cfg.get("smoke_test", {})
    if smoke.get("enabled", False) and not args.skip_smoke:
        _run_one(smoke, save=save, skip_existing=skip_existing, dry_run=args.dry_run)

    experiments = families.filter_experiments(yaml_cfg["experiments"], pattern, args.experiment)
    if args.family or args.experiment:
        print(f"[filter] family={args.family} experiment={args.experiment} "
              f"-> {len(experiments)} match(es)")
    if not experiments:
        print(f"no experiments match family={args.family!r} experiment={args.experiment!r}",
              file=sys.stderr)
        return 1

    for exp in experiments:
        for spec in families.expand_experiment(exp):
            if spec["fold"] >= n_splits:
                print(f"fold {spec['fold']} out of range ({SPLITS} has {n_splits} folds)",
                      file=sys.stderr)
                return 1
            _run_one(spec, save=save, skip_existing=skip_existing, dry_run=args.dry_run)
    return 0


def _run_one(spec: dict, save: bool, skip_existing: bool, dry_run: bool) -> None:
    from skyfinder.training import families
    from skyfinder.training.trainer import run_baseline
    kwargs = {k: spec[k] for k in families.RUN_KEYS if k in spec}
    run_name = kwargs["run_name"]
    if skip_existing and families.completed(run_name):
        print(f"[skip] {run_name} already has a results JSON")
        return
    print(f"[run] {run_name}")
    if dry_run:
        print(f"      {kwargs}")
        return
    run_baseline(save=save, **kwargs)


# ============================================================
# inference
# ============================================================

def _cmd_inference(args: argparse.Namespace) -> int:
    from skyfinder.analysis.config_loader import load_config
    from skyfinder.analysis.test_inference import run as run_inference
    cfg = load_config(args.config)
    run_inference(cfg, out_path=args.out, result_dirs=args.results_dirs,
                  pattern=args.pattern, exclude=args.exclude, include_vit=args.include_vit,
                  with_preds=not args.no_preds,
                  skip_existing=not args.no_skip_existing,
                  batch_size=args.batch_size, num_workers=args.num_workers)
    return 0


# ============================================================
# analyze
# ============================================================

_ANALYZE_KINDS = ("c1", "c2", "d1", "embeddings", "trajectory", "linear_probe",
                  "aggregate", "figures", "all")


def _cmd_analyze(args: argparse.Namespace) -> int:
    from skyfinder.analysis.config_loader import load_config
    cfg = load_config(args.config)

    kind = args.kind
    if kind == "c1":
        from skyfinder.analysis.baselines_constant import run_c1
        run_c1(cfg, out_path=args.out)
    elif kind == "c2":
        from skyfinder.analysis.baselines_metadata import run_c2
        run_c2(cfg, out_path=args.out, seed=args.seed)
    elif kind == "d1":
        from skyfinder.analysis.skymask_inference import run_d1
        run_d1(cfg, out_path=args.out, fold=args.fold,
               batch_size=args.batch_size, num_workers=args.num_workers)
    elif kind == "linear_probe":
        from skyfinder.analysis.linear_probe import run_linear_probe_summary
        run_linear_probe_summary(cfg, out_path=args.out, fold=args.fold)
    elif kind == "embeddings":
        from skyfinder.analysis.extract_embeddings import run_embeddings
        run_embeddings(cfg, out_dir=args.out_dir, fold=args.fold,
                       batch_size=args.batch_size, num_workers=args.num_workers)
    elif kind == "trajectory":
        from skyfinder.analysis.extract_trajectory import (DEFAULT_RUNS,
                                                            run_trajectory)
        runs = tuple(args.runs) if args.runs else DEFAULT_RUNS
        splits = tuple(args.splits) if args.splits else ("train", "val", "test")
        train_subsample = None if args.no_subsample else args.train_subsample
        run_trajectory(cfg, out_dir=args.out_dir, fold=args.fold, runs=runs, splits=splits,
                       train_subsample=train_subsample,
                       batch_size=args.batch_size, num_workers=args.num_workers)
        if args.with_figures:
            from skyfinder.analysis.figures import render_trajectory
            render_trajectory(cfg, runs=runs, fold=args.fold)
    elif kind == "aggregate":
        from skyfinder.analysis.aggregate import main as agg_main
        agg_main(cfg)
    elif kind == "figures":
        from skyfinder.analysis.aggregate import build_dataframe
        from skyfinder.analysis.figures import render_all
        df = build_dataframe(cfg)
        render_all(cfg, df)
    elif kind == "all":
        from skyfinder.analysis.aggregate import main as agg_main
        from skyfinder.analysis.baselines_constant import run_c1
        from skyfinder.analysis.aggregate import build_dataframe
        from skyfinder.analysis.figures import render_all
        run_c1(cfg)
        agg_main(cfg)
        df = build_dataframe(cfg)
        render_all(cfg, df)
    else:
        print(f"unknown analyze kind: {kind!r} (choices: {_ANALYZE_KINDS})", file=sys.stderr)
        return 2
    return 0


# ============================================================
# figures (one-off renderers + same as `analyze figures`)
# ============================================================

def _cmd_figures(args: argparse.Namespace) -> int:
    """Subcommands for one-off figure renders not covered by `analyze figures`."""
    from skyfinder.analysis.config_loader import load_config
    cfg = load_config(args.config)
    kind = args.kind
    if kind == "loss-curves":
        from skyfinder.analysis.figures import fig_training_curves
        from skyfinder.analysis.figures._helpers import KIND_ORDER
        # When --runs given, use it. Otherwise default to all 8 (4 ResNet + 4 ViT) if --all,
        # else the headline 4 ResNet.
        if args.runs:
            runs = [tuple(r.split(":", 1)) if ":" in r else (r, r) for r in args.runs]
        elif args.all_runs:
            runs = [(f"{k}_resnet50", k) for k in KIND_ORDER] + [(f"{k}_vit", k) for k in KIND_ORDER]
        else:
            runs = None  # function default
        grid = tuple(args.grid) if args.grid else None
        fig_training_curves(cfg, runs=runs, fold=args.fold, grid=grid,
                            output_name=args.output_name)
    elif kind == "embed-by-bin":
        from skyfinder.analysis.figures import fig_embed_by_bin
        fig_embed_by_bin(cfg, run_name=args.run, fold=args.fold,
                         split=args.split, method=args.method)
    elif kind == "all" or kind is None:
        # Same as `analyze figures`.
        from skyfinder.analysis.aggregate import build_dataframe
        from skyfinder.analysis.figures import render_all
        df = build_dataframe(cfg)
        render_all(cfg, df)
    else:
        print(f"unknown figures kind: {kind!r}", file=sys.stderr)
        return 2
    return 0


# ============================================================
# dist (passthrough)
# ============================================================

def _cmd_dist(args: argparse.Namespace) -> int:
    from skyfinder.analysis.dist import main as dist_main
    # Forward remaining args to dist.main via sys.argv.
    sys.argv = ["skyfinder dist"] + list(args.passthrough)
    dist_main()
    return 0


# ============================================================
# data-prep
# ============================================================

def _cmd_data_prep(args: argparse.Namespace) -> int:
    if args.migrate_results is not None:
        from skyfinder.training.migrate import run_migration
        root = Path(args.migrate_results)
        return run_migration(root, dry_run=args.dry_run)
    print("nothing to do — pass --migrate-results PATH", file=sys.stderr)
    return 2


# ============================================================
# Top-level argparse
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skyfinder", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Run training experiments (by family or by name)")
    t.add_argument("--family", type=str, default=None)
    t.add_argument("--config", type=Path, default=None)
    t.add_argument("--experiment", type=str, default=None)
    t.add_argument("--list", nargs="?", const="__ALL__", default=None)
    t.add_argument("--count", type=str, default=None)
    t.add_argument("--names-only", action="store_true")
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--skip-smoke", action="store_true")
    t.set_defaults(func=_cmd_train)

    # --- inference ---
    i = sub.add_parser("inference", help="Test-set MAE for every saved checkpoint")
    i.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    i.add_argument("--out", type=Path, default=None)
    i.add_argument("--results-dirs", nargs="+", type=Path, default=None)
    i.add_argument("--pattern", type=str, default=None)
    i.add_argument("--exclude", type=str, default=None)
    i.add_argument("--include-vit", action="store_true")
    i.add_argument("--no-preds", action="store_true")
    i.add_argument("--no-skip-existing", action="store_true")
    i.add_argument("--batch-size", type=int, default=32)
    i.add_argument("--num-workers", type=int, default=2)
    i.set_defaults(func=_cmd_inference)

    # --- analyze ---
    a = sub.add_parser("analyze", help="Run analysis subcommands")
    a.add_argument("kind", choices=_ANALYZE_KINDS)
    a.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    a.add_argument("--out", type=Path, default=None)
    a.add_argument("--out-dir", type=Path, default=None)
    a.add_argument("--fold", type=int, default=0)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--num-workers", type=int, default=2)
    a.add_argument("--runs", nargs="+", default=None,
                   help="for `trajectory`: list of run name prefixes")
    a.add_argument("--splits", nargs="+", default=None,
                   help="for `trajectory`: which split files to extract")
    a.add_argument("--train-subsample", type=int, default=10000)
    a.add_argument("--no-subsample", action="store_true")
    a.add_argument("--with-figures", action="store_true",
                   help="for `trajectory`: also render the 5 trajectory figures")
    a.set_defaults(func=_cmd_analyze)

    # --- figures ---
    f = sub.add_parser("figures", help="One-off figure renders + (default) full sweep")
    f.add_argument("kind", nargs="?", default="all",
                   choices=("all", "loss-curves", "embed-by-bin"))
    f.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    f.add_argument("--fold", type=int, default=0)
    f.add_argument("--runs", nargs="+", default=None,
                   help="for `loss-curves`: list of run prefixes, optionally name:label")
    f.add_argument("--all-runs", action="store_true",
                   help="for `loss-curves`: shortcut for all 8 ResNet+ViT runs in a 2x4 grid")
    f.add_argument("--grid", nargs=2, type=int, default=None,
                   help="for `loss-curves`: (nrows, ncols)")
    f.add_argument("--output-name", type=str, default=None)
    f.add_argument("--run", type=str, default=None,
                   help="for `embed-by-bin`: single run name prefix")
    f.add_argument("--split", type=str, default="val")
    f.add_argument("--method", type=str, default="umap", choices=("pca", "umap"))
    f.set_defaults(func=_cmd_figures)

    # --- dist ---
    d = sub.add_parser("dist", help="Distribution-shift analysis (passes through to dist module)")
    d.add_argument("passthrough", nargs=argparse.REMAINDER)
    d.set_defaults(func=_cmd_dist)

    # --- data-prep ---
    dp = sub.add_parser("data-prep", help="Data + results utilities")
    dp.add_argument("--migrate-results", type=str, default=None,
                    help="Migrate flat-layout results to nested. Pass results dir path.")
    dp.add_argument("--dry-run", action="store_true",
                    help="With --migrate-results: print plan, don't move files.")
    dp.set_defaults(func=_cmd_data_prep)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
