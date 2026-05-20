"""Publication figures for SkyFinder DIR experiments.

Subpackage layout:
    _helpers.py      — shared constants and small helper functions
    main_sweep.py    — fig_main_sweep, fig_pred_vs_true_scatter, fig_dist_and_errbar
    curves.py        — fig_training_curves (reusable; accepts arbitrary `runs`)
    ablations.py     — A1-A5, D1, D4, E1, F1-F5
    embeddings.py    — fig_embed_temp/cam/knn/per_bin/cka, fig_embed_by_bin
    trajectory.py    — fig_traj_pca/per_bin/knn/cka, render_trajectory

All public `fig_*` functions take `config: dict` (from `configs/analysis.yaml`)
as the first argument. Functions that hardcoded run lists now take `runs=` so
you can swap in ResNet, ViT, or mixed.

`render_all(config, df)` is the headline driver — same as the old `make_all`.
"""
from __future__ import annotations

# Re-export every public figure function. New callers do
# `from skyfinder.analysis.figures import fig_main_sweep, ...`.
from skyfinder.analysis.figures.ablations import (fig_a1_sigma, fig_a2_reweight,
                                                   fig_a3_bucket, fig_a4_momentum,
                                                   fig_a5_start_smooth,
                                                   fig_d1_skymask, fig_d4_linprobe,
                                                   fig_e1_seeds, fig_f1_rate,
                                                   fig_f2_range, fig_f3_drop,
                                                   fig_f5_noise)
from skyfinder.analysis.figures.curves import fig_training_curves
from skyfinder.analysis.figures.embeddings import (fig_embed_by_bin, fig_embed_cam,
                                                    fig_embed_cka, fig_embed_knn,
                                                    fig_embed_per_bin, fig_embed_temp)
from skyfinder.analysis.figures.main_sweep import (fig_dist_and_errbar,
                                                    fig_main_sweep,
                                                    fig_pred_vs_true_scatter)
from skyfinder.analysis.figures.trajectory import (fig_traj_cka, fig_traj_knn_mae,
                                                    fig_traj_pca, fig_traj_per_bin,
                                                    make_trajectory,
                                                    render_trajectory)


def render_all(config: dict, df) -> None:
    """Headline driver — render every public figure that has the data it needs.

    Missing data is logged as `[skip]` rather than crashing.
    """
    fig_main_sweep(config, df, metric="test")
    fig_main_sweep(config, df, metric="val")
    fig_training_curves(config)
    fig_pred_vs_true_scatter(config)
    fig_dist_and_errbar(config)
    fig_a1_sigma(config, df)
    fig_a2_reweight(config, df)
    fig_a3_bucket(config, df)
    fig_a4_momentum(config, df)
    fig_a5_start_smooth(config, df)
    fig_d1_skymask(config)
    fig_d4_linprobe(config, df)
    fig_e1_seeds(config, df)
    fig_f1_rate(config, df)
    fig_f2_range(config, df)
    fig_f3_drop(config, df)
    fig_f5_noise(config, df)
    for split in ("val", "test"):
        fig_embed_temp(config, split=split)
        fig_embed_cam(config, split=split)
        fig_embed_knn(config, split=split)
        fig_embed_per_bin(config, split=split)
        fig_embed_cka(config, split=split)


# Backwards-compatible alias (the old name was `make_all`).
make_all = render_all
