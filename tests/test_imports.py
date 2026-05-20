"""Smoke test: every public skyfinder module imports cleanly.

Catches the most common refactor regression: a missing rename, a stale
`from analysis.foo` line, or a circular import. Runs in <2 seconds.
"""


def test_top_level():
    import skyfinder
    assert skyfinder.__version__


def test_training_subpackage():
    from skyfinder.training import config, dataloader, model, engine, checkpoint
    from skyfinder.training import lds, fds, trainer, families, migrate, diagnostics
    assert config.Config
    assert dataloader.SkyFinderDataset
    assert model.build_model
    assert model.FDSModel
    assert engine.train_one_epoch
    assert engine.predict_split
    assert engine.per_bin_mae
    assert checkpoint.find_artifact
    assert checkpoint.save_model_weights
    assert checkpoint.save_training_state
    assert trainer.run_baseline
    assert families.FAMILIES
    assert migrate.run_migration


def test_analysis_subpackage():
    from skyfinder.analysis import (config_loader, style, aggregate,
                                     baselines_constant, baselines_metadata,
                                     skymask_inference, linear_probe,
                                     extract_embeddings, extract_trajectory,
                                     corrupt_labels, dist, test_inference)
    assert config_loader.load_config
    assert aggregate.build_dataframe


def test_figures_subpackage():
    from skyfinder.analysis.figures import (fig_main_sweep, fig_training_curves,
                                             fig_pred_vs_true_scatter,
                                             fig_dist_and_errbar, fig_a1_sigma,
                                             fig_a2_reweight, fig_a3_bucket,
                                             fig_a4_momentum, fig_a5_start_smooth,
                                             fig_d1_skymask, fig_d4_linprobe,
                                             fig_e1_seeds, fig_f1_rate, fig_f2_range,
                                             fig_f3_drop, fig_f5_noise,
                                             fig_embed_temp, fig_embed_cam,
                                             fig_embed_knn, fig_embed_per_bin,
                                             fig_embed_cka, fig_embed_by_bin,
                                             fig_traj_pca, fig_traj_per_bin,
                                             fig_traj_knn_mae, fig_traj_cka,
                                             render_all, render_trajectory)
    assert render_all
    assert fig_training_curves


def test_cli_module():
    from skyfinder import cli
    parser = cli.build_parser()
    args = parser.parse_args(["train", "--list"])
    assert args.cmd == "train"
