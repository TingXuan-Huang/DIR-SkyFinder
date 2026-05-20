"""Regression test: feed a tiny set of JSON fixtures through `build_dataframe`."""
import json

from skyfinder.analysis.aggregate import build_dataframe


def _minimal_run_json(name: str, model: str = "resnet50", fold: int = 0,
                     overall: float = 4.0) -> dict:
    return {
        "run_name": name,
        "config": {"model": model, "fold": fold, "seed": 0, "freeze_backbone": False,
                   "use_lds": False, "use_fds": False, "bin_width": 1.0},
        "final_val": {"overall": overall, "many": overall, "medium": overall, "few": overall},
    }


def test_build_dataframe_basic(tmp_path):
    results_dir = tmp_path / "results"
    (results_dir / "baseline_resnet50").mkdir(parents=True)
    j = results_dir / "baseline_resnet50" / "baseline_resnet50_fold0.json"
    j.write_text(json.dumps(_minimal_run_json("baseline_resnet50_fold0")))

    cfg = {
        "aggregate_result_dirs": [str(results_dir)],
        "ablation_yamls": [],
        "test_inference_path": str(tmp_path / "test_inference.json"),
        "aggregate_csv": str(tmp_path / "aggregate.csv"),
    }
    df = build_dataframe(cfg)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "baseline_resnet50_fold0"
    assert df.iloc[0]["group"] == "main"
    assert df.iloc[0]["config_kind"] == "baseline"
    assert df.iloc[0]["val_overall"] == 4.0


def test_build_dataframe_skips_dist_files(tmp_path):
    """`dist_*.json` outputs should be skipped, not classified as 'unknown'."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "dist_summary.json").write_text(json.dumps({"foo": "bar"}))
    cfg = {
        "aggregate_result_dirs": [str(results_dir)],
        "ablation_yamls": [],
        "test_inference_path": str(tmp_path / "test_inference.json"),
        "aggregate_csv": str(tmp_path / "aggregate.csv"),
    }
    df = build_dataframe(cfg)
    assert len(df) == 0
