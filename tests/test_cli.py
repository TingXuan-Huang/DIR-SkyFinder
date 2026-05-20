"""Smoke test: every `skyfinder <subcmd> --help` exits 0."""
import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "skyfinder.cli", *args],
                          capture_output=True, text=True, timeout=30)


def test_top_help():
    r = _run(["--help"])
    assert r.returncode == 0, r.stderr
    assert "skyfinder" in r.stdout.lower()


def test_train_help():
    r = _run(["train", "--help"])
    assert r.returncode == 0, r.stderr


def test_inference_help():
    r = _run(["inference", "--help"])
    assert r.returncode == 0, r.stderr


def test_analyze_help():
    r = _run(["analyze", "--help"])
    assert r.returncode == 0, r.stderr


def test_figures_help():
    r = _run(["figures", "--help"])
    assert r.returncode == 0, r.stderr


def test_dist_help():
    r = _run(["dist", "--help"])
    assert r.returncode == 0, r.stderr


def test_data_prep_help():
    r = _run(["data-prep", "--help"])
    assert r.returncode == 0, r.stderr


def test_train_list_works():
    """train --list enumerates FAMILIES without loading torch."""
    r = _run(["train", "--list"])
    assert r.returncode == 0, r.stderr
    assert "main" in r.stdout
