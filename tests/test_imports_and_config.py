"""Syntax / imports of every script, config loading and device handling."""
import glob
import os
import py_compile

import pytest


def test_every_script_compiles(root):
    files = glob.glob(os.path.join(root, "src", "*.py")) + glob.glob(os.path.join(root, "app", "*.py")) + \
        [os.path.join(root, f) for f in ("infer.py", "run_pipeline.py", "make_submission.py")]
    assert len(files) > 25
    for f in files:
        py_compile.compile(f, doraise=True)


def test_core_imports():
    import fv_engine, train_gru, train_rf, preprocess, make_windows, eval_causal, cache_poses   # noqa: F401
    import infer, run_pipeline                                                                  # noqa: F401


def test_config_loads_and_is_complete(root):
    from fv_engine import load_config
    cfg = load_config(os.path.join(root, "config.yaml"))
    for key in ("pose_model", "gru", "imgsz", "imgsz_small", "small_px", "det_conf", "tracker", "model_fps",
                "window_s", "stride_s", "history_s", "join_gap_s", "decision_lag_s", "score", "threshold",
                "rearm_s", "label"):
        assert key in cfg, key
    assert cfg["label"] == "person_collapse"
    assert 0 < cfg["threshold"] < 2 and cfg["model_fps"] == 15.0 and cfg["window_s"] == 2.0
    assert os.path.basename(cfg["gru"]) == "gru_hard.pt" and os.path.basename(cfg["pose_model"]) == "yolo11m-pose.pt"


def test_device_values():
    import torch
    from fv_engine import torch_device
    assert torch_device("cpu") == ("cpu", "cpu", None)
    dev, ydev, warn = torch_device("cuda:0")
    if torch.cuda.is_available():
        assert dev == "cuda:0" and ydev == "0" and warn is None
    else:
        assert dev == "cpu" and ydev == "cpu" and "not available" in warn
    assert torch_device("1")[0] in ("cuda:1", "cpu")
    with pytest.raises(ValueError):
        torch_device("tpu")


def test_pipeline_dry_run(root):
    import subprocess, sys
    r = subprocess.run([sys.executable, "run_pipeline.py", "--omnifall-root", "O", "--gmd-root", "G",
                        "--all-models", "--dry-run"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for stage in ("poses", "features", "windows", "train", "choose", "evaluate"):
        assert f"== {stage}:" in r.stdout
    assert "plot_curves.py" in r.stdout and "--write-config" in r.stdout
