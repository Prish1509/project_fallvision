"""Model forward pass, a small training run, checkpoints, learning-curve history, model choice."""
import csv
import os
import subprocess
import sys

import numpy as np
import torch


def test_gru_forward_and_augmentation():
    from train_gru import GRUNet, augment
    m = GRUNet()
    x = torch.zeros(4, 30, 92)
    out = m(x)
    assert out.shape == (4, 3)
    xa = augment(np.random.default_rng(0).normal(size=(4, 30, 92)).astype(np.float32), np.random.default_rng(1))
    assert xa.shape == (4, 30, 92) and np.isfinite(xa).all()


def test_small_training_run_checkpoints_and_history(root, tmp_path, tiny_windows):
    model_out = tmp_path / "models" / "gru_test.pt"
    hist = tmp_path / "gru_test_history.csv"
    r = subprocess.run([sys.executable, os.path.join(root, "src", "train_gru.py"), "--windows", tiny_windows,
                        "--epochs", "2", "--batch", "16", "--hard-weight", "3",
                        "--out", str(tmp_path / "gru_test.md"), "--model-out", str(model_out),
                        "--probs-out", str(tmp_path / "probs" / "gru_test"), "--history-out", str(hist)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # checkpoints: best (the file infer.py reads) and last (with optimizer state)
    assert model_out.exists() and (tmp_path / "models" / "gru_test_last.pt").exists()
    from detect_falls import load_gru
    model, mean, std = load_gru(str(model_out), "cpu")
    assert mean.shape == (92,) and std.shape == (92,)
    with torch.no_grad():
        p = torch.softmax(model(torch.zeros(2, 30, 92)), 1)
    assert p.shape == (2, 3) and abs(float(p.sum()) - 2.0) < 1e-4
    last = torch.load(tmp_path / "models" / "gru_test_last.pt", weights_only=False)
    assert last["epoch"] == 2 and "optimizer" in last and last["best_epoch"] in (1, 2)
    # learning-curve history: one row per epoch with train and val loss / F1
    rows = list(csv.DictReader(open(hist)))
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    for k in ("train_loss", "val_loss", "train_macro_f1", "val_macro_f1", "train_accuracy", "val_accuracy"):
        assert all(np.isfinite(float(r[k])) for r in rows), k
    # outputs the next stage reads
    assert (tmp_path / "probs" / "gru_test" / "val.npz").exists()
    assert "window macro-F1" in open(tmp_path / "gru_test.md").read()


def test_training_is_reproducible(root, tmp_path, tiny_windows):
    outs = []
    for i in range(2):
        mo = tmp_path / f"m{i}.pt"
        r = subprocess.run([sys.executable, os.path.join(root, "src", "train_gru.py"), "--windows", tiny_windows,
                            "--epochs", "1", "--batch", "16", "--out", str(tmp_path / f"g{i}.md"),
                            "--model-out", str(mo), "--history-out", str(tmp_path / f"h{i}.csv")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.append(torch.load(mo, weights_only=False)["state"])
    for k in outs[0]:
        assert torch.allclose(outs[0][k], outs[1][k]), f"weights differ for {k} with the same seed"


def test_random_forest_and_model_choice(root, tmp_path, tiny_windows):
    probs = tmp_path / "probs"
    for cmd in ([sys.executable, os.path.join(root, "src", "train_rf.py"), "--windows", tiny_windows, "--trees", "20",
                 "--out", str(tmp_path / "rf.md"), "--model-out", str(tmp_path / "rf.joblib"),
                 "--probs-out", str(probs / "rf")],
                [sys.executable, os.path.join(root, "src", "train_gru.py"), "--windows", tiny_windows, "--epochs", "1",
                 "--batch", "16", "--out", str(tmp_path / "gru.md"), "--model-out", str(tmp_path / "gru.pt"),
                 "--probs-out", str(probs / "gru"), "--history-out", str(tmp_path / "gru_history.csv")],
                [sys.executable, os.path.join(root, "src", "fuse.py"), "--windows", tiny_windows,
                 "--out", str(tmp_path / "fusion.md"), "--models", f"rf={probs / 'rf'}", f"gru={probs / 'gru'}"],
                [sys.executable, os.path.join(root, "src", "plot_curves.py"), "--history",
                 f"final={tmp_path / 'gru_history.csv'}", "--out", str(tmp_path / "curves.png")]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, " ".join(cmd) + "\n" + r.stdout + r.stderr
    assert "**Chosen:**" in open(tmp_path / "fusion.md").read()
    assert (tmp_path / "curves.png").stat().st_size > 1000 and (tmp_path / "curves.md").exists()
