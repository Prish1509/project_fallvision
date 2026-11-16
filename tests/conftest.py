"""Shared fixtures: the repository root on sys.path, a tiny synthetic windows folder, a tiny config."""
import csv
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def tiny_windows(tmp_path_factory):
    """Synthetic windows in the format of make_windows.py: 30 frames x 92 features, 3 classes,
    8 videos per split. Falls get a visible pattern so a model can learn something in 2 epochs."""
    rng = np.random.default_rng(0)
    out = tmp_path_factory.mktemp("windows")
    for split in ("train", "val", "test_gmdcsa24"):
        X, y, vid, t, src, videos = [], [], [], [], [], []
        for v in range(8):
            has_fall = v % 2 == 0
            for w in range(6):
                lab = 0
                if has_fall and w in (3, 4):
                    lab = 1 if w == 3 else 2
                x = rng.normal(0, 0.1, (30, 92)).astype(np.float32)
                x[:, 91] = 1.0                                   # present flag
                if lab:
                    x[:, 86] += 2.0                              # head-top speed: a fall-like cue
                X.append(x); y.append(lab); vid.append(v); t.append(2.0 + 0.5 * w); src.append(lab)
            videos.append(dict(dataset="synthetic", path=f"v{v}", has_fall=int(has_fall), n_windows=6))
        np.savez_compressed(out / f"{split}.npz", X=np.stack(X), y=np.array(y, np.int8),
                            vid=np.array(vid, np.int32), t=np.array(t, np.float32), src=np.array(src, np.int8))
        with open(out / f"{split}_videos.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(videos[0]))
            wr.writeheader()
            wr.writerows(videos)
    return str(out)
