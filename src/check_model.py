"""
Measure a saved GRU on the validation windows exactly as the learning curves do (train_gru.py):
class-weighted cross-entropy (weights from the training windows), window macro-F1 and accuracy.

Used to put the submitted model (models/gru_hard.pt) on the training curves of a re-run with the
same script, data and seed: if the point lies on the re-run's validation curve at the same epoch,
the curves describe the submitted training.

Usage:
  python src/check_model.py --model models/gru_hard.pt --windows WORK/windows \
                            --out results/curves/submitted_gru_hard.json
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from train_gru import GRUNet, load_raw, predict


def main():
    ap = argparse.ArgumentParser(description="Validation loss / macro-F1 of a saved GRU")
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True, help="JSON file with epoch, val_loss, val_macro_f1, val_accuracy")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = GRUNet(hidden=int(ck.get("hidden", 128))).to(device)
    model.load_state_dict(ck["state"])
    tr, va = load_raw(a.windows, "train"), load_raw(a.windows, "val")
    counts = np.bincount(tr["y"], minlength=3)
    w = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    w = w / w.mean()
    p = predict(model, va["X"], ck["mean"], ck["std"], device)
    y = va["y"]
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.0))
    res = dict(model=os.path.basename(a.model), epoch=int(ck.get("epoch", 0)),
               val_loss=round(float((nll * w[y]).sum() / w[y].sum()), 5),
               val_macro_f1=round(float(f1_score(y, p.argmax(1), average="macro", labels=[0, 1, 2],
                                                 zero_division=0)), 5),
               val_accuracy=round(float((p.argmax(1) == y).mean()), 5))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
