"""
Training and validation curves of the GRU models (for the report).

Reads the per-epoch history written by train_gru.py (<name>_history.csv) and draws, per model:
  left   loss: training vs validation (same class weighting; training measured on a fixed sample of
         training windows without augmentation), plus the loss actually optimized (with augmentation
         and hard-negative weights, dotted)
  right  window macro-F1: training vs validation
A dashed line marks the epoch kept (best validation macro-F1). A last figure compares the validation
macro-F1 of all models. --submitted adds a star for a saved model measured by check_model.py (the
weights actually submitted), so a re-run's curves can be compared with them.

Usage:
  python src/plot_curves.py --history final=results/curves/gru_hard_history.csv \
                            aug=results/curves/gru_aug_history.csv noaug=results/curves/gru_noaug_history.csv \
                            --submitted final=results/curves/submitted_gru_hard.json --out results/training_curves.png
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

TITLES = {"final": "Final GRU (aug. + hard neg. x3)", "aug": "GRU (aug.)", "noaug": "GRU (no aug.)"}


def read(path):
    rows = list(csv.DictReader(open(path, newline="")))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description="Plot training / validation curves")
    ap.add_argument("--history", nargs="+", required=True, help="name=path/to/<model>_history.csv ...")
    ap.add_argument("--submitted", nargs="*", default=[], help="name=check_model.json: star for the submitted weights")
    ap.add_argument("--out", default="results/training_curves.png")
    a = ap.parse_args()
    runs = [(n, read(p)) for n, p in (h.split("=", 1) for h in a.history)]
    subm = {n: json.load(open(p)) for n, p in (s.split("=", 1) for s in a.submitted)}

    fig, axes = plt.subplots(len(runs) + 1, 2, figsize=(12, 3.6 * (len(runs) + 1)), squeeze=False)
    lines = ["# Training and validation curves", "",
             "| Model | Best epoch | Val macro-F1 (best) | Train loss / val loss at best | Train / val macro-F1 at best |",
             "|---|---|---|---|---|"]
    for row, (name, h) in enumerate(runs):
        ep = h["epoch"]
        best = max(range(len(ep)), key=lambda i: (h["val_macro_f1"][i], -i))
        ax = axes[row][0]
        ax.plot(ep, h["train_loss"], label="training", color="#1f77b4")
        ax.plot(ep, h["val_loss"], label="validation", color="#d62728")
        ax.plot(ep, h["train_loss_optimized"], ":", label="training, as optimized (running mean)", color="#7f7f7f")
        ax.axvline(ep[best], ls="--", color="black", lw=1)
        if name in subm:
            ax.plot(subm[name]["epoch"], subm[name]["val_loss"], "*", ms=13, color="#ff7f0e",
                    label=f"submitted weights, validation (epoch {subm[name]['epoch']})")
        ax.set_title(f"{TITLES.get(name, name)}: loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("weighted cross-entropy")
        ax.legend(fontsize=8)
        ax = axes[row][1]
        ax.plot(ep, h["train_macro_f1"], label="training", color="#1f77b4")
        ax.plot(ep, h["val_macro_f1"], label="validation", color="#d62728")
        ax.axvline(ep[best], ls="--", color="black", lw=1, label=f"kept: epoch {int(ep[best])}")
        if name in subm:
            ax.plot(subm[name]["epoch"], subm[name]["val_macro_f1"], "*", ms=13, color="#ff7f0e",
                    label=f"submitted weights, validation (epoch {subm[name]['epoch']})")
        ax.set_title(f"{TITLES.get(name, name)}: window macro-F1")
        ax.set_xlabel("epoch")
        ax.set_ylabel("macro-F1 (normal / falling / fallen)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        lines.append(f"| {TITLES.get(name, name)} | {int(ep[best])} | {h['val_macro_f1'][best]:.3f} | "
                     f"{h['train_loss'][best]:.3f} / {h['val_loss'][best]:.3f} | "
                     f"{h['train_macro_f1'][best]:.3f} / {h['val_macro_f1'][best]:.3f} |")
    for name, r in subm.items():
        lines.append(f"| Submitted weights ({r['model']}) | {r['epoch']} | {r['val_macro_f1']:.3f} | "
                     f"- / {r['val_loss']:.3f} | - / {r['val_macro_f1']:.3f} |")
    ax = axes[-1][0]
    for name, h in runs:
        ax.plot(h["epoch"], h["val_loss"], label=TITLES.get(name, name))
    ax.set_title("Validation loss: all models")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
    ax = axes[-1][1]
    for name, h in runs:
        ax.plot(h["epoch"], h["val_macro_f1"], label=TITLES.get(name, name))
    ax.set_title("Validation window macro-F1: all models")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    md = os.path.splitext(a.out)[0] + ".md"
    open(md, "w").write("\n".join(lines + ["", f"![curves]({os.path.basename(a.out)})"]) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {a.out} and {md}")


if __name__ == "__main__":
    main()
