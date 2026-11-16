"""
Re-run the whole FallVision pipeline, from downloaded videos to the final evaluation.

    python run_pipeline.py --omnifall-root OMNI --gmd-root GMD --work work --device cuda:0
    python run_pipeline.py --work work --from train --to choose --all-models   # stages without videos
    python run_pipeline.py ... --dry-run                         # print the commands only

Before: download the videos with notebooks/01_download_omnifall_videos.ipynb and
notebooks/02_download_gmdcsa24.ipynb (OMNI = the folder with le2i/ caucafall/ mcfd/ OOPS/,
GMD = the folder with "Subject 1" ... "Subject 4"), and put yolo11m-pose.pt in models/.

Stages (each one is a script in src/; outputs of a stage are the inputs of the next):
  splits     src/make_splits.py    train / val / test lists from the OmniFall labels and official splits
                                   (already in splits/; re-made only with --remake-splits, needs internet)
  poses      src/cache_poses.py    YOLO11m-pose + ByteTrack once per video -> <work>/poses   (GPU, ~6 h)
  features   src/preprocess.py     92 features per frame per person          -> <work>/features
  windows    src/make_windows.py   2-s labelled windows                       -> <work>/windows
  train      src/train_gru.py      the final GRU (augmentation, hard negatives x3) -> models/gru_hard.pt
                                   (+ with --all-models: GRU without hard weighting, GRU without
                                   augmentation, Random Forest, for the comparison table)
             src/plot_curves.py    training / validation curves -> results/training_curves.png
  choose     src/fuse.py           model / score / threshold chosen on validation -> results/fusion.md
  evaluate   src/eval_causal.py    the submitted frame-by-frame system (infer.py engine) on validation and
                                   every test set, decision lags 0 / 0.5 / 1 / 1.5 s; lag and threshold
                                   chosen on validation only -> results/causal_eval.md, and written into
                                   config.yaml (retraining on a GPU never gives bit-identical weights)

Small result files are also copied into results/: poses_index.csv (poses), features_manifest.csv
(features), windows_summary.md (windows).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGES = ["splits", "poses", "features", "windows", "train", "choose", "evaluate"]


def commands(a):
    py, w = sys.executable, a.work
    probs = os.path.join(w, "probs")
    own = ["--own-root", a.own_root] if a.own_root else []
    cmd = {
        "splits": [[py, "src/make_splits.py", "--labels", "labels/omnifall", "--out", "splits"]] if a.remake_splits else [],
        "poses": [[py, "src/cache_poses.py", "--omnifall-root", a.omnifall_root, "--gmd-root", a.gmd_root, *own,
                   "--out", os.path.join(w, "poses"), "--model", a.pose_model, "--imgsz", "640", "--device", a.device,
                   "--max-hours", str(a.max_hours)]],
        "features": [[py, "src/preprocess.py", "--poses", os.path.join(w, "poses"), "--out", os.path.join(w, "features")]],
        "windows": [[py, "src/make_windows.py", "--features", os.path.join(w, "features"), "--labels", "labels/omnifall",
                     "--out", os.path.join(w, "windows")]],
        "train": [[py, "src/train_gru.py", "--windows", os.path.join(w, "windows"), "--hard-weight", "3", "--seed", str(a.seed),
                   "--out", "results/gru_hard.md", "--model-out", "models/gru_hard.pt",
                   "--probs-out", os.path.join(probs, "gru_hard"),
                   "--history-out", "results/curves/gru_hard_history.csv"]],
        "choose": [[py, "src/fuse.py", "--windows", os.path.join(w, "windows"), "--out", "results/fusion.md",
                    "--models", f"gru_hard={os.path.join(probs, 'gru_hard')}"]],
        "evaluate": [[py, "src/eval_causal.py", "--run", "--scores", os.path.join(w, "causal_scores.csv"),
                      "--omnifall-root", a.omnifall_root, "--gmd-root", a.gmd_root, *own, "--device", a.device],
                     [py, "src/eval_causal.py", "--report", "--scores", os.path.join(w, "causal_scores.csv"),
                      "--report-out", "results/causal_eval.md", "--write-config"]],
    }
    if a.all_models:
        cmd["train"] += [
            [py, "src/train_gru.py", "--windows", os.path.join(w, "windows"), "--seed", str(a.seed), "--out", "results/gru.md",
             "--model-out", os.path.join(w, "models", "gru.pt"), "--probs-out", os.path.join(probs, "gru"),
             "--history-out", "results/curves/gru_aug_history.csv"],
            [py, "src/train_gru.py", "--windows", os.path.join(w, "windows"), "--no-aug", "--seed", str(a.seed), "--out", "results/gru_noaug.md",
             "--model-out", os.path.join(w, "models", "gru_noaug.pt"), "--probs-out", os.path.join(probs, "gru_noaug"),
             "--history-out", "results/curves/gru_noaug_history.csv"],
            [py, "src/train_rf.py", "--windows", os.path.join(w, "windows"), "--out", "results/rf.md",
             "--model-out", os.path.join(w, "models", "rf.joblib"), "--probs-out", os.path.join(probs, "rf")]]
        cmd["choose"][0] += [f"{m}={os.path.join(probs, m)}" for m in ("gru", "gru_noaug", "rf")]
    curves = ["final=results/curves/gru_hard_history.csv"] + (
        ["aug=results/curves/gru_aug_history.csv", "noaug=results/curves/gru_noaug_history.csv"] if a.all_models else [])
    cmd["train"].append([py, "src/plot_curves.py", "--history", *curves, "--out", "results/training_curves.png"])
    return cmd


RECORDS = {"poses": [("poses/index.csv", "results/poses_index.csv")],
           "features": [("features/manifest.csv", "results/features_manifest.csv")],
           "windows": [("windows/summary.md", "results/windows_summary.md")]}


def record(stage, work):
    """Copy the small summary files of a stage into results/ (they are kept in git)."""
    for src, dst in RECORDS.get(stage, []):
        if os.path.exists(os.path.join(work, src)):
            shutil.copyfile(os.path.join(work, src), dst)
            print(f"   saved {dst}")


def main():
    ap = argparse.ArgumentParser(description="Re-run the whole FallVision pipeline")
    ap.add_argument("--omnifall-root", default=None, help="folder with le2i/ caucafall/ mcfd/ OOPS/ (poses, evaluate)")
    ap.add_argument("--gmd-root", default=None, help="GMDCSA24 folder with 'Subject 1' ... 'Subject 4' (poses, evaluate)")
    ap.add_argument("--own-root", default=None, help="our own test videos (optional, private)")
    ap.add_argument("--work", default="work", help="folder for poses, features, windows, probabilities")
    ap.add_argument("--pose-model", default="models/yolo11m-pose.pt")
    ap.add_argument("--device", default="cuda:0", help="cpu, cuda, cuda:N or N; falls back to cpu if there is no GPU")
    ap.add_argument("--from", dest="start", choices=STAGES, default="splits")
    ap.add_argument("--to", dest="end", choices=STAGES, default="evaluate")
    ap.add_argument("--all-models", action="store_true", help="also train the comparison models")
    ap.add_argument("--max-hours", type=float, default=11.0,
                    help="poses stage: stop cleanly after this long (resumable; Kaggle sessions end after 12 h)")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the GRU training")
    ap.add_argument("--remake-splits", action="store_true", help="rebuild splits/ from the OmniFall Hub files")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    todo = STAGES[STAGES.index(a.start):STAGES.index(a.end) + 1]
    if {"poses", "evaluate"} & set(todo) and not (a.omnifall_root and a.gmd_root):
        ap.error("the poses and evaluate stages read the videos: give --omnifall-root and --gmd-root")
    a.work = os.path.abspath(a.work)
    for k in ("omnifall_root", "gmd_root", "own_root"):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    os.chdir(ROOT)
    os.makedirs(os.path.join(a.work, "models"), exist_ok=True)
    cmds = commands(a)
    t_all = time.time()
    for stage in todo:
        if not cmds[stage]:
            print(f"== {stage}: skipped (splits/ is already in the repository)")
            continue
        for c in cmds[stage]:
            print(f"\n== {stage}: {' '.join(c[1:])}", flush=True)
            if a.dry_run:
                continue
            t0 = time.time()
            r = subprocess.run(c)
            if r.returncode != 0:
                sys.exit(f"Stage '{stage}' failed (exit {r.returncode}); fix it and re-run with --from {stage}")
            print(f"== {stage} done in {(time.time() - t0) / 60:.1f} min", flush=True)
        if not a.dry_run:
            record(stage, a.work)
    print(f"\nPipeline finished ({', '.join(todo)}) in {(time.time() - t_all) / 60:.1f} min.")
    if "evaluate" in todo and not a.dry_run:
        print("config.yaml now holds the decision lag and threshold chosen on validation (see above).")


if __name__ == "__main__":
    main()
