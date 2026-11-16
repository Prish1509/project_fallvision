# FallVision: methodology, from raw video to final results

The complete workflow, in the order the pipeline runs it. Each stage is one script in `src/`;
`run_pipeline.py` runs them in order (`--from` / `--to` select a part). Paths are never written into
the code: the video folders, the work folder and the device are arguments.

```
videos (Le2i, CAUCAFall, MCFD, OOPS, GMDCSA24)      labels/omnifall/*.csv (OmniFall)
        |                                                    |
   [splits]  src/make_splits.py  -------------------->  splits/videos.csv (train / val / 3 test sets)
        |
   [poses]   src/cache_poses.py   YOLO11m-pose + ByteTrack, 15 frames/s  -> work/poses/<dataset>/<video>.pkl
        |
   [features] src/preprocess.py   join tracks, main person, 92 features   -> work/features/<dataset>/<video>.npz
        |
   [windows] src/make_windows.py  2-s windows every 0.5 s, 3 classes      -> work/windows/<split>.npz
        |
   [train]   src/train_gru.py     final GRU (+ comparison GRUs, src/train_rf.py)  -> models/gru_hard.pt,
             src/plot_curves.py   per-epoch history -> results/training_curves.png    results/gru_*.md
        |
   [choose]  src/fuse.py          model / score / threshold on validation  -> results/fusion.md
        |
   [evaluate] src/eval_causal.py  the infer.py engine on every val/test video; lag + threshold chosen on
                                  validation; tests once -> results/causal_eval.md, config.yaml (threshold)
        |
   infer.py + config.yaml + models/   (the submitted system)
```

## 1. Data

| Dataset | Videos used | Role | Source |
|---|---|---|---|
| GMDCSA24 | 160 (4 subjects) | train S1 / val S2 / test S3-4 | GitHub release (MIT) |
| Le2i | 190 | train / val / test (official split) | research use |
| CAUCAFall | 97 | train / val / test (official split) | research use |
| MCFD | 192 | train only | research use |
| OOPS (fall clips) | 818 | test only | non-commercial research |
| own recordings | 20 | test only, private | filmed by the team |

Labels and splits come from **OmniFall** (`simplexsigil2/omnifall`): frame-accurate activity segments
(16 classes) and official cross-subject splits, so no actor appears in two splits. Download notebooks:
`notebooks/01_download_omnifall_videos.ipynb`, `notebooks/02_download_gmdcsa24.ipynb`; the exact videos
used can be packed and checked with `src/check_dataset.py` (`docs/DATA.md`).

Final split sizes (`splits/summary.md`): train 426, validation 76, test_gmdcsa24 80, test_staged 57,
test_oops 818, test_own 20.

## 2. Pose caching (`src/cache_poses.py`)
Every split video is decoded once. Frames are taken at 15 per second (the model's rate), YOLO11m-pose
(input 640, confidence 0.25, people only) gives boxes and 17 joints with confidences, ByteTrack gives
track IDs. One pickle per video; `index.csv` lists status, frames and seconds. The run is resumable
(cached videos are skipped) and stops cleanly after `--max-hours`. Device: `--device cpu | cuda | cuda:N`,
falling back to CPU with a warning when no GPU exists.

## 3. Features (`src/preprocess.py`)
Per video: (1) join broken tracks (a new ID starting within 1.5 s and one body height of a lost one is the
same person); (2) main person = longest visible joined track (for training; at inference every person is
scored); (3) fill gaps ≤ 0.5 s, mark longer gaps as not seen; (4) normalise joints relative to the hip
centre and the standing height (90th percentile of box height); (5) 92 features per frame:
34 positions, 34 velocities, 17 confidences, 7 whole-body cues (head-top height above the floor and its
speed, box height / standing height, aspect, torso angle, hip speed, present flag).

## 4. Windows and labels (`src/make_windows.py`)
2-s windows (30 frames) every 0.5 s. Label from the OmniFall segments: **falling** if the window overlaps
a fall segment by ≥ 0.5 s (or half of a shorter fall), **fallen** if it ends inside a post-fall lying
segment, otherwise **normal** (lying down on purpose, sitting down, exercise are normal). Windows where
the person is seen < 50 % of the time are dropped. Each window also keeps the OmniFall class at its end
(`src`) for hard-negative weighting. Per-video truth (`has_fall`) is kept for video-level metrics.

## 5. Training (`src/train_gru.py`, `src/train_rf.py`)
- Features are standardised with the training-set mean and std (saved inside the model file).
- Final GRU: 2 layers, hidden 128, dropout 0.3, AdamW lr 1e-3 / wd 1e-4, cosine schedule, batch 256,
  40 epochs, class-weighted cross-entropy, hard negatives ×3, augmentation (mirror, rotate, scale, joint
  noise, night, frame drop), seed 0. The epoch with the best **validation window macro-F1** is kept.
  Checkpoints: best model whenever validation improves (`models/gru_hard.pt`) and the last epoch with
  optimizer state (`models/gru_hard_last.pt`). A per-epoch history (train / val loss, macro-F1, accuracy,
  learning rate) is saved to `results/curves/` and drawn in `results/training_curves.png`.
- Comparison models (`--all-models`): GRU without hard weighting, GRU without augmentation, Random
  Forest (300 trees on 460 window statistics). Each model writes window probabilities per split for the
  choice stage and a per-split table (`results/<model>.md`).
- Validation is used for model selection only; the test splits are never looked at during training.

## 6. Model choice on validation (`src/fuse.py`)
Video score = highest window score after averaging neighbouring windows. For every model and every
model average, two window scores (fall+fallen, falling) and two threshold rules (best F1; precision ≥
0.95) are scored on validation. The best validation F1 (ties → fewer false alarms) is chosen and then
reported once on each test split (`results/fusion.md`). Result: `gru_hard`, fall+fallen, best-F1 threshold.

## 7. The submitted engine and its evaluation (`src/fv_engine.py`, `src/eval_causal.py`, `infer.py`)
The engine processes a video sequentially: each frame on the 15/s grid is tracked, features are computed
per person from the **past 10 s only**, a window is scored every 0.5 s, and an alarm is raised the moment
a person's smoothed score reaches the threshold (`decision_timestamp_s` = the latest frame used;
`onset_estimate_s` = the moment of fastest head drop in the window, recorded separately). State is
reset per video. `eval_causal.py --run` runs exactly this engine on every validation and test video
(scores for decision lags 0 / 0.5 / 1.0 / 1.5 s in one pass); `--report` chooses the lag (best validation
average precision, ties → shorter) and then the threshold (best validation F1, ties → fewer false alarms),
reports every test set once, and `--write-config` writes the choice into `config.yaml`.

## 8. Metrics
Video level: a fall video is correct if at least one alarm is raised, a normal video if none is.
Precision, recall, F1, accuracy, false alarms, missed falls, median alarm delay (first crossing minus the
labelled fall start). Window level (training): macro-F1 over normal / falling / fallen, and loss curves.
Localization ("where") was checked by eye on alarm frames (`results/localization.md`); speed with
`src/speed_test.py`.

## 9. Leakage controls
Cross-subject splits; every threshold / lag / model chosen on validation; OOPS and own videos used only
for the final test; feature standardisation from the training set only; the hard validation set is built
from validation videos only; the one test-set peek that happened (threshold 0.864, Step 27b) is disclosed
in `results/summary.md` and was not kept.

## 10. Reproducibility
Seeds fixed; the whole pipeline is one command (`python run_pipeline.py ...`) on CPU or GPU, or the Kaggle
notebooks (`kaggle/project_kaggle.ipynb`, or `notebooks/03-05` in stages); a re-run re-chooses the threshold
on validation for the new weights and writes it to `config.yaml`. Exact steps: `REPRODUCIBILITY.md`.
