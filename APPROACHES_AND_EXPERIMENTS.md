# FallVision: approaches and experiments

Person collapse detection (DS5602 Computer Vision). This file explains every approach, algorithm,
architecture and experiment of the project and how they compare. All numbers come from the files in
`results/` named next to them; nothing here is estimated. For the step-by-step workflow see
`METHODOLOGY.md`; to re-run anything see `REPRODUCIBILITY.md`.

## 1. The task and the design goals

Detect when a person falls (collapses) in a video and say **who** (track ID), **where** (bounding box)
and **when** (decision time, plus a separate estimate of the onset). The goals we designed for:

| Design goal | How it is covered | Where |
|---|---|---|
| A single pretrained vision model (**YOLO11**) | YOLO11m-pose for people and 17 joints; nothing else is pretrained | `config.yaml`, `src/fv_engine.py` |
| Falls from **sitting** must be caught | falls from chairs / beds are in the training data (Le2i, CAUCAFall, GMDCSA24); window labels come from the fall *movement*, not from standing height | `src/make_windows.py` |
| **No false alarms** (lying down, sitting fast, exercise are not falls) | "lie down / lying / sit down / exercise" windows are *normal* and weighted ×3 as hard negatives; threshold chosen for best F1 with ties broken by fewer false alarms | `src/train_gru.py`, `src/eval_causal.py` |
| Works at **night** | skeleton features instead of pixels; "night" augmentation (joints hidden, confidences lowered) | `src/train_gru.py` (`augment`) |
| Tested on **unseen videos** | official OmniFall cross-subject splits; OOPS and our own recordings never used for any choice | `splits/videos.csv` |
| Causal operation: frame by frame, **past frames only**, alarms as they occur, state reset per video | the causal engine used by `infer.py` | `src/fv_engine.py` |
| One `config.yaml`, offline inference, one `infer.py` command, machine-readable outputs | | `infer.py`, `config.yaml` |

The sections below follow the order in which the work was done; the final system is described first.

## 2. The final system (what is submitted)

```
video -> frames sampled on a 15/s time grid -> YOLO11m-pose (people, 17 joints) -> ByteTrack (IDs)
      -> join broken tracks -> 92 features per frame per person (last 10 s)
      -> 2-s window every 0.5 s -> GRU: P(normal), P(falling), P(fallen)
      -> score = P(falling)+P(fallen), averaged with the previous window
      -> alarm when score >= threshold (who = track ID, where = box, when = latest frame; onset estimated separately)
```

Settings (all in `config.yaml`): input 640 px (960 px while people are small: standing height < 150 px
in a 1280-wide frame), detection confidence 0.25, 15 frames/s, 2-s windows every 0.5 s, 10 s of
history, track joining within 1.5 s, decision lag 0 s, threshold **0.8573**, re-arm after 10 s.

## 3. Approaches, in the order we tried them

### 3.1 Rule-based baseline  (`src/baseline_fall.py`, `src/detect_falls.py`)
YOLO11 pose + ByteTrack + a hand-written state machine per person: a fast drop of the head top,
then "down" (box height well below the person's standing height) for 2 s, with a stillness check.
Rules v2 tuned on GMDCSA24 Subjects 1-2 and tested on Subjects 3-4: **F1 0.870**, 1 false alarm
(push-ups); most misses were falls onto or from a bed or chair (`results/gmdcsa24_*`, `results/rules_splits.md`).
Lesson: thresholds on heights and speeds cannot separate "sitting down fast" from "falling".

### 3.2 Pose model size (`results/gmdcsa24_*`)
YOLO11n-pose vs YOLO11m-pose with the same rules: n F1 0.72, m F1 0.87. **YOLO11m-pose** is used.
Larger inputs: 960 helps far-away people (far-simulation F1 0.63 → 0.74); 1280 adds false alarms
(`results/far_*`). Final choice: 640, switching to 960 only while people are small.

### 3.3 Night and far-away simulations (`src/simulate_night.py`, `src/simulate_far.py`, `src/night_analysis.py`)
Validation videos were darkened (gamma, noise) and shrunk (far camera). Night F1 of the rules fell to
0.58, by misses, not false alarms; brightening the image (CLAHE, denoising) made it **worse** (F1
0.35-0.42) and was dropped (`results/night_*`). Analysis: at night the person is lost in 1 of 5 frames
and the torso is unreadable in 1 of 3. Decision: handle this in the learned model (night and frame-drop
augmentation) rather than in the image.

### 3.4 Skeleton features (`src/preprocess.py`)
92 numbers per frame per person: 34 joint positions relative to the hip centre and divided by the
standing height, 34 joint velocities, 17 joint confidences, 7 whole-body cues (head-top height above
the floor and its speed, box height / standing height, box aspect, torso angle, hip speed, "seen"
flag). Short gaps (≤ 0.5 s) are interpolated, longer gaps marked as not seen. Broken tracks are joined
(new ID within 1.5 s and one body height of a lost one). Everything is relative, so camera distance,
resolution and lighting matter less.

### 3.5 Labels and windows (`src/make_windows.py`)
OmniFall's 16 activity labels (frame-accurate) are mapped to 3 classes per 2-s window: **normal**
(including lie down, lying, sit down, exercise…), **falling** (window overlaps a fall segment by ≥ 0.5 s)
and **fallen** (window ends while lying after a fall). 12,587 training windows (9,683 / 1,795 / 1,109),
1,262 validation windows (`results/windows_summary.md`).

### 3.6 Random Forest (`src/train_rf.py`)
300 trees on 460 window-summary statistics of the 92 features (class-balanced, `random_state=0`).
First learned model; beats the rules everywhere, most on real-world video (OOPS F1 0.38 → 0.69,
`results/rf.md`).

### 3.7 GRU sequence classifier (`src/train_gru.py`)
- Input 30 frames × 92 features, standardised with the training mean / std (stored in the model file).
- 2-layer GRU, hidden 128, dropout 0.3; classifier on [last hidden state, mean of hidden states] → 3 classes.
- AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, batch 256, 40 epochs, class-weighted
  cross-entropy (weights ∝ 1/√count), gradient clipping 1.0, seed 0; the epoch with the best validation
  window macro-F1 is kept. The best checkpoint is saved whenever validation improves and
  `<model>_last.pt` keeps the last epoch with optimizer state.
- **Augmentation** (training only, each with probability 0.5): mirror (left/right joints swapped),
  rotation ±15°, scale 0.9-1.1, joint noise, **night** (hide 10-40 % of joints, lower confidences),
  **frame drop** (blank up to 20 % of frames).
- **Hard negatives ×3**: normal windows whose OmniFall class is sit down, lie down, lying, other,
  kneel down, kneeling, squat down, squatting, crawl or jump get 3× loss weight.

Three variants were trained (same data, same seed): without augmentation, with augmentation, with
augmentation + hard negatives ×3 (**final: `gru_hard.pt`**, best epoch 16 of 40, validation window
macro-F1 0.864, `results/gru_hard.md`). Training and validation curves per epoch (loss, macro-F1,
accuracy) are written by the pipeline to `results/curves/*_history.csv` and drawn in
`results/training_curves.png` (`src/plot_curves.py`).

### 3.8 Model / score / threshold choice on validation (`src/fuse.py`)
Every model alone, the mean of every pair and of all models, with two window scores (P(falling)+P(fallen)
vs P(falling) only) and two threshold rules (best F1; precision-first ≥ 0.95), all scored on
**validation only** (`results/fusion.md`). Chosen: `gru_hard` alone, score fall+fallen, best-F1
threshold. Averaging with the Random Forest or the other GRUs did not improve validation F1.

### 3.9 Whole-video evaluation and error analysis (`src/eval_system.py`, `src/eval_own.py`, `results/error_analysis.md`)
The complete detector on every test video (every person scored), 18 failure clips watched by hand,
20 own recordings (day / night, near / far, 1-9 people) checked (`src/check_own_videos.py`).
Localization checked by eye on 36 alarm frames: the alarm box is on the person who fell in 33 (92 %)
(`src/loc_sheets.py`, `results/localization.md`).

### 3.10 Hard validation set (`src/make_hard_val.py`, `results/system_hv_*`)
Validation videos made far away and composed into 3-person crowds, to test ideas without touching the
test sets. Tested and rejected: low-confidence tracking (far 18 → 17 caught, false alarms 22 → 21),
fall-aware track joining (no change), a higher threshold 0.864 chosen by F2 on this set (tested once on
the test sets: lost falls, no fewer false alarms; disclosed in `results/summary.md`).

### 3.11 The causal (frame-by-frame) engine and its evaluation (`src/fv_engine.py`, `src/eval_causal.py`)
Operating causally (current and past frames only) changes the features slightly (floor level and
standing height are known only from the past), so the final threshold was **re-chosen on validation
for this exact engine**: 0.8573 (`results/causal_eval.md`). A **decision lag** of 0 / 0.5 / 1.0 / 1.5 s
(score the window ending a little before the newest frame, still past frames only) was compared by
validation average precision: 0.958 / 0.951 / 0.942 / 0.923 → **lag 0** (`results/causal_eval_lags.md`).

### 3.12 Speed and deployment (`src/speed_test.py`, `app/`)
29.6 frames/s on a Tesla T4 (15 needed for real time), GRU 0.1 ms per frame; on a laptop CPU about
2-3 frames/s with PyTorch and 10-13 frames/s with OpenVINO (optional, used only by `app/`). `app/live_fall.py`
runs the same engine live from a webcam, a phone camera or an RTSP stream.

## 4. Results

### 4.1 Final submitted system (`infer.py` engine, threshold 0.8573 chosen on validation, each test set once; `results/causal_eval.md`)

| Split | Videos | F1 | Precision | Recall | False alarms | Missed falls | Median alarm delay |
|---|---|---|---|---|---|---|---|
| validation (reference) | 76 | 0.965 | 0.932 | 1.000 | 3 | 0 | 1.6 s |
| test GMDCSA24 S3-4 | 80 | 0.875 | 0.833 | 0.921 | 7 | 3 | 1.9 s |
| test staged (Le2i, CAUCAFall) | 57 | 0.968 | 0.968 | 0.968 | 1 | 1 | 1.5 s |
| test OOPS (real world) | 818 | 0.855 | 0.958 | 0.772 | 25 | 169 | 1.1 s |
| own recordings | 20 | 0.857 | 1.000 | 0.750 | 0 | 5 | 1.0 s |

### 4.2 Model comparison (whole-video evaluation, threshold chosen on validation; `results/*.md`)

| Model | GMDCSA24 F1 | Staged F1 | OOPS F1 |
|---|---|---|---|
| Rules v2 | 0.870 | 0.741 | 0.384 |
| Random Forest | 0.909 | 0.949 | 0.691 |
| GRU, no augmentation | 0.944 | 0.830 | 0.566 |
| GRU + augmentation | 0.914 | 0.933 | 0.798 |
| **GRU + augmentation + hard negatives ×3** | 0.947 | 0.912 | 0.763 |

Reading: augmentation is what makes the model work on real-world video (OOPS 0.57 → 0.80); hard
negatives trade a little OOPS recall for fewer false alarms on the staged sets, and gave the best
validation F1, which is why it was chosen (choice made on validation, not on these test numbers).

### 4.3 Other measurements
- "Where": alarm box on the right person in 92 % of checked alarms. "When": estimated onset within
  0.3-0.5 s of the label; alarm 1.0-1.9 s after the fall starts.
- Own recordings: night fall caught; misses are two far-away people, one fall inside a group of three
  (the faller's track splits at the fall), one fall from sitting (score 0.63 < 0.8573) and one at 0.71.

## 5. What fails and why (`results/error_analysis.md`)

| Failure | Cause |
|---|---|
| very small / far person | YOLO loses the person exactly when they fall (9 of the 12 worst OOPS misses had the person detected in only 6-21 % of frames) |
| moving / shaking camera (OOPS) | body motion in the image is not body motion |
| very fast or blurred fall | too few clear frames in the 2-s window |
| fall inside a group | tracker gives the faller a new ID; the window is split |
| false alarm: lying down on purpose, push-ups | look like "fallen" / a fast drop |
| label problems | e.g. GMDCSA24 S2 Fall/03 labelled "lie down"; 77 OOPS clips have no fall label |

## 6. Ideas tested and not kept
Night brightening (CLAHE, denoise), lower detection confidence at night, zoom re-check for small
people, input 1280, shorter rule hold time, low-confidence tracking, fall-aware track joining,
threshold 0.864 from the hard validation set, decision lag > 0, model averaging. Each is recorded with
its numbers in `results/summary.md`.

## 7. Fairness of the comparisons
- One fixed split list (`splits/videos.csv`, OmniFall official cross-subject splits + GMDCSA24 by
  subject); no subject appears in two splits.
- All models see the same windows; every threshold, score, model and setting is chosen on validation;
  test sets are run once with the chosen setting. OOPS and own videos were never used for any choice.
- Fixed seeds (GRU seed 0, Random Forest `random_state=0`); a re-run on the same machine type gives the
  same weights on CPU and nearly the same on a GPU (GPU kernels are not bit-deterministic), which is
  why the `evaluate` stage re-chooses the threshold on validation after a retrain.
