# Results index

**Final numbers of the submitted system** (`infer.py`, frame by frame, threshold 0.8573):
- `causal_eval.md`: threshold chosen on validation, every test set run once  ← **report these**
- `causal_eval_lags.md`: decision lag 0 / 0.5 / 1.0 / 1.5 s compared on validation (0 s chosen)
- `causal_scores.csv`: per-video collapse score over time behind both tables
- `env_kaggle.txt`: exact environment (Python, CUDA, GPU, package versions) of that evaluation

**Training:** `training_curves.png` / `training_curves.md`: training and validation loss and window
macro-F1 per epoch (final GRU and the two comparison GRUs); per-epoch numbers in `curves/`. The
orange star is the submitted `gru_hard.pt` measured the same way.

**Story of all experiments:** `summary.md` (in order, with every decision).

| Files | What they are |
|---|---|
| `fusion.md` | model / score / threshold choice on validation (`src/fuse.py`) |
| `gru.md`, `gru_hard.md`, `gru_noaug.md`, `rf.md`, `rules_splits.md` (+ `_videos.csv`) | each model on every split |
| `system_final.md`, `system_own_final.md` (+ `_videos.csv`) | final model, earlier whole-video evaluation |
| `system_final864*`, `system_own_final864*` | a higher threshold (0.864) tried once and not kept |
| `system_hv_*` | hard validation set experiments (far / crowd), Step 27 |
| `error_analysis.md` | 12 missed falls and 6 false alarms watched by hand |
| `localization.md`, `loc_check.csv` | "where": is the alarm box on the person who fell (by eye) |
| `speed.md` | frames per second on a Tesla T4 |
| `windows_summary.md`, `features_manifest.csv` | training data sizes |
| `night_*`, `far_*`, `day_*`, `gmdcsa24_*` | rule-baseline experiments (Steps 7-16) |
| `system_all*`, `system_own.md`, `system_own_fix*` | earlier runs, kept as a record (superseded) |
| `demo/` | CSVs of the first demo videos |
