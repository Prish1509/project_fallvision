# FallVision: how to reproduce everything

Three ways, same code and same results: (A) your own machine (CPU or NVIDIA GPU), (B) Kaggle in one
notebook, (C) Kaggle in stages. Inference only (the instructor's command) needs no training: the trained
weights are in the submission ZIP.

## 0. What a re-run gives
Retraining on a GPU does not give bit-identical weights (GPU kernels are not deterministic; on CPU the
run is exactly repeatable with the fixed seeds). The pipeline therefore ends by **re-choosing the alarm
threshold on validation** for the new weights and writing it into `config.yaml`. Expect the test numbers
in `results/causal_eval.md` to differ from the committed ones by about ±0.01-0.02 F1.

## A. On your own machine

### A.1 Install (Python 3.13)
```powershell
# Windows PowerShell          (Linux / macOS: python3.13 -m venv .venv && source .venv/bin/activate)
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu   # CPU
#   NVIDIA GPU instead:   ... --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

### A.2 Weights and tests
- `models/yolo11m-pose.pt` (pretrained, in the ZIP; or `python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"`
  inside `models/` to download it once) and, for inference without retraining, `models/gru_hard.pt`.
- `python -m pytest tests -q` runs the unit tests (syntax, config, device handling, model forward, a
  2-epoch training run with checkpoints and curves, metrics, and `infer.py` on the two example clips).

### A.3 The videos
Put the datasets in one folder with this layout (made by `notebooks/06_pack_dataset.ipynb`, or by the
download notebooks 01/02; details in `docs/DATA.md`):
```
DATA/omnifall/{le2i,caucafall,mcfd,OOPS}/video/<path>.mp4
DATA/GMDCSA24/Subject 1 ... Subject 4/{Fall,ADL}/NN.mp4
DATA/own/NNN.mp4                      (optional, the team's private recordings)
```
Check it: `python src/check_dataset.py --omnifall-root DATA/omnifall --gmd-root DATA/GMDCSA24 [--own-root DATA/own]`
→ must end with `ALL VIDEOS FOUND`.

### A.4 Run the pipeline
```powershell
python run_pipeline.py --omnifall-root DATA/omnifall --gmd-root DATA/GMDCSA24 --own-root DATA/own `
                       --work work --all-models --device cuda:0        # or --device cpu
```
- Stages in order: `splits` (skipped: `splits/` is in the repository) → `poses` → `features` → `windows`
  → `train` → `choose` → `evaluate`. Use `--from STAGE --to STAGE` to run a part; `--dry-run` prints the
  commands; `--max-hours H` stops pose caching cleanly (it resumes on the next run).
- Time: pose caching is the long part (about 343,000 frames: ~6 h on a T4, ~2 days on a laptop CPU);
  features + windows minutes; training ~20 min GPU / ~1 h CPU; evaluation ~2 h GPU / ~1 day CPU.
  On a CPU laptop it is realistic to run `--from features` on pose caches made on Kaggle
  (download `work/poses` from notebook 03's Output).
- Outputs: `models/gru_hard.pt` (+ `_last.pt`), `results/gru_*.md`, `results/curves/`,
  `results/training_curves.png`, `results/fusion.md`, `results/causal_eval.md`, `results/causal_scores.csv`,
  `results/poses_index.csv`, `results/features_manifest.csv`, `results/windows_summary.md`, and
  `config.yaml` with the newly chosen threshold.

### A.5 Inference (the instructor's command) and a check
```powershell
python infer.py --source INPUT.mp4 --config config.yaml --output-dir results --device cuda:0 --no-display
python infer.py --source examples/videos --config config.yaml --output-dir results_example --device cpu --no-display
```
`--device` accepts `cpu`, `cuda`, `cuda:N` or `N`; a GPU request without CUDA falls back to CPU with a
warning in `run_summary.json`. Expected output of the example: `examples/expected_output/`.

## B. Kaggle, one notebook (`kaggle/project_kaggle.ipynb`)
See `KAGGLE_INSTRUCTIONS.md`. In short: import the notebook, add the video inputs, GPU T4, Internet on,
set `DATA` (or leave it empty to auto-find the inputs), Save & Run All. About 9-10 h. Result: `export/`
in the Output with `models/gru_hard.pt`, `config.yaml`, `results/`.

## C. Kaggle, in stages (`notebooks/03`, `04`, `05`; safer against the 12-h limit)
| Notebook | Inputs | Time | Output used next |
|---|---|---|---|
| `03_poses_features_windows` | 01 + 02 outputs (or the packed dataset) | ~6-7 h | `work/windows` |
| `04_train_models` | 03 output | ~30 min | `FallVision/models/gru_hard.pt`, curves |
| `05_evaluate` | 01 + 02 outputs (or the packed dataset) + 04 output (+ own videos) | ~2 h | `results/causal_eval.md`, `config.yaml` |

## D. Building the submission package
```powershell
# fill TEAM.md; put models/gru_hard.pt and models/yolo11m-pose.pt in models/; docs/report.pdf in docs/
python make_submission.py --team-id <TeamID>          # runs the example, writes examples/expected_output, builds <TeamID>_Project.zip
python make_submission.py --verify <TeamID>_Project.zip   # unzips to a temp folder, re-runs the example, compares alarms
```
Then test the ZIP in a fresh virtual environment on another machine (install from A.1, run A.5) with
the network off.

## E. Files that are never in git
Videos, weights (`*.pt`), pose caches, features, windows (`.gitignore`). They travel in the submission
ZIP (weights, example clips) or stay in the Kaggle outputs / the packed dataset.
