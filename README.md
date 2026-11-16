# FallVision: person collapse detection

DS5602 Computer Vision course project. **Project: person collapse detection.**
Team, names, roll numbers and contributions: see [`TEAM.md`](TEAM.md).
Approaches and experiments: [`APPROACHES_AND_EXPERIMENTS.md`](APPROACHES_AND_EXPERIMENTS.md). Workflow:
[`METHODOLOGY.md`](METHODOLOGY.md). Re-running everything: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md),
[`KAGGLE_INSTRUCTIONS.md`](KAGGLE_INSTRUCTIONS.md). Report: `docs/report.md` / `docs/report.pdf`.

The system finds every person in a video with **YOLO11m-pose** (17 body joints) and **ByteTrack**,
turns each person's last 2 seconds of skeletons into 92 posture and motion features, and a small
**GRU** network we trained decides *normal / falling / fallen*. When a person's collapse score reaches
the threshold, an alarm is raised with **who** (person ID), **where** (box) and **when** (decision time,
plus a separate estimate of when the collapse started). Video is processed frame by frame using only
the current and past frames.

---

## 1. Requirements

| | Tested with |
|---|---|
| Python | 3.13 (use 3.13; the pinned NumPy has no wheels for newer versions) |
| OS | Linux (Kaggle, glibc 2.39) for the evaluation; Windows 11 for the fresh-environment check |
| GPU (optional) | NVIDIA Tesla T4 (16 GB), CUDA 13.0, driver 580 |
| CPU-only | works (`--device cpu`), about 2-3 frames/s on a laptop CPU with PyTorch |
| RAM / disk | 4 GB RAM; about 3 GB disk for the Python packages, 45 MB for the weights |
| Internet | **not needed** for inference (all weights are in `models/`) |

## 2. Installation (exact commands)

**Windows (PowerShell)**
```powershell
cd TeamID_Project
py -3.13 -m venv .venv
.venv\Scripts\activate
# CPU only (no NVIDIA GPU): CPU build of torch first, then the rest
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# with an NVIDIA GPU instead: CUDA 12.8 build of torch, then the rest
#   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
#   pip install -r requirements.txt
```

**Linux / macOS**
```bash
cd TeamID_Project
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt                     # Linux + NVIDIA GPU: the PyPI torch wheel includes CUDA
# CPU only (or macOS):
#   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements.txt
```

Check: `python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"`, and
`python -m pytest tests -q` (unit tests + `infer.py` on the example clips, about 2 minutes on a CPU).

Weights included in `models/`: `gru_hard.pt` (our trained GRU) and `yolo11m-pose.pt` (Ultralytics
YOLO11m-pose, pretrained on COCO keypoints). Nothing is downloaded at run time.

## 3. Inference (the evaluation command)

```bash
python infer.py --source INPUT.mp4 --config config.yaml --output-dir results --device cuda:0 --no-display
```

| Argument | Meaning |
|---|---|
| `--source` | a video file, or a folder of videos (each video is processed separately, state reset) |
| `--config` | `config.yaml`: every setting; nothing is set in the code |
| `--output-dir` | where the three output files are written |
| `--device` | **`cpu`**, **`cuda`** (= first GPU), **`cuda:N`** (GPU number N), or **`N`**. If a GPU is requested but CUDA is not available, it runs on `cpu` and records a warning in `run_summary.json` |
| `--no-display` | no window. Without it a preview window is shown when a screen exists; nothing is ever asked interactively |
| `--save-video` | optional: also write `<source_id>_annotated.mp4` (boxes, skeletons, red COLLAPSE label) |

### Output files
- **`predictions.jsonl`**: one record per processed frame:
  `{"source_id", "frame_index", "timestamp_s", "predictions": [...]}`. `predictions` is an empty list
  when nobody is collapsing; otherwise one entry per collapsed person:
  `{"label": "person_collapse", "confidence", "person_id", "bbox_xyxy"}`.
- **`alarms.jsonl`**: one record per newly raised alarm, written the moment it is raised:
  `{"source_id", "event_id", "label": "person_collapse", "decision_timestamp_s", "confidence",
  "frame_index", "onset_estimate_s", "person_id", "bbox_xyxy"}`.
  `decision_timestamp_s` is the time of the latest frame used for the alarm; the estimated earlier
  start of the collapse is recorded separately as `onset_estimate_s`.
- **`run_summary.json`**: `status` (`completed` / `completed_with_errors` / `failed`), `frames_read`,
  `frames_processed`, `frames_skipped`, `alarms`, `total_runtime_s`, `errors`, `warnings`, device used
  and the same figures per video.

### How frames are processed
Frames are read in order. A frame is processed when it falls on the next slot of a 15-per-second time
grid (`frame_sampling.model_fps`), whatever the video's frame rate; the others are counted as
skipped. Each processed frame uses only itself and earlier frames; alarms are emitted as they occur;
the tracker and all temporal state are reset for every new video.

## 4. Verification example

Two short shareable clips from GMDCSA24 (MIT licence) are in `examples/videos/` (in the submission
ZIP; videos are not stored in git):

| Clip | Content | Expected |
|---|---|---|
| `event_fall_gmdcsa24_s3_fall02.mp4` | person walks, stands, falls at about 2.6 s | **1 alarm** `person_collapse` |
| `normal_sitting_gmdcsa24_s4_adl13.mp4` | person sitting | **no alarm** |

```bash
python infer.py --source examples/videos --config config.yaml --output-dir results_example --device cpu --no-display
```

The expected output files are in `examples/expected_output/` (`predictions.jsonl`, `alarms.jsonl`,
`run_summary.json`). Times and confidences can differ in the last digits between CPU and GPU.
`python make_submission.py --verify <ZIP>` runs this check automatically on an unzipped package.

## 5. Configuration (`config.yaml`)

| Section | Settings |
|---|---|
| `models` | pose model and classifier paths |
| `pose` | input size 640 (960 when people are small: standing height < 150 px in a 1280-wide frame), detection confidence 0.25, ByteTrack |
| `frame_sampling` | 15 frames per second |
| `temporal` | 2-s windows, every 0.5 s, 10 s of history per person, track joining within 1.5 s, decision lag 0 s |
| `alarm` | label `person_collapse`, score = P(falling) + P(fallen) averaged with the previous window, **threshold 0.8573**, re-arm after 10 s below the threshold |

All values were chosen on validation videos only (`results/causal_eval.md`, `results/causal_eval_lags.md`).

## 6. Results of the submitted system

Frame-by-frame system (`infer.py`), threshold chosen on validation, each test set run once
(`results/causal_eval.md`):

| Test set | Videos | F1 | Precision | Recall | False alarms | Missed falls | Median alarm delay |
|---|---|---|---|---|---|---|---|
| GMDCSA24 Subjects 3-4 | 80 | 0.875 | 0.833 | 0.921 | 7 | 3 | 1.9 s |
| Staged (Le2i, CAUCAFall) | 57 | 0.968 | 0.968 | 0.968 | 1 | 1 | 1.5 s |
| OOPS real-world | 818 | 0.855 | 0.958 | 0.772 | 25 | 169 | 1.1 s |
| Own recordings | 20 | 0.857 | 1.000 | 0.750 | 0 | 5 | 1.0 s |

Training and validation curves (loss and window macro-F1 per epoch): `results/training_curves.png`.
Speed: 29.6 frames/s on a Tesla T4 (2x real time). Localization: the alarm box is on the person who
fell in 92 % of checked alarms. Details, every experiment and the failure analysis: `results/` (start
with `results/README.md`) and `docs/PROJECT.md`.

## 7. Reproducing the whole pipeline (training and evaluation)

Everything after the video download is one command on any machine (CPU or NVIDIA GPU; the device is
detected and a GPU request falls back to CPU with a warning):

```bash
python run_pipeline.py --omnifall-root DATA/omnifall --gmd-root DATA/GMDCSA24 --work work --all-models --device cuda:0
```

On Kaggle the same pipeline runs in **one notebook, `kaggle/project_kaggle.ipynb`** (set the dataset
location in its first cell; `KAGGLE_INSTRUCTIONS.md`), or in stages with the notebooks below. Each
notebook adds the output of the earlier ones as an input; the pipeline notebooks only call
`run_pipeline.py` (all the work is done by the scripts in `src/`) and push their result files to the
repository. Exact steps: `REPRODUCIBILITY.md`.

| Notebook | What it does | Inputs | Time |
|---|---|---|---|
| `01_download_omnifall_videos.ipynb` | downloads Le2i, CAUCAFall, MCFD and OOPS videos | - | |
| `02_download_gmdcsa24.ipynb` | downloads GMDCSA24 | - | |
| `03_poses_features_windows.ipynb` | stages `poses`, `features`, `windows` | 01, 02 | about 6-7 h |
| `04_train_models.ipynb` | stages `train` (+ learning curves) and `choose` | 03 | about 30 min |
| `05_evaluate.ipynb` | stage `evaluate`, writes the chosen lag and threshold into `config.yaml`, runs `infer.py` on two test videos | 01, 02, 04 (+ own videos, optional) | about 2 h |
| `06_pack_dataset.ipynb` | copies exactly the 1,457 videos used (+ own videos, optional) into `FallVision_dataset/` with labels, split list and MD5 manifest (`src/check_dataset.py`) | 01, 02 | about 30 min, CPU |

The packed `FallVision_dataset/` can replace the outputs of 01 and 02 as the video input of 03 and 05.
Data description, licences and folder layout: [`docs/DATA.md`](docs/DATA.md). To check a local copy:
`python src/check_dataset.py --omnifall-root DATA/omnifall --gmd-root DATA/GMDCSA24`.

`run_pipeline.py` runs the same stages on any machine; each stage is one script in `src/`:

```bash
python run_pipeline.py --omnifall-root OMNI --gmd-root GMD --work work --all-models --device cuda:0   # all stages
python run_pipeline.py --work work --from train --to choose --all-models                              # stages without videos
python run_pipeline.py --omnifall-root OMNI --gmd-root GMD --dry-run                                  # show the commands
```

| Stage | Script | Output |
|---|---|---|
| `splits` | `src/make_splits.py` | train / val / test lists (already in `splits/`; re-made only with `--remake-splits`) |
| `poses` | `src/cache_poses.py` | YOLO11m-pose + ByteTrack skeletons per video (GPU, about 6 h); `results/poses_index.csv` |
| `features` | `src/preprocess.py` | 92 features per frame per person; `results/features_manifest.csv` |
| `windows` | `src/make_windows.py` | 2-s windows labelled normal / falling / fallen; `results/windows_summary.md` |
| `train` | `src/train_gru.py` (+ `--all-models`: comparison GRUs and Random Forest), `src/plot_curves.py` | `models/gru_hard.pt`, per-epoch history in `results/curves/`, `results/training_curves.png` |
| `choose` | `src/fuse.py` | every model / score / threshold compared on validation (`results/fusion.md`) |
| `evaluate` | `src/eval_causal.py` | the submitted system on validation and every test set (`results/causal_eval.md`); the lag and threshold chosen on validation are written into `config.yaml` |

Retraining on a GPU never gives bit-identical weights, so a re-run gives slightly different numbers;
`evaluate` re-chooses the threshold on validation for the new weights. Data: Le2i, CAUCAFall, MCFD and
OOPS with OmniFall labels and official splits (`simplexsigil2/omnifall` on Hugging Face), GMDCSA24 from
its GitHub release.

## 8. Other programs (not needed for grading)

| Command | What it does |
|---|---|
| `python app/run_fall.py` | test videos interactively; models load once; results in `outputs/` |
| `python app/live_fall.py` | live: webcam, a video played as it runs, a phone camera (`http://PHONE-IP:8080/video`) or CCTV (`rtsp://...`); `--serve` shows the live picture on a phone browser |

## 9. Repository layout and components

```
FallVision/
├── infer.py                 the evaluation entry point (instruction 6)
├── config.yaml              the one final configuration (instruction 5)
├── requirements.txt         exact package versions (instruction 4)
├── run_pipeline.py          single entry point for the whole pipeline; --from/--to run single stages
├── make_submission.py       builds and verifies <TeamID>_Project.zip
├── src/                     all implementation (flat, one script per step; grouped below)
├── app/                     interactive / live programs (not needed for grading)
├── tests/                   pytest suite: python -m pytest tests -q
├── kaggle/project_kaggle.ipynb   the whole pipeline in one Kaggle notebook
├── notebooks/               01-02 data download, 03-05 the pipeline in stages, 06 dataset packing
├── labels/, splits/         OmniFall labels; train / val / test lists; own-video labels
├── results/                 every result table and figure (index: results/README.md)
├── docs/                    DATA.md, PROJECT.md (long-form explanation), report, recording plan
├── examples/                verification clips + expected outputs (clips are in the ZIP, not in git)
└── models/                  gru_hard.pt, yolo11m-pose.pt (in the ZIP, not in git)
```

| Component | Files in `src/` |
|---|---|
| Data loading and splits | `make_splits.py`, `check_dataset.py` (+ `notebooks/01`, `02`, `06`) |
| Preprocessing: pose + tracking | `cache_poses.py`, `baseline_fall.py` (`extract_tracks`), `small_person.py` |
| Feature engineering | `preprocess.py` (92 features), `make_windows.py` (2-s windows, 3 classes) |
| Models and baselines | `train_gru.py` (GRU, augmentation, hard negatives), `train_rf.py` (Random Forest), `baseline_fall.py` / `detect_falls.py` (rules) |
| Training and validation | `train_gru.py` (checkpoints, per-epoch history), `fuse.py` (choice on validation) |
| Evaluation and metrics | `eval_causal.py` (submitted system), `eval_system.py`, `eval_own.py`, `eval_gmdcsa24.py`, `eval_rules_splits.py`, `loc_sheets.py`, `speed_test.py` |
| Visualization and comparison | `plot_curves.py` (training curves), `fuse.py` (model table), `results/README.md` |
| Robustness experiments | `simulate_night.py`, `simulate_far.py`, `night_analysis.py`, `make_hard_val.py`, `check_model.py` |
| Inference engine and utilities | `fv_engine.py` (causal engine, config, device), `kaggle_helpers.py` (notebook helpers) |

Documentation: `APPROACHES_AND_EXPERIMENTS.md` (every approach and comparison), `METHODOLOGY.md` (raw
video → results), `REPRODUCIBILITY.md` (exact steps, local and Kaggle), `KAGGLE_INSTRUCTIONS.md`,
`docs/DATA.md` (datasets), `docs/PROJECT.md` (long-form), `docs/report.md` (the brief report).

## 10. Third-party models, data and licences
- **YOLO11m-pose** and **ByteTrack** via Ultralytics (AGPL-3.0).
- **GMDCSA24** (MIT licence): the two example clips come from it.
- **Le2i, CAUCAFall, MCFD, OOPS**: research / non-commercial use; not redistributed here.
- **OmniFall** labels and splits.
- Our own recordings are private and not included.
