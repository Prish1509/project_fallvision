# FallVision on Kaggle

The notebooks only call the scripts in `src/` (through `run_pipeline.py`); no code is implemented inside
them, so what runs on Kaggle is exactly what runs on a laptop. Kaggle-specific things live in one place:
the **settings cell** at the top of each notebook (repository URL, where the videos are, what to run) and
`src/kaggle_helpers.py` (finding inputs, pushing result files).

## 1. One-time setup
1. **Repository.** The notebooks clone the code from GitHub (`REPO_URL` in the settings cell). The
   repository must be public, or the clone will fail.
2. **Secret (optional).** To let a notebook push its result files back to GitHub: GitHub → Settings →
   Developer settings → fine-grained token with *Contents: read and write* on the repository only; on
   Kaggle → *Add-ons → Secrets* → name `GITHUB_TOKEN`. The token is read from the secret at push time and
   never printed or stored in the notebook. Without it set `PUSH_RESULTS = False`: everything is still in
   the notebook's Output.
3. **Videos.** Run the two download notebooks once (`notebooks/01_download_omnifall_videos.ipynb`,
   `notebooks/02_download_gmdcsa24.ipynb`) with *Save & Run All*; their Outputs are the video inputs.
   Alternatively upload a dataset with the layout in `docs/DATA.md` (`notebooks/06_pack_dataset.ipynb`
   builds exactly that from the download outputs).
4. **Own recordings (optional).** A private dataset holding the clips named in
   `splits/own_videos_labels.csv`; it is found automatically when attached.

## 2. The one-notebook run: `kaggle/project_kaggle.ipynb`
1. *Create → New Notebook → File → Import Notebook* → upload `kaggle/project_kaggle.ipynb`.
2. *Add Input*: the video inputs from step 1.3 (and the own-video dataset if you have it).
3. *Settings*: Accelerator **GPU T4 x1**, Internet **On**. *Add-ons → Secrets*: `GITHUB_TOKEN` on (optional).
4. Settings cell:
   - `DATA = ""` finds the videos among the inputs automatically; or set it to the folder that holds
     `omnifall/` and `GMDCSA24/` (e.g. `/kaggle/input/datasets/<user>/<dataset>`).
   - `STAGES`: remove stages to run a part (the default runs all six).
   - `ALL_MODELS = True` also trains the comparison models (needed for the model-comparison table).
   - `POSES_MAX_HOURS = 7.5`: pose caching stops cleanly after this; the notebook then stops with a clear
     message instead of training on a partial dataset (Kaggle ends sessions after 12 h).
5. **Save Version → Save & Run All (Commit)** so it runs in the background. About 9-10 h on a T4.
6. When it finishes, open the version → **Output**:
   - `export/models/gru_hard.pt` (and `gru_hard_last.pt`), `export/config.yaml` (with the threshold chosen
     on validation for these weights), `export/results/` (all tables, `training_curves.png`), `export.zip`;
   - `work/` with every intermediate file (pose caches, features, windows, probabilities, scores).
   - If `PUSH_RESULTS = True`, `results/` and `config.yaml` are also committed to the repository.
7. Put `export/models/gru_hard.pt` into the project's `models/` folder and `export/config.yaml` at the
   project root (or `git pull` if it was pushed), and build the submission (`REPRODUCIBILITY.md` D).

Download from a terminal instead of the browser: `kaggle kernels output <user>/<notebook-slug> -p DIR`
(needs the Kaggle CLI and your `kaggle.json`).

## 3. The staged run: `notebooks/03`, `04`, `05`
Same code, split so that no single session comes near the 12-h limit; each notebook adds the previous
one's Output as an input (*Add Input → Your Work → Notebooks*):

| Notebook | Inputs | Accelerator | Time | Check in the log |
|---|---|---|---|---|
| `03_poses_features_windows` | video inputs | GPU T4 | ~6-7 h | "videos with a problem: 0", the windows table, "Pushed to" |
| `04_train_models` | output of 03 only | GPU T4 | ~30 min | the curves figure, "Chosen:" in fusion, "Pushed to" |
| `05_evaluate` | video inputs + output of 04 (+ own videos) | GPU T4 | ~2 h | the "committed version → this run" table, `infer.py gave … OK` twice |
| `06_pack_dataset` | video inputs | None (CPU) | ~30 min | "ALL VIDEOS FOUND" twice |

## 4. What to expect, and what can go wrong
- The numbers of a re-run differ slightly from the committed ones (GPU training is not bit-identical);
  the threshold is re-chosen on validation automatically. Each notebook prints the committed value next
  to the new one.
- "OmniFall videos: NOT FOUND": the download outputs are not attached, or `DATA` points to the wrong
  folder (it must contain `omnifall/` and `GMDCSA24/`).
- "GPU: NONE": turn on the accelerator; the code would run on CPU but far too slowly for 1,457 videos.
- Push failed: check the secret and the token's repository permission; the files are still in the Output.
- Never put the weights or the videos in the repository (`.gitignore` blocks `*.pt`, `*.mp4`, `work/`).
