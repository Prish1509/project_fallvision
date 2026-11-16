# FallVision data

Videos used to train, choose and test FallVision (person collapse detection), DS5602 course project.
This file is copied into the packed dataset folder by `src/check_dataset.py --pack-to`.

## What is in it

| Dataset | train | val | test_gmdcsa24 | test_staged | test_oops | total | Licence / use |
|---|---|---|---|---|---|---|---|
| GMDCSA24 | 32 (Subject 1) | 48 (Subject 2) | 80 (Subjects 3-4) | - | - | 160 | MIT |
| Le2i | 133 | 19 | - | 38 | - | 190 | research use only |
| CAUCAFall | 69 | 9 | - | 19 | - | 97 | research use only |
| MCFD | 192 | - | - | - | - | 192 | research use only |
| OOPS (fall clips) | - | - | - | - | 818 | 818 | non-commercial research only |
| **total** | 426 | 76 | 80 | 57 | 818 | **1457** | |
| Own recordings (optional) | - | - | - | - | - | 20 (`test_own`) | the team's private clips |

- **Labels and splits:** OmniFall (`simplexsigil2/omnifall` on Hugging Face): frame-accurate activity
  labels and the official cross-subject splits for all five datasets. Our split list is
  `videos.csv` (= `splits/videos.csv` of the code); the labels are in `labels/`.
- **Own recordings:** labels in `own_videos_labels.csv` (fall start / end time, setting, notes). Test only.
- The Le2i, CAUCAFall, MCFD and OOPS videos belong to their authors. They are shared here only for
  grading this course project and must not be redistributed or used commercially. Only GMDCSA24 (MIT)
  may be shared freely.

## Folder layout (what the code reads)

```
omnifall/le2i/video/<path>.mp4         <path> as in videos.csv, e.g. "Coffee_room_01/Videos/video (1)"
omnifall/caucafall/video/<path>.mp4
omnifall/mcfd/video/<path>.mp4
omnifall/OOPS/video/<path>.mp4
GMDCSA24/Subject 1 ... Subject 4/{Fall,ADL}/NN.mp4
own/NNN.mp4                            (only if included)
manifest.csv                           size and MD5 of every video
```

## Using it with the code

```bash
# check that every video is there
python src/check_dataset.py --omnifall-root DATA/omnifall --gmd-root DATA/GMDCSA24 --own-root DATA/own

# the whole pipeline (GPU strongly advised: pose caching is about 343,000 frames)
python run_pipeline.py --omnifall-root DATA/omnifall --gmd-root DATA/GMDCSA24 --own-root DATA/own \
                       --work work --all-models --device cuda:0
```

On Kaggle the same videos come from the outputs of `notebooks/01_download_omnifall_videos.ipynb` and
`notebooks/02_download_gmdcsa24.ipynb`, and notebooks 03-05 run the pipeline on them.

## Where the videos came from

- Le2i, CAUCAFall, MCFD, OOPS: downloaded by `notebooks/01_download_omnifall_videos.ipynb` (OOPS from a
  Hugging Face mirror, because the original link no longer works).
- GMDCSA24: its GitHub release, downloaded by `notebooks/02_download_gmdcsa24.ipynb`.
- Own recordings: filmed by the team on phones (day and night, near and far, 1-9 people).
