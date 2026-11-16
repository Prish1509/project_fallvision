# Team

| | |
|---|---|
| **Team ID** | 142502033_142301011_142613001 |
| **Project** | Person Collapse Detection in Video |
| **Course** | DS5602 Computer Vision |

| Member | Name | Roll number | Contributions |
|---|---|---|---|
| Member 1 | Vaibhav Singh | 142502033 | Data and evaluation: labelled 18 of the team's own videos (fall type, start time, lighting, distance, number of people) in the label sheet used by the evaluation scripts; manual localization study (36 alarm frames checked by eye: is the box on the person who fell), which gives the 92 % figure; watched and classified the missed falls and false alarms for the error analysis. |
| Member 2 | Prisha Italiya | 142301011 | Design and implementation of the system: OmniFall-based splits, pose caching with YOLO11m-pose and ByteTrack, the 92 skeleton features and 2-s windows, the GRU models (augmentation, hard negatives) and the Random Forest baseline, model and threshold selection on validation, the frame-by-frame inference engine and `infer.py`, the evaluation scripts, the night / far-away robustness experiments, the Kaggle pipeline notebooks, documentation and the submission package; repository owner. |
| Member 3 | T.S. Hrushikesh | 142613001 | Test data: planned and clipped the 20 own test videos (day and night, near and far, one to nine people; falls from standing, walking, sitting and jumping) following the recording plan, kept in a private dataset; ran the own-video checker and the own-video evaluations and reported their results. |