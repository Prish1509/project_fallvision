"""infer.py on the two verification clips (CPU, about 1-2 minutes): output files and their format.
Needs models/gru_hard.pt and models/yolo11m-pose.pt (they are in the submission ZIP, not in git)."""
import json
import os
import subprocess
import sys

import pytest


def test_infer_outputs(root, tmp_path):
    models = [os.path.join(root, "models", m) for m in ("gru_hard.pt", "yolo11m-pose.pt")]
    videos = os.path.join(root, "examples", "videos")
    if not all(os.path.isfile(m) for m in models) or not os.path.isdir(videos):
        pytest.skip("weights or example videos not present (they are only in the submission ZIP)")
    out = tmp_path / "results"
    r = subprocess.run([sys.executable, "infer.py", "--source", videos, "--config", "config.yaml",
                        "--output-dir", str(out), "--device", "cpu", "--no-display"],
                       cwd=root, capture_output=True, text=True, env={**os.environ, "YOLO_OFFLINE": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.load(open(out / "run_summary.json"))
    assert summary["status"] == "completed" and summary["errors"] == []
    assert summary["frames_processed"] > 0 and summary["frames_read"] >= summary["frames_processed"]
    assert summary["frames_read"] == summary["frames_processed"] + summary["frames_skipped"]
    preds = [json.loads(l) for l in open(out / "predictions.jsonl") if l.strip()]
    assert len(preds) == summary["frames_processed"]
    for p in preds:
        assert set(p) >= {"source_id", "frame_index", "timestamp_s", "predictions"}
        for q in p["predictions"]:
            assert q["label"] == "person_collapse" and 0 <= q["confidence"] <= 1
    alarms = [json.loads(l) for l in open(out / "alarms.jsonl") if l.strip()]
    by_video = {}
    for a in alarms:
        assert set(a) >= {"source_id", "event_id", "label", "decision_timestamp_s", "confidence", "onset_estimate_s"}
        assert a["label"] == "person_collapse" and a["onset_estimate_s"] <= a["decision_timestamp_s"]
        by_video[a["source_id"]] = by_video.get(a["source_id"], 0) + 1
    # the alarms must match the expected output stored with the package (made by make_submission.py
    # with the submitted weights); without that folder only the formats above are checked
    exp_file = os.path.join(root, "examples", "expected_output", "alarms.jsonl")
    if os.path.isfile(exp_file):
        exp = [json.loads(l) for l in open(exp_file) if l.strip()]
        key = lambda a: (a["source_id"], round(a["decision_timestamp_s"], 1))
        assert len(alarms) == len(exp), (alarms, exp)
        for g, e in zip(sorted(alarms, key=key), sorted(exp, key=key)):
            assert g["source_id"] == e["source_id"] and abs(g["decision_timestamp_s"] - e["decision_timestamp_s"]) <= 0.2
    else:
        print("examples/expected_output not present: alarm counts not compared (run make_submission.py --team-id)")
    # per-video timestamps are increasing (sequential processing)
    for sid in by_video:
        ts = [p["timestamp_s"] for p in preds if p["source_id"] == sid]
        assert ts == sorted(ts)
