"""Evaluation metrics, threshold choice, window labelling and the config writer."""
import os
import shutil


def rows(scores_truth):
    return [dict(split="val", has_fall=int(t), true_start_s="1.0" if t else "", max_score=s,
                 series=[(0.5, 0.0), (1.5, s)]) for s, t in scores_truth]


def test_metrics_counts_and_delay():
    from eval_causal import metrics
    m = metrics(rows([(0.9, 1), (0.2, 1), (0.95, 0), (0.1, 0)]), 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert abs(m["precision"] - 0.5) < 1e-9 and abs(m["recall"] - 0.5) < 1e-9 and abs(m["f1"] - 0.5) < 1e-9
    assert abs(m["delay"] - 0.5) < 1e-9                       # first crossing at 1.5 s, fall started at 1.0 s


def test_best_threshold_prefers_fewer_false_alarms_and_average_precision():
    from eval_causal import average_precision, best_threshold
    r = rows([(0.9, 1), (0.8, 1), (0.7, 0), (0.3, 0)])
    thr, m = best_threshold(r)
    assert m["f1"] == 1.0 and 0.7 < thr <= 0.8
    assert average_precision(r) == 1.0
    assert average_precision(rows([(0.9, 0), (0.8, 1)])) == 0.5


def test_window_labels():
    from make_windows import window_label, end_label, FALL, FALLEN
    segs = [(0, 0.0, 5.0), (FALL, 5.0, 6.0), (FALLEN, 6.0, 9.0)]
    assert window_label(segs, 2.0, 4.0) == 0
    assert window_label(segs, 4.5, 6.5) == 1          # overlaps the fall by 1 s
    assert window_label(segs, 6.5, 8.5) == 2          # ends while lying after the fall
    assert end_label(segs, 7.0) == FALLEN and end_label(segs, 20.0) == -1


def test_write_config_keeps_comments(root, tmp_path):
    from eval_causal import write_config
    from fv_engine import load_config
    dst = tmp_path / "config.yaml"
    shutil.copyfile(os.path.join(root, "config.yaml"), dst)
    write_config(str(dst), 0.5, 0.7777)
    cfg = load_config(str(dst))
    assert cfg["threshold"] == 0.7777 and cfg["decision_lag_s"] == 0.5
    assert "# alarm when the score reaches this" in open(dst).read()
