"""
Build the course submission ZIP (TeamID_Project.zip) and check it.

    python make_submission.py --team-id T07              # build T07_Project.zip next to this folder
    python make_submission.py --verify T07_Project.zip   # unzip in a temporary folder, run the example, compare

Building:
  1. checks that TEAM.md is filled in and both weights are in models/;
  2. runs infer.py on examples/videos/ (cpu) and stores the outputs in examples/expected_output/;
  3. zips code (src/, app/, tests/), config.yaml, requirements.txt, the documentation, kaggle/, notebooks/,
     results/, labels/, splits/, examples/ and models/{gru_hard.pt, yolo11m-pose.pt} into <TeamID>_Project/.
     Videos you tested with (test_videos/), outputs/, caches and the OpenVINO copy are left out.
Verifying: runs the example from the unzipped copy (no internet needed) and checks that it raises the
same alarms as examples/expected_output/alarms.jsonl (same videos, decision time within 0.2 s).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
INCLUDE = ["infer.py", "config.yaml", "requirements.txt", "README.md", "TEAM.md", "make_submission.py", "run_pipeline.py",
           "APPROACHES_AND_EXPERIMENTS.md", "METHODOLOGY.md", "REPRODUCIBILITY.md", "KAGGLE_INSTRUCTIONS.md", ".gitignore",
           "app", "src", "tests", "kaggle", "docs", "notebooks", "results", "labels", "splits", "examples",
           "models/gru_hard.pt", "models/yolo11m-pose.pt"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".git", ".pytest_cache"}
EXAMPLE_CMD = ["infer.py", "--source", "examples/videos", "--config", "config.yaml",
               "--output-dir", "{out}", "--device", "cpu", "--no-display"]


def run_example(root, out):
    cmd = [sys.executable] + [c.format(out=out) for c in EXAMPLE_CMD]
    print("Running:", " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=root)
    if r.returncode != 0:
        sys.exit("The example run failed; fix this before submitting.")
    return [json.loads(l) for l in open(os.path.join(out, "alarms.jsonl")) if l.strip()]


def build(a):
    team = open(os.path.join(ROOT, "TEAM.md")).read() if os.path.exists(os.path.join(ROOT, "TEAM.md")) else ""
    if not team or "<" in team:
        sys.exit("Fill in TEAM.md first (team ID, names, roll numbers, contributions; no <...> left).")
    for w in ("models/gru_hard.pt", "models/yolo11m-pose.pt"):
        if not os.path.isfile(os.path.join(ROOT, w)):
            sys.exit(f"Missing {w}: both weights must be in the package (instruction 3).")
    for nb in ("01_*.ipynb", "02_*.ipynb", "03_*.ipynb", "04_*.ipynb", "05_*.ipynb", "06_*.ipynb"):
        if not glob.glob(os.path.join(ROOT, "notebooks", nb)):
            print(f"Warning: notebooks/{nb} is missing.")
    if not glob.glob(os.path.join(ROOT, "docs", "*report*")) and not glob.glob(os.path.join(ROOT, "docs", "*Report*")):
        print("Warning: no report found in docs/.")

    exp = os.path.join(ROOT, "examples", "expected_output")
    if not a.skip_expected:
        if os.path.isdir(exp):
            shutil.rmtree(exp)
        alarms = run_example(ROOT, exp)
        print(f"Expected output written to examples/expected_output ({len(alarms)} alarm(s))")
        n_event = sum(a["source_id"].startswith("event_") for a in alarms)
        n_normal = sum(a["source_id"].startswith("normal_") for a in alarms)
        if n_event == 0 or n_normal > 0:
            print("WARNING: the event clip should give 1 alarm and the normal clip none. Got event "
                  f"{n_event}, normal {n_normal}. Is models/gru_hard.pt the trained model and config.yaml its threshold?")

    name = f"{a.team_id}_Project"
    out = os.path.abspath(a.out or os.path.join(os.path.dirname(ROOT), name + ".zip"))
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            src = os.path.join(ROOT, item)
            if not os.path.exists(src):
                print(f"  (not found, skipped: {item})")
                continue
            files = [src] if os.path.isfile(src) else [os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs]
            for f in sorted(files):
                rel = os.path.relpath(f, ROOT)
                if SKIP_PARTS & set(rel.replace("\\", "/").split("/")) or f.endswith(".pyc"):
                    continue
                z.write(f, f"{name}/{rel}".replace("\\", "/"))
                n += 1
    print(f"\nBuilt {out}: {n} files, {os.path.getsize(out) / 1e6:.1f} MB")
    print(f"Next: python make_submission.py --verify \"{out}\"")


def verify(a):
    tmp = tempfile.mkdtemp(prefix="fallvision_verify_")
    with zipfile.ZipFile(a.verify) as z:
        z.extractall(tmp)
    roots = [d for d in glob.glob(os.path.join(tmp, "*")) if os.path.isdir(d)]
    if len(roots) != 1:
        sys.exit("The ZIP should contain exactly one top folder.")
    root = roots[0]
    exp = [json.loads(l) for l in open(os.path.join(root, "examples", "expected_output", "alarms.jsonl")) if l.strip()]
    got = run_example(root, os.path.join(tmp, "check"))
    key = lambda r: (r["source_id"], round(r["decision_timestamp_s"], 1))
    ok = len(got) == len(exp) and all(g["source_id"] == e["source_id"] and
                                      abs(g["decision_timestamp_s"] - e["decision_timestamp_s"]) <= 0.2
                                      for g, e in zip(sorted(got, key=key), sorted(exp, key=key)))
    summary = json.load(open(os.path.join(tmp, "check", "run_summary.json")))
    print(f"\nrun_summary status: {summary['status']}, errors: {summary['errors']}")
    print("Expected alarms:", [key(r) for r in exp])
    print("Got alarms     :", [key(r) for r in got])
    print("VERIFY OK: the unzipped package reproduces the expected output." if ok and summary["status"] == "completed"
          else "VERIFY FAILED: see above.")
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="Build or verify the submission ZIP")
    ap.add_argument("--team-id", help="e.g. T07 -> T07_Project.zip")
    ap.add_argument("--out", default=None, help="where to write the ZIP (default: next to the project folder)")
    ap.add_argument("--skip-expected", action="store_true", help="keep the existing examples/expected_output")
    ap.add_argument("--verify", metavar="ZIP", help="check a built ZIP instead of building")
    a = ap.parse_args()
    if a.verify:
        verify(a)
    elif a.team_id:
        build(a)
    else:
        ap.error("give --team-id to build, or --verify ZIP")


if __name__ == "__main__":
    main()
