"""
Small helpers for the Kaggle notebooks (notebooks/03, 04, 05): find the input folders whatever
Kaggle calls them, get the pretrained YOLO weights, record the environment, and push result files
to GitHub with the token from Kaggle Secrets (the token is never printed or saved).

Nothing here changes how the system works; the pipeline itself is in the other scripts of src/.
"""

import csv
import os
import platform
import shutil
import subprocess
import sys

INPUT = "/kaggle/input"
OMNI_SETS = ("le2i", "caucafall", "mcfd", "OOPS")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def _walk(root, max_depth=8):
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for d, subs, files in os.walk(root, followlinks=True):
        if d.count(os.sep) - base >= max_depth or os.path.basename(d) == "video":
            subs[:] = []                       # do not list thousands of video folders
        subs.sort()
        yield d, subs, files


def find_omnifall(root=INPUT):
    """Folder that holds le2i/ caucafall/ mcfd/ OOPS/ (each with a video/ folder)."""
    best = (0, None)
    for d, subs, _ in _walk(root):
        n = sum(os.path.isdir(os.path.join(d, s, "video")) for s in OMNI_SETS)
        if n > best[0]:
            best = (n, d)
    if best[1]:
        missing = [s for s in OMNI_SETS if not os.path.isdir(os.path.join(best[1], s, "video"))]
        print(f"OmniFall videos: {best[1]}" + (f"   (missing: {', '.join(missing)})" if missing else "   (all 4 datasets)"))
    else:
        print("OmniFall videos: NOT FOUND - add the output of 01_download_omnifall_videos as an input")
    return best[1]


def find_gmd(root=INPUT):
    """GMDCSA24 folder with 'Subject 1' ... 'Subject 4'."""
    for d, subs, _ in _walk(root):
        if all(f"Subject {i}" in subs for i in (1, 2, 3, 4)):
            print("GMDCSA24 videos:", d)
            return d
    print("GMDCSA24 videos: NOT FOUND - add the output of 02_download_gmdcsa24 as an input")
    return None


def find_own(labels="splits/own_videos_labels.csv", root=INPUT, skip=()):
    """Folder with our own test videos (optional: private, only on the team's Kaggle account)."""
    names = {r["file"].strip() for r in csv.DictReader(open(labels, newline="")) if r.get("file")}
    skip = [os.path.abspath(s) for s in skip if s]
    hits = []
    for d, _, files in _walk(root):
        if any(os.path.abspath(d).startswith(s + os.sep) or os.path.abspath(d) == s for s in skip):
            continue
        hits += [os.path.join(d, f) for f in files if f in names and f.lower().endswith(VIDEO_EXT)]
    if not hits:
        print("Own videos: not found (optional; the test_own rows are skipped)")
        return None
    own = os.path.commonpath([os.path.dirname(h) for h in hits])
    print(f"Own videos: {own}   ({len({os.path.basename(h) for h in hits})} of {len(names)} found)")
    return own


def find_dir_with(*files, root=INPUT):
    """First folder under the inputs that contains all the given file names."""
    for d, _, fs in _walk(root):
        if all(f in fs for f in files):
            return d
    return None


def find_file(name, root=INPUT):
    for d, _, fs in _walk(root):
        if name in fs:
            return os.path.join(d, name)
    return None


def get_yolo(dst="models/yolo11m-pose.pt"):
    """Pretrained YOLO11m-pose weights: from an input if one has them, else downloaded by Ultralytics."""
    name = os.path.basename(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if not os.path.isfile(dst):
        src = find_file(name)
        if src:
            shutil.copyfile(src, dst)
            print(f"{name}: copied from {src}")
        else:
            here = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(dst)))
            try:
                from ultralytics import YOLO
                YOLO(name)                                  # Ultralytics downloads the official weights
            finally:
                os.chdir(here)
            if not os.path.isfile(dst):
                sys.exit(f"{name} could not be downloaded: turn Internet on, or add an input that has it")
            print(f"{name}: downloaded by Ultralytics")
    print(f"{dst}: {os.path.getsize(dst) / 1e6:.1f} MB")
    return dst


def env_report(path="results/env_kaggle.txt"):
    """Python, OS, torch / CUDA, GPU and package versions of this run."""
    import importlib.metadata as md
    import torch
    lines = [f"python {platform.python_version()}", f"os {platform.platform()}",
             f"torch {torch.__version__} cuda {torch.version.cuda}"]
    pkgs = ["lap", "matplotlib", "numpy", "opencv-python", "opencv-python-headless", "pandas", "pillow", "polars",
            "psutil", "PyYAML", "requests", "scikit-learn", "scipy", "torch", "torchvision", "tqdm",
            "ultralytics", "ultralytics-thop"]
    for p in pkgs:
        try:
            lines.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            pass
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        lines.append(gpu or "no GPU")
    except (OSError, subprocess.SubprocessError):
        lines.append("no GPU (nvidia-smi not available)")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def push(repo_dir, paths, message, repo_url, branch="main", secret="GITHUB_TOKEN"):
    """git add <paths> (only those that exist), commit and push. Returns True on success, never raises."""
    def git(*args, hide=None):
        r = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        if hide:
            out = out.replace(hide, "***")
        return r.returncode, out

    paths = [p for p in paths if os.path.exists(os.path.join(repo_dir, p))]
    if not paths:
        print("Nothing to push.")
        return False
    target = repo_url
    if repo_url.startswith("https://github.com/"):
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret(secret)
        except Exception as e:                                   # no secret attached / not on Kaggle
            print(f"Not pushed: Kaggle secret {secret} is not available ({type(e).__name__}). "
                  "The files are still in this notebook's Output.")
            return False
        target = repo_url.replace("https://github.com/", f"https://x-access-token:{token}@github.com/")
    else:
        token = None
    git("config", "user.name", "FallVision pipeline")
    git("config", "user.email", "fallvision-pipeline@users.noreply.github.com")
    git("add", "--", *paths)
    code, out = git("commit", "-q", "-m", message)
    if code != 0:
        print("Nothing new to commit." if "nothing" in out.lower() else out)
        return False
    for attempt in range(2):                     # if another notebook pushed meanwhile: rebase once
        code, out = git("push", target, f"HEAD:{branch}", hide=token)
        if code == 0:
            print(f"Pushed to {repo_url} ({branch}): {', '.join(paths)}")
            return True
        if attempt == 0:
            git("pull", "--rebase", "-q", target, branch, hide=token)
    print("Push failed:", out[-800:])
    print("The files are still in this notebook's Output (project_fallvision folder).")
    return False


def run(cmd, cwd=None):
    """Run a command and show its output line by line in the notebook; stop the notebook if it fails."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise SystemExit(f"FAILED (exit {p.returncode}): read the messages above")


def find_trained_model(name="gru_hard.pt"):
    """models/gru_hard.pt from the output of notebook 04 (its repository copy also holds the training
    curves), so old weights in another input are never picked by mistake."""
    for d, _, fs in _walk(INPUT, max_depth=10):
        if name in fs and os.path.basename(d) == "models":
            repo = os.path.dirname(d)
            if os.path.exists(os.path.join(repo, "results", "curves", "gru_hard_history.csv")):
                print("Trained model (from notebook 04):", os.path.join(d, name))
                return os.path.join(d, name)
    sys.exit(f"{name} from notebook 04 not found: add the output of 04_train_models as an input")


def git_text(repo_dir, path, rev="HEAD"):
    """A file as it is in the last commit (before this run changed it); '' if it was not there."""
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=repo_dir, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def show_change(repo_dir, path):
    """Word-level difference between the committed file and this run's file."""
    r = subprocess.run(["git", "diff", "--no-color", "--word-diff=plain", "-U0", "--", path],
                       cwd=repo_dir, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if not l.startswith(("diff --git", "index ", "--- ", "+++ "))]
    print(f"--- {path}: " + ("same as the committed version" if not lines else
                              "changes ([-old-] {+new+}):"))
    print("\n".join(lines[:60]))


def _chosen_rows(md):
    rows = {}
    for l in md.splitlines():
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) >= 12 and c[1].startswith("chosen"):
            rows[c[0]] = dict(f1=c[3], precision=c[4], recall=c[5], fp=c[8], fn=c[10], delay=c[11], setting=c[1])
    return rows


def compare_eval(old_md, new_md):
    """Side by side: the committed results/causal_eval.md and this run's, chosen setting per split."""
    old, new = _chosen_rows(old_md), _chosen_rows(new_md)
    print(f"{'split':<15}{'F1 old -> new':>18}{'false alarms':>16}{'missed falls':>16}{'delay':>18}")
    for s in ["val", "test_gmdcsa24", "test_staged", "test_oops", "test_own"]:
        if s not in new:
            continue
        o = old.get(s, {})
        n = new[s]
        print(f"{s:<15}{o.get('f1', '-'):>8} -> {n['f1']:<7}{o.get('fp', '-'):>7} -> {n['fp']:<6}"
              f"{o.get('fn', '-'):>7} -> {n['fn']:<6}{o.get('delay', '-'):>8} -> {n['delay']}")
    if new:
        print("\nthis run:", next(iter(new.values()))["setting"])
