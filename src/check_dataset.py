"""
Check that a dataset folder holds every video of our splits, and (optionally) pack exactly those
videos into a clean folder for submission.

The layout the pipeline reads (run_pipeline.py, eval_causal.py, cache_poses.py):
  <omnifall-root>/le2i/video/<path>.mp4, caucafall/..., mcfd/..., OOPS/...   (paths: splits/videos.csv)
  <gmd-root>/Subject 1 ... Subject 4/{Fall,ADL}/NN.mp4
  <own-root>/**/NNN.mp4                                                       (splits/own_videos_labels.csv)

Usage:
  python src/check_dataset.py --omnifall-root D:/dl/omnifall --gmd-root D:/dl/GMDCSA24 [--own-root D:/own]
  python src/check_dataset.py ... --pack-to G:/FallVision_dataset [--with-own]
      copies only the videos our splits use into
        G:/FallVision_dataset/omnifall/<dataset>/video/<path>.mp4
        G:/FallVision_dataset/GMDCSA24/Subject N/...
        G:/FallVision_dataset/own/NNN.mp4               (only with --with-own)
      plus DATA.md, videos.csv, own_videos_labels.csv, labels/ and manifest.csv (size + MD5 of every file),
      then checks the packed copy again.
"""

import argparse
import collections
import csv
import hashlib
import os
import shutil
import sys

from cache_poses import video_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows_of(own_root):
    rows = [dict(dataset=r["dataset"], path=r["path"], split=r["split"])
            for r in csv.DictReader(open(os.path.join(ROOT, "splits", "videos.csv"), newline=""))]
    if own_root is not None:
        for r in csv.DictReader(open(os.path.join(ROOT, "splits", "own_videos_labels.csv"), newline="")):
            rows.append(dict(dataset="own", path=os.path.splitext(r["file"].strip())[0], split="test_own"))
    return rows


def check(rows, omni, gmd, own):
    found, missing = collections.Counter(), collections.defaultdict(list)
    size = collections.Counter()
    for r in rows:
        f = video_file(r["dataset"], r["path"], omni or "", gmd or "", own)
        r["file"] = f
        key = (r["dataset"], r["split"])
        if os.path.isfile(f):
            found[key] += 1
            size[r["dataset"]] += os.path.getsize(f)
        else:
            missing[key].append(f)
    print(f"{'dataset':<11}{'split':<15}{'found':>7}{'missing':>9}")
    for key in sorted(set(found) | set(missing)):
        print(f"{key[0]:<11}{key[1]:<15}{found[key]:>7}{len(missing[key]):>9}")
    for ds in sorted(size):
        print(f"  {ds:<10} {size[ds] / 1e9:6.2f} GB")
    print(f"  {'total':<10} {sum(size.values()) / 1e9:6.2f} GB")
    n_miss = sum(len(v) for k, v in missing.items() if k[0] != "own")
    n_own = sum(len(v) for k, v in missing.items() if k[0] == "own")
    for key, files in sorted(missing.items()):
        for f in files[:3]:
            print("  missing:", f)
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more missing in {key[0]} / {key[1]}")
    if n_own:
        print(f"note: {n_own} of our own test videos are not in --own-root (optional set; those rows are skipped)")
    print("ALL VIDEOS FOUND" if n_miss == 0 else f"MISSING: {n_miss} videos of the public datasets")
    return n_miss


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack(rows, dest):
    omni, gmd, own = (os.path.join(dest, d) for d in ("omnifall", "GMDCSA24", "own"))
    manifest = []
    for i, r in enumerate(rows, 1):
        if not os.path.isfile(r["file"]):
            continue
        dst = video_file(r["dataset"], r["path"], omni, gmd, own)
        if r["dataset"] == "own":
            dst = os.path.join(own, os.path.basename(r["file"]))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not (os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(r["file"])):
            shutil.copyfile(r["file"], dst)
        manifest.append(dict(dataset=r["dataset"], split=r["split"], file=os.path.relpath(dst, dest).replace("\\", "/"),
                             bytes=os.path.getsize(dst), md5=md5(dst)))
        if i % 100 == 0 or i == len(rows):
            print(f"  packed {i}/{len(rows)}", flush=True)
    if not manifest:
        sys.exit("No video was found, nothing packed: check the folders given.")
    with open(os.path.join(dest, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    shutil.copyfile(os.path.join(ROOT, "splits", "videos.csv"), os.path.join(dest, "videos.csv"))
    if any(r["dataset"] == "own" for r in rows):
        shutil.copyfile(os.path.join(ROOT, "splits", "own_videos_labels.csv"), os.path.join(dest, "own_videos_labels.csv"))
    os.makedirs(os.path.join(dest, "labels"), exist_ok=True)
    for ds in ("GMDCSA24", "le2i", "caucafall", "mcfd", "OOPS"):          # the datasets we use
        shutil.copyfile(os.path.join(ROOT, "labels", "omnifall", ds + ".csv"), os.path.join(dest, "labels", ds + ".csv"))
    shutil.copyfile(os.path.join(ROOT, "docs", "DATA.md"), os.path.join(dest, "DATA.md"))
    print(f"\nPacked {len(manifest)} videos into {dest} (manifest.csv has size and MD5 of each)")
    return omni, gmd, (own if any(r["dataset"] == "own" for r in rows) else None)


def main():
    ap = argparse.ArgumentParser(description="Check (and pack) the videos used by FallVision")
    ap.add_argument("--omnifall-root", required=True, help="folder with le2i/ caucafall/ mcfd/ OOPS/")
    ap.add_argument("--gmd-root", required=True, help="GMDCSA24 folder with 'Subject 1' ... 'Subject 4'")
    ap.add_argument("--own-root", default=None, help="our own test videos (optional)")
    ap.add_argument("--pack-to", default=None, help="copy exactly the videos used into this new folder")
    ap.add_argument("--with-own", action="store_true", help="also pack the own videos (only if the team agrees)")
    a = ap.parse_args()

    own = a.own_root if (a.own_root and (a.with_own or not a.pack_to)) else None
    rows = rows_of(own)
    n_miss = check(rows, a.omnifall_root, a.gmd_root, own)
    if not a.pack_to:
        sys.exit(1 if n_miss else 0)
    if n_miss:
        print("\nPacking anyway: only the videos found are copied (see MISSING above).")
    dest = os.path.abspath(a.pack_to)
    if os.path.abspath(dest).startswith(ROOT + os.sep):
        sys.exit("Put the packed dataset OUTSIDE the project folder (videos must never go into git).")
    omni, gmd, own_p = pack(rows, dest)
    print("\nChecking the packed copy:")
    n2 = check(rows_of(own_p), omni, gmd, own_p)
    sys.exit(1 if n2 else 0)


if __name__ == "__main__":
    main()
