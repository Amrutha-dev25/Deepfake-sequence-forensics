"""
build_manifest.py — Build training manifest from the new metadata index
==========================================================================
The dataset format changed. Previously there was one JSON label file per
video (edit_keys, source_video, split_group). Now there is a single CSV
with columns:

    video_name,source_video,sequence,num_frames
    000_FS_FSh_DF.mp4,000.mp4,FaceSwap->FaceShifter->DeepFake,303

`sequence` spells out the full manipulation-method names joined by "->",
not the short codes (FS/FSh/DF/NT/F2F) the model trains on. This script:

  1. Parses `sequence` and maps each method name to its short code.
  2. Builds source-ID-disjoint train/val/test splits (same identity never
     spans two splits — prevents the model from cheating on face identity
     instead of learning manipulation order).
  3. Writes manifest.csv + split_assignment.json into --out-dir, which
     train.py reads directly.
  4. Prints a per-position class-balance report so a skewed dataset is
     visible before you spend GPU hours on it.

Run this FIRST, before train.py, any time the metadata index changes.

Usage:
    python build_manifest.py \\
        --metadata-csv _metadata_index.csv \\
        --video-root dataset/generated_videos \\
        --out-dir checkpoints \\
        --verify-files
"""

import csv
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

from model import EDIT_TO_ID, MAX_SEQ_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Full manipulation-method name (as written in the metadata index) -> short
# code used everywhere else in the pipeline (model.EDIT_TO_ID).
EDIT_NAME_TO_CODE = {
    "FaceSwap":       "FS",
    "FaceShifter":     "FSh",
    "DeepFake":        "DF",
    "NeuralTextures":  "NT",
    "Face2Face":       "F2F",
}


def parse_sequence(sequence: str) -> list:
    """'FaceSwap->FaceShifter->DeepFake' -> ['FS', 'FSh', 'DF']"""
    parts = [p.strip() for p in sequence.split("->")]
    codes = []
    for p in parts:
        if p not in EDIT_NAME_TO_CODE:
            raise ValueError(f"Unknown manipulation name '{p}' in sequence '{sequence}'")
        codes.append(EDIT_NAME_TO_CODE[p])
    return codes


def build_manifest_rows(metadata_csv: Path, video_root: Path, verify_files: bool) -> list:
    rows, skipped = [], []

    with open(metadata_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            video_name   = r["video_name"].strip()
            source_video = r["source_video"].strip()
            sequence     = r["sequence"].strip()
            num_frames   = int(r["num_frames"])

            try:
                codes = parse_sequence(sequence)
            except ValueError as e:
                skipped.append((video_name, str(e)))
                continue

            if len(codes) != MAX_SEQ_LEN:
                skipped.append((video_name, f"expected {MAX_SEQ_LEN} steps, got {len(codes)}: {sequence}"))
                continue

            unknown = [c for c in codes if c not in EDIT_TO_ID]
            if unknown:
                skipped.append((video_name, f"code(s) {unknown} not in model.EDIT_TO_ID vocabulary"))
                continue

            video_path = video_root / video_name
            if verify_files and not video_path.exists():
                skipped.append((video_name, f"file not found at {video_path}"))
                continue

            source_id = source_video.replace(".mp4", "")
            rows.append({
                "video_path": str(video_path),
                "source_id":  source_id,
                "edit_keys":  "|".join(codes),
                "num_frames": num_frames,
            })

    log.info(f"Parsed {len(rows)} usable rows, skipped {len(skipped)}")
    if skipped:
        log.info("First 10 skip reasons:")
        for name, reason in skipped[:10]:
            log.info(f"  {name}: {reason}")
    return rows


def save_manifest(rows: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "source_id", "edit_keys", "num_frames"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Manifest saved -> {out_path}  ({len(rows)} rows)")


def build_splits(rows: list, seed: int, train_r: float, val_r: float) -> dict:
    source_ids = sorted({r["source_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(source_ids)

    n       = len(source_ids)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    split = {}
    for i, sid in enumerate(source_ids):
        if i < n_train:
            split[sid] = "train"
        elif i < n_train + n_val:
            split[sid] = "val"
        else:
            split[sid] = "test"

    counts = defaultdict(int)
    for s in split.values():
        counts[s] += 1
    log.info(f"Source-ID split: train={counts['train']} val={counts['val']} test={counts['test']} "
              f"(of {n} unique source identities)")
    return split


def class_balance_report(rows: list):
    pos_counts = [defaultdict(int) for _ in range(MAX_SEQ_LEN)]
    for r in rows:
        keys = r["edit_keys"].split("|")
        for i, k in enumerate(keys):
            pos_counts[i][k] += 1
    for i, counts in enumerate(pos_counts):
        total = sum(counts.values())
        dist = "  ".join(f"{k}={v} ({v/total:.1%})" for k, v in sorted(counts.items()))
        log.info(f"  Position {i+1}: {dist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", required=True, help="Path to _metadata_index.csv")
    ap.add_argument("--video-root",   required=True, help="Folder containing the actual .mp4 files")
    ap.add_argument("--out-dir",      required=True, help="Where to write manifest.csv + split_assignment.json")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--train-frac",   type=float, default=0.70)
    ap.add_argument("--val-frac",     type=float, default=0.15)
    ap.add_argument("--verify-files", action="store_true",
                     help="Check every video file actually exists on disk (slower; recommended at least once).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_manifest_rows(Path(args.metadata_csv), Path(args.video_root), args.verify_files)
    if not rows:
        raise SystemExit("No usable rows parsed — check --metadata-csv / --video-root and re-run.")

    save_manifest(rows, out_dir / "manifest.csv")

    log.info("Class balance across the full manifest:")
    class_balance_report(rows)

    split = build_splits(rows, args.seed, args.train_frac, args.val_frac)
    with open(out_dir / "split_assignment.json", "w") as f:
        json.dump(split, f, indent=2)
    log.info(f"Split assignment saved -> {out_dir / 'split_assignment.json'}")
    log.info("Done. Now run train.py --debug-overfit as the sanity check.")


if __name__ == "__main__":
    main()
