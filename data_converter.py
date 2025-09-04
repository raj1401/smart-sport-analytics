#!/usr/bin/env python3
"""
Build a YOLOv8 detection dataset from SoccerNet MOT folders.

Input layout (example):
  train/
    SNMOT-060/
      img1/000001.jpg ...
      gt/gt.txt                  # MOT: frame,id,x,y,w,h,score,-1,-1,-1
      gt/classes.csv   (optional, per-track map: track_id,class_name where class_name in {ball,player,referee,goalkeeper,other})
    SNMOT-061/
      ...

Output (example):
  OUT_ROOT/
    images/train/ SNMOT-060_000001.jpg ...
    labels/train/ SNMOT-060_000001.txt  # "cls cx cy w h" (normalized), one line per box

Default is single-class (class 0 = "object"). Use --ball-split to output two classes:
  0 = ball, 1 = personlike. If classes.csv is absent, a heuristic is used.

Heuristic for ball:
  - tiny median area fraction (bbox_area / image_area) < --ball-area-frac (default 0.00015 = 0.015%)
  - and motion magnitude percentile >= --ball-motion-pctl (default 60th percentile)

Usage:
  python data_converter.py \
    --train-root /path/to/train \
    --out-root datasets/sn_yolo \
    --split train \
    --copy-images \
    --ball-split   # optional (enables ball vs. personlike)
"""

import argparse, csv, os, glob, math, shutil
from collections import defaultdict
from typing import Dict, Tuple, List
import cv2
import numpy as np


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_classes_csv(csv_path: str) -> Dict[int, str]:
    """Read optional classes.csv mapping track_id -> class_name (ball/player/referee/goalkeeper/other)."""
    mapping = {}
    if not os.path.isfile(csv_path):
        return mapping
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            try:
                tid = int(row[0])
                cname = row[1].strip().lower()
                mapping[tid] = cname
            except Exception:
                continue
    return mapping


def load_mot_gt(gt_path: str) -> List[Tuple[int,int,float,float,float,float,float]]:
    """
    Load MOT gt.txt rows -> list of (frame, tid, x, y, w, h, score).
    Keeps only boxes with positive w, h.
    """
    rows = []
    with open(gt_path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 7:
                continue
            try:
                frame = int(float(row[0]))
                tid   = int(float(row[1]))
                x, y, w, h = map(float, row[2:6])
                score = float(row[6])
            except Exception:
                continue
            if w > 0 and h > 0:
                rows.append((frame, tid, x, y, w, h, score))
    return rows


def compute_track_stats(rows, W, H):
    """Per-track median area fraction and per-step motion magnitudes (in pixels)."""
    track_boxes = defaultdict(list)  # tid -> [(frame, cx, cy, w, h)]
    for (frame, tid, x, y, w, h, _) in rows:
        cx, cy = x + w/2.0, y + h/2.0
        track_boxes[tid].append((frame, cx, cy, w, h))

    areas_frac = {}
    motion_mag = {}
    for tid, seq in track_boxes.items():
        seq.sort(key=lambda z: z[0])
        areas = [(w*h)/(W*H) for (_, _, _, w, h) in seq]
        areas_frac[tid] = float(np.median(areas)) if areas else 1.0

        mm = []
        for i in range(1, len(seq)):
            _, cx1, cy1, _, _ = seq[i-1]
            _, cx2, cy2, _, _ = seq[i]
            mm.append(math.hypot(cx2-cx1, cy2-cy1))
        motion_mag[tid] = np.array(mm) if mm else np.array([0.0])
    return areas_frac, motion_mag


def decide_ball_tracks(rows, W, H, area_frac_thresh=0.00015, motion_percentile=60.0, px_thresh=0.5) -> Dict[int, bool]:
    """Heuristic ball detection per track."""
    areas_frac, motion_mag = compute_track_stats(rows, W, H)
    is_ball = {}
    for tid in areas_frac:
        area_ok = areas_frac[tid] < area_frac_thresh
        mm = motion_mag[tid]
        pctl = float(np.percentile(mm, motion_percentile)) if mm.size else 0.0
        motion_ok = pctl >= px_thresh  # px/frame
        is_ball[tid] = bool(area_ok and motion_ok)
    return is_ball


def detect_image_size(img1_dir: str, ext_candidates=("jpg","png","jpeg")) -> Tuple[int,int,str,List[str]]:
    """Return (W, H, ext_used, sorted_image_paths)."""
    for ext in ext_candidates:
        imgs = sorted(glob.glob(os.path.join(img1_dir, f"*.{ext}")))
        if imgs:
            im0 = cv2.imread(imgs[0])
            if im0 is None:
                continue
            H, W = im0.shape[:2]
            return W, H, ext, imgs
    raise RuntimeError(f"No readable images found in {img1_dir} (tried {ext_candidates}).")


def process_sequence(seq_dir: str, out_img_dir: str, out_lab_dir: str, copy_images: bool,
                     ball_split: bool, ball_area_frac: float, ball_motion_pctl: float, px_thresh: float):
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    img1_dir = os.path.join(seq_dir, "img1")
    gt_path  = os.path.join(seq_dir, "gt", "gt.txt")
    if not (os.path.isdir(img1_dir) and os.path.isfile(gt_path)):
        print(f"[SKIP] {seq_name}: missing img1/ or gt/gt.txt")
        return

    # read images/size
    W, H, ext, img_paths = detect_image_size(img1_dir)

    # read GT
    rows = load_mot_gt(gt_path)

    # decide classes
    cls_map_path = os.path.join(seq_dir, "gt", "classes.csv")
    explicit_map = read_classes_csv(cls_map_path)
    if ball_split:
        if explicit_map:
            is_ball = {tid: (explicit_map.get(tid, "") == "ball") for tid in set([r[1] for r in rows])}
        else:
            is_ball = decide_ball_tracks(rows, W, H, ball_area_frac, ball_motion_pctl, px_thresh)
    else:
        is_ball = {}  # ignored

    # per-frame lists of YOLO boxes
    by_frame = defaultdict(list)  # frame -> [(cls, cx, cy, w, h)]
    for (frame, tid, x, y, w, h, score) in rows:
        cx = (x + w/2.0) / W
        cy = (y + h/2.0) / H
        nw = w / W
        nh = h / H
        if ball_split:
            cls = 0 if is_ball.get(tid, False) else 1  # 0=ball, 1=personlike
        else:
            cls = 0  # single-class
        by_frame[frame].append((cls, cx, cy, nw, nh))

    # write labels and copy images (prefix with seq)
    for p in img_paths:
        fname = os.path.basename(p)         # e.g., 000001.jpg
        stem, _ = os.path.splitext(fname)
        try:
            frame_id = int(stem)
        except Exception:
            # fallback: try stripping non-digits
            import re
            m = re.search(r"(\d+)", stem)
            frame_id = int(m.group(1)) if m else None

        # label name
        lab_path = os.path.join(out_lab_dir, f"{seq_name}_{stem}.txt")
        with open(lab_path, "w") as f:
            for (cls, cx, cy, nw, nh) in by_frame.get(frame_id, []):
                f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        if copy_images:
            out_img_path = os.path.join(out_img_dir, f"{seq_name}_{fname}")
            if p != out_img_path:
                shutil.copy2(p, out_img_path)

    print(f"[OK] {seq_name}: labels -> {out_lab_dir}" + (" | images copied" if copy_images else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True, help="Path to SoccerNet train/ containing SNMOT-xxx folders")
    ap.add_argument("--out-root", required=True, help="Root for YOLO dataset")
    ap.add_argument("--split", default="train", choices=["train","val","test"], help="Subfolder under images/ and labels/")
    ap.add_argument("--copy-images", action="store_true", help="Copy frames into out-root/images/<split>")
    ap.add_argument("--ball-split", action="store_true", help="Output two classes: 0=ball, 1=personlike (else single-class 0=object)")
    ap.add_argument("--ball-area-frac", type=float, default=0.00015, help="Heuristic: median area fraction threshold for ball")
    ap.add_argument("--ball-motion-pctl", type=float, default=60.0, help="Heuristic: motion percentile (0-100)")
    ap.add_argument("--ball-motion-px", type=float, default=0.5, help="Heuristic: min px/frame at the chosen percentile")
    args = ap.parse_args()

    out_img_dir = os.path.join(args.out_root, "images", args.split)
    out_lab_dir = os.path.join(args.out_root, "labels", args.split)
    ensure_dir(out_lab_dir)
    if args.copy_images:
        ensure_dir(out_img_dir)

    # each immediate subfolder in train-root is a sequence (e.g., SNMOT-060)
    seq_dirs = sorted([p for p in glob.glob(os.path.join(args.train_root, "*")) if os.path.isdir(p)])
    if not seq_dirs:
        raise RuntimeError(f"No sequence folders found under {args.train_root}")

    for seq in seq_dirs:
        process_sequence(
            seq_dir=seq,
            out_img_dir=out_img_dir,
            out_lab_dir=out_lab_dir,
            copy_images=args.copy_images,
            ball_split=args.ball_split,
            ball_area_frac=args.ball_area_frac,
            ball_motion_pctl=args.ball_motion_pctl,
            px_thresh=args.ball_motion_px,
        )

    # dataset YAML suggestion
    if args.ball_split:
        names = '["ball","personlike"]'
    else:
        names = '["object"]'
    print("\nCreate a data yaml like:\n"
          f"path: {args.out_root}\n"
          f"train: images/{args.split}\n"
          f"val: images/val\n"
          f"names: {names}\n")


if __name__ == "__main__":
    main()
