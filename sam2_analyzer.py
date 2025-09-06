"""
Given a sequence of frames, this script runs both ByteTrack and SAM-2 to generate object tracks and masks.
"""

import os
import gc
import numpy as np
import cv2
import torch
import glob
import re
import json

from typing import Dict, List, Tuple, Optional

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO


sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"


def _natural_key(s: str):
    """Sort helper: 'frame10.jpg' after 'frame9.jpg'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(s))]


# -------------- SAM2 --------------
def _clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def _color_from_id(tid: int) -> Tuple[int, int, int]:
    """
    Deterministic bright color per id (BGR for OpenCV).
    Uses tab10-ish palette; wraps by modulo.
    """
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
        (140, 86, 75),  (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)
    ]
    # Convert RGB->BGR for OpenCV
    r, g, b = palette[abs(tid) % len(palette)]
    return (b, g, r)


def sam2_track_video(
    video_path: str,
    yolo_model_path: str = "yolov8n.pt",
    model_cfg: str = model_cfg,
    sam2_checkpoint: str = sam2_checkpoint,
    device: str | int | torch.device = device,
    out_video_path: Optional[str] = "outputs/sam2_tracked_output.mp4",
    imgsz: int = 1280,
    conf: float = 0.25,
    tracker_cfg: str = "bytetrack.yaml",
    max_frames: int = -1,
    mask_threshold: float = 0.0,
    draw_ids: bool = True,
    alpha: float = 0.55,
    keep_temp: bool = False,
) -> str:
    """
    Extract frames from a video, run YOLO+ByteTrack on the first frame to get detections,
    seed SAM2 with those boxes, propagate masks across the video, and save an overlay MP4.

    Returns the path to the saved video.
    """
    def _ultra_device_arg(dev):
        if isinstance(dev, torch.device):
            if dev.type == "cuda":
                # assume first cuda device
                return 0
            return dev.type
        return dev

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Determine how many frames to extract
    if max_frames is not None and max_frames > 0:
        n_extract = min(total_frames or max_frames, max_frames)
    else:
        n_extract = total_frames if total_frames > 0 else 10**9  # until EOF

    import tempfile
    import shutil
    temp_frames_dir_sam = tempfile.mkdtemp(prefix="sam2_frames_")

    # Extract frames
    pad = max(6, len(str(n_extract)))
    frame_paths: List[str] = []
    idx = 0
    while idx < n_extract:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:
            break
        if H <= 0 or W <= 0:
            H, W = frame.shape[:2]
        fname = f"{idx:0{pad}d}.jpg"
        fpath = os.path.join(temp_frames_dir_sam, fname)
        cv2.imwrite(fpath, frame)
        frame_paths.append(fpath)
        idx += 1
    cap.release()

    if not frame_paths:
        if not keep_temp:
            shutil.rmtree(temp_frames_dir_sam, ignore_errors=True)
        raise RuntimeError("No frames extracted from video.")

    # Read first frame
    first_img = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if first_img is None:
        if not keep_temp:
            shutil.rmtree(temp_frames_dir_sam, ignore_errors=True)
        raise RuntimeError(f"Failed to read first extracted frame: {frame_paths[0]}")
    H, W = first_img.shape[:2]

    # Run YOLO ByteTrack on first frame only to get IDs and boxes
    yolo = YOLO(yolo_model_path)
    results = yolo.track(
        source=first_img,
        imgsz=imgsz,
        device=_ultra_device_arg(device),
        conf=conf,
        tracker=tracker_cfg,
        persist=False,
        verbose=False,
    )

    first_frame_dets: List[Tuple[int, int, int, int, int, float]] = []
    if results:
        res0 = results[0]
        boxes = res0.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids_t = boxes.id
            ids = ids_t.cpu().numpy().astype(int) if ids_t is not None else None
            # Fallback to sequential IDs if tracker didn't assign
            if ids is None:
                ids = list(range(1, len(xyxy) + 1))
            for i in range(len(xyxy)):
                tid = int(ids[i]) if not isinstance(ids, list) else int(ids[i])
                x1, y1, x2, y2 = map(int, xyxy[i])
                conf_i = float(confs[i])
                first_frame_dets.append((tid, x1, y1, x2, y2, conf_i))

    # Initialize SAM2 on the extracted frames dir
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=temp_frames_dir_sam)

    # Seed SAM2 with first-frame boxes
    ann_frame_idx = 0
    for tid, x1, y1, x2, y2, _conf in first_frame_dets:
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, W, H)
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=int(tid),
            box=box,
        )

    # Propagate masks
    video_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        masks = {}
        for i, obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > mask_threshold).cpu().numpy()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            masks[int(obj_id)] = mask.astype(np.bool_)
        video_segments[int(out_frame_idx)] = masks

    # Write overlay video
    if out_video_path is None:
        out_video_path = os.path.join(os.path.dirname(video_path), "sam2_tracked_output.mp4")
    os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)

    # Try H.264, fall back to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (W, H))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (W, H))
    if not writer.isOpened():
        if not keep_temp:
            shutil.rmtree(temp_frames_dir_sam, ignore_errors=True)
        raise RuntimeError(f"Could not open video writer for: {out_video_path}")

    # Ensure deterministic order
    frame_paths.sort(key=_natural_key)
    for idx, fp in enumerate(frame_paths):
        frame = cv2.imread(fp, cv2.IMREAD_COLOR)
        if frame is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

        overlay = frame.copy()
        masks = video_segments.get(idx, {})
        for obj_id, mask in masks.items():
            if mask is None or mask.shape != (H, W):
                continue
            color = _color_from_id(obj_id)
            overlay[mask] = (np.array(color, dtype=np.uint8) * 0.7 + overlay[mask] * 0.3).astype(np.uint8)

            if draw_ids:
                m8 = (mask.astype(np.uint8) * 255)
                x, y, w, h = cv2.boundingRect(m8)
                if w > 0 and h > 0:
                    label_pt = (x, max(0, y - 5))
                    cv2.putText(
                        overlay,
                        f"id={obj_id}",
                        label_pt,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )

        blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        writer.write(blended)

    writer.release()

    # Cleanup
    if not keep_temp:
        shutil.rmtree(temp_frames_dir_sam, ignore_errors=True)

    print(f"[OK] SAM2 tracked video saved: {out_video_path}")
    return out_video_path


if __name__ == "__main__":
    # Example usage
    video_path = os.path.join("videos", "raw_video.mp4")
    _ = sam2_track_video(
        video_path=video_path,
        yolo_model_path="models/yolov8x.pt",
        model_cfg=model_cfg,
        sam2_checkpoint=sam2_checkpoint,
        device=device,
        out_video_path="outputs/sam2_tracked_output.mp4",
        imgsz=1280,
        conf=0.25,
        tracker_cfg="bytetrack.yaml",
        max_frames=125,
        mask_threshold=0.0,
        draw_ids=True,
        alpha=0.55,
        keep_temp=False,
    )