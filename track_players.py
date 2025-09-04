"""
Track players in a video using YOLOv8 and ByteTrack, saving an annotated video and optional JSON of tracks.
"""

from typing import Dict, List, Tuple, Optional
import cv2, json, os
import numpy as np
from ultralytics import YOLO


def track_players_video(
    model_path: str,
    video_path: str,
    out_video_path: str = "tracked.mp4",
    imgsz: int = 1280,
    device: str | int = 0,
    tracker_cfg: str = "bytetrack.yaml",
    conf: float = 0.25,
    save_tracks_json: Optional[str] = None,
    filter_ids: Optional[List[int]] = None,
) -> Dict[int, List[Tuple[int, int, int, int, int, float]]]:
    """
    Track players in a video and collect per-ID bounding boxes.

    Returns:
        tracks: dict {track_id: [(frame_idx, x1, y1, x2, y2, conf), ...]}
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    names = model.names
    tracks: Dict[int, List[Tuple[int, int, int, int, int, float]]] = {}

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.track(
            source=frame,
            imgsz=imgsz,
            device=device,
            conf=conf,
            tracker=tracker_cfg,
            persist=True,  # keep tracker state across frames
            verbose=False,
        )

        if not results:
            writer.write(frame)
            continue

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            writer.write(frame)
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss  = boxes.cls.cpu().numpy().astype(int)
        ids_t = boxes.id
        ids   = ids_t.cpu().numpy().astype(int) if ids_t is not None else None

        # Draw and record tracks
        for i in range(len(xyxy)):
            if ids is None:
                continue
            tid = int(ids[i])
            if filter_ids is not None and tid not in filter_ids:
                continue

            x1, y1, x2, y2 = map(int, xyxy[i])
            conf_i = float(confs[i])

            tracks.setdefault(tid, []).append((frame_idx, x1, y1, x2, y2, conf_i))

            color = (255, 255, 0) if (filter_ids and tid in filter_ids) else (0, 255, 0)
            label = f"id={tid} {conf_i:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()

    if save_tracks_json:
        os.makedirs(os.path.dirname(save_tracks_json) or ".", exist_ok=True)
        with open(save_tracks_json, "w") as f:
            json.dump(tracks, f)

    print(f"[OK] Annotated video saved: {out_video_path}")
    if save_tracks_json:
        print(f"[OK] Tracks JSON saved: {save_tracks_json}")

    return tracks


def crop_track_video(
    video_path: str,
    tracks_json_path: str,
    track_id: int,
    out_path: str = "crop.mp4",
    aspect_ratio: Tuple[int, int] = (1, 1),   # e.g., (16, 9) or (1, 1)
    padding: float = 0.15,                    # 15% padding beyond max box size
    smooth_alpha: float = 0.0,                # 0 = no smoothing, e.g., 0.4 for EMA smoothing
    bg_color: Tuple[int, int, int] = (0, 0, 0) # letterbox color when near edges
) -> None:
    """
    Crop a video around a specific tracker ID using a fixed output size.
    The fixed crop size is determined as the maximum bbox size for that ID (over all frames),
    padded by `padding`, and adjusted to `aspect_ratio`.

    Args:
        video_path: path to the source video (e.g., .mp4)
        tracks_json_path: path to tracks JSON {track_id: [(frame_idx, x1,y1,x2,y2,conf), ...], ...}
        track_id: the tracker ID to follow
        out_path: output cropped video file
        aspect_ratio: desired (w,h) ratio for the crop (e.g., (16,9))
        padding: extra margin as a fraction of the max bbox size
        smooth_alpha: EMA smoothing factor for center (0=no smoothing; 0.2–0.5 = smoother)
        bg_color: RGB tuple for padding near edges

    Writes:
        out_path video with fixed resolution following the selected track.
    """
    # --- Load tracks ---
    with open(tracks_json_path, "r") as f:
        data: Dict[str, List[List[float]]] = json.load(f)

    # Keys may be str in JSON; normalize to int
    # and extract this track's list: [(frame_idx, x1,y1,x2,y2,conf), ...]
    # If not found, raise error.
    str_tid = str(track_id)
    if str_tid in data:
        dets = data[str_tid]
    elif track_id in data:
        dets = data[track_id]
    else:
        raise ValueError(f"Track id {track_id} not found in {tracks_json_path}")

    # Convert to dict for quick lookup: frame -> (x1,y1,x2,y2,conf)
    dets_sorted = sorted(dets, key=lambda r: int(r[0]))
    by_frame = {int(fr): (float(x1), float(y1), float(x2), float(y2), float(conf))
                for fr, x1, y1, x2, y2, conf in dets_sorted}

    # --- Open video and get properties ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Determine fixed crop size from max box + padding + aspect ratio ---
    # Compute max w/h across all detections of this ID
    max_w = 1
    max_h = 1
    for _, (x1, y1, x2, y2, _) in by_frame.items():
        w = x2 - x1
        h = y2 - y1
        if w > max_w: max_w = w
        if h > max_h: max_h = h

    # Add padding
    pad_w = max_w * padding
    pad_h = max_h * padding
    w_padded = max_w + 2 * pad_w
    h_padded = max_h + 2 * pad_h

    # Adjust to requested aspect ratio
    ar_w, ar_h = aspect_ratio
    desired_ar = (ar_w / ar_h) if ar_h != 0 else 1.0

    crop_w = w_padded
    crop_h = h_padded
    # Fit the larger of the two so both w and h are covered
    if crop_w / crop_h < desired_ar:
        # too tall -> widen
        crop_w = crop_h * desired_ar
    else:
        # too wide -> heighten
        crop_h = crop_w / desired_ar

    # Final integer sizes, clamp to source dims
    crop_w = int(round(min(crop_w, src_w)))
    crop_h = int(round(min(crop_h, src_h)))

    # Make divisible by 2 for codecs
    if crop_w % 2: crop_w -= 1
    if crop_h % 2: crop_h -= 1
    crop_size = (crop_w, crop_h)

    # --- Prepare writer ---
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
    writer = cv2.VideoWriter(out_path, fourcc, src_fps if src_fps > 0 else 25, crop_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer: {out_path}")

    # --- Helper: crop with padding if near borders ---
    def crop_with_padding(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
        H, W = frame.shape[:2]
        half_w = crop_w // 2
        half_h = crop_h // 2

        # Desired crop coords
        x1 = int(round(cx - half_w))
        y1 = int(round(cy - half_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Compute in-frame region
        in_x1 = max(0, x1)
        in_y1 = max(0, y1)
        in_x2 = min(W, x2)
        in_y2 = min(H, y2)

        # Create output canvas
        out = np.full((crop_h, crop_w, 3), bg_color, dtype=np.uint8)

        # Destination where to paste the valid region
        dst_x1 = in_x1 - x1
        dst_y1 = in_y1 - y1
        dst_x2 = dst_x1 + (in_x2 - in_x1)
        dst_y2 = dst_y1 + (in_y2 - in_y1)

        if in_x2 > in_x1 and in_y2 > in_y1:
            out[dst_y1:dst_y2, dst_x1:dst_x2] = frame[in_y1:in_y2, in_x1:in_x2]

        return out

    # --- Iterate frames, follow the track ---
    # Optional smoothing with EMA on center
    cx_ema: Optional[float] = None
    cy_ema: Optional[float] = None
    alpha = float(smooth_alpha)

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Pick center: use this frame's box if present; else last known center
        if frame_idx in by_frame:
            x1, y1, x2, y2, _ = by_frame[frame_idx]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
        else:
            # If no detection this frame, keep last EMA center;
            # if none yet, default to center of the frame.
            if cx_ema is not None and cy_ema is not None:
                cx, cy = cx_ema, cy_ema
            else:
                cx, cy = src_w / 2.0, src_h / 2.0

        # Apply EMA smoothing if requested
        if alpha > 0.0:
            if cx_ema is None:
                cx_ema, cy_ema = cx, cy
            else:
                cx_ema = alpha * cx + (1 - alpha) * cx_ema
                cy_ema = alpha * cy + (1 - alpha) * cy_ema
            cx_use, cy_use = cx_ema, cy_ema
        else:
            cx_use, cy_use = cx, cy

        crop = crop_with_padding(frame, cx_use, cy_use)
        writer.write(crop)

    cap.release()
    writer.release()
    print(f"[OK] Cropped video saved to {out_path}")


if __name__ == "__main__":
    MODEL_PATH = "models/yolov8x.pt"
    IMG_SEQ_PATH = "datasets/SoccerNet/tracking/test/test/SNMOT-116/img1/"
    VIDEO_PATH = "videos/raw_video.mp4"
    OUT_PATH = "track_videos/output_video_annotated.mp4"

    IMGSZ = 1280
    FPS = 30

    # track_players_video(
    #     model_path=MODEL_PATH,
    #     video_path=VIDEO_PATH,
    #     out_video_path=OUT_PATH,
    #     imgsz=IMGSZ,
    #     device=0,
    #     save_tracks_json="track_videos/tracks.json",
    #     filter_ids=None  # e.g., [1,2,3] to only keep these IDs
    # )

    crop_track_video(
        video_path=VIDEO_PATH,
        tracks_json_path="track_videos/tracks.json",
        track_id=2,  # specify the ID to crop around
        out_path="track_videos/track_id2_crop.mp4",
        aspect_ratio=(9, 16),
        padding=0.15,
        smooth_alpha=0.0,
        bg_color=(0, 0, 0)
    )