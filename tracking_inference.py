
"""
Run YOLOv8 inference on a folder of images
or on a video file and save an annotated video.
"""

import glob, os, cv2
from ultralytics import YOLO

def run_inference(model_path, source_dir, out_path, imgsz=1280, fps=25):
    # Load model
    model = YOLO(model_path)

    # Collect images (sorted by filename)
    img_paths = sorted(glob.glob(os.path.join(source_dir, "*.jpg")) +
                       glob.glob(os.path.join(source_dir, "*.png")))

    if not img_paths:
        raise RuntimeError(f"No images found in {source_dir}")

    # Read first image to set video size
    im0 = cv2.imread(img_paths[0])
    if im0 is None:
        raise RuntimeError(f"Cannot read first image {img_paths[0]}")
    H, W = im0.shape[:2]

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Class names
    names = model.names

    for i, img_path in enumerate(img_paths):
        im = cv2.imread(img_path)
        if im is None:
            print(f"[WARN] Cannot read {img_path}, skipping")
            continue

        # Run inference
        results = model.predict(source=im, imgsz=imgsz, verbose=False, device=0)

        # results[0] is the first (and only) image
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()   # (N,4)
        confs = res.boxes.conf.cpu().numpy()   # (N,)
        clss  = res.boxes.cls.cpu().numpy().astype(int)  # (N,)

        # Draw boxes
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
            label = f"{names[cls]} {conf:.2f}"
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(im, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(im)
        if (i+1) % 50 == 0:
            print(f"[INFO] Processed {i+1}/{len(img_paths)} frames")

    writer.release()
    print(f"[OK] Video saved to {out_path}")


def run_inference_video(model_path, video_path, out_path="out.mp4", imgsz=1280, device=0, fps=None):
    """
    Run YOLOv8 inference on a video and save an annotated video.

    Args:
        model_path (str): Path to YOLOv8 .pt weights.
        video_path (str): Path to input video (e.g., .mp4).
        out_path (str): Path to output annotated video.
        imgsz (int): Inference image size (square side).
        device (int|str): CUDA device index (e.g., 0) or "cpu".
        fps (float|None): Output FPS; if None, uses source FPS.

    Returns:
        None (saves video to out_path)
    """
    from ultralytics import YOLO
    import cv2
    import time

    # Load model
    model = YOLO(model_path)

    # Open source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Infer properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps if fps is not None else (src_fps if src_fps > 0 else 25)

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (src_w, src_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer: {out_path}")

    names = model.names
    t0 = time.time()
    i = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # YOLO inference on this frame
            results = model.predict(source=frame, imgsz=imgsz, verbose=False, device=device)

            res = results[0]
            boxes = res.boxes.xyxy
            confs = res.boxes.conf
            clss  = res.boxes.cls

            if boxes is not None and len(boxes):
                boxes = boxes.cpu().numpy()
                confs = confs.cpu().numpy()
                clss  = clss.cpu().numpy().astype(int)

                # Draw detections
                for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    label = f"{names.get(cls, str(cls))} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            writer.write(frame)
            i += 1
            if i % 50 == 0:
                elapsed = time.time() - t0
                print(f"[INFO] Processed {i}/{n_frames if n_frames>0 else '?'} frames "
                      f"({i/max(elapsed,1e-6):.1f} FPS)")

    finally:
        cap.release()
        writer.release()

    print(f"[OK] Annotated video saved to {out_path}")



if __name__ == "__main__":
    MODEL_PATH = "models/yolov8x.pt"
    IMG_SEQ_PATH = "datasets/SoccerNet/tracking/test/test/SNMOT-116/img1/"
    VIDEO_PATH = "videos/raw_video.mp4"
    OUT_PATH = "videos/output_video.mp4"

    IMGSZ = 1280
    FPS = 30

    # run_inference(MODEL_PATH, IMG_SEQ_PATH, OUT_PATH, IMGSZ, FPS)

    run_inference_video(MODEL_PATH, VIDEO_PATH, OUT_PATH, IMGSZ, device=0, fps=FPS)
