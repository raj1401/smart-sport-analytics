import subprocess
import os
from pathlib import Path
import cv2


def convert_to_avi(input_video: str) -> str:
    avi_path = str(Path(input_video).with_suffix('.avi'))
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    return avi_path


def run_openpose_video(input_video: str, output_video: str = None) -> str:
    """
    Run OpenPose on a video and save the pose-annotated output.
    Converts input video to AVI format before processing.

    Args:
        openpose_bin (str): Path to OpenPoseDemo.exe (e.g., "bin/OpenPoseDemo.exe")
        input_video (str): Path to the input video file
        output_video (str, optional): Path where the output video will be saved.
                                      If None, saves next to input with '_pose.avi'.

    Returns:
        str: Path to the saved output video
    """
    OPENPOSE_BIN_PATH = os.path.join("openpose", "bin", "OpenPoseDemo.exe")
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Convert to AVI format
    avi_input = convert_to_avi(str(input_path))
    # Convert to absolute path for OpenPose
    avi_input = str(Path(avi_input).resolve())

    if output_video is None:
        output_video = str(input_path.with_name(input_path.stem + "_pose.avi"))
    output_video = str(Path(output_video).resolve())

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    cmd = [
        OPENPOSE_BIN_PATH,
        "--video", avi_input,
        "--write_video", str(output_video),
        "--display", "0",
    ]

    try:
        subprocess.run(cmd, check=True, cwd="openpose")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenPose failed with code {e.returncode}") from e

    if not Path(output_video).exists():
        raise RuntimeError(f"Expected output not found: {output_video}")

    return output_video