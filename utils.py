from pathlib import Path

import os
import shutil
import tempfile
import subprocess


def write_h264_for_streamlit(input_path: str) -> str:
    """
    Transcode/remux 'input_path' to a browser-friendly H.264 MP4:
    - H.264 (libx264)
    - yuv420p pixel format
    - +faststart for streaming in <video> / st.video
    Returns the path to a temp MP4 you can pass to st.video() (bytes or path).
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # 1) Find ffmpeg: PATH -> imageio-ffmpeg fallback
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise FileNotFoundError(
                "ffmpeg not found. Install it (e.g., choco install ffmpeg on Windows) "
                "or `pip install imageio-ffmpeg` so we can use its bundled binary."
            ) from e

    # 2) Make a temp output mp4
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp.name
    tmp.close()

    # 3) Run ffmpeg
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(in_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.0",
        "-movflags", "+faststart",
        "-crf", "23",
        "-preset", "fast",
        out_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,  # important on Windows
        )
    except subprocess.CalledProcessError as e:
        # Clean up the temp file if ffmpeg failed
        try:
            os.unlink(out_path)
        except OSError:
            pass
        # Surface ffmpeg stderr for debugging
        raise RuntimeError(
            "ffmpeg failed:\n" + e.stderr.decode(errors="ignore")
        ) from e

    return out_path