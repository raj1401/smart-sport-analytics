# Smart Sport Analytics

## Project Goal

Smart Sport Analytics is a computer vision application designed to analyze sports videos, specifically soccer, by tracking players and the ball, estimating individual player poses, and providing visual insights. The app enables users to upload a video, run tracking and pose estimation, and visualize results interactively.

## Technologies Used

- Python 3.8+
- Streamlit (for interactive web UI)
- OpenCV (video processing)
- Ultralytics YOLOv8 (object detection and tracking)
- ByteTrack (multi-object tracking)
- OpenPose (pose estimation)
- Other dependencies: numpy, json, pathlib

## Installation

### 1. Install SAM 2 (Segment Anything Model v2)

Follow the official instructions from Facebook Research:

```
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

See the [SAM 2 GitHub page](https://github.com/facebookresearch/sam2) for details and troubleshooting.

### 2. Download SAM 2.1 Small Model Checkpoint

Create a `checkpoints/` directory in your project root (if it doesn't exist):

```
mkdir checkpoints
```

Download the small model checkpoint from the official release:

- [sam2.1_hiera_small.pt](https://github.com/facebookresearch/sam2/releases/download/v1.0.0/sam2.1_hiera_small.pt)

Save the downloaded file to your `checkpoints/` directory:

```
mv sam2.1_hiera_small.pt checkpoints/
```

1. **Clone the repository:**

   ```bash
   git clone https://github.com/raj1401/smart-sport-analytics.git
   cd smart-sport-analytics
   ```

2. **Create and activate a Python virtual environment (recommended):**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies:**

   ```bash
   uv pip install --system --all
   ```

   If you need to add new packages, use:

   ```bash
   uv pip install <package-name>
   uv pip freeze > uv.lock
   ```

   > **Note:** You may need to install OpenPose and ByteTrack separately. See their official documentation for setup.

   > **Note:** SAM 2 must be installed and the checkpoint downloaded as described above for SAM 2.1 analysis features to work.

4. **Download YOLOv8 model weights:**

   - Place your YOLOv8 weights (e.g., `yolov8x.pt`) in the `models/` directory.

5. **Set up OpenPose:**
   - Download and install OpenPose from [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
   - Ensure `OpenPoseDemo.exe` is available at `openpose/bin/OpenPoseDemo.exe` (or update the path in `pose_estimation.py`)

## Running the App

1. **Start the Streamlit app:**

   ```bash
   streamlit run main.py
   ```

2. **Using the App:**
   - Upload a soccer video file (MP4 format recommended).
   - Use the "Track Players and Ball" section to run YOLOv8-based tracking and view the annotated video.
   - Use the "Track Individual Players" section to track all players, select a player ID, and estimate their pose using OpenPose.
   - All processed videos and JSON files are stored in the `temp/` folder.

## Folder Structure

```
smart-sport-analytics/
├── main.py
├── tracking_inference.py
├── track_players.py
├── pose_estimation.py
├── models/
│   └── yolov8x.pt
├── temp/
├── datasets/
├── requirements.txt
└── README.md
```

## Troubleshooting

- Ensure all dependencies are installed and model weights are present.
- For OpenPose, verify that the binary path is correct and all required DLLs are available.
- If you encounter errors, check the terminal output and logs for details.

## License

This project is for educational and research purposes. See LICENSE for details.
