import streamlit as st
import os
from utils import write_h264_for_streamlit
import json
from tracking_inference import run_inference_video
from track_players import track_players_video, crop_track_video
from pose_estimation import run_openpose_video
from sam2_analyzer import sam2_track_video


# Helper to save uploaded file to temp
def save_uploaded_file(uploaded_file):
    temp_dir = os.path.join("temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Smart Sport Analytics</h1>", unsafe_allow_html=True)

# Initialize session state for cached paths
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "tracked_players_ball_path" not in st.session_state:
    st.session_state.tracked_players_ball_path = None
if "tracked_individuals_path" not in st.session_state:
    st.session_state.tracked_individuals_path = None
if "tracks_json_path" not in st.session_state:
    st.session_state.tracks_json_path = None
if "crop_out_path" not in st.session_state:
    st.session_state.crop_out_path = None
if "sam2_output_path" not in st.session_state:
    st.session_state.sam2_output_path = None
# if "sam2_max_frames" not in st.session_state:
#     st.session_state.sam2_max_frames = 125

# Reset cached paths if a new file is uploaded
def on_file_upload():
    st.session_state.uploaded_video_path = None
    st.session_state.tracked_players_ball_path = None
    st.session_state.tracked_individuals_path = None
    st.session_state.tracks_json_path = None
    st.session_state.crop_out_path = None
    st.session_state.sam2_output_path = None
    st.session_state.sam2_max_frames = 125



uploaded_file = st.file_uploader("Upload a video file", type=["mp4"], on_change=on_file_upload)

if uploaded_file:
    video_path = save_uploaded_file(uploaded_file)
    st.session_state.uploaded_video_path = video_path
    st.video(open(write_h264_for_streamlit(video_path), "rb").read())

# --- Track Players and Ball ---
def on_track_players_ball():
    temp_dir = os.path.join("temp")
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, "tracked_players_ball.mp4")
    try:
        model_path = "models/yolov8x.pt"
        with st.spinner("Running tracking inference..."):
            run_inference_video(model_path, st.session_state.uploaded_video_path, out_path)
        st.session_state.tracked_players_ball_path = out_path
        st.success("Inference complete!")
    except Exception as e:
        st.error(f"Error: {e}")

if st.session_state.uploaded_video_path:
    st.markdown("<h2 style='text-align: center;'>Track Players and Ball</h2>", unsafe_allow_html=True)
    st.button("Run Tracking Inference", on_click=on_track_players_ball, use_container_width=True)
    if st.session_state.tracked_players_ball_path:
        st.video(open(write_h264_for_streamlit(st.session_state.tracked_players_ball_path), "rb").read())

# --- SAM 2.1 Analysis ---
def on_run_sam2_analysis():
    temp_dir = os.path.join("temp")
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, "sam2_tracked_output.mp4")
    try:
        with st.spinner("Running SAM 2.1 analysis..."):
            result_path = sam2_track_video(
                video_path=st.session_state.uploaded_video_path,
                yolo_model_path="models/yolov8x.pt",
                out_video_path=out_path,
                max_frames=int(st.session_state.sam2_max_frames),
            )
        st.session_state.sam2_output_path = result_path
        st.success("SAM 2.1 analysis complete!")
    except Exception as e:
        st.error(f"Error: {e}")

if st.session_state.uploaded_video_path:
    st.markdown("<h2 style='text-align: center;'>SAM 2.1 Analysis</h2>", unsafe_allow_html=True)
    st.number_input(
        "Max frames",
        min_value=1,
        max_value=10000,
        value=st.session_state.sam2_max_frames,
        step=1,
        key="sam2_max_frames",
    )
    st.button("Run SAM 2.1 Analysis", on_click=on_run_sam2_analysis, use_container_width=True)
    if st.session_state.sam2_output_path:
        st.video(open(write_h264_for_streamlit(st.session_state.sam2_output_path), "rb").read())

# --- Track Individual Players ---
def on_track_individual_players():
    temp_dir = os.path.join("temp")
    os.makedirs(temp_dir, exist_ok=True)
    out_video_path = os.path.join(temp_dir, "tracked_individuals.mp4")
    tracks_json_path = os.path.join(temp_dir, "tracks.json")
    try:
        model_path = "models/yolov8x.pt"
        with st.spinner("Tracking individual players..."):
            track_players_video(
                model_path=model_path,
                video_path=st.session_state.uploaded_video_path,
                out_video_path=out_video_path,
                save_tracks_json=tracks_json_path
            )
        st.session_state.tracked_individuals_path = out_video_path
        st.session_state.tracks_json_path = tracks_json_path
        st.success("Player tracking complete!")
    except Exception as e:
        st.error(f"Error: {e}")

def on_estimate_pose():
    crop_out_path = os.path.join("temp", f"crop_id_{selected_id}.mp4")
    try:
        with st.spinner("Estimating pose..."):
            crop_track_video(
                video_path=st.session_state.uploaded_video_path,
                tracks_json_path=st.session_state.tracks_json_path,
                track_id=int(selected_id),
                aspect_ratio=(9,16),
                out_path=crop_out_path
            )
            st.session_state.crop_out_path = run_openpose_video(
                input_video=crop_out_path
            )
        st.success("Pose estimation complete!")
    except Exception as e:
        st.error(f"Error: {e}")


if st.session_state.uploaded_video_path:
    st.markdown("<h2 style='text-align: center;'>Track Individual Players</h2>", unsafe_allow_html=True)
    st.button("Track Players", on_click=on_track_individual_players, use_container_width=True)
    if st.session_state.tracked_individuals_path and st.session_state.tracks_json_path:
        col1, col2 = st.columns((3.98, 1.02))
        with col1:
            st.video(open(write_h264_for_streamlit(st.session_state.tracked_individuals_path), "rb").read())
        with col2:
            with open(st.session_state.tracks_json_path, "r") as f:
                tracks_data = json.load(f)
            track_ids = list(tracks_data.keys())
            selected_id = st.selectbox("Select player ID", track_ids)
            st.button("Estimate Pose", on_click=on_estimate_pose, use_container_width=True)
            if st.session_state.crop_out_path:
                st.video(open(write_h264_for_streamlit(st.session_state.crop_out_path), "rb").read())

