"""
app.py — YogaQuest  🧘  Main Streamlit Application
=====================================================
Run with: python -m streamlit run files/app.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, time, tempfile, json, random, math
from datetime import datetime, date

# ── third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ── project modules ───────────────────────────────────────────────────────────
import database as db
from poses_config import (
    YOGA_POSES, LEVEL_XP, LEVEL_NAMES, LEVEL_ICONS,
    ACHIEVEMENTS, SURYA_NAMASKAR_SEQUENCE, SURYA_CORE_POSES,
)
from pose_analyzer import (
    analyze_frame, annotate_frame, SessionAnalyzer,
    score_pose, extract_landmarks, get_grade,
)
from game_engine import GameEngine, level_from_xp, level_progress
from video_processor import process_video, build_pose_segments
from pose_detector import YoloPoseModel
from motion_analyzer import MotionAnalyzer
from wall_game_engine import WallGameEngine, WallGameState, DIFFICULTY_PRESETS
from wall_renderer import render_wall_frame

# ─────────────────────────────────────────────────────────────────────────────
# Page & global config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YogaQuest 🧘",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global background */
.stApp { 
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); 
}

/* Main content area - prevent overlapping */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    z-index: 1000;
}

/* Prevent header overlap */
header[data-testid="stHeader"] {
    background: transparent;
    height: 0px;
}

/* Fix main content positioning */
.main {
    overflow: visible !important;
}

/* Game canvas containers */
.element-container {
    margin-bottom: 0.5rem !important;
}

/* Image containers - prevent overlap */
[data-testid="stImage"] {
    margin: 0 !important;
    position: relative !important;
}

/* Column containers */
[data-testid="column"] {
    padding: 0.25rem !important;
    overflow: visible !important;
}

/* Cards */
.yoga-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    position: relative;
    z-index: 1;
}

/* Score circle */
.score-big {
    font-size: 72px; font-weight: 900; text-align: center;
    background: linear-gradient(135deg, #f9d423, #ff4e50);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}

/* XP badge */
.xp-badge {
    display:inline-block; background:linear-gradient(135deg,#667eea,#764ba2);
    color:#fff; font-weight:700; font-size:14px; border-radius:20px;
    padding:4px 14px;
}

/* Achievement card */
.ach-card {
    display:inline-block; background:rgba(255,215,0,0.12);
    border:1px solid rgba(255,215,0,0.35); border-radius:12px;
    padding:12px 16px; margin:6px; text-align:center; min-width:130px;
}

/* Pose pill */
.pose-pill {
    display:inline-block; border-radius:20px; padding:3px 14px;
    font-size:13px; font-weight:600; margin:3px;
}

/* Grade colours */
.grade-S{color:#FFD700;} .grade-A{color:#4ECDC4;} .grade-B{color:#2ECC71;}
.grade-C{color:#F39C12;} .grade-D{color:#E74C3C;} .grade-F{color:#95A5A6;}

/* Status info styling - fix positioning */
[data-testid="stAlert"] {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    position: relative !important;
    z-index: 100 !important;
}

/* Progress bar override */
.stProgress > div > div > div { background: linear-gradient(90deg,#667eea,#764ba2) !important; }

/* Metric card */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px;
}

/* Header gradient */
h1 { background: linear-gradient(90deg,#f9d423,#ff4e50);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
h2 { color: #e0d7ff; }
h3 { color: #c8baff; }

/* Tab styling */
.stTabs [data-baseweb="tab"] { color: #a090cc; }
.stTabs [aria-selected="true"] { color: #f9d423 !important; border-bottom-color: #f9d423 !important; }

/* Leaderboard rows */
.lb-row {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.lb-row.gold   { background: rgba(255,215,0,0.10); border:1px solid rgba(255,215,0,0.25); }
.lb-row.silver { background: rgba(192,192,192,0.08); border:1px solid rgba(200,200,200,0.2); }
.lb-row.bronze { background: rgba(205,127,50,0.09); border:1px solid rgba(205,127,50,0.25); }

/* ── Wall Game styles ─────────────────────────────────────────────────── */
.wall-verdict {
    text-align:center; border-radius:16px; padding:18px 28px;
    margin:12px 0; font-weight:900; font-size:28px;
    letter-spacing:2px; text-transform:uppercase;
    animation: verdict-pulse 0.6s ease-out;
}
.wall-verdict.perfect {
    background: linear-gradient(135deg, rgba(255,215,0,0.25), rgba(255,78,80,0.15));
    border: 2px solid rgba(255,215,0,0.5);
    color: #FFD700;
    text-shadow: 0 0 20px rgba(255,215,0,0.4);
}
.wall-verdict.great {
    background: linear-gradient(135deg, rgba(78,205,196,0.25), rgba(46,204,113,0.15));
    border: 2px solid rgba(78,205,196,0.4);
    color: #4ECDC4;
}
.wall-verdict.good {
    background: linear-gradient(135deg, rgba(46,204,113,0.2), rgba(39,174,96,0.12));
    border: 2px solid rgba(46,204,113,0.35);
    color: #2ECC71;
}
.wall-verdict.miss {
    background: rgba(243,156,18,0.15);
    border: 2px solid rgba(243,156,18,0.35);
    color: #F39C12;
}
.wall-verdict.collision {
    background: linear-gradient(135deg, rgba(231,76,60,0.25), rgba(192,57,43,0.15));
    border: 2px solid rgba(231,76,60,0.5);
    color: #E74C3C;
    animation: collision-shake 0.4s ease-out;
}
@keyframes verdict-pulse {
    0% { transform: scale(0.85); opacity: 0; }
    60% { transform: scale(1.05); }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes collision-shake {
    0%, 100% { transform: translateX(0); }
    20% { transform: translateX(-6px); }
    40% { transform: translateX(6px); }
    60% { transform: translateX(-4px); }
    80% { transform: translateX(4px); }
}

/* Wall game stat card */
.wall-stat {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}
.wall-stat .stat-value {
    font-size: 36px; font-weight: 900;
    background: linear-gradient(135deg, #f9d423, #ff4e50);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.wall-stat .stat-label {
    color: #a090cc; font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;
}

/* Flow state glow */
.flow-state-badge {
    display:inline-block; padding:6px 20px; border-radius:24px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color:#FFD700; font-weight:800; font-size:14px;
    letter-spacing:1px; text-transform:uppercase;
    animation: flow-glow 1.5s ease-in-out infinite alternate;
    box-shadow: 0 0 20px rgba(102,126,234,0.4);
}
@keyframes flow-glow {
    0% { box-shadow: 0 0 10px rgba(102,126,234,0.3); }
    100% { box-shadow: 0 0 30px rgba(118,75,162,0.6), 0 0 60px rgba(102,126,234,0.2); }
}

/* Collision heatmap bar */
.collision-bar {
    display:flex; align-items:center; gap:8px;
    padding: 4px 0;
}
.collision-bar .bar-fill {
    height:14px; border-radius:7px;
    background: linear-gradient(90deg, #E74C3C, #C0392B);
    transition: width 0.5s ease-out;
}
.collision-bar .bar-label {
    color:#c8baff; font-size:12px; min-width:130px;
}
.collision-bar .bar-count {
    color:#ff6b6b; font-size:12px; font-weight:700;
}

/* Difficulty badge pills */
.diff-easy   { background:rgba(78,205,196,0.15); color:#4ECDC4; border:1px solid rgba(78,205,196,0.3); }
.diff-normal { background:rgba(255,165,0,0.15);  color:#FFA500; border:1px solid rgba(255,165,0,0.3); }
.diff-hard   { background:rgba(231,76,60,0.15);  color:#E74C3C; border:1px solid rgba(231,76,60,0.3); }
.diff-insane { background:rgba(155,89,182,0.15); color:#9B59B6; border:1px solid rgba(155,89,182,0.3); }
.diff-pill {
    display:inline-block; border-radius:20px; padding:4px 16px;
    font-size:13px; font-weight:700;
}

/* ── Pose Target Game ───────────────────────────────────────────────────── */
.ptg-step-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px; padding: 16px 20px; margin-bottom: 10px;
}
.ptg-step-done {
    background: rgba(46,204,113,0.15);
    border: 1px solid rgba(46,204,113,0.4);
}
.ptg-step-active {
    background: rgba(249,212,35,0.12);
    border: 2px solid rgba(249,212,35,0.5);
    animation: ptg-pulse 1.2s ease-in-out infinite alternate;
}
@keyframes ptg-pulse {
    0%   { box-shadow: 0 0 0 rgba(249,212,35,0); }
    100% { box-shadow: 0 0 18px rgba(249,212,35,0.35); }
}

/* ── Falling Pose Game ──────────────────────────────────────────────────── */
.fpg-card {
    display:inline-block; background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18); border-radius: 12px;
    padding: 10px 18px; margin: 6px; text-align:center;
}
.fpg-score-flash {
    font-size:32px; font-weight:900; text-align:center;
    color: #f9d423;
    animation: fpg-flash 0.5s ease-out;
}
@keyframes fpg-flash {
    0%   { transform: scale(1.4); opacity:0; }
    100% { transform: scale(1);   opacity:1; }
}
/* Combo badge */
.fpg-combo-badge {
    display:inline-block; padding:8px 22px; border-radius:30px;
    font-size:22px; font-weight:900; letter-spacing:2px;
    background: linear-gradient(135deg, #f9d423, #ff4e50);
    color:#1a1a2e; text-align:center;
    animation: combo-pop 0.4s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow: 0 0 25px rgba(249,212,35,0.5);
}
@keyframes combo-pop {
    0%   { transform: scale(0.5) rotate(-5deg); opacity:0; }
    60%  { transform: scale(1.15) rotate(2deg); }
    100% { transform: scale(1) rotate(0deg); opacity:1; }
}
.fpg-hud {
    display:flex; justify-content:space-between; align-items:center;
    padding:12px 20px; border-radius:14px;
    background: rgba(10,8,30,0.85);
    border:1px solid rgba(255,255,255,0.12);
    margin-bottom:10px;
}
.fpg-hud-stat { text-align:center; }
.fpg-hud-stat .val {
    font-size:28px; font-weight:900;
    background: linear-gradient(135deg,#f9d423,#ff4e50);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.fpg-hud-stat .lbl { color:#a090cc; font-size:11px; font-weight:600; text-transform:uppercase; }
.fpg-timer-bar-wrap {
    height:10px; border-radius:5px;
    background:rgba(255,255,255,0.08); overflow:hidden; margin:4px 0 10px;
}
.fpg-timer-bar-fill {
    height:100%; border-radius:5px;
    transition: width 0.1s linear;
}
.fpg-caught-banner {
    text-align:center; font-size:34px; font-weight:900;
    padding:14px; border-radius:16px;
    background: linear-gradient(135deg,rgba(0,255,120,0.2),rgba(0,200,255,0.1));
    border:2px solid rgba(0,255,120,0.5);
    color:#00ff88;
    animation: fpg-flash 0.4s ease-out;
    text-shadow: 0 0 20px rgba(0,255,136,0.6);
}
.fpg-miss-banner {
    text-align:center; font-size:26px; font-weight:900;
    padding:10px; border-radius:14px;
    background:rgba(231,76,60,0.15);
    border:2px solid rgba(231,76,60,0.4);
    color:#ff6b6b;
    animation: collision-shake 0.3s ease-out;
}

/* ── Practice Mode Fun Features ──────────────────────────────────────── */
.practice-preset-btn {
    display:inline-block; border-radius:12px; padding:14px 18px;
    text-align:center; cursor:pointer; min-width:120px;
    transition: all 0.25s ease;
}
.practice-preset-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.preset-easy {
    background: linear-gradient(135deg, rgba(78,205,196,0.2), rgba(46,204,113,0.15));
    border: 1px solid rgba(78,205,196,0.4);
}
.preset-warrior {
    background: linear-gradient(135deg, rgba(255,78,80,0.2), rgba(255,165,0,0.15));
    border: 1px solid rgba(255,78,80,0.4);
}
.preset-full {
    background: linear-gradient(135deg, rgba(155,89,182,0.2), rgba(102,126,234,0.15));
    border: 1px solid rgba(155,89,182,0.4);
}
.preset-daily {
    background: linear-gradient(135deg, rgba(249,212,35,0.2), rgba(255,215,0,0.15));
    border: 1px solid rgba(249,212,35,0.4);
}

/* Score reaction animations */
.score-reaction {
    text-align:center; padding:16px; border-radius:16px;
    margin:8px 0; animation: score-reveal 0.6s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes score-reveal {
    0%   { transform: scale(0.3) rotate(-10deg); opacity:0; }
    60%  { transform: scale(1.1) rotate(2deg); }
    100% { transform: scale(1) rotate(0deg); opacity:1; }
}
.score-incredible {
    background: linear-gradient(135deg, rgba(255,215,0,0.25), rgba(255,78,80,0.15));
    border: 2px solid rgba(255,215,0,0.6);
    box-shadow: 0 0 30px rgba(255,215,0,0.3);
    animation: score-reveal 0.6s cubic-bezier(0.34,1.56,0.64,1), golden-glow 1.5s ease-in-out infinite alternate;
}
@keyframes golden-glow {
    0%   { box-shadow: 0 0 15px rgba(255,215,0,0.2); }
    100% { box-shadow: 0 0 40px rgba(255,215,0,0.5), 0 0 80px rgba(255,78,80,0.2); }
}
.score-great {
    background: linear-gradient(135deg, rgba(78,205,196,0.2), rgba(46,204,113,0.15));
    border: 2px solid rgba(78,205,196,0.5);
    box-shadow: 0 0 20px rgba(78,205,196,0.2);
}
.score-good {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.12));
    border: 2px solid rgba(102,126,234,0.4);
}
.score-keep-going {
    background: rgba(255,255,255,0.06);
    border: 2px solid rgba(255,255,255,0.15);
}

/* Combo badge for practice */
.practice-combo {
    display:inline-block; padding:8px 24px; border-radius:30px;
    font-size:20px; font-weight:900; letter-spacing:1px;
    text-align:center;
    animation: combo-pop 0.4s cubic-bezier(0.34,1.56,0.64,1);
}
.combo-fire {
    background: linear-gradient(135deg, #f9d423, #ff4e50);
    color:#1a1a2e;
    box-shadow: 0 0 25px rgba(249,212,35,0.5);
}
.combo-break {
    background: rgba(231,76,60,0.2);
    color:#ff6b6b;
    border: 1px solid rgba(231,76,60,0.4);
    animation: collision-shake 0.4s ease-out;
}

/* Speed challenge timer */
.speed-timer {
    display:inline-block; padding:6px 18px; border-radius:20px;
    background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.5);
    color:#e0d7ff; font-weight:700; font-size:16px;
    animation: timer-pulse 1s ease-in-out infinite;
}
@keyframes timer-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.7; }
}

/* MVP / Needs Love cards */
.mvp-card {
    background: linear-gradient(135deg, rgba(255,215,0,0.15), rgba(255,78,80,0.08));
    border: 2px solid rgba(255,215,0,0.4);
    border-radius: 16px; padding: 20px; text-align:center;
    animation: score-reveal 0.6s cubic-bezier(0.34,1.56,0.64,1);
}
.needs-love-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.08));
    border: 2px solid rgba(102,126,234,0.3);
    border-radius: 16px; padding: 20px; text-align:center;
    animation: score-reveal 0.8s cubic-bezier(0.34,1.56,0.64,1);
}

/* Session title banner */
.session-title-banner {
    text-align:center; padding:20px; border-radius:20px;
    margin:12px 0;
    background: linear-gradient(135deg, rgba(249,212,35,0.12), rgba(255,78,80,0.08), rgba(102,126,234,0.12));
    border: 1px solid rgba(255,255,255,0.15);
    animation: score-reveal 0.5s cubic-bezier(0.34,1.56,0.64,1);
}

/* Motivational quote */
.motive-quote {
    text-align:center; padding:12px 20px; border-radius:12px;
    background: rgba(255,255,255,0.04);
    border-left: 3px solid rgba(249,212,35,0.5);
    color:#c8baff; font-style:italic; font-size:14px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DB bootstrap
# ─────────────────────────────────────────────────────────────────────────────
db.init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Cached YOLO pose model (shared across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_pose_model():
    return YoloPoseModel(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

@st.cache_resource
def get_static_pose_model():
    """For single-frame (camera_input) analysis."""
    return YoloPoseModel(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "user_id":          None,
        "username":         "",
        "page":             "home",
        # Practice session
        "practice_poses":   [],
        "practice_results": None,
        # Live webcam session
        "live_active":      False,
        "live_frames":      [],
        "live_start":       None,
        "live_target":      "Auto-Detect",
        "live_results":     None,
        # Video upload session
        "video_results":    None,
        "video_out_path":   None,
        # Dynamic Pose Wall Challenge
        "wall_game_engine":  None,
        "wall_game_results": None,
        "motion_analyzer":   None,
        # Pose Target Game
        "ptg_active":        False,
        "ptg_pose":          "Tree Pose",
        "ptg_step":          0,
        "ptg_completed":     False,
        "ptg_completed_at":  None,
        "ptg_particles":     [],
        "ptg_score":         0,
        "ptg_rounds_done":   0,
        "ptg_total_rounds":  5,
        "ptg_start_time":    None,
        "ptg_results":       None,
        # Falling Pose Game
        "fpg_active":        False,
        "fpg_score":         0,
        "fpg_start_time":    None,
        "fpg_active_cards":  [],   # list of dicts
        "fpg_results":       None,
        "fpg_duration":      60,
        # Practice Mode fun features
        "practice_speed_challenge": False,
        "practice_speed_start":     None,
        "practice_combo":           0,
        "practice_best_combo":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


@st.cache_resource
def open_camera(camera_index: int = 0):
    """
    Open the webcam once and keep it alive across Streamlit reruns.
    Optimized for fast startup on Windows.
    """
    # Try CAP_DSHOW first (Windows optimized backend)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        cap.release()
        # Fallback to default backend
        cap = cv2.VideoCapture(camera_index)
    
    if cap.isOpened():
        # Optimize for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        
        # Disable auto-focus and auto-exposure for faster startup
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure mode
        
        # Set FPS to match game requirements (reduces latency)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Warm up: flush initial buffered frames that may be stale
        for _ in range(3):
            cap.read()
        
    return cap


def release_camera() -> None:
    """Release the cached OpenCV camera capture."""
    cap = open_camera()
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    open_camera.clear()


def enhance_frame_brightness(frame, alpha=1.4, beta=30):
    """
    Enhance frame brightness and contrast for better visibility.
    
    Args:
        frame: Input BGR frame
        alpha: Contrast control (1.0-3.0, default 1.4)
        beta: Brightness control (0-100, default 30)
    
    Returns:
        Enhanced BGR frame
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def should_analyze_now(key: str, interval: float = 0.18) -> bool:
    """Throttle expensive pose detection while keeping the camera preview live."""
    now = time.time()
    last_key = f"{key}_last_analysis_ts"
    if now - st.session_state.get(last_key, 0.0) >= interval:
        st.session_state[last_key] = now
        return True
    return False


def get_camera():
    """
    Return an open camera capture.
    Maintains stable connection during active use - only recreates if truly necessary.
    """
    cap = open_camera()
    
    # Only recreate if the capture object is None (not just isOpened check)
    if cap is None:
        open_camera.clear()
        cap = open_camera()
    
    return cap


def prewarm_camera():
    """
    Pre-warm camera in background before game starts.
    Call this when user is on config screen to reduce startup delay.
    """
    try:
        cap = get_camera()
        if cap.isOpened():
            # Flush stale frames
            for _ in range(5):
                cap.read()
            return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — login & navigation
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("# 🧘 YogaQuest")
        st.markdown("*Your AI-powered yoga companion*")
        st.divider()

        # ── Login ─────────────────────────────────────────────────────────────
        if not st.session_state.user_id:
            st.markdown("### 👤 Enter your name")
            existing = [r[1] for r in db.get_all_users()]
            uname = st.text_input("Username", placeholder="e.g. Arjun")
            if st.button("▶️  Start Playing", type="primary", use_container_width=True):
                if uname.strip():
                    uid = db.get_or_create_user(uname.strip())
                    st.session_state.user_id  = uid
                    st.session_state.username = uname.strip()
                    st.rerun()
                else:
                    st.warning("Please enter a username.")
            if existing:
                st.markdown("**Or pick an existing player:**")
                choice = st.selectbox("", ["— select —"] + existing, label_visibility="collapsed")
                if choice != "— select —":
                    uid = db.get_or_create_user(choice)
                    st.session_state.user_id  = uid
                    st.session_state.username = choice
                    st.rerun()
            return

        # ── User card ──────────────────────────────────────────────────────────
        stats = db.get_user_stats(st.session_state.user_id)
        level = stats["level"] if stats else 1
        xp    = stats["total_xp"] if stats else 0
        _, xp_need, xp_pct = level_progress(xp, level)
        lname = LEVEL_NAMES.get(level, "Yogi")
        licon = LEVEL_ICONS.get(level, "🧘")

        st.markdown(f"""
        <div class="yoga-card" style="margin-bottom:10px">
          <div style="font-size:22px;font-weight:700;color:#f9d423">{licon} {st.session_state.username}</div>
          <div style="color:#a090cc;font-size:13px">{lname} — Level {level}</div>
          <div style="color:#ccc;font-size:12px;margin-top:6px">⚡ {xp:,} XP
               &nbsp;|&nbsp; 🔥 {stats.get('streak_days',0)} day streak</div>
        </div>
        """, unsafe_allow_html=True)

        xp_pct_clamped = min(1.0, xp_pct)
        st.progress(xp_pct_clamped, text=f"XP to Lv {level+1}: {int(xp_pct_clamped*100)}%")

        st.divider()

        # ── Navigation ────────────────────────────────────────────────────────
        pages = [
            ("🏠 Home",                  "home"),
            ("📸 Practice Mode",         "practice"),
            ("📹 Live Session",          "live"),
            ("🧱 Pose Wall Game",         "wall_game"),
            ("🎯 Pose Target Game",      "pose_target_game"),
            ("🌊 Falling Pose Game",     "falling_pose_game"),
            ("🎬 Video Analysis",        "video"),
            ("☀️ Surya Namaskar",        "surya"),
            ("🏆 Leaderboard",           "leaderboard"),
            ("👤 My Profile",            "profile"),
        ]
        st.markdown("### Navigate")
        for label, key in pages:
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if st.session_state.page == key else "secondary"):
                if key != st.session_state.page:
                    release_camera()
                    st.session_state.live_active = False
                st.session_state.page = key
                st.rerun()

        st.divider()
        if st.button("🚪 Log Out", use_container_width=True):
            release_camera()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_state()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Utility renderers
# ─────────────────────────────────────────────────────────────────────────────
GRADE_COLORS = {
    "S": "#FFD700", "A": "#4ECDC4", "B": "#2ECC71",
    "C": "#F39C12", "D": "#E74C3C", "F": "#95A5A6",
}

def score_gauge(score: float, title: str = "Overall Score") -> go.Figure:
    color = "#FFD700" if score >= 90 else "#4ECDC4" if score >= 70 else "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"color": "#e0d7ff", "size": 14}},
        number={"font": {"color": color, "size": 36}, "suffix": ""},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#666", "tickwidth": 1},
            "bar":   {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 50],  "color": "rgba(231,76,60,0.15)"},
                {"range": [50, 70], "color": "rgba(243,156,18,0.15)"},
                {"range": [70, 90], "color": "rgba(46,204,113,0.12)"},
                {"range": [90,100], "color": "rgba(255,215,0,0.18)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=30, b=10, l=10, r=10),
        font_color="#ccc",
    )
    return fig


def pose_bar_chart(pose_details: list) -> go.Figure:
    names   = [p["english_name"] for p in pose_details]
    scores  = [p["adjusted_score"] for p in pose_details]
    colors  = [GRADE_COLORS.get(p["grade"], "#888") for p in pose_details]

    fig = go.Figure(go.Bar(
        x=names, y=scores, marker_color=colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.06)",
                   color="#ccc", title="Score"),
        xaxis=dict(color="#ccc"),
        font_color="#ccc", height=280,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


def xp_breakdown_chart(xp_bp: dict) -> go.Figure:
    labels, vals = [], []
    label_map = {
        "base_xp":          "Base (poses)",
        "score_bonus":      "Score bonus",
        "streak_bonus":     "Streak bonus",
        "combo_bonus":      "Combo bonus",
        "hold_bonus":       "Hold bonus",
        "perfect_bonus":    "Perfect pose",
        "achievement_bonus": "Achievements",
        "stability_bonus":  "Stability",
        "smoothness_bonus": "Smoothness",
        "flow_state_bonus": "Flow state",
        "wall_bonus":       "Wall game",
    }
    for k, lbl in label_map.items():
        v = xp_bp.get(k, 0)
        if v > 0:
            labels.append(lbl)
            vals.append(v)
    if not vals:
        return go.Figure()
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.55,
        marker_colors=["#667eea","#764ba2","#f9d423","#ff4e50",
                        "#4ECDC4","#2ECC71","#FFD700"],
        textinfo="label+value", textfont_size=11,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", height=240,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(font_color="#ccc"),
    )
    return fig


def render_achievements(new_ach: list):
    if not new_ach:
        return
    st.markdown("### 🎉 New Achievements Unlocked!")
    cols = st.columns(min(len(new_ach), 4))
    for i, ach in enumerate(new_ach):
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="ach-card">
              <div style="font-size:32px">{ach['icon']}</div>
              <div style="font-weight:700;color:#f9d423;font-size:13px">{ach['name']}</div>
              <div style="color:#aaa;font-size:11px">{ach['description']}</div>
              {f'<div class="xp-badge">+{ach["xp_bonus"]} XP</div>' if ach.get("xp_bonus") else ""}
            </div>
            """, unsafe_allow_html=True)


def render_session_report(session_results: dict, game_report: dict):
    """Full results page shown after completing any session."""
    st.balloons()
    st.markdown("# 🎊 Session Complete!")
    st.divider()

    # ── Top metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎯 Overall Score",   f"{session_results['total_score']:.0f} / 100")
    c2.metric("⚡ XP Earned",       f"+{game_report['xp_earned']:,}")
    c3.metric("🧘 Poses Completed", session_results['poses_completed'])
    c4.metric("⏱️ Duration",        f"{session_results['duration']:.0f}s")
    c5.metric("🔥 Streak",          f"{game_report['streak_days']} days")

    st.divider()

    # ── Gauge + bar chart ──────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.plotly_chart(score_gauge(session_results["total_score"]), use_container_width=True)
        grade = get_grade(session_results["total_score"])
        gc = GRADE_COLORS.get(grade, "#888")
        st.markdown(f"""
        <div style="text-align:center;font-size:48px;font-weight:900;color:{gc}">
          Grade: {grade}
        </div>""", unsafe_allow_html=True)

    with col_right:
        if session_results.get("pose_details"):
            st.markdown("#### 📊 Per-Pose Scores")
            st.plotly_chart(pose_bar_chart(session_results["pose_details"]),
                            use_container_width=True)

    st.divider()

    # ── Level-up banner ────────────────────────────────────────────────────
    if game_report.get("level_up"):
        st.success(f"""
        🎉 **LEVEL UP!**  {game_report['level_icon']}  You reached **Level {game_report['level']}**
        — {game_report['level_name']}!
        """)

    # ── XP breakdown + new achievements ────────────────────────────────────
    col_xp, col_ach = st.columns(2)
    with col_xp:
        st.markdown("#### ⚡ XP Breakdown")
        st.plotly_chart(xp_breakdown_chart(game_report["xp_breakdown"]),
                        use_container_width=True)
        st.markdown(f"**Total XP earned:** `{game_report['xp_earned']:,}` &nbsp;&nbsp; "
                    f"**Total XP:** `{game_report['total_xp']:,}`")
        if game_report.get("xp_needed", 0) > 0:
            pct = int(game_report["xp_pct"] * 100)
            st.progress(game_report["xp_pct"],
                        text=f"Level {game_report['level']+1} progress: {pct}%")

    with col_ach:
        if game_report.get("new_achievements"):
            render_achievements(game_report["new_achievements"])
        else:
            st.markdown("#### 🏅 Keep Going!")
            st.info("No new achievements this session, but you're making great progress!")

    st.divider()

    # ── Per-pose details ────────────────────────────────────────────────────
    if session_results.get("pose_details"):
        st.markdown("#### 🧘 Detailed Pose Breakdown")
        for p in session_results["pose_details"]:
            grade_col = GRADE_COLORS.get(p["grade"], "#888")
            with st.expander(
                f"{p['emoji']} {p['english_name']}  —  "
                f"Score: {p['adjusted_score']:.0f}  |  Grade: {p['grade']}", expanded=False
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Base Score",   f"{p['score']:.0f}")
                c2.metric("Hold Bonus",   f"+{p['hold_bonus']:.1f}")
                c3.metric("Final Score",  f"{p['adjusted_score']:.0f}")
                c4.metric("Hold Time",    f"{p['hold_time']:.1f}s")

                st.markdown(f"**Stability index:** {p['stability']:.0f} / 100")
                st.progress(p["stability"] / 100)

                if p.get("suggestions"):
                    st.markdown("**💡 Improvement tips:**")
                    for tip in p["suggestions"]:
                        st.markdown(f"- {tip}")

    # ── Overall suggestions ────────────────────────────────────────────────
    if session_results.get("suggestions"):
        st.divider()
        st.markdown("#### 💡 Top Session Suggestions")
        for tip in session_results["suggestions"]:
            st.markdown(f"- {tip}")

    # ── Surya Namaskar progress ────────────────────────────────────────────
    sn = session_results.get("surya_namaskar_pct", 0)
    if sn > 0:
        st.divider()
        st.markdown(f"#### ☀️ Surya Namaskar Completion: {sn*100:.0f}%")
        st.progress(sn)
        if sn >= 0.9:
            st.success("🌟 You completed a full Surya Namaskar cycle!")

    st.divider()
    if st.button("🔄 Practice Again", type="primary"):
        st.session_state.practice_results = None
        st.session_state.video_results    = None
        st.session_state.page             = "practice"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Page: Home
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("# 🧘 Welcome to YogaQuest")
    st.markdown("*Your AI-powered yoga journey begins here*")
    st.divider()

    uid   = st.session_state.user_id
    stats = db.get_user_stats(uid)

    if not stats:
        st.info("Start your first session to see your stats!")
        return

    level   = stats["level"]
    xp      = stats["total_xp"]
    _, xp_need, xp_pct = level_progress(xp, level)

    # ── Stats row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⚡ Total XP",         f"{xp:,}")
    c2.metric(f"{LEVEL_ICONS.get(level,'🧘')} Level", f"{level} — {LEVEL_NAMES.get(level,'')}")
    c3.metric("🔥 Streak",           f"{stats['streak_days']} days")
    c4.metric("📅 Sessions",         stats["total_sessions"])
    c5.metric("⭐ Perfect Poses",    stats["perfect_poses"])

    if xp_need:
        st.progress(xp_pct, text=f"Progress to Level {level+1}: {int(xp_pct*100)}%")

    st.divider()

    # ── Daily challenge ────────────────────────────────────────────────────
    daily_poses = list(YOGA_POSES.keys())
    challenge_idx = date.today().toordinal() % len(daily_poses)
    challenge_pose = daily_poses[challenge_idx]
    cp_cfg = YOGA_POSES[challenge_pose]

    st.markdown("### 🌟 Daily Challenge")
    st.markdown(f"""
    <div class="yoga-card">
      <div style="font-size:36px">{cp_cfg['emoji']}</div>
      <div style="font-size:22px;font-weight:700;color:#f9d423">{cp_cfg['english']}</div>
      <div style="color:#c8baff;font-size:13px">{cp_cfg['description']}</div>
      <div style="margin-top:8px;color:#aaa;font-size:12px">
        Difficulty: {'⭐' * cp_cfg['difficulty']}{'☆' * (3-cp_cfg['difficulty'])}
        &nbsp;|&nbsp; Target hold: {cp_cfg['hold_time']}s
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🎯 Accept Daily Challenge", type="primary"):
        st.session_state.page = "practice"
        st.session_state.practice_poses = [challenge_pose]
        st.rerun()

    st.divider()

    # ── Quick start grid ───────────────────────────────────────────────────
    st.markdown("### 🚀 Quick Start")
    cols = st.columns(4)
    quick = [
        ("📸 Practice Mode",      "practice",          "Snapshot-based pose training"),
        ("📹 Live Session",       "live",               "Real-time webcam analysis"),
        ("🧱 Pose Wall Game",     "wall_game",          "Arcade wall challenge"),
        ("🎯 Pose Target Game",   "pose_target_game",   "Hit targets with your limbs"),
        ("🌊 Falling Pose Game",  "falling_pose_game",  "Catch falling pose cards"),
        ("🎬 Video Upload",       "video",              "Analyse a recorded session"),
        ("☀️ Surya Namaskar",     "surya",              "Guided sun salutation"),
    ]
    for i, (label, page_key, desc) in enumerate(quick):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="yoga-card" style="text-align:center;cursor:pointer">
              <div style="font-size:26px">{label[:2]}</div>
              <div style="font-weight:700;color:#e0d7ff">{label[2:]}</div>
              <div style="color:#888;font-size:12px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Go →", key=f"qs_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

    # ── Recent history ─────────────────────────────────────────────────────
    history = db.get_session_history(uid, limit=5)
    if history:
        st.divider()
        st.markdown("### 📜 Recent Sessions")
        rows = []
        for row in history:
            _, sdate, stype, tscore, xp_e, dur, _ = row
            rows.append({
                "Date":   sdate[:16].replace("T", " "),
                "Type":   stype.title(),
                "Score":  f"{tscore:.0f}",
                "XP":     f"+{xp_e}",
                "Duration": f"{dur:.0f}s",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Practice Mode — helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_motivational_quote() -> str:
    """Return a random motivational quote for between poses."""
    quotes = [
        "🌬️ Breathe in strength, breathe out doubt.",
        "🏛️ Your body is a temple — honour it.",
        "💫 One pose at a time, one breath at a time.",
        "🌊 Flow like water, stand like a mountain.",
        "🔥 You're doing amazing — keep that fire burning!",
        "🌸 Progress, not perfection.",
        "⚡ Energy flows where intention goes.",
        "🧘 The pose begins when you want to leave it.",
        "🌙 Be patient with yourself — growth takes time.",
        "🌈 Every expert was once a beginner.",
        "💎 Diamonds are made under pressure — keep going!",
        "🦋 Flexibility of mind leads to flexibility of body.",
        "🌟 You showed up today — that's already a win.",
        "🎯 Focus on the journey, not the destination.",
        "🍃 Let go of what you can't control.",
    ]
    return random.choice(quotes)


def _speed_challenge_rating(seconds: float) -> tuple:
    """Return (emoji, title, color) based on completion time."""
    if seconds < 60:
        return "⚡", "Speed Demon", "#FFD700"
    elif seconds < 120:
        return "🐇", "Quick Flow", "#4ECDC4"
    elif seconds < 180:
        return "🏃", "Steady Pace", "#2ECC71"
    else:
        return "🐌", "Chill Yogi", "#a090cc"


def _score_reaction_html(score: float, grade: str) -> str:
    """Return animated HTML for a dramatic score reveal."""
    gc = GRADE_COLORS.get(grade, "#888")
    if score >= 90:
        css_class = "score-incredible"
        emoji = "🔥"
        label = "INCREDIBLE!"
        sub = "Perfect form — you're on fire!"
    elif score >= 70:
        css_class = "score-great"
        emoji = "✨"
        label = "Great Form!"
        sub = "Almost perfect — keep refining!"
    elif score >= 50:
        css_class = "score-good"
        emoji = "💪"
        label = "Keep Pushing!"
        sub = "Good effort — you're getting there!"
    else:
        css_class = "score-keep-going"
        emoji = "🧘"
        label = "Keep Practicing!"
        sub = "Every attempt makes you better."

    return f"""
    <div class="score-reaction {css_class}">
      <div style="font-size:42px">{emoji}</div>
      <div style="font-size:36px;font-weight:900;color:{gc};margin:4px 0">
        {score:.0f}/100 &nbsp; Grade: {grade}
      </div>
      <div style="font-size:20px;font-weight:700;color:#e0d7ff">{label}</div>
      <div style="font-size:13px;color:#a090cc;margin-top:4px">{sub}</div>
    </div>"""


def _session_title(avg_score: float) -> tuple:
    """Return (emoji, title) based on average session score."""
    if avg_score >= 95:
        return "👑", "Yoga Legend"
    elif avg_score >= 85:
        return "🌟", "Asana Master"
    elif avg_score >= 75:
        return "⚡", "Rising Star"
    elif avg_score >= 60:
        return "💪", "Determined Warrior"
    elif avg_score >= 40:
        return "🌱", "Growing Yogi"
    else:
        return "🧘", "Brave Beginner"


# ─────────────────────────────────────────────────────────────────────────────
# Page: Practice Mode (camera_input snapshot) — with fun features!
# ─────────────────────────────────────────────────────────────────────────────
def page_practice():
    st.markdown("# 📸 Practice Mode")
    st.markdown("Strike a pose, snap a photo — the AI scores your form with style! 🎮")
    st.divider()

    uid     = st.session_state.user_id
    results = st.session_state.practice_results

    # ── Show report if session already finished ───────────────────────────
    if results:
        ge = GameEngine(uid)
        game_report = ge.process_session(results)

        # ── Enhanced session summary (MVP, Needs Love, Title) ─────────
        st.balloons()
        st.markdown("# 🎊 Session Complete!")

        pose_details = results.get("pose_details", [])
        avg_score = results.get("total_score", 0)
        title_emoji, title_text = _session_title(avg_score)

        # Session title banner
        st.markdown(f"""
        <div class="session-title-banner">
          <div style="font-size:48px">{title_emoji}</div>
          <div style="font-size:28px;font-weight:900;
                      background:linear-gradient(135deg,#f9d423,#ff4e50);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            {title_text}
          </div>
          <div style="color:#a090cc;font-size:14px;margin-top:4px">
            Average Score: {avg_score:.0f}/100
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Speed challenge result
        if results.get("speed_challenge") and results.get("speed_time"):
            spd_emoji, spd_title, spd_color = _speed_challenge_rating(results["speed_time"])
            st.markdown(f"""
            <div style="text-align:center;margin:10px 0">
              <span class="speed-timer" style="font-size:20px">
                {spd_emoji} {spd_title} — {results['speed_time']:.1f}s
              </span>
            </div>
            """, unsafe_allow_html=True)

        # Combo result
        best_combo = results.get("best_combo", 0)
        if best_combo >= 2:
            st.markdown(f"""
            <div style="text-align:center;margin:8px 0">
              <span class="practice-combo combo-fire">
                🔥 Best Combo: {best_combo}x
              </span>
            </div>
            """, unsafe_allow_html=True)

        # MVP and Needs Love
        if len(pose_details) >= 2:
            mvp = max(pose_details, key=lambda p: p["adjusted_score"])
            needs_love = min(pose_details, key=lambda p: p["adjusted_score"])

            col_mvp, col_love = st.columns(2)
            with col_mvp:
                st.markdown(f"""
                <div class="mvp-card">
                  <div style="font-size:36px">🏆</div>
                  <div style="font-size:12px;color:#a090cc;text-transform:uppercase;
                              font-weight:700;letter-spacing:1px">MVP Pose</div>
                  <div style="font-size:22px;font-weight:800;color:#FFD700;margin:4px 0">
                    {mvp['emoji']} {mvp['english_name']}
                  </div>
                  <div style="font-size:28px;font-weight:900;color:#4ECDC4">
                    {mvp['adjusted_score']:.0f}/100
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with col_love:
                st.markdown(f"""
                <div class="needs-love-card">
                  <div style="font-size:36px">💜</div>
                  <div style="font-size:12px;color:#a090cc;text-transform:uppercase;
                              font-weight:700;letter-spacing:1px">Needs Love</div>
                  <div style="font-size:22px;font-weight:800;color:#c8baff;margin:4px 0">
                    {needs_love['emoji']} {needs_love['english_name']}
                  </div>
                  <div style="font-size:28px;font-weight:900;color:#667eea">
                    {needs_love['adjusted_score']:.0f}/100
                  </div>
                  <div style="font-size:11px;color:#888;margin-top:4px">
                    Try this one again next session!
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Delegate to the shared report renderer for XP, achievements, etc.
        render_session_report(results, game_report)
        return

    # ══════════════════════════════════════════════════════════════════════
    #  ACTIVE SESSION — pose selection & capture
    # ══════════════════════════════════════════════════════════════════════

    all_names = list(YOGA_POSES.keys())

    # ── 🌈 Difficulty Quick-Pick Presets ──────────────────────────────────
    st.markdown("### 🎮 Quick Start Presets")
    st.markdown("<div style='color:#888;font-size:13px;margin-bottom:8px'>"
                "Pick a preset or build your own below!</div>", unsafe_allow_html=True)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown("""
        <div class="practice-preset-btn preset-easy">
          <div style="font-size:24px">🌱</div>
          <div style="font-weight:700;color:#4ECDC4;font-size:13px">Easy Flow</div>
          <div style="color:#888;font-size:10px">3 easiest</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="preset_easy", use_container_width=True):
            easy = sorted(all_names, key=lambda n: YOGA_POSES[n]["difficulty"])[:3]
            st.session_state.practice_poses = easy
            st.rerun()

    with p2:
        st.markdown("""
        <div class="practice-preset-btn preset-warrior">
          <div style="font-size:24px">⚔️</div>
          <div style="font-weight:700;color:#ff4e50;font-size:13px">Warrior Challenge</div>
          <div style="color:#888;font-size:10px">Warriors + Tree</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="preset_warrior", use_container_width=True):
            warriors = [n for n in all_names if "Virabhadrasana" in n or "Vrikshasana" in n]
            st.session_state.practice_poses = warriors or all_names[:3]
            st.rerun()

    with p3:
        st.markdown("""
        <div class="practice-preset-btn preset-full">
          <div style="font-size:24px">🔥</div>
          <div style="font-weight:700;color:#9B59B6;font-size:13px">Full Send</div>
          <div style="color:#888;font-size:10px">All poses</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="preset_full", use_container_width=True):
            st.session_state.practice_poses = list(all_names)
            st.rerun()

    with p4:
        st.markdown("""
        <div class="practice-preset-btn preset-daily">
          <div style="font-size:24px">🎯</div>
          <div style="font-weight:700;color:#f9d423;font-size:13px">Daily Focus</div>
          <div style="color:#888;font-size:10px">Today's + 2</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="preset_daily", use_container_width=True):
            daily_idx = date.today().toordinal() % len(all_names)
            daily = all_names[daily_idx]
            others = [n for n in all_names if n != daily]
            picks = [daily] + random.sample(others, min(2, len(others)))
            st.session_state.practice_poses = picks
            st.rerun()

    with p5:
        st.markdown("""
        <div class="practice-preset-btn" style="background:linear-gradient(135deg,rgba(255,255,255,0.08),rgba(255,255,255,0.04));
             border:1px solid rgba(255,255,255,0.2)">
          <div style="font-size:24px">🎲</div>
          <div style="font-weight:700;color:#e0d7ff;font-size:13px">Surprise Me!</div>
          <div style="color:#888;font-size:10px">Random 3</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select", key="preset_random", use_container_width=True):
            picks = random.sample(all_names, min(3, len(all_names)))
            st.session_state.practice_poses = picks
            st.rerun()

    st.divider()

    # ── Pose selector (manual) ────────────────────────────────────────────
    st.markdown("### 1️⃣ Select poses for this session")
    selected = st.multiselect(
        "Poses to practice (pick one or more)",
        options=all_names,
        default=st.session_state.practice_poses or all_names[:3],
        format_func=lambda n: f"{YOGA_POSES[n]['emoji']} {YOGA_POSES[n]['english']}",
    )
    st.session_state.practice_poses = selected

    if not selected:
        st.warning("Please select at least one pose.")
        return

    st.divider()

    # ── ⏱️ Speed Challenge Toggle ─────────────────────────────────────────
    st.markdown("### 2️⃣ Capture each pose")

    opt_col1, opt_col2, opt_col3 = st.columns([2, 1, 1])
    with opt_col1:
        st.info("💡 Click 'Start 3-Second Countdown' — a 3-2-1 timer runs before auto-capture!")
    with opt_col2:
        speed_on = st.toggle("⚡ Speed Challenge", value=st.session_state.practice_speed_challenge,
                             help="Race the clock! See how fast you can nail all your poses.")
        st.session_state.practice_speed_challenge = speed_on
    with opt_col3:
        brightness_boost = st.slider("☀️ Brightness", min_value=1.0, max_value=2.0,
                                     value=1.4, step=0.1, key="brightness_setting",
                                     help="Adjust camera brightness")

    # Show speed timer if active
    if speed_on and st.session_state.practice_speed_start is not None:
        elapsed = time.time() - st.session_state.practice_speed_start
        st.markdown(f"""
        <div style="text-align:center;margin:6px 0">
          <span class="speed-timer">⏱️ {elapsed:.1f}s</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Motivational quote ────────────────────────────────────────────────
    st.markdown(f'<div class="motive-quote">{_random_motivational_quote()}</div>',
                unsafe_allow_html=True)

    pose_model   = get_static_pose_model()
    session_data = []
    combo_count  = 0

    for pose_idx, pose_name in enumerate(selected):
        cfg = YOGA_POSES[pose_name]
        with st.expander(f"{cfg['emoji']} {cfg['english']}  •  {'⭐'*cfg['difficulty']}{'☆'*(3-cfg['difficulty'])}",
                         expanded=True):
            c_info, c_cam = st.columns([1, 2])

            with c_info:
                st.markdown(f"**{cfg['description']}**")
                st.markdown(f"🎯 Difficulty: {'⭐'*cfg['difficulty']}{'☆'*(3-cfg['difficulty'])}")
                st.markdown(f"⏱️ Hold target: **{cfg['hold_time']}s**")
                st.markdown("**💡 Tips:**")
                for tip in cfg["tips"][:3]:
                    st.markdown(f"- {tip}")

                # Show a mini motivational between poses (not on first)
                if pose_idx > 0:
                    st.markdown(f'<div class="motive-quote" style="margin-top:12px">{_random_motivational_quote()}</div>',
                                unsafe_allow_html=True)

            with c_cam:
                # Check if countdown is active for this pose
                countdown_key = f"countdown_{pose_name}"
                photo_key = f"photo_{pose_name}"
                
                if countdown_key not in st.session_state:
                    st.session_state[countdown_key] = None
                if photo_key not in st.session_state:
                    st.session_state[photo_key] = None
                
                # If no photo taken yet, show camera and take photo button
                if st.session_state[photo_key] is None:
                    # Live camera preview
                    camera_placeholder = st.empty()
                    
                    # Show countdown or normal camera
                    if st.session_state[countdown_key] is not None:
                        # Countdown active - show countdown overlay
                        remaining = 3 - (time.time() - st.session_state[countdown_key])
                        if remaining > 0:
                            # Show countdown
                            countdown_val = int(remaining) + 1
                            if countdown_val > 3:
                                countdown_val = 3
                                
                            # Create countdown display with enhanced camera background
                            try:
                                cap = get_camera()
                                if cap and cap.isOpened():
                                    ret, frame = cap.read()
                                    if ret:
                                        frame = cv2.flip(frame, 1)
                                        enhanced_frame = enhance_frame_brightness(frame, alpha=brightness_boost, beta=30)
                                        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                                        camera_placeholder.image(rgb_frame, 
                                                                caption=f"📷 {countdown_val} - Get ready!",
                                                                use_container_width=True)
                            except:
                                pass
                                
                            # Show countdown number prominently
                            countdown_colors = {3: "#4ECDC4", 2: "#f9d423", 1: "#ff4e50"}
                            cd_color = countdown_colors.get(countdown_val, "#ff4e50")
                            st.markdown(f"""
                            <div style="text-align:center;padding:20px 0;
                                        background:linear-gradient(135deg, {cd_color}dd, {cd_color}88);
                                        color:white;border-radius:16px;margin:10px 0;
                                        border:3px solid white;
                                        animation:pulse 1s infinite;">
                                <div style="font-size:56px;font-weight:900;text-shadow:0 0 20px rgba(0,0,0,0.3)">{countdown_val}</div>
                                <div style="font-size:18px;font-weight:600">Get ready to pose!</div>
                            </div>
                            <style>
                            @keyframes pulse {{
                                0% {{ transform: scale(1); }}
                                50% {{ transform: scale(1.05); }}
                                100% {{ transform: scale(1); }}
                            }}
                            </style>
                            """, unsafe_allow_html=True)
                            
                            time.sleep(0.3)
                            st.rerun()
                        else:
                            # Countdown finished - take photo
                            try:
                                cap = get_camera()
                                if cap and cap.isOpened():
                                    ret, frame = cap.read()
                                    if ret:
                                        frame = cv2.flip(frame, 1)
                                        enhanced_frame = enhance_frame_brightness(frame, alpha=brightness_boost, beta=30)
                                        st.session_state[photo_key] = enhanced_frame
                                        st.session_state[countdown_key] = None
                                        # Start speed timer on first capture
                                        if (speed_on and
                                                st.session_state.practice_speed_start is None):
                                            st.session_state.practice_speed_start = time.time()
                                        st.success("📸 Photo captured!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to capture photo. Please try again.")
                                        st.session_state[countdown_key] = None
                                        st.rerun()
                                else:
                                    st.error("Camera not available. Please check your camera permissions.")
                                    st.session_state[countdown_key] = None
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Camera error: {e}")
                                st.session_state[countdown_key] = None
                                st.rerun()
                    else:
                        # Show smooth live camera preview
                        try:
                            cap = get_camera()
                            if cap and cap.isOpened():
                                for _ in range(2):
                                    cap.read()
                                ret, frame = cap.read()
                                if ret:
                                    frame = cv2.flip(frame, 1)
                                    enhanced_frame = enhance_frame_brightness(frame, alpha=brightness_boost, beta=25)
                                    rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                                    camera_placeholder.image(rgb_frame, 
                                                           caption=f"📷 Live: Strike the {cfg['english']} pose",
                                                           use_container_width=True)
                                    time.sleep(0.15)
                                    st.rerun()
                                else:
                                    camera_placeholder.warning("⚠ Camera frame not available")
                                    time.sleep(1.0)
                                    st.rerun()
                            else:
                                camera_placeholder.warning("📹 Camera not accessible. Please check permissions.")
                                time.sleep(2.0)
                                st.rerun()
                        except Exception as e:
                            camera_placeholder.error(f"📹 Camera error: {e}")
                            time.sleep(2.0)
                            st.rerun()
                    
                    # Take photo button
                    st.markdown("### 📸 Ready to capture?")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(f"🎯 Start 3-Second Countdown", key=f"take_photo_{pose_name}", 
                                    type="primary", use_container_width=True):
                            st.session_state[countdown_key] = time.time()
                            st.rerun()
                        
                else:
                    # ── Photo taken — analyse & show animated reaction ────
                    frame_bgr = st.session_state[photo_key]
                    analysis  = analyze_frame(frame_bgr, pose_model, target_pose=pose_name)
                    annotated = annotate_frame(frame_bgr, analysis)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption=f"📸 {cfg['english']} — Analysis Complete",
                             use_container_width=True)

                    score = analysis.get("score", 0.0)
                    grade = analysis.get("grade", "F")

                    # 🎯 Animated score reaction
                    st.markdown(_score_reaction_html(score, grade),
                                unsafe_allow_html=True)

                    # 🔥 Combo tracking
                    adj_score = min(100.0, score + min(10.0, cfg["hold_time"] * 0.8))
                    if adj_score >= 70:
                        combo_count += 1
                        if combo_count >= 2:
                            st.markdown(f"""
                            <div style="text-align:center;margin:8px 0">
                              <span class="practice-combo combo-fire">
                                🔥 {combo_count}x Combo!
                              </span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        if combo_count >= 2:
                            st.markdown(f"""
                            <div style="text-align:center;margin:8px 0">
                              <span class="practice-combo combo-break">
                                💔 Combo Broken at {combo_count}x
                              </span>
                            </div>
                            """, unsafe_allow_html=True)
                        combo_count = 0

                    # Per-check breakdown
                    with st.expander("📐 Angle Breakdown", expanded=False):
                        for chk in analysis.get("checks", []):
                            lvl = chk.get("level", "good")
                            ico = {"good": "✅", "warning": "⚠️", "error": "❌"}.get(lvl, "•")
                            st.markdown(f"{ico} **{chk['name']}**: {chk['measured']:.0f}° "
                                        f"(ideal {chk['ideal']}°) — {chk['score']:.0f}pts")
                    
                    # Retake button
                    if st.button(f"🔄 Retake Photo", key=f"retake_{pose_name}", 
                                type="secondary", use_container_width=True):
                        st.session_state[photo_key] = None
                        st.session_state[countdown_key] = None
                        st.rerun()

                    # Store result
                    session_data.append({
                        "pose_name":      pose_name,
                        "english_name":   cfg["english"],
                        "emoji":          cfg["emoji"],
                        "score":          score,
                        "adjusted_score": adj_score,
                        "hold_time":      float(cfg["hold_time"]),
                        "hold_bonus":     min(10.0, cfg["hold_time"] * 0.8),
                        "stability":      85.0,
                        "grade":          grade,
                        "suggestions":    [chk["tip"] for chk in analysis.get("checks", [])
                                           if chk.get("level") in ("warning", "error")][:3],
                    })

    # ── Finish session ─────────────────────────────────────────────────────
    st.divider()
    n_captured = len(session_data)

    # Progress indicator
    progress_pct = n_captured / len(selected) if selected else 0
    st.progress(progress_pct,
                text=f"📸 Captured: {n_captured} / {len(selected)} poses")

    if combo_count >= 2:
        st.markdown(f"""
        <div style="text-align:center;margin:6px 0">
          <span class="practice-combo combo-fire">🔥 Current Combo: {combo_count}x</span>
        </div>
        """, unsafe_allow_html=True)

    if n_captured > 0 and st.button("🏁 Finish Session & See Results",
                                    type="primary", use_container_width=True):
        overall = np.mean([p["adjusted_score"] for p in session_data])
        done_set = {p["pose_name"] for p in session_data}
        surya_pct = len(done_set & SURYA_CORE_POSES) / len(SURYA_CORE_POSES)

        # Compute real combo (consecutive poses ≥70)
        max_combo = 0
        cur_combo = 0
        for p in session_data:
            if p["adjusted_score"] >= 70:
                cur_combo += 1
                max_combo = max(max_combo, cur_combo)
            else:
                cur_combo = 0

        all_tips = []
        for p in session_data:
            all_tips.extend(p.get("suggestions", []))

        # Speed challenge time
        speed_time = None
        if (st.session_state.practice_speed_challenge and
                st.session_state.practice_speed_start is not None):
            speed_time = time.time() - st.session_state.practice_speed_start

        st.session_state.practice_results = {
            "total_score":         round(float(overall), 1),
            "duration":            speed_time if speed_time else len(session_data) * 10.0,
            "poses_completed":     n_captured,
            "unique_poses":        len(done_set),
            "pose_details":        session_data,
            "suggestions":         list(dict.fromkeys(all_tips))[:5],
            "completed_poses_list": list(done_set),
            "surya_namaskar_pct":  surya_pct,
            "max_combo":           max_combo,
            "best_combo":          max_combo,
            "session_type":        "practice",
            "speed_challenge":     st.session_state.practice_speed_challenge,
            "speed_time":          speed_time,
        }
        # Reset speed timer for next session
        st.session_state.practice_speed_start = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Live Session (continuous webcam via cv2 + auto-rerun)
# ─────────────────────────────────────────────────────────────────────────────
def page_live():
    st.markdown("# 📹 Live Yoga Session")
    st.markdown("Start the webcam and perform any poses — the AI tracks everything in real-time.")
    st.divider()

    uid          = st.session_state.user_id
    live_results = st.session_state.live_results

    # ── Show results if session just finished ─────────────────────────────
    if live_results and not st.session_state.live_active:
        ge = GameEngine(uid)
        gr = ge.process_session(live_results)
        render_session_report(live_results, gr)
        if st.button("🔄 New Live Session"):
            st.session_state.live_results = None
            st.rerun()
        return

    # ── Controls ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        start_btn = st.button("▶️ Start Session", type="primary",
                              disabled=st.session_state.live_active,
                              use_container_width=True)
    with c2:
        stop_btn  = st.button("⏹️ Stop & Score", type="secondary",
                              disabled=not st.session_state.live_active,
                              use_container_width=True)
    with c3:
        target_options = ["Auto-Detect"] + [
            f"{cfg['emoji']} {cfg['english']}" for cfg in YOGA_POSES.values()
        ]
        live_target_display = st.selectbox(
            "Target Pose", target_options, disabled=st.session_state.live_active,
            key="live_target_sel",
        )
        # Map back to pose key
        if live_target_display == "Auto-Detect":
            st.session_state.live_target = "Auto-Detect"
        else:
            for k, cfg in YOGA_POSES.items():
                if cfg["english"] in live_target_display:
                    st.session_state.live_target = k
                    break

    # ── Start ─────────────────────────────────────────────────────────────
    if start_btn:
        get_camera()
        st.session_state.live_active  = True
        st.session_state.live_frames  = []
        st.session_state.live_start   = time.time()
        st.session_state.live_results = None
        st.session_state.live_cached_analysis = None
        st.session_state.live_last_analysis_ts = 0.0
        st.rerun()

    # ── Stop ──────────────────────────────────────────────────────────────
    if stop_btn and st.session_state.live_active:
        st.session_state.live_active = False
        st.session_state.live_last_frame = None  # clear buffer
        release_camera()
        frames = st.session_state.live_frames
        if frames:
            # Compile via SessionAnalyzer
            sa = SessionAnalyzer()
            sa.session_start = st.session_state.live_start or time.time()
            sa.completed_poses = []
            # Replay frames into analyser
            for f in frames:
                sa.add_frame(f["analysis"], f["ts"])
            results = sa.finish_session()
            results["session_type"] = "live"
            st.session_state.live_results = results
        st.rerun()

    # ── Live capture loop ─────────────────────────────────────────────────
    if st.session_state.live_active:
        # Frame buffer initialization
        if "live_last_frame" not in st.session_state:
            st.session_state.live_last_frame = None
        
        frame_ph  = st.empty()
        status_ph = st.empty()
        
        # Display buffered frame immediately to prevent flicker
        if st.session_state.live_last_frame is not None:
            frame_ph.image(st.session_state.live_last_frame, use_container_width=True)

        try:
            cap = get_camera()
            if not cap or not cap.isOpened():
                st.error("❌ Cannot access webcam. Check permissions.")
                st.session_state.live_active = False
                st.session_state.live_last_frame = None
                st.rerun()

            ret, frame = cap.read()
            if not ret:
                for _ in range(2):
                    ret, frame = cap.read()
                    if ret:
                        break
                    time.sleep(0.02)

            if ret:
                target = (None if st.session_state.live_target == "Auto-Detect"
                          else st.session_state.live_target)
                analysis = st.session_state.get("live_cached_analysis")
                if analysis is None or should_analyze_now("live", 0.20):
                    pose_model = get_pose_model()
                    analysis = analyze_frame(frame, pose_model, target)
                    st.session_state.live_cached_analysis = analysis
                annotated  = annotate_frame(frame, analysis)

                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Store in buffer FIRST, then display
                st.session_state.live_last_frame = rgb_frame
                frame_ph.image(rgb_frame, use_container_width=True)

                ts = time.time() - (st.session_state.live_start or time.time())
                st.session_state.live_frames.append({
                    "ts":       round(ts, 2),
                    "analysis": {k: v for k, v in analysis.items() if k not in ("results", "landmarks")},
                })

                elapsed = int(ts)
                n_frames = len(st.session_state.live_frames)
                pname    = analysis.get("pose_name", "…")
                pscore   = analysis.get("score", 0.0)
                status_ph.info(
                    f"⏱️ **{elapsed}s** &nbsp;|&nbsp; 📷 **{n_frames}** frames "
                    f"&nbsp;|&nbsp; 🧘 **{pname}** &nbsp;|&nbsp; 📊 **{pscore:.0f}/100**"
                )

        except Exception as e:
            # Keep showing last frame and retry during transient camera errors.
            if st.session_state.live_last_frame is None:
                st.warning(f"Webcam issue: {e} - retrying...")

        time.sleep(0.15)   # ~6-7 FPS, slightly faster for smoother experience
        st.rerun()

    elif not st.session_state.live_active and not live_results:
        st.markdown("""
        <div class="yoga-card">
          <h3 style="color:#f9d423">How to use Live Session</h3>
          <ol style="color:#c8baff;line-height:2">
            <li>Choose a target pose or leave on Auto-Detect</li>
            <li>Click <strong>▶️ Start Session</strong></li>
            <li>Step back so your full body is visible in the webcam</li>
            <li>Perform your yoga poses — hold each for 5-10 seconds</li>
            <li>Click <strong>⏹️ Stop & Score</strong> when done</li>
          </ol>
          <p style="color:#888;font-size:12px">
            ⚠️ Live mode requires a connected webcam and runs locally.
            If you see a camera error, use <strong>Video Analysis</strong> instead.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Video Analysis
# ─────────────────────────────────────────────────────────────────────────────
def page_video():
    st.markdown("# 🎬 Video Analysis")
    st.markdown("Upload any yoga session video — get a fully annotated output + detailed report.")
    st.divider()

    uid          = st.session_state.user_id
    video_results = st.session_state.video_results
    video_out     = st.session_state.video_out_path

    # ── Show results ───────────────────────────────────────────────────────
    if video_results:
        # Annotated video download
        if video_out and os.path.exists(video_out):
            st.markdown("### 🎞️ Annotated Output Video")
            with open(video_out, "rb") as vf:
                st.download_button(
                    "⬇️ Download Annotated Video",
                    vf, file_name="yogaquest_annotated.mp4",
                    mime="video/mp4", use_container_width=True,
                )
            st.video(video_out)

        st.divider()
        ge = GameEngine(uid)
        gr = ge.process_session(video_results)
        render_session_report(video_results, gr)

        if st.button("🔄 Analyse Another Video"):
            st.session_state.video_results = None
            st.session_state.video_out_path = None
            st.rerun()
        return

    # ── Upload ─────────────────────────────────────────────────────────────
    col_up, col_opt = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload your yoga session video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
        )
    with col_opt:
        target_opt = st.selectbox(
            "Target Pose (optional)",
            ["Auto-Detect"] + [f"{v['emoji']} {v['english']}"
                               for v in YOGA_POSES.values()],
        )
        target_pose = None
        if target_opt != "Auto-Detect":
            for k, v in YOGA_POSES.items():
                if v["english"] in target_opt:
                    target_pose = k
                    break

    if uploaded and st.button("🚀 Analyse Video", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_path  = os.path.join(tmp_dir, "input.mp4")
            out_path = os.path.join(tmp_dir, "output.mp4")

            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            st.info("🔄 Processing video — this may take a minute …")
            progress_bar = st.progress(0.0, text="Analysing frames …")

            def _cb(p: float):
                progress_bar.progress(min(p, 1.0),
                                      text=f"Analysing frames … {int(p*100)}%")

            try:
                results = process_video(in_path, out_path, target_pose, _cb)
                progress_bar.progress(1.0, text="Done! ✅")

                # Copy output to a persistent location
                persistent_out = os.path.join(
                    tempfile.gettempdir(),
                    f"yogaquest_out_{int(time.time())}.mp4",
                )
                import shutil
                shutil.copy2(out_path, persistent_out)

                results["session_type"] = "video"
                st.session_state.video_results  = results
                st.session_state.video_out_path = persistent_out
                st.rerun()

            except Exception as e:
                st.error(f"Processing failed: {e}")

    if not uploaded:
        st.markdown("""
        <div class="yoga-card">
          <h3 style="color:#f9d423">📋 What this mode does</h3>
          <ul style="color:#c8baff;line-height:2">
            <li>Detects all yoga poses frame-by-frame using MediaPipe</li>
            <li>Scores pose accuracy, alignment, and hold duration</li>
            <li>Generates a fully annotated video with skeleton overlay, scores, and feedback</li>
            <li>Produces a comprehensive session report with XP, achievements &amp; suggestions</li>
            <li>Supports MP4, AVI, MOV, MKV, WebM</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Surya Namaskar Guide
# ─────────────────────────────────────────────────────────────────────────────
def page_surya():
    st.markdown("# ☀️ Surya Namaskar")
    st.markdown("*Sun Salutation — 12-step guided sequence*")
    st.divider()

    uid = st.session_state.user_id

    st.markdown("""
    <div class="yoga-card">
      <p style="color:#c8baff;line-height:1.8">
        Surya Namaskar is a flowing sequence of 12 yoga poses performed in harmony with the breath.
        It builds strength, flexibility, and mindfulness — a complete practice in itself.
        Follow the steps below, then capture your key poses in Practice Mode to earn
        the <strong>☀️ Sun Salutation</strong> achievement.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sequence visualiser ────────────────────────────────────────────────
    st.markdown("### 🔢 The 12-Step Sequence")
    for step in SURYA_NAMASKAR_SEQUENCE:
        key_cfg = YOGA_POSES.get(step["key_pose"], {})
        with st.expander(
            f"**Step {step['step']}** — {step['emoji']} {step['english']}", expanded=False
        ):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Sanskrit:** *{step['name']}*")
                st.markdown(f"**Action:** {step['description']}")
                if key_cfg:
                    st.markdown(f"**Key tips:**")
                    for tip in key_cfg.get("tips", [])[:2]:
                        st.markdown(f"- {tip}")
            with c2:
                st.markdown(f"""
                <div style="text-align:center;padding:16px">
                  <div style="font-size:56px">{step['emoji']}</div>
                  <div style="color:#f9d423;font-weight:700">~{step['duration']}s</div>
                  <div style="color:#888;font-size:12px">hold / transition</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── Timeline visualisation ─────────────────────────────────────────────
    st.markdown("### 📈 Sequence Timeline")
    seq_df = pd.DataFrame([
        {"Step": f"S{s['step']} {s['emoji']}", "Duration (s)": s["duration"],
         "Pose": s["english"]}
        for s in SURYA_NAMASKAR_SEQUENCE
    ])
    fig = px.bar(
        seq_df, x="Step", y="Duration (s)", hover_data=["Pose"],
        color="Duration (s)", color_continuous_scale="Viridis",
        title="Time per Step",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", height=260,
        margin=dict(t=30,b=10,l=10,r=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🎯 Practice the Key Poses")
    st.markdown("Hit these four poses in Practice Mode to unlock the ☀️ Sun Salutation achievement:")

    cols = st.columns(4)
    for i, pname in enumerate(sorted(SURYA_CORE_POSES)):
        cfg = YOGA_POSES[pname]
        with cols[i]:
            st.markdown(f"""
            <div class="yoga-card" style="text-align:center">
              <div style="font-size:36px">{cfg['emoji']}</div>
              <div style="font-weight:700;color:#f9d423;font-size:13px">{cfg['english']}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("☀️ Start Surya Namaskar Practice", type="primary", use_container_width=True):
        st.session_state.practice_poses = list(SURYA_CORE_POSES)
        st.session_state.practice_results = None
        st.session_state.page = "practice"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
def page_leaderboard():
    st.markdown("# 🏆 Leaderboard")
    st.markdown("*Top YogaQuest players ranked by total XP*")
    st.divider()

    rows = db.get_leaderboard(limit=15)
    if not rows:
        st.info("No players yet — start a session to appear on the leaderboard!")
        return

    medal = {0: "🥇", 1: "🥈", 2: "🥉"}
    row_cls = {0: "gold", 1: "silver", 2: "bronze"}
    current = st.session_state.username

    for i, (username, xp, level, streak, sessions, perfect, total_poses) in enumerate(rows):
        icon    = LEVEL_ICONS.get(level, "🧘")
        lname   = LEVEL_NAMES.get(level, "Yogi")
        cls     = row_cls.get(i, "")
        is_me   = (username == current)
        prefix  = medal.get(i, f"#{i+1}")
        bold_me = "font-weight:900;color:#f9d423" if is_me else "color:#e0d7ff"

        accuracy = f"{(perfect/total_poses*100):.0f}%" if total_poses else "—"

        st.markdown(f"""
        <div class="lb-row {cls}">
          <span style="font-size:22px;width:40px">{prefix}</span>
          <span style="font-size:20px">{icon}</span>
          <span style="{bold_me};font-size:16px;flex:1">{username}{'  👈 you' if is_me else ''}</span>
          <span style="color:#f9d423;font-weight:700">⚡ {xp:,} XP</span>
          <span style="color:#c8baff">Lv {level}</span>
          <span style="color:#ff6b6b">🔥 {streak}d</span>
          <span style="color:#4ECDC4">📅 {sessions} sessions</span>
          <span style="color:#2ECC71">⭐ {accuracy} accurate</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── XP distribution chart ─────────────────────────────────────────────
    if len(rows) >= 2:
        df = pd.DataFrame(rows, columns=[
            "Username","XP","Level","Streak","Sessions","PerfectPoses","TotalPoses"
        ])
        fig = px.bar(
            df.head(10), x="Username", y="XP",
            color="Level", color_continuous_scale="Viridis",
            title="Top 10 — XP Comparison",
            labels={"XP": "Total XP"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", height=300,
            margin=dict(t=30,b=10,l=10,r=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Profile
# ─────────────────────────────────────────────────────────────────────────────
def page_profile():
    st.markdown(f"# 👤 {st.session_state.username}'s Profile")
    st.divider()

    uid   = st.session_state.user_id
    stats = db.get_user_stats(uid)
    if not stats:
        st.info("No stats yet — complete a session first!")
        return

    level   = stats["level"]
    xp      = stats["total_xp"]
    xp_in, xp_need, xp_pct = level_progress(xp, level)
    lname   = LEVEL_NAMES.get(level, "Yogi")
    licon   = LEVEL_ICONS.get(level, "🧘")

    # ── Hero card ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="yoga-card" style="text-align:center">
      <div style="font-size:64px">{licon}</div>
      <div style="font-size:28px;font-weight:900;color:#f9d423">{st.session_state.username}</div>
      <div style="color:#a090cc;font-size:16px">{lname} — Level {level}</div>
      <div style="margin-top:10px" class="xp-badge">⚡ {xp:,} Total XP</div>
    </div>
    """, unsafe_allow_html=True)

    if xp_need:
        st.progress(xp_pct, text=f"{xp_in:,} / {xp_need:,} XP to Level {level+1}")

    # ── Stats grid ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📅 Sessions",      stats["total_sessions"])
    c2.metric("🧘 Total Poses",   stats["total_poses"])
    c3.metric("⭐ Perfect Poses", stats["perfect_poses"])
    c4.metric("🔥 Best Streak",   f"{stats['best_streak']} days")
    c5.metric("🔥 Current Streak",f"{stats['streak_days']} days")

    st.divider()

    # ── Achievements ───────────────────────────────────────────────────────
    st.markdown("### 🏅 Achievements")
    unlocked = set(stats.get("achievements", []))
    all_ach  = list(ACHIEVEMENTS.items())

    un_list  = [(k, v) for k, v in all_ach if k in unlocked]
    lock_list= [(k, v) for k, v in all_ach if k not in unlocked]

    st.markdown(f"**Unlocked:** {len(un_list)} / {len(all_ach)}")
    st.progress(len(un_list) / len(all_ach) if all_ach else 0)

    if un_list:
        st.markdown("#### ✅ Unlocked")
        cols = st.columns(4)
        for i, (kid, ach) in enumerate(un_list):
            with cols[i % 4]:
                bonus_tag = (f'<div class="xp-badge">+{ach["xp_bonus"]} XP</div>'
                             if ach.get("xp_bonus") else "")
                st.markdown(f"""
                <div class="ach-card">
                  <div style="font-size:28px">{ach['icon']}</div>
                  <div style="font-weight:700;color:#f9d423;font-size:12px">{ach['name']}</div>
                  <div style="color:#aaa;font-size:10px;line-height:1.3">{ach['description']}</div>
                  {bonus_tag}
                </div>
                """, unsafe_allow_html=True)

    if lock_list:
        with st.expander(f"🔒 Locked ({len(lock_list)} remaining)"):
            cols = st.columns(4)
            for i, (kid, ach) in enumerate(lock_list):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="ach-card" style="opacity:0.45;filter:grayscale(80%)">
                      <div style="font-size:28px">{ach['icon']}</div>
                      <div style="font-weight:700;color:#888;font-size:12px">{ach['name']}</div>
                      <div style="color:#666;font-size:10px">{ach['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()

    # ── Session history ────────────────────────────────────────────────────
    st.markdown("### 📜 Session History")
    history = db.get_session_history(uid, limit=20)
    if not history:
        st.info("No sessions yet!")
        return

    rows = []
    scores_over_time = []
    for sid, sdate, stype, tscore, xp_e, dur, poses_json in history:
        rows.append({
            "Date":     sdate[:16].replace("T", " "),
            "Type":     stype.title(),
            "Score":    f"{tscore:.0f}",
            "Grade":    get_grade(tscore),
            "XP":       f"+{xp_e}",
            "Duration": f"{dur:.0f}s",
        })
        scores_over_time.append({"Date": sdate[:10], "Score": tscore})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Score trend
    if len(scores_over_time) > 1:
        df_trend = pd.DataFrame(scores_over_time).sort_values("Date")
        fig = px.line(
            df_trend, x="Date", y="Score",
            title="Score Trend Over Time",
            markers=True,
            color_discrete_sequence=["#f9d423"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", height=260,
            yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(t=30,b=10,l=10,r=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Pose Wall Game
# ─────────────────────────────────────────────────────────────────────────────
def page_wall_game():
    st.markdown("# 🧱 Dynamic Pose Wall Challenge")
    st.markdown("*Match the yoga pose before the wall reaches you!*")
    st.divider()

    uid = st.session_state.user_id

    # ── Show results if game just finished ────────────────────────────────
    wall_results = st.session_state.wall_game_results
    if wall_results and not st.session_state.live_active:
        results, game_report = wall_results
        render_wall_results(results, game_report)
        if st.button("🔄 Play Again", type="primary"):
            st.session_state.wall_game_results = None
            st.session_state.wall_game_engine  = None
            st.rerun()
        return

    # ── Config panel (when not playing) ───────────────────────────────────
    if not st.session_state.live_active:
        # Pre-warm camera in background while user reviews settings
        # This reduces the perceived "camera opening" delay when START is clicked
        if "camera_prewarmed" not in st.session_state:
            st.session_state.camera_prewarmed = False
        
        if not st.session_state.camera_prewarmed:
            with st.spinner("📷 Preparing camera..."):
                if prewarm_camera():
                    st.session_state.camera_prewarmed = True
                    st.success("✅ Camera ready!", icon="✅")
                    time.sleep(0.5)  # Brief feedback
                    st.rerun()
        
        stats = db.get_user_stats(uid)
        user_level = stats["level"] if stats else 1

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### ⚙️ Game Settings")
            diff_options = []
            for key, preset in DIFFICULTY_PRESETS.items():
                unlocked = user_level >= preset["unlock_level"]
                if unlocked:
                    diff_options.append(f"{preset['label']}")
                else:
                    diff_options.append(f"{preset['label']} 🔒 (Level {preset['unlock_level']})")

            diff_choice = st.selectbox(
                "Difficulty",
                diff_options,
                index=0,
                key="wall_diff_sel",
            )
            # Map back to difficulty key
            diff_key = "easy"
            for key, preset in DIFFICULTY_PRESETS.items():
                if preset["label"] in diff_choice:
                    diff_key = key
                    break

            preset = DIFFICULTY_PRESETS[diff_key]
            is_locked = user_level < preset["unlock_level"]

            rounds = st.slider(
                "Rounds", 5, 20, preset["total_rounds"],
                key="wall_rounds_slider",
                disabled=is_locked,
            )

        with col2:
            st.markdown("### 📋 Difficulty Details")
            st.markdown(f"""
            <div class="yoga-card">
              <div style="font-size:20px;font-weight:700;color:#f9d423">{preset['label']}</div>
              <div style="color:#c8baff;margin-top:8px;line-height:1.8;font-size:13px">
                {preset['description']}<br/>
                ⏱️ Prep time: {preset['prep_time_s']}s &nbsp;|&nbsp;
                🏃 Wall speed: {preset['travel_time_s']}s<br/>
                ❤️ Lives: {preset['lives']} &nbsp;|&nbsp;
                🎯 Collision zone: {preset['collision_threshold_px']}px<br/>
                ⚡ XP multiplier: {preset['xp_multiplier']}x
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        if is_locked:
            st.warning(f"🔒 {preset['label']} difficulty requires Level {preset['unlock_level']}. "
                       f"You are Level {user_level}.")
        else:
            if st.button("🎮 START WALL GAME", type="primary", use_container_width=True):
                # Camera is already pre-warmed, just flush any stale frames
                _cap = get_camera()
                if _cap.isOpened():
                    for _ in range(2):  # Quick flush (camera already warmed)
                        _cap.read()
                
                # Initialize game engine + motion analyzer
                st.session_state.wall_game_engine = WallGameEngine(
                    difficulty=diff_key,
                    frame_w=640,
                    frame_h=480,
                    total_rounds=rounds,
                )
                st.session_state.wall_game_engine.start_game()
                st.session_state.motion_analyzer = MotionAnalyzer(
                    window_size=20, frame_w=640, frame_h=480
                )
                st.session_state.live_active = True
                st.session_state.wall_game_results = None
                # Initialize round tracking state
                st.session_state.wall_current_round = 0
                st.session_state.wall_last_verdict = None
                st.session_state.wall_last_round_score = 0
                st.session_state.wall_last_xp = 0
                st.session_state.wall_round_history = []
                st.session_state.wall_cached_analysis = None
                st.session_state.wall_last_analysis_ts = 0.0
                st.rerun()

        # ── How to play ───────────────────────────────────────────────────
        st.markdown("""
        <div class="yoga-card">
          <h3 style="color:#f9d423">How to Play</h3>
          <ol style="color:#c8baff;line-height:2">
            <li>A silhouette wall slides toward you from the right side</li>
            <li>The wall has a yoga pose cut-out — match the pose with your body!</li>
            <li>You must fit through the cut-out before the wall reaches you</li>
            <li>Colliding limbs are highlighted in red, clear limbs in green</li>
            <li>Score points, build combos, and survive all rounds to win!</li>
          </ol>
          <p style="color:#888;font-size:12px">
            ⚠️ Requires a connected webcam. Stand back so your full body is visible.
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Game loop (live_active = True) ────────────────────────────────────
    # Wrapped in @st.fragment so that per-frame reruns only re-render the
    # game feed — NOT the entire page (sidebar, title, etc.).  This keeps
    # the camera window stable and open across round transitions.

    wge = st.session_state.wall_game_engine
    if wge is None:
        st.session_state.live_active = False
        st.rerun()
        return

    @st.fragment
    def _wall_game_fragment():
        """Self-contained game loop fragment — only this block re-renders
        on each frame tick, so the camera feed never flickers."""
        _uid = st.session_state.user_id

        frame_ph  = st.empty()
        status_ph = st.empty()
        stop_col1, _ = st.columns([1, 4])
        with stop_col1:
            stop_btn = st.button("⏹ Exit Game", type="secondary",
                                 key="wall_exit_btn")

        if stop_btn:
            st.session_state.live_active = False
            release_camera()
            st.session_state.wall_game_engine = None
            st.session_state.wall_game_results = None
            st.session_state.wall_last_frame = None  # clear buffer
            # Clear round tracking state
            st.session_state.wall_current_round = 0
            st.session_state.wall_last_verdict = None
            st.session_state.wall_last_round_score = 0
            st.session_state.wall_last_xp = 0
            st.session_state.wall_round_history = []
            st.rerun(scope="app")        # full-page rerun to leave game
            return

        _wge = st.session_state.wall_game_engine
        if _wge is None or not st.session_state.live_active:
            return

        # ── Frame buffer: display last good frame immediately ─────────────
        # This prevents flicker during state transitions by always showing
        # the most recent frame while the new one is being processed
        if "wall_last_frame" not in st.session_state:
            st.session_state.wall_last_frame = None

        if st.session_state.wall_last_frame is not None:
            frame_ph.image(
                st.session_state.wall_last_frame,
                use_container_width=True,
            )

        try:
            cap = get_camera()
            
            # Optimized retry - only if truly needed
            if not cap or not cap.isOpened():
                retry_count = 0
                while retry_count < 2:  # Reduced from 3 to 2
                    time.sleep(0.05)  # Reduced from 0.1
                    cap = get_camera()
                    if cap and cap.isOpened():
                        break
                    retry_count += 1

            if not cap or not cap.isOpened():
                # Only show warning if no buffer exists
                if st.session_state.wall_last_frame is None:
                    frame_ph.warning("⚠️ Camera temporarily unavailable")
                time.sleep(0.02)  # Reduced delay
                st.rerun()
                return

            ret, frame = cap.read()
            # Quick retry for first read failures
            if not ret:
                for _ in range(3):  # Reduced from 5
                    ret, frame = cap.read()
                    if ret:
                        break
                    time.sleep(0.02)  # Reduced from 0.05

            if not ret:
                # Keep showing last frame
                if st.session_state.wall_last_frame is None:
                    frame_ph.warning("⚠️ Camera frame dropped")
                time.sleep(0.02)
                st.rerun()
                return

            # ── Process the frame ─────────────────────────────────────────
            analysis = st.session_state.get("wall_cached_analysis")
            if analysis is None or should_analyze_now("wall", 0.14):
                pose_model = get_pose_model()
                analysis = analyze_frame(frame, pose_model,
                                         target_pose=_wge.current_pose)
                st.session_state.wall_cached_analysis = analysis

            ma = st.session_state.motion_analyzer
            motion_metrics = {}
            if ma is not None:
                motion_metrics = ma.push(analysis, time.time())

            render_data = _wge.tick(analysis, motion_metrics, delta_t=0.05)

            # ── Track round transitions in session state ──────────────────
            current_round = render_data.get("round", 0)
            current_state = render_data.get("state", "")
            
            # Initialize round tracking
            if "wall_current_round" not in st.session_state:
                st.session_state.wall_current_round = 0
                st.session_state.wall_last_verdict = None
                st.session_state.wall_last_round_score = 0
                st.session_state.wall_last_xp = 0
                st.session_state.wall_round_history = []
            
            # Detect round transition (RESULT → PREPARE or new round started)
            if current_round > st.session_state.wall_current_round:
                # Store completed round data
                st.session_state.wall_last_verdict = render_data.get("verdict", "")
                st.session_state.wall_last_round_score = render_data.get("round_score", 0)
                st.session_state.wall_last_xp = render_data.get("xp_this_round", 0)
                
                # Add to history
                round_info = {
                    "round": st.session_state.wall_current_round + 1,
                    "pose": _wge.current_pose or "unknown",
                    "verdict": st.session_state.wall_last_verdict,
                    "score": st.session_state.wall_last_round_score,
                    "xp": st.session_state.wall_last_xp,
                    "lives_remaining": render_data.get("lives", 0),
                    "combo": render_data.get("combo", 0),
                }
                st.session_state.wall_round_history.append(round_info)
                
                # Update current round tracker
                st.session_state.wall_current_round = current_round

            # ── Render & display ──────────────────────────────────────────
            annotated = render_wall_frame(frame, render_data, analysis)
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Store in buffer FIRST, then display
            st.session_state.wall_last_frame = rgb_frame
            frame_ph.image(rgb_frame, use_container_width=True)

            state_label = (render_data.get("state", "")
                           .replace("_", " ").upper())
            
            # Enhanced status display with round transition feedback
            status_msg = (
                f"🎮 **{state_label}** &nbsp;|&nbsp; "
                f"🏆 **{render_data.get('score', 0):,}** pts &nbsp;|&nbsp; "
                f"❤️ **{render_data.get('lives', 0)}** lives &nbsp;|&nbsp; "
                f"Round **{min(render_data.get('round', 0) + 1, render_data.get('total_rounds', 0))}/"
                f"{render_data.get('total_rounds', 0)}**"
            )
            
            # Show last round result during PREPARE state
            if current_state == "prepare" and st.session_state.wall_last_verdict:
                verdict = st.session_state.wall_last_verdict
                xp = st.session_state.wall_last_xp
                status_msg += f" &nbsp;|&nbsp; Last: **{verdict}** (+{xp} pts)"
            
            # Use a container with proper spacing to prevent overlap
            with st.container():
                st.markdown("---")  # Add separator
                status_ph.info(status_msg)

            # ── Game-end check ────────────────────────────────────────────
            if _wge.state in (WallGameState.GAME_OVER, WallGameState.VICTORY):
                st.session_state.live_active = False
                results = _wge.get_final_results()

                if ma is not None:
                    mm = ma.get_current_metrics()
                    results["max_stability"] = mm.get("max_stability", 0.0)
                    results["min_transition_ms"] = mm.get(
                        "min_transition_ms", 9999)
                    results["smooth_transitions_85plus"] = mm.get(
                        "smooth_transitions_85plus", 0)

                ge = GameEngine(_uid)
                game_report = ge.process_session(results)
                st.session_state.wall_game_results = (results, game_report)
                st.session_state.wall_last_frame = None  # clear buffer
                # Clear round tracking state on game end
                st.session_state.wall_current_round = 0
                st.session_state.wall_last_verdict = None
                st.session_state.wall_last_round_score = 0
                st.session_state.wall_last_xp = 0
                st.session_state.wall_round_history = []
                st.rerun(scope="app")    # full-page rerun → show results
                return

            # ── Smooth continuous loop without sleep ──────────────────────
            # Immediately trigger next frame to maintain 60fps feel
            time.sleep(0.016)  # ~60fps, reduced from 0.04 for smoother transitions
            st.rerun()  # fragment-scoped rerun

        except Exception as e:
            # Keep showing last frame during errors
            if st.session_state.wall_last_frame is None:
                frame_ph.warning(f"⚠️ Camera issue: {e} — retrying…")
            time.sleep(0.04)
            st.rerun()                   # fragment-scoped rerun
            return

        time.sleep(0.04)                 # ~25 FPS cap
        st.rerun()                       # fragment-scoped rerun (camera stays)

    # Invoke the fragment
    _wall_game_fragment()



# ─────────────────────────────────────────────────────────────────────────────
# Page: Pose Target Game  🎯
# ─────────────────────────────────────────────────────────────────────────────

_PTG_TARGET_RADIUS = 45
_PTG_HIT_THRESHOLD = 65

_PTG_POSES = {
    "Tree Pose": [
        {"body_part": 27, "name": "LEFT FOOT",  "x": 0.42, "y": 0.58},
        {"body_part": 15, "name": "LEFT HAND",  "x": 0.42, "y": 0.18},
        {"body_part": 16, "name": "RIGHT HAND", "x": 0.58, "y": 0.18},
    ],
    "Mountain Pose": [
        {"body_part": 27, "name": "LEFT FOOT",  "x": 0.46, "y": 0.88},
        {"body_part": 28, "name": "RIGHT FOOT", "x": 0.54, "y": 0.88},
        {"body_part": 15, "name": "LEFT HAND",  "x": 0.42, "y": 0.60},
        {"body_part": 16, "name": "RIGHT HAND", "x": 0.58, "y": 0.60},
    ],
    "Warrior I": [
        {"body_part": 27, "name": "FRONT FOOT", "x": 0.35, "y": 0.82},
        {"body_part": 28, "name": "BACK FOOT",  "x": 0.65, "y": 0.82},
        {"body_part": 15, "name": "LEFT HAND",  "x": 0.48, "y": 0.18},
        {"body_part": 16, "name": "RIGHT HAND", "x": 0.52, "y": 0.18},
    ],
    "Warrior II": [
        {"body_part": 27, "name": "LEFT FOOT",  "x": 0.25, "y": 0.84},
        {"body_part": 28, "name": "RIGHT FOOT", "x": 0.75, "y": 0.84},
        {"body_part": 15, "name": "LEFT HAND",  "x": 0.08, "y": 0.45},
        {"body_part": 16, "name": "RIGHT HAND", "x": 0.92, "y": 0.45},
    ],
    "Triangle Pose": [
        {"body_part": 27, "name": "LEFT FOOT",  "x": 0.25, "y": 0.84},
        {"body_part": 28, "name": "RIGHT FOOT", "x": 0.75, "y": 0.84},
        {"body_part": 15, "name": "LOWER HAND", "x": 0.32, "y": 0.70},
        {"body_part": 16, "name": "UPPER HAND", "x": 0.68, "y": 0.18},
    ],
    "Cobra Pose": [
        {"body_part": 15, "name": "LEFT HAND",  "x": 0.42, "y": 0.60},
        {"body_part": 16, "name": "RIGHT HAND", "x": 0.58, "y": 0.60},
    ],
    "Downward Dog": [
        {"body_part": 15, "name": "LEFT HAND",  "x": 0.35, "y": 0.30},
        {"body_part": 16, "name": "RIGHT HAND", "x": 0.65, "y": 0.30},
        {"body_part": 27, "name": "LEFT FOOT",  "x": 0.40, "y": 0.85},
        {"body_part": 28, "name": "RIGHT FOOT", "x": 0.60, "y": 0.85},
    ],
}


def _ptg_draw_overlay(frame, step, total_steps, pose_name, completed, hit_px=None):
    """
    Renders the target circle, instruction text, and completion banner
    directly onto *frame* (BGR, in-place).
    Returns the annotated frame.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Header strip
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    cv2.putText(out, pose_name, (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Step {min(step+1, total_steps)} / {total_steps}",
                (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)

    if not completed:
        # Active target ring — drawn further below by caller
        pass
    else:
        # Victory banner
        banner = out.copy()
        cv2.rectangle(banner, (0, h//2 - 60), (w, h//2 + 70), (10, 40, 10), -1)
        cv2.addWeighted(banner, 0.75, out, 0.25, 0, out)
        cv2.putText(out, "POSE COMPLETE!", (w//2 - 185, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 80), 3, cv2.LINE_AA)
        cv2.putText(out, "Next pose in 2s ...", (w//2 - 145, h//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    # Hit-flash overlay
    if hit_px:
        cv2.circle(out, hit_px, _PTG_TARGET_RADIUS + 10, (0, 255, 100), 4, cv2.LINE_AA)

    return out


def _ptg_check_hit(landmarks, target, w, h):
    """Return True + pixel coords if the target body part is within hit threshold."""
    idx = target["body_part"]
    if landmarks is None or idx >= len(landmarks):
        return False, None
    lm = landmarks[idx]
    if lm[3] < 0.25:
        return False, None
    jx = int(lm[0] * w)
    jy = int(lm[1] * h)
    tx = int(target["x"] * w)
    ty = int(target["y"] * h)
    dist = math.sqrt((jx - tx)**2 + (jy - ty)**2)
    return dist < _PTG_HIT_THRESHOLD, (tx, ty)


def page_pose_target_game():
    """
    Pose Target Game — guide each body part into glowing targets,
    one step at a time, across multiple rounds.
    """
    st.markdown("# 🎯 Pose Target Game")
    st.markdown(
        "*Hit the glowing targets by moving your hands and feet "
        "to the correct positions for each yoga pose.*"
    )
    st.divider()

    uid = st.session_state.user_id

    # ── Show results after game ends ───────────────────────────────────────
    if st.session_state.ptg_results and not st.session_state.ptg_active:
        results = st.session_state.ptg_results
        ge = GameEngine(uid)
        gr = ge.process_session(results)
        st.balloons()
        st.markdown("# 🎊 Game Over — Well Done!")
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Poses Completed",  results["poses_completed"])
        c2.metric("⏱️ Duration",          f"{results['duration']:.0f}s")
        c3.metric("⚡ XP Earned",         f"+{gr['xp_earned']:,}")
        c4.metric("🔥 Streak",            f"{gr['streak_days']} days")
        st.divider()
        st.plotly_chart(score_gauge(results["total_score"]), use_container_width=True)
        if gr.get("new_achievements"):
            render_achievements(gr["new_achievements"])
        if st.button("🔄 Play Again", type="primary", use_container_width=True):
            st.session_state.ptg_results  = None
            st.session_state.ptg_active   = False
            st.session_state.ptg_score    = 0
            st.session_state.ptg_rounds_done = 0
            st.session_state.ptg_completed_at = None
            st.rerun()
        return

    # ── Config panel ───────────────────────────────────────────────────────
    if not st.session_state.ptg_active:
        # Pre-warm camera in background
        if "ptg_camera_prewarmed" not in st.session_state:
            st.session_state.ptg_camera_prewarmed = False
        
        if not st.session_state.ptg_camera_prewarmed:
            with st.spinner("📷 Preparing camera..."):
                if prewarm_camera():
                    st.session_state.ptg_camera_prewarmed = True
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚙️ Game Settings")
            pose_choice = st.selectbox(
                "Starting Pose",
                list(_PTG_POSES.keys()),
                key="ptg_pose_sel",
            )
            rounds = st.slider("Rounds (poses to complete)", 1, 10, 5, key="ptg_rounds_sl")

        with col2:
            steps = _PTG_POSES[pose_choice]
            st.markdown("### 📋 How to Play")
            st.markdown(f"""
            <div class="yoga-card">
              <ul style="color:#c8baff;line-height:2;margin:0;padding-left:18px">
                <li>A glowing target circle appears on screen</li>
                <li>Move the indicated body part into the circle</li>
                <li>All <strong>{len(steps)}</strong> targets must be hit to complete the pose</li>
                <li>Each completed pose earns XP and increases your score</li>
                <li>Complete all rounds as fast as you can!</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        if st.button("▶️ START GAME", type="primary", use_container_width=True):
            st.session_state.ptg_active     = True
            st.session_state.ptg_pose       = pose_choice
            st.session_state.ptg_step       = 0
            st.session_state.ptg_completed  = False
            st.session_state.ptg_completed_at = None
            st.session_state.ptg_score      = 0
            st.session_state.ptg_rounds_done= 0
            st.session_state.ptg_total_rounds = rounds
            st.session_state.ptg_start_time  = time.time()
            st.session_state.ptg_results     = None
            st.session_state.ptg_cached_analysis = None
            st.session_state.ptg_last_analysis_ts = 0.0
            st.rerun()

        st.markdown("""
        <div class="yoga-card">
          <h3 style="color:#f9d423">🏆 Scoring</h3>
          <ul style="color:#c8baff;line-height:2;margin:0;padding-left:18px">
            <li><strong>+50 XP</strong> per completed pose</li>
            <li><strong>+10 XP</strong> bonus for fast completion (&lt;15 s per pose)</li>
            <li>Score and XP saved to your profile</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Live game loop — wrapped in @st.fragment ──────────────────────
    # Only this fragment re-renders on each frame tick, keeping the
    # camera feed persistent across round transitions.

    @st.fragment
    def _ptg_game_fragment():
        """Self-contained game loop fragment — only this block re-renders
        on each frame tick, so the camera feed never flickers."""

        # ── Frame buffer initialization ───────────────────────────────────
        if "ptg_last_frame" not in st.session_state:
            st.session_state.ptg_last_frame = None

        # ── Live game loop ─────────────────────────────────────────────────────
        frame_ph  = st.empty()
        status_ph = st.empty()
        stop_col, _ = st.columns([1, 4])
        with stop_col:
            stop_btn = st.button("⏹ Exit Game", key="ptg_stop", type="secondary")

        if stop_btn:
            st.session_state.ptg_active = False
            st.session_state.ptg_last_frame = None  # clear buffer
            release_camera()
            st.rerun(scope="app")
            return

        # Display buffered frame immediately to prevent flicker
        if st.session_state.ptg_last_frame is not None:
            frame_ph.image(st.session_state.ptg_last_frame, use_container_width=True)

        # Fetch current game state from session
        pose_name  = st.session_state.ptg_pose
        cur_step   = st.session_state.ptg_step
        completed  = st.session_state.ptg_completed
        completed_at = st.session_state.get("ptg_completed_at")
        score      = st.session_state.ptg_score
        rounds_done= st.session_state.ptg_rounds_done
        total_rounds = st.session_state.ptg_total_rounds
        steps      = _PTG_POSES[pose_name]

        # ── Pose-completed cooldown handling ─────────────────────────────────
        if completed:
            now_completed = time.time()
            if completed_at is None:
                st.session_state.ptg_completed_at = now_completed
                completed_at = now_completed
            if now_completed - completed_at >= 2.0:
                rounds_done += 1
            st.session_state.ptg_rounds_done = rounds_done

            if now_completed - completed_at < 2.0:
                pass
            elif rounds_done >= total_rounds:
                # Game finished — build results and store
                elapsed = time.time() - (st.session_state.ptg_start_time or time.time())
                avg_score = min(100.0, score / max(1, rounds_done))
                pose_details = [{
                    "pose_name":      "PoseTargetGame",
                    "english_name":   "Pose Target Game",
                    "emoji":          "🎯",
                    "score":          avg_score,
                    "adjusted_score": avg_score,
                    "hold_time":      elapsed / max(1, rounds_done),
                    "hold_bonus":     5.0,
                    "stability":      80.0,
                    "grade":          get_grade(avg_score),
                    "suggestions":    [],
                }]
                st.session_state.ptg_results = {
                    "total_score":          round(avg_score, 1),
                    "duration":             round(elapsed, 1),
                    "poses_completed":      rounds_done,
                    "unique_poses":         1,
                    "pose_details":         pose_details,
                    "suggestions":          [],
                    "completed_poses_list": [],
                    "surya_namaskar_pct":   0.0,
                    "max_combo":            rounds_done,
                    "session_type":         "pose_target_game",
                }
                st.session_state.ptg_active = False
                st.session_state.ptg_completed_at = None
                # Keep the cached camera alive through the final transition too.
                # It is released only on explicit Exit Game, page navigation, or logout.
                st.rerun(scope="app")
                return
            else:
                # Advance to next pose (random)
                available = list(_PTG_POSES.keys())
                next_pose = random.choice(available)
                st.session_state.ptg_pose      = next_pose
                st.session_state.ptg_step      = 0
                st.session_state.ptg_completed = False
                st.session_state.ptg_completed_at = None
                st.rerun()
                return

        try:
            cap = get_camera()
        
            # Optimized retry
            if not cap or not cap.isOpened():
                retry_count = 0
                while retry_count < 2:
                    time.sleep(0.05)
                    cap = get_camera()
                    if cap and cap.isOpened():
                        break
                    retry_count += 1

            if not cap or not cap.isOpened():
                if st.session_state.ptg_last_frame is None:
                    frame_ph.warning("⚠️ Camera unavailable")
                time.sleep(0.05)
                st.rerun()
                return

            ret, frame = cap.read()
            if not ret:
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret:
                        break
                    time.sleep(0.02)

            if not ret:
                if st.session_state.ptg_last_frame is None:
                    frame_ph.warning("⚠️ Frame dropped")
                time.sleep(0.05)
                st.rerun()
                return

            if ret:
                frame = cv2.flip(frame, 1)
                # Enhance brightness for better visibility in Wall Game
                frame = enhance_frame_brightness(frame, alpha=1.3, beta=25)
                h, w  = frame.shape[:2]

                analysis = st.session_state.get("ptg_cached_analysis")
                if analysis is None or should_analyze_now("ptg", 0.16):
                    pose_model = get_pose_model()
                    analysis = analyze_frame(frame, pose_model)
                    st.session_state.ptg_cached_analysis = analysis
                landmarks  = analysis.get("landmarks")

                if landmarks is not None:
                    from pose_detector import draw_skeleton, PoseDetectionResult
                    res = analysis.get("results")
                    if isinstance(res, PoseDetectionResult) and res.landmarks is not None:
                        draw_skeleton(frame, res, (0, 220, 80), thickness=2)

                # Check current target
                hit_px = None
                if cur_step < len(steps):
                    target   = steps[cur_step]
                    tx = int(target["x"] * w)
                    ty = int(target["y"] * h)

                    # Draw target ring
                    cv2.circle(frame, (tx, ty), _PTG_TARGET_RADIUS,
                               (255, 230, 0), 3, cv2.LINE_AA)
                    # Animated inner ring
                    cv2.circle(frame, (tx, ty), _PTG_TARGET_RADIUS - 12,
                               (255, 180, 0), 1, cv2.LINE_AA)

                    # Instruction label
                    label_y = max(ty - _PTG_TARGET_RADIUS - 10, 20)
                    cv2.putText(frame, f"MOVE {target['name']} HERE",
                                (max(0, tx - 90), label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 0), 2, cv2.LINE_AA)

                    hit, hit_px = _ptg_check_hit(landmarks, target, w, h)
                    if hit:
                        score += 10
                        st.session_state.ptg_score = score
                        st.session_state.ptg_step  = cur_step + 1
                        if st.session_state.ptg_step >= len(steps):
                            score += 40          # pose completion bonus
                            st.session_state.ptg_score     = score
                            st.session_state.ptg_completed = True
                            st.session_state.ptg_completed_at = time.time()

                # Overlay HUD
                frame = _ptg_draw_overlay(
                    frame, cur_step, len(steps), pose_name,
                    st.session_state.ptg_completed, hit_px
                )

                # Score HUD
                cv2.putText(frame, f"Score: {score}",
                            (w - 160, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (249, 212, 35), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Round {rounds_done+1}/{total_rounds}",
                            (w - 200, 66), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (200, 200, 200), 1, cv2.LINE_AA)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Store in buffer FIRST, then display
                st.session_state.ptg_last_frame = rgb_frame
                frame_ph.image(rgb_frame, use_container_width=True)

                status_ph.info(
                    f"🎯 **{pose_name}** &nbsp;|&nbsp; "
                    f"Step **{min(cur_step+1, len(steps))}/{len(steps)}** &nbsp;|&nbsp; "
                    f"🏆 Score: **{score}** &nbsp;|&nbsp; "
                    f"Round **{rounds_done+1}/{total_rounds}**"
                )

        except Exception as e:
            # Keep showing last frame and retry during transient camera errors.
            if st.session_state.ptg_last_frame is None:
                frame_ph.warning(f"Camera issue: {e} - retrying...")
            time.sleep(0.05)
            st.rerun()
            return

        time.sleep(0.05)  # ~20 FPS, faster for better responsiveness
        st.rerun()

    # Invoke the fragment
    _ptg_game_fragment()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Falling Pose Game  🌊
# ─────────────────────────────────────────────────────────────────────────────

# Game constants
_FPG_IMAGE_SIZE   = 80
_FPG_HOLD_TIME    = 0.5    # seconds the pose must be held to score
_FPG_SPAWN_MIN    = 1.5
_FPG_SPAWN_MAX    = 3.5
_FPG_MAX_ACTIVE   = 6
_FPG_SPEED_SCALE  = 0.012  # how fast cards accelerate over time

# Each pose gets a unique vivid BGR color for its card
_FPG_POSE_INFO = {
    "Tadasana":             {"emoji": "M",  "english": "Mountain",      "points": 5,  "card_col": (200, 80, 30),   "glow": (255, 120, 50)},
    "Vrikshasana":          {"emoji": "T",  "english": "Tree Pose",     "points": 10, "card_col": (40, 180, 40),   "glow": (80, 255, 80)},
    "Virabhadrasana_I":     {"emoji": "W1", "english": "Warrior I",     "points": 15, "card_col": (30, 80, 220),   "glow": (60, 140, 255)},
    "Virabhadrasana_II":    {"emoji": "W2", "english": "Warrior II",    "points": 15, "card_col": (180, 30, 180),  "glow": (240, 60, 240)},
    "Trikonasana":          {"emoji": "Tr", "english": "Triangle",      "points": 20, "card_col": (0, 180, 220),   "glow": (0, 230, 255)},
    "Bhujangasana":         {"emoji": "Co", "english": "Cobra",         "points": 12, "card_col": (20, 160, 160),  "glow": (40, 210, 210)},
    "Adho_Mukha_Svanasana": {"emoji": "DD", "english": "Down Dog",      "points": 12, "card_col": (140, 40, 180),  "glow": (200, 70, 255)},
}

# Emoji dict for display (front-end labels)
_FPG_EMOJI_MAP = {
    "Tadasana": "🏔️", "Vrikshasana": "🌳", "Virabhadrasana_I": "⚔️",
    "Virabhadrasana_II": "🗡️", "Trikonasana": "🔺", "Bhujangasana": "🐍",
    "Adho_Mukha_Svanasana": "🐕",
}


def _fpg_draw_canvas(canvas, active_cards, current_hold_name, current_hold_start,
                     last_catch, last_catch_pts, score, remaining, duration,
                     combo, particles, now, elapsed, GAME_W, GAME_H):
    """Render the full game canvas with fancy animations."""

    # ── Starfield background ──────────────────────────────────────────────
    canvas[:] = (12, 8, 28)
    # Animated subtle grid
    grid_offset = int(now * 20) % 60
    for gx in range(-grid_offset, GAME_W, 60):
        cv2.line(canvas, (gx, 0), (gx, GAME_H), (28, 18, 50), 1)
    for gy in range(0, GAME_H, 60):
        cv2.line(canvas, (0, gy), (GAME_W, gy), (28, 18, 50), 1)

    # ── Particles ────────────────────────────────────────────────────────
    for p in particles:
        px, py, pr, pc, palpha = int(p[0]), int(p[1]), max(1, int(p[2])), p[3], p[4]
        if 0 <= px < GAME_W and 0 <= py < GAME_H and palpha > 10:
            # Draw glowing circle
            cv2.circle(canvas, (px, py), pr + 2, (int(pc[0]*0.3), int(pc[1]*0.3), int(pc[2]*0.3)), -1)
            cv2.circle(canvas, (px, py), pr, pc, -1)

    # ── Draw cards ───────────────────────────────────────────────────────
    for card in active_cards:
        x      = int(card["x"])
        y      = int(card["y"])
        info   = _FPG_POSE_INFO.get(card["pose_name"], {})
        english = info.get("english", card["pose_name"][:8])
        pts_val = info.get("points", 5)
        c_col   = info.get("card_col", (80, 40, 120))
        g_col   = info.get("glow",     (120, 80, 200))
        is_matched = (card["pose_name"] == current_hold_name)

        W = _FPG_IMAGE_SIZE + 30
        H = _FPG_IMAGE_SIZE + 28
        bx1 = max(0, x);      by1 = max(0, y)
        bx2 = min(GAME_W, x + W); by2 = min(GAME_H, y + H)
        if bx2 <= bx1 or by2 <= by1:
            continue

        # Glow outer ring
        glow_r = (40, 28, 60) if not is_matched else (
            int(g_col[0]*0.15), int(g_col[1]*0.15), int(g_col[2]*0.15))
        cv2.rectangle(canvas, (max(0, x-4), max(0, y-4)),
                      (min(GAME_W, x+W+4), min(GAME_H, y+H+4)), glow_r, -1)

        # Card body
        body_dark = (max(0, c_col[0]//4), max(0, c_col[1]//4), max(0, c_col[2]//4))
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), body_dark, -1)

        # Top colour stripe
        stripe_h = 12
        cv2.rectangle(canvas, (bx1, by1), (bx2, min(by2, by1+stripe_h)), c_col, -1)

        # Border — thicker + brighter when matched
        brd_col   = g_col if is_matched else c_col
        brd_thick = 3 if is_matched else 2
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), brd_col, brd_thick, cv2.LINE_AA)

        # Pose short label (big, centered)
        label = english[:10]
        lscale = 0.45 if len(label) > 8 else 0.50
        lsize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
        lx = bx1 + max(0, (bx2 - bx1 - lsize[0]) // 2)
        ly = by1 + stripe_h + 26
        cv2.putText(canvas, label, (lx, min(GAME_H-4, ly)),
                    cv2.FONT_HERSHEY_SIMPLEX, lscale, (230, 220, 255), 1, cv2.LINE_AA)

        # Points badge — bottom-right
        pts_str = f"+{pts_val}"
        pt_y = min(GAME_H - 6, by2 - 6)
        cv2.putText(canvas, pts_str, (max(0, bx2 - 34), pt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (249, 212, 35), 2, cv2.LINE_AA)

        # Hold-progress arc
        if is_matched and current_hold_start:
            pct    = min(1.0, (now - current_hold_start) / _FPG_HOLD_TIME)
            cx_c   = bx1 + (bx2 - bx1) // 2
            cy_c   = by1 + (by2 - by1) // 2
            radius = min((bx2-bx1), (by2-by1)) // 2 + 10
            # Track ring
            cv2.circle(canvas, (cx_c, cy_c), radius, (60, 50, 80), 5, cv2.LINE_AA)
            # Progress arc (yellow)
            cv2.ellipse(canvas, (cx_c, cy_c), (radius, radius),
                        -90, 0, int(360 * pct), (249, 212, 35), 5, cv2.LINE_AA)
            # Pulsing centre dot
            dot_r = max(3, int(6 * pct))
            cv2.circle(canvas, (cx_c, cy_c), dot_r, (249, 212, 35), -1)

    # ── CAUGHT! flash ────────────────────────────────────────────────────
    age = now - last_catch
    if age < 0.9:
        alpha = max(0.0, 1.0 - age / 0.9)
        caught_text = f"CAUGHT! +{last_catch_pts}"
        ts, _ = cv2.getTextSize(caught_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        tx = (GAME_W - ts[0]) // 2
        ty = GAME_H // 2 - 20
        # Shadow
        cv2.putText(canvas, caught_text, (tx+3, ty+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 80, 40), 3, cv2.LINE_AA)
        cv2.putText(canvas, caught_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 120), 3, cv2.LINE_AA)
        if combo > 1:
            combo_txt = f"x{combo} COMBO!"
            cs, _ = cv2.getTextSize(combo_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cx2 = (GAME_W - cs[0]) // 2
            cv2.putText(canvas, combo_txt, (cx2, ty + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2, cv2.LINE_AA)

    # ── Speed indicator bar (right edge) ─────────────────────────────────
    speed_pct  = min(1.0, elapsed / max(1, duration))
    speed_h    = int((GAME_H - 56) * speed_pct)
    cv2.rectangle(canvas, (GAME_W - 8, 50), (GAME_W, GAME_H), (30, 20, 50), -1)
    spd_col = (0, 180, 255) if speed_pct < 0.6 else (0, 80, 255) if speed_pct < 0.85 else (0, 30, 220)
    cv2.rectangle(canvas, (GAME_W - 8, GAME_H - speed_h), (GAME_W, GAME_H), spd_col, -1)

    # ── HUD bar (top) ────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (GAME_W, 50), (8, 6, 20), -1)
    cv2.line(canvas, (0, 50), (GAME_W, 50), (60, 40, 100), 1)
    # Score
    cv2.putText(canvas, f"Score: {score}", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (249, 212, 35), 2, cv2.LINE_AA)
    # Timer
    t_str = f"{remaining}s"
    ts2, _ = cv2.getTextSize(t_str, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 2)
    cv2.putText(canvas, t_str, (GAME_W - ts2[0] - 16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2, cv2.LINE_AA)
    # Combo badge in HUD
    if combo > 1:
        c_str = f"x{combo}"
        cv2.putText(canvas, c_str, (GAME_W//2 - 20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 180, 0), 2, cv2.LINE_AA)

    # ── Timer bar (bottom) ───────────────────────────────────────────────
    if remaining > 20:
        t_col = (0, 210, 80)
    elif remaining > 10:
        t_col = (0, 160, 255)
    else:
        # Urgent red pulsing
        pulse = int(abs(np.sin(now * 6)) * 80)
        t_col = (0, pulse, 220)
    cv2.rectangle(canvas, (0, GAME_H - 10), (GAME_W, GAME_H), (25, 20, 40), -1)
    timer_w = int(GAME_W * remaining / max(1, duration))
    cv2.rectangle(canvas, (0, GAME_H - 10), (timer_w, GAME_H), t_col, -1)
    # Glow tip on timer bar
    if timer_w > 4:
        cv2.rectangle(canvas, (timer_w - 4, GAME_H - 10), (timer_w, GAME_H),
                      (min(255, t_col[0]+80), min(255, t_col[1]+80), min(255, t_col[2]+80)), -1)

    return canvas


def page_falling_pose_game():
    """
    Falling Pose Game — Enhanced two-panel layout.
    LEFT  : animated game canvas with particle effects, combo system, progressive speed.
    RIGHT : live webcam with skeleton overlay and pose detection.
    """
    st.markdown("# 🌊 Falling Pose Game")
    st.markdown(
        "*Pose cards fall from the sky — match the pose shown and hold it to catch the card! "
        "Build combos for bonus points! Cards speed up over time!* 🔥"
    )
    st.divider()

    uid = st.session_state.user_id

    # ── Results screen ────────────────────────────────────────────────────
    if st.session_state.get("fpg_results") and not st.session_state.get("fpg_active"):
        results = st.session_state.fpg_results
        ge = GameEngine(uid)
        gr = ge.process_session(results)
        st.balloons()

        final_score  = results.get("fpg_score", 0)
        caught_count = results.get("poses_completed", 0)
        best_combo   = results.get("max_combo", 1)

        # Big score display
        st.markdown(f"""
        <div style="text-align:center;padding:30px 0;">
          <div style="font-size:16px;color:#a090cc;text-transform:uppercase;letter-spacing:3px">Final Score</div>
          <div style="font-size:88px;font-weight:900;line-height:1;
               background:linear-gradient(135deg,#f9d423,#ff4e50);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent">{final_score}</div>
          <div style="font-size:14px;color:#888;margin-top:4px">Game Points</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧘 Poses Caught",   caught_count)
        c2.metric("🔥 Best Combo",      f"x{best_combo}")
        c3.metric("⚡ XP Earned",       f"+{gr['xp_earned']:,}")
        c4.metric("⏱️ Duration",        f"{results['duration']:.0f}s")
        st.divider()
        st.plotly_chart(score_gauge(results["total_score"]), use_container_width=True)
        if gr.get("new_achievements"):
            render_achievements(gr["new_achievements"])
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🔄 Play Again", type="primary", use_container_width=True):
                for k in ["fpg_results","fpg_active","fpg_score","fpg_active_cards",
                          "fpg_next_spawn","fpg_hold_pose_name","fpg_hold_start",
                          "fpg_combo","fpg_particles","fpg_last_catch_pts","fpg_max_combo",
                          "fpg_poses_caught","fpg_last_catch_time"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with bcol2:
            if st.button("🏠 Back to Home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
        return

    # ── Config screen ─────────────────────────────────────────────────────
    if not st.session_state.get("fpg_active"):
        # Pre-warm camera in background
        if "fpg_camera_prewarmed" not in st.session_state:
            st.session_state.fpg_camera_prewarmed = False
        
        if not st.session_state.fpg_camera_prewarmed:
            with st.spinner("📷 Preparing camera..."):
                if prewarm_camera():
                    st.session_state.fpg_camera_prewarmed = True
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚙️ Settings")
            duration = st.slider("Game Duration (s)", 30, 120, 60, step=10, key="fpg_dur_sl")
            st.session_state.fpg_duration = duration
            st.markdown("### 🎮 New Features")
            st.markdown("""
            <div class="yoga-card">
              <ul style="color:#c8baff;line-height:2.2;margin:0;padding-left:18px">
                <li>🔥 <strong>Combo System</strong> — catch 2+ in a row for bonus!</li>
                <li>⚡ <strong>Speed Ramp</strong> — cards fall faster over time</li>
                <li>✨ <strong>Particle Effects</strong> — sparks on every catch!</li>
                <li>🌈 <strong>Color-coded cards</strong> — each pose has its color</li>
                <li>⏳ <strong>Hold ring</strong> — visual arc shows hold progress</li>
              </ul>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("### 📋 How to Play")
            st.markdown("""
            <div class="yoga-card">
              <ol style="color:#c8baff;line-height:2.2;margin:0;padding-left:18px">
                <li>Press <strong>START</strong> and stand back from camera</li>
                <li>Watch the <strong>left panel</strong> — colored pose cards fall from the top</li>
                <li>Strike the matching yoga pose in the <strong>right panel</strong> (camera)</li>
                <li>A yellow ring appears — hold until it completes (0.5s)</li>
                <li>🎉 <strong>CAUGHT!</strong> — Points added, keep the combo going!</li>
              </ol>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 🃏 Card Values")
        cols = st.columns(4)
        for i, (pname, info) in enumerate(_FPG_POSE_INFO.items()):
            emoji = _FPG_EMOJI_MAP.get(pname, "🧘")
            col_hex = "#{:02x}{:02x}{:02x}".format(*info["glow"][::-1])  # BGR->RGB
            with cols[i % 4]:
                st.markdown(
                    f'<div class="yoga-card" style="text-align:center;padding:10px;'
                    f'border-color:{col_hex};border-width:2px">' +
                    f'<div style="font-size:28px">{emoji}</div>' +
                    f'<div style="color:{col_hex};font-size:12px;font-weight:700">{info["english"]}</div>' +
                    f'<div style="color:#f9d423;font-weight:800;font-size:16px">+{info["points"]} pts</div></div>',
                    unsafe_allow_html=True)

        st.divider()
        if st.button("▶️ START GAME", type="primary", use_container_width=True):
            # Camera already pre-warmed, quick start
            _cap = get_camera()
            if _cap.isOpened():
                _cap.read()  # Single flush
            
            st.session_state.fpg_active         = True
            st.session_state.fpg_score          = 0
            st.session_state.fpg_start_time     = time.time()
            st.session_state.fpg_active_cards   = []
            st.session_state.fpg_results        = None
            st.session_state.fpg_next_spawn     = time.time() + random.uniform(_FPG_SPAWN_MIN, _FPG_SPAWN_MAX)
            st.session_state.fpg_hold_pose_name = None
            st.session_state.fpg_cached_analysis = None
            st.session_state.fpg_last_analysis_ts = 0.0
            st.session_state.fpg_hold_start     = None
            st.session_state.fpg_combo          = 0
            st.session_state.fpg_max_combo      = 0
            st.session_state.fpg_particles      = []
            st.session_state.fpg_last_catch_pts = 0
            st.session_state.fpg_poses_caught   = 0
            st.rerun()
        return

    # ═════════════════════════════════════════════════════════════════════
    # GAME LOOP — wrapped in @st.fragment so only this block re-renders
    # on each frame tick, keeping the camera feed persistent.
    # ═════════════════════════════════════════════════════════════════════

    @st.fragment
    def _fpg_game_fragment():
        """Self-contained game loop fragment — only this block re-renders
        on each frame tick, so the camera feed never flickers."""

        # ═════════════════════════════════════════════════════════════════════
        # GAME LOOP
        # ═════════════════════════════════════════════════════════════════════

        # ── Frame buffer initialization ───────────────────────────────────
        if "fpg_last_cam_frame" not in st.session_state:
            st.session_state.fpg_last_cam_frame = None
        if "fpg_last_game_frame" not in st.session_state:
            st.session_state.fpg_last_game_frame = None

        # ── Stop button ───────────────────────────────────────────────────────
        if st.button("⏹ Exit Game", key="fpg_stop", type="secondary"):
            st.session_state.fpg_active = False
            st.session_state.fpg_last_cam_frame = None
            st.session_state.fpg_last_game_frame = None
            release_camera()
            st.rerun(scope="app")
            return

        # ── HUD row ───────────────────────────────────────────────────────────
        now          = time.time()
        start_time   = st.session_state.get("fpg_start_time") or now
        elapsed      = now - start_time
        duration     = st.session_state.get("fpg_duration", 60)
        remaining    = max(0, int(duration - elapsed))
        score        = st.session_state.get("fpg_score", 0)
        combo        = st.session_state.get("fpg_combo", 0)
        max_combo    = st.session_state.get("fpg_max_combo", 0)
        active_cards = st.session_state.get("fpg_active_cards", [])
        particles    = st.session_state.get("fpg_particles", [])
        last_catch   = st.session_state.get("fpg_last_catch_time", 0)
        last_catch_pts = st.session_state.get("fpg_last_catch_pts", 0)

        # ── Fancy inline HUD ─────────────────────────────────────────────────
        timer_pct = remaining / max(1, duration) * 100
        t_bar_col = "#00d452" if remaining > 20 else "#0050ff" if remaining > 10 else "#cc0000"
        combo_html = (
            f'<div class="fpg-combo-badge">🔥 x{combo} COMBO!</div>'
            if combo > 1 else ""
        )
        st.markdown(f"""
        <div class="fpg-hud">
          <div class="fpg-hud-stat"><div class="val">🏆 {score}</div><div class="lbl">Score</div></div>
          <div class="fpg-hud-stat"><div class="val">⏱️ {remaining}s</div><div class="lbl">Remaining</div></div>
          <div class="fpg-hud-stat"><div class="val">🃏 {len(active_cards)}</div><div class="lbl">Cards</div></div>
          <div class="fpg-hud-stat"><div class="val">🔥 x{max(1,combo)}</div><div class="lbl">Combo</div></div>
        </div>
        <div class="fpg-timer-bar-wrap">
          <div class="fpg-timer-bar-fill" style="width:{timer_pct:.1f}%;background:{t_bar_col}"></div>
        </div>
        {combo_html}
        """, unsafe_allow_html=True)

        # ── Two panels ────────────────────────────────────────────────────────
        col_game, col_cam = st.columns(2)
        with col_game:
            st.markdown("#### 🎮 Falling Cards")
            game_ph = st.empty()
            # Display buffered game frame immediately
            if st.session_state.fpg_last_game_frame is not None:
                game_ph.image(st.session_state.fpg_last_game_frame, use_container_width=True)
        with col_cam:
            st.markdown("#### 📷 Camera — Strike the Pose!")
            cam_ph  = st.empty()
            pose_ph = st.empty()
            # Display buffered camera frame immediately
            if st.session_state.fpg_last_cam_frame is not None:
                cam_ph.image(st.session_state.fpg_last_cam_frame, use_container_width=True)

        GAME_W, GAME_H = 480, 540

        # ── Time-up? ──────────────────────────────────────────────────────────
        if elapsed >= duration:
            st.session_state.fpg_active = False
            st.session_state.fpg_last_cam_frame = None
            st.session_state.fpg_last_game_frame = None
            avg_score = min(100.0, score * 1.5)
            caught_count = st.session_state.get("fpg_poses_caught", 0)
            st.session_state.fpg_results = {
                "total_score":          round(avg_score, 1),
                "duration":             round(elapsed, 1),
                "poses_completed":      caught_count,
                "unique_poses":         min(len(_FPG_POSE_INFO), max(1, caught_count // 2)),
                "pose_details": [{
                    "pose_name": "FallingPoseGame", "english_name": "Falling Pose Game",
                    "emoji": "🌊", "score": avg_score, "adjusted_score": avg_score,
                    "hold_time": float(duration), "hold_bonus": 5.0,
                    "stability": 75.0, "grade": get_grade(avg_score), "suggestions": [],
                }],
                "suggestions": [], "completed_poses_list": [],
                "surya_namaskar_pct": 0.0, "max_combo": max_combo,
                "session_type": "falling_pose_game", "fpg_score": score,
            }
            st.rerun(scope="app")
            return

        # ── Progressive speed factor ──────────────────────────────────────────
        speed_boost = 1.0 + elapsed * _FPG_SPEED_SCALE

        # ── Spawn cards ───────────────────────────────────────────────────────
        next_spawn = st.session_state.get("fpg_next_spawn", now)
        # Spawn interval shrinks as game progresses
        spawn_min = max(0.8, _FPG_SPAWN_MIN - elapsed * 0.015)
        spawn_max = max(1.5, _FPG_SPAWN_MAX - elapsed * 0.015)
        if len(active_cards) < _FPG_MAX_ACTIVE and now >= next_spawn:
            new_pose = random.choice(list(_FPG_POSE_INFO.keys()))
            active_cards.append({
                "pose_name": new_pose,
                "x": float(random.randint(10, max(10, GAME_W - _FPG_IMAGE_SIZE - 50))),
                "y": float(-_FPG_IMAGE_SIZE - 10),
                "speed": random.uniform(1.8, 3.8) * speed_boost,
            })
            st.session_state.fpg_next_spawn = now + random.uniform(spawn_min, spawn_max)

        # ── Camera: read one frame + detect pose ──────────────────────────────
        detected_pose  = "Unknown"
        detected_score = 0.0
        cam_frame_rgb  = None

        try:
            cap = get_camera()
            if not cap or not cap.isOpened():
                cap = get_camera()

            ret, frame = cap.read()
            if not ret:
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret:
                        break
                    time.sleep(0.02)

            if not ret:
                if st.session_state.fpg_last_cam_frame is None:
                    cam_ph.warning("⚠️ Camera frame dropped")
            else:
                frame = cv2.flip(frame, 1)
                # Enhance brightness for better visibility in Live Session
                frame = enhance_frame_brightness(frame, alpha=1.3, beta=25)
                analysis = st.session_state.get("fpg_cached_analysis")
                if analysis is None or should_analyze_now("fpg", 0.16):
                    pose_model = get_pose_model()
                    analysis = analyze_frame(frame, pose_model)
                    st.session_state.fpg_cached_analysis = analysis

                res = analysis.get("results")
                from pose_detector import draw_skeleton, PoseDetectionResult
                if isinstance(res, PoseDetectionResult) and res.landmarks is not None:
                    # Coloured skeleton based on match
                    sk_col = (0, 220, 80) if detected_score >= 65 else (80, 140, 255)
                    draw_skeleton(frame, res, sk_col, thickness=2)

                detected_pose  = analysis.get("pose_name", "Unknown")
                detected_score = analysis.get("score", 0.0)

                # Overlay: pose score bar
                cam_h, cam_w = frame.shape[:2]
                cv2.rectangle(frame, (0, cam_h - 12), (cam_w, cam_h), (15, 15, 25), -1)
                if detected_pose not in ("Unknown", "No pose detected"):
                    has_card = any(c["pose_name"] == detected_pose for c in active_cards)
                    bar_col = (0, 220, 80) if (detected_score >= 65 and has_card) else (
                               (0, 160, 255) if detected_score >= 65 else (80, 60, 200))
                    cv2.rectangle(frame, (0, cam_h - 12),
                                  (int(cam_w * detected_score / 100), cam_h), bar_col, -1)
                    cv2.putText(frame,
                        f"{detected_pose.replace('_',' ')} {detected_score:.0f}%" +
                        (" MATCH!" if has_card and detected_score >= 65 else ""),
                        (8, cam_h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw hold-ring on camera if actively holding
                hold_name  = st.session_state.get("fpg_hold_pose_name")
                hold_start = st.session_state.get("fpg_hold_start")
                if hold_name and hold_start and detected_pose == hold_name:
                    pct = min(1.0, (now - hold_start) / _FPG_HOLD_TIME)
                    ring_cx, ring_cy = cam_w // 2, cam_h // 2
                    ring_r = min(cam_w, cam_h) // 6
                    cv2.circle(frame, (ring_cx, ring_cy), ring_r, (60, 60, 80), 6, cv2.LINE_AA)
                    cv2.ellipse(frame, (ring_cx, ring_cy), (ring_r, ring_r),
                                -90, 0, int(360 * pct), (249, 212, 35), 6, cv2.LINE_AA)
                    pct_txt = f"HOLD {int(pct*100)}%"
                    pt_s, _ = cv2.getTextSize(pct_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(frame, pct_txt,
                        (ring_cx - pt_s[0]//2, ring_cy + pt_s[1]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (249, 212, 35), 2, cv2.LINE_AA)

                cam_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception as exc:
            # Silent recovery
            if st.session_state.fpg_last_cam_frame is None:
                cam_ph.warning(f"⚠️ Camera issue: {exc}")

        # ── Hold-detection ────────────────────────────────────────────────────
        hold_pose_name = st.session_state.get("fpg_hold_pose_name")
        hold_start_t   = st.session_state.get("fpg_hold_start")

        matched_pose_name = None
        if detected_pose not in ("Unknown", "No pose detected") and detected_score >= 65:
            for card in active_cards:
                if card["pose_name"] == detected_pose:
                    matched_pose_name = detected_pose
                    break

        just_caught = False
        caught_x, caught_y = GAME_W // 2, GAME_H // 2

        if matched_pose_name:
            if hold_pose_name != matched_pose_name:
                st.session_state.fpg_hold_pose_name = matched_pose_name
                st.session_state.fpg_hold_start     = now
            elif hold_start_t and (now - hold_start_t) >= _FPG_HOLD_TIME:
                for card in active_cards:
                    if card["pose_name"] == matched_pose_name:
                        pts       = _FPG_POSE_INFO[matched_pose_name]["points"]
                        combo     = combo + 1
                        max_combo = max(max_combo, combo)
                        # Combo multiplier
                        multiplier = 1 + (combo - 1) * 0.25
                        bonus_pts  = int(pts * multiplier)
                        score     += bonus_pts
                        caught_x   = int(card["x"]) + _FPG_IMAGE_SIZE // 2
                        caught_y   = int(card["y"]) + _FPG_IMAGE_SIZE // 2
                        just_caught = True
                        st.session_state.fpg_score          = score
                        st.session_state.fpg_combo          = combo
                        st.session_state.fpg_max_combo      = max_combo
                        st.session_state.fpg_last_catch_time = now
                        st.session_state.fpg_last_catch_pts  = bonus_pts
                        last_catch_pts = bonus_pts
                        last_catch     = now
                        # Increment pose count
                        st.session_state.fpg_poses_caught = (
                            st.session_state.get("fpg_poses_caught", 0) + 1)
                        active_cards = [c for c in active_cards if c is not card]
                        break
                st.session_state.fpg_hold_pose_name = None
                st.session_state.fpg_hold_start     = None
        else:
            if combo > 0 and matched_pose_name is None and hold_pose_name:
                # Reset combo only if we had a match and lost it
                pass  # keep combo alive while mid-hold-fail
            if not matched_pose_name:
                st.session_state.fpg_hold_pose_name = None
                st.session_state.fpg_hold_start     = None

        # ── Spawn sparkle particles on catch ─────────────────────────────────
        if just_caught:
            info_c  = _FPG_POSE_INFO.get(matched_pose_name, {})
            p_col   = info_c.get("glow", (249, 212, 35))
            for _ in range(22):
                angle  = random.uniform(0, 2 * np.pi)
                speed  = random.uniform(3, 10)
                particles.append([
                    caught_x + random.randint(-10, 10),
                    caught_y + random.randint(-10, 10),
                    random.randint(3, 7),                  # radius
                    p_col,                                  # color
                    255.0,                                  # alpha
                    np.cos(angle) * speed,                  # vx
                    np.sin(angle) * speed,                  # vy
                ])

        # ── Update particles ─────────────────────────────────────────────────
        updated_particles = []
        for p in particles:
            p[0] += p[5]   # x += vx
            p[1] += p[6]   # y += vy
            p[6] += 0.4    # gravity
            p[4] -= 18     # fade
            p[2] = max(1, p[2] - 0.2)  # shrink
            if p[4] > 0:
                updated_particles.append(p)
        particles = updated_particles
        st.session_state.fpg_particles = particles

        # ── Move cards down (with speed boost) ───────────────────────────────
        for card in active_cards:
            card["speed"] = min(card["speed"] + 0.02, 12.0)  # gradual per-card acceleration
            card["y"] += card["speed"]
        active_cards = [c for c in active_cards if c["y"] <= GAME_H + _FPG_IMAGE_SIZE + 30]
        st.session_state.fpg_active_cards = active_cards

        # ── Build game canvas ─────────────────────────────────────────────────
        canvas = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)
        current_hold_name  = st.session_state.get("fpg_hold_pose_name")
        current_hold_start = st.session_state.get("fpg_hold_start")

        canvas = _fpg_draw_canvas(
            canvas, active_cards, current_hold_name, current_hold_start,
            last_catch, last_catch_pts, score, remaining, duration,
            combo, particles, now, elapsed, GAME_W, GAME_H
        )

        # ── Render panels ─────────────────────────────────────────────────────
        game_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Store in buffer FIRST, then display
        st.session_state.fpg_last_game_frame = game_rgb
        game_ph.image(game_rgb, use_container_width=True)

        if cam_frame_rgb is not None:
            # Store camera frame in buffer
            st.session_state.fpg_last_cam_frame = cam_frame_rgb
            cam_ph.image(cam_frame_rgb, use_container_width=True)

        # ── Pose feedback below camera ────────────────────────────────────────
        if detected_pose not in ("Unknown", "No pose detected"):
            has_match = any(c["pose_name"] == detected_pose for c in active_cards)
            info_e    = _FPG_POSE_INFO.get(detected_pose, {})
            eng_name  = info_e.get("english", detected_pose.replace("_", " "))
            emoji_e   = _FPG_EMOJI_MAP.get(detected_pose, "🧘")
            if detected_score >= 65 and has_match:
                feedback_html = (
                    f'<div class="fpg-caught-banner">'
                    f'{emoji_e} {eng_name} — {detected_score:.0f}% ✅ HOLD IT!</div>'
                )
            elif detected_score >= 65:
                col_txt = "#F39C12"
                feedback_html = (
                    f'<div style="text-align:center;padding:8px;border-radius:8px;'
                    f'background:rgba(243,156,18,0.12);border:1px solid rgba(243,156,18,0.3);'
                    f'margin-top:4px"><span style="color:{col_txt};font-weight:700;font-size:15px">'
                    f'{emoji_e} {eng_name} — {detected_score:.0f}% (no card yet)</span></div>'
                )
            else:
                feedback_html = (
                    f'<div style="text-align:center;padding:8px;border-radius:8px;'
                    f'background:rgba(155,89,182,0.1);border:1px solid rgba(155,89,182,0.3);'
                    f'margin-top:4px"><span style="color:#9B59B6;font-weight:600;font-size:14px">'
                    f'{emoji_e} {eng_name} — {detected_score:.0f}% (improve pose)</span></div>'
                )
        else:
            feedback_html = (
                '<div style="text-align:center;padding:8px;color:#666;font-size:13px;'
                'border-radius:8px;background:rgba(255,255,255,0.03);'
                'margin-top:4px">'
                '📷 Step back — show your full body</div>'
            )
        pose_ph.markdown(feedback_html, unsafe_allow_html=True)

        time.sleep(0.04)  # ~25 FPS, reduced for smoother experience
        st.rerun()

    # Invoke the fragment
    _fpg_game_fragment()

# ─────────────────────────────────────────────────────────────────────────────
# Wall Game Results
# ─────────────────────────────────────────────────────────────────────────────
def render_wall_results(results: dict, game_report: dict):
    """Full results page for the Pose Wall Challenge."""
    st.balloons()

    is_victory = results.get("wall_completed", False)
    if is_victory:
        st.markdown("# 🏆 Wall Challenge — VICTORY!")
    else:
        st.markdown("# 💥 Wall Challenge — Game Over")

    st.divider()

    # ── Top metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🏆 Wall Score",       f"{results.get('wall_score', 0):,}")
    c2.metric("🧱 Survived",         f"{results.get('wall_survived', 0)}/{results.get('wall_rounds', 0)}")
    c3.metric("🎯 Perfect Fits",     results.get("wall_perfect_fits", 0))
    c4.metric("🔥 Best Combo",       results.get("max_combo", 0))
    c5.metric("⚡ XP Earned",        f"+{game_report.get('xp_earned', 0):,}")
    c6.metric("⏱️ Duration",         f"{results.get('duration', 0):.0f}s")

    st.divider()

    # ── Round-by-round chart ───────────────────────────────────────────────
    round_results = results.get("round_results", [])
    if round_results:
        st.markdown("#### 📊 Round-by-Round Scores")
        round_nums   = [f"R{rr.round_num}" for rr in round_results]
        round_scores = [rr.wall_round_score for rr in round_results]
        verdicts     = [rr.verdict for rr in round_results]

        verdict_colors = {
            "PERFECT FIT": "#FFD700",
            "GREAT FIT":   "#4ECDC4",
            "GOOD FIT":    "#2ECC71",
            "CLOSE MISS":  "#F39C12",
            "COLLISION":   "#E74C3C",
        }
        bar_colors = [verdict_colors.get(v, "#888") for v in verdicts]

        fig = go.Figure(go.Bar(
            x=round_nums, y=round_scores,
            marker_color=bar_colors,
            text=[f"{s:.0f}" for s in round_scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.06)",
                       color="#ccc", title="Score"),
            xaxis=dict(color="#ccc"),
            font_color="#ccc", height=280,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Collision heatmap ──────────────────────────────────────────────────
    collision_hm = results.get("collision_heatmap", {})
    if any(v > 0 for v in collision_hm.values()):
        st.markdown("#### 🔥 Collision Heatmap")
        st.markdown("*Which limbs collided most often:*")
        col_hm, col_chart = st.columns([1, 2])

        with col_hm:
            for limb, count in sorted(collision_hm.items(), key=lambda x: -x[1]):
                if count > 0:
                    label = limb.replace("_", " ").title()
                    bar_len = min(count * 15, 100)
                    st.markdown(f"""
                    <div style="margin:3px 0;display:flex;align-items:center;gap:8px">
                      <span style="color:#c8baff;font-size:12px;min-width:130px">{label}</span>
                      <div style="background:#E74C3C;height:12px;width:{bar_len}px;border-radius:6px"></div>
                      <span style="color:#ff6b6b;font-size:12px;font-weight:700">{count}×</span>
                    </div>
                    """, unsafe_allow_html=True)

        with col_chart:
            limbs_with_hits = {k: v for k, v in collision_hm.items() if v > 0}
            if limbs_with_hits:
                fig_hm = go.Figure(go.Bar(
                    x=[k.replace("_", " ").title() for k in limbs_with_hits.keys()],
                    y=list(limbs_with_hits.values()),
                    marker_color="#E74C3C",
                ))
                fig_hm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", height=220,
                    margin=dict(t=10, b=10, l=10, r=10),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Hits"),
                    xaxis=dict(tickangle=-45),
                )
                st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # ── Difficulty badge ───────────────────────────────────────────────────
    diff = results.get("difficulty", "normal")
    diff_label = DIFFICULTY_PRESETS.get(diff, {}).get("label", diff.title())
    flow_tag = " 🌀 FLOW STATE REACHED!" if results.get("flow_state_reached") else ""
    st.markdown(f"""
    <div class="yoga-card" style="text-align:center">
      <span style="font-size:13px;color:#a090cc">Difficulty: </span>
      <span style="font-size:16px;font-weight:700;color:#f9d423">{diff_label}</span>
      <span style="color:#4ECDC4;font-weight:700">{flow_tag}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Full session report (XP, level, achievements) ──────────────────────
    render_session_report(results, game_report)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    if not st.session_state.user_id:
        # Landing page for unauthenticated users
        st.markdown("""
        <div style="text-align:center;padding:60px 20px">
          <div style="font-size:96px">🧘</div>
          <h1 style="font-size:48px">YogaQuest</h1>
          <p style="color:#c8baff;font-size:18px;max-width:520px;margin:0 auto">
            AI-powered yoga training with real-time pose detection, scoring,
            and full gamification — XP, levels, achievements, streaks &amp; leaderboard.
          </p>
          <br/>
          <div style="color:#888;font-size:14px">
            Enter your username in the sidebar to begin your journey. 👈
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        features = [
            ("🤖 AI Pose Detection",    "YOLO pose detection + custom angle scoring for 7 poses"),
            ("🧱 Pose Wall Game",       "Real-time arcade wall challenge with collision detection"),
            ("🎯 Pose Target Game",     "Guide limbs into glowing targets pose-by-pose"),
            ("🌊 Falling Pose Game",    "Hold poses to catch falling yoga cards and score"),
            ("☀️ Surya Namaskar",       "Guided 12-step sun salutation sequence"),
            ("🎮 Gamification",         "XP, levels, achievements, streaks & leaderboard"),
        ]
        cols = st.columns(3)
        for i, (title, desc) in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="yoga-card" style="text-align:center;margin-top:12px">
                  <div style="font-size:28px">{title[:2]}</div>
                  <div style="font-weight:700;color:#e0d7ff">{title[2:]}</div>
                  <div style="color:#888;font-size:12px;margin-top:6px">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        return

    # Route to the selected page
    page_map = {
        "home":              page_home,
        "practice":          page_practice,
        "live":              page_live,
        "video":             page_video,
        "surya":             page_surya,
        "leaderboard":       page_leaderboard,
        "profile":           page_profile,
        "wall_game":         page_wall_game,
        "pose_target_game":  page_pose_target_game,
        "falling_pose_game": page_falling_pose_game,
    }
    page_fn = page_map.get(st.session_state.page, page_home)
    page_fn()


if __name__ == "__main__":
    main()
