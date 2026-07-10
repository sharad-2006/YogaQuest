"""
video_processor.py — Processes uploaded video files frame-by-frame.
Detects yoga poses, annotates each frame, writes an output video,
and returns a full session-results dict.
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any, List

from pose_analyzer import (
    analyze_frame, annotate_frame, SessionAnalyzer, get_grade
)
from pose_detector import YoloPoseModel
from poses_config import YOGA_POSES


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    input_path:       str,
    output_path:      str,
    target_pose:      Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    analysis_fps:     float = 8.0,
) -> Dict[str, Any]:
    """
    Analyse *input_path* frame-by-frame with YOLO pose detection.

    Parameters
    ----------
    input_path       : path to the source video (mp4 / avi / mov …)
    output_path      : destination path for the annotated video
    target_pose      : if set, score only this pose; otherwise auto-detect
    progress_callback: called with a 0→1 float as frames are processed
    analysis_fps     : maximum analysis rate (sub-sample heavy videos)

    Returns
    -------
    Session results dict (same schema as SessionAnalyzer.finish_session()).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # ── Video metadata ────────────────────────────────────────────────────────
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / src_fps

    # Sub-sample: analyse at most analysis_fps frames per second
    analyse_every = max(1, int(src_fps / analysis_fps))

    # ── Writer setup ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

    # ── Session tracker ───────────────────────────────────────────────────────
    session    = SessionAnalyzer()
    session.start_session()

    last_analysis: Dict[str, Any] = {"pose_detected": False,
                                     "pose_name": "", "score": 0,
                                     "grade": "F", "checks": [],
                                     "landmarks": None, "results": None}
    frame_idx = 0
    timeline: List[Dict] = []

    with YoloPoseModel(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=False,
    ) as pose_model:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / src_fps

            # ── Analyse (sub-sampled) ─────────────────────────────────────────
            if frame_idx % analyse_every == 0:
                last_analysis = analyze_frame(frame, pose_model, target_pose)
                session.add_frame(last_analysis, timestamp)
                timeline.append({
                    "ts":        round(timestamp, 2),
                    "pose_name": last_analysis.get("pose_name", "Unknown"),
                    "score":     last_analysis.get("score", 0.0),
                })

            # ── Annotate every frame (reuse last analysis for speed) ──────────
            annotated = annotate_frame(frame, last_analysis, show_skeleton=True)

            # Add timestamp watermark
            cv2.putText(
                annotated,
                f"t={timestamp:.1f}s  |  frame {frame_idx}",
                (width - 260, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA,
            )

            out.write(annotated)
            frame_idx += 1

            if progress_callback:
                progress_callback(min(1.0, frame_idx / total_frames))

    cap.release()
    out.release()

    # ── Compile results ───────────────────────────────────────────────────────
    results = session.finish_session()
    results["duration"]     = round(duration_s, 1)
    results["total_frames"] = total_frames
    results["fps"]          = src_fps
    results["session_type"] = "video"
    results["timeline"]     = timeline

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Timeline utility
# ─────────────────────────────────────────────────────────────────────────────

def build_pose_segments(timeline: List[Dict]) -> List[Dict]:
    """
    Collapse the per-frame timeline into contiguous pose segments.
    Returns list of {pose_name, start, end, duration, avg_score}.
    """
    if not timeline:
        return []

    segments: List[Dict] = []
    seg_pose  = timeline[0]["pose_name"]
    seg_start = timeline[0]["ts"]
    seg_scores= [timeline[0]["score"]]

    for entry in timeline[1:]:
        if entry["pose_name"] == seg_pose:
            seg_scores.append(entry["score"])
        else:
            segments.append({
                "pose_name": seg_pose,
                "start":     seg_start,
                "end":       entry["ts"],
                "duration":  round(entry["ts"] - seg_start, 2),
                "avg_score": round(float(np.mean(seg_scores)), 1),
            })
            seg_pose   = entry["pose_name"]
            seg_start  = entry["ts"]
            seg_scores = [entry["score"]]

    # Last segment
    last_ts = timeline[-1]["ts"]
    segments.append({
        "pose_name": seg_pose,
        "start":     seg_start,
        "end":       last_ts,
        "duration":  round(last_ts - seg_start, 2),
        "avg_score": round(float(np.mean(seg_scores)), 1),
    })

    return [s for s in segments
            if s["pose_name"] not in ("Unknown", "No pose detected")
            and s["duration"] >= 1.0]
