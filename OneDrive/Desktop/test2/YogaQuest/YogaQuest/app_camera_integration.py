"""
app_camera_integration.py — Integration helper for Streamlit app
Provides simple API to use the optimized camera and processing system.
"""

import streamlit as st
import cv2
import numpy as np
from typing import Optional, Dict, Any
import time

from camera_manager import get_global_camera, release_global_camera
from frame_processor import FrameProcessor, FrameAnnotator, AdaptiveFrameSkipper
from pose_analyzer import analyze_frame
from pose_detector import YoloPoseModel


class StreamlitCameraSession:
    """
    Manages a camera session for Streamlit with automatic state handling.
    Solves flickering and lag issues with minimal code changes.
    """
    
    def __init__(self, 
                 session_key: str,
                 target_ui_fps: float = 20.0,
                 target_process_fps: float = 8.0,
                 use_full_annotation: bool = False):
        """
        Args:
            session_key: Unique key for this camera session in st.session_state
            target_ui_fps: Target frames per second for UI updates
            target_process_fps: Target rate for pose detection processing
            use_full_annotation: Use detailed annotation (slower) vs simple overlay
        """
        self.session_key = session_key
        self.target_ui_fps = target_ui_fps
        self.target_process_fps = target_process_fps
        self.use_full_annotation = use_full_annotation
        
        # Initialize session state keys
        self._init_session_state()
        
    def _init_session_state(self):
        """Initialize required session state variables."""
        prefix = f"{self.session_key}_"
        
        if f"{prefix}initialized" not in st.session_state:
            st.session_state[f"{prefix}initialized"] = False
        if f"{prefix}pose_model" not in st.session_state:
            st.session_state[f"{prefix}pose_model"] = None
        if f"{prefix}processor" not in st.session_state:
            st.session_state[f"{prefix}processor"] = None
        if f"{prefix}frame_skipper" not in st.session_state:
            st.session_state[f"{prefix}frame_skipper"] = None
        if f"{prefix}last_analysis" not in st.session_state:
            st.session_state[f"{prefix}last_analysis"] = None
        if f"{prefix}last_rgb_frame" not in st.session_state:
            st.session_state[f"{prefix}last_rgb_frame"] = None
        if f"{prefix}fps_tracker" not in st.session_state:
            st.session_state[f"{prefix}fps_tracker"] = {"times": [], "last_update": 0}
    
    def start(self, target_pose: Optional[str] = None) -> bool:
        """
        Start the camera session.
        Returns True if successful, False otherwise.
        """
        prefix = f"{self.session_key}_"
        
        # Get camera
        camera = get_global_camera(camera_index=0, target_fps=30)
        if not camera.is_running():
            return False
        
        # Initialize pose model (cached)
        if st.session_state[f"{prefix}pose_model"] is None:
            st.session_state[f"{prefix}pose_model"] = YoloPoseModel(
                static_image_mode=False,
                min_detection_confidence=0.5,
            )
        
        pose_model = st.session_state[f"{prefix}pose_model"]
        
        # Create processing function
        def process_frame(frame: np.ndarray) -> Dict[str, Any]:
            return analyze_frame(frame, pose_model, target_pose)
        
        # Initialize frame processor
        if st.session_state[f"{prefix}processor"] is None:
            processor = FrameProcessor(process_frame, self.target_process_fps)
            processor.start()
            st.session_state[f"{prefix}processor"] = processor
        
        # Initialize frame skipper
        if st.session_state[f"{prefix}frame_skipper"] is None:
            st.session_state[f"{prefix}frame_skipper"] = AdaptiveFrameSkipper(self.target_ui_fps)
        
        st.session_state[f"{prefix}initialized"] = True
        return True
    
    def get_frame_and_analysis(self, 
                              frame_placeholder,
                              show_fps: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get current frame with analysis overlay and display it.
        This is the main method to call in your Streamlit loop.
        
        Uses persistent display pattern to prevent flicker on st.rerun().
        
        Args:
            frame_placeholder: Streamlit placeholder to display frame (st.empty())
            show_fps: Show FPS counter on frame
            
        Returns:
            Analysis dict or None
        """
        prefix = f"{self.session_key}_"
        
        if not st.session_state[f"{prefix}initialized"]:
            return None
        
        camera = get_global_camera()
        processor = st.session_state[f"{prefix}processor"]
        frame_skipper = st.session_state[f"{prefix}frame_skipper"]
        
        if not camera.is_running() or processor is None:
            return None
        
        # Get latest frame from camera
        frame = camera.get_latest_frame()
        if frame is None:
            # Show last displayed frame to prevent blank screen
            last_rgb = st.session_state.get(f"{prefix}last_rgb_frame")
            if last_rgb is not None:
                frame_placeholder.image(last_rgb, use_container_width=True, channels="RGB")
            return st.session_state[f"{prefix}last_analysis"]
        
        # Submit frame for processing if it's time
        if frame_skipper.should_process_frame():
            processor.submit_frame(frame)
        
        # Get latest analysis result (may be from previous frame)
        analysis = processor.get_latest_result()
        if analysis is not None:
            st.session_state[f"{prefix}last_analysis"] = analysis
        else:
            analysis = st.session_state[f"{prefix}last_analysis"]
        
        # Calculate FPS
        fps_tracker = st.session_state[f"{prefix}fps_tracker"]
        now = time.time()
        fps_tracker["times"].append(now)
        # Keep last 30 samples
        fps_tracker["times"] = fps_tracker["times"][-30:]
        
        if len(fps_tracker["times"]) >= 2:
            time_span = fps_tracker["times"][-1] - fps_tracker["times"][0]
            current_fps = (len(fps_tracker["times"]) - 1) / time_span if time_span > 0 else 0
        else:
            current_fps = 0
        
        # Annotate frame
        if self.use_full_annotation and analysis is not None:
            from pose_analyzer import annotate_frame as full_annotate
            annotated = full_annotate(frame, analysis, show_skeleton=True)
        else:
            annotated = FrameAnnotator.annotate_simple(
                frame, analysis, show_fps=show_fps, fps=current_fps
            )
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Store last frame for buffering
        st.session_state[f"{prefix}last_rgb_frame"] = rgb_frame
        
        # Display using the SAME placeholder (this prevents DOM recreation flicker)
        frame_placeholder.image(rgb_frame, use_container_width=True, channels="RGB")
        
        return analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        prefix = f"{self.session_key}_"
        
        camera = get_global_camera()
        processor = st.session_state.get(f"{prefix}processor")
        frame_skipper = st.session_state.get(f"{prefix}frame_skipper")
        
        stats = {
            "initialized": st.session_state.get(f"{prefix}initialized", False),
            "camera": camera.get_stats() if camera else {},
            "processor": processor.get_stats() if processor else {},
            "ui_fps": frame_skipper.get_actual_fps() if frame_skipper else 0,
        }
        
        return stats
    
    def stop(self):
        """Stop the camera session and clean up resources."""
        prefix = f"{self.session_key}_"
        
        # Stop processor
        processor = st.session_state.get(f"{prefix}processor")
        if processor is not None:
            processor.stop()
            st.session_state[f"{prefix}processor"] = None
        
        # Clear model
        st.session_state[f"{prefix}pose_model"] = None
        
        # Reset state
        st.session_state[f"{prefix}initialized"] = False
        st.session_state[f"{prefix}last_analysis"] = None
        st.session_state[f"{prefix}frame_skipper"] = None
        st.session_state[f"{prefix}fps_tracker"] = {"times": [], "last_update": 0}
    
    @staticmethod
    def release_all_cameras():
        """Release all camera resources globally."""
        release_global_camera()


# Helper functions for easy integration

def create_camera_session(session_key: str, **kwargs) -> StreamlitCameraSession:
    """
    Create a new camera session.
    
    Usage:
        session = create_camera_session("my_game", target_ui_fps=25.0)
        if session.start():
            # In your loop:
            analysis = session.get_frame_and_analysis(frame_placeholder)
    """
    return StreamlitCameraSession(session_key, **kwargs)


def quick_camera_frame(frame_placeholder, 
                       session_key: str = "default",
                       target_pose: Optional[str] = None,
                       show_fps: bool = True) -> Optional[Dict[str, Any]]:
    """
    Simplified API for displaying camera with pose detection.
    Handles all initialization and state management automatically.
    
    Usage in Streamlit:
        frame_ph = st.empty()
        while game_active:
            analysis = quick_camera_frame(frame_ph, session_key="game1")
            if analysis:
                score = analysis.get("score", 0)
                st.write(f"Score: {score}")
            time.sleep(0.05)  # ~20 FPS
            st.rerun()
    """
    session_state_key = f"_camera_session_{session_key}"
    
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = create_camera_session(session_key)
    
    session: StreamlitCameraSession = st.session_state[session_state_key]
    
    if not session.start(target_pose):
        frame_placeholder.error("Failed to initialize camera")
        return None
    
    return session.get_frame_and_analysis(frame_placeholder, show_fps)
