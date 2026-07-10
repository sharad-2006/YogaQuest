"""
camera_manager.py — Stable camera management with frame buffering
Solves flickering and lag issues by maintaining a persistent camera connection
and using a background thread for frame capture.
"""

import cv2
import threading
import time
import queue
from typing import Optional, Tuple
import numpy as np


class CameraManager:
    """
    Thread-safe camera manager that maintains a stable connection and
    continuously captures frames in the background.
    
    This solves two critical issues:
    1. Flickering: Camera stays open across Streamlit reruns
    2. Lag: Fresh frames are always available without blocking
    """
    
    def __init__(self, camera_index: int = 0, target_fps: int = 30):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Threading components
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=2)  # Keep only latest frames
        
        # Status tracking
        self._is_initialized = False
        self._last_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._error_count = 0
        
    def start(self) -> bool:
        """
        Initialize camera and start background capture thread.
        Returns True if successful.
        """
        if self._is_initialized:
            return True
            
        # Open camera with optimized settings
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not self._cap.isOpened():
            self._cap.release()
            # Fallback to default backend
            self._cap = cv2.VideoCapture(self.camera_index)
            
        if not self._cap.isOpened():
            return False
            
        # Configure for stability and performance
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        # Warm up camera - discard first few frames
        for _ in range(5):
            self._cap.read()
        
        # Start background capture thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        self._is_initialized = True
        return True
    
    def _capture_loop(self):
        """
        Background thread that continuously captures frames.
        Runs at target FPS and always provides fresh frames.
        """
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            
            # Grab latest frame (skip buffer)
            if self._cap.grab():
                ret, frame = self._cap.retrieve()
                
                if ret and frame is not None:
                    # Flip for mirror view
                    frame = cv2.flip(frame, 1)
                    
                    # Update queue - discard old frames if full
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    try:
                        self._frame_queue.put_nowait(frame)
                        self._last_frame = frame
                        self._frame_count += 1
                        self._error_count = 0
                    except queue.Full:
                        pass
                else:
                    self._error_count += 1
                    # If too many errors, try to recover
                    if self._error_count > 30:
                        self._reinitialize_camera()
            else:
                self._error_count += 1
                
            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _reinitialize_camera(self):
        """Attempt to recover from camera errors."""
        try:
            if self._cap is not None:
                self._cap.release()
            time.sleep(0.5)
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = cv2.VideoCapture(self.camera_index)
            self._error_count = 0
        except Exception:
            pass
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get the latest frame from the camera.
        Returns immediately with the most recent frame or last known frame.
        
        Args:
            timeout: Maximum time to wait for a new frame (seconds)
            
        Returns:
            BGR frame (already flipped for mirror view) or None
        """
        if not self._is_initialized:
            return None
        
        try:
            # Try to get fresh frame
            frame = self._frame_queue.get(timeout=timeout)
            self._last_frame = frame
            return frame
        except queue.Empty:
            # Return last known frame if queue is empty
            return self._last_frame
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest available frame without waiting.
        Returns last known frame immediately.
        """
        if not self._is_initialized:
            return None
        return self._last_frame
    
    def is_running(self) -> bool:
        """Check if camera is initialized and thread is running."""
        return self._is_initialized and self._thread is not None and self._thread.is_alive()
    
    def get_stats(self) -> dict:
        """Get camera performance statistics."""
        return {
            "initialized": self._is_initialized,
            "frame_count": self._frame_count,
            "error_count": self._error_count,
            "queue_size": self._frame_queue.qsize(),
            "thread_alive": self._thread.is_alive() if self._thread else False,
        }
    
    def stop(self):
        """Stop background thread and release camera."""
        if not self._is_initialized:
            return
        
        # Stop capture thread
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        
        # Release camera
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self._is_initialized = False
        self._last_frame = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


# Global singleton instance for Streamlit
_global_camera: Optional[CameraManager] = None


def get_global_camera(camera_index: int = 0, target_fps: int = 30) -> CameraManager:
    """
    Get or create the global camera manager instance.
    This maintains camera state across Streamlit reruns.
    """
    global _global_camera
    
    if _global_camera is None:
        _global_camera = CameraManager(camera_index, target_fps)
        _global_camera.start()
    elif not _global_camera.is_running():
        # Reinitialize if stopped
        _global_camera = CameraManager(camera_index, target_fps)
        _global_camera.start()
    
    return _global_camera


def release_global_camera():
    """Release the global camera instance."""
    global _global_camera
    if _global_camera is not None:
        _global_camera.stop()
        _global_camera = None
