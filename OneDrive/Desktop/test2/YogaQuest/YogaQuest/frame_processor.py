"""
frame_processor.py — Optimized async frame processing for pose detection
Solves lag by processing frames in background without blocking the UI.
"""

import threading
import queue
import time
from typing import Optional, Dict, Any, Callable
import numpy as np
import cv2


class FrameProcessor:
    """
    Background frame processor that handles expensive pose detection
    without blocking the main thread.
    
    Benefits:
    - UI remains responsive during YOLO inference
    - Frames can be skipped if processing is slow
    - Always returns latest available result
    """
    
    def __init__(self, 
                 process_func: Callable[[np.ndarray], Dict[str, Any]],
                 target_process_fps: float = 10.0):
        """
        Args:
            process_func: Function that takes a BGR frame and returns analysis dict
            target_process_fps: Maximum processing rate (actual may be lower if model is slow)
        """
        self.process_func = process_func
        self.process_interval = 1.0 / target_process_fps
        
        # Threading components
        self._input_queue = queue.Queue(maxsize=2)
        self._result_queue = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # State tracking
        self._is_running = False
        self._last_result: Optional[Dict[str, Any]] = None
        self._processed_count = 0
        self._skipped_count = 0
        self._last_process_time = 0.0
        
    def start(self):
        """Start the background processing thread."""
        if self._is_running:
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        self._is_running = True
    
    def _process_loop(self):
        """Background thread that processes frames."""
        while not self._stop_event.is_set():
            try:
                # Get frame with timeout
                frame = self._input_queue.get(timeout=0.1)
                
                # Process frame
                process_start = time.time()
                result = self.process_func(frame)
                process_time = time.time() - process_start
                
                self._last_process_time = process_time
                self._processed_count += 1
                
                # Store result
                if self._result_queue.full():
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self._result_queue.put_nowait(result)
                    self._last_result = result
                except queue.Full:
                    pass
                
                # Rate limiting
                if process_time < self.process_interval:
                    time.sleep(self.process_interval - process_time)
                    
            except queue.Empty:
                continue
            except Exception as e:
                # Log error but continue processing
                print(f"Frame processing error: {e}")
                continue
    
    def submit_frame(self, frame: np.ndarray) -> bool:
        """
        Submit a frame for processing.
        Returns False if queue is full (frame will be skipped).
        """
        if not self._is_running:
            return False
        
        try:
            self._input_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            self._skipped_count += 1
            return False
    
    def get_result(self, timeout: float = 0.01) -> Optional[Dict[str, Any]]:
        """
        Get the latest processing result.
        Returns immediately with cached result if no new result is available.
        """
        try:
            result = self._result_queue.get(timeout=timeout)
            self._last_result = result
            return result
        except queue.Empty:
            return self._last_result
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent cached result without waiting."""
        return self._last_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor performance statistics."""
        return {
            "running": self._is_running,
            "processed_frames": self._processed_count,
            "skipped_frames": self._skipped_count,
            "last_process_time_ms": round(self._last_process_time * 1000, 1),
            "input_queue_size": self._input_queue.qsize(),
            "result_queue_size": self._result_queue.qsize(),
        }
    
    def stop(self):
        """Stop the background thread."""
        if not self._is_running:
            return
        
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        
        # Clear queues
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        
        self._is_running = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class FrameAnnotator:
    """
    Lightweight frame annotator that draws analysis results on frames.
    This is fast enough to run in the main thread.
    """
    
    @staticmethod
    def annotate_simple(frame: np.ndarray, 
                       analysis: Optional[Dict[str, Any]],
                       show_fps: bool = False,
                       fps: float = 0.0) -> np.ndarray:
        """
        Quick annotation showing only essential info.
        Much faster than the full annotate_frame() function.
        """
        if analysis is None:
            return frame
        
        out = frame.copy()
        h, w = out.shape[:2]
        
        pose_name = analysis.get("pose_name", "Unknown")
        score = analysis.get("score", 0.0)
        detected = analysis.get("pose_detected", False)
        
        # Simple top banner
        banner_h = 60
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        
        # Draw text
        if detected:
            color = FrameAnnotator._score_color(score)
            text = f"{pose_name}: {score:.0f}/100"
            cv2.putText(out, text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(out, "No pose detected", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 200), 2, cv2.LINE_AA)
        
        # FPS indicator
        if show_fps and fps > 0:
            cv2.putText(out, f"FPS: {fps:.1f}", (w - 120, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
        
        return out
    
    @staticmethod
    def _score_color(score: float) -> tuple:
        """Get BGR color based on score."""
        if score >= 80:
            return (0, 220, 60)  # green
        elif score >= 60:
            return (0, 165, 255)  # orange
        else:
            return (0, 80, 255)  # red


class AdaptiveFrameSkipper:
    """
    Intelligently skips frames based on processing performance
    to maintain target FPS.
    """
    
    def __init__(self, target_ui_fps: float = 20.0):
        self.target_ui_fps = target_ui_fps
        self.frame_interval = 1.0 / target_ui_fps
        self._last_frame_time = 0.0
        self._frame_times = []
        self._max_samples = 30
    
    def should_process_frame(self) -> bool:
        """
        Determine if enough time has passed to process another frame.
        Maintains smooth UI updates at target FPS.
        """
        now = time.time()
        elapsed = now - self._last_frame_time
        
        if elapsed >= self.frame_interval:
            self._last_frame_time = now
            self._frame_times.append(elapsed)
            
            # Keep only recent samples
            if len(self._frame_times) > self._max_samples:
                self._frame_times.pop(0)
            
            return True
        return False
    
    def get_actual_fps(self) -> float:
        """Calculate actual FPS based on recent frame times."""
        if len(self._frame_times) < 2:
            return 0.0
        avg_time = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def reset(self):
        """Reset timing statistics."""
        self._last_frame_time = 0.0
        self._frame_times = []
