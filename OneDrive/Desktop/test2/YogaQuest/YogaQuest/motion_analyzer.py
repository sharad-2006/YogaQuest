"""
motion_analyzer.py — Temporal Motion Analysis Engine for YogaQuest Phase 3.

Maintains a sliding window (ring buffer) of the last N frames and computes
four temporal metrics per push:
    • Stability       – spatial jitter of keypoints while holding a pose
    • Consistency     – how steady the pose score remains across the window
    • Transition      – smoothness of keypoint movement when switching poses
    • Velocity        – per-keypoint speed (pixels/sec, normalised)

The ring buffer stores FrameWindowEntry dataclass instances.  All heavy maths
uses NumPy vectorised operations for performance on the Streamlit rerun loop.

Public API:
    analyzer = MotionAnalyzer(window_size=20, frame_w=640, frame_h=480)
    metrics  = analyzer.push(analysis_dict, timestamp)
    metrics  = analyzer.get_current_metrics()
    analyzer.reset()
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

NUM_COCO_KP = 17  # COCO 17-keypoint model (YOLO native)


@dataclass
class FrameWindowEntry:
    """One slot in the sliding window ring buffer."""
    timestamp:  float
    landmarks:  np.ndarray          # (17, 3)  — px, py, conf  (COCO order)
    pose_name:  str
    pose_score: float
    velocity:   np.ndarray          # (17, 2)  — Δpx/s, Δpy/s per keypoint
    accel:      np.ndarray          # (17, 2)  — Δvel/s² per keypoint


# ─────────────────────────────────────────────────────────────────────────────
# MotionAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class MotionAnalyzer:
    """
    Sliding-window manager + temporal metric calculator.

    Parameters
    ----------
    window_size : int
        Number of frames to keep in the ring buffer (default 20 ≈ 0.7 s @ 28 FPS).
    frame_w, frame_h : int
        Frame resolution — used to normalise velocities against the frame diagonal
        so that the metrics are resolution-independent.
    """

    # ── Tuning constants ─────────────────────────────────────────────────────
    CONF_THRESHOLD:   float = 0.25   # min keypoint confidence to include
    JITTER_PENALTY:   float = 3.5    # stability pts lost per pixel of jitter
    MAX_EXPECTED_V:   float = 0.08   # normalised peak velocity (tune empirically)
    TRANS_WINDOW:     int   = 12     # frames to inspect for transition smoothness
    LAND_SCORE_THR:   float = 65.0   # pose_score threshold to count "landed"

    def __init__(
        self,
        window_size: int = 20,
        frame_w: int = 640,
        frame_h: int = 480,
    ):
        self.window_size: int = max(3, window_size)
        self.frame_w:     int = frame_w
        self.frame_h:     int = frame_h
        self._diag:       float = float(np.sqrt(frame_w ** 2 + frame_h ** 2))

        # Ring buffer implemented as a bounded deque (O(1) append / pop)
        self.window: deque[FrameWindowEntry] = deque(maxlen=self.window_size)

        # Transition tracking state
        self._last_transition_idx: int = -1     # index in window of most recent transition
        self._transition_smoothness: float = 100.0
        self._transition_speed_ms: int = 0
        self._transition_detected: bool = False
        self._peak_velocity_norm: float = 0.0

        # Session-level trackers (for achievements)
        self._smooth_85_count: int = 0
        self._max_stability: float = 0.0
        self._min_transition_ms: int = 9999
        self._flow_transitions: int = 0  # consecutive perfect transitions

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, analysis: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """
        Push a new frame analysis into the ring buffer and return metrics.

        Parameters
        ----------
        analysis : dict
            Output of ``analyze_frame()`` — must contain at minimum:
                pose_detected (bool), pose_name (str), score (float),
                results (PoseDetectionResult with .keypoints_px, .keypoints_conf)
        timestamp : float
            ``time.time()`` of this frame capture.

        Returns
        -------
        dict
            motion_metrics — see ``get_current_metrics()`` for schema.
        """
        landmarks_17 = self._extract_coco17(analysis)
        pose_name  = analysis.get("pose_name", "Unknown")
        pose_score = analysis.get("score", 0.0)

        # ── Compute velocity & acceleration from previous frame ──────────────
        velocity = np.zeros((NUM_COCO_KP, 2), dtype=float)
        accel    = np.zeros((NUM_COCO_KP, 2), dtype=float)

        if len(self.window) > 0:
            prev = self.window[-1]
            dt   = timestamp - prev.timestamp
            if dt > 0 and dt < 2.0:  # ignore gaps > 2 s (e.g. pause)
                velocity = self._compute_velocity(prev.landmarks, landmarks_17, dt)
                if len(self.window) >= 2:
                    accel = self._compute_accel(prev.velocity, velocity, dt)

        # ── Insert into ring buffer ──────────────────────────────────────────
        entry = FrameWindowEntry(
            timestamp=timestamp,
            landmarks=landmarks_17,
            pose_name=pose_name,
            pose_score=pose_score,
            velocity=velocity,
            accel=accel,
        )
        self.window.append(entry)

        # ── Detect transition ────────────────────────────────────────────────
        self._transition_detected = self._detect_transition()
        if self._transition_detected:
            self._transition_smoothness = self._compute_transition_smoothness()
            self._transition_speed_ms   = self._estimate_transition_speed_ms()

            # Session-level achievement tracking
            if self._transition_smoothness >= 85.0:
                self._smooth_85_count += 1
            if self._transition_speed_ms <= 800:
                self._min_transition_ms = min(self._min_transition_ms,
                                              self._transition_speed_ms)

        # ── Compute per-window metrics ───────────────────────────────────────
        stability   = self._compute_stability()
        consistency = self._compute_consistency()

        # Track peak stability for achievements
        if stability > self._max_stability:
            self._max_stability = stability

        # Peak velocity norm across current window
        self._peak_velocity_norm = self._compute_peak_velocity_norm()

        return self.get_current_metrics()

    def get_current_metrics(self) -> Dict[str, Any]:
        """Return the latest motion metrics snapshot."""
        stability   = self._compute_stability()
        consistency = self._compute_consistency()

        # Composite: weighted blend of three sub-scores
        trans = self._transition_smoothness if self._transition_detected else 100.0
        composite = (
            stability   * 0.40 +
            consistency * 0.35 +
            trans       * 0.25
        )

        return {
            "stability":             round(stability, 1),
            "consistency":           round(consistency, 1),
            "transition_smoothness": round(self._transition_smoothness, 1),
            "transition_detected":   self._transition_detected,
            "transition_speed_ms":   self._transition_speed_ms,
            "peak_velocity_norm":    round(self._peak_velocity_norm, 4),
            "composite_motion":      round(composite, 1),
            "window_size":           self.window_size,
            "frames_in_window":      len(self.window),
            # Session-level trackers (for achievement checks)
            "smooth_transitions_85plus": self._smooth_85_count,
            "max_stability":             round(self._max_stability, 1),
            "min_transition_ms":         self._min_transition_ms,
        }

    def reset(self):
        """Clear the window and all tracking state."""
        self.window.clear()
        self._last_transition_idx = -1
        self._transition_smoothness = 100.0
        self._transition_speed_ms = 0
        self._transition_detected = False
        self._peak_velocity_norm = 0.0
        self._smooth_85_count = 0
        self._max_stability = 0.0
        self._min_transition_ms = 9999
        self._flow_transitions = 0

    # ── Internal: keypoint extraction ─────────────────────────────────────────

    def _extract_coco17(self, analysis: Dict[str, Any]) -> np.ndarray:
        """
        Extract COCO-17 keypoints as (17, 3) [px, py, conf] from analysis dict.

        The YOLO pipeline stores raw COCO keypoints in
        ``analysis["results"].keypoints_px``  (17, 2) and
        ``analysis["results"].keypoints_conf`` (17,).
        """
        out = np.zeros((NUM_COCO_KP, 3), dtype=float)
        results = analysis.get("results")
        if results is None:
            return out

        kp_px   = getattr(results, "keypoints_px", None)
        kp_conf = getattr(results, "keypoints_conf", None)

        if kp_px is not None and kp_conf is not None:
            out[:, :2] = kp_px[:NUM_COCO_KP]
            out[:, 2]  = kp_conf[:NUM_COCO_KP]

        return out

    # ── Internal: velocity & acceleration ─────────────────────────────────────

    def _compute_velocity(
        self,
        prev_lm: np.ndarray,
        curr_lm: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Per-keypoint velocity in px/s.  Shape (17, 2).

        Only computed for keypoints that are confident in *both* frames.
        """
        vel = np.zeros((NUM_COCO_KP, 2), dtype=float)
        for i in range(NUM_COCO_KP):
            if prev_lm[i, 2] >= self.CONF_THRESHOLD and curr_lm[i, 2] >= self.CONF_THRESHOLD:
                vel[i] = (curr_lm[i, :2] - prev_lm[i, :2]) / dt
        return vel

    def _compute_accel(
        self,
        prev_vel: np.ndarray,
        curr_vel: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Per-keypoint acceleration in px/s².  Shape (17, 2)."""
        if dt <= 0:
            return np.zeros((NUM_COCO_KP, 2), dtype=float)
        return (curr_vel - prev_vel) / dt

    # ── Internal: stability ───────────────────────────────────────────────────

    def _compute_stability(self) -> float:
        """
        Spatial jitter of visible keypoints over the window.

        For each COCO keypoint with sufficient confidence, compute the standard
        deviation of its (x, y) positions across all frames in the window.
        The jitter magnitude is ``sqrt(std_x² + std_y²)`` in pixels.
        Mean jitter is penalised linearly to produce a 0–100 score.
        """
        if len(self.window) < 3:
            return 80.0  # not enough data, return generous default

        jitters: list[float] = []
        for kp_idx in range(NUM_COCO_KP):
            # Vectorised: gather positions only where confidence is high
            positions = np.array([
                f.landmarks[kp_idx, :2]
                for f in self.window
                if f.landmarks[kp_idx, 2] >= self.CONF_THRESHOLD
            ])
            if len(positions) < 2:
                continue
            std_x = float(positions[:, 0].std())
            std_y = float(positions[:, 1].std())
            jitters.append(np.sqrt(std_x ** 2 + std_y ** 2))

        if not jitters:
            return 50.0  # no confident keypoints at all

        mean_jitter = float(np.mean(jitters))
        return float(np.clip(100.0 - mean_jitter * self.JITTER_PENALTY, 0.0, 100.0))

    # ── Internal: consistency ─────────────────────────────────────────────────

    def _compute_consistency(self) -> float:
        """
        How consistently the pose score stays high across the window.

        Filters to frames matching the current (most recent) pose name,
        then penalises variance: ``mean_score - 0.5 * std_score``.
        """
        if len(self.window) < 1:
            return 0.0

        current_pose = self.window[-1].pose_name
        scores = np.array([
            f.pose_score
            for f in self.window
            if f.pose_name == current_pose
        ])

        if len(scores) < 3:
            return float(self.window[-1].pose_score)

        mean_s = float(scores.mean())
        std_s  = float(scores.std())
        consistency = mean_s - std_s * 0.5
        return float(np.clip(consistency, 0.0, 100.0))

    # ── Internal: transition detection ────────────────────────────────────────

    def _detect_transition(self) -> bool:
        """True if pose_name changed between the two most recent frames."""
        if len(self.window) < 2:
            return False
        a, b = self.window[-2], self.window[-1]
        return (
            a.pose_name != b.pose_name
            and a.pose_name not in ("Unknown", "No pose detected")
            and b.pose_name not in ("Unknown", "No pose detected")
        )

    # ── Internal: transition smoothness ───────────────────────────────────────

    def _compute_transition_smoothness(self) -> float:
        """
        Score the smoothness of the most-recently-detected transition.

        Looks at the velocity profile across the last TRANS_WINDOW frames.
        For each frame, computes the mean velocity magnitude of confident
        keypoints, then finds the peak value.

        **Velocity normalisation maths:**

        1.  Raw velocity ``v`` is in px/s — resolution-dependent.
        2.  We divide by ``diag * MAX_EXPECTED_V`` where:
              - ``diag = sqrt(frame_w² + frame_h²)`` converts px to a
                fraction-of-diagonal unit (range ≈ 0–1 for normal movement).
              - ``MAX_EXPECTED_V`` (default 0.08) is the empirically-tuned
                peak normalised velocity for a *smooth* transition.
            This produces ``v_norm`` ∈ [0, ∞) where 1.0 means "at the
            expected peak of a smooth transition".
        3.  Smoothness = ``clip(100 - v_norm × 80, 0, 100)``.
            A perfectly still frame scores 100.
            A frame at exactly MAX_EXPECTED_V scores 20.
            Anything faster than ~1.25× MAX_EXPECTED_V scores 0.
            The 80× scaling factor controls the sensitivity: it maps the
            0–1.25 normalised velocity range onto the full 0–100 score range.

        Adding ``1e-6`` to the denominator guards against division-by-zero
        when ``diag`` or ``MAX_EXPECTED_V`` would be zero (impossible in
        practice, but safe).
        """
        recent = list(self.window)[-self.TRANS_WINDOW:]
        peak_v = 0.0

        for f in recent:
            conf_mask = f.landmarks[:, 2] >= self.CONF_THRESHOLD
            if not conf_mask.any():
                continue
            # Mean velocity magnitude across confident keypoints
            mag = float(np.linalg.norm(f.velocity[conf_mask], axis=1).mean())
            peak_v = max(peak_v, mag)

        v_norm = peak_v / (self._diag * self.MAX_EXPECTED_V + 1e-6)
        smooth = float(np.clip(100.0 - v_norm * 80.0, 0.0, 100.0))
        return smooth

    # ── Internal: transition speed ────────────────────────────────────────────

    def _estimate_transition_speed_ms(self) -> int:
        """
        Estimate how many milliseconds it took the user to "land" the new pose
        after a transition was detected.

        Scans forward from the transition point (last frame where pose changed)
        and returns the time to first frame where ``pose_score >= LAND_SCORE_THR``.
        If the pose hasn't been landed yet within the window, returns a large value.
        """
        if len(self.window) < 2:
            return 9999

        # Transition happened at window[-1] (the new pose name)
        transition_ts = self.window[-1].timestamp
        new_pose      = self.window[-1].pose_name

        # Look through the window for the first "landed" frame
        for f in self.window:
            if f.timestamp >= transition_ts and f.pose_name == new_pose:
                if f.pose_score >= self.LAND_SCORE_THR:
                    elapsed_ms = int((f.timestamp - transition_ts) * 1000)
                    return max(0, elapsed_ms)

        return 9999  # not yet landed

    # ── Internal: peak velocity ───────────────────────────────────────────────

    def _compute_peak_velocity_norm(self) -> float:
        """
        Peak normalised velocity across the entire current window.
        Returns a dimensionless value in [0, ~1+] where 1.0 ≈ MAX_EXPECTED_V.
        """
        peak = 0.0
        for f in self.window:
            conf_mask = f.landmarks[:, 2] >= self.CONF_THRESHOLD
            if not conf_mask.any():
                continue
            mag = float(np.linalg.norm(f.velocity[conf_mask], axis=1).max())
            peak = max(peak, mag)

        return peak / (self._diag * self.MAX_EXPECTED_V + 1e-6)
