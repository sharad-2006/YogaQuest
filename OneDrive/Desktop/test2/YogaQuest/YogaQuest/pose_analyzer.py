"""
pose_analyzer.py — YOLO-based pose detection, angle scoring,
                   frame annotation, and full-session analysis.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any

from poses_config import YOGA_POSES, SURYA_CORE_POSES
from pose_detector import PoseDetectionResult, draw_skeleton


# ─────────────────────────────────────────────────────────────────────────────
# Low-level geometry
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Angle (degrees) at vertex p2 formed by rays p2→p1 and p2→p3.
    Uses 2-D (x, y) projection — reliable for front / side camera views.
    Returns a value in [0, 180].
    """
    a = np.array(p1[:2], dtype=float)
    b = np.array(p2[:2], dtype=float)
    c = np.array(p3[:2], dtype=float)

    ba = a - b
    bc = c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-7 or n_bc < 1e-7:
        return 0.0

    cos_a = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def angle_score(measured: float, ideal: float, tolerance: float) -> float:
    """
    100 when |measured - ideal| ≤ tolerance.
    Drops at 2.5 pts / degree beyond tolerance, floor 0.
    """
    dev = abs(measured - ideal)
    if dev <= tolerance:
        return 100.0
    return max(0.0, 100.0 - (dev - tolerance) * 2.5)


def score_color_bgr(score: float) -> Tuple[int, int, int]:
    """BGR colour coding for overlay text."""
    if score >= 80:  return (0, 220, 60)     # green
    if score >= 60:  return (0, 165, 255)    # orange
    if score >= 40:  return (0, 80, 255)     # red
    return (130, 130, 130)                    # grey


def get_grade(score: float) -> str:
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


# ─────────────────────────────────────────────────────────────────────────────
# Landmark extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks(results: Optional[PoseDetectionResult]) -> Optional[np.ndarray]:
    """
    Returns shape (33, 4) array: [x, y, z, visibility] in normalised coords,
    or None if no pose was detected.
    """
    if results is None or results.landmarks is None:
        return None
    return results.landmarks


# ─────────────────────────────────────────────────────────────────────────────
# Per-pose scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_pose(pose_name: str, landmarks: np.ndarray) -> Dict[str, Any]:
    """
    Score a named yoga pose against the current landmarks.
    Returns a dict with total_score, grade, and per-check details.
    """
    cfg = YOGA_POSES.get(pose_name)
    if cfg is None:
        return {"pose_name": pose_name, "total_score": 0.0,
                "grade": "F", "checks": []}

    total_w   = 0.0
    total_ws  = 0.0
    checks_out = []

    for chk in cfg["angle_checks"]:
        idx = chk["points"]
        p1, p2, p3 = landmarks[idx[0]], landmarks[idx[1]], landmarks[idx[2]]

        # Skip checks where any landmark is poorly visible
        min_vis = min(p1[3], p2[3], p3[3])
        if min_vis < 0.30:
            continue

        measured = calculate_angle(p1, p2, p3)
        cscore   = angle_score(measured, chk["ideal"], chk["tolerance"])
        w        = chk["weight"]

        dev = measured - chk["ideal"]
        if abs(dev) <= chk["tolerance"]:
            feedback, level = "✅ Perfect alignment", "good"
        elif abs(dev) <= chk["tolerance"] * 2:
            dir_str = "too large" if dev > 0 else "too small"
            feedback = f"⚠️ Angle {dir_str} by {abs(dev):.0f}°"
            level    = "warning"
        else:
            feedback = f"❌ {chk['tip']}"
            level    = "error"

        checks_out.append({
            "name":     chk["name"],
            "measured": round(measured, 1),
            "ideal":    chk["ideal"],
            "score":    round(cscore, 1),
            "weight":   w,
            "feedback": feedback,
            "level":    level,
            "tip":      chk["tip"],
        })
        total_ws += cscore * w
        total_w  += w

    final = (total_ws / total_w) if total_w > 0 else 0.0

    return {
        "pose_name":   pose_name,
        "total_score": round(final, 1),
        "grade":       get_grade(final),
        "checks":      checks_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pose auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_current_pose(landmarks: np.ndarray,
                        min_score: float = 42.0) -> Tuple[str, float]:
    """
    Score all known poses and return (best_pose_name, best_score).
    Returns ("Unknown", score) if best score is below min_score.
    """
    best_name  = "Unknown"
    best_score = 0.0
    for pname in YOGA_POSES:
        res = score_pose(pname, landmarks)
        s   = res["total_score"]
        if s > best_score:
            best_score = s
            best_name  = pname
    if best_score < min_score:
        return "Unknown", best_score
    return best_name, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Frame analysis (processes one frame end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_frame(frame: np.ndarray,
                  pose_model,
                  target_pose: Optional[str] = None) -> Dict[str, Any]:
    """
    Run YOLO pose detection on *frame* (BGR) and return a serialisable analysis dict.

    Keys:
        pose_detected : bool
        pose_name     : str
        score         : float 0-100
        grade         : str
        checks        : list[dict]
        landmarks     : np.ndarray | None  (33×4, normalised)
        results       : PoseDetectionResult (not stored long-term)
    """
    results = pose_model.process(frame)

    base = {
        "pose_detected": False,
        "pose_name":     "No pose detected",
        "score":         0.0,
        "grade":         "F",
        "checks":        [],
        "landmarks":     None,
        "results":       results,
    }

    landmarks = extract_landmarks(results)
    if landmarks is None:
        return base

    base["landmarks"] = landmarks
    base["pose_detected"] = True

    if target_pose:
        scoring = score_pose(target_pose, landmarks)
        base.update({
            "pose_name": target_pose,
            "score":     scoring["total_score"],
            "grade":     scoring["grade"],
            "checks":    scoring["checks"],
        })
    else:
        detected, score = detect_current_pose(landmarks)
        if detected != "Unknown":
            scoring = score_pose(detected, landmarks)
            base.update({
                "pose_name": detected,
                "score":     score,
                "grade":     scoring["grade"],
                "checks":    scoring["checks"],
            })
        else:
            base["pose_name"] = "Unknown"
            base["score"]     = score

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotation (draws skeleton + HUD on a copy of frame)
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray,
                   analysis: Dict[str, Any],
                   show_skeleton: bool = True) -> np.ndarray:
    """
    Return an annotated BGR copy of *frame* with:
      • skeleton overlay (coloured by score)
      • top-left info panel (pose name, score bar, grade)
      • bottom feedback strip (per-check messages)
    """
    out = frame.copy()
    h, w = out.shape[:2]

    results  = analysis.get("results")
    score    = analysis.get("score", 0.0)
    detected = analysis.get("pose_detected", False)

    # ── Skeleton ────────────────────────────────────────────────────────────
    if show_skeleton and isinstance(results, PoseDetectionResult) and results.landmarks is not None:
        sc = score_color_bgr(score)
        draw_skeleton(out, results, sc, thickness=2)

    # ── Top info panel ───────────────────────────────────────────────────────
    panel_h = 110
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (360, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    if detected:
        pname   = analysis.get("pose_name", "Unknown")
        cfg     = YOGA_POSES.get(pname, {})
        emoji   = cfg.get("emoji", "")
        english = cfg.get("english", pname.replace("_", " "))
        grade   = analysis.get("grade", "?")

        title = f"{emoji}  {english}"
        cv2.putText(out, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

        sc = score_color_bgr(score)
        cv2.putText(out, f"Score: {score:.0f} / 100", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, sc, 2, cv2.LINE_AA)

        # Score bar
        bar_x, bar_y, bar_w, bar_h2 = 10, 68, 310, 12
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h2),
                      (60, 60, 60), -1)
        fill = int(bar_w * score / 100)
        if fill > 0:
            cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h2), sc, -1)

        grade_colors = {"S": (0,215,255), "A": (0,220,60), "B": (0,180,40),
                        "C": (0,165,255), "D": (0,100,255), "F": (0,50,200)}
        gc = grade_colors.get(grade, (200, 200, 200))
        cv2.putText(out, f"Grade: {grade}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, gc, 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "No pose detected", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "Step back so your full body is visible", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Bottom feedback strip ────────────────────────────────────────────────
    checks = analysis.get("checks", [])
    if checks:
        n = min(len(checks), 5)
        strip_h = n * 22 + 12
        strip_y = h - strip_h
        ov2 = out.copy()
        cv2.rectangle(ov2, (0, strip_y), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(ov2, 0.60, out, 0.40, 0, out)

        level_colors = {"good": (0, 200, 60), "warning": (0, 155, 255), "error": (50, 50, 255)}
        for i, chk in enumerate(checks[:5]):
            y   = strip_y + 18 + i * 22
            col = level_colors.get(chk.get("level", "good"), (200, 200, 200))
            cv2.putText(out, chk.get("feedback", ""), (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Session Analyser
# ─────────────────────────────────────────────────────────────────────────────

class SessionAnalyzer:
    """
    Tracks pose detections over time during a session.
    Call add_frame() for each analysed frame, finish_session() at the end.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.frames_data:     List[Dict] = []
        self.completed_poses: List[Dict] = []
        self.current_pose:    Optional[str] = None
        self.current_start:   float = 0.0
        self.current_scores:  List[float] = []
        self.current_checks:  List[Dict] = []
        self.session_start:   float = 0.0

    def start_session(self):
        self._reset()
        self.session_start = time.time()

    def add_frame(self, analysis: Dict, timestamp: float):
        self.frames_data.append({"ts": timestamp, "analysis": analysis})

        pose_name = analysis.get("pose_name", "Unknown")
        score     = analysis.get("score", 0.0)

        if pose_name not in ("Unknown", "No pose detected") and score >= 45:
            if pose_name != self.current_pose:
                self._commit_current(timestamp)
                self.current_pose   = pose_name
                self.current_start  = timestamp
                self.current_scores = [score]
                self.current_checks = analysis.get("checks", [])
            else:
                self.current_scores.append(score)
                # Keep best-quality checks
                if score > (self.current_scores[-2] if len(self.current_scores) > 1 else 0):
                    self.current_checks = analysis.get("checks", [])
        elif self.current_pose:
            self._commit_current(timestamp)
            self.current_pose = None

    def _commit_current(self, end_ts: float):
        if not self.current_pose or not self.current_scores:
            return
        hold = end_ts - self.current_start
        if hold < 1.0:
            return
        avg  = float(np.mean(self.current_scores))
        peak = float(np.max(self.current_scores))
        self.completed_poses.append({
            "pose_name":    self.current_pose,
            "hold_time":    round(hold, 2),
            "avg_score":    round(avg,  1),
            "peak_score":   round(peak, 1),
            "start_ts":     round(self.current_start, 2),
            "best_checks":  self.current_checks,
        })

    def finish_session(self) -> Dict[str, Any]:
        now = time.time()
        self._commit_current(now - self.session_start)
        duration = now - self.session_start
        return self._compile(duration)

    # ── Result compilation ───────────────────────────────────────────────────
    def _compile(self, duration: float) -> Dict[str, Any]:
        if not self.completed_poses:
            return {
                "total_score": 0.0, "duration": round(duration, 1),
                "poses_completed": 0, "unique_poses": 0,
                "pose_details": [], "suggestions": [],
                "completed_poses_list": [],
                "surya_namaskar_pct": 0.0,
                "max_combo": 0,
                "session_type": "practice",
            }

        # Keep only best attempt per pose
        best: Dict[str, Dict] = {}
        for p in self.completed_poses:
            n = p["pose_name"]
            if n not in best or p["avg_score"] > best[n]["avg_score"]:
                best[n] = p

        pose_details = []
        score_sum = 0.0
        all_tips: List[str] = []
        combo = 0
        max_combo = 0

        for pose_name, pd in best.items():
            avg   = pd["avg_score"]
            hold  = pd["hold_time"]
            cfg   = YOGA_POSES.get(pose_name, {})
            target_hold = cfg.get("hold_time", 5)

            # Hold bonus: up to +10 pts for holding target duration
            hold_bonus = min(10.0, (hold / target_hold) * 10.0)
            adj_score  = min(100.0, avg + hold_bonus)

            # Stability measure: std-dev of scores (lower = more stable)
            all_s = [fr["analysis"]["score"]
                     for fr in self.frames_data
                     if fr["analysis"].get("pose_name") == pose_name
                     and fr["analysis"].get("score", 0) >= 45]
            stability = max(0.0, 100.0 - float(np.std(all_s))) if len(all_s) > 1 else 80.0

            # Combo tracking
            if adj_score >= 70:
                combo += 1
                max_combo = max(max_combo, combo)
            else:
                combo = 0

            # Suggestions from failed checks
            suggestions = []
            for chk in pd.get("best_checks", []):
                if chk.get("level") in ("warning", "error"):
                    suggestions.append(chk["tip"])
            if not suggestions and cfg.get("tips"):
                suggestions = cfg["tips"][:2]
            all_tips.extend(suggestions)

            pose_details.append({
                "pose_name":    pose_name,
                "english_name": cfg.get("english", pose_name.replace("_", " ")),
                "emoji":        cfg.get("emoji", "🧘"),
                "score":        avg,
                "adjusted_score": round(adj_score, 1),
                "hold_time":    hold,
                "hold_bonus":   round(hold_bonus, 1),
                "stability":    round(stability, 1),
                "grade":        get_grade(adj_score),
                "suggestions":  suggestions[:3],
            })
            score_sum += adj_score

        overall = round(score_sum / len(pose_details), 1) if pose_details else 0.0
        done_set = set(best.keys())
        surya_pct = len(done_set & SURYA_CORE_POSES) / len(SURYA_CORE_POSES)

        return {
            "total_score":         overall,
            "duration":            round(duration, 1),
            "poses_completed":     len(pose_details),
            "unique_poses":        len(best),
            "pose_details":        pose_details,
            "suggestions":         list(dict.fromkeys(all_tips))[:6],
            "completed_poses_list": list(done_set),
            "surya_namaskar_pct":  round(surya_pct, 2),
            "max_combo":           max_combo,
            "session_type":        "practice",
        }
