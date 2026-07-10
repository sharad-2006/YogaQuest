"""
wall_game_engine.py — State machine, scoring, and collision logic
for the Dynamic Pose Wall Challenge arcade game.

Works with the existing YOLO keypoint pipeline — no changes to core detection.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from poses_config import YOGA_POSES


# ─────────────────────────────────────────────────────────────────────────────
# Enums & constants
# ─────────────────────────────────────────────────────────────────────────────

class WallGameState(str, Enum):
    IDLE        = "idle"
    COUNTDOWN   = "countdown"
    PREPARE     = "prepare"       # announce pose, wall not moving yet
    WALL_MOVING = "wall_moving"   # wall animates toward user
    JUDGEMENT   = "judgement"      # 1-frame evaluation + freeze
    RESULT      = "result"        # show result, brief hold
    GAME_OVER   = "game_over"
    VICTORY     = "victory"       # survived all rounds


LIMB_SEGMENTS: Dict[str, Tuple[int, int]] = {
    "left_upper_arm":  (5,  7),
    "left_lower_arm":  (7,  9),
    "right_upper_arm": (6,  8),
    "right_lower_arm": (8, 10),
    "left_upper_leg":  (11, 13),
    "left_lower_leg":  (13, 15),
    "right_upper_leg": (12, 14),
    "right_lower_leg": (14, 16),
    "torso_left":      (5,  11),
    "torso_right":     (6,  12),
    "head":            (0,   5),
}


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty presets
# ─────────────────────────────────────────────────────────────────────────────

DIFFICULTY_PRESETS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "prep_time_s":           4.0,
        "travel_time_s":         4.0,
        "lives":                 5,
        "collision_threshold_px": 50,
        "total_rounds":          8,
        "pose_pool_size":        4,
        "xp_multiplier":         0.75,
        "wall_color_bgr":        (200, 140, 40),   # warm blue
        "label":                 "Easy",
        "description":           "Generous timing, forgiving collision zones, 5 lives.",
        "unlock_level":          1,
    },
    "normal": {
        "prep_time_s":           2.5,
        "travel_time_s":         2.5,
        "lives":                 3,
        "collision_threshold_px": 35,
        "total_rounds":          12,
        "pose_pool_size":        7,
        "xp_multiplier":         1.0,
        "wall_color_bgr":        (0, 140, 255),     # orange
        "label":                 "Normal",
        "description":           "Standard timing and scoring. 3 lives, all 7 poses.",
        "unlock_level":          3,
    },
    "hard": {
        "prep_time_s":           1.5,
        "travel_time_s":         1.8,
        "lives":                 2,
        "collision_threshold_px": 25,
        "total_rounds":          15,
        "pose_pool_size":        7,
        "xp_multiplier":         1.5,
        "wall_color_bgr":        (50, 50, 220),     # red
        "label":                 "Hard",
        "description":           "Tight timing, strict collision. Only 2 lives.",
        "unlock_level":          7,
    },
    "insane": {
        "prep_time_s":           1.0,
        "travel_time_s":         1.2,
        "lives":                 1,
        "collision_threshold_px": 18,
        "total_rounds":          20,
        "pose_pool_size":        7,
        "xp_multiplier":         2.5,
        "wall_color_bgr":        (180, 40, 180),    # purple
        "label":                 "Insane",
        "description":           "One life. Blazing speed. Only the Grand Guru survives.",
        "unlock_level":          12,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Round result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WallRoundResult:
    round_num:        int
    target_pose:      str
    pose_score:       float
    wall_round_score: float
    verdict:          str
    collision_map:    Dict[str, bool]
    n_collisions:     int
    timing_factor:    float
    xp_awarded:       int
    combo_at_time:    int
    timestamp:        float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Verdict thresholds & point values
# ─────────────────────────────────────────────────────────────────────────────

VERDICTS = [
    ("PERFECT FIT",  90, 150),
    ("GREAT FIT",    75, 100),
    ("GOOD FIT",     55,  60),
    ("CLOSE MISS",   35,  20),
    ("COLLISION",     0,   0),
]

COMBO_THRESHOLDS = [
    (10, 2.00),
    (7,  1.75),
    (4,  1.50),
    (2,  1.25),
    (0,  1.00),
]


def _verdict_for_score(score: float) -> Tuple[str, int]:
    """Return (verdict_label, base_points) for a given wall_round_score."""
    for label, threshold, pts in VERDICTS:
        if score >= threshold:
            return label, pts
    return "COLLISION", 0


def _combo_multiplier(combo: int) -> float:
    for threshold, mult in COMBO_THRESHOLDS:
        if combo >= threshold:
            return mult
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: normalised keypoints → pixel coords
# ─────────────────────────────────────────────────────────────────────────────

def landmarks_to_pixel(
    landmarks_norm: list,
    frame_w: int,
    frame_h: int,
    x_offset: int = 0,
) -> np.ndarray:
    """
    Convert normalised ideal keypoints ``[[nx, ny], ...]`` from poses_config
    to pixel coordinates.  *x_offset* shifts the silhouette horizontally
    (used for wall animation).

    Returns np.ndarray shape (17, 2) in pixel coordinates.
    """
    out = np.zeros((len(landmarks_norm), 2), dtype=float)
    for i, (nx, ny) in enumerate(landmarks_norm):
        out[i] = [nx * frame_w + x_offset, ny * frame_h]
    return out


def align_ideal_to_user(
    ideal_kps_px: np.ndarray,
    user_kps: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """
    Translate and scale the ideal silhouette to the player's visible body box.

    The pose cut-out remains game-authored, but judgement should not require the
    player to stand at an exact pixel location.  This keeps the YOLO pipeline
    untouched while making wall fits feel fair on different webcams and rooms.
    """
    if ideal_kps_px.size == 0 or user_kps.size == 0:
        return ideal_kps_px

    visible = user_kps[:, 2] >= 0.25
    if visible.sum() < 5:
        return ideal_kps_px

    user_pts = user_kps[visible, :2]
    ideal_pts = ideal_kps_px[visible, :2]

    user_min = user_pts.min(axis=0)
    user_max = user_pts.max(axis=0)
    ideal_min = ideal_pts.min(axis=0)
    ideal_max = ideal_pts.max(axis=0)

    user_size = np.maximum(user_max - user_min, 1.0)
    ideal_size = np.maximum(ideal_max - ideal_min, 1.0)

    user_center = (user_min + user_max) / 2.0
    ideal_center = (ideal_min + ideal_max) / 2.0

    scale = float(np.mean(user_size / ideal_size))
    scale = float(np.clip(scale, 0.55, 1.75))

    aligned = (ideal_kps_px - ideal_center) * scale + user_center
    aligned[:, 0] = np.clip(aligned[:, 0], 0, frame_w - 1)
    aligned[:, 1] = np.clip(aligned[:, 1], 0, frame_h - 1)
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# WallGameEngine
# ─────────────────────────────────────────────────────────────────────────────

class WallGameEngine:
    """
    State machine + scoring logic for the Dynamic Pose Wall Challenge.

    Usage::

        engine = WallGameEngine("normal", 640, 480)
        engine.start_game()

        # Per-frame in the Streamlit rerun loop:
        render_data = engine.tick(analysis, motion_metrics, delta_t)

        if engine.state in (WallGameState.GAME_OVER, WallGameState.VICTORY):
            results = engine.get_final_results()
    """

    # Timing constants
    COUNTDOWN_DURATION = 3.0        # 3-2-1-GO
    RESULT_DISPLAY_S   = 1.2        # how long to show verdict
    WALL_WIDTH_FRAC    = 0.12       # wall width = 12% of frame

    def __init__(
        self,
        difficulty: str = "normal",
        frame_w: int = 640,
        frame_h: int = 480,
        total_rounds: Optional[int] = None,
    ):
        preset = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["normal"])

        self.difficulty   = difficulty
        self.frame_w      = frame_w
        self.frame_h      = frame_h
        self.prep_time_s  = preset["prep_time_s"]
        self.travel_time_s = preset["travel_time_s"]
        self.max_lives    = preset["lives"]
        self.collision_thr = preset["collision_threshold_px"]
        self.total_rounds = total_rounds or preset["total_rounds"]
        self.pose_pool_size = preset["pose_pool_size"]
        self.xp_multiplier = preset["xp_multiplier"]
        self.wall_color    = preset["wall_color_bgr"]

        self.wall_width = int(frame_w * self.WALL_WIDTH_FRAC)

        # ── State ─────────────────────────────────────────────────────────
        self.state:        WallGameState = WallGameState.IDLE
        self.score:        int   = 0
        self.lives:        int   = self.max_lives
        self.combo:        int   = 0
        self.max_combo:    int   = 0
        self.current_round: int  = 0
        self.current_pose: Optional[str] = None

        # Timing
        self._state_start:  float = 0.0   # time.time() when current state began
        self._state_elapsed: float = 0.0  # elapsed time in current state
        self._wall_x:       int   = frame_w  # wall left-edge x-position
        self._game_start:   float = 0.0

        # Last judgement data
        self._last_collision_map: Dict[str, bool] = {}
        self._last_round_score:   float = 0.0
        self._last_verdict:       str   = ""
        self._last_xp:            int   = 0

        # Session tracking
        self.round_results: List[WallRoundResult] = []
        self.collision_heatmap: Dict[str, int] = {k: 0 for k in LIMB_SEGMENTS}
        self.perfect_fits:  int  = 0
        self.walls_survived: int = 0
        self.max_consecutive_survived: int = 0
        self._consecutive_survived:    int = 0
        self.flow_state_reached: bool = False
        self._total_pose_scores: List[float] = []

        # Pre-generated pose sequence
        self._pose_sequence: List[str] = []
        self._build_pose_sequence()

    # ── Pose sequence generation ──────────────────────────────────────────

    def _build_pose_sequence(self):
        all_poses = list(YOGA_POSES.keys())
        pool = all_poses[: self.pose_pool_size]

        seq = []
        prev = None
        for _ in range(self.total_rounds):
            choices = [p for p in pool if p != prev]
            if not choices:
                choices = pool
            pick = random.choice(choices)
            seq.append(pick)
            prev = pick
        self._pose_sequence = seq

    # ── Public API ────────────────────────────────────────────────────────

    def start_game(self):
        """Transition from IDLE → COUNTDOWN."""
        self._game_start = time.time()
        self._advance_state(WallGameState.COUNTDOWN)

    def tick(
        self,
        analysis: Dict[str, Any],
        motion_metrics: Dict[str, Any],
        delta_t: float,
    ) -> Dict[str, Any]:
        """
        Advance the game by one frame.  Returns a ``render_data`` dict
        consumed by ``wall_renderer.render_wall_frame()``.
        """
        now = time.time()
        self._state_elapsed = now - self._state_start

        # ── State machine ─────────────────────────────────────────────────
        if self.state == WallGameState.COUNTDOWN:
            self._tick_countdown()

        elif self.state == WallGameState.PREPARE:
            self._tick_prepare()

        elif self.state == WallGameState.WALL_MOVING:
            self._tick_wall_moving(analysis)

        elif self.state == WallGameState.JUDGEMENT:
            self._tick_judgement(analysis)

        elif self.state == WallGameState.RESULT:
            self._tick_result()

        # GAME_OVER / VICTORY / IDLE — no-ops (UI handles these)

        return self._build_render_data(analysis)

    def get_final_results(self) -> Dict[str, Any]:
        """
        Return a ``session_results``-compatible dict that can be passed
        to ``GameEngine.process_session()``.
        """
        # Build per-pose detail summaries (best attempt per pose)
        pose_best: Dict[str, Dict] = {}
        for rr in self.round_results:
            cfg = YOGA_POSES.get(rr.target_pose, {})
            entry = {
                "pose_name":      rr.target_pose,
                "english_name":   cfg.get("english", rr.target_pose.replace("_", " ")),
                "emoji":          cfg.get("emoji", "🧘"),
                "score":          rr.pose_score,
                "adjusted_score": rr.wall_round_score,
                "hold_time":      self.travel_time_s,
                "hold_bonus":     0.0,
                "stability":      80.0,
                "grade":          _grade(rr.wall_round_score),
                "suggestions":    [],
            }
            if rr.target_pose not in pose_best or rr.wall_round_score > pose_best[rr.target_pose]["adjusted_score"]:
                pose_best[rr.target_pose] = entry

        pose_details = list(pose_best.values())
        done_set = set(pose_best.keys())
        overall = float(np.mean([p["adjusted_score"] for p in pose_details])) if pose_details else 0.0
        duration = time.time() - self._game_start if self._game_start else 0.0

        # XP from wall game
        wall_xp = int(
            (self.score / max(1, self.total_rounds * 150)) * 80 * self.xp_multiplier
            + self.perfect_fits * 15
            + self.max_combo * 5
            + (50 if self.flow_state_reached else 0)
            + (25 if self.state == WallGameState.VICTORY else 0)
        )

        from poses_config import SURYA_CORE_POSES
        surya_pct = len(done_set & SURYA_CORE_POSES) / len(SURYA_CORE_POSES) if SURYA_CORE_POSES else 0.0

        return {
            "session_type":                   "wall_game",
            "difficulty":                     self.difficulty,
            "total_score":                    round(overall, 1),
            "duration":                       round(duration, 1),
            "poses_completed":                len(pose_details),
            "unique_poses":                   len(done_set),
            "pose_details":                   pose_details,
            "wall_xp_earned":                 wall_xp,
            "wall_rounds":                    self.total_rounds,
            "wall_survived":                  self.walls_survived,
            "wall_perfect_fits":              self.perfect_fits,
            "wall_max_consecutive_survived":  self.max_consecutive_survived,
            "wall_completed":                 self.state == WallGameState.VICTORY,
            "wall_difficulty":                self.difficulty,
            "max_combo":                      self.max_combo,
            "flow_state_reached":             self.flow_state_reached,
            "round_results":                  self.round_results,
            "collision_heatmap":              dict(self.collision_heatmap),
            "completed_poses_list":           list(done_set),
            "surya_namaskar_pct":             round(surya_pct, 2),
            "suggestions":                    [],
            "wall_score":                     self.score,
            # Motion metrics (populated if available)
            "max_stability":                  0.0,
            "min_transition_ms":              9999,
            "smooth_transitions_85plus":      0,
        }

    # ── State transitions ─────────────────────────────────────────────────

    def _advance_state(self, new_state: WallGameState):
        self.state = new_state
        self._state_start = time.time()
        self._state_elapsed = 0.0

        if new_state == WallGameState.PREPARE:
            if self.current_round < self.total_rounds:
                self.current_pose = self._pose_sequence[self.current_round]
            self._wall_x = self.frame_w

        elif new_state == WallGameState.WALL_MOVING:
            self._wall_x = self.frame_w

    # ── Per-state tick functions ──────────────────────────────────────────

    def _tick_countdown(self):
        if self._state_elapsed >= self.COUNTDOWN_DURATION:
            self.current_round = 0
            self._advance_state(WallGameState.PREPARE)

    def _tick_prepare(self):
        if self._state_elapsed >= self.prep_time_s:
            self._advance_state(WallGameState.WALL_MOVING)

    def _tick_wall_moving(self, analysis: Dict[str, Any]):
        # Animate wall from right edge to centre
        progress = min(1.0, self._state_elapsed / self.travel_time_s)
        self._wall_x = int(self.frame_w * (1.0 - progress))

        # Judgement triggers when the wall centre reaches roughly frame centre
        judgement_x = self.frame_w // 2 - self.wall_width // 2
        if self._wall_x <= judgement_x:
            self._advance_state(WallGameState.JUDGEMENT)

    def _tick_judgement(self, analysis: Dict[str, Any]):
        """Run collision detection and score this round."""
        self._run_judgement(analysis)
        self._advance_state(WallGameState.RESULT)

    def _tick_result(self):
        if self._state_elapsed >= self.RESULT_DISPLAY_S:
            # Check game-end conditions
            if self.lives <= 0:
                self._advance_state(WallGameState.GAME_OVER)
            elif self.current_round >= self.total_rounds:
                self._advance_state(WallGameState.VICTORY)
            else:
                self._advance_state(WallGameState.PREPARE)

    # ── Judgement & collision ──────────────────────────────────────────────

    def _run_judgement(self, analysis: Dict[str, Any]):
        """Evaluate the user's pose against the wall silhouette."""
        pose_score = analysis.get("score", 0.0)
        self._total_pose_scores.append(pose_score)

        # Get user's COCO keypoints (pixel coords + confidence)
        results_obj = analysis.get("results")
        user_kps = np.zeros((17, 3), dtype=float)
        if results_obj is not None:
            kp_px   = getattr(results_obj, "keypoints_px", None)
            kp_conf = getattr(results_obj, "keypoints_conf", None)
            if kp_px is not None and kp_conf is not None:
                user_kps[:, :2] = kp_px[:17]
                user_kps[:, 2]  = kp_conf[:17]

        # Get ideal keypoints in pixel coords (centred in frame)
        cfg = YOGA_POSES.get(self.current_pose, {})
        ideal_norm = cfg.get("ideal_keypoints_norm", [])
        if ideal_norm:
            ideal_px = landmarks_to_pixel(ideal_norm, self.frame_w, self.frame_h, x_offset=0)
            ideal_px = align_ideal_to_user(ideal_px, user_kps, self.frame_w, self.frame_h)
        else:
            ideal_px = np.zeros((17, 2), dtype=float)

        # Collision detection
        collision_map = self._collision_detection(user_kps, ideal_px, self.collision_thr)
        self._last_collision_map = collision_map

        # Update collision heatmap
        for limb, collided in collision_map.items():
            if collided:
                self.collision_heatmap[limb] = self.collision_heatmap.get(limb, 0) + 1

        # Timing factor: how close the user was to ideal arrival
        timing_factor = max(0.0, 1.0 - abs(self._state_elapsed) / max(0.01, self.travel_time_s))

        # Round score
        round_score = self._compute_round_score(collision_map, pose_score, timing_factor)
        self._last_round_score = round_score

        # Verdict
        verdict, base_pts = _verdict_for_score(round_score)
        self._last_verdict = verdict

        # Combo + multiplier
        if verdict in ("PERFECT FIT", "GREAT FIT"):
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self._consecutive_survived += 1
            self.max_consecutive_survived = max(
                self.max_consecutive_survived, self._consecutive_survived
            )
        else:
            self.combo = 0
            self._consecutive_survived = 0

        combo_mult = _combo_multiplier(self.combo)
        if combo_mult >= 2.0:
            self.flow_state_reached = True

        # Points
        pts = int(base_pts * combo_mult * self.xp_multiplier)
        self.score += pts
        self._last_xp = pts

        # Lives
        if verdict == "COLLISION":
            self.lives -= 1

        # Track perfect fits and walls survived
        if verdict == "PERFECT FIT":
            self.perfect_fits += 1
        if verdict != "COLLISION":
            self.walls_survived += 1

        # Record round result
        n_collisions = sum(1 for v in collision_map.values() if v)
        self.round_results.append(WallRoundResult(
            round_num=self.current_round + 1,
            target_pose=self.current_pose or "",
            pose_score=round(pose_score, 1),
            wall_round_score=round(round_score, 1),
            verdict=verdict,
            collision_map=dict(collision_map),
            n_collisions=n_collisions,
            timing_factor=round(timing_factor, 2),
            xp_awarded=pts,
            combo_at_time=self.combo,
        ))

        # Advance round counter
        self.current_round += 1

    def _collision_detection(
        self,
        user_kps: np.ndarray,
        ideal_kps_px: np.ndarray,
        threshold_px: int,
    ) -> Dict[str, bool]:
        """
        Per-limb collision: compare midpoint of user's limb segment
        to midpoint of ideal limb segment.  True = collision (too far apart).
        """
        results: Dict[str, bool] = {}
        if user_kps.size == 0 or int((user_kps[:, 2] >= 0.25).sum()) < 6:
            return {limb: True for limb in LIMB_SEGMENTS}

        for limb, (i, j) in LIMB_SEGMENTS.items():
            # Missing keypoints count as a hit: the wall challenge needs the
            # player's body visible enough to judge the cut-out.
            if user_kps[i, 2] < 0.25 or user_kps[j, 2] < 0.25:
                results[limb] = True
                continue
            user_mid  = (user_kps[i, :2] + user_kps[j, :2]) / 2
            ideal_mid = (ideal_kps_px[i] + ideal_kps_px[j]) / 2
            dist = float(np.linalg.norm(user_mid - ideal_mid))
            results[limb] = (dist > threshold_px)
        return results

    def _compute_round_score(
        self,
        collision_results: Dict[str, bool],
        pose_score: float,
        timing_factor: float,
    ) -> float:
        """
        wall_round_score = limb_accuracy * 60 + pose_score * 0.30 + timing * 10
        """
        n_total = len(collision_results) or 1
        n_clear = sum(1 for v in collision_results.values() if not v)
        limb_accuracy = n_clear / n_total

        raw = limb_accuracy * 60 + pose_score * 0.30 + timing_factor * 10
        return float(np.clip(raw, 0, 100))

    # ── Render data builder ───────────────────────────────────────────────

    def _build_render_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build the dict consumed by wall_renderer."""
        # Compute ideal keypoints in pixel space for current pose
        target_kps_px = np.zeros((17, 2), dtype=float)
        if self.current_pose:
            cfg = YOGA_POSES.get(self.current_pose, {})
            ideal_norm = cfg.get("ideal_keypoints_norm", [])
            if ideal_norm:
                # During wall movement, offset keypoints to track wall position
                if self.state == WallGameState.WALL_MOVING:
                    # Silhouette is centred within the wall
                    x_offset = self._wall_x - self.frame_w // 2
                    target_kps_px = landmarks_to_pixel(
                        ideal_norm, self.frame_w, self.frame_h, x_offset
                    )
                else:
                    target_kps_px = landmarks_to_pixel(
                        ideal_norm, self.frame_w, self.frame_h, 0
                    )

        # Countdown value (3, 2, 1, 0)
        countdown_val = 0
        if self.state == WallGameState.COUNTDOWN:
            remaining = self.COUNTDOWN_DURATION - self._state_elapsed
            countdown_val = max(0, int(remaining) + 1)
            if countdown_val > 3:
                countdown_val = 3

        # Prep timer
        prep_timer = 0.0
        if self.state == WallGameState.PREPARE:
            prep_timer = max(0.0, self.prep_time_s - self._state_elapsed)

        pose_emoji = ""
        pose_english = ""
        if self.current_pose:
            cfg = YOGA_POSES.get(self.current_pose, {})
            pose_emoji = cfg.get("emoji", "🧘")
            pose_english = cfg.get("english", self.current_pose.replace("_", " "))

        return {
            "state":           self.state.value,
            "wall_x":          self._wall_x,
            "wall_width":      self.wall_width,
            "wall_color":      self.wall_color,
            "target_pose":     self.current_pose or "",
            "target_kps_px":   target_kps_px,
            "pose_emoji":      pose_emoji,
            "pose_english":    pose_english,
            "score":           self.score,
            "lives":           self.lives,
            "max_lives":       self.max_lives,
            "combo":           self.combo,
            "combo_mult":      _combo_multiplier(self.combo),
            "round":           self.current_round,
            "total_rounds":    self.total_rounds,
            "prep_timer":      prep_timer,
            "verdict":         self._last_verdict if self.state == WallGameState.RESULT else "",
            "collision_map":   self._last_collision_map if self.state in (
                WallGameState.RESULT, WallGameState.JUDGEMENT
            ) else {},
            "round_score":     self._last_round_score if self.state == WallGameState.RESULT else 0.0,
            "xp_this_round":   self._last_xp if self.state == WallGameState.RESULT else 0,
            "countdown_val":   countdown_val,
            "show_silhouette": self.state in (
                WallGameState.PREPARE,
                WallGameState.WALL_MOVING,
                WallGameState.JUDGEMENT,
                WallGameState.RESULT,
            ),
            "difficulty":      self.difficulty,
            "flow_state":      self.flow_state_reached,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grade(score: float) -> str:
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"
