"""
wall_renderer.py — Pure OpenCV rendering for the Dynamic Pose Wall Challenge.

Draws the wall, silhouette cut-out, HUD, collision feedback, and verdict
banners onto a BGR webcam frame.  All functions are stateless — they take
a frame + render_data dict and return the annotated frame.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np

from pose_detector import COCO_CONNECTIONS, draw_skeleton


# ─────────────────────────────────────────────────────────────────────────────
# Colour constants (BGR)
# ─────────────────────────────────────────────────────────────────────────────

WALL_COLOR_DEFAULT = (30, 30, 30)         # dark charcoal
CUT_OUT_COLOR      = (255, 255, 255)      # white
IDEAL_SKEL_COLOR   = (255, 230, 0)        # cyan (BGR)
COLLISION_COLOR    = (0,   0,   255)      # red
CLEAR_COLOR        = (80, 220, 0)         # green
HUD_BG             = (18, 10, 10)         # near-black
VERDICT_COLORS     = {
    "PERFECT FIT": (0,   215, 255),       # gold
    "GREAT FIT":   (60,  220, 0),         # green
    "GOOD FIT":    (0,   165, 255),       # orange
    "CLOSE MISS":  (0,   80,  255),       # red-ish
    "COLLISION":   (50,  50,  255),        # bright red
}
LIFE_COLOR         = (50, 50, 255)        # red heart

# COCO connections for drawing ideal skeleton
SKEL_CONNECTIONS = COCO_CONNECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Master renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_wall_frame(
    frame: np.ndarray,
    render_data: Dict[str, Any],
    analysis: Dict[str, Any],
) -> np.ndarray:
    """
    Master renderer — calls sub-renderers based on current game state.
    Returns the annotated BGR frame (a copy — does not mutate input).
    """
    out = frame.copy()
    state = render_data.get("state", "idle")

    # ── Always: draw user's skeleton ──────────────────────────────────────
    results_obj = analysis.get("results")
    if results_obj is not None and hasattr(results_obj, "landmarks") and results_obj.landmarks is not None:
        score = analysis.get("score", 0.0)
        skel_color = _score_color_bgr(score)
        draw_skeleton(out, results_obj, skel_color, thickness=2)

    # ── State-specific rendering ──────────────────────────────────────────
    if state == "countdown":
        out = draw_countdown(out, render_data.get("countdown_val", 3))

    elif state == "prepare":
        out = draw_prep_announcement(
            out,
            render_data.get("pose_english", ""),
            render_data.get("pose_emoji", "🧘"),
            render_data.get("prep_timer", 0.0),
        )
        # Show static silhouette preview at centre
        if render_data.get("show_silhouette"):
            target_kps = render_data.get("target_kps_px")
            if target_kps is not None:
                out = draw_ideal_skeleton(out, target_kps, IDEAL_SKEL_COLOR, alpha=0.4)

    elif state == "wall_moving":
        wall_x = render_data.get("wall_x", 0)
        wall_w = render_data.get("wall_width", 80)
        wall_color = render_data.get("wall_color", WALL_COLOR_DEFAULT)
        out = draw_wall_overlay(out, wall_x, wall_w, wall_color, alpha=0.72)

        if render_data.get("show_silhouette"):
            target_kps = render_data.get("target_kps_px")
            if target_kps is not None:
                out = draw_pose_silhouette(out, target_kps, wall_x, wall_w)
                out = draw_ideal_skeleton(out, target_kps, IDEAL_SKEL_COLOR, alpha=0.9)

    elif state == "judgement":
        # Flash white overlay briefly
        flash = out.copy()
        cv2.rectangle(flash, (0, 0), (out.shape[1], out.shape[0]), (255, 255, 255), -1)
        cv2.addWeighted(flash, 0.15, out, 0.85, 0, out)

    elif state == "result":
        # Show collision feedback on user's limbs
        collision_map = render_data.get("collision_map", {})
        if collision_map:
            out = draw_collision_feedback(out, analysis, collision_map)
        # Verdict banner
        verdict = render_data.get("verdict", "")
        if verdict:
            round_score = render_data.get("round_score", 0.0)
            xp = render_data.get("xp_this_round", 0)
            out = draw_verdict_banner(out, verdict, round_score, xp)

    elif state in ("game_over", "victory"):
        out = draw_game_end(out, state, render_data)

    # ── Always: draw HUD ──────────────────────────────────────────────────
    if state not in ("idle", "countdown"):
        out = draw_hud(out, render_data)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Wall overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_wall_overlay(
    frame: np.ndarray,
    wall_x: int,
    wall_w: int,
    wall_color: Tuple[int, int, int] = WALL_COLOR_DEFAULT,
    alpha: float = 0.72,
) -> np.ndarray:
    """Draw a semi-transparent vertical wall band."""
    out = frame.copy()
    h = frame.shape[0]

    overlay = frame.copy()
    x1 = max(0, wall_x)
    x2 = min(frame.shape[1], wall_x + wall_w)
    if x2 > x1:
        cv2.rectangle(overlay, (x1, 0), (x2, h), wall_color, -1)
        # Bright edge lines
        cv2.line(overlay, (x1, 0), (x1, h), _brighten(wall_color, 80), 3)
        cv2.line(overlay, (x2, 0), (x2, h), _brighten(wall_color, 80), 3)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette cut-out (white convex hull)
# ─────────────────────────────────────────────────────────────────────────────

# Body outline indices for hull (COCO-17)
BODY_OUTLINE_INDICES = [0, 5, 7, 9, 7, 5, 11, 13, 15, 16, 14, 12, 6, 8, 10, 8, 6, 0]

def draw_pose_silhouette(
    frame: np.ndarray,
    target_kps_px: np.ndarray,
    wall_x: int,
    wall_w: int,
    margin: int = 25,
) -> np.ndarray:
    """Draw a white convex-hull 'hole' in the wall representing the pose shape."""
    out = frame.copy()

    # Collect body outline points
    pts = []
    for idx in BODY_OUTLINE_INDICES:
        if idx < len(target_kps_px):
            x, y = target_kps_px[idx]
            if x > 0 and y > 0:
                pts.append([int(x), int(y)])

    if len(pts) < 4:
        return out

    pts_arr = np.array(pts, dtype=np.int32)

    # Compute convex hull
    try:
        hull = cv2.convexHull(pts_arr)
    except cv2.error:
        return out

    # Expand hull by margin for forgiveness
    if margin > 0:
        center = hull.mean(axis=0).astype(int)
        expanded = []
        for pt in hull:
            direction = pt[0] - center[0]
            norm = np.linalg.norm(direction.astype(float))
            if norm > 0:
                unit = direction.astype(float) / norm
                new_pt = pt[0] + (unit * margin).astype(int)
                expanded.append(new_pt)
            else:
                expanded.append(pt[0])
        hull = np.array(expanded, dtype=np.int32).reshape(-1, 1, 2)

    # Draw the cut-out: fill with semi-transparent white
    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Only apply within the wall band
    wall_mask = np.zeros_like(mask)
    x1 = max(0, wall_x)
    x2 = min(out.shape[1], wall_x + wall_w)
    wall_mask[:, x1:x2] = 255
    combined = cv2.bitwise_and(mask, wall_mask)

    white_overlay = out.copy()
    white_overlay[combined > 0] = CUT_OUT_COLOR
    cv2.addWeighted(white_overlay, 0.35, out, 0.65, 0, out)

    # Outline the hull
    cv2.polylines(out, [hull], True, IDEAL_SKEL_COLOR, 2, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Ideal skeleton
# ─────────────────────────────────────────────────────────────────────────────

def draw_ideal_skeleton(
    frame: np.ndarray,
    target_kps_px: np.ndarray,
    color: Tuple[int, int, int] = IDEAL_SKEL_COLOR,
    alpha: float = 0.9,
) -> np.ndarray:
    """Draw the ideal COCO skeleton (lines + joints) for the target pose."""
    out = frame.copy()
    overlay = frame.copy()

    for i, j in SKEL_CONNECTIONS:
        if i < len(target_kps_px) and j < len(target_kps_px):
            pt1 = (int(target_kps_px[i][0]), int(target_kps_px[i][1]))
            pt2 = (int(target_kps_px[j][0]), int(target_kps_px[j][1]))
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(overlay, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw joints
    for i in range(min(17, len(target_kps_px))):
        x, y = int(target_kps_px[i][0]), int(target_kps_px[i][1])
        if x > 0 and y > 0:
            cv2.circle(overlay, (x, y), 5, color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Collision feedback (red / green limb highlighting)
# ─────────────────────────────────────────────────────────────────────────────

# Same segment indices used in wall_game_engine.py
_LIMB_SEGMENTS = {
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

def draw_collision_feedback(
    frame: np.ndarray,
    analysis: Dict[str, Any],
    collision_map: Dict[str, bool],
) -> np.ndarray:
    """Highlight user's limbs red (collision) or green (clear)."""
    out = frame.copy()

    results_obj = analysis.get("results")
    if results_obj is None:
        return out
    kp_px   = getattr(results_obj, "keypoints_px", None)
    kp_conf = getattr(results_obj, "keypoints_conf", None)
    if kp_px is None or kp_conf is None:
        return out

    for limb, collided in collision_map.items():
        seg = _LIMB_SEGMENTS.get(limb)
        if seg is None:
            continue
        i, j = seg
        if i >= len(kp_conf) or j >= len(kp_conf):
            continue
        if kp_conf[i] < 0.25 or kp_conf[j] < 0.25:
            continue

        pt1 = (int(kp_px[i][0]), int(kp_px[i][1]))
        pt2 = (int(kp_px[j][0]), int(kp_px[j][1]))

        color = COLLISION_COLOR if collided else CLEAR_COLOR
        thickness = 5 if collided else 4
        cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

        if collided:
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.circle(out, mid, 8, COLLISION_COLOR, -1)
            cv2.putText(out, "X", (mid[0] - 6, mid[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# HUD (heads-up display)
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, render_data: Dict[str, Any]) -> np.ndarray:
    """
    Draw the game HUD panels with better positioning to avoid overlap:
      Top-left:  Score, lives, combo
      Top-right: Target pose, timer/progress
      Bottom-left: Difficulty badge
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # ── Top-left panel (reduced size to prevent overlap) ──────────────────
    panel_w, panel_h = min(280, w//2 - 10), 60  # Responsive width, smaller height
    overlay = out.copy()
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    score = render_data.get("score", 0)
    lives = render_data.get("lives", 0)
    max_lives = render_data.get("max_lives", 3)
    combo = render_data.get("combo", 0)
    combo_mult = render_data.get("combo_mult", 1.0)
    rnd = render_data.get("round", 0)
    total = render_data.get("total_rounds", 12)

    # Score (smaller font)
    cv2.putText(out, f"SCORE: {score:,}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2, cv2.LINE_AA)

    # Lives (hearts) - positioned better
    hearts = "".join(["♥" for _ in range(lives)] + ["♡" for _ in range(max_lives - lives)])
    lives_text = f"LIVES: {hearts}"
    cv2.putText(out, lives_text, (170, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, LIFE_COLOR, 1, cv2.LINE_AA)

    # Combo + round (second line, smaller)
    combo_color = (0, 215, 255) if combo_mult >= 2.0 else (0, 200, 60) if combo >= 2 else (200, 200, 200)
    cv2.putText(out, f"Combo: x{combo_mult:.1f}", (12, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, combo_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"Round: {min(rnd + 1, total)}/{total}", (150, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Flow state banner (smaller)
    if render_data.get("flow_state"):
        cv2.putText(out, "FLOW!", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 215, 255), 1, cv2.LINE_AA)

    # ── Top-right panel (better positioning) ──────────────────────────────
    state = render_data.get("state", "")
    if state in ("prepare", "wall_moving"):
        tr_w, tr_h = min(260, w//2 - 10), 50  # Responsive, smaller
        tr_x = max(w - tr_w - 5, panel_w + 10)  # Don't overlap with left panel
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (tr_x, 5), (w - 5, tr_h + 5), HUD_BG, -1)
        cv2.addWeighted(overlay2, 0.75, out, 0.25, 0, out)

        pose_name = render_data.get("pose_english", "")
        pose_emoji = render_data.get("pose_emoji", "")
        
        # Truncate long pose names
        if len(pose_name) > 12:
            pose_name = pose_name[:12] + "..."
            
        cv2.putText(out, f"Target: {pose_name}", (tr_x + 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 0), 1, cv2.LINE_AA)

        if state == "prepare":
            timer_val = render_data.get("prep_timer", 0.0)
            cv2.putText(out, f"Wall in: {timer_val:.1f}s", (tr_x + 8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            # Progress bar (smaller)
            bar_x = tr_x + 120
            bar_w_max = min(100, w - bar_x - 10)
            if bar_w_max > 20:  # Only draw if there's space
                progress = max(0.0, 1.0 - timer_val / 4.0)
                bar_fill = int(bar_w_max * progress)
                cv2.rectangle(out, (bar_x, 32), (bar_x + bar_w_max, 40), (60, 60, 60), -1)
                if bar_fill > 0:
                    cv2.rectangle(out, (bar_x, 32), (bar_x + bar_fill, 40), (0, 165, 255), -1)

    # ── Bottom-left badge (better positioning) ────────────────────────────
    diff = render_data.get("difficulty", "normal").upper()
    badge_text = f"POSE WALL | {diff}"
    cv2.putText(out, badge_text, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Verdict banner
# ─────────────────────────────────────────────────────────────────────────────

def draw_verdict_banner(
    frame: np.ndarray,
    verdict: str,
    round_score: float,
    xp: int,
) -> np.ndarray:
    """Draw a centered verdict banner during RESULT state with better positioning."""
    out = frame.copy()
    h, w = out.shape[:2]

    color = VERDICT_COLORS.get(verdict, (200, 200, 200))

    # Banner background (smaller and better positioned)
    bh = 80  # Reduced height
    by = h // 2 - bh // 2 + 20  # Moved down to avoid top HUD
    overlay = out.copy()
    cv2.rectangle(overlay, (15, by), (w - 15, by + bh), HUD_BG, -1)  # Add margins
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

    # Border lines
    cv2.line(out, (15, by), (w - 15, by), color, 2)
    cv2.line(out, (15, by + bh), (w - 15, by + bh), color, 2)

    # Verdict text (smaller if too long)
    font_scale = 1.0
    if len(verdict) > 12:
        font_scale = 0.8
    
    text_size = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
    tx = (w - text_size[0]) // 2
    cv2.putText(out, verdict, (tx, by + 35),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3, cv2.LINE_AA)

    # Score + XP (smaller text)
    info = f"Score: {round_score:.0f}   +{xp} pts"
    info_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    ix = (w - info_size[0]) // 2
    cv2.putText(out, info, (ix, by + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 2, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Countdown (3-2-1-GO!)
# ─────────────────────────────────────────────────────────────────────────────

def draw_countdown(frame: np.ndarray, value: int) -> np.ndarray:
    """Draw large centred countdown number."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Dim the background
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, out, 0.50, 0, out)

    text = str(value) if value > 0 else "GO!"
    color = (0, 215, 255) if value > 0 else (60, 220, 0)

    scale = 4.0 if value > 0 else 3.0
    thickness = 8 if value > 0 else 6

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    tx = (w - text_size[0]) // 2
    ty = (h + text_size[1]) // 2

    # Shadow
    cv2.putText(out, text, (tx + 3, ty + 3),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Main
    cv2.putText(out, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Prep announcement ("GET READY — TREE POSE!")
# ─────────────────────────────────────────────────────────────────────────────

def draw_prep_announcement(
    frame: np.ndarray,
    pose_name: str,
    pose_emoji: str,
    timer: float,
) -> np.ndarray:
    """Draw the pose announcement overlay during PREPARE state with better positioning."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent overlay band (positioned to avoid top HUD)
    bh = 70  # Reduced height
    by = h // 2 - bh // 2 - 20  # Moved up slightly to avoid bottom elements
    overlay = out.copy()
    cv2.rectangle(overlay, (10, by), (w - 10, by + bh), HUD_BG, -1)  # Add margins
    cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)

    # "GET READY" text (smaller)
    ready_text = "GET READY"
    ready_size = cv2.getTextSize(ready_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    ready_x = (w - ready_size[0]) // 2
    cv2.putText(out, ready_text, (ready_x, by + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

    # Pose name (truncate if too long)
    display_name = pose_name
    if len(display_name) > 20:
        display_name = display_name[:17] + "..."
    
    name_size = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    nx = (w - name_size[0]) // 2
    cv2.putText(out, display_name, (nx, by + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 0), 2, cv2.LINE_AA)

    # Timer countdown (positioned below the box to avoid overlap)
    timer_text = f"Wall in {timer:.1f}s"
    ts = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(out, timer_text, ((w - ts[0]) // 2, by + bh + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Game end screens
# ─────────────────────────────────────────────────────────────────────────────

def draw_game_end(
    frame: np.ndarray,
    state: str,
    render_data: Dict[str, Any],
) -> np.ndarray:
    """Draw GAME OVER or VICTORY overlay."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Full-screen dim
    overlay = out.copy()
    tint = (0, 0, 40) if state == "game_over" else (0, 40, 0)
    cv2.rectangle(overlay, (0, 0), (w, h), tint, -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    if state == "game_over":
        title = "GAME OVER"
        color = (50, 50, 255)
    else:
        title = "VICTORY!"
        color = (0, 215, 255)

    # Title
    scale = 2.5
    thickness = 5
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    tx = (w - ts[0]) // 2
    ty = h // 2 - 30

    cv2.putText(out, title, (tx + 3, ty + 3),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, title, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    # Score summary
    score = render_data.get("score", 0)
    survived = render_data.get("round", 0)
    total = render_data.get("total_rounds", 0)

    info = f"Score: {score:,}  |  Rounds: {survived}/{total}"
    info_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    ix = (w - info_size[0]) // 2
    cv2.putText(out, info, (ix, ty + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    # Instruction
    cv2.putText(out, "Results loading...", ((w - 200) // 2, ty + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_color_bgr(score: float) -> Tuple[int, int, int]:
    """BGR colour coding by score."""
    if score >= 80: return (0, 220, 60)
    if score >= 60: return (0, 165, 255)
    if score >= 40: return (0, 80, 255)
    return (130, 130, 130)


def _brighten(color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
    """Lighten a BGR colour."""
    return tuple(min(255, c + amount) for c in color)  # type: ignore
