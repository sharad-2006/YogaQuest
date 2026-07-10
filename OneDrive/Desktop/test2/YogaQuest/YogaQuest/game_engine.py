"""
game_engine.py — Gamification engine for YogaQuest.
Handles XP calculation, level progression, achievement unlocks,
daily streak tracking, and combo multipliers.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Tuple, Optional

import database as db
from poses_config import (
    LEVEL_XP, LEVEL_NAMES, LEVEL_ICONS,
    ACHIEVEMENTS, YOGA_POSES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def xp_for_level(level: int) -> int:
    """Total XP required to *reach* this level."""
    return LEVEL_XP.get(max(1, min(level, 20)), 0)


def level_from_xp(total_xp: int) -> int:
    """Current level given total accumulated XP."""
    level = 1
    for lvl, req in sorted(LEVEL_XP.items()):
        if total_xp >= req:
            level = lvl
    return level


def level_progress(total_xp: int, current_level: int) -> Tuple[int, int, float]:
    """
    Returns (xp_into_this_level, xp_needed_for_next_level, pct_0_to_1).
    At max level returns (total_xp, 0, 1.0).
    """
    next_lvl = current_level + 1
    if next_lvl not in LEVEL_XP:
        return total_xp, 0, 1.0
    cur_req  = LEVEL_XP[current_level]
    next_req = LEVEL_XP[next_lvl]
    into     = total_xp - cur_req
    needed   = next_req - cur_req
    pct      = min(1.0, into / needed) if needed else 1.0
    return into, needed, pct


# ─────────────────────────────────────────────────────────────────────────────
# GameEngine class
# ─────────────────────────────────────────────────────────────────────────────

class GameEngine:
    """Stateful game engine tied to a single user."""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.stats   = db.get_user_stats(user_id) or self._default_stats(user_id)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _default_stats(user_id: int) -> Dict[str, Any]:
        return {
            "user_id":           user_id,
            "total_xp":          0,
            "level":             1,
            "streak_days":       0,
            "best_streak":       0,
            "last_session_date": None,
            "achievements":      [],
            "total_sessions":    0,
            "perfect_poses":     0,
            "total_poses":       0,
        }

    def _current_achievements(self) -> set:
        return set(self.stats.get("achievements", []))

    # ── Streak ───────────────────────────────────────────────────────────────

    def update_streak(self) -> Dict[str, Any]:
        """
        Call once per session.  Updates streak_days and returns a summary dict:
          previous_streak, new_streak, streak_broken, new_best
        """
        today     = date.today()
        last_str  = self.stats.get("last_session_date")
        old_streak = self.stats.get("streak_days", 0)
        info = {
            "previous_streak": old_streak,
            "new_streak":      1,
            "streak_broken":   False,
            "new_best":        False,
        }

        if last_str:
            try:
                last = date.fromisoformat(last_str) if isinstance(last_str, str) else last_str
                delta = (today - last).days
                if delta == 0:
                    info["new_streak"] = old_streak            # same day
                elif delta == 1:
                    info["new_streak"] = old_streak + 1        # consecutive
                else:
                    info["new_streak"] = 1                     # broke streak
                    info["streak_broken"] = (old_streak > 1)
            except Exception:
                info["new_streak"] = 1

        self.stats["streak_days"]       = info["new_streak"]
        self.stats["last_session_date"] = today.isoformat()

        if info["new_streak"] > self.stats.get("best_streak", 0):
            self.stats["best_streak"] = info["new_streak"]
            info["new_best"] = True

        return info

    # ── XP calculation ───────────────────────────────────────────────────────

    def calculate_xp(self, session_results: Dict[str, Any]) -> Dict[str, int]:
        """
        Returns a breakdown dict and 'total' key.
        Sources:
          base_xp          10 XP per completed pose
          score_bonus      up to 60 XP (% of overall score × 0.6)
          streak_bonus     10 / 25 / 60 XP for 3 / 7 / 30+ days
          combo_bonus      5 XP per combo step above 1
          hold_bonus       2 XP per pose for each hold-bonus point
          perfect_bonus    25 XP for each pose scoring 90+
        """
        bp = {"base_xp": 0, "score_bonus": 0, "streak_bonus": 0,
              "combo_bonus": 0, "hold_bonus": 0, "perfect_bonus": 0, "total": 0}

        poses     = session_results.get("pose_details", [])
        score_avg = session_results.get("total_score", 0.0)
        combo     = session_results.get("max_combo", 0)
        streak    = self.stats.get("streak_days", 0)

        bp["base_xp"]    = len(poses) * 10
        bp["score_bonus"] = int(score_avg * 0.6)

        if streak >= 30:  bp["streak_bonus"] = 60
        elif streak >= 7: bp["streak_bonus"] = 25
        elif streak >= 3: bp["streak_bonus"] = 10

        bp["combo_bonus"] = max(0, combo - 1) * 5

        for p in poses:
            bp["hold_bonus"]   += int(p.get("hold_bonus", 0) * 2)
            if p.get("adjusted_score", 0) >= 90:
                bp["perfect_bonus"] += 25

        # ── Phase 3: Temporal motion + wall game XP sources ───────────────
        mean_stability  = session_results.get("max_stability", 0.0)
        mean_smoothness = 0.0
        smooth_count    = session_results.get("smooth_transitions_85plus", 0)
        if smooth_count > 0:
            mean_smoothness = min(100.0, smooth_count * 28.0)  # rough estimate

        bp["stability_bonus"]  = int(mean_stability  * 0.30)   # 0–30 XP
        bp["smoothness_bonus"] = int(mean_smoothness * 0.20)   # 0–20 XP
        bp["flow_state_bonus"] = 50 if session_results.get("flow_state_reached") else 0
        bp["wall_bonus"]       = session_results.get("wall_xp_earned", 0)

        bp["total"] = sum(v for k, v in bp.items() if k != "total")
        return bp

    # ── Achievement detection ─────────────────────────────────────────────────

    def check_achievements(self, session_results: Dict[str, Any]) -> Tuple[List[Dict], int]:
        """
        Compare session data against achievement criteria.
        Returns (list_of_new_achievement_dicts, bonus_xp_total).
        Mutates self.stats['achievements'] in-place.
        """
        existing = self._current_achievements()
        new_ids: List[str] = []

        poses   = session_results.get("pose_details", [])
        done    = set(session_results.get("completed_poses_list", []))
        score   = session_results.get("total_score", 0.0)
        combo   = session_results.get("max_combo", 0)
        streak  = self.stats.get("streak_days", 0)
        level   = self.stats.get("level", 1)
        sessions= self.stats.get("total_sessions", 0)
        surya   = session_results.get("surya_namaskar_pct", 0.0)
        dur     = session_results.get("duration", 999.0)

        def _unlock(aid: str) -> None:
            if aid not in existing and aid in ACHIEVEMENTS:
                new_ids.append(aid)

        # First pose ever
        if poses:
            _unlock("first_pose")

        # Perfect pose (any ≥ 90)
        for p in poses:
            if p.get("adjusted_score", 0) >= 90:
                _unlock("perfect_pose")
                break

        # All 7 poses
        if set(YOGA_POSES.keys()).issubset(done):
            _unlock("all_7_poses")

        # Streak milestones
        if streak >= 3:  _unlock("streak_3")
        if streak >= 7:  _unlock("streak_7")
        if streak >= 30: _unlock("streak_30")

        # Surya Namaskar
        if surya >= 0.9:
            _unlock("surya_namaskar")

        # Combo 5
        if combo >= 5:
            _unlock("combo_5")

        # Warrior path
        if "Virabhadrasana_I" in done and "Virabhadrasana_II" in done:
            _unlock("warrior_path")

        # Balance master — Tree ≥ 85
        for p in poses:
            if p["pose_name"] == "Vrikshasana" and p.get("adjusted_score", 0) >= 85:
                _unlock("balance_master")
                break

        # Iron core — Down Dog ≥ 85
        for p in poses:
            if p["pose_name"] == "Adho_Mukha_Svanasana" and p.get("adjusted_score", 0) >= 85:
                _unlock("iron_core")
                break

        # Level milestones
        if level >= 5:  _unlock("level_5")
        if level >= 10: _unlock("level_10")
        if level >= 20: _unlock("level_20")

        # 100 sessions
        if sessions >= 100:
            _unlock("hundred_sessions")

        # Speed runner: 5 poses in < 3 min
        if len(poses) >= 5 and dur < 180:
            _unlock("speed_runner")

        # ── Phase 3: Temporal motion achievements ─────────────────────────
        if session_results.get("smooth_transitions_85plus", 0) >= 3:
            _unlock("silky_smooth")
        if session_results.get("max_stability", 0) >= 95:
            _unlock("iron_statue")
        if session_results.get("flow_state_reached", False):
            _unlock("flow_state")
        if session_results.get("min_transition_ms", 9999) <= 800:
            _unlock("speed_demon")

        # ── Phase 3: Wall game achievements ───────────────────────────────
        if session_results.get("wall_max_consecutive_survived", 0) >= 5:
            _unlock("wall_survivor_5")
        if session_results.get("wall_perfect_fits", 0) >= 10:
            _unlock("wall_perfect_10")
        if (session_results.get("wall_difficulty") == "insane"
                and session_results.get("wall_completed")):
            _unlock("wall_master")

        # Deduplicate
        new_ids = list(dict.fromkeys(new_ids))
        bonus   = sum(ACHIEVEMENTS[a].get("xp_bonus", 0) for a in new_ids)

        self.stats["achievements"] = list(existing | set(new_ids))
        return [ACHIEVEMENTS[a] for a in new_ids], bonus

    # ── Master session processor ──────────────────────────────────────────────

    def process_session(self, session_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full post-session processing pipeline:
          1. Update streak
          2. Calculate XP
          3. Update session/pose counts
          4. Update level
          5. Check achievements (including achievement bonus XP)
          6. Persist to DB
          7. Return rich summary for the UI
        """
        # 1. Streak
        streak_info = self.update_streak()

        # 2. XP
        xp_bp = self.calculate_xp(session_results)

        # 3. Counters
        self.stats["total_sessions"] = self.stats.get("total_sessions", 0) + 1
        self.stats["total_poses"]    = (self.stats.get("total_poses", 0)
                                        + session_results.get("poses_completed", 0))

        # Perfect pose count
        for p in session_results.get("pose_details", []):
            if p.get("adjusted_score", 0) >= 90:
                self.stats["perfect_poses"] = self.stats.get("perfect_poses", 0) + 1

        # 4. Tentative level (before achievement XP)
        prev_level  = self.stats.get("level", 1)
        self.stats["total_xp"] = self.stats.get("total_xp", 0) + xp_bp["total"]
        self.stats["level"]    = level_from_xp(self.stats["total_xp"])

        # 5. Achievements
        new_achievements, ach_bonus = self.check_achievements(session_results)
        if ach_bonus:
            xp_bp["achievement_bonus"] = ach_bonus
            xp_bp["total"] += ach_bonus
            self.stats["total_xp"] += ach_bonus
            self.stats["level"]     = level_from_xp(self.stats["total_xp"])

        level_up  = self.stats["level"] > prev_level
        xp_in, xp_need, xp_pct = level_progress(self.stats["total_xp"], self.stats["level"])

        # 6. Persist
        db.update_user_stats(self.user_id, self.stats)
        db.save_session(self.user_id, {
            "date":        datetime.now().isoformat(),
            "type":        session_results.get("session_type", "practice"),
            "total_score": session_results.get("total_score", 0),
            "xp_earned":   xp_bp["total"],
            "poses_data":  session_results.get("pose_details", []),
            "duration":    session_results.get("duration", 0),
            "combo_bonus": xp_bp.get("combo_bonus", 0),
        })

        # 7. Return summary
        return {
            "xp_earned":      xp_bp["total"],
            "xp_breakdown":   xp_bp,
            "streak_info":    streak_info,
            "new_achievements": new_achievements,
            "level":          self.stats["level"],
            "prev_level":     prev_level,
            "level_up":       level_up,
            "level_name":     LEVEL_NAMES.get(self.stats["level"], "Master"),
            "level_icon":     LEVEL_ICONS.get(self.stats["level"], "🧘"),
            "total_xp":       self.stats["total_xp"],
            "xp_in_level":    xp_in,
            "xp_needed":      xp_need,
            "xp_pct":         xp_pct,
            "streak_days":    self.stats["streak_days"],
            "best_streak":    self.stats["best_streak"],
            "stats":          self.stats,
        }
