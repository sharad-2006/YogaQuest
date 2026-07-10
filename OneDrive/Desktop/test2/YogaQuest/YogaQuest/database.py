"""
database.py — YogaQuest SQLite persistence layer.
Manages users, session history, XP/level stats, and leaderboard data.
"""

import sqlite3
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any

DB_PATH = "yogaquest.db"


# ─────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't already exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            created_at  TEXT    NOT NULL,
            avatar      TEXT    DEFAULT '🧘'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id          INTEGER PRIMARY KEY,
            total_xp         INTEGER DEFAULT 0,
            level            INTEGER DEFAULT 1,
            streak_days      INTEGER DEFAULT 0,
            best_streak      INTEGER DEFAULT 0,
            last_session_date TEXT,
            achievements     TEXT    DEFAULT '[]',
            total_sessions   INTEGER DEFAULT 0,
            perfect_poses    INTEGER DEFAULT 0,
            total_poses      INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            session_date  TEXT    NOT NULL,
            session_type  TEXT    NOT NULL,
            total_score   REAL    DEFAULT 0,
            xp_earned     INTEGER DEFAULT 0,
            poses_data    TEXT    DEFAULT '[]',
            duration      REAL    DEFAULT 0,
            combo_bonus   INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_wall (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            wall_date  TEXT    NOT NULL,
            score      REAL    DEFAULT 0,
            difficulty TEXT    DEFAULT 'normal',
            completed  INTEGER DEFAULT 0,
            UNIQUE(user_id, wall_date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# User helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_user(username: str) -> int:
    """Return user_id, creating the user (and empty stats row) if needed."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = c.fetchone()

    if row:
        user_id = row[0]
    else:
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO users (username, created_at) VALUES (?, ?)",
            (username, now),
        )
        user_id = c.lastrowid
        c.execute("INSERT INTO user_stats (user_id) VALUES (?)", (user_id,))
        conn.commit()

    conn.close()
    return user_id


def get_all_users() -> List[tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username FROM users ORDER BY username")
    rows = c.fetchall()
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_user_stats(user_id: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "user_id":           row[0],
        "total_xp":          row[1],
        "level":             row[2],
        "streak_days":       row[3],
        "best_streak":       row[4],
        "last_session_date": row[5],
        "achievements":      json.loads(row[6]) if row[6] else [],
        "total_sessions":    row[7],
        "perfect_poses":     row[8],
        "total_poses":       row[9],
    }


def update_user_stats(user_id: int, stats: Dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        UPDATE user_stats SET
            total_xp          = ?,
            level             = ?,
            streak_days       = ?,
            best_streak       = ?,
            last_session_date = ?,
            achievements      = ?,
            total_sessions    = ?,
            perfect_poses     = ?,
            total_poses       = ?
        WHERE user_id = ?
        """,
        (
            stats["total_xp"],
            stats["level"],
            stats["streak_days"],
            stats["best_streak"],
            stats.get("last_session_date"),
            json.dumps(stats.get("achievements", [])),
            stats["total_sessions"],
            stats["perfect_poses"],
            stats["total_poses"],
            user_id,
        ),
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_session(user_id: int, session_data: Dict[str, Any]) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO sessions
            (user_id, session_date, session_type, total_score,
             xp_earned, poses_data, duration, combo_bonus)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            session_data.get("date", datetime.now().isoformat()),
            session_data.get("type", "practice"),
            session_data.get("total_score", 0),
            session_data.get("xp_earned", 0),
            json.dumps(session_data.get("poses_data", [])),
            session_data.get("duration", 0),
            session_data.get("combo_bonus", 0),
        ),
    )
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_session_history(user_id: int, limit: int = 10) -> List[tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT id, session_date, session_type, total_score,
               xp_earned, duration, poses_data
        FROM   sessions
        WHERE  user_id = ?
        ORDER  BY session_date DESC
        LIMIT  ?
        """,
        (user_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def get_leaderboard(limit: int = 15) -> List[tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT u.username,
               s.total_xp,
               s.level,
               s.streak_days,
               s.total_sessions,
               s.perfect_poses,
               s.total_poses
        FROM   users u
        JOIN   user_stats s ON u.id = s.user_id
        ORDER  BY s.total_xp DESC
        LIMIT  ?
        """,
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows
