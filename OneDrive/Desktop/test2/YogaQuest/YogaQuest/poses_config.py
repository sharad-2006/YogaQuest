"""
poses_config.py — Pure data: pose definitions, angle checks, levels, achievements,
                  and the Surya Namaskar sequence.  Landmark indices are
                  MediaPipe-compatible (populated from YOLO COCO keypoints).
"""

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Pose landmark indices (for reference)
# ─────────────────────────────────────────────────────────────────────────────
MP_POSE = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,  "LEFT_EYE": 2,  "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,  "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,  "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,     "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,     "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,     "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,     "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,     "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,       "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,      "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,     "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,      "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,"RIGHT_FOOT_INDEX": 32,
}

L  = MP_POSE  # shorthand alias


# ─────────────────────────────────────────────────────────────────────────────
# Level progression
# ─────────────────────────────────────────────────────────────────────────────
LEVEL_XP = {
     1:      0,  2:    150,  3:    350,  4:    700,  5:   1_200,
     6:  1_900,  7:  2_800,  8:  4_000,  9:  5_500, 10:   7_500,
    11: 10_000, 12: 13_500, 13: 17_500, 14: 22_500, 15:  29_000,
    16: 37_000, 17: 46_000, 18: 57_000, 19: 70_000, 20: 100_000,
}

LEVEL_NAMES = {
     1: "Beginner Yogi",     2: "Calm Seeker",       3: "Flexible Learner",
     4: "Balance Finder",    5: "Breath Follower",    6: "Mindful Mover",
     7: "Grounded Warrior",  8: "Flow Master",        9: "Inner Light",
    10: "Yoga Explorer",    11: "Drishti Holder",    12: "Prana Channeler",
    13: "Asana Artist",     14: "Zen Guardian",      15: "Chakra Awakened",
    16: "Lotus Sage",       17: "Kundalini Riser",   18: "Yoga Scholar",
    19: "Enlightened One",  20: "Grand Guru",
}

LEVEL_ICONS = {
     1: "🌱",  2: "🌿",  3: "🌾",  4: "🍃",  5: "🌸",
     6: "🌺",  7: "⚡",  8: "💫",  9: "🌟", 10: "✨",
    11: "🔥", 12: "💎", 13: "🎯", 14: "🛡️", 15: "🌈",
    16: "🪷", 17: "🐉", 18: "📿", 19: "🌙", 20: "👑",
}


# ─────────────────────────────────────────────────────────────────────────────
# Achievements catalogue
# ─────────────────────────────────────────────────────────────────────────────
ACHIEVEMENTS = {
    "first_pose": {
        "id": "first_pose", "name": "First Step",
        "description": "Complete your very first yoga pose",
        "icon": "🎯", "xp_bonus": 50,
    },
    "perfect_pose": {
        "id": "perfect_pose", "name": "Perfect Form",
        "description": "Score 90 or above on any single pose",
        "icon": "⭐", "xp_bonus": 100,
    },
    "all_7_poses": {
        "id": "all_7_poses", "name": "Complete Yogi",
        "description": "Hit all 7 poses in a single session",
        "icon": "🌟", "xp_bonus": 300,
    },
    "streak_3": {
        "id": "streak_3", "name": "Three-Day Streak",
        "description": "Practice 3 days in a row",
        "icon": "🔥", "xp_bonus": 150,
    },
    "streak_7": {
        "id": "streak_7", "name": "Week Warrior",
        "description": "Practice 7 days in a row",
        "icon": "🏆", "xp_bonus": 500,
    },
    "streak_30": {
        "id": "streak_30", "name": "Monthly Master",
        "description": "Practice every day for a full month",
        "icon": "👑", "xp_bonus": 2_000,
    },
    "surya_namaskar": {
        "id": "surya_namaskar", "name": "Sun Salutation",
        "description": "Complete a full Surya Namaskar cycle",
        "icon": "☀️", "xp_bonus": 300,
    },
    "combo_5": {
        "id": "combo_5", "name": "Combo King",
        "description": "Score ≥ 70 on 5 consecutive poses",
        "icon": "💥", "xp_bonus": 200,
    },
    "warrior_path": {
        "id": "warrior_path", "name": "Warrior Path",
        "description": "Complete Warrior I and Warrior II in one session",
        "icon": "⚔️", "xp_bonus": 200,
    },
    "balance_master": {
        "id": "balance_master", "name": "Balance Master",
        "description": "Score 85+ on Tree Pose",
        "icon": "🌳", "xp_bonus": 150,
    },
    "iron_core": {
        "id": "iron_core", "name": "Iron Core",
        "description": "Score 85+ on Downward Dog",
        "icon": "💪", "xp_bonus": 150,
    },
    "level_5": {
        "id": "level_5", "name": "Level 5 Ascent",
        "description": "Reach Level 5",
        "icon": "🌙", "xp_bonus": 0,
    },
    "level_10": {
        "id": "level_10", "name": "Level 10 Enlightened",
        "description": "Reach Level 10",
        "icon": "☀️", "xp_bonus": 0,
    },
    "level_20": {
        "id": "level_20", "name": "Grand Guru",
        "description": "Reach the maximum Level 20",
        "icon": "👑", "xp_bonus": 0,
    },
    "hundred_sessions": {
        "id": "hundred_sessions", "name": "Century Club",
        "description": "Complete 100 practice sessions",
        "icon": "💯", "xp_bonus": 1_000,
    },
    "speed_runner": {
        "id": "speed_runner", "name": "Speed Runner",
        "description": "Complete a session with 5 poses in under 3 minutes",
        "icon": "⚡", "xp_bonus": 200,
    },
    # ── Phase 3: Temporal Motion Analysis & Wall Game achievements ─────────
    "silky_smooth": {
        "id": "silky_smooth", "name": "Silky Smooth",
        "description": "Score 85+ on transition smoothness 3× in one session",
        "icon": "🌊", "xp_bonus": 180,
    },
    "iron_statue": {
        "id": "iron_statue", "name": "Iron Statue",
        "description": "Achieve 95+ stability on any held pose",
        "icon": "🗿", "xp_bonus": 150,
    },
    "flow_state": {
        "id": "flow_state", "name": "Flow State",
        "description": "Reach 2.0× transition multiplier in a session",
        "icon": "🌀", "xp_bonus": 250,
    },
    "speed_demon": {
        "id": "speed_demon", "name": "Speed Demon",
        "description": "Land a pose in under 800 ms from transition",
        "icon": "⚡", "xp_bonus": 200,
    },
    "wall_survivor_5": {
        "id": "wall_survivor_5", "name": "Wall Survivor",
        "description": "Survive 5 consecutive walls in Pose Wall Challenge",
        "icon": "🧱", "xp_bonus": 200,
    },
    "wall_perfect_10": {
        "id": "wall_perfect_10", "name": "Wall Perfect 10",
        "description": "Get 10 perfect fits in a single Wall session",
        "icon": "💫", "xp_bonus": 300,
    },
    "wall_master": {
        "id": "wall_master", "name": "Wall Master",
        "description": "Survive a full Insane difficulty session",
        "icon": "👹", "xp_bonus": 500,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Yoga Pose Definitions
#
# Each pose has:
#   angle_checks — list of dicts:
#       points   : [landmark_idx_A, landmark_idx_B (vertex), landmark_idx_C]
#       ideal    : ideal angle in degrees
#       tolerance: acceptable deviation in degrees (full marks within range)
#       weight   : contribution to total pose score (should sum ~1.0)
#       tip      : displayed if the check fails
# ─────────────────────────────────────────────────────────────────────────────
YOGA_POSES = {
    # ── Mountain Pose ─────────────────────────────────────────────────────
    "Tadasana": {
        "name": "Tadasana", "english": "Mountain Pose",
        "description": "Stand tall, feet together, spine erect, arms alongside body.",
        "difficulty": 1, "hold_time": 5,
        "emoji": "🏔️", "color": "#4ECDC4",
        "angle_checks": [
            {
                "name": "Left Knee Straight",
                "points": [L["LEFT_HIP"], L["LEFT_KNEE"], L["LEFT_ANKLE"]],
                "ideal": 175, "tolerance": 8, "weight": 0.20,
                "tip": "Straighten your left knee fully",
            },
            {
                "name": "Right Knee Straight",
                "points": [L["RIGHT_HIP"], L["RIGHT_KNEE"], L["RIGHT_ANKLE"]],
                "ideal": 175, "tolerance": 8, "weight": 0.20,
                "tip": "Straighten your right knee fully",
            },
            {
                "name": "Left Spine Alignment",
                "points": [L["LEFT_SHOULDER"], L["LEFT_HIP"], L["LEFT_KNEE"]],
                "ideal": 178, "tolerance": 8, "weight": 0.30,
                "tip": "Stack your shoulder directly above your hip",
            },
            {
                "name": "Right Spine Alignment",
                "points": [L["RIGHT_SHOULDER"], L["RIGHT_HIP"], L["RIGHT_KNEE"]],
                "ideal": 178, "tolerance": 8, "weight": 0.30,
                "tip": "Keep the right side of your torso straight",
            },
        ],
        "tips": [
            "Root both feet evenly into the ground.",
            "Draw your navel gently inward.",
            "Relax your shoulders down and back.",
            "Breathe slowly and steadily.",
            "Soften your face and jaw.",
        ],
        # ── Ideal normalised keypoints for wall silhouette (COCO-17) ──────
        # [norm_x, norm_y]  x=0 left, y=0 top, front-facing, person centred
        "ideal_keypoints_norm": [
            [0.500, 0.080],  # 0  Nose
            [0.485, 0.070],  # 1  Left Eye
            [0.515, 0.070],  # 2  Right Eye
            [0.470, 0.075],  # 3  Left Ear
            [0.530, 0.075],  # 4  Right Ear
            [0.430, 0.220],  # 5  Left Shoulder
            [0.570, 0.220],  # 6  Right Shoulder
            [0.415, 0.390],  # 7  Left Elbow
            [0.585, 0.390],  # 8  Right Elbow
            [0.410, 0.560],  # 9  Left Wrist
            [0.590, 0.560],  # 10 Right Wrist
            [0.445, 0.540],  # 11 Left Hip
            [0.555, 0.540],  # 12 Right Hip
            [0.445, 0.730],  # 13 Left Knee
            [0.555, 0.730],  # 14 Right Knee
            [0.445, 0.920],  # 15 Left Ankle
            [0.555, 0.920],  # 16 Right Ankle
        ],
    },

    # ── Tree Pose ──────────────────────────────────────────────────────────
    "Vrikshasana": {
        "name": "Vrikshasana", "english": "Tree Pose",
        "description": "Balance on one foot, other foot on inner thigh, arms raised overhead.",
        "difficulty": 3, "hold_time": 10,
        "emoji": "🌳", "color": "#2ECC71",
        "angle_checks": [
            {
                "name": "Standing Leg Straight",
                "points": [L["LEFT_HIP"], L["LEFT_KNEE"], L["LEFT_ANKLE"]],
                "ideal": 175, "tolerance": 8, "weight": 0.30,
                "tip": "Lock your standing leg straight and strong",
            },
            {
                "name": "Body Upright (L)",
                "points": [L["LEFT_SHOULDER"], L["LEFT_HIP"], L["LEFT_ANKLE"]],
                "ideal": 175, "tolerance": 10, "weight": 0.20,
                "tip": "Keep your torso vertical — do not lean sideways",
            },
            {
                "name": "Left Arm Raised",
                "points": [L["LEFT_HIP"], L["LEFT_SHOULDER"], L["LEFT_ELBOW"]],
                "ideal": 162, "tolerance": 15, "weight": 0.25,
                "tip": "Raise your left arm fully overhead",
            },
            {
                "name": "Right Arm Raised",
                "points": [L["RIGHT_HIP"], L["RIGHT_SHOULDER"], L["RIGHT_ELBOW"]],
                "ideal": 162, "tolerance": 15, "weight": 0.25,
                "tip": "Keep both arms symmetrically raised",
            },
        ],
        "tips": [
            "Fix your gaze on a single unmoving point (drishti).",
            "Press your raised foot firmly into your inner thigh.",
            "Never place the raised foot on the knee joint.",
            "Engage your core to prevent swaying.",
            "Keep your hips level and square.",
        ],
        "ideal_keypoints_norm": [
            [0.500, 0.080],  # 0  Nose
            [0.485, 0.070],  # 1  Left Eye
            [0.515, 0.070],  # 2  Right Eye
            [0.470, 0.075],  # 3  Left Ear
            [0.530, 0.075],  # 4  Right Ear
            [0.430, 0.210],  # 5  Left Shoulder
            [0.570, 0.210],  # 6  Right Shoulder
            [0.400, 0.120],  # 7  Left Elbow (arms overhead)
            [0.600, 0.120],  # 8  Right Elbow
            [0.450, 0.040],  # 9  Left Wrist (above head)
            [0.550, 0.040],  # 10 Right Wrist
            [0.445, 0.530],  # 11 Left Hip
            [0.555, 0.530],  # 12 Right Hip
            [0.445, 0.730],  # 13 Left Knee (standing leg)
            [0.580, 0.580],  # 14 Right Knee (bent, foot on thigh)
            [0.445, 0.920],  # 15 Left Ankle
            [0.500, 0.650],  # 16 Right Ankle (raised to inner thigh)
        ],
    },

    # ── Cobra Pose ────────────────────────────────────────────────────────
    "Bhujangasana": {
        "name": "Bhujangasana", "english": "Cobra Pose",
        "description": "Prone backbend — lift the chest using back muscles with arms supporting.",
        "difficulty": 2, "hold_time": 8,
        "emoji": "🐍", "color": "#E74C3C",
        "angle_checks": [
            {
                "name": "Left Elbow Angle",
                "points": [L["LEFT_SHOULDER"], L["LEFT_ELBOW"], L["LEFT_WRIST"]],
                "ideal": 135, "tolerance": 25, "weight": 0.20,
                "tip": "Keep elbows slightly bent and hugged close to your ribs",
            },
            {
                "name": "Right Elbow Angle",
                "points": [L["RIGHT_SHOULDER"], L["RIGHT_ELBOW"], L["RIGHT_WRIST"]],
                "ideal": 135, "tolerance": 25, "weight": 0.20,
                "tip": "Keep both elbows equally bent",
            },
            {
                "name": "Spine Curve (L)",
                "points": [L["LEFT_SHOULDER"], L["LEFT_HIP"], L["LEFT_KNEE"]],
                "ideal": 148, "tolerance": 20, "weight": 0.30,
                "tip": "Lift your chest higher off the mat and arch your spine",
            },
            {
                "name": "Spine Curve (R)",
                "points": [L["RIGHT_SHOULDER"], L["RIGHT_HIP"], L["RIGHT_KNEE"]],
                "ideal": 148, "tolerance": 20, "weight": 0.30,
                "tip": "Distribute the backbend evenly across your whole spine",
            },
        ],
        "tips": [
            "Press the tops of your feet into the floor.",
            "Draw your shoulders back and down.",
            "Look forward or slightly upward — do not crunch your neck.",
            "Breathe into your expanding chest.",
            "Keep your glutes soft, not clenched.",
        ],
        "ideal_keypoints_norm": [
            [0.500, 0.300],  # 0  Nose (lifted chest)
            [0.485, 0.290],  # 1  Left Eye
            [0.515, 0.290],  # 2  Right Eye
            [0.470, 0.300],  # 3  Left Ear
            [0.530, 0.300],  # 4  Right Ear
            [0.430, 0.420],  # 5  Left Shoulder
            [0.570, 0.420],  # 6  Right Shoulder
            [0.420, 0.550],  # 7  Left Elbow (supporting)
            [0.580, 0.550],  # 8  Right Elbow
            [0.410, 0.650],  # 9  Left Wrist (on mat)
            [0.590, 0.650],  # 10 Right Wrist
            [0.445, 0.720],  # 11 Left Hip (on mat)
            [0.555, 0.720],  # 12 Right Hip
            [0.445, 0.850],  # 13 Left Knee (on mat)
            [0.555, 0.850],  # 14 Right Knee
            [0.445, 0.950],  # 15 Left Ankle (on mat)
            [0.555, 0.950],  # 16 Right Ankle
        ],
    },

    # ── Downward Dog ──────────────────────────────────────────────────────
    "Adho_Mukha_Svanasana": {
        "name": "Adho_Mukha_Svanasana", "english": "Downward Facing Dog",
        "description": "Inverted V-shape — hips high, spine long, heels reaching toward the floor.",
        "difficulty": 2, "hold_time": 10,
        "emoji": "🐕", "color": "#9B59B6",
        "angle_checks": [
            {
                "name": "Left Hip Apex",
                "points": [L["LEFT_SHOULDER"], L["LEFT_HIP"], L["LEFT_KNEE"]],
                "ideal": 55, "tolerance": 15, "weight": 0.30,
                "tip": "Push your hips further up and back to deepen the V",
            },
            {
                "name": "Right Hip Apex",
                "points": [L["RIGHT_SHOULDER"], L["RIGHT_HIP"], L["RIGHT_KNEE"]],
                "ideal": 55, "tolerance": 15, "weight": 0.30,
                "tip": "Create a symmetrical inverted-V on both sides",
            },
            {
                "name": "Left Knee Straight",
                "points": [L["LEFT_HIP"], L["LEFT_KNEE"], L["LEFT_ANKLE"]],
                "ideal": 175, "tolerance": 10, "weight": 0.20,
                "tip": "Straighten your left leg (or soften if hamstrings are tight)",
            },
            {
                "name": "Right Knee Straight",
                "points": [L["RIGHT_HIP"], L["RIGHT_KNEE"], L["RIGHT_ANKLE"]],
                "ideal": 175, "tolerance": 10, "weight": 0.20,
                "tip": "Straighten your right leg",
            },
        ],
        "tips": [
            "Press through all ten fingers equally.",
            "Externally rotate your upper arms.",
            "Move your chest toward your thighs.",
            "Bend the knees slightly if hamstrings are very tight.",
            "Press heels toward the floor — they don't need to touch.",
        ],
        "ideal_keypoints_norm": [
            [0.500, 0.550],  # 0  Nose (head down, inverted V)
            [0.490, 0.540],  # 1  Left Eye
            [0.510, 0.540],  # 2  Right Eye
            [0.480, 0.545],  # 3  Left Ear
            [0.520, 0.545],  # 4  Right Ear
            [0.430, 0.440],  # 5  Left Shoulder
            [0.570, 0.440],  # 6  Right Shoulder
            [0.400, 0.520],  # 7  Left Elbow (arms straight)
            [0.600, 0.520],  # 8  Right Elbow
            [0.380, 0.650],  # 9  Left Wrist (on floor)
            [0.620, 0.650],  # 10 Right Wrist
            [0.450, 0.200],  # 11 Left Hip (apex of V)
            [0.550, 0.200],  # 12 Right Hip
            [0.450, 0.500],  # 13 Left Knee
            [0.550, 0.500],  # 14 Right Knee
            [0.450, 0.700],  # 15 Left Ankle
            [0.550, 0.700],  # 16 Right Ankle
        ],
    },

    # ── Warrior I ─────────────────────────────────────────────────────────
    "Virabhadrasana_I": {
        "name": "Virabhadrasana_I", "english": "Warrior I",
        "description": "Powerful lunge — front knee at 90°, back leg straight, arms raised overhead.",
        "difficulty": 3, "hold_time": 8,
        "emoji": "⚔️", "color": "#E67E22",
        "angle_checks": [
            {
                "name": "Front Knee 90°",
                "points": [L["LEFT_HIP"], L["LEFT_KNEE"], L["LEFT_ANKLE"]],
                "ideal": 90, "tolerance": 15, "weight": 0.30,
                "tip": "Bend your front knee to a full 90-degree angle",
            },
            {
                "name": "Back Leg Straight",
                "points": [L["RIGHT_HIP"], L["RIGHT_KNEE"], L["RIGHT_ANKLE"]],
                "ideal": 175, "tolerance": 10, "weight": 0.20,
                "tip": "Fully extend and firm your back leg",
            },
            {
                "name": "Left Arm Overhead",
                "points": [L["LEFT_HIP"], L["LEFT_SHOULDER"], L["LEFT_ELBOW"]],
                "ideal": 165, "tolerance": 15, "weight": 0.25,
                "tip": "Sweep your left arm fully overhead, shoulder away from ear",
            },
            {
                "name": "Right Arm Overhead",
                "points": [L["RIGHT_HIP"], L["RIGHT_SHOULDER"], L["RIGHT_ELBOW"]],
                "ideal": 165, "tolerance": 15, "weight": 0.25,
                "tip": "Keep both arms parallel and reaching upward",
            },
        ],
        "tips": [
            "Square your hips to face directly forward.",
            "Ground your back heel firmly into the mat.",
            "Keep the front knee stacked over the ankle, not past the toes.",
            "Lengthen your spine as you reach upward.",
            "Draw your lower belly in and up.",
        ],
        "ideal_keypoints_norm": [
            [0.450, 0.080],  # 0  Nose
            [0.440, 0.070],  # 1  Left Eye
            [0.460, 0.070],  # 2  Right Eye
            [0.430, 0.075],  # 3  Left Ear
            [0.470, 0.075],  # 4  Right Ear
            [0.400, 0.220],  # 5  Left Shoulder
            [0.530, 0.220],  # 6  Right Shoulder
            [0.380, 0.120],  # 7  Left Elbow (arms overhead)
            [0.550, 0.120],  # 8  Right Elbow
            [0.420, 0.040],  # 9  Left Wrist
            [0.500, 0.040],  # 10 Right Wrist
            [0.380, 0.500],  # 11 Left Hip
            [0.560, 0.500],  # 12 Right Hip
            [0.340, 0.650],  # 13 Left Knee (front, bent 90°)
            [0.620, 0.700],  # 14 Right Knee (back, straight)
            [0.340, 0.850],  # 15 Left Ankle
            [0.680, 0.880],  # 16 Right Ankle (back foot)
        ],
    },

    # ── Warrior II ────────────────────────────────────────────────────────
    "Virabhadrasana_II": {
        "name": "Virabhadrasana_II", "english": "Warrior II",
        "description": "Wide-leg stance — front knee at 90°, arms extended horizontally to both sides.",
        "difficulty": 3, "hold_time": 8,
        "emoji": "🗡️", "color": "#C0392B",
        "angle_checks": [
            {
                "name": "Front Knee 90°",
                "points": [L["LEFT_HIP"], L["LEFT_KNEE"], L["LEFT_ANKLE"]],
                "ideal": 90, "tolerance": 15, "weight": 0.30,
                "tip": "Deepen your front knee bend to reach 90°",
            },
            {
                "name": "Back Leg Straight",
                "points": [L["RIGHT_HIP"], L["RIGHT_KNEE"], L["RIGHT_ANKLE"]],
                "ideal": 175, "tolerance": 10, "weight": 0.20,
                "tip": "Keep your back leg strong and fully extended",
            },
            {
                "name": "Front Arm Extended",
                "points": [L["LEFT_SHOULDER"], L["LEFT_ELBOW"], L["LEFT_WRIST"]],
                "ideal": 175, "tolerance": 10, "weight": 0.25,
                "tip": "Extend your front arm straight, in line with the shoulder",
            },
            {
                "name": "Back Arm Extended",
                "points": [L["RIGHT_SHOULDER"], L["RIGHT_ELBOW"], L["RIGHT_WRIST"]],
                "ideal": 175, "tolerance": 10, "weight": 0.25,
                "tip": "Extend your back arm equally in the opposite direction",
            },
        ],
        "tips": [
            "Open your hips to the side — unlike Warrior I, do NOT square them.",
            "Stack your front knee directly above the ankle.",
            "Hold both arms level at shoulder height.",
            "Gaze forward over your front fingertips.",
            "Keep your torso upright — do not lean over the front leg.",
        ],
        "ideal_keypoints_norm": [
            [0.420, 0.090],  # 0  Nose (looking left)
            [0.410, 0.080],  # 1  Left Eye
            [0.430, 0.085],  # 2  Right Eye
            [0.400, 0.088],  # 3  Left Ear
            [0.440, 0.090],  # 4  Right Ear
            [0.400, 0.230],  # 5  Left Shoulder
            [0.580, 0.230],  # 6  Right Shoulder
            [0.250, 0.230],  # 7  Left Elbow (arm extended left)
            [0.730, 0.230],  # 8  Right Elbow (arm extended right)
            [0.100, 0.230],  # 9  Left Wrist (fingertips)
            [0.880, 0.230],  # 10 Right Wrist
            [0.380, 0.520],  # 11 Left Hip
            [0.580, 0.520],  # 12 Right Hip
            [0.320, 0.660],  # 13 Left Knee (front, bent 90°)
            [0.640, 0.700],  # 14 Right Knee (back, straight)
            [0.320, 0.880],  # 15 Left Ankle
            [0.700, 0.890],  # 16 Right Ankle
        ],
    },

    # ── Triangle Pose ─────────────────────────────────────────────────────
    "Trikonasana": {
        "name": "Trikonasana", "english": "Triangle Pose",
        "description": "Wide-leg side bend — both legs straight, one hand to ankle/shin, other arm straight up.",
        "difficulty": 3, "hold_time": 8,
        "emoji": "△", "color": "#3498DB",
        "angle_checks": [
            {
                "name": "Front Leg Straight",
                "points": [L["LEFT_HIP"], L["LEFT_KNEE"], L["LEFT_ANKLE"]],
                "ideal": 175, "tolerance": 8, "weight": 0.20,
                "tip": "Keep your front leg completely straight",
            },
            {
                "name": "Back Leg Straight",
                "points": [L["RIGHT_HIP"], L["RIGHT_KNEE"], L["RIGHT_ANKLE"]],
                "ideal": 175, "tolerance": 8, "weight": 0.20,
                "tip": "Keep your back leg completely straight",
            },
            {
                "name": "Side Bend Depth",
                "points": [L["LEFT_SHOULDER"], L["LEFT_HIP"], L["LEFT_ANKLE"]],
                "ideal": 72, "tolerance": 18, "weight": 0.35,
                "tip": "Reach your hand lower — try to touch your ankle or the floor",
            },
            {
                "name": "Arms in One Line",
                "points": [L["LEFT_WRIST"], L["LEFT_SHOULDER"], L["RIGHT_SHOULDER"]],
                "ideal": 168, "tolerance": 15, "weight": 0.25,
                "tip": "Stack top shoulder directly above bottom — both arms in one vertical line",
            },
        ],
        "tips": [
            "Extend your spine long before bending sideways.",
            "Keep both legs fully straight throughout.",
            "Open your chest to the ceiling.",
            "Reach the top arm straight up toward the sky.",
            "Look up at your top hand or straight forward.",
        ],
        "ideal_keypoints_norm": [
            [0.380, 0.200],  # 0  Nose (tilted, side bend)
            [0.370, 0.190],  # 1  Left Eye
            [0.390, 0.195],  # 2  Right Eye
            [0.360, 0.200],  # 3  Left Ear
            [0.400, 0.205],  # 4  Right Ear
            [0.380, 0.310],  # 5  Left Shoulder (lower)
            [0.520, 0.250],  # 6  Right Shoulder (upper)
            [0.330, 0.480],  # 7  Left Elbow (reaching down)
            [0.560, 0.120],  # 8  Right Elbow (reaching up)
            [0.310, 0.650],  # 9  Left Wrist (near ankle)
            [0.580, 0.040],  # 10 Right Wrist (sky)
            [0.400, 0.530],  # 11 Left Hip
            [0.560, 0.530],  # 12 Right Hip
            [0.350, 0.720],  # 13 Left Knee (straight)
            [0.620, 0.720],  # 14 Right Knee (straight)
            [0.320, 0.910],  # 15 Left Ankle
            [0.680, 0.910],  # 16 Right Ankle
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Surya Namaskar (Sun Salutation) — 12-step sequence
# ─────────────────────────────────────────────────────────────────────────────
SURYA_NAMASKAR_SEQUENCE = [
    {
        "step": 1,  "name": "Pranamasana",
        "english": "Prayer Pose",
        "description": "Stand in Tadasana, palms together at heart centre (Namaste).",
        "key_pose": "Tadasana", "duration": 3, "emoji": "🙏",
    },
    {
        "step": 2,  "name": "Hasta Uttanasana",
        "english": "Raised Arms Pose",
        "description": "Inhale — sweep arms overhead with a gentle backbend.",
        "key_pose": "Virabhadrasana_I", "duration": 3, "emoji": "🙌",
    },
    {
        "step": 3,  "name": "Uttanasana",
        "english": "Standing Forward Fold",
        "description": "Exhale — hinge at hips and fold forward, hands toward the floor.",
        "key_pose": "Adho_Mukha_Svanasana", "duration": 3, "emoji": "🙇",
    },
    {
        "step": 4,  "name": "Ashwa Sanchalanasana",
        "english": "Equestrian / Low Lunge",
        "description": "Inhale — step right leg back into a deep low lunge.",
        "key_pose": "Virabhadrasana_I", "duration": 3, "emoji": "🐎",
    },
    {
        "step": 5,  "name": "Adho Mukha Svanasana",
        "english": "Downward Facing Dog",
        "description": "Exhale — step left leg back, press into an inverted V.",
        "key_pose": "Adho_Mukha_Svanasana", "duration": 5, "emoji": "🐕",
    },
    {
        "step": 6,  "name": "Ashtanga Namaskara",
        "english": "Eight-Limb Salutation",
        "description": "Lower knees → chest → chin to the mat (six points touch).",
        "key_pose": "Bhujangasana", "duration": 2, "emoji": "🤸",
    },
    {
        "step": 7,  "name": "Bhujangasana",
        "english": "Cobra Pose",
        "description": "Inhale — slide forward, lift the chest into a cobra backbend.",
        "key_pose": "Bhujangasana", "duration": 5, "emoji": "🐍",
    },
    {
        "step": 8,  "name": "Adho Mukha Svanasana",
        "english": "Downward Facing Dog",
        "description": "Exhale — push back and up into downward dog.",
        "key_pose": "Adho_Mukha_Svanasana", "duration": 5, "emoji": "🐕",
    },
    {
        "step": 9,  "name": "Ashwa Sanchalanasana",
        "english": "Equestrian / Low Lunge",
        "description": "Inhale — step right foot forward between the hands.",
        "key_pose": "Virabhadrasana_I", "duration": 3, "emoji": "🐎",
    },
    {
        "step": 10, "name": "Uttanasana",
        "english": "Standing Forward Fold",
        "description": "Exhale — bring left foot forward and fold deeply.",
        "key_pose": "Adho_Mukha_Svanasana", "duration": 3, "emoji": "🙇",
    },
    {
        "step": 11, "name": "Hasta Uttanasana",
        "english": "Raised Arms Pose",
        "description": "Inhale — rise up sweeping arms overhead with a gentle backbend.",
        "key_pose": "Virabhadrasana_I", "duration": 3, "emoji": "🙌",
    },
    {
        "step": 12, "name": "Pranamasana",
        "english": "Prayer Pose",
        "description": "Exhale — return to standing Namaste. One full cycle complete.",
        "key_pose": "Tadasana", "duration": 3, "emoji": "🙏",
    },
]

# Core poses needed to count Surya Namaskar complete
SURYA_CORE_POSES = {
    "Tadasana", "Adho_Mukha_Svanasana", "Bhujangasana", "Virabhadrasana_I",
}
