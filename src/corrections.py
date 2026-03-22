"""
Rule-based correction checkers for all 6 yoga poses.
Each function returns (is_correct: bool, corrections: list[dict])

Every correction dict has:
    {
        "message": str,          # human-readable instruction
        "severity": int,         # 1 (low) | 2 (medium) | 3 (high)
        "key": str,              # unique stable id, used by priority queue for cooldown
    }

Severity guide:
    3 — safety risk or pose is fundamentally wrong (e.g. knee caving in)
    2 — major form issue that reduces effectiveness
    1 — fine-tuning, polish
"""

import numpy as np

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def _angle(a, b, c):
    """Angle at point b, given three (x, y) points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _pt(lms, idx):
    return (lms[idx].x, lms[idx].y)


def _lm(lms, idx):
    return lms[idx]


def _correction(key, message, severity):
    return {"key": key, "message": message, "severity": severity}


# ─────────────────────────────────────────────
#  1. Tree Pose  (Vrikshasana)
# ─────────────────────────────────────────────


def check_tree_pose(landmarks):
    """
    Key checks:
      - Standing leg straight
      - Raised knee pointing outward
      - Raised foot high enough
      - Arms raised above head
      - Spine upright (no side lean)
    """
    lms = landmarks
    corrections = []

    left_ankle_y = _lm(lms, 27).y
    right_ankle_y = _lm(lms, 28).y

    if left_ankle_y < right_ankle_y:
        standing_hip, standing_knee, standing_ankle = 24, 26, 28
        _raised_hip, raised_knee, raised_ankle = 23, 25, 27
    else:
        standing_hip, standing_knee, standing_ankle = 23, 25, 27
        _raised_hip, raised_knee, raised_ankle = 24, 26, 28

    # Standing leg straight
    leg_angle = _angle(
        _pt(lms, standing_hip), _pt(lms, standing_knee), _pt(lms, standing_ankle)
    )
    if leg_angle < 160:
        corrections.append(
            _correction(
                "tree_standing_leg",
                f"Straighten your standing leg (angle: {leg_angle:.0f}°)",
                severity=3,
            )
        )

    # Raised knee outward
    hip_center_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    knee_offset = abs(_lm(lms, raised_knee).x - hip_center_x)
    if knee_offset < 0.08:
        corrections.append(
            _correction(
                "tree_knee_out", "Open your raised knee outward to the side", severity=2
            )
        )

    # Raised foot height
    if _lm(lms, raised_ankle).y > _lm(lms, standing_ankle).y - 0.05:
        corrections.append(
            _correction(
                "tree_foot_height",
                "Raise your foot higher — place it on your inner thigh or calf",
                severity=2,
            )
        )

    # Arms raised
    if _lm(lms, 15).y > _lm(lms, 11).y - 0.05:
        corrections.append(
            _correction(
                "tree_left_arm", "Raise your left arm above your head", severity=1
            )
        )
    if _lm(lms, 16).y > _lm(lms, 12).y - 0.05:
        corrections.append(
            _correction(
                "tree_right_arm", "Raise your right arm above your head", severity=1
            )
        )

    # Spine upright
    shoulder_mid_x = (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    if abs(shoulder_mid_x - hip_mid_x) > 0.07:
        corrections.append(
            _correction(
                "tree_lean", "Stand upright — you're leaning to the side", severity=2
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  2. Chair Pose  (Utkatasana)
# ─────────────────────────────────────────────


def check_chair_pose(landmarks):
    """
    Key checks:
      - Knees bent ~90° (80–110°)
      - Knees not past toes (knee x should not exceed ankle x significantly)
      - Spine upright / slight forward lean only — not collapsing
      - Arms raised overhead, roughly parallel
      - Weight in heels (ankles behind knees)
      - Feet together / hip-width (knees not caving inward)
    """
    lms = landmarks
    corrections = []

    # Knee bend angle (both legs)
    for side, hip, knee, ankle, label in [
        ("left", 23, 25, 27, "left"),
        ("right", 24, 26, 28, "right"),
    ]:
        angle = _angle(_pt(lms, hip), _pt(lms, knee), _pt(lms, ankle))
        if angle > 120:
            corrections.append(
                _correction(
                    f"chair_{label}_knee_bend",
                    f"Bend your {label} knee more — aim for about 90 degrees",
                    severity=2,
                )
            )
        elif angle < 60:
            corrections.append(
                _correction(
                    f"chair_{label}_knee_deep",
                    f"You're squatting too deep on your {label} side — rise up slightly",
                    severity=3,
                )
            )

    # Knees not caving inward — knee x distance should be close to hip x distance
    knee_width = abs(_lm(lms, 25).x - _lm(lms, 26).x)
    hip_width = abs(_lm(lms, 23).x - _lm(lms, 24).x)
    if knee_width < hip_width * 0.6:
        corrections.append(
            _correction(
                "chair_knees_cave",
                "Push your knees outward — keep them in line with your toes",
                severity=3,
            )
        )

    # Arms raised — wrists above shoulders
    if _lm(lms, 15).y > _lm(lms, 11).y - 0.05:
        corrections.append(
            _correction(
                "chair_left_arm", "Raise your left arm straight overhead", severity=1
            )
        )
    if _lm(lms, 16).y > _lm(lms, 12).y - 0.05:
        corrections.append(
            _correction(
                "chair_right_arm", "Raise your right arm straight overhead", severity=1
            )
        )

    # Torso — shoulders should be ahead of hips slightly (forward lean is ok)
    # but not collapsed (shoulders dropping too far forward past knees)
    (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    (_lm(lms, 25).x + _lm(lms, 26).x) / 2
    # In a side view this would be more meaningful; in frontal view check side lean
    shoulder_mid_x_r = (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    if abs(shoulder_mid_x_r - hip_mid_x) > 0.10:
        corrections.append(
            _correction(
                "chair_lean",
                "Keep your torso centred — you're leaning to one side",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  3. Warrior Pose  (Virabhadrasana II)
# ─────────────────────────────────────────────


def check_warrior_pose(landmarks):
    """
    Warrior II specific checks:
      - Front knee bent ~90°, directly over ankle
      - Back leg straight
      - Arms extended horizontally (wrists near shoulder height)
      - Torso upright — not leaning toward front leg
      - Hips open (facing side, not forward) — approximated by hip width vs shoulder width
    """
    lms = landmarks
    corrections = []

    # Detect which leg is forward (lower ankle y = higher on screen = back leg)
    # Forward leg = more bent knee
    left_knee_angle = _angle(_pt(lms, 23), _pt(lms, 25), _pt(lms, 27))
    right_knee_angle = _angle(_pt(lms, 24), _pt(lms, 26), _pt(lms, 28))

    if left_knee_angle < right_knee_angle:
        front = {"hip": 23, "knee": 25, "ankle": 27, "label": "left"}
        back = {"hip": 24, "knee": 26, "ankle": 28, "label": "right"}
    else:
        front = {"hip": 24, "knee": 26, "ankle": 28, "label": "right"}
        back = {"hip": 23, "knee": 25, "ankle": 27, "label": "left"}

    # Front knee ~90°
    front_angle = _angle(
        _pt(lms, front["hip"]), _pt(lms, front["knee"]), _pt(lms, front["ankle"])
    )
    if front_angle > 110:
        corrections.append(
            _correction(
                "warrior_front_knee_bend",
                f"Bend your {front['label']} (front) knee more — aim for 90 degrees",
                severity=2,
            )
        )
    elif front_angle < 70:
        corrections.append(
            _correction(
                "warrior_front_knee_over",
                f"Your {front['label']} knee is too far forward — press it back over your ankle",
                severity=3,
            )
        )

    # Back leg straight
    back_angle = _angle(
        _pt(lms, back["hip"]), _pt(lms, back["knee"]), _pt(lms, back["ankle"])
    )
    if back_angle < 155:
        corrections.append(
            _correction(
                "warrior_back_leg",
                f"Straighten your {back['label']} (back) leg fully",
                severity=2,
            )
        )

    # Arms horizontal — wrists should be near shoulder height (y difference small)
    left_wrist_diff = abs(_lm(lms, 15).y - _lm(lms, 11).y)
    right_wrist_diff = abs(_lm(lms, 16).y - _lm(lms, 12).y)
    if left_wrist_diff > 0.08:
        corrections.append(
            _correction(
                "warrior_left_arm",
                "Extend your left arm straight out at shoulder height",
                severity=1,
            )
        )
    if right_wrist_diff > 0.08:
        corrections.append(
            _correction(
                "warrior_right_arm",
                "Extend your right arm straight out at shoulder height",
                severity=1,
            )
        )

    # Arms spread wide — wrist distance should be large
    wrist_distance = abs(_lm(lms, 15).x - _lm(lms, 16).x)
    if wrist_distance < 0.35:
        corrections.append(
            _correction(
                "warrior_arms_wide",
                "Spread your arms wider — reach through your fingertips in both directions",
                severity=2,
            )
        )

    # Torso upright — no side lean
    shoulder_mid_x = (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    if abs(shoulder_mid_x - hip_mid_x) > 0.08:
        corrections.append(
            _correction(
                "warrior_torso_lean",
                "Keep your torso upright — don't lean toward the front leg",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  4. Cobra Pose  (Bhujangasana)
# ─────────────────────────────────────────────


def check_cobra_pose(landmarks):
    """
    Cobra is a floor backbend. In camera view (frontal or slight angle):
      - Shoulders should be lifted — shoulder y well above hip y
      - Elbows should be bent and close to body (not flared wide)
      - Neck not crunched — head not dropping
      - Hips stay on floor — hip y should be lower than shoulder y (already implied)
      - Shoulders back and down — not hunched up near ears

    Note: many checks here are approximate for a frontal camera.
    """
    lms = landmarks
    corrections = []

    left_shoulder = _lm(lms, 11)
    right_shoulder = _lm(lms, 12)
    left_hip = _lm(lms, 23)
    right_hip = _lm(lms, 24)
    left_elbow = _lm(lms, 13)
    right_elbow = _lm(lms, 14)
    _lm(lms, 15)
    _lm(lms, 16)
    nose = _lm(lms, 0)

    # Chest lifted — shoulders should be noticeably above hips (smaller y = higher)
    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_avg_y = (left_hip.y + right_hip.y) / 2
    lift = hip_avg_y - shoulder_avg_y
    if lift < 0.10:
        corrections.append(
            _correction(
                "cobra_chest_lift",
                "Lift your chest higher — press through your palms and open your heart upward",
                severity=3,
            )
        )

    # Shoulders not shrugged up to ears — shoulder y should not be too close to nose y
    shoulder_ear_gap = abs(shoulder_avg_y - nose.y)
    if shoulder_ear_gap < 0.12:
        corrections.append(
            _correction(
                "cobra_shoulders_shrug",
                "Roll your shoulders back and down — away from your ears",
                severity=2,
            )
        )

    # Elbows not flaring too wide — elbow x should be close to shoulder x
    left_elbow_flare = abs(left_elbow.x - left_shoulder.x)
    right_elbow_flare = abs(right_elbow.x - right_shoulder.x)
    if left_elbow_flare > 0.12:
        corrections.append(
            _correction(
                "cobra_left_elbow",
                "Draw your left elbow in closer to your body",
                severity=1,
            )
        )
    if right_elbow_flare > 0.12:
        corrections.append(
            _correction(
                "cobra_right_elbow",
                "Draw your right elbow in closer to your body",
                severity=1,
            )
        )

    # Arms roughly symmetrical — no tilting to one side
    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
    if shoulder_tilt > 0.06:
        corrections.append(
            _correction(
                "cobra_shoulder_tilt",
                "Level your shoulders — you're tilting to one side",
                severity=2,
            )
        )

    # Head position — nose should be above shoulders (not drooping)
    if nose.y > shoulder_avg_y:
        corrections.append(
            _correction(
                "cobra_head_drop",
                "Lift your head and gaze forward or slightly upward",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  5. Downward Dog  (Adho Mukha Svanasana)
# ─────────────────────────────────────────────


def check_downward_dog(landmarks):
    """
    Key checks:
      - Hips high — hips should be highest point (smallest y among shoulders/hips/knees)
      - Arms straight — elbow angle near 180°
      - Legs straight — knee angle near 180° (heels toward floor)
      - Spine long — shoulders, hips roughly form an inverted V
      - Head between arms / neutral — not craning up or dropping too low
      - Weight distributed — not all in wrists (can't measure directly, check arm angle)
    """
    lms = landmarks
    corrections = []

    left_shoulder = _lm(lms, 11)
    right_shoulder = _lm(lms, 12)
    left_hip = _lm(lms, 23)
    right_hip = _lm(lms, 24)
    nose = _lm(lms, 0)

    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_avg_y = (left_hip.y + right_hip.y) / 2

    # Hips should be the high point — hips higher than shoulders
    # In down-dog from frontal view this is tricky; we check hips are above shoulders
    if hip_avg_y > shoulder_avg_y - 0.05:
        corrections.append(
            _correction(
                "ddog_hips_high",
                "Lift your hips higher toward the ceiling — press back and up",
                severity=3,
            )
        )

    # Arms straight — elbow angle
    for side, shoulder, elbow, wrist, label in [
        ("left", 11, 13, 15, "left"),
        ("right", 12, 14, 16, "right"),
    ]:
        angle = _angle(_pt(lms, shoulder), _pt(lms, elbow), _pt(lms, wrist))
        if angle < 155:
            corrections.append(
                _correction(
                    f"ddog_{label}_arm",
                    f"Straighten your {label} arm fully — no bend at the elbow",
                    severity=2,
                )
            )

    # Legs straight — knee angle
    for side, hip, knee, ankle, label in [
        ("left", 23, 25, 27, "left"),
        ("right", 24, 26, 28, "right"),
    ]:
        angle = _angle(_pt(lms, hip), _pt(lms, knee), _pt(lms, ankle))
        if angle < 150:
            corrections.append(
                _correction(
                    f"ddog_{label}_knee",
                    f"Try to straighten your {label} leg — work on pressing the heel down",
                    severity=1,  # bent knees ok for beginners, hence severity 1
                )
            )

    # Head neutral — nose should be roughly between wrists in y
    # Simpler check: nose should not be above shoulder level (craning up)
    if nose.y < shoulder_avg_y - 0.05:
        corrections.append(
            _correction(
                "ddog_head_crane",
                "Relax your neck — let your head hang freely between your arms",
                severity=1,
            )
        )

    # Symmetry — shoulders level
    if abs(left_shoulder.y - right_shoulder.y) > 0.06:
        corrections.append(
            _correction(
                "ddog_shoulder_level",
                "Level your shoulders — distribute weight equally through both hands",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  6. Goddess Pose  (Utkata Konasana)
# ─────────────────────────────────────────────


def check_goddess_pose(landmarks):
    """
    Key checks:
      - Knees bent ~90° and tracking over toes (wide squat)
      - Knees not caving inward — knee width proportional to ankle width
      - Torso upright — no forward lean
      - Arms in goal-post position: elbows at shoulder height, bent 90°
      - Feet turned out (approximated by ankle width being wider than hip width)
    """
    lms = landmarks
    corrections = []

    # Knee bend — both sides
    for hip, knee, ankle, label in [
        (23, 25, 27, "left"),
        (24, 26, 28, "right"),
    ]:
        angle = _angle(_pt(lms, hip), _pt(lms, knee), _pt(lms, ankle))
        if angle > 115:
            corrections.append(
                _correction(
                    f"goddess_{label}_knee_bend",
                    f"Bend your {label} knee more — sink deeper into the pose",
                    severity=2,
                )
            )
        elif angle < 65:
            corrections.append(
                _correction(
                    f"goddess_{label}_knee_deep",
                    f"Rise up slightly on your {label} side — you're too deep",
                    severity=3,
                )
            )

    # Knees not caving — knee width should be >= ankle width
    knee_width = abs(_lm(lms, 25).x - _lm(lms, 26).x)
    ankle_width = abs(_lm(lms, 27).x - _lm(lms, 28).x)
    if knee_width < ankle_width * 0.85:
        corrections.append(
            _correction(
                "goddess_knees_cave",
                "Press your knees outward — open them wide over your toes",
                severity=3,
            )
        )

    # Wide stance — ankles should be noticeably wider than hips
    hip_width = abs(_lm(lms, 23).x - _lm(lms, 24).x)
    if ankle_width < hip_width * 1.3:
        corrections.append(
            _correction(
                "goddess_stance_wide",
                "Widen your stance — step your feet further apart",
                severity=2,
            )
        )

    # Torso upright — shoulder mid over hip mid
    shoulder_mid_x = (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    if abs(shoulder_mid_x - hip_mid_x) > 0.08:
        corrections.append(
            _correction(
                "goddess_torso_lean",
                "Keep your torso upright — stack your shoulders over your hips",
                severity=2,
            )
        )

    # Arms — goal post: elbows at shoulder height
    for shoulder, elbow, wrist, label in [
        (11, 13, 15, "left"),
        (12, 14, 16, "right"),
    ]:
        elbow_shoulder_diff = abs(_lm(lms, elbow).y - _lm(lms, shoulder).y)
        if elbow_shoulder_diff > 0.08:
            corrections.append(
                _correction(
                    f"goddess_{label}_elbow",
                    f"Raise your {label} elbow to shoulder height — goal-post arms",
                    severity=1,
                )
            )

        # Elbow bent ~90° — wrist should be above elbow
        if _lm(lms, wrist).y > _lm(lms, elbow).y - 0.03:
            corrections.append(
                _correction(
                    f"goddess_{label}_wrist",
                    f"Bend your {label} elbow to 90 degrees — wrist directly above elbow",
                    severity=1,
                )
            )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  Dispatcher
# ─────────────────────────────────────────────

POSE_CHECKERS = {
    "TreePose": check_tree_pose,
    "ChairPose": check_chair_pose,
    "WarriorPose": check_warrior_pose,
    "CobraPose": check_cobra_pose,
    "DownwardDog": check_downward_dog,
    "GoddessPose": check_goddess_pose,
}


def check_pose(pose_label: str, landmarks):
    """
    Main entry point.
    Returns (is_correct: bool, corrections: list[dict])
    corrections are already sorted by severity descending.
    """
    checker = POSE_CHECKERS.get(pose_label)
    if checker is None:
        return True, []
    is_correct, corrections = checker(landmarks)
    corrections.sort(key=lambda c: c["severity"], reverse=True)
    return is_correct, corrections
