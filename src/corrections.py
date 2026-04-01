"""
Rule-based correction checkers for all 6 yoga poses.

** IMPORTANT — coordinate space **
All landmarks arriving here have already been processed by LandmarkPipeline:
  - Origin   : midpoint of the two hips
  - Scale    : 1 unit == torso height (hip-mid → shoulder-mid distance)
  - Axis     : x grows right, y grows DOWN (same as screen space)

Typical normalised ranges:
  - Shoulders are at y ≈ -1.0  (one torso-height ABOVE hips)
  - Hips      are at y ≈  0.0  (origin)
  - Knees     are at y ≈ +0.8  (below hips)
  - Ankles    are at y ≈ +1.5
  - Wrists (arms raised) at y ≈ -1.6 to -2.0
  - x spans  ≈ ±0.4 for hips, ±0.6 for shoulders

All threshold comments below include the unit "tu" (torso-units) so future
maintainers know what they are comparing.

Each function returns (is_correct: bool, corrections: list[dict])

Every correction dict has:
    {
        "message":     str,    # human-readable instruction (English)
        "message_hi":  str,    # human-readable instruction (Hindi)
        "severity":    int,    # 1 (low) | 2 (medium) | 3 (high)
        "key":         str,    # unique stable id used by FeedbackManager
    }

Severity guide:
    3 — safety risk or pose is fundamentally wrong
    2 — major form issue that reduces effectiveness
    1 — fine-tuning / polish
"""

# Reference translations for client.py (built-in visibility checks):
# "Please step into frame, your {missing_part} is not visible" -> "कृपया frame में आएं, आपका {missing_part} नहीं दिख रहा है"
# "Please step into frame, no body detected" -> "कृपया frame में आएं, कोई body नहीं दिख रही है"

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


def _correction(key, message, message_hi, severity):
    return {
        "key": key,
        "message": message,
        "message_hi": message_hi,
        "severity": severity,
    }


# ─────────────────────────────────────────────
#  1. Tree Pose  (Vrikshasana)
# ─────────────────────────────────────────────


def check_tree_pose(landmarks):
    """
    Normalised-space checks (1 tu = torso height):
      - Standing leg straight
      - Raised knee pointing outward (x offset > 0.15 tu from hip centre)
      - Raised foot above standing ankle by at least 0.3 tu
      - Arms raised above shoulders
      - Spine upright (shoulder_mid_x ≈ hip_mid_x, tolerance 0.12 tu)
    """
    lms = landmarks
    corrections = []

    # Determine which foot is raised (higher on screen = smaller y in normalised space)
    left_ankle_y = _lm(lms, 27).y
    right_ankle_y = _lm(lms, 28).y

    if left_ankle_y < right_ankle_y:  # left foot is higher → left is raised
        standing_hip, standing_knee, standing_ankle = 24, 26, 28
        _raised_hip, raised_knee, raised_ankle = 23, 25, 27
    else:
        standing_hip, standing_knee, standing_ankle = 23, 25, 27
        _raised_hip, raised_knee, raised_ankle = 24, 26, 28

    # Standing leg straight (angle > 160°)
    leg_angle = _angle(
        _pt(lms, standing_hip), _pt(lms, standing_knee), _pt(lms, standing_ankle)
    )
    if leg_angle < 160:
        corrections.append(
            _correction(
                "tree_standing_leg",
                f"Straighten your standing leg (angle: {leg_angle:.0f}°)",
                f"अपना standing leg सीधा करें (angle: {leg_angle:.0f}°)",
                severity=3,
            )
        )

    # Raised knee outward — in normalised space hip centre is at x≈0,
    # so |knee_x| > 0.15 tu means it is opening to the side
    hip_center_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    knee_offset = abs(_lm(lms, raised_knee).x - hip_center_x)
    if knee_offset < 0.15:  # 0.15 tu
        corrections.append(
            _correction(
                "tree_knee_out",
                "Open your raised knee outward to the side",
                "अपने उठे हुए knee को side में खोलें",
                severity=2,
            )
        )

    # Raised foot height — raised ankle must be above standing ankle by ≥ 0.30 tu
    foot_height_diff = _lm(lms, standing_ankle).y - _lm(lms, raised_ankle).y
    if foot_height_diff < 0.30:
        corrections.append(
            _correction(
                "tree_foot_height",
                "Raise your foot higher — place it on your inner thigh or calf",
                "अपना foot और ऊपर उठाएं — इसे अपनी inner thigh या calf पर रखें",
                severity=2,
            )
        )

    # Arms raised — wrists must be above shoulders by ≥ 0.10 tu
    # In normalised space y is more negative as you go up (shoulders ≈ -1)
    if _lm(lms, 15).y > _lm(lms, 11).y - 0.10:
        corrections.append(
            _correction(
                "tree_left_arm",
                "Raise your left arm above your head",
                "अपना left arm सिर के ऊपर उठाएं",
                severity=1,
            )
        )
    if _lm(lms, 16).y > _lm(lms, 12).y - 0.10:
        corrections.append(
            _correction(
                "tree_right_arm",
                "Raise your right arm above your head",
                "अपना right arm सिर के ऊपर उठाएं",
                severity=1,
            )
        )

    # Spine upright — |shoulder_mid_x - hip_mid_x| < 0.12 tu
    shoulder_mid_x = (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    if abs(shoulder_mid_x - hip_mid_x) > 0.12:
        corrections.append(
            _correction(
                "tree_lean",
                "Stand upright — you're leaning to the side",
                "सीधे खड़े हों — आप side में झुक रहे हैं",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  2. Chair Pose  (Utkatasana)
# ─────────────────────────────────────────────


def check_chair_pose(landmarks):
    """
    Normalised-space checks:
      - Knee angle 70°–115° (bent but not over-deep)
      - Knees not caving inward (knee_width ≥ 0.55 × hip_width)
      - Arms raised (wrists above shoulders by ≥ 0.10 tu)
      - No side lean (|shoulder_mid_x - hip_mid_x| < 0.15 tu)
    """
    lms = landmarks
    corrections = []

    for _side, hip, knee, ankle, label in [
        ("left", 23, 25, 27, "left"),
        ("right", 24, 26, 28, "right"),
    ]:
        angle = _angle(_pt(lms, hip), _pt(lms, knee), _pt(lms, ankle))
        if angle > 115:
            corrections.append(
                _correction(
                    f"chair_{label}_knee_bend",
                    f"Bend your {label} knee more — aim for about 90 degrees",
                    f"अपना {label} knee और मोड़ें — लगभग 90 degrees तक लाएं",
                    severity=2,
                )
            )
        elif angle < 60:
            corrections.append(
                _correction(
                    f"chair_{label}_knee_deep",
                    f"You're squatting too deep on your {label} side — rise up slightly",
                    f"आप अपने {label} side पर बहुत नीचे झुक रहे हैं — थोड़ा ऊपर उठें",
                    severity=3,
                )
            )

    # Knees not caving inward
    knee_width = abs(_lm(lms, 25).x - _lm(lms, 26).x)
    hip_width = abs(_lm(lms, 23).x - _lm(lms, 24).x)
    # In normalised space hip_width ≈ 0.5–0.8 tu; threshold ratio stays the same
    if hip_width > 0.05 and knee_width < hip_width * 0.55:
        corrections.append(
            _correction(
                "chair_knees_cave",
                "Push your knees outward — keep them in line with your toes",
                "अपने knees को बाहर की तरफ दबाएं — उन्हें toes की line में रखें",
                severity=3,
            )
        )

    # Arms raised above shoulders by ≥ 0.10 tu
    if _lm(lms, 15).y > _lm(lms, 11).y - 0.10:
        corrections.append(
            _correction(
                "chair_left_arm",
                "Raise your left arm straight overhead",
                "अपना left arm सीधा सिर के ऊपर उठाएं",
                severity=1,
            )
        )
    if _lm(lms, 16).y > _lm(lms, 12).y - 0.10:
        corrections.append(
            _correction(
                "chair_right_arm",
                "Raise your right arm straight overhead",
                "अपना right arm सीधा सिर के ऊपर उठाएं",
                severity=1,
            )
        )

    # No side lean — tolerance 0.15 tu (generous; some forward lean is correct)
    shoulder_mid_x = (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    if abs(shoulder_mid_x - hip_mid_x) > 0.15:
        corrections.append(
            _correction(
                "chair_lean",
                "Keep your torso centred — you're leaning to one side",
                "अपने torso को centre में रखें — आप एक side झुक रहे हैं",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  3. Warrior Pose  (Virabhadrasana II)
# ─────────────────────────────────────────────


def check_warrior_pose(landmarks):
    """
    Normalised-space checks:
      - Front knee angle 70°–115°
      - Back leg straight (> 155°)
      - Arms horizontal — |wrist_y - shoulder_y| < 0.20 tu
      - Arms spread wide — wrist-to-wrist x distance > 1.6 tu
      - Torso upright — |shoulder_mid_x - hip_mid_x| < 0.15 tu
    """
    lms = landmarks
    corrections = []

    left_knee_angle = _angle(_pt(lms, 23), _pt(lms, 25), _pt(lms, 27))
    right_knee_angle = _angle(_pt(lms, 24), _pt(lms, 26), _pt(lms, 28))

    if left_knee_angle < right_knee_angle:
        front = {"hip": 23, "knee": 25, "ankle": 27, "label": "left"}
        back = {"hip": 24, "knee": 26, "ankle": 28, "label": "right"}
    else:
        front = {"hip": 24, "knee": 26, "ankle": 28, "label": "right"}
        back = {"hip": 23, "knee": 25, "ankle": 27, "label": "left"}

    front_angle = _angle(
        _pt(lms, front["hip"]), _pt(lms, front["knee"]), _pt(lms, front["ankle"])
    )
    if front_angle > 115:
        corrections.append(
            _correction(
                "warrior_front_knee_bend",
                f"Bend your {front['label']} (front) knee more — aim for 90 degrees",
                f"अपना {front['label']} (front) knee और मोड़ें — 90 degrees का लक्ष्य रखें",
                severity=2,
            )
        )
    elif front_angle < 65:
        corrections.append(
            _correction(
                "warrior_front_knee_over",
                f"Your {front['label']} knee is too far forward — press it back over your ankle",
                f"आपका {front['label']} knee बहुत आगे है — इसे वापस ankle के ऊपर लाएं",
                severity=3,
            )
        )

    back_angle = _angle(
        _pt(lms, back["hip"]), _pt(lms, back["knee"]), _pt(lms, back["ankle"])
    )
    if back_angle < 155:
        corrections.append(
            _correction(
                "warrior_back_leg",
                f"Straighten your {back['label']} (back) leg fully",
                f"अपना {back['label']} (back) leg पूरी तरह सीधा करें",
                severity=2,
            )
        )

    # Arms horizontal — in normalised space shoulder_y ≈ -1.0;
    # wrist should be within 0.20 tu of shoulder height
    left_wrist_diff = abs(_lm(lms, 15).y - _lm(lms, 11).y)
    right_wrist_diff = abs(_lm(lms, 16).y - _lm(lms, 12).y)
    if left_wrist_diff > 0.20:
        corrections.append(
            _correction(
                "warrior_left_arm",
                "Extend your left arm straight out at shoulder height",
                "अपना left arm shoulder की height पर सीधा बाहर फैलाएं",
                severity=1,
            )
        )
    if right_wrist_diff > 0.20:
        corrections.append(
            _correction(
                "warrior_right_arm",
                "Extend your right arm straight out at shoulder height",
                "अपना right arm shoulder की height पर सीधा बाहर फैलाएं",
                severity=1,
            )
        )

    # Arms spread wide — wrist-to-wrist horizontal distance > 1.6 tu
    # (shoulder width ≈ 0.7 tu; arms fully extended ≈ 2.0+ tu)
    wrist_distance = abs(_lm(lms, 15).x - _lm(lms, 16).x)
    if wrist_distance < 1.6:
        corrections.append(
            _correction(
                "warrior_arms_wide",
                "Spread your arms wider — reach through your fingertips in both directions",
                "अपने arms को और चौड़ा फैलाएं — दोनों दिशाओं में fingertips से खिंचाव महसूस करें",
                severity=2,
            )
        )

    # Torso upright — 0.15 tu tolerance
    shoulder_mid_x = (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    if abs(shoulder_mid_x - hip_mid_x) > 0.15:
        corrections.append(
            _correction(
                "warrior_torso_lean",
                "Keep your torso upright — don't lean toward the front leg",
                "अपने torso को सीधा रखें — front leg की तरफ न झुकें",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  4. Cobra Pose  (Bhujangasana)
# ─────────────────────────────────────────────


def check_cobra_pose(landmarks):
    """
    Normalised-space checks (person is lying down, so y-axis orientation differs):
      - Chest lifted: hip_y - shoulder_y > 0.15 tu  (shoulders above hips)
      - Shoulders not shrugged: |shoulder_y - nose_y| > 0.20 tu
      - Elbows close to body: |elbow_x - shoulder_x| < 0.25 tu
      - Shoulders level: |left_shoulder_y - right_shoulder_y| < 0.12 tu
      - Head not drooping: nose_y < shoulder_avg_y  (nose above shoulders on screen)
    """
    lms = landmarks
    corrections = []

    left_shoulder = _lm(lms, 11)
    right_shoulder = _lm(lms, 12)
    left_hip = _lm(lms, 23)
    right_hip = _lm(lms, 24)
    left_elbow = _lm(lms, 13)
    right_elbow = _lm(lms, 14)
    nose = _lm(lms, 0)

    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_avg_y = (left_hip.y + right_hip.y) / 2

    # Chest lifted — shoulders should be higher (smaller y) than hips by ≥ 0.15 tu
    lift = hip_avg_y - shoulder_avg_y
    if lift < 0.15:
        corrections.append(
            _correction(
                "cobra_chest_lift",
                "Lift your chest higher — press through your palms and open your heart upward",
                "अपनी chest और ऊपर उठाएं — हथेली से दबाएं और ऊपर की ओर देखें",
                severity=3,
            )
        )

    # Shoulders not shrugged — gap between shoulder and nose ≥ 0.20 tu
    shoulder_ear_gap = abs(shoulder_avg_y - nose.y)
    if shoulder_ear_gap < 0.20:
        corrections.append(
            _correction(
                "cobra_shoulders_shrug",
                "Roll your shoulders back and down — away from your ears",
                "अपने shoulders को पीछे और नीचे रोल करें — कानों से दूर",
                severity=2,
            )
        )

    # Elbows close to body — |elbow_x - shoulder_x| < 0.25 tu
    left_elbow_flare = abs(left_elbow.x - left_shoulder.x)
    right_elbow_flare = abs(right_elbow.x - right_shoulder.x)
    if left_elbow_flare > 0.25:
        corrections.append(
            _correction(
                "cobra_left_elbow",
                "Draw your left elbow in closer to your body",
                "अपने left elbow को शरीर के करीब लाएं",
                severity=1,
            )
        )
    if right_elbow_flare > 0.25:
        corrections.append(
            _correction(
                "cobra_right_elbow",
                "Draw your right elbow in closer to your body",
                "अपने right elbow को शरीर के करीब लाएं",
                severity=1,
            )
        )

    # Shoulders level — |left_y - right_y| < 0.12 tu
    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
    if shoulder_tilt > 0.12:
        corrections.append(
            _correction(
                "cobra_shoulder_tilt",
                "Level your shoulders — you're tilting to one side",
                "अपने shoulders बराबर रखें — आप एक side झुक रहे हैं",
                severity=2,
            )
        )

    # Head not drooping — nose should be above (smaller y than) shoulder midpoint
    if nose.y > shoulder_avg_y:
        corrections.append(
            _correction(
                "cobra_head_drop",
                "Lift your head and gaze forward or slightly upward",
                "अपना head उठाएं और सामने या थोड़ा ऊपर देखें",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  5. Downward Dog  (Adho Mukha Svanasana)
# ─────────────────────────────────────────────


def check_downward_dog(landmarks):
    """
    Normalised-space checks:
      - Hips high: hip_avg_y < shoulder_avg_y - 0.10 tu
        (hips higher on screen = more negative y in normalised coords)
      - Arms straight: elbow angle > 155°
      - Legs straight: knee angle > 150°
      - Head neutral: nose not above (smaller y than) shoulder level by > 0.10 tu
      - Shoulders level: |left_shoulder_y - right_shoulder_y| < 0.12 tu
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

    # Hips high — in down-dog hips should be the apex;
    # hip_y < shoulder_y - 0.10 tu (more negative y = higher on screen)
    if hip_avg_y > shoulder_avg_y - 0.10:
        corrections.append(
            _correction(
                "ddog_hips_high",
                "Lift your hips higher toward the ceiling — press back and up",
                "अपने hips को छत की ओर ऊपर उठाएं — पीछे और ऊपर की तरफ दबाएं",
                severity=3,
            )
        )

    # Arms straight
    for _side, shoulder, elbow, wrist, label in [
        ("left", 11, 13, 15, "left"),
        ("right", 12, 14, 16, "right"),
    ]:
        angle = _angle(_pt(lms, shoulder), _pt(lms, elbow), _pt(lms, wrist))
        if angle < 155:
            corrections.append(
                _correction(
                    f"ddog_{label}_arm",
                    f"Straighten your {label} arm fully — no bend at the elbow",
                    f"अपना {label} arm पूरी तरह सीधा करें — elbow से न मोड़ें",
                    severity=2,
                )
            )

    # Legs straight (soft — bent knees ok for beginners)
    for _side, hip, knee, ankle, label in [
        ("left", 23, 25, 27, "left"),
        ("right", 24, 26, 28, "right"),
    ]:
        angle = _angle(_pt(lms, hip), _pt(lms, knee), _pt(lms, ankle))
        if angle < 150:
            corrections.append(
                _correction(
                    f"ddog_{label}_knee",
                    f"Try to straighten your {label} leg — work on pressing the heel down",
                    f"अपना {label} leg सीधा करने की कोशिश करें — heel को नीचे दबाने का प्रयास करें",
                    severity=1,
                )
            )

    # Head neutral — don't crane up; nose should not be more than 0.10 tu
    # above (smaller y than) shoulder level
    if nose.y < shoulder_avg_y - 0.10:
        corrections.append(
            _correction(
                "ddog_head_crane",
                "Relax your neck — let your head hang freely between your arms",
                "अपनी neck को relax करें — अपने head को arms के बीच ढीला छोड़ दें",
                severity=1,
            )
        )

    # Shoulders level
    if abs(left_shoulder.y - right_shoulder.y) > 0.12:
        corrections.append(
            _correction(
                "ddog_shoulder_level",
                "Level your shoulders — distribute weight equally through both hands",
                "अपने shoulders बराबर रखें — दोनों हाथों पर बराबर वजन दें",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  6. Goddess Pose  (Utkata Konasana)
# ─────────────────────────────────────────────


def check_goddess_pose(landmarks):
    """
    Normalised-space checks:
      - Knee angle 65°–115°
      - Knees not caving: knee_width ≥ 0.85 × ankle_width
      - Wide stance: ankle_width ≥ 1.3 × hip_width
      - Torso upright: |shoulder_mid_x - hip_mid_x| < 0.15 tu
      - Goal-post arms: |elbow_y - shoulder_y| < 0.20 tu
      - Elbows bent: wrist_y < elbow_y - 0.05 tu  (wrists above elbows)
    """
    lms = landmarks
    corrections = []

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
                    f"अपना {label} knee और मोड़ें — pose में और गहराई से बैठें",
                    severity=2,
                )
            )
        elif angle < 65:
            corrections.append(
                _correction(
                    f"goddess_{label}_knee_deep",
                    f"Rise up slightly on your {label} side — you're too deep",
                    f"अपने {label} side पर थोड़ा ऊपर उठें — आप बहुत ज्यादा नीचे हैं",
                    severity=3,
                )
            )

    # Knees not caving
    knee_width = abs(_lm(lms, 25).x - _lm(lms, 26).x)
    ankle_width = abs(_lm(lms, 27).x - _lm(lms, 28).x)
    if ankle_width > 0.05 and knee_width < ankle_width * 0.85:
        corrections.append(
            _correction(
                "goddess_knees_cave",
                "Press your knees outward — open them wide over your toes",
                "अपने knees को बाहर की तरफ दबाएं — उन्हें toes के ऊपर चौड़ा खोलें",
                severity=3,
            )
        )

    # Wide stance — ankle_width ≥ 1.3 × hip_width
    hip_width = abs(_lm(lms, 23).x - _lm(lms, 24).x)
    if hip_width > 0.05 and ankle_width < hip_width * 1.3:
        corrections.append(
            _correction(
                "goddess_stance_wide",
                "Widen your stance — step your feet further apart",
                "अपना stance और चौड़ा करें — पैरों को और दूर ले जाएं",
                severity=2,
            )
        )

    # Torso upright
    shoulder_mid_x = (_lm(lms, 11).x + _lm(lms, 12).x) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    if abs(shoulder_mid_x - hip_mid_x) > 0.15:
        corrections.append(
            _correction(
                "goddess_torso_lean",
                "Keep your torso upright — stack your shoulders over your hips",
                "अपने torso को सीधा रखें — shoulders को hips के ऊपर रखें",
                severity=2,
            )
        )

    # Goal-post arms — elbow within 0.20 tu of shoulder height
    for shoulder, elbow, wrist, label in [
        (11, 13, 15, "left"),
        (12, 14, 16, "right"),
    ]:
        elbow_shoulder_diff = abs(_lm(lms, elbow).y - _lm(lms, shoulder).y)
        if elbow_shoulder_diff > 0.20:
            corrections.append(
                _correction(
                    f"goddess_{label}_elbow",
                    f"Raise your {label} elbow to shoulder height — goal-post arms",
                    f"अपना {label} elbow shoulder की height तक उठाएं — goal-post arms बनाएं",
                    severity=1,
                )
            )

        # Wrist above elbow by ≥ 0.05 tu  (y is more negative = higher)
        if _lm(lms, wrist).y > _lm(lms, elbow).y - 0.05:
            corrections.append(
                _correction(
                    f"goddess_{label}_wrist",
                    f"Bend your {label} elbow to 90 degrees — wrist directly above elbow",
                    f"अपने {label} elbow को 90 degrees पर मोड़ें — wrist को सीधे elbow के ऊपर रखें",
                    severity=1,
                )
            )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  7. Corpse Pose  (Savasana)
# ─────────────────────────────────────────────


def check_corpse_pose(landmarks):
    """
    Normalised-space checks:
      - Horizontal: |shoulder_avg_y - hip_avg_y| < 0.30 tu
      - Arms away: |wrist_x - shoulder_x| > 0.20 tu
      - Legs not crossed: ankle_width > 0.30 tu
      - Head neutral: |nose_x - hip_mid_x| < 0.15 tu
    """
    lms = landmarks
    corrections = []

    shoulder_avg_y = (_lm(lms, 11).y + _lm(lms, 12).y) / 2
    hip_avg_y = (_lm(lms, 23).y + _lm(lms, 24).y) / 2
    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2

    # Horizontal
    if abs(shoulder_avg_y - hip_avg_y) > 0.30:
        corrections.append(
            _correction(
                "corpse_horizontal",
                "Lie flat on your back — keep your body horizontal",
                "पीठ के बल सीधे लेटें — अपने शरीर को horizontal रखें",
                severity=3,
            )
        )

    # Arms away
    if abs(_lm(lms, 15).x - _lm(lms, 11).x) < 0.20:
        corrections.append(
            _correction(
                "corpse_left_arm",
                "Move your left arm slightly away from your body",
                "अपने left arm को शरीर से थोड़ा दूर ले जाएं",
                severity=1,
            )
        )
    if abs(_lm(lms, 16).x - _lm(lms, 12).x) < 0.20:
        corrections.append(
            _correction(
                "corpse_right_arm",
                "Move your right arm slightly away from your body",
                "अपने right arm को शरीर से थोड़ा दूर ले जाएं",
                severity=1,
            )
        )

    # Legs not crossed
    ankle_width = abs(_lm(lms, 27).x - _lm(lms, 28).x)
    if ankle_width < 0.30:
        corrections.append(
            _correction(
                "corpse_legs_crossed",
                "Keep your feet slightly apart — let your toes fall open",
                "अपने पैरों को थोड़ा अलग रखें — toes को बाहर की तरफ ढीला छोड़ दें",
                severity=1,
            )
        )

    # Head neutral
    nose_x = _lm(lms, 0).x
    if abs(nose_x - hip_mid_x) > 0.15:
        corrections.append(
            _correction(
                "corpse_head",
                "Keep your head neutral and centred",
                "अपने head को सीधा और centre में रखें",
                severity=1,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  8. Bridge Pose  (Setu Bandha Sarvangasana)
# ─────────────────────────────────────────────


def check_bridge_pose(landmarks):
    """
    Normalised-space checks:
      - Hips raised: hip_avg_y < shoulder_avg_y - 0.10 tu AND hip_avg_y < ankle_avg_y - 0.10 tu
      - Knees not caving: knee_width ≥ 0.8 * ankle_width
      - Feet flat: |ankle_y - heel_y| < 0.10 tu
    """
    lms = landmarks
    corrections = []

    shoulder_avg_y = (_lm(lms, 11).y + _lm(lms, 12).y) / 2
    hip_avg_y = (_lm(lms, 23).y + _lm(lms, 24).y) / 2
    ankle_avg_y = (_lm(lms, 27).y + _lm(lms, 28).y) / 2

    # Hips raised (smaller y is higher)
    if hip_avg_y > shoulder_avg_y - 0.10 or hip_avg_y > ankle_avg_y - 0.10:
        corrections.append(
            _correction(
                "bridge_hips_raised",
                "Lift your hips higher toward the ceiling — they should be above your shoulders and knees",
                "अपने hips को छत की ओर और ऊपर उठाएं — वे shoulders और knees से ऊपर होने चाहिए",
                severity=3,
            )
        )

    # Knees not caving
    knee_width = abs(_lm(lms, 25).x - _lm(lms, 26).x)
    ankle_width = abs(_lm(lms, 27).x - _lm(lms, 28).x)
    if ankle_width > 0.05 and knee_width < ankle_width * 0.8:
        corrections.append(
            _correction(
                "bridge_knees_cave",
                "Keep your knees parallel — don't let them cave inward",
                "अपने knees को parallel रखें — उन्हें अंदर की तरफ न झुकने दें",
                severity=2,
            )
        )

    # Feet flat (ankles and heels should be at floor level)
    if (
        abs(_lm(lms, 27).y - _lm(lms, 29).y) > 0.10
        or abs(_lm(lms, 28).y - _lm(lms, 30).y) > 0.10
    ):
        corrections.append(
            _correction(
                "bridge_feet_flat",
                "Keep your feet flat on the floor",
                "अपने पैरों को floor पर सीधा रखें",
                severity=2,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  9. Supine Twist Pose  (Supta Matsyendrasana)
# ─────────────────────────────────────────────


def check_supine_twist_pose(landmarks):
    """
    Normalised-space checks:
      - One knee crossed: |knee_x - hip_mid_x| > 0.40 tu
      - Shoulders flat: |left_shoulder_y - right_shoulder_y| < 0.12 tu
      - Arms extended: |wrist_y - shoulder_y| < 0.20 tu
    """
    lms = landmarks
    corrections = []

    hip_mid_x = (_lm(lms, 23).x + _lm(lms, 24).x) / 2
    left_knee_x = _lm(lms, 25).x
    right_knee_x = _lm(lms, 26).x

    # Knee crossed
    if abs(left_knee_x - hip_mid_x) < 0.40 and abs(right_knee_x - hip_mid_x) < 0.40:
        corrections.append(
            _correction(
                "supine_twist_knee",
                "Cross one knee over your body to the opposite side",
                "एक knee को शरीर के ऊपर से दूसरी side ले जाएं",
                severity=3,
            )
        )

    # Shoulders flat
    shoulder_tilt = abs(_lm(lms, 11).y - _lm(lms, 12).y)
    if shoulder_tilt > 0.12:
        corrections.append(
            _correction(
                "supine_twist_shoulders",
                "Keep both shoulders flat on the mat",
                "दोनों shoulders को mat पर सीधा रखें",
                severity=2,
            )
        )

    # Arms extended
    if abs(_lm(lms, 15).y - _lm(lms, 11).y) > 0.20:
        corrections.append(
            _correction(
                "supine_twist_left_arm",
                "Extend your left arm out to the side at shoulder height",
                "अपने left arm को shoulder की height पर side में फैलाएं",
                severity=1,
            )
        )
    if abs(_lm(lms, 16).y - _lm(lms, 12).y) > 0.20:
        corrections.append(
            _correction(
                "supine_twist_right_arm",
                "Extend your right arm out to the side at shoulder height",
                "अपने right arm को shoulder की height पर side में फैलाएं",
                severity=1,
            )
        )

    return len(corrections) == 0, corrections


# ─────────────────────────────────────────────
#  10. Happy Baby Pose  (Ananda Balasana)
# ─────────────────────────────────────────────


def check_happy_baby_pose(landmarks):
    """
    Normalised-space checks:
      - Knees above hips: knee_y < hip_y - 0.20 tu
      - Knees wide: knee_width > 0.70 tu
      - Ankles above knees: ankle_y < knee_y - 0.10 tu
    """
    lms = landmarks
    corrections = []

    left_knee_y = _lm(lms, 25).y
    right_knee_y = _lm(lms, 26).y
    left_hip_y = _lm(lms, 23).y
    right_hip_y = _lm(lms, 24).y

    # Knees above hips
    if left_knee_y > left_hip_y - 0.20 or right_knee_y > right_hip_y - 0.20:
        corrections.append(
            _correction(
                "happy_baby_knees_up",
                "Pull your knees closer to your chest",
                "अपने knees को अपनी chest के करीब लाएं",
                severity=3,
            )
        )

    # Knees wide
    knee_width = abs(_lm(lms, 25).x - _lm(lms, 26).x)
    if knee_width < 0.70:
        corrections.append(
            _correction(
                "happy_baby_knees_wide",
                "Open your knees wide toward your armpits",
                "अपने knees को अपनी armpits की तरफ चौड़ा खोलें",
                severity=2,
            )
        )

    # Ankles above knees
    if _lm(lms, 27).y > left_knee_y - 0.10 or _lm(lms, 28).y > right_knee_y - 0.10:
        corrections.append(
            _correction(
                "happy_baby_ankles",
                "Stack your ankles directly over your knees",
                "अपने ankles को सीधे अपने knees के ऊपर रखें",
                severity=2,
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
    "CorpsePose": check_corpse_pose,
    "BridgePose": check_bridge_pose,
    "SupineTwist": check_supine_twist_pose,
    "HappyBabyPose": check_happy_baby_pose,
}


def check_pose(pose_label: str, landmarks):
    """
    Main entry point.
    Returns (is_correct: bool, corrections: list[dict])
    Corrections are sorted by severity descending.
    """
    checker = POSE_CHECKERS.get(pose_label)
    if checker is None:
        return True, []
    is_correct, corrections = checker(landmarks)
    corrections.sort(key=lambda c: c["severity"], reverse=True)
    return is_correct, corrections
