"""
realtime.py
-----------
Real-time yoga pose classification + correction + voice feedback.

Per-frame pipeline:
  raw landmarks
    → visibility check     (full body in frame?)
    → LandmarkPipeline     (EMA smooth → body-normalize)
    → classifier model     (which pose?)
    → check_pose           (what needs fixing?)
    → FeedbackManager      (voice on 5-second ticker)
    → SessionLogger        (session tracking)
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

from src.corrections import check_pose
from src.feedback import FeedbackManager
from src.pipeline import LandmarkPipeline
from src.session_logger import SessionLogger

CLASSIFIER_MODEL = "../models/pose_classifier.h5"
ENCODER_MODEL = "../models/label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.6
SHOW_LIVE_STATS = True
FEEDBACK_TICK_SECONDS = 5.0
FEEDBACK_COOLDOWN_SECONDS = 12.0

# Landmarks that must all be visible for a "full body" frame.
# If ANY of these drops below VISIBILITY_THRESHOLD we warn the user.
_REQUIRED_LANDMARKS = {
    "nose": 0,
    "left shoulder": 11,
    "right shoulder": 12,
    "left elbow": 13,
    "right elbow": 14,
    "left wrist": 15,
    "right wrist": 16,
    "left hip": 23,
    "right hip": 24,
    "left knee": 25,
    "right knee": 26,
    "left ankle": 27,
    "right ankle": 28,
    "left foot": 31,
    "right foot": 32,
}
VISIBILITY_THRESHOLD = 0.6


def _check_visibility(landmark_list) -> tuple[bool, str]:
    """
    Check whether all required landmarks are visible.

    Returns
    -------
    (all_visible: bool, missing_part: str)
        missing_part is the human-readable name of the first low-visibility
        landmark, used to build a specific warning message.
    """
    for name, idx in _REQUIRED_LANDMARKS.items():
        if landmark_list[idx].visibility < VISIBILITY_THRESHOLD:
            return False, name
    return True, ""


def classify(
    classifier_model: str = CLASSIFIER_MODEL,
    encoder_model: str = ENCODER_MODEL,
):
    # ── Load model & encoder ──────────────────
    model = tf.keras.models.load_model(classifier_model)
    with open(encoder_model, "rb") as f:
        le = pickle.load(f)

    # ── Pipeline components ───────────────────
    lm_pipeline = LandmarkPipeline(smooth_alpha=0.4)
    fm = FeedbackManager(
        cooldown_seconds=FEEDBACK_COOLDOWN_SECONDS,
        tick_seconds=FEEDBACK_TICK_SECONDS,
    )
    logger = SessionLogger()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    fm.start()
    logger.start()

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # ── MediaPipe detection ────────────
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # ── Visibility check ──────────────────────────────────
                all_visible, missing_part = _check_visibility(
                    results.pose_landmarks.landmark
                )

                if not all_visible:
                    # On-screen warning
                    cv2.putText(
                        image,
                        f"Step into frame — {missing_part} not visible",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                    # Voice — severity 3 so it overrides pose corrections
                    fm.update(
                        [
                            {
                                "key": "frame_visibility",
                                "message": f"Please step into frame, your {missing_part} is not visible",
                                "severity": 3,
                            }
                        ]
                    )
                    lm_pipeline.reset()

                else:
                    # ── Preprocess ────────────────────────────────────
                    raw_lms = results.pose_landmarks.landmark
                    feat_vec = lm_pipeline.process_for_classify(raw_lms)
                    processed = lm_pipeline.process(raw_lms)

                    # ── Classify ──────────────────────────────────────
                    probs = model.predict(feat_vec.reshape(1, -1), verbose=0)[0]
                    top_idx = int(np.argmax(probs))
                    confidence = float(probs[top_idx])
                    # TEMP DEBUG
                    if not hasattr(classify, "_fc"):
                        classify._fc = 0
                    classify._fc += 1
                    if classify._fc % 30 == 0:
                        print("\n--- Probabilities ---", flush=True)
                        for i, p in enumerate(probs):
                            print(
                                f"  {le.inverse_transform([i])[0]:<15} {p*100:5.1f}%",
                                flush=True,
                            )

                    if confidence >= CONFIDENCE_THRESHOLD:
                        label_text = le.inverse_transform([top_idx])[0]
                        color = (0, 255, 0)

                        logger.log_pose(label_text)

                        is_correct, corrections = check_pose(label_text, processed)
                        logger.log_corrections(corrections)

                        if is_correct:
                            fm.update_good()
                            cv2.putText(
                                image,
                                "Good Form!",
                                (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                        else:
                            fm.update(corrections)
                            for i, c in enumerate(corrections[:3]):
                                sev_color = (
                                    (0, 0, 255)
                                    if c["severity"] == 3
                                    else (
                                        (0, 165, 255)
                                        if c["severity"] == 2
                                        else (0, 255, 255)
                                    )
                                )
                                cv2.putText(
                                    image,
                                    f"• {c['message']}",
                                    (10, 120 + i * 28),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    sev_color,
                                    2,
                                )

                        cv2.putText(
                            image,
                            label_text,
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            color,
                            3,
                        )
                        cv2.putText(
                            image,
                            f"{confidence * 100:.1f}%",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2,
                        )

                    else:
                        cv2.putText(
                            image,
                            "Unknown Pose",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 165, 255),
                            3,
                        )
                        cv2.putText(
                            image,
                            f"{confidence * 100:.1f}%",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 165, 255),
                            2,
                        )

            else:
                # No landmarks detected at all
                cv2.putText(
                    image,
                    "No Pose Detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
                fm.update(
                    [
                        {
                            "key": "frame_visibility",
                            "message": "Please step into frame, no body detected",
                            "severity": 3,
                        }
                    ]
                )
                lm_pipeline.reset()

            # ── Voice HUD ─────────────────────────────────────────────
            if fm.last_message:
                cv2.putText(
                    image,
                    f"Voice: {fm.last_message}",
                    (10, image.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

            # ── Session stats ──────────────────────────────────────────
            if SHOW_LIVE_STATS:
                stats = logger.get_live_stats()
                if stats:
                    t = stats.get("elapsed_seconds", 0)
                    n = stats.get("total_corrections", 0)
                    cv2.putText(
                        image,
                        f"Session {int(t//60):02d}:{int(t%60):02d}  |  Corrections: {n}",
                        (10, image.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (180, 180, 180),
                        1,
                    )

            cv2.imshow("AI Yoga Assist", image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    fm.stop()
    logger.stop()
