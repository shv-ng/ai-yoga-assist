"""
realtime.py
-----------
Real-time yoga pose classification + correction + voice feedback.

Per-frame pipeline:
  raw landmarks
    → LandmarkPipeline  (EMA smooth → body-normalize)
    → classifier model  (which pose?)
    → check_pose        (what needs fixing?)
    → FeedbackManager   (voice output)
    → SessionLogger     (session tracking)
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



CLASSIFIER_MODEL     = "../models/pose_classifier.h5"
ENCODER_MODEL        = "../models/label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.6
CORRECTION_EVERY_N   = 10   # check corrections every N frames (saves CPU)
SHOW_LIVE_STATS      = True


def classify(
    classifier_model: str = CLASSIFIER_MODEL,
    encoder_model:    str = ENCODER_MODEL,
):
    # ── Load model & encoder ──────────────────
    model = tf.keras.models.load_model(classifier_model)
    with open(encoder_model, "rb") as f:
        le = pickle.load(f)

    # ── Pipeline components ───────────────────
    lm_pipeline = LandmarkPipeline(smooth_alpha=0.4)
    fm          = FeedbackManager(cooldown_seconds=6.0, speak_interval=4.0)
    logger      = SessionLogger()

    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose           = mp.solutions.pose

    fm.start()
    logger.start()

    cap         = cv2.VideoCapture(0)
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            frame_count += 1

            # ── MediaPipe detection ────────────
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            label_text      = "No Pose Detected"
            confidence_text = ""
            color           = (0, 0, 255)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # ── Preprocess ────────────────
                raw_lms   = results.pose_landmarks.landmark
                processed = lm_pipeline.process(raw_lms)
                feat_vec  = lm_pipeline.to_feature_vector(processed)

                # ── Classify ──────────────────
                probs      = model.predict(feat_vec.reshape(1, -1), verbose=0)[0]
                top_idx    = int(np.argmax(probs))
                confidence = float(probs[top_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    label_text = le.inverse_transform([top_idx])[0]
                    color      = (0, 255, 0)

                    logger.log_pose(label_text)

                    # ── Correction check ──────
                    if frame_count % CORRECTION_EVERY_N == 0:
                        is_correct, corrections = check_pose(label_text, processed)
                        logger.log_corrections(corrections)

                        if is_correct:
                            fm.update_good()
                            cv2.putText(image, "Good Form!", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            fm.update(corrections)
                            for i, c in enumerate(corrections[:3]):
                                sev_color = (0, 0, 255)   if c["severity"] == 3 else \
                                            (0, 165, 255) if c["severity"] == 2 else \
                                            (0, 255, 255)
                                cv2.putText(image, f"• {c['message']}",
                                            (10, 120 + i * 28),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, sev_color, 2)
                else:
                    label_text = "Unknown Pose"
                    color      = (0, 165, 255)
                    lm_pipeline.reset()

                confidence_text = f"{confidence * 100:.1f}%"

            else:
                color = (0, 0, 255)
                lm_pipeline.reset()

            # ── HUD ───────────────────────────
            cv2.putText(image, label_text,      (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(image, confidence_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if fm.last_message:
                cv2.putText(image, f"Voice: {fm.last_message}",
                            (10, image.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if SHOW_LIVE_STATS:
                stats = logger.get_live_stats()
                if stats:
                    t = stats.get("elapsed_seconds", 0)
                    n = stats.get("total_corrections", 0)
                    cv2.putText(image,
                                f"Session {int(t//60):02d}:{int(t%60):02d}  |  Corrections: {n}",
                                (10, image.shape[0] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("AI Yoga Assist", image)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    fm.stop()
    logger.stop()
