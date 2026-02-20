import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import numpy as np

# MediaPipe landmark indices
# 11=left shoulder, 12=right shoulder
# 23=left hip, 24=right hip
# 25=left knee, 26=right knee
# 27=left ankle, 28=right ankle
# 15=left wrist, 16=right wrist

def calculate_angle(a, b, c):
    """Angle at point b, given points a, b, c as (x, y) tuples."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def check_tree_pose(landmarks):
    """
    landmarks: list of mediapipe landmark objects (results.pose_landmarks.landmark)
    Returns: (is_correct: bool, feedback: list of strings)
    """
    feedback = []

    def pt(idx):
        return (landmarks[idx].x, landmarks[idx].y)

    def lm(idx):
        return landmarks[idx]

    # --- Detect which leg is raised (higher foot = raised leg) ---
    left_ankle_y  = lm(27).y
    right_ankle_y = lm(28).y

    # In mediapipe, y increases downward, so smaller y = higher on screen
    if left_ankle_y < right_ankle_y:
        # Left leg is raised, right leg is standing
        standing_hip, standing_knee, standing_ankle = 24, 26, 28
        raised_hip,   raised_knee,   raised_ankle   = 23, 25, 27
    else:
        # Right leg is raised, left leg is standing
        standing_hip, standing_knee, standing_ankle = 23, 25, 27
        raised_hip,   raised_knee,   raised_ankle   = 24, 26, 28

    left_wrist  = lm(15)
    right_wrist = lm(16)
    left_shoulder  = lm(11)
    right_shoulder = lm(12)
    left_hip  = lm(23)
    right_hip = lm(24)

    # ── CHECK 1: Standing leg should be straight (angle ~170-180°) ──
    standing_leg_angle = calculate_angle(
        pt(standing_hip), pt(standing_knee), pt(standing_ankle)
    )
    if standing_leg_angle < 160:
        feedback.append(f"Straighten your standing leg (angle: {standing_leg_angle:.0f}°)")

    # ── CHECK 2: Raised knee should be pointing outward ──
    # The raised knee x should be noticeably away from center
    hip_center_x = (lm(23).x + lm(24).x) / 2
    raised_knee_x = lm(raised_knee).x
    knee_offset = abs(raised_knee_x - hip_center_x)
    if knee_offset < 0.08:
        feedback.append("Open your raised knee outward to the side")

    # ── CHECK 3: Raised foot should be off the ground (above standing ankle) ──
    raised_foot_y   = lm(raised_ankle).y
    standing_foot_y = lm(standing_ankle).y
    if raised_foot_y > standing_foot_y - 0.05:
        feedback.append("Raise your foot higher — place it on your inner thigh or calf")

    # ── CHECK 4: Arms should be raised (wrists above shoulders) ──
    if left_wrist.y > left_shoulder.y - 0.05:
        feedback.append("Raise your left arm above your head")
    if right_wrist.y > right_shoulder.y - 0.05:
        feedback.append("Raise your right arm above your head")

    # ── CHECK 5: Spine should be upright (shoulders level with hips horizontally) ──
    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    hip_mid_x      = (left_hip.x + right_hip.x) / 2
    lean = abs(shoulder_mid_x - hip_mid_x)
    if lean > 0.07:
        feedback.append("Stand upright — you're leaning to the side")

    is_correct = len(feedback) == 0
    return is_correct, feedback

def classify(classifier_model="../models/pose_classifier.h5",encoder_model="../models/label_encoder.pkl"):
    # --- Load Model & Encoder ---
    model = tf.keras.models.load_model(classifier_model)
    with open(encoder_model, "rb") as f:
        le = pickle.load(f)

    CONFIDENCE_THRESHOLD = 0.6

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            label_text = "No Pose Detected"
            confidence_text = ""

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                data = []
                for lm in landmarks:
                    data.extend([lm.x, lm.y, lm.z])

                x = np.array(data).reshape(1, -1)
                probs = model.predict(x, verbose=0)[0]
                top_idx = np.argmax(probs)
                confidence = probs[top_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    label_text = le.inverse_transform([top_idx])[0]
                    color = (0, 255, 0)
                    if label_text == "TreePose" and results.pose_landmarks:
                        is_correct, feedback = check_tree_pose(results.pose_landmarks.landmark)

                        if is_correct:
                            cv2.putText(image, "✓ Good Form!", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            for i, tip in enumerate(feedback):
                                cv2.putText(image, f"• {tip}", (10, 120 + i * 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    label_text = "Unknown Pose"
                    color = (0, 165, 255)

                confidence_text = f"{confidence * 100:.1f}%"
            else:
                color = (0, 0, 255)

            cv2.putText(image, label_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(image, confidence_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("AI Yoga Assist", image)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()
