import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

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
