"""
client.py
---------
Client-side script for AI Yoga Assist.
Captures webcam, extracts landmarks via MediaPipe, sends them to the server,
and provides real-time visual and voice feedback.
"""

import cv2
import mediapipe as mp
import requests
import argparse
from src.feedback import FeedbackManager

# ── Config ────────────────────────────────────────────────────────────────────
REQUIRED_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
VISIBILITY_THRESHOLD = 0.6
COLOR_MAP = {
    3: (0, 0, 255),  # Red (Severity 3)
    2: (0, 165, 255),  # Orange (Severity 2)
    1: (0, 255, 255),  # Yellow (Severity 1)
}

POSE_DESCRIPTIONS = {
    "TreePose": {
        "en": "To get into Tree Pose, stand on one leg and place the other foot on your inner thigh.",
        "hi": "Tree Pose mein aane ke liye, ek pair par khade ho jayein aur doosre pair ko apni jaangh par rakhein.",
    },
    "ChairPose": {
        "en": "To get into Chair Pose, bend your knees and sink your hips back as if sitting in a chair.",
        "hi": "Chair Pose mein aane ke liye, apne ghutno ko modein aur hips ko peeche jhukayein jaise chair par baithe hon.",
    },
    "WarriorPose": {
        "en": "To get into Warrior 2, step your feet wide, bend your front knee, and extend your arms out.",
        "hi": "Warrior 2 mein aane ke liye, apne pairon ko chaudha failayein, agle ghutne ko modein aur apne haath bahar failayein.",
    },
    "CobraPose": {
        "en": "To get into Cobra Pose, lie on your stomach and lift your chest off the floor using your back muscles.",
        "hi": "Cobra Pose mein aane ke liye, apne pet ke bal letein aur apni chest ko farsh se upar uthayein.",
    },
    "DownwardDog": {
        "en": "To get into Downward Dog, press your hands and feet into the floor and lift your hips toward the ceiling.",
        "hi": "Downward Dog mein aane ke liye, apne haathon aur pairon ko farsh par dabayein aur apne hips ko chhat ki ore uthayein.",
    },
    "GoddessPose": {
        "en": "To get into Goddess Pose, step your feet wide, turn your toes out, and bend your knees into a wide squat.",
        "hi": "Goddess Pose mein aane ke liye, apne pairon ko chaudha failayein, toes ko bahar modein aur ghutno ko modkar squat karein.",
    },
    "CorpsePose": {
        "en": "To get into Corpse Pose, lie flat on your back and relax your entire body.",
        "hi": "Corpse Pose mein aane ke liye, apni peeth ke bal letein aur apne pure sharir ko relax karein.",
    },
    "BridgePose": {
        "en": "To get into Bridge Pose, lie on your back, bend your knees, and lift your hips off the floor.",
        "hi": "Bridge Pose mein aane ke liye, apni peeth ke bal letein, ghutno ko modein aur apne hips ko farsh se upar uthayein.",
    },
    "SupineTwist": {
        "en": "To get into Supine Twist, lie on your back and drop one knee over to the opposite side.",
        "hi": "Supine Twist mein aane ke liye, apni peeth ke bal letein aur ek ghutne ko doosri side jhukayein.",
    },
    "HappyBabyPose": {
        "en": "To get into Happy Baby, lie on your back, pull your knees to your chest, and hold the outside of your feet.",
        "hi": "Happy Baby mein aane ke liye, apni peeth ke bal letein, ghutno ko chest tak layein aur apne pairon ke bahari hisse ko pakdein.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="AI Yoga Assist Client")
    parser.add_argument(
        "--server", type=str, default="http://localhost:8000", help="Server URL"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--lang", type=str, choices=["en", "hi"], default="hi", help="Voice language"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    # Initialize Feedback Manager (Local Voice)
    fm = FeedbackManager(lang=args.lang)
    fm.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    print(f"Client started. Connecting to server at {args.server}...")

    server_online = True
    last_pose_label = ""

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose.process(rgb_frame)

            # Draw landmarks (skeleton) on frame
            if results.pose_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            status_msg = ""
            pose_label = "None"
            confidence = 0.0
            corrections = []
            is_correct = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Visibility check
                all_visible = True
                missing_part = ""
                for idx in REQUIRED_LANDMARK_INDICES:
                    if landmarks[idx].visibility < VISIBILITY_THRESHOLD:
                        all_visible = False
                        # Get a rough name for the landmark for UI feedback
                        missing_part = mp.solutions.pose.PoseLandmark(idx).name
                        break

                if not all_visible:
                    status_msg = f"Step into frame - {missing_part} not visible"
                    fm.update(
                        [
                            {
                                "key": "frame_visibility",
                                "message": f"Step into frame, {missing_part} not visible",
                                "message_hi": f"frame में आएं, आपका {missing_part} नहीं दिख रहा",
                                "severity": 3,
                            }
                        ]
                    )
                else:
                    # Extract landmarks as list of [x, y, z]
                    lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks]

                    # POST to server
                    try:
                        response = requests.post(
                            f"{args.server}/process",
                            json={"landmarks": lm_list},
                            timeout=0.5,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            pose_label = data.get("pose") or "None"
                            confidence = data.get("confidence", 0.0)
                            is_correct = data.get("is_correct", False)
                            corrections = data.get("corrections", [])
                            server_online = True

                            # Announcement on pose transition
                            if pose_label != "None" and pose_label != last_pose_label:
                                desc = POSE_DESCRIPTIONS.get(pose_label, {})
                                msg = desc.get(args.lang, pose_label)
                                fm.speak_immediate(msg)
                                last_pose_label = pose_label

                            # Update FeedbackManager
                            if is_correct:
                                fm.update_good()
                            else:
                                fm.update(corrections)
                        else:
                            status_msg = f"Server Error: {response.status_code}"
                    except requests.exceptions.RequestException:
                        server_online = False

            else:
                status_msg = "No body detected"
                fm.update(
                    [
                        {
                            "key": "no_body",
                            "message": "come into frame",
                            "message_hi": "कोई body नहीं मिली, frame में आएं",
                            "severity": 3,
                        }
                    ]
                )

            # ── Drawing ───────────────────────────────────────────────────────

            if not server_online:
                cv2.putText(
                    frame,
                    "Server offline",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if status_msg:
                cv2.putText(
                    frame,
                    status_msg,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # Pose info
            cv2.putText(
                frame,
                f"Pose: {pose_label} ({confidence:.1%}) | Lang: {args.lang.upper()}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Draw up to 3 corrections
            for i, corr in enumerate(corrections[:3]):
                color = COLOR_MAP.get(corr["severity"], (255, 255, 255))
                y_pos = 120 + i * 30
                msg = (
                    corr["message_hi"]
                    if args.lang == "hi" and "message_hi" in corr
                    else corr["message"]
                )
                cv2.putText(
                    frame,
                    f"! {msg}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Voice feedback status
            cv2.putText(
                frame,
                f"Voice: {fm.last_message}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("AI Yoga Assist Client", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("h") or key == ord("H"):
                args.lang = "hi" if args.lang == "en" else "en"
                fm.set_lang(args.lang)

    finally:
        fm.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
