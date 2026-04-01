"""
client.py
---------
Client-side script for AI Yoga Assist.
Captures webcam, extracts landmarks via MediaPipe, sends them to the server,
and provides real-time visual and voice feedback.
"""

import cv2
import mediapipe as mp
import numpy as np
import requests
import argparse
import time
from src.feedback import FeedbackManager

# ── Config ────────────────────────────────────────────────────────────────────
REQUIRED_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
VISIBILITY_THRESHOLD = 0.6
COLOR_MAP = {
    3: (0, 0, 255),    # Red (Severity 3)
    2: (0, 165, 255),  # Orange (Severity 2)
    1: (0, 255, 255),  # Yellow (Severity 1)
}

def parse_args():
    parser = argparse.ArgumentParser(description="AI Yoga Assist Client")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--lang", type=str, choices=["en", "hi"], default="en", help="Voice language")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize Feedback Manager (Local Voice)
    fm = FeedbackManager(cooldown_seconds=12.0, tick_seconds=5.0, lang=args.lang)
    fm.start()
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    print(f"Client started. Connecting to server at {args.server}...")
    
    server_online = True

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
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
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
                    # FIX: Voice feedback for visibility
                    msg = f"Step into frame, {missing_part} not visible"
                    if args.lang == "hi":
                        msg = f"frame में आएं, आपका {missing_part} नहीं दिख रहा"
                    fm.update([{"key": "frame_visibility", "message": msg, "severity": 3}])
                else:
                    # Extract landmarks as list of [x, y, z]
                    lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                    
                    # POST to server
                    try:
                        response = requests.post(
                            f"{args.server}/process",
                            json={"landmarks": lm_list},
                            timeout=0.5
                        )
                        if response.status_code == 200:
                            data = response.json()
                            pose_label = data.get("pose") or "None"
                            confidence = data.get("confidence", 0.0)
                            is_correct = data.get("is_correct", False)
                            corrections = data.get("corrections", [])
                            server_online = True
                            
                            # Update FeedbackManager
                            if is_correct:
                                fm.update_good()
                            else:
                                # HINDI SUPPORT: Use message_hi if lang is hi
                                processed_corrections = []
                                for c in corrections:
                                    msg = c["message_hi"] if args.lang == "hi" and "message_hi" in c else c["message"]
                                    processed_corrections.append({
                                        "key": c["key"],
                                        "message": msg,
                                        "severity": c["severity"]
                                    })
                                fm.update(processed_corrections)
                        else:
                            status_msg = f"Server Error: {response.status_code}"
                    except requests.exceptions.RequestException:
                        server_online = False

            else:
                status_msg = "No body detected"
                # FIX: Voice feedback for no body
                msg = "come into frame"
                if args.lang == "hi":
                    msg = "कोई body नहीं मिली, frame में आएं"
                fm.update([{"key": "no_body", "message": msg, "severity": 3}])

            # ── Drawing ───────────────────────────────────────────────────────
            
            if not server_online:
                cv2.putText(frame, "Server offline", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if status_msg:
                cv2.putText(frame, status_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Pose info
            cv2.putText(frame, f"Pose: {pose_label} ({confidence:.1%}) | Lang: {args.lang.upper()}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw up to 3 corrections
            for i, corr in enumerate(corrections[:3]):
                color = COLOR_MAP.get(corr["severity"], (255, 255, 255))
                y_pos = 120 + i * 30
                msg = corr["message_hi"] if args.lang == "hi" and "message_hi" in corr else corr["message"]
                cv2.putText(frame, f"! {msg}", (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Voice feedback status
            cv2.putText(frame, f"Voice: {fm.last_message}", (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("AI Yoga Assist Client", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('h') or key == ord('H'):
                args.lang = "hi" if args.lang == "en" else "en"
                fm.set_lang(args.lang)

    finally:
        fm.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
