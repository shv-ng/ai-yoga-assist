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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize Feedback Manager (Local Voice)
    fm = FeedbackManager(cooldown_seconds=12.0, tick_seconds=5.0)
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
                                fm.update(corrections)
                        else:
                            status_msg = f"Server Error: {response.status_code}"
                    except requests.exceptions.RequestException:
                        server_online = False

            else:
                status_msg = "No body detected"

            # ── Drawing ───────────────────────────────────────────────────────
            
            if not server_online:
                cv2.putText(frame, "Server offline", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if status_msg:
                cv2.putText(frame, status_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Pose info
            cv2.putText(frame, f"Pose: {pose_label} ({confidence:.1%})", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw up to 3 corrections
            for i, corr in enumerate(corrections[:3]):
                color = COLOR_MAP.get(corr["severity"], (255, 255, 255))
                y_pos = 120 + i * 30
                cv2.putText(frame, f"! {corr['message']}", (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Voice feedback status
            cv2.putText(frame, f"Voice: {fm.last_message}", (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("AI Yoga Assist Client", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        fm.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
