import cv2
import mediapipe as mp
import requests
import argparse
import time

def run_client(server_url):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    
    last_voice_time = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in results.pose_landmarks.landmark
                ]
                
                try:
                    resp = requests.post(f"{server_url}/process_landmarks", json={"landmarks": landmarks}, timeout=0.5)
                    if resp.status_code == 200:
                        vid = resp.json().get("voice_id")
                        if vid and (time.time() - last_voice_time > 5):
                            print(f"Playing voice ID: {vid}")
                            # In a real app, we'd fetch and play the MP3 here.
                            # For the prototype, we'll just log it.
                            last_voice_time = time.time()
                except Exception as e:
                    print(f"Server error: {e}")

            cv2.imshow("AI Yoga Assist Client", image)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()
    run_client(args.server)
