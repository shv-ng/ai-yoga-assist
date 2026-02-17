import cv2
import mediapipe as mp
import csv
import os


def start_collection(csv_path="../data/poses_raw.csv"):
    # --- Setup MediaPipe Drawing ---
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # --- Configuration ---
    LABELS = [
        "TreePose",
        "ChairPose",
        "WarriorPose",
        "CobraPose",
        "DownwardDog",
        "GoddessPose",
    ]
    CSV_FILE = csv_path

    current_capturing = 0
    pose_count = len(LABELS)

    # State Variables
    sequence_id = 0
    frame_id = 0
    is_recording = False

    # 1. Ensure the directory exists
    directory = os.path.dirname(csv_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # 2. Initialize CSV with Headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["frame_id", "sequence_id", "label"]
            # Add x, y, z for all 33 landmarks
            for i in range(33):
                header.extend([f"x{i}", f"y{i}", f"z{i}"])
            writer.writerow(header)

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            if current_capturing >= pose_count:
                break
            LABEL = LABELS[current_capturing]

            success, image = cap.read()
            if not success:
                break

            # Process the image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw landmarks on the image for visual feedback
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # DRAW THE LINES ON THE BODY HERE
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # 3. Record Data Logic
                if is_recording:
                    landmarks = results.pose_landmarks.landmark
                    # Flatten landmarks into a list [x0, y0, z0, x1, y1, z1...]
                    current_frame_data = []
                    for lm in landmarks:
                        current_frame_data.extend([lm.x, lm.y, lm.z])

                    # Combine with metadata and write to CSV
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [frame_id, sequence_id, LABEL] + current_frame_data
                        )

                    frame_id += 1

            # UI Overlays
            status_color = (0, 0, 255) if is_recording else (0, 255, 0)
            status_text = (
                "RECORDING (Press R to stop)"
                if is_recording
                else "IDLE (Press R to Start)"
            )
            cv2.putText(
                image,
                f"Pose: {LABEL} (Press N for Next Pose)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                f"Status: {status_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                image,
                f"Seq ID: {sequence_id} | Frame: {frame_id}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Pose Data Collector", image)

            # Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):  # Start recording a new sequence
                if is_recording:
                    sequence_id += 1
                    print(f"Sequence {sequence_id-1} finished and saved.")
                else:
                    frame_id = 0
                    print(f"Recording Sequence {sequence_id}...")
                is_recording = not is_recording
            elif key == ord("n"):  # Start recording a new sequence
                current_capturing += 1
            elif key == 27:  # ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()
