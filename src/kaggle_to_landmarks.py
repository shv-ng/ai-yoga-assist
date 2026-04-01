import cv2
import mediapipe as mp
import csv
import os
import glob

def process_kaggle_data(input_dir="./data/image/", output_csv="data/kaggle_landmarks.csv"):
    mp_pose = mp.solutions.pose
    
    # Mapping kaggle folder names to our project labels
    LABEL_MAP = {
        "downwarddog": "DownwardDog",
        "goddesspose": "GoddessPose",
        "treepose": "TreePose",
        "warriorpose": "WarriorPose",
        "chairpose": "ChairPose",
        "cobrapose": "CobraPose",
        "bridgepose": "BridgePose", # Added based on common kaggle yoga sets
        "corpsepose": "CorpsePose",
        "happybabypose": "HappyBabyPose",
        "supinetwist": "SupineTwist"
    }

    # Initialize CSV with Headers
    header = ["frame_id", "sequence_id", "label"]
    for i in range(33):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    sequence_id = 100000
    label_counts = {}

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Iterate through subdirectories
        for folder_name in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # Normalize folder name for mapping
            clean_folder_name = folder_name.lower().replace(" ", "").replace("_", "")
            label = LABEL_MAP.get(clean_folder_name, folder_name)
            
            print(f"Processing label: {label} (from folder: {folder_name})...")
            count = 0
            
            # Get all images in the folder
            image_paths = glob.glob(os.path.join(folder_path, "*.[jJ][pP]*[gG]")) + \
                          glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
            
            for img_path in image_paths:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    current_frame_data = []
                    for lm in landmarks:
                        current_frame_data.extend([lm.x, lm.y, lm.z])
                    
                    # Each image is treated as a single-frame sequence
                    with open(output_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([0, sequence_id, label] + current_frame_data)
                    
                    sequence_id += 1
                    count += 1
            
            label_counts[label] = count
            print(f"Finished {label}: {count} images processed.")

    print("\nFinal Count per Label:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    print(f"\nTotal sequences created: {sequence_id - 100000}")

if __name__ == "__main__":
    process_kaggle_data()
