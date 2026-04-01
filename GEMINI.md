# GEMINI.md

## Project Overview
**AI Yoga Assist** is a real-time yoga pose classification and correction system. It uses computer vision (MediaPipe) to detect body landmarks, processes them through a normalized pipeline, and employs a machine learning model (MLP) to identify poses. A rule-based engine then evaluates the pose for biomechanical correctness and provides spoken feedback.

### Key Technologies
- **Python 3.10+**
- **MediaPipe:** Body landmark detection (33 points).
- **TensorFlow/TFLite:** Pose classification (Tree, Chair, Warrior II, Cobra, Downward Dog, Goddess).
- **FastAPI:** Server for ESP32 integration.
- **pyttsx3 / gTTS:** Voice feedback system.
- **OpenCV:** Video processing and visualization.

### Architecture
1. **Detection:** MediaPipe extracts 33 landmarks.
2. **Normalization:** `LandmarkPipeline` transforms landmarks to hip-origin and scales them by torso height (invariance to distance/position).
3. **Classification:** MLP model predicts the pose label.
4. **Correction:** `src/corrections.py` runs pose-specific rule checks (angles, alignment).
5. **Feedback:** `FeedbackManager` prioritizes and speaks corrections.
6. **Logging:** `SessionLogger` records session stats to `logs/` as JSON.

---

## Building and Running

### Environment Setup
The project uses `uv` or `pip`. Install dependencies:
```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy pyttsx3 fastapi uvicorn
```

### Key Commands
- **Main Entry Point:** `python main.py`
  - Option 1: Data Collection (saves to `data/poses.csv`).
  - Option 2: Live Classification & Correction (webcam mode).
- **ESP32 Server:** `uvicorn server:app --host 0.0.0.0 --port 8000`
  - Handles JPEG frames and serves MP3 voice feedback.
- **Training:** Run `notebooks/pose_classifier.py` or the Jupyter notebook to retrain the model.

---

## Development Conventions

### Coordinate System (Normalised Space)
All correction logic in `src/corrections.py` uses "Torso Units" (tu):
- **Origin (0,0):** Midpoint of the hips.
- **Scale (1.0 tu):** Distance from hip midpoint to shoulder midpoint.
- **Axes:** X increases right, Y increases **downward** (screen space).
- **Typical Ranges:** Shoulders at Y ≈ -1.0, Ankles at Y ≈ +1.5.

### Pose Corrections
Checkers in `src/corrections.py` return `(is_correct: bool, corrections: list[dict])`.
- **Severity Levels:**
  - `3`: Safety risk or fundamental error (e.g., straight leg vs bent).
  - `2`: Major form issue affecting effectiveness.
  - `1`: Fine-tuning / polish.
- **Deduplication:** Each correction has a stable `key` used by the `FeedbackManager` for cooldowns.

### Adding a New Pose
1. **Data Collection:** Use `main.py` (Option 1) to record sequences for the new pose.
2. **Training:** Update the label list in `notebooks/pose_classifier.py` and retrain.
3. **Logic:** Implement a `check_new_pose` function in `src/corrections.py` and register it in `POSE_CHECKERS`.
4. **Voice:** (If using ESP32) Ensure `FeedbackManager` generates corresponding MP3s.

### Testing and Validation
- **Visual Check:** Run `main.py` and verify landmark overlays and classification labels.
- **Logs:** Review JSON files in `logs/` to verify session tracking accuracy.
- **ESP32 Sim:** Use `curl` or a script to POST images to `/process` and verify `voice_id` responses.
