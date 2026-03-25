# AI Yoga Assist

A real-time yoga pose classifier and correction coach that uses computer vision to detect your pose, identify form errors, and deliver spoken feedback — live from your webcam or from an ESP32 camera module.

---

## Features

- **6 Pose Recognition** — Tree, Chair, Warrior II, Cobra, Downward Dog, Goddess
- **Rule-based Corrections** — Biomechanically grounded checks (joint angles, limb alignment, stance width) with severity levels 1–3
- **Voice Feedback** — Priority-queued TTS via pyttsx3; highest-severity correction spoken every 5 seconds
- **Session Logging** — Per-session JSON logs tracking pose durations, correction frequency, and resolution rate
- **ESP32 Support** — FastAPI server accepts JPEG frames over HTTP and returns pre-generated MP3 voice files
- **Body-Normalised Pipeline** — All landmark features are translated to hip-origin and scaled by torso height, making the model invariant to camera distance and body position

---

## Project Structure

```
.
├── main.py                  # CLI entry point (data collection or live classification)
├── server.py                # FastAPI server for ESP32 integration
├── server_pipeline.py       # Server-side inference pipeline (singleton-based)
│
├── src/
│   ├── realtime.py          # Webcam loop: detect → classify → correct → speak
│   ├── pipeline.py          # LandmarkPipeline: EMA smoothing + body normalisation
│   ├── corrections.py       # Rule-based pose checkers for all 6 poses
│   ├── feedback.py          # FeedbackManager: threaded TTS with cooldown + priority queue
│   ├── session_logger.py    # SessionLogger: per-session stats and JSON log writer
│   └── collect_data.py      # Webcam data collection tool (saves to CSV)
│
├── notebooks/
│   └── pose_classifier.py   # Training notebook (exported as script)
│
├── models/
│   ├── pose_classifier.h5   # Trained Keras model
│   ├── pose_classifier.tflite  # TFLite export (for edge devices)
│   └── label_encoder.pkl    # Scikit-learn LabelEncoder
│
├── data/
│   └── poses.csv            # Collected landmark data (created by collect_data.py)
│
├── logs/                    # Auto-created; one JSON file per session
└── voice_files/             # Auto-created; cached MP3 files for the ESP32 server
```

---

## Setup

### Requirements

```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy
pip install pyttsx3          # local TTS (webcam mode)
pip install gtts             # cloud TTS (server/ESP32 mode, optional)
pip install fastapi uvicorn  # only needed for ESP32 server mode
```

Python 3.10+ is recommended (uses `X | Y` union type hints).

### Models

Place trained model files in `models/`:
- `pose_classifier.h5`
- `label_encoder.pkl`

To train from scratch, collect data first (see below), then run `notebooks/pose_classifier.py`.

---

## Usage

### Interactive CLI

```bash
python main.py
```

Options:
- **1** — Launch webcam data collector
- **2** — Launch live classification and correction
- **q** — Quit

### Webcam Mode (direct)

```python
from src.realtime import classify
classify("./models/pose_classifier.h5", "./models/label_encoder.pkl")
```

Press **ESC** to end the session. A JSON log is saved to `logs/` automatically.

### ESP32 / HTTP Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The ESP32 should:
1. `POST /process` with raw JPEG bytes → receives `{"voice_id": int | null}`
2. `GET /voice/{voice_id}` → streams the `.mp3` file to play

---

## Data Collection

```bash
python main.py  # select option 1
```

Keyboard controls during collection:

| Key | Action |
|-----|--------|
| `R` | Start / stop recording a sequence |
| `N` | Move to the next pose label |
| `ESC` | Exit |

Data is appended to `data/poses.csv`. Each row contains a frame ID, sequence ID, pose label, and flattened x/y/z coordinates for all 33 MediaPipe landmarks.

---

## Training

Open `notebooks/pose_classifier.py` (or the `.ipynb` equivalent). The notebook:

1. Loads `data/poses.csv`
2. Normalises landmarks (hip-origin, torso-height scale) — identical to `pipeline.py`
3. Splits by `sequence_id` to prevent data leakage
4. Trains a 3-layer MLP (128 → 64 → softmax)
5. Saves `pose_classifier.h5`, `label_encoder.pkl`, and a TFLite export

---

## Pose Corrections

Each pose checker in `corrections.py` returns a list of corrections, each with:

| Field | Description |
|-------|-------------|
| `key` | Stable string ID (used for cooldown deduplication) |
| `message` | Human-readable instruction spoken aloud |
| `severity` | 1 = polish, 2 = major form issue, 3 = safety/fundamental error |

Corrections are sorted by severity descending before being passed to the feedback manager. The voice system speaks one correction per tick, always prioritising the highest severity.

---

## Coordinate System

All correction logic operates in **normalised landmark space**:

- **Origin** — midpoint of the two hips
- **Scale** — 1 unit = torso height (hip-mid to shoulder-mid distance)
- **Axes** — x grows right, y grows downward (screen space convention)
- Shoulders sit at approximately y = −1.0; ankles at y ≈ +1.5

This normalisation happens in `LandmarkPipeline.process_for_classify()` (for the model) and `LandmarkPipeline.process()` (for corrections), and mirrors the preprocessing done during training.

---

## Session Logs

Each session produces a JSON file in `logs/session_YYYYMMDD_HHMMSS.json` containing:

- Total duration and correction count
- Resolution rate (fraction of corrections that were fixed during the session)
- Per-pose time breakdown and correction counts
- Full pose interval timeline

---

## Architecture Overview

```
Webcam / ESP32 frame
        │
        ▼
  MediaPipe Pose
        │
        ▼
  Visibility Check ──── (warn if body not fully in frame)
        │
        ▼
  LandmarkPipeline
    ├─ EMA smoothing (corrections path only)
    └─ Hip-origin + torso-scale normalisation
        │
        ├──► Classifier Model  →  Pose Label + Confidence
        │
        └──► check_pose()      →  Corrections list
                │
                ▼
         FeedbackManager  →  Spoken TTS (priority queue, 5s tick)
                │
                ▼
          SessionLogger   →  JSON log on exit
```
