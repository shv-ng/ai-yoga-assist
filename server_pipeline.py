"""
server_pipeline.py
------------------
Wires together MediaPipe → LandmarkPipeline → classifier → corrections →
voice file generation into a single run_pipeline(frame) call for server.py.

Loaded ONCE at FastAPI startup; all heavy objects are module-level singletons.
"""

import os
import pickle
import hashlib
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from src.pipeline import LandmarkPipeline
from src.corrections import check_pose

# ── Config ────────────────────────────────────────────────────────────────────
CLASSIFIER_MODEL     = "models/pose_classifier.h5"
ENCODER_PATH         = "models/label_encoder.pkl"
VOICE_DIR            = Path("voice_files")
CONFIDENCE_THRESHOLD = 0.6
VISIBILITY_THRESHOLD = 0.6

_REQUIRED_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

# ── Singletons (initialised once at import time) ──────────────────────────────
_model   = tf.keras.models.load_model(CLASSIFIER_MODEL)
with open(ENCODER_PATH, "rb") as _f:
    _le = pickle.load(_f)

_lm_pipeline  = LandmarkPipeline(smooth_alpha=0.4)
_mp_pose      = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

VOICE_DIR.mkdir(parents=True, exist_ok=True)

# ── Voice file cache  {message_hash → voice_id int} ──────────────────────────
_voice_cache:  dict[str, int] = {}
_voice_counter: int           = 0
_voice_lock                   = threading.Lock()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_visibility(landmarks) -> tuple[bool, str]:
    for idx in _REQUIRED_LANDMARK_INDICES:
        if landmarks[idx].visibility < VISIBILITY_THRESHOLD:
            return False, str(idx)
    return True, ""


def _generate_voice_file(message: str) -> int:
    """
    Return a voice_id for *message*.

    If the message was seen before, reuse the existing file.
    Otherwise generate a new .mp3 via gTTS and cache it.
    Falls back to pyttsx3 save_to_file if gTTS is unavailable.
    """
    global _voice_counter

    key = hashlib.md5(message.encode()).hexdigest()

    with _voice_lock:
        if key in _voice_cache:
            return _voice_cache[key]

        _voice_counter += 1
        vid  = _voice_counter
        path = VOICE_DIR / f"{vid:03d}.mp3"

        _voice_cache[key] = vid   # reserve slot before releasing lock

    # Generate outside the lock so other threads aren't blocked
    _synthesise(message, path)
    logging.info("Voice file created: %s → %s", message, path)
    return vid


def _synthesise(text: str, path: Path):
    """Try gTTS first; fall back to pyttsx3 save_to_file."""
    try:
        from gtts import gTTS
        gTTS(text=text, lang="en", slow=False).save(str(path))
        return
    except Exception as e:
        logging.warning("gTTS failed (%s), trying pyttsx3", e)

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 0.9)
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        engine.stop()
        return
    except Exception as e:
        logging.error("pyttsx3 also failed: %s", e)

    # Last resort — write a tiny silent WAV renamed .mp3 so the route
    # doesn't 404 (ESP32 will just play silence rather than hang).
    path.write_bytes(b"")


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline_landmarks(landmarks_list: list) -> int | None:
    """
    Client-side MediaPipe pipeline. Receives 33 landmarks from the client.

    Parameters
    ----------
    landmarks_list : list of dicts or lists
        Raw landmarks sent from client-side MediaPipe.

    Returns
    -------
    voice_id : int | None
    """
    # Convert list to MediaPipe-like objects (must have .x, .y, .z, .visibility)
    from types import SimpleNamespace
    landmarks = [SimpleNamespace(**lm) if isinstance(lm, dict) else lm for lm in landmarks_list]

    all_visible, _ = _check_visibility(landmarks)

    if not all_visible:
        _lm_pipeline.reset()
        vid = _generate_voice_file("Please step into frame fully")
        return vid

    # Preprocess
    feat_vec  = _lm_pipeline.process_for_classify(landmarks)
    processed = _lm_pipeline.process(landmarks)

    # Classify
    probs      = _model.predict(feat_vec.reshape(1, -1), verbose=0)[0]
    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        _lm_pipeline.reset()
        return None

    label = _le.inverse_transform([top_idx])[0]
    is_correct, corrections = check_pose(label, processed)

    if is_correct:
        return None   # or "Good form!" if you want constant feedback

    if not corrections:
        return None

    top = corrections[0]
    vid = _generate_voice_file(top["message"])
    return vid


def run_pipeline(frame: np.ndarray) -> int | None:
    """
    Full per-frame pipeline.

    Parameters
    ----------
    frame : np.ndarray
        BGR image decoded from the JPEG bytes sent by the ESP32.

    Returns
    -------
    voice_id : int | None
        ID of the .mp3 file to fetch, or None if nothing should be spoken
        this frame (pose correct, low confidence, no landmarks, etc.).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _mp_pose.process(rgb)

    if not results.pose_landmarks:
        _lm_pipeline.reset()
        vid = _generate_voice_file("Please step into frame, no body detected")
        return vid

    landmarks = results.pose_landmarks.landmark
    all_visible, _ = _check_visibility(landmarks)

    if not all_visible:
        _lm_pipeline.reset()
        vid = _generate_voice_file("Please step into frame fully")
        return vid

    # Preprocess
    feat_vec  = _lm_pipeline.process_for_classify(landmarks)
    processed = _lm_pipeline.process(landmarks)

    # Classify
    probs      = _model.predict(feat_vec.reshape(1, -1), verbose=0)[0]
    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        _lm_pipeline.reset()
        return None   # too uncertain — stay silent

    label = _le.inverse_transform([top_idx])[0]
    is_correct, corrections = check_pose(label, processed)

    if is_correct:
        vid = _generate_voice_file("Good form! Hold this position.")
        return vid

    if not corrections:
        return None

    # Pick highest-severity correction
    top = corrections[0]   # already sorted by severity desc in check_pose()
    vid = _generate_voice_file(top["message"])
    return vid
