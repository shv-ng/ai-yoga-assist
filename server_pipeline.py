"""
server_pipeline.py
------------------
Yoga pose classification and correction pipeline for the server.
"""

import pickle
import numpy as np
import tensorflow as tf
from types import SimpleNamespace

from src.pipeline import LandmarkPipeline
from src.corrections import check_pose

# ── Config ────────────────────────────────────────────────────────────────────
CLASSIFIER_MODEL     = "models/pose_classifier.h5"
ENCODER_PATH         = "models/label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.6

# ── Singletons (initialised once at import time) ──────────────────────────────
_model = tf.keras.models.load_model(CLASSIFIER_MODEL)
with open(ENCODER_PATH, "rb") as _f:
    _le = pickle.load(_f)

_lm_pipeline = LandmarkPipeline(smooth_alpha=0.4)

# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(landmarks_list: list) -> dict:
    """
    Process landmarks, classify pose, and check for corrections.

    Parameters
    ----------
    landmarks_list : list of [x, y, z]
        33 landmarks from client-side MediaPipe.

    Returns
    -------
    dict
        {"pose": str|None, "confidence": float, "is_correct": bool, "corrections": list}
    """
    # Convert list of [x,y,z] to numpy array (33, 3)
    landmarks_array = np.array(landmarks_list)
    
    # Create SimpleNamespace objects for LandmarkPipeline (expects .x, .y, .z)
    landmarks = [SimpleNamespace(x=lm[0], y=lm[1], z=lm[2]) for lm in landmarks_array]

    # Preprocess
    feat_vec  = _lm_pipeline.process_for_classify(landmarks)
    processed = _lm_pipeline.process(landmarks)

    # Classify
    probs      = _model.predict(feat_vec.reshape(1, -1), verbose=0)[0]
    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        _lm_pipeline.reset()
        return {
            "pose": None,
            "confidence": confidence,
            "is_correct": False,
            "corrections": []
        }

    label = _le.inverse_transform([top_idx])[0]
    is_correct, corrections = check_pose(label, processed)

    return {
        "pose": str(label),
        "confidence": confidence,
        "is_correct": bool(is_correct),
        "corrections": corrections
    }
