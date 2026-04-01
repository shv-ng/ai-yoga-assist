"""
pipeline.py
-----------
Landmark preprocessing: normalization + temporal smoothing.

Two steps applied before classification and correction:

1. Normalize
   Raw MediaPipe coordinates are image-relative (0-1 range, top-left origin).
   We normalize relative to the person's own body so the model is invariant to
   their position in the frame and their distance from the camera.

   Method:
     - Translate so the hip midpoint is the origin
     - Scale so the torso height (hip-mid → shoulder-mid) == 1.0
     - This makes the feature vector body-size invariant

2. Temporal smoothing
   MediaPipe landmarks jitter frame-to-frame. We apply a simple exponential
   moving average (EMA) per landmark across consecutive frames.
   Alpha controls the smoothing strength (lower = smoother but more lag).

Usage:
    from pipeline import LandmarkPipeline

    pipeline = LandmarkPipeline(smooth_alpha=0.4)

    # in your frame loop, after mediapipe:
    landmarks = results.pose_landmarks.landmark
    processed  = pipeline.process(landmarks)   # list of SimpleNamespace(x, y, z)
    feature_vec = pipeline.to_feature_vector(processed)  # flat np.array for model

    # reset between sessions or when person leaves frame:
    pipeline.reset()
"""

import numpy as np
from types import SimpleNamespace

N_LANDMARKS = 33

# MediaPipe indices used for normalization
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12


class LandmarkPipeline:
    """
    Stateful per-session landmark preprocessor.

    Parameters
    ----------
    smooth_alpha : float
        EMA weight for the current frame (0 < alpha <= 1).
        1.0 = no smoothing, 0.2 = heavy smoothing.
    """

    def __init__(self, smooth_alpha: float = 0.4):
        if not 0 < smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in (0, 1]")
        self.alpha = smooth_alpha
        self._ema: np.ndarray | None = None  # shape (N_LANDMARKS, 3)

    def reset(self):
        """Call when a new person enters frame or session restarts."""
        self._ema = None

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, landmarks) -> list:
        """
        Full pipeline: smooth → normalize.

        Parameters
        ----------
        landmarks : mediapipe landmark list (results.pose_landmarks.landmark)

        Returns
        -------
        list of SimpleNamespace(x, y, z) — same structure as mediapipe landmarks
        so existing correction functions work without modification.
        """
        raw = self._to_array(landmarks)  # (33, 3)
        smoothed = self._smooth(raw)  # (33, 3)
        normed = self._normalize(smoothed)  # (33, 3)
        return self._to_landmark_list(normed)

    def to_feature_vector(self, processed_landmarks) -> np.ndarray:
        """
        Flatten processed landmarks to a 1-D feature vector for the classifier.
        Shape: (99,)  — 33 landmarks × (x, y, z)
        """
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in processed_landmarks], dtype=np.float32
        ).flatten()

    def process_for_classify(self, landmarks) -> np.ndarray:
        """
        Normalize only — NO EMA smoothing.
        Use this to generate the feature vector for the classifier.
        Matches exactly what the notebook did during training.
        """
        raw = self._to_array(landmarks)  # (33, 3)
        normed = self._normalize(raw)  # (33, 3) — no smoothing
        lm_list = self._to_landmark_list(normed)
        return self.to_feature_vector(lm_list)  # (99,) flat array

    # ── Internal steps ────────────────────────────────────────────────────────

    def _to_array(self, landmarks) -> np.ndarray:
        """MediaPipe landmark list → (33, 3) float32 array."""
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    def _smooth(self, raw: np.ndarray) -> np.ndarray:
        """Exponential moving average across frames."""
        if self._ema is None:
            self._ema = raw.copy()
        else:
            self._ema = self.alpha * raw + (1.0 - self.alpha) * self._ema
        return self._ema.copy()

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Translate + scale so the result is body-position and body-size invariant.

        Origin  → midpoint of the two hips
        Scale   → distance from hip-midpoint to shoulder-midpoint (torso height)
        """
        # Hip midpoint as origin
        hip_mid = (arr[_LEFT_HIP] + arr[_RIGHT_HIP]) / 2.0
        translated = arr - hip_mid

        # Torso height for scale
        shoulder_mid = (arr[_LEFT_SHOULDER] + arr[_RIGHT_SHOULDER]) / 2.0
        torso_height = np.linalg.norm(shoulder_mid - hip_mid)

        if torso_height < 1e-6:
            # Person not detected properly — return translated but unscaled
            return translated

        normalized = translated / torso_height
        return normalized

    def _to_landmark_list(self, arr: np.ndarray) -> list:
        """(33, 3) array → list of SimpleNamespace(x, y, z)."""
        return [
            SimpleNamespace(x=float(arr[i, 0]), y=float(arr[i, 1]), z=float(arr[i, 2]))
            for i in range(N_LANDMARKS)
        ]
