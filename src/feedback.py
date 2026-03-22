"""
Priority-queue based voice feedback manager.

How it works:
  1. After each pose check, call feedback_manager.update(corrections)
  2. The manager holds a queue of pending corrections, deduplicated by key
  3. A background thread speaks the highest-severity pending correction
  4. Each correction key has an individual cooldown — once spoken it won't
     repeat until the cooldown expires (default 6 seconds)
  5. Corrections that are resolved (pose improved) are dropped automatically

Usage:
    from feedback import FeedbackManager

    fm = FeedbackManager(cooldown_seconds=6)
    fm.start()

    # in your frame loop:
    is_correct, corrections = check_pose(label, landmarks)
    fm.update(corrections)

    # on exit:
    fm.stop()
"""

import threading
import time
import queue
import logging

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning(
        "pyttsx3 not installed — voice output disabled. Run: pip install pyttsx3"
    )


# ─────────────────────────────────────────────
#  CorrectionEntry  (one slot in the queue)
# ─────────────────────────────────────────────


class CorrectionEntry:
    def __init__(self, key: str, message: str, severity: int):
        self.key = key
        self.message = message
        self.severity = severity

    # heapq is a min-heap, so invert severity for max-priority behaviour
    def __lt__(self, other):
        return self.severity > other.severity


# ─────────────────────────────────────────────
#  FeedbackManager
# ─────────────────────────────────────────────


class FeedbackManager:
    """
    Thread-safe voice feedback manager with per-key cooldowns.

    Parameters
    ----------
    cooldown_seconds : float
        How long (seconds) before the same correction can be spoken again.
    speak_interval : float
        Minimum gap between any two spoken messages (avoids rapid-fire output).
    rate : int
        TTS speech rate (words per minute). pyttsx3 default is ~200.
    volume : float
        TTS volume 0.0 – 1.0.
    """

    def __init__(
        self,
        cooldown_seconds: float = 6.0,
        speak_interval: float = 4.0,
        rate: int = 155,
        volume: float = 0.9,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.speak_interval = speak_interval

        # key → timestamp when last spoken
        self._cooldowns: dict[str, float] = {}

        # current active correction keys (updated each frame)
        self._active_keys: set[str] = set()

        # internal queue (thread-safe)
        self._q: queue.PriorityQueue = queue.PriorityQueue()

        # lock protecting cooldowns + active keys
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._speak_thread = None

        # TTS engine lives on the speaker thread (pyttsx3 is not thread-safe)
        self._rate = rate
        self._volume = volume
        self._engine = None  # initialised inside speaker thread

        # last spoken timestamp
        self._last_spoken: float = 0.0

        # for display / debug — last spoken message
        self.last_message: str = ""

    # ── Public API ────────────────────────────

    def start(self):
        """Start the background speaker thread."""
        self._stop_event.clear()
        self._speak_thread = threading.Thread(
            target=self._speaker_loop,
            name="FeedbackSpeaker",
            daemon=True,
        )
        self._speak_thread.start()
        logging.info("FeedbackManager started")

    def stop(self):
        """Gracefully stop the speaker thread."""
        self._stop_event.set()
        if self._speak_thread:
            self._speak_thread.join(timeout=3)
        logging.info("FeedbackManager stopped")

    def update(self, corrections: list):
        """
        Called every frame (or every N frames) with the latest correction list.
        Corrections that have been resolved (no longer in list) are dropped.
        New corrections above cooldown threshold are queued.

        Parameters
        ----------
        corrections : list[dict]
            Each dict: {"key": str, "message": str, "severity": int}
        """
        now = time.time()

        with self._lock:
            new_active = {c["key"] for c in corrections}
            self._active_keys = new_active

            for c in corrections:
                key = c["key"]
                message = c["message"]
                severity = c["severity"]

                # Skip if on cooldown
                last_time = self._cooldowns.get(key, 0.0)
                if now - last_time < self.cooldown_seconds:
                    continue

                # Queue it (duplicates are fine — speaker thread checks active_keys)
                self._q.put(CorrectionEntry(key, message, severity))

    def update_good(self):
        """Call this when the pose is fully correct to speak a positive cue."""
        with self._lock:
            self._active_keys = set()
        self._q.put(
            CorrectionEntry("__good__", "Good form! Hold this position.", severity=0)
        )

    # ── Speaker thread ─────────────────────────

    def _speaker_loop(self):
        if TTS_AVAILABLE:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)

        while not self._stop_event.is_set():
            try:
                entry: CorrectionEntry = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            now = time.time()

            # Enforce minimum gap between any two utterances
            if now - self._last_spoken < self.speak_interval:
                # Put it back only if still active, else discard
                with self._lock:
                    if entry.key in self._active_keys or entry.key == "__good__":
                        self._q.put(entry)
                time.sleep(0.2)
                continue

            # Drop if correction is no longer active (pose was fixed)
            if entry.key != "__good__":
                with self._lock:
                    if entry.key not in self._active_keys:
                        continue  # silently discard — already resolved

            # Check cooldown again (may have changed while waiting in queue)
            if entry.key != "__good__":
                with self._lock:
                    last_time = self._cooldowns.get(entry.key, 0.0)
                    if now - last_time < self.cooldown_seconds:
                        continue
                    self._cooldowns[entry.key] = now

            self.last_message = entry.message
            self._last_spoken = now
            self._speak(entry.message)

    def _speak(self, text: str):
        if TTS_AVAILABLE and self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logging.warning(f"TTS error: {e}")
        else:
            # Fallback: just print so development works without a speaker
            print(f"[VOICE] {text}")
