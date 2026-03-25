"""
Priority-queue based voice feedback manager.

Fix: pyttsx3 engine is created fresh for every utterance and immediately
destroyed after. This avoids the well-known pyttsx3 threading bug where
runAndWait() deadlocks after a few calls when the engine is reused across
multiple speak() invocations on a background thread.
"""

import heapq
import threading
import time
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
#  CorrectionEntry
# ─────────────────────────────────────────────

class CorrectionEntry:
    def __init__(self, key: str, message: str, severity: int):
        self.key      = key
        self.message  = message
        self.severity = severity

    def __lt__(self, other):
        return self.severity > other.severity   # max-heap on severity


# ─────────────────────────────────────────────
#  FeedbackManager
# ─────────────────────────────────────────────

class FeedbackManager:
    """
    Thread-safe voice feedback manager driven by a fixed-interval ticker.

    Every tick_seconds the speaker thread picks the highest-severity
    pending correction that is still active and not on cooldown, speaks
    it once using a fresh TTS engine, then waits for the next tick.

    Parameters
    ----------
    cooldown_seconds : float
        How long before the same correction key can be spoken again.
    tick_seconds : float
        Fixed interval between voice feedback attempts (default 5 s).
    rate : int
        TTS speech rate (words per minute).
    volume : float
        TTS volume 0.0 – 1.0.
    """

    def __init__(
        self,
        cooldown_seconds: float = 10.0,
        tick_seconds:     float = 5.0,
        rate:             int   = 155,
        volume:           float = 0.9,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.tick_seconds     = tick_seconds
        self._rate            = rate
        self._volume          = volume

        self._cooldowns:    dict[str, float]  = {}
        self._pending:      list              = []   # heapq
        self._pending_keys: set[str]          = set()
        self._active_keys:  set[str]          = set()
        self._good_pending: bool              = False

        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._speak_thread: threading.Thread | None = None

        self.last_message: str = ""

    # ── Public API ────────────────────────────

    def start(self):
        self._stop_event.clear()
        self._speak_thread = threading.Thread(
            target=self._speaker_loop,
            name="FeedbackSpeaker",
            daemon=True,
        )
        self._speak_thread.start()
        logging.info("FeedbackManager started (tick=%.1fs)", self.tick_seconds)

    def stop(self):
        self._stop_event.set()
        if self._speak_thread:
            self._speak_thread.join(timeout=5)
        logging.info("FeedbackManager stopped")

    def update(self, corrections: list):
        """
        Call every frame with the current correction list.
        Adds new corrections to the heap; resolved ones are dropped at speak time.
        """
        with self._lock:
            self._active_keys = {c["key"] for c in corrections}
            for c in corrections:
                key = c["key"]
                if key not in self._pending_keys:
                    heapq.heappush(self._pending, CorrectionEntry(key, c["message"], c["severity"]))
                    self._pending_keys.add(key)

    def update_good(self):
        """Call when the pose is fully correct."""
        with self._lock:
            self._active_keys  = set()
            self._good_pending = True

    # ── Speaker thread ────────────────────────

    def _speaker_loop(self):
        while not self._stop_event.is_set():

            # ── Wait one full tick (interruptible) ────────────────────
            tick_start = time.time()
            while not self._stop_event.is_set():
                if time.time() - tick_start >= self.tick_seconds:
                    break
                time.sleep(0.1)

            if self._stop_event.is_set():
                break

            # ── Pick what to say ──────────────────────────────────────
            message = None
            now     = time.time()

            with self._lock:
                if self._good_pending:
                    self._good_pending = False
                    last_good = self._cooldowns.get("__good__", 0.0)
                    if now - last_good >= self.cooldown_seconds:
                        message = "Good form! Hold this position."
                        self._cooldowns["__good__"] = now

                else:
                    scratch: list[CorrectionEntry] = []

                    while self._pending:
                        entry = heapq.heappop(self._pending)
                        self._pending_keys.discard(entry.key)

                        # Drop if no longer active
                        if entry.key not in self._active_keys:
                            continue

                        # Still on cooldown — save for re-insertion
                        last_time = self._cooldowns.get(entry.key, 0.0)
                        if now - last_time < self.cooldown_seconds:
                            scratch.append(entry)
                            continue

                        # ✓ Speak this one
                        self._cooldowns[entry.key] = now
                        message = entry.message
                        break   # only one per tick

                    # Re-insert cooled-down active entries
                    for e in scratch:
                        if e.key in self._active_keys:
                            heapq.heappush(self._pending, e)
                            self._pending_keys.add(e.key)

            # ── Speak ─────────────────────────────────────────────────
            if message:
                self.last_message = message
                self._speak(message)

    def _speak(self, text: str):
        """
        Speak text using a brand-new pyttsx3 engine instance.

        A fresh engine is created, used once, and immediately stopped.
        This is intentional — reusing a single engine across calls causes
        pyttsx3 to deadlock silently after a few utterances on a background
        thread, which is exactly the "voice stops after a while" bug.
        """
        if not TTS_AVAILABLE:
            print(f"[VOICE] {text}")
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logging.warning("TTS error: %s", e)
            # If pyttsx3 fails entirely, fall back to print so the
            # developer can still see what would have been spoken.
            print(f"[VOICE] {text}")
