"""
Priority-queue based voice feedback manager.
Uses piper-tts for high-quality local voice synthesis.
"""

import heapq
import threading
import time
import logging
import subprocess
import tempfile
import os

# ─────────────────────────────────────────────
#  CorrectionEntry
# ─────────────────────────────────────────────

class CorrectionEntry:
    def __init__(self, key: str, message_en: str, severity: int, message_hi: str = None):
        self.key        = key
        self.message_en = message_en
        self.message_hi = message_hi
        self.severity   = severity

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
    it once using piper-tts, then waits for the next tick.

    Parameters
    ----------
    cooldown_seconds : float
        How long before the same correction key can be spoken again.
    tick_seconds : float
        Fixed interval between voice feedback attempts (default 5 s).
    """

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        tick_seconds:     float = 2.0,
        lang:             str = "en",
    ):
        self.cooldown_seconds = cooldown_seconds
        self.tick_seconds     = tick_seconds
        self.lang             = lang

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
        logging.info("FeedbackManager started (tick=%.1fs, lang=%s)", self.tick_seconds, self.lang)

    def stop(self):
        self._stop_event.set()
        if self._speak_thread:
            self._speak_thread.join(timeout=5)
        logging.info("FeedbackManager stopped")

    def set_lang(self, lang: str):
        """Switch language dynamically."""
        with self._lock:
            self.lang = lang
            logging.info("FeedbackManager language switched to %s", lang)

    def speak_immediate(self, text: str):
        """Speak text immediately in a separate thread, bypassing the queue."""
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

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
                    msg_en = c.get("message", "")
                    msg_hi = c.get("message_hi", "")
                    heapq.heappush(self._pending, CorrectionEntry(key, msg_en, c["severity"], msg_hi))
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
                        if self.lang == "hi":
                            message = "बहुत अच्छे! इसी स्थिति में रहें।"
                        else:
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
                        if self.lang == "hi" and entry.message_hi:
                            message = entry.message_hi
                        else:
                            message = entry.message_en
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
        Speak text using piper-tts via subprocess.
        Pipes text to piper, outputs to a temp wav file, and plays with aplay.
        """
        model = "models/piper/en_US-lessac-medium.onnx"
        if self.lang == "hi":
            model = "models/piper/hi_IN-pratham-medium.onnx"
            
        command = f"echo '{text}' | piper --model {model} --output_file /tmp/speak.wav && aplay /tmp/speak.wav"
        
        try:
            subprocess.run(command, shell=True, check=True, capture_output=True)
        except Exception as e:
            logging.warning("Piper TTS error: %s", e)
            # Fallback to print if synthesis fails
            print(f"[VOICE] {text}")
