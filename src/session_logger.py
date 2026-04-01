"""
session_logger.py
-----------------
Lightweight session logger for yoga practice.

Tracks per-session:
  - Which pose was detected each frame, and for how long
  - Which corrections fired, how often, and when they were resolved
  - A simple improvement score per pose (correction count decreases over time)

Writes a JSON log to  logs/session_YYYYMMDD_HHMMSS.json  on stop().

The log is intentionally simple — no cloud, no PII, just local files.
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

LOGS_DIR = Path(__file__).parent.parent / "logs"


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────


@dataclass
class CorrectionEvent:
    key: str
    message: str
    severity: int
    timestamp: float  # seconds since session start
    resolved: bool = False
    resolved_at: float | None = None


@dataclass
class PoseInterval:
    pose: str
    started_at: float  # seconds since session start
    ended_at: float | None = None
    corrections: list = field(default_factory=list)  # list of CorrectionEvent

    @property
    def duration(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at


# ─────────────────────────────────────────────
#  SessionLogger
# ─────────────────────────────────────────────


class SessionLogger:
    """
    Usage
    -----
        logger = SessionLogger()
        logger.start()

        # each frame:
        logger.log_pose(label)
        logger.log_corrections(corrections)   # list[dict] from check_pose / BiLSTMCorrector

        # when a correction is resolved (not in latest correction list):
        logger.resolve_correction(key)

        # on session end:
        logger.stop()   # writes JSON to logs/
    """

    def __init__(self, logs_dir: Path = LOGS_DIR):
        self._logs_dir = Path(logs_dir)
        self._start_time: float | None = None
        self._session_id: str = ""

        self._pose_intervals: list[PoseInterval] = []
        self._current_interval: PoseInterval | None = None

        # key → CorrectionEvent (currently active corrections)
        self._active_corrections: dict[str, CorrectionEvent] = {}

        # all corrections ever fired this session (for summary)
        self._all_corrections: list[CorrectionEvent] = []

    # ── Lifecycle ─────────────────────────────

    def start(self):
        self._start_time = time.time()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Session {self._session_id} started")

    def stop(self):
        """Finalise and write the session log."""
        if self._start_time is None:
            return

        # Close any open pose interval
        if self._current_interval:
            self._current_interval.ended_at = self._elapsed()

        path = self._logs_dir / f"session_{self._session_id}.json"
        data = self._build_summary()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logging.info(f"Session log saved to {path}")
        print(f"\n[SessionLogger] Log saved → {path}")
        return path

    # ── Per-frame calls ────────────────────────

    def log_pose(self, pose_label: str):
        """
        Call each frame with the classified pose label.
        Automatically opens/closes pose intervals on transitions.
        """
        if self._start_time is None:
            return

        if self._current_interval is None or self._current_interval.pose != pose_label:
            # Close previous interval
            if self._current_interval:
                self._current_interval.ended_at = self._elapsed()
                self._pose_intervals.append(self._current_interval)

            # Open new interval
            self._current_interval = PoseInterval(
                pose=pose_label, started_at=self._elapsed()
            )

    def log_corrections(self, corrections: list):
        """
        Call each correction cycle with the current list of active corrections.
        New keys are registered; keys absent from this list are auto-resolved.

        Parameters
        ----------
        corrections : list[dict]
            Each dict: {"key", "message", "severity", ...}
        """
        if self._start_time is None:
            return

        now = self._elapsed()
        current_keys = {c["key"] for c in corrections}

        # Resolve corrections that are no longer active
        for key in list(self._active_corrections.keys()):
            if key not in current_keys:
                self.resolve_correction(key)

        # Register new corrections
        for c in corrections:
            key = c["key"]
            if key not in self._active_corrections:
                event = CorrectionEvent(
                    key=key,
                    message=c["message"],
                    severity=c["severity"],
                    timestamp=now,
                )
                self._active_corrections[key] = event
                self._all_corrections.append(event)

                # Also attach to current pose interval
                if self._current_interval:
                    self._current_interval.corrections.append(event)

    def resolve_correction(self, key: str):
        """Mark a correction as resolved (pose improved)."""
        event = self._active_corrections.pop(key, None)
        if event:
            event.resolved = True
            event.resolved_at = self._elapsed()

    # ── Summary / stats ───────────────────────

    def get_live_stats(self) -> dict:
        """
        Returns a lightweight stats dict suitable for on-screen display.
        Call at any time during a session.
        """
        if self._start_time is None:
            return {}

        elapsed = self._elapsed()
        len(self._pose_intervals) + (1 if self._current_interval else 0)

        # Time per pose
        time_per_pose: dict[str, float] = {}
        for interval in self._pose_intervals:
            time_per_pose[interval.pose] = (
                time_per_pose.get(interval.pose, 0) + interval.duration
            )
        if self._current_interval:
            p = self._current_interval.pose
            time_per_pose[p] = time_per_pose.get(p, 0) + self._current_interval.duration

        # Most frequent correction
        freq: dict[str, int] = {}
        for e in self._all_corrections:
            freq[e.key] = freq.get(e.key, 0) + 1
        top_correction = max(freq, key=freq.get) if freq else None

        return {
            "elapsed_seconds": round(elapsed, 1),
            "total_corrections": len(self._all_corrections),
            "active_corrections": len(self._active_corrections),
            "time_per_pose": {k: round(v, 1) for k, v in time_per_pose.items()},
            "top_correction": top_correction,
        }

    def _build_summary(self) -> dict:
        """Full session summary for JSON output."""
        elapsed = self._elapsed()
        freq: dict[str, int] = {}
        resolved_count = 0

        for e in self._all_corrections:
            freq[e.key] = freq.get(e.key, 0) + 1
            if e.resolved:
                resolved_count += 1

        # Per-pose correction breakdown
        pose_breakdown = {}
        for interval in self._pose_intervals:
            p = interval.pose
            if p not in pose_breakdown:
                pose_breakdown[p] = {"total_seconds": 0, "corrections_fired": 0}
            pose_breakdown[p]["total_seconds"] += round(interval.duration, 2)
            pose_breakdown[p]["corrections_fired"] += len(interval.corrections)
        if self._current_interval:
            p = self._current_interval.pose
            if p not in pose_breakdown:
                pose_breakdown[p] = {"total_seconds": 0, "corrections_fired": 0}
            pose_breakdown[p]["total_seconds"] += round(
                self._current_interval.duration, 2
            )
            pose_breakdown[p]["corrections_fired"] += len(
                self._current_interval.corrections
            )

        return {
            "session_id": self._session_id,
            "duration_seconds": round(elapsed, 2),
            "total_corrections": len(self._all_corrections),
            "resolved_corrections": resolved_count,
            "resolution_rate": round(
                resolved_count / max(len(self._all_corrections), 1), 2
            ),
            "correction_frequency": dict(sorted(freq.items(), key=lambda x: -x[1])),
            "pose_breakdown": pose_breakdown,
            "pose_intervals": [
                {
                    "pose": iv.pose,
                    "start": round(iv.started_at, 2),
                    "end": round(iv.ended_at, 2) if iv.ended_at else None,
                    "duration": round(iv.duration, 2),
                    "n_corrections": len(iv.corrections),
                }
                for iv in self._pose_intervals
            ],
        }

    def _elapsed(self) -> float:
        return time.time() - self._start_time
