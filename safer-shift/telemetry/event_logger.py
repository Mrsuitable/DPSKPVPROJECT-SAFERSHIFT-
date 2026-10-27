"""Local Event and Telemetry Logger.

Writes telemetry samples and alert events to local CSV storage.
Ensures zero transmission of raw video footage.
Handles corrupt rows, empty files, and partial writes gracefully.
"""

import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from config import LOGS_DIR


CSV_FIELDNAMES = [
    "timestamp",
    "timestamp_iso",
    "event_type",
    "mode",
    "ear",
    "mar",
    "yaw",
    "pitch",
    "roll",
    "closure_duration",
    "distraction_duration",
    "yawn_duration",
    "blink_rate_bpm",
    "total_blinks",
    "calibration_status",
    "warning_reason",
    "tracking_quality",
    "measured_fps",
]


@dataclass
class TelemetryRecord:
    timestamp: float
    timestamp_iso: str
    event_type: str  # "sample", "prolonged_eye_closure", "sustained_distraction", etc.
    mode: str        # "standard" or "personalized"
    ear: float
    mar: float
    yaw: float
    pitch: float
    roll: float
    closure_duration: float
    distraction_duration: float
    yawn_duration: float
    blink_rate_bpm: float
    total_blinks: int
    calibration_status: str
    warning_reason: str
    tracking_quality: float
    measured_fps: float


class EventLogger:
    """Manages buffered writing of telemetry and immediate flushing of critical alert events."""

    def __init__(
        self,
        output_filepath: Optional[Path] = None,
        sample_interval_sec: float = 0.20,  # 5 Hz time-series sample
        buffer_size: int = 15,
    ):
        if output_filepath is None:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filepath = LOGS_DIR / f"telemetry_{now_str}.csv"
        else:
            self.filepath = Path(output_filepath)

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.sample_interval_sec = sample_interval_sec
        self.buffer_size = buffer_size

        self.last_sample_time: float = -1.0
        self.buffer: List[TelemetryRecord] = []
        self._ensure_header()

    def _ensure_header(self):
        if not self.filepath.exists() or self.filepath.stat().st_size == 0:
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()

    def log_state(
        self,
        state: Any,
        event_type: str = "sample",
        warning_reason: str = "",
        calibration_status: str = "calibrated",
        measured_fps: float = 0.0,
        force_flush: bool = False,
    ):
        """Logs a state record. Alerts trigger immediate flush; periodic samples are buffered."""
        now = state.timestamp
        is_event = event_type != "sample"

        if not is_event and not force_flush:
            if (now - self.last_sample_time) < self.sample_interval_sec:
                return

        self.last_sample_time = now
        iso_str = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        record = TelemetryRecord(
            timestamp=round(now, 3),
            timestamp_iso=iso_str,
            event_type=event_type,
            mode=getattr(state, "mode", "standard"),
            ear=round(getattr(state, "ear", 0.0), 4),
            mar=round(getattr(state, "mar", 0.0), 4),
            yaw=round(getattr(state, "yaw", 0.0), 2),
            pitch=round(getattr(state, "pitch", 0.0), 2),
            roll=round(getattr(state, "roll", 0.0), 2),
            closure_duration=round(getattr(state, "eye_closure_duration", 0.0), 2),
            distraction_duration=round(getattr(state, "distraction_duration", 0.0), 2),
            yawn_duration=round(getattr(state, "yawn_duration", 0.0), 2),
            blink_rate_bpm=round(getattr(state, "blink_rate_bpm", 0.0), 1),
            total_blinks=int(getattr(state, "total_blinks", 0)),
            calibration_status=calibration_status,
            warning_reason=warning_reason,
            tracking_quality=round(getattr(state, "tracking_quality", 1.0), 2),
            measured_fps=round(measured_fps, 1),
        )

        self.buffer.append(record)

        if is_event or force_flush or len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Flushes buffered records to the CSV file on disk."""
        if not self.buffer:
            return

        self._ensure_header()
        try:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                for rec in self.buffer:
                    writer.writerow(asdict(rec))
            self.buffer.clear()
        except Exception:
            # Prevent logging failures from halting real-time monitor
            pass

    def close(self):
        self.flush()


def load_telemetry_records(filepath: Path) -> List[TelemetryRecord]:
    """Robustly reads a telemetry CSV, safely skipping corrupted or truncated rows."""
    filepath = Path(filepath)
    if not filepath.exists() or filepath.stat().st_size == 0:
        return []

    records: List[TelemetryRecord] = []
    try:
        with open(filepath, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rec = TelemetryRecord(
                        timestamp=float(row.get("timestamp", 0.0)),
                        timestamp_iso=row.get("timestamp_iso", ""),
                        event_type=row.get("event_type", "sample"),
                        mode=row.get("mode", "standard"),
                        ear=float(row.get("ear", 0.0)),
                        mar=float(row.get("mar", 0.0)),
                        yaw=float(row.get("yaw", 0.0)),
                        pitch=float(row.get("pitch", 0.0)),
                        roll=float(row.get("roll", 0.0)),
                        closure_duration=float(row.get("closure_duration", 0.0)),
                        distraction_duration=float(row.get("distraction_duration", 0.0)),
                        yawn_duration=float(row.get("yawn_duration", 0.0)),
                        blink_rate_bpm=float(row.get("blink_rate_bpm", 0.0)),
                        total_blinks=int(float(row.get("total_blinks", 0))),
                        calibration_status=row.get("calibration_status", "unknown"),
                        warning_reason=row.get("warning_reason", ""),
                        tracking_quality=float(row.get("tracking_quality", 1.0)),
                        measured_fps=float(row.get("measured_fps", 0.0)),
                    )
                    records.append(rec)
                except (ValueError, TypeError):
                    continue
    except Exception:
        return []

    return records
