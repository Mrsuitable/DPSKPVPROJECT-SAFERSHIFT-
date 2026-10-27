"""Runtime System Metrics Tracking.

Measures actual frame rate (FPS), latency distributions, and tracking continuity.
Enforces the project principle: Never fabricate or hardcode FPS numbers.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class SystemMetrics:
    fps: float
    avg_latency_ms: float
    total_frames: int
    dropped_frames: int


class FPSMeter:
    """Calculates running average FPS using a deque of frame timestamps."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        self.latencies: Deque[float] = deque(maxlen=window_size)
        self.total_frames: int = 0
        self.dropped_frames: int = 0

    def tick(self, latency_sec: float = 0.0) -> float:
        now = time.monotonic()
        self.timestamps.append(now)
        self.latencies.append(latency_sec)
        self.total_frames += 1

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0

        fps = (len(self.timestamps) - 1) / elapsed
        return round(fps, 1)

    def get_metrics(self) -> SystemMetrics:
        if len(self.timestamps) < 2:
            fps = 0.0
        else:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            fps = round((len(self.timestamps) - 1) / max(1e-4, elapsed), 1)

        avg_lat_ms = (
            round((sum(self.latencies) / len(self.latencies)) * 1000.0, 1)
            if self.latencies
            else 0.0
        )

        return SystemMetrics(
            fps=fps,
            avg_latency_ms=avg_lat_ms,
            total_frames=self.total_frames,
            dropped_frames=self.dropped_frames,
        )
