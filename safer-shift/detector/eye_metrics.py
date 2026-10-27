"""Eye metrics module: Eye Aspect Ratio (EAR) and blink tracking.

Computes EAR for left and right eyes using standard 6-point Euclidean geometry,
detects discrete blink transitions, and computes sliding-window blink frequency.
Handles missing, empty, or degenerate landmark coordinates gracefully.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any


# Standard MediaPipe facial mesh landmark indices for eyes
# Left eye: p1=33 (lateral), p2=160, p3=158, p4=133 (medial), p5=153, p6=144
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Right eye: p1=362 (lateral), p2=385, p3=387, p4=263 (medial), p5=373, p6=380
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


@dataclass
class EyeMetrics:
    left_ear: float
    right_ear: float
    ear: float
    is_closed: bool
    is_valid: bool


def _euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Computes Euclidean 2D distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_single_eye_ear(eye_points: List[Tuple[float, float]]) -> Optional[float]:
    """Calculates the Eye Aspect Ratio (EAR) for a single eye given 6 (x, y) coordinates.

    p1, p2, p3, p4, p5, p6
    EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
    """
    if not eye_points or len(eye_points) < 6:
        return None

    # Check for invalid coordinates
    for pt in eye_points[:6]:
        if pt is None or len(pt) < 2:
            return None
        if not (isinstance(pt[0], (int, float)) and isinstance(pt[1], (int, float))):
            return None
        if math.isnan(pt[0]) or math.isnan(pt[1]):
            return None

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    horizontal_dist = _euclidean_dist(p1, p4)
    if horizontal_dist < 1e-6:
        # Avoid division by zero or degenerate eye geometry
        return 0.0

    vertical_dist_1 = _euclidean_dist(p2, p6)
    vertical_dist_2 = _euclidean_dist(p3, p5)

    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    return max(0.0, float(ear))


def calculate_ear(
    landmarks: Any,
    ear_threshold: float = 0.21,
    image_width: int = 1,
    image_height: int = 1,
) -> EyeMetrics:
    """Calculates EAR from a landmark collection (dict, list, or MediaPipe landmark object).

    Gracefully returns invalid metrics if landmarks are missing, empty, or incomplete.
    """
    if landmarks is None:
        return EyeMetrics(0.0, 0.0, 0.0, is_closed=False, is_valid=False)

    try:
        def get_coords(idx: int) -> Optional[Tuple[float, float]]:
            x, y = None, None
            if isinstance(landmarks, (list, tuple)):
                if idx >= len(landmarks):
                    return None
                item = landmarks[idx]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    x, y = float(item[0]), float(item[1])
                elif hasattr(item, "x") and hasattr(item, "y"):
                    x, y = float(item.x), float(item.y)
            elif isinstance(landmarks, dict) and idx in landmarks:
                item = landmarks[idx]
                x, y = float(item[0]), float(item[1])

            if x is None or y is None:
                return None

            if abs(x) > 1.0 or abs(y) > 1.0:
                return (x, y)
            return (x * image_width, y * image_height)

        left_pts = [get_coords(idx) for idx in LEFT_EYE_INDICES]
        right_pts = [get_coords(idx) for idx in RIGHT_EYE_INDICES]

        if any(pt is None for pt in left_pts) or any(pt is None for pt in right_pts):
            return EyeMetrics(0.0, 0.0, 0.0, is_closed=False, is_valid=False)

        left_ear = calculate_single_eye_ear(left_pts)  # type: ignore
        right_ear = calculate_single_eye_ear(right_pts)  # type: ignore

        if left_ear is None or right_ear is None:
            return EyeMetrics(0.0, 0.0, 0.0, is_closed=False, is_valid=False)

        avg_ear = (left_ear + right_ear) / 2.0
        is_closed = avg_ear < ear_threshold

        return EyeMetrics(
            left_ear=round(left_ear, 4),
            right_ear=round(right_ear, 4),
            ear=round(avg_ear, 4),
            is_closed=is_closed,
            is_valid=True,
        )

    except Exception:
        return EyeMetrics(0.0, 0.0, 0.0, is_closed=False, is_valid=False)


class BlinkTracker:
    """Tracks discrete eye blink events and sliding window blink rate (blinks per minute).

    Prevents counting a single sustained closure as multiple blinks.
    A blink is registered upon rising edge (closing then reopening).
    """

    def __init__(self, window_sec: float = 60.0):
        self.window_sec = window_sec
        self.blink_timestamps: List[float] = []
        self.was_closed: bool = False
        self.closed_start_time: Optional[float] = None
        self.total_blinks: int = 0

    def update(self, is_closed: bool, timestamp: float) -> bool:
        """Updates blink state. Returns True if a discrete blink completed on this frame."""
        blink_occurred = False

        if is_closed and not self.was_closed:
            # Transition from open to closed
            self.was_closed = True
            self.closed_start_time = timestamp
        elif not is_closed and self.was_closed:
            # Transition from closed to open
            closure_dur = timestamp - (self.closed_start_time or timestamp)
            # A natural blink lasts between 0.05s and 0.45s.
            if 0.04 <= closure_dur <= 0.60:
                self.blink_timestamps.append(timestamp)
                self.total_blinks += 1
                blink_occurred = True
            self.was_closed = False
            self.closed_start_time = None

        # Clean old timestamps outside sliding window
        cutoff = timestamp - self.window_sec
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff]

        return blink_occurred

    def get_blinks_per_minute(self, current_time: float) -> float:
        """Calculates blinks per minute normalized over the active window."""
        cutoff = current_time - self.window_sec
        recent = [t for t in self.blink_timestamps if t >= cutoff]
        if not recent:
            return 0.0

        # Normalization factor if tracking window is shorter than 60s
        window_duration = max(1.0, min(self.window_sec, current_time - recent[0] + 1.0))
        return round((len(recent) / window_duration) * 60.0, 1)

    def reset(self):
        self.blink_timestamps.clear()
        self.was_closed = False
        self.closed_start_time = None
        self.total_blinks = 0


def detect_blinks(ear_series: List[float], threshold: float = 0.21) -> int:
    """Helper for offline event analysis: counts discrete blink dips below threshold."""
    blinks = 0
    in_blink = False
    for val in ear_series:
        if val < threshold and not in_blink:
            in_blink = True
        elif val >= threshold and in_blink:
            blinks += 1
            in_blink = False
    return blinks
