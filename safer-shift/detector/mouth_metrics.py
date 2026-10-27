"""Mouth metrics module: Mouth Aspect Ratio (MAR) and yawn analysis.

Computes MAR from facial landmarks to capture mouth vertical opening normalized
against horizontal lip width. Handles invalid or missing coordinates gracefully.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any


# MediaPipe FaceMesh key indices for mouth
UPPER_LIP_MID = 13
LOWER_LIP_MID = 14
UPPER_LIP_LEFT = 82
LOWER_LIP_LEFT = 87
UPPER_LIP_RIGHT = 312
LOWER_LIP_RIGHT = 317
MOUTH_CORNER_LEFT = 78
MOUTH_CORNER_RIGHT = 308


@dataclass
class MouthMetrics:
    mar: float
    is_open: bool
    is_valid: bool


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_mar(
    landmarks: Any,
    mar_threshold: float = 0.55,
    image_width: int = 1,
    image_height: int = 1,
) -> MouthMetrics:
    """Calculates Mouth Aspect Ratio (MAR) from landmarks.

    MAR = (vertical_mid + 0.5 * (vertical_left + vertical_right)) / (2.0 * horizontal_width)
    """
    if landmarks is None:
        return MouthMetrics(0.0, is_open=False, is_valid=False)

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

            # If already pixel coordinates (> 1.0), use directly without stretching aspect ratio
            if abs(x) > 1.0 or abs(y) > 1.0:
                return (x, y)
            return (x * image_width, y * image_height)
            return None

        p_up_mid = get_coords(UPPER_LIP_MID)
        p_low_mid = get_coords(LOWER_LIP_MID)
        p_up_left = get_coords(UPPER_LIP_LEFT)
        p_low_left = get_coords(LOWER_LIP_LEFT)
        p_up_right = get_coords(UPPER_LIP_RIGHT)
        p_low_right = get_coords(LOWER_LIP_RIGHT)
        p_corner_left = get_coords(MOUTH_CORNER_LEFT)
        p_corner_right = get_coords(MOUTH_CORNER_RIGHT)

        pts = [p_up_mid, p_low_mid, p_up_left, p_low_left, p_up_right, p_low_right, p_corner_left, p_corner_right]
        if any(pt is None for pt in pts):
            return MouthMetrics(0.0, is_open=False, is_valid=False)

        # Horizontal distance
        h_dist = _dist(p_corner_left, p_corner_right)  # type: ignore
        if h_dist < 1e-5:
            return MouthMetrics(0.0, is_open=False, is_valid=False)

        # Vertical distances
        v_mid = _dist(p_up_mid, p_low_mid)  # type: ignore
        v_left = _dist(p_up_left, p_low_left)  # type: ignore
        v_right = _dist(p_up_right, p_low_right)  # type: ignore

        mar = (v_mid + 0.5 * (v_left + v_right)) / (2.0 * h_dist)
        mar = max(0.0, float(mar))

        return MouthMetrics(
            mar=round(mar, 4),
            is_open=mar >= mar_threshold,
            is_valid=True,
        )

    except Exception:
        return MouthMetrics(0.0, is_open=False, is_valid=False)
