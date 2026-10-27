"""SaferShift Detector Package.
Provides facial landmark analysis, eye and mouth metrics, head pose estimation,
and temporal alert detection logic.
"""

from .alerts import AlertCategory, AlertEvent, AlertSeverity
from .eye_metrics import EyeMetrics, calculate_ear, detect_blinks
from .mouth_metrics import MouthMetrics, calculate_mar
from .head_pose import HeadPose, estimate_head_pose
from .temporal_engine import TemporalEngine, DetectionState

__all__ = [
    "AlertCategory",
    "AlertEvent",
    "AlertSeverity",
    "EyeMetrics",
    "calculate_ear",
    "detect_blinks",
    "MouthMetrics",
    "calculate_mar",
    "HeadPose",
    "estimate_head_pose",
    "TemporalEngine",
    "DetectionState",
]
