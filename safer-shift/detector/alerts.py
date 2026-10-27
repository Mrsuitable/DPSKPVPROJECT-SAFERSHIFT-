"""SaferShift Alert Definitions.

Defines alert categories, severities, structured alert events,
and transparent explanation generators for human driver safety feedback.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import time


class AlertCategory(str, Enum):
    PROLONGED_EYE_CLOSURE = "prolonged_eye_closure"
    ELEVATED_BLINK_RATE = "elevated_blink_rate"
    POSSIBLE_YAWNING = "possible_yawning"
    SUSTAINED_DISTRACTION = "sustained_distraction"
    TRACKING_UNRELIABLE = "tracking_unreliable"
    NORMAL = "normal"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"      # Immediate collision risk (e.g. eyes closed >1.5s)
    WARNING = "warning"        # High risk (sustained distraction, yawning)
    ADVISORY = "advisory"      # Experimental indicator (high blink rate)
    INFO = "info"              # Tracking loss / neutral status


@dataclass
class AlertEvent:
    category: AlertCategory
    severity: AlertSeverity
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    mode: str = "standard"  # "standard" or "personalized"
    trigger_metric: str = ""
    trigger_value: float = 0.0
    threshold_value: float = 0.0
    explanation: str = ""
    telemetry_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.start_time,
            "category": self.category.value,
            "severity": self.severity.value,
            "duration": round(self.duration, 2),
            "mode": self.mode,
            "trigger_metric": self.trigger_metric,
            "trigger_value": round(self.trigger_value, 4),
            "threshold_value": round(self.threshold_value, 4),
            "explanation": self.explanation,
        }
