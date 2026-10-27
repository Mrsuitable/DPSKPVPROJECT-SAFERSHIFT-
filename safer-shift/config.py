"""SaferShift Configuration and Constants.

Personalized, Privacy-Preserving Driver Safety Intelligence.
Defines baseline standard thresholds, temporal parameters, personalization safety bounds,
and filesystem paths.
"""

from dataclasses import dataclass
from pathlib import Path


# Root directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROFILES_DIR = DATA_DIR / "profiles"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StandardThresholds:
    """Standard baseline thresholds (unpersonalized)."""
    # Eye Aspect Ratio (EAR) below which eyes are considered closed
    EAR_THRESHOLD: float = 0.21
    # Mouth Aspect Ratio (MAR) above which mouth is considered yawning
    MAR_THRESHOLD: float = 0.55
    # Head pose deviation limits from neutral (degrees)
    YAW_THRESHOLD: float = 25.0
    PITCH_DOWN_THRESHOLD: float = 18.0
    PITCH_UP_THRESHOLD: float = 22.0


@dataclass(frozen=True)
class TemporalSettings:
    """Sliding-window temporal parameters and continuity settings."""
    # Mandatory requirement: sustained eye closure > 1.5 seconds triggers critical alarm
    PROLONGED_CLOSURE_SEC: float = 1.5
    # Sustained head-pose deviation duration to trigger distraction alert
    DISTRACTION_DURATION_SEC: float = 2.0
    # Sustained mouth opening duration to trigger yawning alert
    YAWN_DURATION_SEC: float = 2.2
    # Sliding window duration for blink rate calculation (seconds)
    BLINK_WINDOW_SEC: float = 60.0
    # Alert cooldown duration to prevent repetitive warning spam (seconds)
    ALERT_COOLDOWN_SEC: float = 3.0
    # Maximum brief tracking dropout tolerated before resetting continuous closure (seconds)
    MAX_TRACKING_LOSS_TOLERANCE_SEC: float = 0.4


@dataclass(frozen=True)
class PersonalizationBounds:
    """Safety bounds for personalized adaptive thresholds."""
    # Hard bounds preventing unsafe extreme threshold values
    MIN_EAR_THRESHOLD: float = 0.15
    MAX_EAR_THRESHOLD: float = 0.28
    # Multiplier applied to driver's calibrated open-eye baseline EAR
    ADAPTIVE_EAR_RATIO: float = 0.75
    # Yaw and pitch deviation bounds
    MIN_YAW_LIMIT: float = 18.0
    MAX_YAW_LIMIT: float = 35.0
    MIN_PITCH_LIMIT: float = 14.0
    MAX_PITCH_LIMIT: float = 30.0
    # Minimum frames and quality score for valid calibration
    MIN_CALIBRATION_FRAMES: int = 45
    MIN_CALIBRATION_QUALITY: float = 0.70


STANDARD_CONFIG = StandardThresholds()
TEMPORAL_CONFIG = TemporalSettings()
BOUNDS_CONFIG = PersonalizationBounds()
