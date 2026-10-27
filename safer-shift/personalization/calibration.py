"""Driver Calibration Workflow.

Captures baseline observations while driver is alert and stationary looking straight ahead.
Computes robust baseline open-eye geometry, neutral head orientation, and derived thresholds.
Assesses calibration quality and enforces safety bounds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import List, Optional, Tuple, Dict, Any

from config import BOUNDS_CONFIG, STANDARD_CONFIG
from .profiles import DriverProfile


@dataclass
class CalibrationSample:
    ear: float
    mar: float
    yaw: float
    pitch: float
    is_valid: bool


@dataclass
class CalibrationResult:
    success: bool
    quality_score: float  # 0.0 to 1.0
    message: str
    profile: Optional[DriverProfile] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _percentile(values: List[float], p: float) -> float:
    """Computes p-th percentile (0-100) using linear interpolation without external dependencies."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return float(d0 + d1)


class CalibrationSession:
    """Collects and analyzes frame measurements to construct an adaptive driver profile."""

    def __init__(self, profile_id: str, driver_name: Optional[str] = None):
        self.profile_id = profile_id.strip() or "driver_default"
        self.driver_name = driver_name.strip() if driver_name else self.profile_id
        self.samples: List[CalibrationSample] = []

    def add_sample(self, ear: float, mar: float, yaw: float, pitch: float, is_valid: bool = True):
        try:
            values = tuple(float(value) for value in (ear, mar, yaw, pitch))
            sample_is_valid = bool(is_valid and all(math.isfinite(value) for value in values))
        except (TypeError, ValueError, OverflowError):
            values = (0.0, 0.0, 0.0, 0.0)
            sample_is_valid = False
        self.samples.append(
            CalibrationSample(ear=values[0], mar=values[1], yaw=values[2], pitch=values[3], is_valid=sample_is_valid)
        )

    @property
    def valid_sample_count(self) -> int:
        return sum(1 for s in self.samples if s.is_valid)

    def finalize(self) -> CalibrationResult:
        """Processes collected calibration observations and builds a validated driver profile."""
        valid_samples = [s for s in self.samples if s.is_valid]
        n = len(valid_samples)

        # 1. Check Sample Count Requirement
        if n < BOUNDS_CONFIG.MIN_CALIBRATION_FRAMES:
            return CalibrationResult(
                success=False,
                quality_score=round(n / BOUNDS_CONFIG.MIN_CALIBRATION_FRAMES, 2),
                message=f"Insufficient calibration samples ({n}/{BOUNDS_CONFIG.MIN_CALIBRATION_FRAMES} required). Please remain facing camera for at least 3 seconds.",
                profile=None,
                diagnostics={"samples_collected": n},
            )

        ears = [s.ear for s in valid_samples]
        mars = [s.mar for s in valid_samples]
        yaws = [s.yaw for s in valid_samples]
        pitches = [s.pitch for s in valid_samples]

        ear_mean = sum(ears) / n
        ear_var = sum((x - ear_mean) ** 2 for x in ears) / max(1, n - 1)
        ear_std = math.sqrt(ear_var)

        mar_mean = sum(mars) / n
        yaw_median = _percentile(yaws, 50.0)
        pitch_median = _percentile(pitches, 50.0)

        ear_p25 = _percentile(ears, 25.0)
        ear_p50 = _percentile(ears, 50.0)

        # 2. Quality Evaluation
        # If driver had their eyes closed during baseline, reject calibration!
        if ear_p50 < 0.17:
            return CalibrationResult(
                success=False,
                quality_score=0.2,
                message="Calibration rejected: Eyes appeared predominantly closed or obstructed during calibration. Please keep eyes alert and open.",
                profile=None,
                diagnostics={"median_ear": round(ear_p50, 3)},
            )

        # Check jitter / variance (excessive head motion or unstable landmarks)
        yaw_var = sum((y - yaw_median) ** 2 for y in yaws) / max(1, n - 1)
        yaw_std = math.sqrt(yaw_var)

        quality = 1.0
        if ear_std > 0.08:
            quality -= 0.25
        if yaw_std > 12.0:
            quality -= 0.25
        if n < 60:
            quality -= 0.10

        quality = max(0.1, min(1.0, quality))

        if quality < BOUNDS_CONFIG.MIN_CALIBRATION_QUALITY:
            return CalibrationResult(
                success=False,
                quality_score=round(quality, 2),
                message=f"Calibration quality ({quality:.2f}) below threshold ({BOUNDS_CONFIG.MIN_CALIBRATION_QUALITY:.2f}) due to high head movement or tracking jitter. Please try again while stationary.",
                profile=None,
                diagnostics={"quality": quality, "ear_std": ear_std, "yaw_std": yaw_std},
            )

        # 3. Compute Adaptive Thresholds
        # Scale driver's 25th percentile open-eye EAR by safety factor
        raw_adapted_ear = ear_p25 * BOUNDS_CONFIG.ADAPTIVE_EAR_RATIO

        # Clamp within strict safety bounds
        adapted_ear = max(
            BOUNDS_CONFIG.MIN_EAR_THRESHOLD,
            min(BOUNDS_CONFIG.MAX_EAR_THRESHOLD, raw_adapted_ear),
        )

        # Yaw and Pitch bounds centered around driver's calibrated neutral position
        yaw_limit = STANDARD_CONFIG.YAW_THRESHOLD
        pitch_down_limit = STANDARD_CONFIG.PITCH_DOWN_THRESHOLD
        pitch_up_limit = STANDARD_CONFIG.PITCH_UP_THRESHOLD

        profile = DriverProfile(
            profile_id=self.profile_id,
            driver_name=self.driver_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            calibration_quality=round(quality, 3),
            baseline_ear_mean=round(ear_mean, 4),
            baseline_ear_std=round(ear_std, 4),
            open_eye_ear_p25=round(ear_p25, 4),
            open_eye_ear_p50=round(ear_p50, 4),
            adapted_ear_threshold=round(adapted_ear, 4),
            neutral_yaw=round(yaw_median, 2),
            neutral_pitch=round(pitch_median, 2),
            yaw_limit=round(yaw_limit, 2),
            pitch_down_limit=round(pitch_down_limit, 2),
            pitch_up_limit=round(pitch_up_limit, 2),
            baseline_mar_mean=round(mar_mean, 4),
            sample_count=n,
            notes=f"Calibrated successfully with {n} frames. Adapted EAR threshold: {adapted_ear:.3f}",
        )

        return CalibrationResult(
            success=True,
            quality_score=round(quality, 2),
            message=f"Calibration successful! Quality: {int(quality * 100)}%. Adapted EAR threshold: {adapted_ear:.3f}",
            profile=profile,
            diagnostics={
                "samples": n,
                "baseline_ear_mean": round(ear_mean, 4),
                "adapted_ear_threshold": round(adapted_ear, 4),
                "standard_ear_threshold": STANDARD_CONFIG.EAR_THRESHOLD,
                "neutral_yaw": round(yaw_median, 2),
                "neutral_pitch": round(pitch_median, 2),
            },
        )
