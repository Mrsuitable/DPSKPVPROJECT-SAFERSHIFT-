"""Temporal Fatigue and Distraction Engine.

Implements monotonic sliding-window temporal state tracking.
Detects:
1. Prolonged eye closure (>1.5 consecutive seconds)
2. Sustained distraction / head-pose deviation (>2.0s)
3. Sustained yawning events (MAR > threshold for >2.2s)
4. Elevated blink frequency (experimental advisory indicator)
5. Tracking quality drops and discontinuity handling
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from .alerts import AlertCategory, AlertEvent, AlertSeverity
from .eye_metrics import EyeMetrics, BlinkTracker, calculate_ear
from .mouth_metrics import MouthMetrics, calculate_mar
from .head_pose import HeadPose, estimate_head_pose


@dataclass
class DetectionState:
    timestamp: float
    mode: str
    ear: float
    mar: float
    yaw: float
    pitch: float
    roll: float
    is_face_detected: bool
    tracking_quality: float  # 0.0 to 1.0
    eye_closure_duration: float
    distraction_duration: float
    yawn_duration: float
    blink_rate_bpm: float
    total_blinks: int
    active_alerts: List[AlertEvent] = field(default_factory=list)


class TemporalEngine:
    """Monotonic sliding-window temporal engine for driver fatigue and distraction monitoring."""

    def __init__(
        self,
        mode: str = "standard",
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.55,
        neutral_yaw: float = 0.0,
        neutral_pitch: float = 0.0,
        yaw_threshold: float = 25.0,
        pitch_down_threshold: float = 18.0,
        pitch_up_threshold: float = 22.0,
        prolonged_closure_sec: float = 1.5,
        distraction_sec: float = 2.0,
        yawn_sec: float = 2.2,
        blink_window_sec: float = 60.0,
        alert_cooldown_sec: float = 3.0,
        max_tracking_loss_tolerance_sec: float = 0.4,
        geometric_head_pose: bool = False,
    ):
        self.mode = mode
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.neutral_yaw = neutral_yaw
        self.neutral_pitch = neutral_pitch
        self.yaw_threshold = yaw_threshold
        self.pitch_down_threshold = pitch_down_threshold
        self.pitch_up_threshold = pitch_up_threshold
        self.geometric_head_pose = geometric_head_pose

        self.prolonged_closure_sec = prolonged_closure_sec
        self.distraction_sec = distraction_sec
        self.yawn_sec = yawn_sec
        self.alert_cooldown_sec = alert_cooldown_sec
        self.max_tracking_loss_tolerance_sec = max_tracking_loss_tolerance_sec

        # State trackers
        self.blink_tracker = BlinkTracker(window_sec=blink_window_sec)

        # Eye closure state
        self.eye_closed_start: Optional[float] = None
        self.eye_closed_last_seen: Optional[float] = None
        self.eye_closed_alert_fired: bool = False

        # Distraction state
        self.distraction_start: Optional[float] = None
        self.distraction_alert_fired: bool = False

        # Yawn state
        self.yawn_start: Optional[float] = None
        self.yawn_alert_fired: bool = False
        self.total_yawns: int = 0

        # Tracking state
        self.last_valid_tracking_time: Optional[float] = None
        self.tracking_unreliable_alert_fired: bool = False

        # Cooldown timestamps by alert category
        self.last_alert_time: Dict[AlertCategory, float] = {}

    def configure_thresholds(
        self,
        ear_threshold: Optional[float] = None,
        neutral_yaw: Optional[float] = None,
        neutral_pitch: Optional[float] = None,
        yaw_threshold: Optional[float] = None,
        pitch_down_threshold: Optional[float] = None,
        pitch_up_threshold: Optional[float] = None,
        mode: Optional[str] = None,
    ):
        """Allows dynamic switching between standard and personalized configurations."""
        if ear_threshold is not None:
            self.ear_threshold = ear_threshold
        if neutral_yaw is not None:
            self.neutral_yaw = neutral_yaw
        if neutral_pitch is not None:
            self.neutral_pitch = neutral_pitch
        if yaw_threshold is not None:
            self.yaw_threshold = yaw_threshold
        if pitch_down_threshold is not None:
            self.pitch_down_threshold = pitch_down_threshold
        if pitch_up_threshold is not None:
            self.pitch_up_threshold = pitch_up_threshold
        if mode is not None:
            self.mode = mode

    def reset_state(self):
        """Resets all active timers and buffers."""
        self.blink_tracker.reset()
        self.eye_closed_start = None
        self.eye_closed_last_seen = None
        self.eye_closed_alert_fired = False
        self.distraction_start = None
        self.distraction_alert_fired = False
        self.yawn_start = None
        self.yawn_alert_fired = False
        self.total_yawns = 0
        self.last_valid_tracking_time = None
        self.tracking_unreliable_alert_fired = False
        self.last_alert_time.clear()

    def _can_fire_alert(self, category: AlertCategory, current_time: float) -> bool:
        last_t = self.last_alert_time.get(category, -1e9)
        return (current_time - last_t) >= self.alert_cooldown_sec

    def process_frame(
        self,
        landmarks: Any,
        timestamp: Optional[float] = None,
        image_width: int = 640,
        image_height: int = 480,
    ) -> Tuple[DetectionState, List[AlertEvent]]:
        """Processes a single video/webcam frame with monotonic temporal analysis.

        Returns the comprehensive DetectionState and any new AlertEvent generated.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        generated_alerts: List[AlertEvent] = []

        # 1. Compute Base Metrics
        eye_metrics = calculate_ear(
            landmarks,
            ear_threshold=self.ear_threshold,
            image_width=image_width,
            image_height=image_height,
        )
        mouth_metrics = calculate_mar(
            landmarks,
            mar_threshold=self.mar_threshold,
            image_width=image_width,
            image_height=image_height,
        )
        head_pose = estimate_head_pose(
            landmarks,
            image_width=image_width,
            image_height=image_height,
            neutral_yaw=self.neutral_yaw,
            neutral_pitch=self.neutral_pitch,
            yaw_threshold=self.yaw_threshold,
            pitch_down_threshold=self.pitch_down_threshold,
            pitch_up_threshold=self.pitch_up_threshold,
            force_geometric=self.geometric_head_pose,
        )

        is_face_detected = eye_metrics.is_valid and head_pose.is_valid
        tracking_quality = 1.0 if (is_face_detected and mouth_metrics.is_valid) else (0.5 if is_face_detected else 0.0)

        # 2. Tracking Continuity & Failure Handling
        if is_face_detected:
            self.last_valid_tracking_time = timestamp
            self.tracking_unreliable_alert_fired = False
        else:
            time_since_valid = (
                timestamp - self.last_valid_tracking_time
                if self.last_valid_tracking_time is not None
                else 999.0
            )

            # If tracking was briefly interrupted (e.g. quick motion blur <= 0.4s),
            # we do NOT immediately cancel ongoing timers, but we do NOT increment closure either.
            if time_since_valid > self.max_tracking_loss_tolerance_sec:
                # Definite tracking loss: Reset eye closure to avoid falsely declaring fatigue!
                self.eye_closed_start = None
                self.eye_closed_last_seen = None
                self.eye_closed_alert_fired = False
                self.distraction_start = None
                self.yawn_start = None

                if time_since_valid >= 1.0 and self._can_fire_alert(AlertCategory.TRACKING_UNRELIABLE, timestamp):
                    alert = AlertEvent(
                        category=AlertCategory.TRACKING_UNRELIABLE,
                        severity=AlertSeverity.INFO,
                        start_time=timestamp,
                        duration=time_since_valid,
                        mode=self.mode,
                        trigger_metric="face_detected",
                        trigger_value=0.0,
                        threshold_value=1.0,
                        explanation=f"Face tracking lost for {time_since_valid:.1f}s. Cannot verify driver state.",
                    )
                    generated_alerts.append(alert)
                    self.last_alert_time[AlertCategory.TRACKING_UNRELIABLE] = timestamp

            state = DetectionState(
                timestamp=timestamp,
                mode=self.mode,
                ear=0.0,
                mar=0.0,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                is_face_detected=False,
                tracking_quality=0.0,
                eye_closure_duration=0.0,
                distraction_duration=0.0,
                yawn_duration=0.0,
                blink_rate_bpm=self.blink_tracker.get_blinks_per_minute(timestamp),
                total_blinks=self.blink_tracker.total_blinks,
                active_alerts=generated_alerts,
            )
            return state, generated_alerts

        # 3. Eye Closure & Prolonged Eye Closure Alert (>1.5 seconds)
        closure_duration = 0.0
        if eye_metrics.is_closed:
            if self.eye_closed_start is None:
                self.eye_closed_start = timestamp
            self.eye_closed_last_seen = timestamp
            closure_duration = timestamp - self.eye_closed_start

            if closure_duration >= self.prolonged_closure_sec and not self.eye_closed_alert_fired:
                if self._can_fire_alert(AlertCategory.PROLONGED_EYE_CLOSURE, timestamp):
                    alert = AlertEvent(
                        category=AlertCategory.PROLONGED_EYE_CLOSURE,
                        severity=AlertSeverity.CRITICAL,
                        start_time=self.eye_closed_start,
                        duration=closure_duration,
                        mode=self.mode,
                        trigger_metric="ear_duration",
                        trigger_value=closure_duration,
                        threshold_value=self.prolonged_closure_sec,
                        explanation=(
                            f"Eyes closed continuously for {closure_duration:.2f}s "
                            f"(EAR: {eye_metrics.ear:.3f} < threshold {self.ear_threshold:.3f}). "
                            f"Exceeds safety limit of {self.prolonged_closure_sec:.1f}s."
                        ),
                    )
                    generated_alerts.append(alert)
                    self.last_alert_time[AlertCategory.PROLONGED_EYE_CLOSURE] = timestamp
                    self.eye_closed_alert_fired = True
        else:
            # Eyes are open: update blink tracking
            self.blink_tracker.update(is_closed=False, timestamp=timestamp)
            self.eye_closed_start = None
            self.eye_closed_last_seen = None
            self.eye_closed_alert_fired = False

        # If eyes are transitioning to closed, update blink tracker
        if eye_metrics.is_closed:
            self.blink_tracker.update(is_closed=True, timestamp=timestamp)

        # 4. Blink Rate Assessment
        blink_rate_bpm = self.blink_tracker.get_blinks_per_minute(timestamp)
        if blink_rate_bpm >= 38.0 and self._can_fire_alert(AlertCategory.ELEVATED_BLINK_RATE, timestamp):
            alert = AlertEvent(
                category=AlertCategory.ELEVATED_BLINK_RATE,
                severity=AlertSeverity.ADVISORY,
                start_time=timestamp,
                duration=60.0,
                mode=self.mode,
                trigger_metric="blink_rate_bpm",
                trigger_value=blink_rate_bpm,
                threshold_value=38.0,
                explanation=f"High blink frequency detected ({blink_rate_bpm:.1f} bpm). Indicative of potential visual fatigue.",
            )
            generated_alerts.append(alert)
            self.last_alert_time[AlertCategory.ELEVATED_BLINK_RATE] = timestamp

        # 5. Distraction / Sustained Head-Pose Deviation
        distraction_duration = 0.0
        if head_pose.is_distracted:
            if self.distraction_start is None:
                self.distraction_start = timestamp
            distraction_duration = timestamp - self.distraction_start

            if distraction_duration >= self.distraction_sec and not self.distraction_alert_fired:
                if self._can_fire_alert(AlertCategory.SUSTAINED_DISTRACTION, timestamp):
                    direction = (
                        "left" if (head_pose.yaw - self.neutral_yaw) < -self.yaw_threshold
                        else "right" if (head_pose.yaw - self.neutral_yaw) > self.yaw_threshold
                        else "down" if (head_pose.pitch - self.neutral_pitch) < -self.pitch_down_threshold
                        else "up"
                    )
                    alert = AlertEvent(
                        category=AlertCategory.SUSTAINED_DISTRACTION,
                        severity=AlertSeverity.WARNING,
                        start_time=self.distraction_start,
                        duration=distraction_duration,
                        mode=self.mode,
                        trigger_metric="head_pose_deviation",
                        trigger_value=distraction_duration,
                        threshold_value=self.distraction_sec,
                        explanation=(
                            f"Head turned {direction} away from road for {distraction_duration:.1f}s "
                            f"(Yaw: {head_pose.yaw:.1f}°, Pitch: {head_pose.pitch:.1f}°, "
                            f"Calibrated neutral: {self.neutral_yaw:.1f}°/{self.neutral_pitch:.1f}°)."
                        ),
                    )
                    generated_alerts.append(alert)
                    self.last_alert_time[AlertCategory.SUSTAINED_DISTRACTION] = timestamp
                    self.distraction_alert_fired = True
        else:
            self.distraction_start = None
            self.distraction_alert_fired = False

        # 6. Possible Yawning / Sustained Mouth Opening
        yawn_duration = 0.0
        if mouth_metrics.is_valid and mouth_metrics.is_open:
            if self.yawn_start is None:
                self.yawn_start = timestamp
            yawn_duration = timestamp - self.yawn_start

            if yawn_duration >= self.yawn_sec and not self.yawn_alert_fired:
                if self._can_fire_alert(AlertCategory.POSSIBLE_YAWNING, timestamp):
                    self.total_yawns += 1
                    alert = AlertEvent(
                        category=AlertCategory.POSSIBLE_YAWNING,
                        severity=AlertSeverity.WARNING,
                        start_time=self.yawn_start,
                        duration=yawn_duration,
                        mode=self.mode,
                        trigger_metric="mar_duration",
                        trigger_value=mouth_metrics.mar,
                        threshold_value=self.mar_threshold,
                        explanation=(
                            f"Sustained wide mouth opening for {yawn_duration:.1f}s "
                            f"(MAR: {mouth_metrics.mar:.3f} >= threshold {self.mar_threshold:.3f}). "
                            f"Potential yawning event #{self.total_yawns}."
                        ),
                    )
                    generated_alerts.append(alert)
                    self.last_alert_time[AlertCategory.POSSIBLE_YAWNING] = timestamp
                    self.yawn_alert_fired = True
        else:
            self.yawn_start = None
            self.yawn_alert_fired = False

        state = DetectionState(
            timestamp=timestamp,
            mode=self.mode,
            ear=eye_metrics.ear,
            mar=mouth_metrics.mar,
            yaw=head_pose.yaw,
            pitch=head_pose.pitch,
            roll=head_pose.roll,
            is_face_detected=True,
            tracking_quality=tracking_quality,
            eye_closure_duration=closure_duration,
            distraction_duration=distraction_duration,
            yawn_duration=yawn_duration,
            blink_rate_bpm=blink_rate_bpm,
            total_blinks=self.blink_tracker.total_blinks,
            active_alerts=generated_alerts,
        )
        return state, generated_alerts
