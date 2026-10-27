"""SaferShift Driver Safety Intelligence Monitor.

Real-time, edge-based driver fatigue and distraction monitoring prototype.
Features:
- Live physical webcam capture (e.g. index 0)
- Prerecorded video file playback
- Synthetic / demo mode with simulated driver scenarios
- Real-time facial landmark tracking (MediaPipe FaceMesh/FaceLandmarker)
- Continuous Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose estimation
- Monotonic temporal engine: prolonged eye closure (>1.5s), yawning, head turns, blink rate
- Interactive stationary calibration with quality validation and adaptive thresholds
- On-screen heads-up display (HUD) with live metrics, alerts, and plain-English explanations
- Local privacy-preserving CSV telemetry logging
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Any

# Ensure project root is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from config import STANDARD_CONFIG, TEMPORAL_CONFIG
from detector.face_landmarks import FaceLandmarkDetector, FaceLandmarkResult
from detector.eye_metrics import calculate_ear, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from detector.mouth_metrics import calculate_mar
from detector.head_pose import estimate_head_pose
from detector.temporal_engine import TemporalEngine, DetectionState
from detector.alerts import AlertCategory, AlertSeverity, AlertEvent
from personalization.profiles import ProfileManager, DriverProfile
from personalization.calibration import CalibrationSession
from telemetry.event_logger import EventLogger
from telemetry.metrics import FPSMeter


def create_synthetic_landmarks(
    ear: float,
    mar: float,
    yaw: float,
    pitch: float,
    image_width: int = 640,
    image_height: int = 480,
) -> list:
    """Build normalized FaceMesh-like landmarks for deterministic demo inference."""
    landmarks = [(0.0, 0.0, 0.0) for _ in range(478)]

    def set_point(index: int, x: float, y: float) -> None:
        landmarks[index] = (x / image_width, y / image_height, 0.0)

    def set_eye(indices: list, outer_x: float, inner_x: float, center_y: float) -> None:
        eye_width = abs(inner_x - outer_x)
        half_opening = max(0.0, ear) * eye_width / 2.0
        set_point(indices[0], outer_x, center_y)
        set_point(indices[3], inner_x, center_y)
        set_point(indices[1], outer_x + (inner_x - outer_x) * 0.35, center_y - half_opening)
        set_point(indices[2], outer_x + (inner_x - outer_x) * 0.65, center_y - half_opening)
        set_point(indices[4], outer_x + (inner_x - outer_x) * 0.65, center_y + half_opening)
        set_point(indices[5], outer_x + (inner_x - outer_x) * 0.35, center_y + half_opening)

    set_eye([33, 160, 158, 133, 153, 144], 250.0, 300.0, 215.0)
    set_eye([362, 385, 387, 263, 373, 380], 340.0, 390.0, 215.0)

    mouth_width = 80.0
    mouth_opening = max(0.0, mar) * mouth_width
    set_point(78, 280.0, 315.0)
    set_point(308, 360.0, 315.0)
    set_point(13, 320.0, 315.0 - mouth_opening / 2.0)
    set_point(14, 320.0, 315.0 + mouth_opening / 2.0)
    set_point(82, 304.0, 315.0 - mouth_opening / 2.0)
    set_point(87, 304.0, 315.0 + mouth_opening / 2.0)
    set_point(312, 336.0, 315.0 - mouth_opening / 2.0)
    set_point(317, 336.0, 315.0 + mouth_opening / 2.0)

    eye_mid_x = 320.0
    eye_width = 140.0
    face_height = 100.0
    expected_nose_y = 215.0 + 0.4 * face_height
    set_point(1, eye_mid_x + (yaw / 45.0) * (eye_width / 2.0), expected_nose_y - (pitch / 55.0) * face_height)
    set_point(152, 320.0, 385.0)
    set_point(33, 250.0, 215.0)
    set_point(263, 390.0, 215.0)
    set_point(61, 290.0, 315.0)
    set_point(291, 350.0, 315.0)
    return landmarks


class SyntheticDriverSource:
    """Generates synthetic driver video frames for demonstration without physical webcam."""

    def __init__(self, fps: float = 30.0, width: int = 640, height: int = 480):
        self.fps = max(1.0, fps)
        self.width = width
        self.height = height
        self.start_time = time.monotonic()
        self.frame_count = 0
        self.elapsed = 0.0

    def current_measurements(self) -> Tuple[float, float, float, float]:
        t = self.elapsed % 40.0
        ear, mar, yaw, pitch = 0.28, 0.20, 0.0, 0.0
        if 3.0 <= t <= 3.2:
            ear = 0.06
        elif 8.0 <= t <= 10.5:
            ear = 0.05
        elif 16.0 <= t <= 19.0:
            yaw = 32.0
        elif 24.0 <= t <= 27.2:
            mar = 0.68
        return ear, mar, yaw, pitch

    def get_landmarks(self) -> list:
        ear, mar, yaw, pitch = self.current_measurements()
        return create_synthetic_landmarks(ear, mar, yaw, pitch, self.width, self.height)

    def read(self) -> Tuple[bool, np.ndarray]:
        target_time = self.start_time + self.frame_count / self.fps
        delay = target_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.frame_count += 1
        self.elapsed = time.monotonic() - self.start_time
        t = self.elapsed

        # Create base dashboard/cab frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (35, 30, 30)

        # Draw vehicle window background
        cv2.rectangle(frame, (40, 40), (600, 420), (50, 45, 45), -1)

        ear, mar, yaw, pitch = self.current_measurements()

        # Draw face oval
        center_x = int(320 + (yaw / 45.0) * 40)
        center_y = int(240 + (pitch / 45.0) * 30)
        cv2.ellipse(frame, (center_x, center_y), (100, 135), 0, 0, 360, (190, 205, 225), -1)
        cv2.ellipse(frame, (center_x, center_y), (100, 135), 0, 0, 360, (120, 140, 160), 2)

        # Draw eyes
        eye_w = 34
        eye_h = max(2, int(ear * eye_w * 1.5))
        left_eye_pos = (center_x - 42, center_y - 25)
        right_eye_pos = (center_x + 42, center_y - 25)

        for e_pos in [left_eye_pos, right_eye_pos]:
            if ear < 0.12:
                # Closed eye line
                cv2.line(frame, (e_pos[0] - eye_w // 2, e_pos[1]), (e_pos[0] + eye_w // 2, e_pos[1]), (50, 50, 70), 3)
            else:
                cv2.ellipse(frame, e_pos, (eye_w // 2, eye_h), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(frame, e_pos, max(2, eye_h // 2), (90, 50, 40), -1)
                cv2.ellipse(frame, e_pos, (eye_w // 2, eye_h), 0, 0, 360, (70, 80, 90), 2)

        # Draw nose
        nose_pt = (center_x, center_y + 15)
        cv2.line(frame, (center_x, center_y - 10), nose_pt, (140, 150, 170), 3)

        # Draw mouth
        m_w = 40
        m_h = max(3, int(mar * m_w * 0.9))
        mouth_pos = (center_x, center_y + 60)
        if mar > 0.45:
            cv2.ellipse(frame, mouth_pos, (m_w // 2, m_h), 0, 0, 360, (30, 20, 50), -1)
            cv2.ellipse(frame, mouth_pos, (m_w // 2, m_h), 0, 0, 360, (140, 60, 60), 2)
        else:
            cv2.ellipse(frame, mouth_pos, (m_w // 2, m_h), 0, 0, 360, (140, 60, 60), -1)

        # Add simulated metadata stamp
        cv2.putText(
            frame,
            f"SYNTHETIC DRIVER STREAM [t={t:.1f}s]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
        )

        return True, frame

    def release(self):
        pass


def draw_hud(
    frame: np.ndarray,
    state: Optional[DetectionState],
    active_alerts: list,
    fps: float,
    mode: str,
    profile_id: str,
    ear_threshold: float,
    mar_threshold: float,
    neutral_yaw: float,
    neutral_pitch: float,
    tracking_valid: bool = True,
):
    """Draws rich heads-up telemetry display and explainable alert banners on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top HUD banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 24, 30), -1)

    # Semi-transparent bottom status bar
    cv2.rectangle(overlay, (0, h - 65), (w, h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & Mode
    mode_color = (130, 215, 60) if mode == "personalized" else (240, 170, 50)
    mode_text = f"MODE: {mode.upper()}" + (f" ({profile_id})" if mode == "personalized" else " (DEFAULT)")
    cv2.putText(frame, "SAFERSHIFT DRIVER INTELLIGENCE", (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(frame, mode_text, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)

    # Measured FPS counter
    fps_color = (0, 255, 120) if fps >= 15.0 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, fps_color, 2)

    # Telemetry metrics row
    if tracking_valid and state is not None:
        ear_color = (0, 80, 255) if state.ear < ear_threshold else (0, 255, 120)
        mar_color = (0, 80, 255) if state.mar > mar_threshold else (0, 255, 120)
        yaw_dev = abs(state.yaw - neutral_yaw)
        pose_color = (0, 80, 255) if yaw_dev > STANDARD_CONFIG.YAW_THRESHOLD else (0, 255, 120)

        metrics_line = (
            f"EAR: {state.ear:.3f} (thresh {ear_threshold:.2f})  |  "
            f"MAR: {state.mar:.2f}  |  "
            f"YAW: {state.yaw:+.1f} deg  |  "
            f"BLINKS: {state.total_blinks} ({state.blink_rate_bpm:.0f}/min)"
        )
        cv2.putText(frame, metrics_line, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    else:
        cv2.putText(frame, "TRACKING: FACE NOT DETECTED / SEARCHING...", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)

    # Bottom status / Alert banner
    if active_alerts:
        alert = active_alerts[-1]
        is_crit = alert.severity == AlertSeverity.CRITICAL
        banner_bg = (30, 20, 180) if is_crit else (20, 120, 210)
        cv2.rectangle(frame, (10, h - 58), (w - 10, h - 8), banner_bg, -1)
        cv2.rectangle(frame, (10, h - 58), (w - 10, h - 8), (255, 255, 255), 1)

        badge = f"[{alert.category.value.upper()}]"
        cv2.putText(frame, badge, (22, h - 35), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, alert.explanation, (22, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1)
    else:
        cv2.putText(frame, "STATUS: DRIVER ALERT & ATTENTIVE", (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 2)
        cv2.putText(frame, "[Q] Quit  |  [M] Toggle Mode  |  [C] Calibrate Profile", (w - 380, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def draw_face_geometry(frame: np.ndarray, landmarks: list, image_width: int, image_height: int, ear_closed: bool, is_yawning: bool):
    """Draws eye and mouth contours on frame."""
    if not landmarks or len(landmarks) < 400:
        return

    def get_pt(idx):
        if idx < len(landmarks):
            p = landmarks[idx]
            return int(p[0] * image_width), int(p[1] * image_height)
        return None

    # Eyes
    left_pts = [get_pt(i) for i in LEFT_EYE_INDICES if get_pt(i) is not None]
    right_pts = [get_pt(i) for i in RIGHT_EYE_INDICES if get_pt(i) is not None]
    eye_color = (0, 50, 255) if ear_closed else (0, 255, 100)

    for pts in [left_pts, right_pts]:
        if len(pts) >= 6:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=True, color=eye_color, thickness=1)
            for pt in pts:
                cv2.circle(frame, pt, 2, eye_color, -1)

    # Mouth
    mouth_indices = [78, 82, 13, 312, 308, 317, 14, 87]
    mouth_pts = [get_pt(i) for i in mouth_indices if get_pt(i) is not None]
    m_color = (0, 140, 255) if is_yawning else (120, 220, 120)
    if len(mouth_pts) >= 8:
        cv2.polylines(frame, [np.array(mouth_pts, dtype=np.int32)], isClosed=True, color=m_color, thickness=1)

    # Nose tip reference
    nose_pt = get_pt(1)
    if nose_pt:
        cv2.circle(frame, nose_pt, 3, (255, 200, 0), -1)


def run_live_calibration(cap: Any, profile_id: str, driver_name: str, detector: FaceLandmarkDetector, duration_sec: float = 4.0, headless: bool = False) -> Optional[DriverProfile]:
    """Performs live calibration by collecting driver baseline observations."""
    print("\n" + "=" * 60)
    print(f"[*] STARTING SAFERSHIFT CALIBRATION FOR: {profile_id}")
    print("[*] Please sit stationary, look straight ahead, and keep eyes alert.")
    print("=" * 60)

    session = CalibrationSession(profile_id=profile_id, driver_name=driver_name)
    fps = 30.0
    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret or frame is None:
            session.add_sample(ear=0.0, mar=0.0, yaw=0.0, pitch=0.0, is_valid=False)
            time.sleep(1.0 / fps)
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process_rgb_frame(rgb)

        if result.is_valid and result.landmarks:
            ear_m = calculate_ear(result.landmarks, image_width=w, image_height=h)
            mar_m = calculate_mar(result.landmarks, image_width=w, image_height=h)
            pose = estimate_head_pose(result.landmarks, image_width=w, image_height=h)
            sample_valid = ear_m.is_valid and mar_m.is_valid and pose.is_valid
            session.add_sample(ear=ear_m.ear, mar=mar_m.mar, yaw=pose.yaw, pitch=pose.pitch, is_valid=sample_valid)
            draw_face_geometry(frame, result.landmarks, w, h, ear_m.is_closed, mar_m.is_open)
        else:
            session.add_sample(ear=0.0, mar=0.0, yaw=0.0, pitch=0.0, is_valid=False)

        # On-screen calibration progress banner
        pct = int(((i + 1) / total_frames) * 100)
        cv2.rectangle(frame, (50, h // 2 - 40), (w - 50, h // 2 + 40), (20, 24, 30), -1)
        cv2.rectangle(frame, (50, h // 2 - 40), (w - 50, h // 2 + 40), (0, 200, 255), 2)
        cv2.putText(frame, f"CALIBRATING: SIT STATIONARY & LOOK STRAIGHT ({pct}%)", (65, h // 2), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

        # Progress bar
        bar_w = int((w - 140) * (pct / 100.0))
        cv2.rectangle(frame, (70, h // 2 + 15), (70 + bar_w, h // 2 + 25), (0, 220, 100), -1)

        if not headless:
            try:
                cv2.imshow("SaferShift - Calibration Routine", frame)
                cv2.waitKey(1)
            except Exception:
                pass

        dt = time.monotonic() - t0
        time.sleep(max(0.001, (1.0 / fps) - dt))

    if not headless:
        try:
            cv2.destroyWindow("SaferShift - Calibration Routine")
        except Exception:
            pass

    result = session.finalize()
    if not result.success or result.profile is None:
        print(f"[!] Calibration rejected: {result.message}")
        return None

    mgr = ProfileManager()
    mgr.save_profile(result.profile)
    print(f"[+] Calibration successful! Quality: {result.quality_score:.2f}")
    print(f"[+] Adapted EAR threshold: {result.profile.adapted_ear_threshold:.3f}")
    return result.profile


def main():
    parser = argparse.ArgumentParser(description="SaferShift Driver Safety Intelligence Monitor")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0), path to video file, or 'demo'")
    parser.add_argument("--mode", choices=["standard", "personalized"], default="standard", help="Detection mode")
    parser.add_argument("--profile", default="driver_default", help="Driver Profile ID for personalized mode")
    parser.add_argument("--calibrate", action="store_true", help="Run interactive calibration before monitoring")
    parser.add_argument("--duration", type=float, default=0.0, help="Session duration in seconds (0 = run until 'q')")
    parser.add_argument("--headless", action="store_true", help="Run in headless console mode without GUI window")
    parser.add_argument("--log-file", default=None, help="Custom destination path for telemetry CSV")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      SaferShift: Personalized Driver Safety Intelligence       ")
    print("=" * 65)
    print(f"  Source:      {args.source}")
    print(f"  Mode:        {args.mode.upper()}")
    print(f"  Profile ID:  {args.profile}")
    print(f"  Headless:    {args.headless}")
    print("=" * 65 + "\n")

    is_demo = str(args.source).lower() in ["demo", "synthetic"]
    if is_demo and args.mode == "personalized" and args.calibrate:
        print("[!] Calibration requires a live camera; synthetic demo measurements cannot create a driver profile.")
        return 2

    # 1. Initialize a detector only for camera/video inputs. Demo landmarks are generated directly.
    detector = None if is_demo else FaceLandmarkDetector()
    if detector is not None and detector.detector is not None:
        print(f"[+] Facial landmark detector loaded (Backend: {detector.backend})")
    elif detector is not None:
        print("[!] Warning: MediaPipe detector not loaded. Operating in fallback / unit test mode.")

    # 2. Open Video Source
    cap = None

    if is_demo:
        print("[*] Launching synthetic driver simulation stream...")
        cap = SyntheticDriverSource()
    else:
        try:
            cam_idx = int(args.source)
            print(f"[*] Connecting to physical webcam at index {cam_idx}...")
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        except ValueError:
            print(f"[*] Opening prerecorded video file: {args.source}")
            cap = cv2.VideoCapture(args.source)

        if not cap or not cap.isOpened():
            print(f"[!] Warning: Could not open camera source '{args.source}'.")
            print("[*] Falling back to synthetic driver demonstration stream.")
            cap = SyntheticDriverSource()
            is_demo = True

    # 3. Handle Personalized Profile / Calibration
    profile_mgr = ProfileManager()
    active_profile = None

    if args.mode == "personalized":
        if args.calibrate:
            if is_demo or detector is None:
                print("[!] Calibration requires a live camera; synthetic demo measurements cannot create a driver profile.")
                cap.release()
                if detector is not None:
                    detector.close()
                return 2
            active_profile = run_live_calibration(cap, args.profile, args.profile, detector, headless=args.headless)
            if active_profile is None:
                cap.release()
                detector.close()
                return 2
        else:
            active_profile = profile_mgr.load_profile(args.profile)
            if active_profile is None:
                if is_demo:
                    print(f"[!] Profile '{args.profile}' not found. Calibrate from a live camera before using personalized mode.")
                    cap.release()
                    if detector is not None:
                        detector.close()
                    return 2
                print(f"[!] Profile '{args.profile}' not found on disk. Launching initial calibration...")
                if detector is None:
                    cap.release()
                    return 2
                active_profile = run_live_calibration(cap, args.profile, args.profile, detector, headless=args.headless)
                if active_profile is None:
                    cap.release()
                    detector.close()
                    return 2

    current_mode = args.mode
    ear_thresh = active_profile.adapted_ear_threshold if active_profile else STANDARD_CONFIG.EAR_THRESHOLD
    neutral_yaw = active_profile.neutral_yaw if active_profile else 0.0
    neutral_pitch = active_profile.neutral_pitch if active_profile else 0.0

    print(f"[+] Active EAR Threshold:    {ear_thresh:.3f}")
    print(f"[+] Neutral Head Pose:       Yaw: {neutral_yaw:.1f} deg | Pitch: {neutral_pitch:.1f} deg")

    # 4. Initialize Temporal Engine & Telemetry Logger
    engine = TemporalEngine(
        mode=current_mode,
        ear_threshold=ear_thresh,
        mar_threshold=STANDARD_CONFIG.MAR_THRESHOLD,
        neutral_yaw=neutral_yaw,
        neutral_pitch=neutral_pitch,
        geometric_head_pose=is_demo,
    )

    logger = EventLogger(output_filepath=Path(args.log_file) if args.log_file else None)
    fps_meter = FPSMeter()
    print(f"[+] Telemetry logging to:    {logger.filepath}")
    print("[*] Monitoring running. Press 'q' to quit, 'm' to toggle mode, 'c' to recalibrate.\n")

    start_time = time.monotonic()
    frame_count = 0
    active_alerts = []

    try:
        while True:
            loop_start = time.monotonic()
            elapsed = loop_start - start_time
            if args.duration > 0 and elapsed >= args.duration:
                print(f"[*] Session duration limit ({args.duration}s) reached.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                if not is_demo:
                    print("[*] Video source ended or disconnected.")
                    break
                continue

            h, w = frame.shape[:2]
            if is_demo:
                landmarks = cap.get_landmarks()
                result = FaceLandmarkResult(
                    landmarks=landmarks,
                    face_count=1,
                    bounding_box=(0.3, 0.2, 0.4, 0.6),
                    is_valid=True,
                )
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process_rgb_frame(rgb)

            state = None
            new_alerts = []

            if result.is_valid and result.landmarks:
                state, new_alerts = engine.process_frame(result.landmarks, timestamp=loop_start, image_width=w, image_height=h)
                ear_m = calculate_ear(result.landmarks, ear_threshold=ear_thresh, image_width=w, image_height=h)
                mar_m = calculate_mar(result.landmarks, mar_threshold=STANDARD_CONFIG.MAR_THRESHOLD, image_width=w, image_height=h)
                draw_face_geometry(frame, result.landmarks, w, h, ear_m.is_closed, mar_m.is_open)
            else:
                state, new_alerts = engine.process_frame(None, timestamp=loop_start, image_width=w, image_height=h)

            latency = time.monotonic() - loop_start
            current_fps = fps_meter.tick(latency)

            # Keep active alerts for banner display (fade after alert cooldown)
            if new_alerts:
                active_alerts = new_alerts
                for a in new_alerts:
                    print(f"\n[ALERT - {a.severity.value.upper()}] {a.category.value.upper()}")
                    print(f"  Explanation: {a.explanation}")
                    print(f"  Trigger: {a.trigger_metric} = {a.trigger_value:.3f}")
                    logger.log_state(
                        state,
                        event_type=a.category.value,
                        warning_reason=a.explanation,
                        measured_fps=current_fps,
                        force_flush=True,
                    )
            elif active_alerts and (loop_start - active_alerts[-1].start_time) > TEMPORAL_CONFIG.ALERT_COOLDOWN_SEC:
                active_alerts = []

            # Log periodic telemetry sample
            logger.log_state(state, event_type="sample", measured_fps=current_fps)

            # Draw HUD
            draw_hud(
                frame=frame,
                state=state,
                active_alerts=active_alerts,
                fps=current_fps,
                mode=current_mode,
                profile_id=args.profile,
                ear_threshold=ear_thresh,
                mar_threshold=STANDARD_CONFIG.MAR_THRESHOLD,
                neutral_yaw=neutral_yaw,
                neutral_pitch=neutral_pitch,
                tracking_valid=result.is_valid,
            )

            frame_count += 1

            # GUI display & interactive key handling
            if not args.headless:
                try:
                    cv2.imshow("SaferShift - Driver Safety Monitor", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), 27]:  # 'q' or ESC
                        print("\n[*] Terminating monitoring session...")
                        break
                    elif key == ord("m"):  # Toggle mode
                        if current_mode == "standard":
                            if active_profile is None:
                                active_profile = profile_mgr.load_profile(args.profile)
                            if active_profile is None and is_demo:
                                print("[!] Personalized mode needs a profile calibrated from a live camera.")
                                continue
                            if active_profile is None and detector is not None:
                                active_profile = run_live_calibration(cap, args.profile, args.profile, detector, headless=args.headless)
                            if active_profile is None:
                                continue
                            current_mode = "personalized"
                            ear_thresh = active_profile.adapted_ear_threshold
                            neutral_yaw = active_profile.neutral_yaw
                            neutral_pitch = active_profile.neutral_pitch
                        else:
                            current_mode = "standard"
                            ear_thresh = STANDARD_CONFIG.EAR_THRESHOLD
                            neutral_yaw = 0.0
                            neutral_pitch = 0.0
                        engine.configure_thresholds(ear_threshold=ear_thresh, neutral_yaw=neutral_yaw, neutral_pitch=neutral_pitch, mode=current_mode)
                        print(f"[*] Switched mode to: {current_mode.upper()} (EAR threshold: {ear_thresh:.3f})")
                    elif key == ord("c"):  # Live recalibration
                        if detector is None or is_demo:
                            print("[!] Calibration requires a live camera, not the synthetic demo stream.")
                            continue
                        calibrated_profile = run_live_calibration(cap, args.profile, args.profile, detector, headless=args.headless)
                        if calibrated_profile is None:
                            print("[!] The existing profile and monitoring mode were left unchanged.")
                            continue
                        active_profile = calibrated_profile
                        ear_thresh = active_profile.adapted_ear_threshold
                        neutral_yaw = active_profile.neutral_yaw
                        neutral_pitch = active_profile.neutral_pitch
                        engine.configure_thresholds(ear_threshold=ear_thresh, neutral_yaw=neutral_yaw, neutral_pitch=neutral_pitch, mode="personalized")
                        current_mode = "personalized"
                except Exception:
                    # In headless or non-GUI environments, ignore cv2.imshow errors
                    pass

    except KeyboardInterrupt:
        print("\n[*] Monitoring terminated by user.")
    finally:
        logger.close()
        cap.release()
        if detector is not None:
            detector.close()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        metrics = fps_meter.get_metrics()
        print("\n" + "=" * 55)
        print("SaferShift Session Summary:")
        print(f"  Total Frames:    {metrics.total_frames}")
        print(f"  Measured FPS:    {metrics.fps:.1f}")
        print(f"  Avg Latency:     {metrics.avg_latency_ms:.2f} ms")
        print(f"  Telemetry File:  {logger.filepath}")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
