"""Standalone MediaPipe & OpenCV Detector Verification.

Tests:
1. MediaPipe import & compatible Face Landmark backend initialization.
2. OpenCV camera capture & fallback handling.
3. Facial landmark extraction, EAR, MAR, and Head Pose calculation.
4. Actual processing FPS measurement.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.face_landmarks import FaceLandmarkDetector
from detector.eye_metrics import calculate_ear
from detector.mouth_metrics import calculate_mar
from detector.head_pose import estimate_head_pose
from telemetry.metrics import FPSMeter


def run_verification(source=0):
    print("=" * 60)
    print("SaferShift Standalone Detector Verification")
    print("=" * 60)

    # 1. Test OpenCV
    try:
        import cv2
        print(f"[OK] OpenCV installed: version {cv2.__version__}")
    except ImportError as e:
        print(f"[FAIL] OpenCV not available: {e}")
        return False

    # 2. Test MediaPipe
    try:
        import mediapipe as mp
        print(f"[OK] MediaPipe installed: version {getattr(mp, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"[FAIL] MediaPipe not available: {e}")
        return False

    # 3. Test Detector Backend
    detector = FaceLandmarkDetector()
    if detector.detector is None:
        print("[WARN] MediaPipe FaceMesh/FaceLandmarker not initialized yet (checking backends...)")
    else:
        print("[OK] Face landmark detector backend initialized successfully.")

    # 4. Test Video Source
    print(f"\n[*] Opening video source: {source}")
    try:
        src = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[!] Could not open video source {source}. Testing with synthetic frame fallback.")
            cap = None
    except Exception as e:
        print(f"[!] Camera error: {e}. Testing with synthetic frame fallback.")
        cap = None

    fps_meter = FPSMeter()
    frames_processed = 0

    print("\n[*] Running 30 inference iterations...")
    start_time = time.monotonic()

    for i in range(30):
        t0 = time.monotonic()
        frame = None
        if cap is not None:
            ret, cap_frame = cap.read()
            if ret and cap_frame is not None:
                frame = cap_frame

        if frame is None:
            # Create a 640x480 test image with a simple human face representation
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (40, 40, 40)
            cv2.ellipse(frame, (320, 240), (120, 160), 0, 0, 360, (180, 200, 220), -1)

        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process_rgb_frame(rgb_frame)

        # Calculate metrics if face detected
        if result.is_valid and result.landmarks:
            ear_metrics = calculate_ear(result.landmarks, image_width=640, image_height=480)
            mar_metrics = calculate_mar(result.landmarks, image_width=640, image_height=480)
            pose = estimate_head_pose(result.landmarks, image_width=640, image_height=480)
            if i == 0:
                print(f"[+] Face detected! Landmarks: {len(result.landmarks)}")
                print(f"    Left EAR: {ear_metrics.left_ear:.3f}, Right EAR: {ear_metrics.right_ear:.3f}, Avg: {ear_metrics.ear:.3f}")
                print(f"    MAR: {mar_metrics.mar:.3f}")
                print(f"    Head Pose: Yaw={pose.yaw:.1f}°, Pitch={pose.pitch:.1f}°, Roll={pose.roll:.1f}°")

        dt = time.monotonic() - t0
        fps_meter.tick(dt)
        frames_processed += 1

    if cap is not None:
        cap.release()
    detector.close()

    metrics = fps_meter.get_metrics()
    total_time = time.monotonic() - start_time
    print("\n" + "=" * 60)
    print("Verification Summary:")
    print(f"  Total Frames:     {metrics.total_frames}")
    print(f"  Measured FPS:     {metrics.fps:.1f}")
    print(f"  Avg Latency:      {metrics.avg_latency_ms:.2f} ms")
    print(f"  Execution Time:   {total_time:.2f} s")
    print("=" * 60)
    return True


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    run_verification(source=src)
