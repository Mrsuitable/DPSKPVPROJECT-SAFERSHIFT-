"""Head pose estimation module using 3D facial landmarks and solvePnP.

Estimates approximate Yaw (turn left/right), Pitch (tilt up/down), and Roll (tilt sideways).
Establishes driver neutral orientation during calibration and checks for sustained deviations.
Includes pure-math geometric fallback if OpenCV solvePnP fails.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any


# 3D Reference Model Points (in millimeters, based on standard human face anthropometry)
MODEL_POINTS_3D = [
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (index 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (index 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
    (150.0, -150.0, -125.0),     # Right mouth corner (index 291)
]

# MediaPipe landmark indices corresponding to above 3D points
HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


@dataclass
class HeadPose:
    yaw: float       # Negative = driver turning left; positive = driver turning right
    pitch: float     # Negative = driver looking down; positive = driver looking up
    roll: float      # Tilt to shoulder
    is_valid: bool
    is_distracted: bool = False


def _extract_2d_points(
    landmarks: Any,
    image_width: int,
    image_height: int,
) -> Optional[List[Tuple[float, float]]]:
    """Extracts the 6 key 2D points from the landmark set."""
    if landmarks is None:
        return None

    pts = []
    try:
        for idx in HEAD_POSE_LANDMARK_INDICES:
            pt = None
            if isinstance(landmarks, (list, tuple)):
                if idx < len(landmarks):
                    item = landmarks[idx]
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        pt = (float(item[0]) * image_width, float(item[1]) * image_height)
                    elif hasattr(item, "x") and hasattr(item, "y"):
                        pt = (float(item.x) * image_width, float(item.y) * image_height)
            elif isinstance(landmarks, dict) and idx in landmarks:
                item = landmarks[idx]
                pt = (float(item[0]) * image_width, float(item[1]) * image_height)

            if pt is None:
                return None
            pts.append(pt)
        return pts
    except Exception:
        return None


def estimate_head_pose_geometric(pts: List[Tuple[float, float]], image_width: int, image_height: int) -> HeadPose:
    """Robust geometric fallback for head pose estimation from 2D facial landmarks.

    Uses symmetry of eye corners, nose position, and mouth baseline to compute
    yaw, pitch, and roll in degrees without requiring OpenCV solvePnP.
    """
    try:
        nose, chin, left_eye, right_eye, left_mouth, right_mouth = pts[:6]

        # Roll: angle between eyes
        d_x = right_eye[0] - left_eye[0]
        d_y = right_eye[1] - left_eye[1]
        roll = math.degrees(math.atan2(d_y, d_x))

        # Yaw: asymmetry of nose relative to eye midpoint
        eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_width = math.hypot(d_x, d_y)
        if eye_width > 1e-4:
            # Shift of nose from eye midpoint
            yaw_ratio = (nose[0] - eye_mid_x) / (eye_width * 0.5)
            yaw = float(yaw_ratio * 45.0)  # ~45 deg at edge
        else:
            yaw = 0.0

        # Pitch: vertical proportion of nose between eye-line and mouth-line
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
        face_height = mouth_mid_y - eye_mid_y
        if face_height > 1e-4:
            # Expected neutral nose position is ~40% between eye line and mouth line
            expected_nose_y = eye_mid_y + 0.40 * face_height
            pitch_shift = (nose[1] - expected_nose_y) / face_height
            pitch = float(-pitch_shift * 55.0)  # positive = looking up, negative = looking down
        else:
            pitch = 0.0

        # Clamp to realistic head range
        yaw = max(-90.0, min(90.0, yaw))
        pitch = max(-90.0, min(90.0, pitch))
        roll = max(-90.0, min(90.0, roll))

        return HeadPose(
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            is_valid=True,
        )
    except Exception:
        return HeadPose(0.0, 0.0, 0.0, is_valid=False)


def estimate_head_pose(
    landmarks: Any,
    image_width: int = 640,
    image_height: int = 480,
    neutral_yaw: float = 0.0,
    neutral_pitch: float = 0.0,
    yaw_threshold: float = 25.0,
    pitch_down_threshold: float = 18.0,
    pitch_up_threshold: float = 22.0,
    force_geometric: bool = False,
) -> HeadPose:
    """Estimates head pose (yaw, pitch, roll) and evaluates deviation from neutral baseline."""
    pts = _extract_2d_points(landmarks, image_width, image_height)
    if not pts or len(pts) < 6:
        return HeadPose(0.0, 0.0, 0.0, is_valid=False)

    pose = None
    if not force_geometric:
        # Try OpenCV solvePnP if cv2 and numpy are available.
        try:
            import numpy as np
            import cv2

            points_2d = np.array(pts, dtype=np.float64)
            points_3d = np.array(MODEL_POINTS_3D, dtype=np.float64)

            focal_length = image_width
            center = (image_width / 2.0, image_height / 2.0)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                # Decompose rotation matrix into Euler angles
                sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                singular = sy < 1e-6

                if not singular:
                    x = math.atan2(rmat[2, 1], rmat[2, 2])
                    y = math.atan2(-rmat[2, 0], sy)
                    z = math.atan2(rmat[1, 0], rmat[0, 0])
                else:
                    x = math.atan2(-rmat[1, 2], rmat[1, 1])
                    y = math.atan2(-rmat[2, 0], sy)
                    z = 0.0

                pitch = math.degrees(x)
                yaw = math.degrees(y)
                roll = math.degrees(z)

                pose = HeadPose(
                    yaw=round(yaw, 2),
                    pitch=round(pitch, 2),
                    roll=round(roll, 2),
                    is_valid=True,
                )
        except Exception:
            pose = None

    if pose is None or not pose.is_valid:
        pose = estimate_head_pose_geometric(pts, image_width, image_height)

    # Check deviation from neutral reference
    delta_yaw = abs(pose.yaw - neutral_yaw)
    delta_pitch = pose.pitch - neutral_pitch

    is_distracted = (
        delta_yaw >= yaw_threshold or
        delta_pitch <= -pitch_down_threshold or
        delta_pitch >= pitch_up_threshold
    )
    pose.is_distracted = is_distracted
    return pose
