# SaferShift Model Assets

## Overview
SaferShift utilizes CPU-optimized facial landmark tracking.
By default, the system leverages:
1. **MediaPipe FaceMesh**: Standard lightweight 468-point 3D facial mesh running on local CPU.
2. **OpenCV Perspective-n-Point (solvePnP)**: 3D head pose estimation using calibrated anthropometric 3D facial landmarks.
3. **Robust Geometric Fallback**: Self-contained mathematical estimator embedded in `detector/head_pose.py` that computes Yaw, Pitch, and Roll directly from 2D coordinates when offline or in lightweight test environments.

## Privacy Note
No proprietary weights or biometric model biometric embeddings are stored here. Model inference runs entirely on the local device.
