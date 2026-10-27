"""Face landmark detector wrapper.

Supports MediaPipe FaceMesh (solutions) and FaceLandmarker (tasks) with CPU execution.
Implements:
1. Documented primary subject selection rule for multiple faces:
   Selects the face with the largest area and highest central alignment.
2. Graceful fallback when MediaPipe is not installed, model is missing, or running offline.
3. Dual-backend compatibility across MediaPipe versions (0.9, 0.10, 1.x).
4. Landmark normalization and extraction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any
import math
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@dataclass
class FaceLandmarkResult:
    landmarks: Optional[List[Tuple[float, float, float]]]  # Normalized [x, y, z]
    face_count: int
    bounding_box: Optional[Tuple[float, float, float, float]]  # (x_min, y_min, width, height)
    is_valid: bool


class FaceLandmarkDetector:
    """Wrapper around MediaPipe FaceMesh/FaceLandmarker with CPU inference and subject selection."""

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 2, min_detection_confidence: float = 0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.detector = None
        self.backend = None  # "solutions" or "tasks"
        self._initialize_backend()

    def _initialize_backend(self):
        # 1. Attempt MediaPipe solutions backend (traditional lightweight FaceMesh)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=self.static_image_mode,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=0.5,
                )
                self.backend = "solutions"
                return
        except Exception:
            pass

        # 2. Attempt MediaPipe tasks backend (modern FaceLandmarker)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not MODEL_PATH.exists():
                self._download_model_asset()

            if MODEL_PATH.exists():
                base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=self.max_num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.detector = mp_vision.FaceLandmarker.create_from_options(options)
                self.backend = "tasks"
                return
        except Exception:
            pass

        self.detector = None
        self.backend = None

    def _download_model_asset(self):
        """Attempts to download the official MediaPipe FaceLandmarker model if missing."""
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"[*] Downloading MediaPipe face landmark model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print("[+] Model asset downloaded successfully.")
        except Exception as e:
            print(f"[!] Warning: Could not auto-download model asset: {e}")

    def process_rgb_frame(self, rgb_image: Any) -> FaceLandmarkResult:
        """Processes an RGB image and returns normalized landmarks for the primary driver face."""
        if self.detector is None or rgb_image is None:
            return FaceLandmarkResult(landmarks=None, face_count=0, bounding_box=None, is_valid=False)

        try:
            candidate_faces = []

            if self.backend == "solutions":
                results = self.detector.process(rgb_image)
                if not results or not results.multi_face_landmarks:
                    return FaceLandmarkResult(landmarks=None, face_count=0, bounding_box=None, is_valid=False)
                candidate_faces = [fl.landmark for fl in results.multi_face_landmarks]

            elif self.backend == "tasks":
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                results = self.detector.detect(mp_image)
                if not results or not results.face_landmarks:
                    return FaceLandmarkResult(landmarks=None, face_count=0, bounding_box=None, is_valid=False)
                candidate_faces = results.face_landmarks

            else:
                return FaceLandmarkResult(landmarks=None, face_count=0, bounding_box=None, is_valid=False)

            if not candidate_faces:
                return FaceLandmarkResult(landmarks=None, face_count=0, bounding_box=None, is_valid=False)

            face_count = len(candidate_faces)

            # Subject Selection Rule:
            # If multiple faces appear in frame, select the primary driver face based on:
            # Score = Area * max(0.1, 1.0 - distance_from_center)
            best_face = None
            best_score = -1.0
            best_bbox = None

            for face_pts in candidate_faces:
                x_vals = [p.x for p in face_pts]
                y_vals = [p.y for p in face_pts]
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)
                w = max(1e-4, x_max - x_min)
                h = max(1e-4, y_max - y_min)
                area = w * h

                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                dist_from_center = math.hypot(center_x - 0.5, center_y - 0.5)

                score = area * max(0.1, 1.0 - dist_from_center)
                if score > best_score:
                    best_score = score
                    best_face = face_pts
                    best_bbox = (x_min, y_min, w, h)

            if best_face is None:
                return FaceLandmarkResult(landmarks=None, face_count=0, bounding_box=None, is_valid=False)

            extracted_pts = [(p.x, p.y, getattr(p, "z", 0.0)) for p in best_face]
            return FaceLandmarkResult(
                landmarks=extracted_pts,
                face_count=face_count,
                bounding_box=best_bbox,
                is_valid=True,
            )

        except Exception:
            return FaceLandmarkResult(landmarks=None, face_count=0, bounding_box=None, is_valid=False)

    def close(self):
        if self.detector is not None and hasattr(self.detector, "close"):
            try:
                self.detector.close()
            except Exception:
                pass
