"""Local Driver Profile Management.

Handles persistence, retrieval, inspection, and deletion of local driver profiles.
Privacy principle: Stores only derived numerical threshold statistics.
Does NOT store facial imagery, biometric identity keys, or video recordings.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import PROFILES_DIR, BOUNDS_CONFIG, STANDARD_CONFIG


@dataclass
class DriverProfile:
    profile_id: str
    driver_name: str
    created_at: str
    calibration_quality: float  # 0.0 to 1.0
    baseline_ear_mean: float
    baseline_ear_std: float
    open_eye_ear_p25: float
    open_eye_ear_p50: float
    adapted_ear_threshold: float
    neutral_yaw: float
    neutral_pitch: float
    yaw_limit: float
    pitch_down_limit: float
    pitch_up_limit: float
    baseline_mar_mean: float
    sample_count: int
    notes: str = ""

    def validate_bounds(self) -> bool:
        """Verifies that personalized thresholds stay within critical safety boundaries."""
        if not (BOUNDS_CONFIG.MIN_EAR_THRESHOLD <= self.adapted_ear_threshold <= BOUNDS_CONFIG.MAX_EAR_THRESHOLD):
            return False
        if not (BOUNDS_CONFIG.MIN_YAW_LIMIT <= self.yaw_limit <= BOUNDS_CONFIG.MAX_YAW_LIMIT):
            return False
        return True


class ProfileManager:
    """Manages local JSON profiles in the data/profiles directory."""

    def __init__(self, profiles_dir: Path = PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, profile_id: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", profile_id)
        return f"{clean}.json"

    def save_profile(self, profile: DriverProfile) -> Path:
        """Saves a driver profile to local disk."""
        filepath = self.profiles_dir / self._safe_filename(profile.profile_id)
        data = asdict(profile)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def load_profile(self, profile_id: str) -> Optional[DriverProfile]:
        """Loads a driver profile by ID. Returns None if file does not exist or is corrupted."""
        filepath = self.profiles_dir / self._safe_filename(profile_id)
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DriverProfile(**data)
        except Exception:
            return None

    def list_profiles(self) -> List[DriverProfile]:
        """Lists all valid local driver profiles."""
        profiles = []
        for file in self.profiles_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profiles.append(DriverProfile(**data))
            except Exception:
                continue
        profiles.sort(key=lambda p: p.created_at, reverse=True)
        return profiles

    def delete_profile(self, profile_id: str) -> bool:
        """Deletes a local driver profile permanently for privacy compliance."""
        filepath = self.profiles_dir / self._safe_filename(profile_id)
        if filepath.exists():
            try:
                filepath.unlink()
                return True
            except Exception:
                return False
        return False
