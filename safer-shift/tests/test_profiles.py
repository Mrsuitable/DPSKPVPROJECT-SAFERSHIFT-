"""Tests for Driver Profile Persistence and Deletion."""

import shutil
import tempfile
import unittest
from pathlib import Path

from personalization.profiles import DriverProfile, ProfileManager


class TestProfiles(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.mgr = ProfileManager(profiles_dir=self.test_dir)

        self.sample_profile = DriverProfile(
            profile_id="test_driver_01",
            driver_name="Jordan Doe",
            created_at="2026-10-22T10:00:00Z",
            calibration_quality=0.88,
            baseline_ear_mean=0.27,
            baseline_ear_std=0.012,
            open_eye_ear_p25=0.26,
            open_eye_ear_p50=0.27,
            adapted_ear_threshold=0.195,
            neutral_yaw=0.5,
            neutral_pitch=-1.2,
            yaw_limit=25.0,
            pitch_down_limit=18.0,
            pitch_up_limit=22.0,
            baseline_mar_mean=0.21,
            sample_count=80,
            notes="Unit test sample profile",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_profile(self):
        path = self.mgr.save_profile(self.sample_profile)
        self.assertTrue(path.exists())

        loaded = self.mgr.load_profile("test_driver_01")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.profile_id, "test_driver_01")
        self.assertEqual(loaded.driver_name, "Jordan Doe")
        self.assertEqual(loaded.adapted_ear_threshold, 0.195)
        self.assertTrue(loaded.validate_bounds())

    def test_list_profiles(self):
        self.mgr.save_profile(self.sample_profile)
        profiles = self.mgr.list_profiles()
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].profile_id, "test_driver_01")

    def test_delete_profile(self):
        self.mgr.save_profile(self.sample_profile)
        self.assertIsNotNone(self.mgr.load_profile("test_driver_01"))

        deleted = self.mgr.delete_profile("test_driver_01")
        self.assertTrue(deleted)
        self.assertIsNone(self.mgr.load_profile("test_driver_01"))
        self.assertEqual(len(self.mgr.list_profiles()), 0)

    def test_corrupted_profile_handling(self):
        bad_file = self.test_dir / "bad_driver.json"
        with open(bad_file, "w") as f:
            f.write("corrupted { invalid json")

        # Loading bad file should return None safely
        self.assertIsNone(self.mgr.load_profile("bad_driver"))
        # list_profiles should skip the corrupt file without failing
        profiles = self.mgr.list_profiles()
        self.assertEqual(len(profiles), 0)


if __name__ == "__main__":
    unittest.main()
