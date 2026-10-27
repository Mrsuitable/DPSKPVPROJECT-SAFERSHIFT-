"""Tests for Driver Calibration and Safety Bound Enforcement."""

import unittest
from config import BOUNDS_CONFIG
from personalization.calibration import CalibrationSession


class TestCalibration(unittest.TestCase):

    def test_successful_calibration_creates_valid_profile(self):
        session = CalibrationSession(profile_id="driver_test_valid", driver_name="Alex Smith")

        # Feed 60 alert open-eye samples (EAR ~0.28, yaw ~0.0, pitch ~0.0)
        for i in range(60):
            session.add_sample(
                ear=0.28 + 0.005 * (i % 3),
                mar=0.20,
                yaw=1.0 * (i % 2),
                pitch=0.0,
                is_valid=True,
            )

        result = session.finalize()
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.quality_score, BOUNDS_CONFIG.MIN_CALIBRATION_QUALITY)
        self.assertIsNotNone(result.profile)

        profile = result.profile
        self.assertEqual(profile.profile_id, "driver_test_valid")
        # Adapted threshold should be within bounds
        self.assertGreaterEqual(profile.adapted_ear_threshold, BOUNDS_CONFIG.MIN_EAR_THRESHOLD)
        self.assertLessEqual(profile.adapted_ear_threshold, BOUNDS_CONFIG.MAX_EAR_THRESHOLD)
        self.assertTrue(profile.validate_bounds())

    def test_insufficient_samples_rejected(self):
        session = CalibrationSession(profile_id="driver_insufficient")

        # Feed only 10 samples (well below 45 required)
        for _ in range(10):
            session.add_sample(ear=0.28, mar=0.20, yaw=0.0, pitch=0.0, is_valid=True)

        result = session.finalize()
        self.assertFalse(result.success)
        self.assertIsNone(result.profile)
        self.assertIn("Insufficient calibration samples", result.message)

    def test_sleepy_eyes_during_calibration_rejected(self):
        """Driver calibrating with closed or squinting eyes must be rejected."""
        session = CalibrationSession(profile_id="driver_sleepy_calib")

        # Median EAR is 0.12 (predominantly closed)
        for _ in range(50):
            session.add_sample(ear=0.12, mar=0.20, yaw=0.0, pitch=0.0, is_valid=True)

        result = session.finalize()
        self.assertFalse(result.success)
        self.assertIn("predominantly closed", result.message)

    def test_high_jitter_drops_quality_score(self):
        """High head movement / tracking jitter drops quality."""
        session = CalibrationSession(profile_id="driver_jittery")

        for i in range(50):
            # Extreme yaw swinging
            session.add_sample(ear=0.28, mar=0.20, yaw=float((i % 2) * 40.0), pitch=0.0, is_valid=True)

        result = session.finalize()
        self.assertFalse(result.success)
        self.assertIn("below threshold", result.message)

    def test_adaptive_threshold_safety_bounds_clamping(self):
        """Even with very high or low baseline EAR, adapted threshold stays within safety bounds."""
        # Unusually high baseline (EAR ~0.42)
        session = CalibrationSession(profile_id="driver_huge_eyes")
        for _ in range(50):
            session.add_sample(ear=0.42, mar=0.20, yaw=0.0, pitch=0.0, is_valid=True)

        result = session.finalize()
        self.assertTrue(result.success)
        self.assertLessEqual(result.profile.adapted_ear_threshold, BOUNDS_CONFIG.MAX_EAR_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
