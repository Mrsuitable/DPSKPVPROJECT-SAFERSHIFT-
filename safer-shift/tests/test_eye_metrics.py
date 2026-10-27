"""Tests for Eye Aspect Ratio (EAR) and blink tracking."""

import unittest
from detector.eye_metrics import (
    calculate_single_eye_ear,
    calculate_ear,
    BlinkTracker,
    detect_blinks,
    EyeMetrics,
)


class TestEyeMetrics(unittest.TestCase):

    def setUp(self):
        # Canonical open eye coordinates (width=50, height=15 -> EAR ~0.30)
        self.open_eye = [
            (100.0, 100.0),  # p1
            (115.0, 92.5),   # p2
            (135.0, 92.5),   # p3
            (150.0, 100.0),  # p4
            (135.0, 107.5),  # p5
            (115.0, 107.5),  # p6
        ]

        # Canonical closed eye coordinates (width=50, height=2 -> EAR ~0.04)
        self.closed_eye = [
            (100.0, 100.0),
            (115.0, 99.0),
            (135.0, 99.0),
            (150.0, 100.0),
            (135.0, 101.0),
            (115.0, 101.0),
        ]

    def test_single_eye_open(self):
        ear = calculate_single_eye_ear(self.open_eye)
        self.assertIsNotNone(ear)
        self.assertGreater(ear, 0.25)
        self.assertLess(ear, 0.35)

    def test_single_eye_closed(self):
        ear = calculate_single_eye_ear(self.closed_eye)
        self.assertIsNotNone(ear)
        self.assertLess(ear, 0.10)

    def test_invalid_and_empty_inputs(self):
        # Empty list
        self.assertIsNone(calculate_single_eye_ear([]))
        # None input
        self.assertIsNone(calculate_single_eye_ear(None))
        # Insufficient points
        self.assertIsNone(calculate_single_eye_ear([(10.0, 20.0), (30.0, 40.0)]))
        # Non-numeric coordinate
        bad_pts = [(10.0, "bad")] * 6
        self.assertIsNone(calculate_single_eye_ear(bad_pts))  # type: ignore

    def test_full_landmarks_ear_calculation(self):
        # Build dictionary of landmarks matching MediaPipe indices
        landmarks = {}
        for idx, pt in zip([33, 160, 158, 133, 153, 144], self.open_eye):
            landmarks[idx] = pt
        for idx, pt in zip([362, 385, 387, 263, 373, 380], self.open_eye):
            landmarks[idx] = (pt[0] + 100.0, pt[1])

        metrics = calculate_ear(landmarks, ear_threshold=0.21)
        self.assertTrue(metrics.is_valid)
        self.assertFalse(metrics.is_closed)
        self.assertGreater(metrics.ear, 0.25)

    def test_missing_landmarks_graceful_handling(self):
        # Missing right eye landmarks completely
        incomplete_landmarks = {33: (100.0, 100.0)}
        metrics = calculate_ear(incomplete_landmarks)
        self.assertFalse(metrics.is_valid)
        self.assertFalse(metrics.is_closed)
        self.assertEqual(metrics.ear, 0.0)

    def test_blink_tracker_counts_discrete_blinks(self):
        tracker = BlinkTracker(window_sec=60.0)
        t = 100.0

        # Eyes open initially
        self.assertFalse(tracker.update(is_closed=False, timestamp=t))
        # Eyes close for 0.15s (natural blink)
        self.assertFalse(tracker.update(is_closed=True, timestamp=t + 0.05))
        self.assertFalse(tracker.update(is_closed=True, timestamp=t + 0.15))
        # Eyes reopen: discrete blink completes
        blinked = tracker.update(is_closed=False, timestamp=t + 0.20)
        self.assertTrue(blinked)
        self.assertEqual(tracker.total_blinks, 1)

    def test_blink_tracker_does_not_count_prolonged_closure_as_blink(self):
        tracker = BlinkTracker(window_sec=60.0)
        t = 100.0

        # Eyes close for 2.0s (microsleep/prolonged closure, NOT a natural blink)
        tracker.update(is_closed=True, timestamp=t)
        tracker.update(is_closed=True, timestamp=t + 1.0)
        tracker.update(is_closed=True, timestamp=t + 2.0)
        # Reopening after 2.0s: should NOT be counted as a normal blink
        blinked = tracker.update(is_closed=False, timestamp=t + 2.1)
        self.assertFalse(blinked)
        self.assertEqual(tracker.total_blinks, 0)

    def test_detect_blinks_series_helper(self):
        ear_series = [0.30, 0.29, 0.12, 0.10, 0.30, 0.31, 0.09, 0.28]
        count = detect_blinks(ear_series, threshold=0.21)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
