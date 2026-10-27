"""Tests for the Sliding-Window Temporal Detection Engine."""

import unittest
from detector.alerts import AlertCategory, AlertSeverity
from detector.temporal_engine import TemporalEngine


def create_mock_landmarks(ear: float = 0.30, mar: float = 0.20, yaw: float = 0.0, pitch: float = 0.0):
    """Builds synthetic landmark coordinates matching given EAR, MAR, and yaw/pitch."""
    landmarks = {}
    h_eye = 50.0
    v_eye = ear * h_eye

    # Left eye: [33, 160, 158, 133, 153, 144]
    landmarks[33] = (100.0, 100.0)
    landmarks[133] = (100.0 + h_eye, 100.0)
    landmarks[160] = (100.0 + h_eye * 0.3, 100.0 - v_eye * 0.5)
    landmarks[158] = (100.0 + h_eye * 0.7, 100.0 - v_eye * 0.5)
    landmarks[144] = (100.0 + h_eye * 0.3, 100.0 + v_eye * 0.5)
    landmarks[153] = (100.0 + h_eye * 0.7, 100.0 + v_eye * 0.5)

    # Right eye: [362, 385, 387, 263, 373, 380]
    landmarks[362] = (200.0, 100.0)
    landmarks[263] = (200.0 + h_eye, 100.0)
    landmarks[385] = (200.0 + h_eye * 0.3, 100.0 - v_eye * 0.5)
    landmarks[387] = (200.0 + h_eye * 0.7, 100.0 - v_eye * 0.5)
    landmarks[380] = (200.0 + h_eye * 0.3, 100.0 + v_eye * 0.5)
    landmarks[373] = (200.0 + h_eye * 0.7, 100.0 + v_eye * 0.5)

    # Mouth landmarks: 13, 14, 82, 87, 312, 317, 78, 308
    w_mouth = 60.0
    v_mouth = mar * w_mouth
    landmarks[78] = (120.0, 180.0)
    landmarks[308] = (120.0 + w_mouth, 180.0)
    landmarks[13] = (120.0 + w_mouth * 0.5, 180.0 - v_mouth * 0.5)
    landmarks[14] = (120.0 + w_mouth * 0.5, 180.0 + v_mouth * 0.5)
    landmarks[82] = (120.0 + w_mouth * 0.3, 180.0 - v_mouth * 0.5)
    landmarks[87] = (120.0 + w_mouth * 0.3, 180.0 + v_mouth * 0.5)
    landmarks[312] = (120.0 + w_mouth * 0.7, 180.0 - v_mouth * 0.5)
    landmarks[317] = (120.0 + w_mouth * 0.7, 180.0 + v_mouth * 0.5)

    # Head pose landmarks: [1, 152, 61, 291]
    # Eye midpoint between 100.0 (left eye 33) and 250.0 (right eye 263) is 175.0 (half-width = 75.0)
    nose_shift = (yaw / 45.0) * 75.0
    landmarks[1] = (175.0 + nose_shift, 140.0)
    landmarks[152] = (175.0, 220.0)
    landmarks[61] = (145.0, 180.0)
    landmarks[291] = (205.0, 180.0)

    return landmarks


class TestTemporalEngine(unittest.TestCase):

    def setUp(self):
        self.engine = TemporalEngine(
            mode="standard",
            ear_threshold=0.21,
            mar_threshold=0.55,
            prolonged_closure_sec=1.5,
            distraction_sec=2.0,
            yawn_sec=2.2,
            alert_cooldown_sec=3.0,
            max_tracking_loss_tolerance_sec=0.4,
        )

    def test_short_closure_does_not_trigger_alert(self):
        """Eye closure < 1.5s (e.g. 1.0s) must NOT fire prolonged eye closure alert."""
        closed_lm = create_mock_landmarks(ear=0.08)

        # 1.0s of continuous closure
        all_alerts = []
        for step in range(11):
            t = 100.0 + (step * 0.1)
            _, alerts = self.engine.process_frame(closed_lm, timestamp=t)
            all_alerts.extend(alerts)

        closure_alerts = [a for a in all_alerts if a.category == AlertCategory.PROLONGED_EYE_CLOSURE]
        self.assertEqual(len(closure_alerts), 0)

    def test_closure_over_1_5s_triggers_prolonged_closure_alert(self):
        """Eye closure exceeding 1.5s must trigger a CRITICAL prolonged closure alert."""
        closed_lm = create_mock_landmarks(ear=0.08)

        all_alerts = []
        for step in range(18):  # 1.7 seconds total
            t = 100.0 + (step * 0.1)
            state, alerts = self.engine.process_frame(closed_lm, timestamp=t)
            all_alerts.extend(alerts)

        closure_alerts = [a for a in all_alerts if a.category == AlertCategory.PROLONGED_EYE_CLOSURE]
        self.assertEqual(len(closure_alerts), 1)
        alert = closure_alerts[0]
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        self.assertGreaterEqual(alert.duration, 1.5)
        self.assertIn("Eyes closed continuously", alert.explanation)

    def test_alert_spam_prevention(self):
        """Continuing the same closed-eye state must NOT spam new alerts on every frame."""
        closed_lm = create_mock_landmarks(ear=0.08)

        all_alerts = []
        for step in range(35):  # 3.4 seconds total
            t = 100.0 + (step * 0.1)
            _, alerts = self.engine.process_frame(closed_lm, timestamp=t)
            all_alerts.extend(alerts)

        closure_alerts = [a for a in all_alerts if a.category == AlertCategory.PROLONGED_EYE_CLOSURE]
        # Only 1 initial alert triggered before condition clears or cooldown elapses
        self.assertEqual(len(closure_alerts), 1)

    def test_brief_tracking_interruption_continuity(self):
        """Brief tracking dropout (<=0.4s) does not wipe closure state immediately."""
        closed_lm = create_mock_landmarks(ear=0.08)

        # Close eyes for 1.0s
        for step in range(10):
            t = 100.0 + (step * 0.1)
            self.engine.process_frame(closed_lm, timestamp=t)

        # Brief dropout for 0.2s (2 frames missing)
        self.engine.process_frame(None, timestamp=101.1)
        self.engine.process_frame(None, timestamp=101.2)

        # Eyes still closed for another 0.6s -> total closure exceeds 1.5s!
        all_alerts = []
        for step in range(3, 9):
            t = 101.0 + (step * 0.1)
            _, alerts = self.engine.process_frame(closed_lm, timestamp=t)
            all_alerts.extend(alerts)

        closure_alerts = [a for a in all_alerts if a.category == AlertCategory.PROLONGED_EYE_CLOSURE]
        self.assertGreaterEqual(len(closure_alerts), 1)

    def test_sustained_tracking_loss_resets_closure_and_warns(self):
        """Long tracking loss (>0.4s) must NEVER be counted as confirmed closure."""
        closed_lm = create_mock_landmarks(ear=0.08)

        # Close eyes for 0.8s
        for step in range(8):
            t = 100.0 + (step * 0.1)
            self.engine.process_frame(closed_lm, timestamp=t)

        # Now tracking is lost for 1.5s
        all_alerts = []
        for step in range(15):
            t = 100.8 + (step * 0.1)
            state, alerts = self.engine.process_frame(None, timestamp=t)
            all_alerts.extend(alerts)

        # Must NOT generate prolonged eye closure
        closure_alerts = [a for a in all_alerts if a.category == AlertCategory.PROLONGED_EYE_CLOSURE]
        self.assertEqual(len(closure_alerts), 0)

        # Must flag tracking unreliable info
        tracking_alerts = [a for a in all_alerts if a.category == AlertCategory.TRACKING_UNRELIABLE]
        self.assertGreaterEqual(len(tracking_alerts), 1)

    def test_sustained_distraction_detection(self):
        """Head turn >25 degrees sustained for >2.0s triggers distraction alert."""
        turned_lm = create_mock_landmarks(ear=0.30, yaw=32.0)

        all_alerts = []
        for step in range(25):  # 2.4 seconds
            t = 100.0 + (step * 0.1)
            _, alerts = self.engine.process_frame(turned_lm, timestamp=t)
            all_alerts.extend(alerts)

        distraction_alerts = [a for a in all_alerts if a.category == AlertCategory.SUSTAINED_DISTRACTION]
        self.assertEqual(len(distraction_alerts), 1)
        self.assertEqual(distraction_alerts[0].severity, AlertSeverity.WARNING)

    def test_possible_yawning_detection(self):
        """Mouth open > 0.55 sustained for >2.2s triggers yawning alert."""
        yawn_lm = create_mock_landmarks(ear=0.30, mar=0.68)

        all_alerts = []
        for step in range(26):  # 2.5 seconds
            t = 100.0 + (step * 0.1)
            _, alerts = self.engine.process_frame(yawn_lm, timestamp=t)
            all_alerts.extend(alerts)

        yawn_alerts = [a for a in all_alerts if a.category == AlertCategory.POSSIBLE_YAWNING]
        self.assertEqual(len(yawn_alerts), 1)


if __name__ == "__main__":
    unittest.main()
