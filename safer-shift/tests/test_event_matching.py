"""Tests for Event Matching and Metric Calculation."""

import shutil
import tempfile
import unittest
from pathlib import Path

from evaluation.event_matching import (
    GroundTruthEvent,
    PredictedEvent,
    match_events,
    calculate_metrics,
)
from telemetry.event_logger import load_telemetry_records, EventLogger


class TestEventMatching(unittest.TestCase):

    def test_perfect_event_matching(self):
        gt = [
            GroundTruthEvent("clip1", 5.0, 8.0, "prolonged_eye_closure"),
            GroundTruthEvent("clip1", 15.0, 18.0, "sustained_distraction"),
        ]
        preds = [
            PredictedEvent("clip1", 5.2, 8.2, "prolonged_eye_closure", mode="standard"),
            PredictedEvent("clip1", 15.1, 17.9, "sustained_distraction", mode="standard"),
        ]

        res = match_events(gt, preds)
        self.assertEqual(len(res.matched_pairs), 2)
        self.assertEqual(len(res.unmatched_ground_truths), 0)
        self.assertEqual(len(res.unmatched_predictions), 0)
        self.assertEqual(len(res.duplicate_predictions), 0)

        metrics = calculate_metrics(gt, preds, total_duration_sec=30.0)
        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.false_positives, 0)
        self.assertEqual(metrics.false_negatives, 0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1_score, 1.0)

    def test_duplicate_prediction_counted_as_false_positive(self):
        """If two predictions fire for a single ground truth event, the second is a duplicate FP."""
        gt = [GroundTruthEvent("clip1", 5.0, 8.0, "prolonged_eye_closure")]
        preds = [
            PredictedEvent("clip1", 5.1, 7.0, "prolonged_eye_closure"),
            PredictedEvent("clip1", 6.0, 8.0, "prolonged_eye_closure"),  # Duplicate!
        ]

        res = match_events(gt, preds)
        self.assertEqual(len(res.matched_pairs), 1)
        self.assertEqual(len(res.duplicate_predictions), 1)

        metrics = calculate_metrics(gt, preds, total_duration_sec=60.0)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.precision, 0.5)

    def test_false_alarms_per_minute(self):
        gt = []  # No ground truth events (clean driving)
        preds = [
            PredictedEvent("clip1", 10.0, 12.0, "prolonged_eye_closure"),
            PredictedEvent("clip1", 40.0, 42.0, "sustained_distraction"),
        ]
        # 2 false alarms in 2 minutes = 1.0 FA/min
        metrics = calculate_metrics(gt, preds, total_duration_sec=120.0)
        self.assertEqual(metrics.false_positives, 2)
        self.assertEqual(metrics.false_alarms_per_minute, 1.0)

    def test_telemetry_corrupted_and_empty_file_loading(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Nonexistent file
            records = load_telemetry_records(temp_dir / "nonexistent.csv")
            self.assertEqual(records, [])

            # 2. Empty file
            empty_file = temp_dir / "empty.csv"
            empty_file.touch()
            records = load_telemetry_records(empty_file)
            self.assertEqual(records, [])

            # 3. File with partial/malformed row
            bad_file = temp_dir / "bad.csv"
            with open(bad_file, "w") as f:
                f.write("timestamp,timestamp_iso,event_type,mode,ear,mar,yaw,pitch,roll,closure_duration,distraction_duration,yawn_duration,blink_rate_bpm,total_blinks,calibration_status,warning_reason,tracking_quality,measured_fps\n")
                f.write("100.1,iso,sample,standard,0.25,0.18,0.0,0.0,0.0,0.0,0.0,0.0,15.0,2,calibrated,,1.0,28.5\n")
                f.write("corrupted,line,with,missing,numbers\n")
                f.write("100.3,iso,sample,standard,0.24,0.19,0.0,0.0,0.0,0.0,0.0,0.0,15.0,2,calibrated,,1.0,28.2\n")

            records = load_telemetry_records(bad_file)
            # Should have parsed the 2 valid lines and gracefully ignored the corrupt one
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].ear, 0.25)
            self.assertEqual(records[1].ear, 0.24)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_event_logger_writing_and_immediate_alert_flush(self):
        """Verifies that EventLogger logs samples and immediately flushes critical alerts."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            log_path = temp_dir / "test_telemetry.csv"
            logger = EventLogger(output_filepath=log_path, buffer_size=10)

            class DummyState:
                timestamp = 100.0
                mode = "standard"
                ear = 0.28
                mar = 0.20
                yaw = 0.0
                pitch = 0.0
                roll = 0.0
                eye_closure_duration = 0.0
                distraction_duration = 0.0
                yawn_duration = 0.0
                blink_rate_bpm = 16.0
                total_blinks = 4
                tracking_quality = 1.0

            # Log periodic sample (buffered)
            logger.log_state(DummyState(), event_type="sample", measured_fps=30.0)

            # Log critical alert (should flush immediately)
            DummyState.timestamp = 101.5
            DummyState.eye_closure_duration = 1.5
            logger.log_state(
                DummyState(),
                event_type="prolonged_eye_closure",
                warning_reason="Eyes closed >1.5s",
                measured_fps=30.0,
            )

            records = load_telemetry_records(log_path)
            self.assertGreaterEqual(len(records), 1)
            event_records = [r for r in records if r.event_type == "prolonged_eye_closure"]
            self.assertEqual(len(event_records), 1)
            self.assertEqual(event_records[0].warning_reason, "Eyes closed >1.5s")
            self.assertEqual(event_records[0].closure_duration, 1.5)
            logger.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
