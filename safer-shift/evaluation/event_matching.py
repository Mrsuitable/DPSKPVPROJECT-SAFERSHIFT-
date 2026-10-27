"""Event Matching and Comparative Metric Calculation.

Implements temporal event-level matching (not per-frame inflation).
Calculates:
- Event-level Precision, Recall, F1 Score
- Confusion Matrix (TP, FP, FN)
- False Alarms per Minute
- Mean Detection Delay (seconds)
- Robustness subgroup breakdowns (glasses vs no-glasses, daylight vs night)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class GroundTruthEvent:
    recording_id: str
    start_time: float
    end_time: float
    event_type: str  # "prolonged_eye_closure", "sustained_distraction", "possible_yawning"
    driver_id: str = "driver_1"
    lighting: str = "normal"      # "normal", "low_light", "glare"
    glasses: str = "no"          # "yes", "no"
    notes: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class PredictedEvent:
    recording_id: str
    start_time: float
    end_time: float
    event_type: str
    mode: str = "standard"  # "standard" or "personalized"
    trigger_metric: str = ""
    trigger_value: float = 0.0
    explanation: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class EventMatchResult:
    matched_pairs: List[Tuple[GroundTruthEvent, PredictedEvent, float]]  # (GT, Pred, delay)
    unmatched_ground_truths: List[GroundTruthEvent]                       # False Negatives (FN)
    unmatched_predictions: List[PredictedEvent]                           # False Positives (FP)
    duplicate_predictions: List[PredictedEvent]                           # Duplicate FP detections


@dataclass
class EvaluationMetrics:
    mode: str
    total_evaluated_duration_sec: float
    total_ground_truth_events: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_alarms_per_minute: float
    mean_detection_delay_sec: float
    category_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    robustness_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def compute_temporal_iou(gt_start: float, gt_end: float, pred_start: float, pred_end: float) -> float:
    """Computes temporal Intersection over Union (tIoU) between two time segments."""
    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    intersection = max(0.0, inter_end - inter_start)

    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def match_events(
    ground_truths: List[GroundTruthEvent],
    predictions: List[PredictedEvent],
    tolerance_sec: float = 1.5,
    min_tiou: float = 0.05,
) -> EventMatchResult:
    """Matches predictions against ground truth events using temporal proximity and overlap.

    Rules:
    - Same event_type and recording_id
    - Temporal overlap or prediction starts within [gt.start_time - tolerance, gt.end_time + tolerance]
    - First matching prediction gets the match (True Positive)
    - Additional predictions for an already-matched GT are flagged as duplicates (False Positives)
    """
    matched_pairs: List[Tuple[GroundTruthEvent, PredictedEvent, float]] = []
    unmatched_preds: List[PredictedEvent] = []
    duplicate_preds: List[PredictedEvent] = []
    matched_gt_indices = set()

    # Sort predictions chronologically
    sorted_preds = sorted(predictions, key=lambda p: p.start_time)

    for pred in sorted_preds:
        candidate_matches = []

        for idx, gt in enumerate(ground_truths):
            if gt.recording_id != pred.recording_id:
                continue
            if gt.event_type != pred.event_type:
                continue

            # Check overlap or timing tolerance
            tiou = compute_temporal_iou(gt.start_time, gt.end_time, pred.start_time, pred.end_time)
            is_within_window = (
                (gt.start_time - tolerance_sec) <= pred.start_time <= (gt.end_time + tolerance_sec) or
                (pred.start_time - tolerance_sec) <= gt.start_time <= (pred.end_time + tolerance_sec)
            )

            if tiou >= min_tiou or is_within_window:
                delay = pred.start_time - gt.start_time
                candidate_matches.append((idx, gt, abs(delay), delay))

        if not candidate_matches:
            unmatched_preds.append(pred)
        else:
            # Pick closest in time
            candidate_matches.sort(key=lambda x: x[2])
            best_idx, best_gt, _, delay = candidate_matches[0]

            if best_idx not in matched_gt_indices:
                matched_gt_indices.add(best_idx)
                matched_pairs.append((best_gt, pred, max(0.0, delay)))
            else:
                # Already matched this ground truth: duplicate prediction counts as FP
                duplicate_preds.append(pred)

    unmatched_gts = [gt for idx, gt in enumerate(ground_truths) if idx not in matched_gt_indices]

    return EventMatchResult(
        matched_pairs=matched_pairs,
        unmatched_ground_truths=unmatched_gts,
        unmatched_predictions=unmatched_preds,
        duplicate_predictions=duplicate_preds,
    )


def calculate_metrics(
    ground_truths: List[GroundTruthEvent],
    predictions: List[PredictedEvent],
    total_duration_sec: float,
    mode: str = "standard",
) -> EvaluationMetrics:
    """Calculates overall and subgroup evaluation metrics."""
    match_result = match_events(ground_truths, predictions)

    tp = len(match_result.matched_pairs)
    fp = len(match_result.unmatched_predictions) + len(match_result.duplicate_predictions)
    fn = len(match_result.unmatched_ground_truths)

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    dur_min = max(0.1, total_duration_sec / 60.0)
    fa_per_min = round(fp / dur_min, 2)

    delays = [delay for _, _, delay in match_result.matched_pairs]
    mean_delay = round(sum(delays) / len(delays), 2) if delays else 0.0

    # Category breakdown
    categories = sorted(list(set([gt.event_type for gt in ground_truths] + [p.event_type for p in predictions])))
    cat_metrics = {}
    for cat in categories:
        cat_gts = [g for g in ground_truths if g.event_type == cat]
        cat_preds = [p for p in predictions if p.event_type == cat]
        res = match_events(cat_gts, cat_preds)
        c_tp = len(res.matched_pairs)
        c_fp = len(res.unmatched_predictions) + len(res.duplicate_predictions)
        c_fn = len(res.unmatched_ground_truths)
        c_prec = round(c_tp / (c_tp + c_fp), 3) if (c_tp + c_fp) > 0 else 0.0
        c_rec = round(c_tp / (c_tp + c_fn), 3) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = round(2 * c_prec * c_rec / (c_prec + c_rec), 3) if (c_prec + c_rec) > 0 else 0.0
        cat_metrics[cat] = {
            "gt_count": len(cat_gts),
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": c_prec,
            "recall": c_rec,
            "f1": c_f1,
        }

    # Robustness subgroups (glasses yes/no, lighting normal/low_light)
    subgroups = {
        "glasses_yes": [g for g in ground_truths if g.glasses.lower() in ("yes", "true", "1")],
        "glasses_no": [g for g in ground_truths if g.glasses.lower() in ("no", "false", "0")],
        "lighting_normal": [g for g in ground_truths if "normal" in g.lighting.lower()],
        "lighting_low_light": [g for g in ground_truths if "low" in g.lighting.lower()],
    }
    robustness_metrics = {}
    for sub_name, sub_gts in subgroups.items():
        if not sub_gts:
            continue
        sub_preds = [p for p in predictions if any(p.recording_id == g.recording_id for g in sub_gts)]
        res = match_events(sub_gts, sub_preds)
        s_tp = len(res.matched_pairs)
        s_fp = len(res.unmatched_predictions) + len(res.duplicate_predictions)
        s_fn = len(res.unmatched_ground_truths)
        s_prec = round(s_tp / (s_tp + s_fp), 3) if (s_tp + s_fp) > 0 else 0.0
        s_rec = round(s_tp / (s_tp + s_fn), 3) if (s_tp + s_fn) > 0 else 0.0
        s_f1 = round(2 * s_prec * s_rec / (s_prec + s_rec), 3) if (s_prec + s_rec) > 0 else 0.0
        robustness_metrics[sub_name] = {
            "gt_count": len(sub_gts),
            "tp": s_tp,
            "fp": s_fp,
            "fn": s_fn,
            "precision": s_prec,
            "recall": s_rec,
            "f1": s_f1,
        }

    return EvaluationMetrics(
        mode=mode,
        total_evaluated_duration_sec=total_duration_sec,
        total_ground_truth_events=len(ground_truths),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_alarms_per_minute=fa_per_min,
        mean_detection_delay_sec=mean_delay,
        category_metrics=cat_metrics,
        robustness_metrics=robustness_metrics,
    )
