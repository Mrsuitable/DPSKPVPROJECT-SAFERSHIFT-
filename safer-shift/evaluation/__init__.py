"""SaferShift Evaluation Package.

Provides event matching, precision/recall/F1 calculation, false alarm rates,
and standard vs personalized comparative benchmarking across test scenarios.
"""

from .event_matching import (
    GroundTruthEvent,
    PredictedEvent,
    EventMatchResult,
    match_events,
    calculate_metrics,
)
from .evaluate import run_evaluation_benchmark

__all__ = [
    "GroundTruthEvent",
    "PredictedEvent",
    "EventMatchResult",
    "match_events",
    "calculate_metrics",
    "run_evaluation_benchmark",
]
