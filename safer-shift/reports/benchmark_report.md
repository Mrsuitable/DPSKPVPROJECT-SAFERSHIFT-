# SaferShift: Standard vs Personalized Detection Benchmark Report

**Generated:** 2026-10-22  
**Total Evaluated Duration:** 128.2 seconds across 5 test scenarios  
**Total Ground Truth Events:** 11  

---

## Executive Summary & Research Question Findings

> **Core Research Question:** *Can personalized calibration reduce false alarms while preserving the detection of genuine fatigue indicators?*

### Experimental Findings:
- **False Alarm Reduction:** Personalized calibration reduced false alarms from **1** to **0** (a **100.0% decrease**).
- **Fatigue Recall Preservation:** Recall for genuine prolonged eye closures remained at **100%** (both modes successfully detected all genuine critical events).
- **Overall F1 Score:** Improved from **0.957** (Standard) to **1.000** (Personalized).

---

## 1. Overall Performance Comparison

| Metric | Standard Mode (Baseline) | Personalized Mode (Calibrated) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 11 | 11 | +0 |
| **False Positives (FP)** | 1 | 0 | -1 (+100.0%) |
| **False Negatives (FN)** | 0 | 0 | +0 |
| **Precision** | **0.917** | **1.000** | **+0.083** |
| **Recall** | **1.000** | **1.000** | **+0.000** |
| **F1-Score** | **0.957** | **1.000** | **+0.043** |
| **False Alarms / Min** | **0.47** | **0.00** | **-0.47** |
| **Mean Detection Delay** | 0.00s | 0.00s | +0.00s |

---

## 2. Breakdown by Event Category

### Standard Mode
```json
{
  "possible_yawning": {
    "gt_count": 3,
    "tp": 3,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "prolonged_eye_closure": {
    "gt_count": 5,
    "tp": 5,
    "fp": 1,
    "fn": 0,
    "precision": 0.833,
    "recall": 1.0,
    "f1": 0.909
  },
  "sustained_distraction": {
    "gt_count": 3,
    "tp": 3,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  }
}
```

### Personalized Mode
```json
{
  "possible_yawning": {
    "gt_count": 3,
    "tp": 3,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "prolonged_eye_closure": {
    "gt_count": 5,
    "tp": 5,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "sustained_distraction": {
    "gt_count": 3,
    "tp": 3,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  }
}
```

---

## 3. Robustness Checks (Glasses & Lighting)

| Subgroup | Condition | Standard F1 | Personalized F1 | Standard FP | Personalized FP |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Glasses | Spectacles Worn | 1.000 | 1.000 | 0 | 0 |
| No Glasses | Bare Face | 0.933 | 1.000 | 1 | 0 |
| Lighting | Normal Daylight | 0.933 | 1.000 | 1 | 0 |
| Lighting | Low Light / Glare | 1.000 | 1.000 | 0 | 0 |

---

## 4. Honest Methodology and Data Limitations
1. **Sample Size:** Tested across 5 distinct scenarios with 11 annotated fatigue/distraction events.
2. **Personalization Bounds:** Hard safety boundaries [0.15, 0.28] prevented adaptive thresholds from disabling prolonged eye closure alerts.
3. **Hypothesis Evaluation:** Supports the hypothesis that personalized baseline calibration curbs nuisance alarms caused by anatomical variance (e.g. naturally narrow palpebral fissures) while maintaining high safety sensitivity for true prolonged eye closures (>1.5s).
