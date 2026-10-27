export type DetectionMode = 'standard' | 'personalized';

export type AlertSeverity = 'critical' | 'warning' | 'advisory' | 'info';

export type AlertCategory =
  | 'prolonged_eye_closure'
  | 'sustained_distraction'
  | 'possible_yawning'
  | 'elevated_blink_rate'
  | 'tracking_unreliable'
  | 'normal';

export interface AlertEvent {
  id: string;
  category: AlertCategory;
  severity: AlertSeverity;
  timestamp: number;
  duration: number;
  mode: DetectionMode;
  triggerMetric: string;
  triggerValue: number;
  thresholdValue: number;
  explanation: string;
}

export interface DriverProfile {
  profile_id: string;
  driver_name: string;
  created_at: string;
  calibration_quality: number;
  baseline_ear_mean: number;
  baseline_ear_std: number;
  open_eye_ear_p25: number;
  open_eye_ear_p50: number;
  adapted_ear_threshold: number;
  neutral_yaw: number;
  neutral_pitch: number;
  yaw_limit: number;
  pitch_down_limit: number;
  pitch_up_limit: number;
  baseline_mar_mean: number;
  sample_count: number;
  notes?: string;
}

export interface TelemetryPoint {
  timestamp: number;
  ear: number;
  mar: number;
  yaw: number;
  pitch: number;
  roll: number;
  closureDuration: number;
  distractionDuration: number;
  yawnDuration: number;
  blinkRateBpm: number;
  totalBlinks: number;
  fps: number;
  eventType: string;
  warningReason?: string;
}

export interface BenchmarkMetrics {
  mode: string;
  total_evaluated_duration_sec: number;
  total_ground_truth_events: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  f1_score: number;
  false_alarms_per_minute: number;
  mean_detection_delay_sec: number;
  category_metrics?: Record<string, any>;
  robustness_metrics?: Record<string, any>;
}

export interface BenchmarkReport {
  standard: BenchmarkMetrics;
  personalized: BenchmarkMetrics;
  false_alarm_reduction_pct: number;
  total_evaluated_duration_sec: number;
  ground_truth_count: number;
}
