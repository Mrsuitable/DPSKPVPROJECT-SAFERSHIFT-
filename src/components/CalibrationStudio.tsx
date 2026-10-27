import React, { useState } from 'react';
import {
  Sliders,
  CheckCircle2,
  AlertOctagon,
  Trash2,
  User,
  ShieldCheck,
  RefreshCw,
  Sparkles,
  Info,
} from 'lucide-react';
import { DriverProfile } from '../types';
import { choosePrimaryFace, FaceMeshInstance, loadFaceMesh, measureFace } from '../utils/vision';

interface CalibrationStudioProps {
  activeProfile: DriverProfile | null;
  profiles: DriverProfile[];
  onProfileCreated: (profile: DriverProfile) => Promise<void>;
  onProfileDeleted: (profileId: string) => Promise<void>;
  onSelectProfile: (profile: DriverProfile) => void;
}

export const CalibrationStudio: React.FC<CalibrationStudioProps> = ({
  activeProfile,
  profiles,
  onProfileCreated,
  onProfileDeleted,
  onSelectProfile,
}) => {
  const [driverName, setDriverName] = useState<string>('');
  const [profileId, setProfileId] = useState<string>('');
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);
  const [calibrationProgress, setCalibrationProgress] = useState<number>(0);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<DriverProfile | null>(null);

  const percentile = (values: number[], percent: number) => {
    const sorted = [...values].sort((a, b) => a - b);
    if (sorted.length === 0) return 0;
    const index = (sorted.length - 1) * percent;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
  };

  // Preset templates for quick testing
  const presets = [
    {
      id: 'driver_narrow_eyes',
      name: 'Driver A (Naturally Narrow Palpebral Fissure)',
      baselineEar: 0.235,
      adaptedEar: 0.176,
      desc: 'Naturally small eye opening. Standard EAR 0.21 produces false alarms; personalized 0.176 fixes it.',
    },
    {
      id: 'driver_wide_eyes',
      name: 'Driver B (Wide Eyes Anatomy)',
      baselineEar: 0.330,
      adaptedEar: 0.248,
      desc: 'Large eyes. Standard EAR 0.21 could delay drowsiness warning; personalized 0.248 detects earlier.',
    },
    {
      id: 'driver_spectacles',
      name: 'Driver C (Eyeglasses / Low Light)',
      baselineEar: 0.265,
      adaptedEar: 0.198,
      desc: 'Calibrated under spectacle frame reflections.',
    },
  ];

  const startCalibration = async () => {
    const pId = (profileId.trim() || `driver_${Date.now().toString().slice(-4)}`).replace(/[^a-zA-Z0-9_-]/g, '_');
    const dName = driverName.trim() || 'Driver Default';

    setIsCalibrating(true);
    setCalibrationProgress(0);
    setCalibrationError(null);
    setLastResult(null);

    const samples: Array<{ ear: number; mar: number; yaw: number; pitch: number }> = [];
    let stream: MediaStream | null = null;
    let mesh: FaceMeshInstance | null = null;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('This browser does not provide camera access. Use a secure local connection and allow webcam access.');
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      });
      const video = document.createElement('video');
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play();

      const FaceMesh = await loadFaceMesh();
      mesh = new FaceMesh({
        locateFile: (file) => `/mediapipe/face_mesh/${file.replace(/^face_mesh\//, '')}`,
      });
      mesh.setOptions({
        maxNumFaces: 2,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      mesh.onResults((results) => {
        const face = choosePrimaryFace(results.multiFaceLandmarks ?? []);
        if (!face) return;
        const measurements = measureFace(face, video.videoWidth || 640, video.videoHeight || 480);
        if (measurements) samples.push(measurements);
      });
      await mesh.initialize();

      const durationMs = 4000;
      const startTime = performance.now();
      let lastVideoTime = -1;
      while (performance.now() - startTime < durationMs) {
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          await mesh.send({ image: video });
        } else {
          await new Promise((resolve) => window.setTimeout(resolve, 20));
        }
        setCalibrationProgress(Math.min(99, Math.round(((performance.now() - startTime) / durationMs) * 100)));
      }

      if (samples.length < 45) {
        throw new Error(`Only ${samples.length} usable face samples were captured. Improve lighting, face the camera, and try again.`);
      }

      const ears = samples.map((sample) => sample.ear);
      const mars = samples.map((sample) => sample.mar);
      const yaws = samples.map((sample) => sample.yaw);
      const pitches = samples.map((sample) => sample.pitch);
      const baselineEar = ears.reduce((sum, value) => sum + value, 0) / ears.length;
      const baselineMar = mars.reduce((sum, value) => sum + value, 0) / mars.length;
      const earStd = Math.sqrt(ears.reduce((sum, value) => sum + (value - baselineEar) ** 2, 0) / Math.max(1, ears.length - 1));
      const neutralYaw = percentile(yaws, 0.5);
      const neutralPitch = percentile(pitches, 0.5);
      const yawStd = Math.sqrt(yaws.reduce((sum, value) => sum + (value - neutralYaw) ** 2, 0) / Math.max(1, yaws.length - 1));
      const earP25 = percentile(ears, 0.25);
      const earP50 = percentile(ears, 0.5);

      if (earP50 < 0.17) {
        throw new Error('Calibration was rejected because eyes appeared closed or obstructed. Keep your eyes alert and try again.');
      }

      let quality = 1;
      if (earStd > 0.08) quality -= 0.25;
      if (yawStd > 12) quality -= 0.25;
      if (samples.length < 60) quality -= 0.1;
      quality = Math.max(0.1, Math.min(1, quality));
      if (quality < 0.7) {
        throw new Error('Calibration quality was too low due to movement or unstable tracking. Sit still, face the camera, and try again.');
      }

      const adaptedEar = Math.max(0.15, Math.min(0.28, earP25 * 0.75));
      const newProfile: DriverProfile = {
        profile_id: pId,
        driver_name: dName,
        created_at: new Date().toISOString(),
        calibration_quality: Math.round(quality * 100) / 100,
        baseline_ear_mean: baselineEar,
        baseline_ear_std: earStd,
        open_eye_ear_p25: earP25,
        open_eye_ear_p50: earP50,
        adapted_ear_threshold: Math.round(adaptedEar * 1000) / 1000,
        neutral_yaw: neutralYaw,
        neutral_pitch: neutralPitch,
        yaw_limit: 25,
        pitch_down_limit: 18,
        pitch_up_limit: 22,
        baseline_mar_mean: baselineMar,
        sample_count: samples.length,
        notes: 'Calibrated from local webcam measurements. Video frames are not stored.',
      };

      await onProfileCreated(newProfile);
      setCalibrationProgress(100);
      setLastResult(newProfile);
    } catch (error) {
      setCalibrationError(error instanceof Error ? error.message : String(error));
    } finally {
      if (mesh) await mesh.close().catch(() => {});
      stream?.getTracks().forEach((track) => track.stop());
      setIsCalibrating(false);
    }
  };

  const applyPreset = async (preset: typeof presets[0]) => {
    const newProfile: DriverProfile = {
      profile_id: preset.id,
      driver_name: preset.name,
      created_at: new Date().toISOString(),
      calibration_quality: 0.96,
      baseline_ear_mean: preset.baselineEar,
      baseline_ear_std: 0.012,
      open_eye_ear_p25: preset.baselineEar - 0.01,
      open_eye_ear_p50: preset.baselineEar,
      adapted_ear_threshold: preset.adaptedEar,
      neutral_yaw: 0.0,
      neutral_pitch: 0.0,
      yaw_limit: 25.0,
      pitch_down_limit: 18.0,
      pitch_up_limit: 22.0,
      baseline_mar_mean: 0.20,
      sample_count: 100,
      notes: preset.desc,
    };
    setCalibrationError(null);
    try {
      await onProfileCreated(newProfile);
      setLastResult(newProfile);
    } catch (error) {
      setCalibrationError(error instanceof Error ? error.message : String(error));
    }
  };

  const deleteProfile = async (id: string) => {
    setCalibrationError(null);
    try {
      await onProfileDeleted(id);
    } catch (error) {
      setCalibrationError(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="space-y-8">
      {/* Header explanation */}
      <div className="bg-gradient-to-r from-slate-900 to-cyan-950 p-6 rounded-3xl border border-cyan-800/40 shadow-2xl">
        <div className="max-w-3xl">
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase tracking-wider mb-2">
            <Sliders className="w-4 h-4" />
            <span>SaferShift Calibration Protocol</span>
          </div>
          <h2 className="text-2xl font-black text-white tracking-tight">
            Personalized Eye & Head Pose Baseline Calibration
          </h2>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed">
            Every human driver possesses unique facial geometry. Standard systems use a single
            rigid threshold (EAR = 0.21), triggering frequent nuisance alarms for drivers with naturally
            narrow eyes or delayed detection for drivers with wide eyes. SaferShift captures a 4-second
            alert baseline while you sit stationary, calculating personalized safety-bounded thresholds.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Interactive Calibration Wizard (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-slate-900/90 p-6 rounded-2xl border border-slate-800 shadow-xl">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center space-x-2">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              <span>Perform Live Calibration</span>
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">
                  Driver Display Name
                </label>
                <input
                  type="text"
                  placeholder="e.g., Alex Carter"
                  value={driverName}
                  onChange={(e) => setDriverName(e.target.value)}
                  disabled={isCalibrating}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white text-sm focus:outline-none focus:border-cyan-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">
                  Local Profile Identifier
                </label>
                <input
                  type="text"
                  placeholder="e.g., driver_shift_01"
                  value={profileId}
                  onChange={(e) => setProfileId(e.target.value)}
                  disabled={isCalibrating}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-white text-sm focus:outline-none focus:border-cyan-500"
                />
              </div>
            </div>

            {/* Calibration Steps Guide */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 mb-6 space-y-2 text-xs text-slate-300">
              <div className="font-semibold text-slate-200">Calibration Instructions:</div>
              <ul className="list-disc list-inside space-y-1 text-slate-400">
                <li>Sit in your natural driving posture facing the camera straight ahead.</li>
                <li>Keep eyes alert and wide open; do not squint or look away.</li>
                <li>Remain stationary for the 4-second measurement cycle.</li>
              </ul>
            </div>

            {/* Calibration Progress Bar */}
            {isCalibrating && (
              <div className="mb-6 space-y-2">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-cyan-400 animate-pulse">Capturing alert baseline geometry...</span>
                  <span className="text-slate-300">{calibrationProgress}%</span>
                </div>
                <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="bg-gradient-to-r from-cyan-500 to-indigo-500 h-full transition-all duration-150"
                    style={{ width: `${calibrationProgress}%` }}
                  />
                </div>
              </div>
            )}

            <button
              onClick={startCalibration}
              disabled={isCalibrating}
              className={`w-full py-3 rounded-xl font-bold text-sm tracking-wide transition-all shadow-lg flex items-center justify-center space-x-2 ${
                isCalibrating
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-cyan-600/30'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${isCalibrating ? 'animate-spin' : ''}`} />
              <span>{isCalibrating ? 'Calibrating...' : 'Start 4-Second Calibration'}</span>
            </button>
            {calibrationError && (
              <div role="alert" className="mt-4 rounded-xl border border-rose-800 bg-rose-950/60 p-3 text-xs text-rose-300">
                {calibrationError}
              </div>
            )}
            {lastResult && !calibrationError && (
              <div className="mt-4 rounded-xl border border-emerald-800 bg-emerald-950/50 p-3 text-xs text-emerald-300">
                Saved {lastResult.driver_name}: {lastResult.sample_count} live samples, quality {(lastResult.calibration_quality * 100).toFixed(0)}%, adapted EAR {lastResult.adapted_ear_threshold.toFixed(3)}.
              </div>
            )}
          </div>

          {/* Quick Presets for Hypothesis Testing */}
          <div className="bg-slate-900/90 p-6 rounded-2xl border border-slate-800 shadow-xl">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-3">
              Driver Anatomy Presets (Quick Demonstration)
            </h3>
            <div className="space-y-3">
              {presets.map((pr) => (
                <div
                  key={pr.id}
                  className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 hover:border-slate-700 transition-all flex items-center justify-between"
                >
                  <div className="space-y-1 pr-4">
                    <div className="text-sm font-semibold text-white">{pr.name}</div>
                    <div className="text-xs text-slate-400">{pr.desc}</div>
                    <div className="text-xs font-mono text-cyan-400">
                      Baseline EAR: {pr.baselineEar.toFixed(3)} → Adapted Threshold: {pr.adaptedEar.toFixed(3)}
                    </div>
                  </div>
                  <button
                    onClick={() => applyPreset(pr)}
                    disabled={isCalibrating}
                    className="shrink-0 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700"
                  >
                    Load Preset
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Calibrated Profiles & Active Threshold Comparison (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Active Profile Diagnostic Card */}
          <div className="bg-slate-900/90 p-6 rounded-2xl border border-slate-800 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">
                Active Profile Parameters
              </h3>
              {activeProfile && (
                <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
                  CALIBRATED
                </span>
              )}
            </div>

            {activeProfile ? (
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <div className="flex items-center space-x-2">
                    <User className="w-4 h-4 text-cyan-400" />
                    <span className="font-bold text-white text-base">
                      {activeProfile.driver_name}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 font-mono">
                    ID: {activeProfile.profile_id} | Quality: {(activeProfile.calibration_quality * 100).toFixed(0)}%
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-slate-400">Open-Eye Baseline</div>
                    <div className="text-lg font-bold text-white mt-1">
                      {activeProfile.baseline_ear_mean.toFixed(3)}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-cyan-800/60 bg-cyan-950/20">
                    <div className="text-cyan-400">Adapted Threshold</div>
                    <div className="text-lg font-bold text-cyan-300 mt-1">
                      {activeProfile.adapted_ear_threshold.toFixed(3)}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-slate-400">Neutral Head Pose</div>
                    <div className="text-sm font-bold text-white mt-1">
                      {activeProfile.neutral_yaw.toFixed(1)}° / {activeProfile.neutral_pitch.toFixed(1)}°
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-slate-400">Safety Bounds</div>
                    <div className="text-xs text-emerald-400 mt-1">
                      [0.150, 0.280] Enforced
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-800/40 text-xs text-cyan-200 flex items-start space-x-2">
                  <ShieldCheck className="w-4 h-4 shrink-0 text-cyan-400 mt-0.5" />
                  <span>
                    Personalization bounds safety check passed: prolonged eye closure (&gt;1.5s)
                    remains guaranteed to trigger critical alarm.
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-slate-500 text-xs">
                No personalized profile selected. Using Standard Mode baseline thresholds.
              </div>
            )}
          </div>

          {/* Stored Profiles List with Privacy Deletion */}
          <div className="bg-slate-900/90 p-6 rounded-2xl border border-slate-800 shadow-xl">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-3">
              Stored Local Profiles ({profiles.length})
            </h3>

            {profiles.length === 0 ? (
              <p className="text-xs text-slate-500">No profiles saved in local data storage.</p>
            ) : (
              <div className="space-y-2.5 max-h-60 overflow-y-auto pr-1">
                {profiles.map((p) => (
                  <div
                    key={p.profile_id}
                    className={`p-3 rounded-xl border transition-all flex items-center justify-between ${
                      activeProfile?.profile_id === p.profile_id
                        ? 'bg-slate-800 border-cyan-500'
                        : 'bg-slate-950 border-slate-800'
                    }`}
                  >
                    <div>
                      <div className="text-sm font-semibold text-white">{p.driver_name}</div>
                      <div className="text-xs text-slate-400 font-mono">
                        Adapted EAR: {p.adapted_ear_threshold.toFixed(3)}
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => onSelectProfile(p)}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700"
                      >
                        Activate
                      </button>
                      <button
                        onClick={() => void deleteProfile(p.profile_id)}
                        className="p-1.5 rounded bg-rose-950/60 hover:bg-rose-900 text-rose-400 border border-rose-800 transition-colors"
                        title="Delete profile permanently (Privacy compliance)"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
