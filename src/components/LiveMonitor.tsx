import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  Play,
  Pause,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Compass,
  Eye,
  Activity,
  Info,
  Maximize2,
  ShieldAlert,
} from 'lucide-react';
import { DetectionMode, DriverProfile, AlertEvent, TelemetryPoint } from '../types';
import { soundManager } from '../utils/audio';
import {
  choosePrimaryFace,
  FACE_MESH_EYES,
  FACE_MESH_MOUTH,
  FaceMeshInstance,
  FacePoint,
  loadFaceMesh,
  measureFace,
} from '../utils/vision';

interface LiveMonitorProps {
  mode: DetectionMode;
  activeProfile: DriverProfile | null;
  isMuted: boolean;
  onTelemetryUpdate: (point: TelemetryPoint) => void;
  onAlertTriggered: (alert: AlertEvent) => void;
}

type ScenarioType = 'camera' | 'normal' | 'microsleep' | 'distraction' | 'yawn' | 'narrow_eyes';

export const LiveMonitor: React.FC<LiveMonitorProps> = ({
  mode,
  activeProfile,
  isMuted,
  onTelemetryUpdate,
  onAlertTriggered,
}) => {
  const [sourceType, setSourceType] = useState<ScenarioType>('normal');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraRetry, setCameraRetry] = useState<number>(0);

  // Runtime measurements
  const [ear, setEar] = useState<number>(0.28);
  const [mar, setMar] = useState<number>(0.18);
  const [yaw, setYaw] = useState<number>(0);
  const [pitch, setPitch] = useState<number>(0);
  const [roll, setRoll] = useState<number>(0);
  const [closureDuration, setClosureDuration] = useState<number>(0);
  const [distractionDuration, setDistractionDuration] = useState<number>(0);
  const [yawnDuration, setYawnDuration] = useState<number>(0);
  const [fps, setFps] = useState<number>(30.0);
  const [bpm, setBpm] = useState<number>(18.0);
  const [totalBlinks, setTotalBlinks] = useState<number>(14);
  const [isFaceDetected, setIsFaceDetected] = useState<boolean>(true);

  // Active Alert
  const [activeAlert, setActiveAlert] = useState<AlertEvent | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceMeshRef = useRef<FaceMeshInstance | null>(null);
  const latestFaceRef = useRef<FacePoint[] | null>(null);
  const latestMeasurementsRef = useRef<ReturnType<typeof measureFace>>(null);
  const inferenceInFlightRef = useRef(false);
  const lastInferenceTimeRef = useRef(0);
  const modelFrameTimesRef = useRef<number[]>([]);
  const modelFpsRef = useRef(0);
  const activeAlertRef = useRef<AlertEvent | null>(null);
  const onTelemetryUpdateRef = useRef(onTelemetryUpdate);
  const onAlertTriggeredRef = useRef(onAlertTriggered);
  onTelemetryUpdateRef.current = onTelemetryUpdate;
  onAlertTriggeredRef.current = onAlertTriggered;

  // Active thresholds based on mode
  const currentEarThreshold =
    mode === 'personalized' && activeProfile
      ? activeProfile.adapted_ear_threshold
      : 0.21;
  const currentNeutralYaw =
    mode === 'personalized' && activeProfile ? activeProfile.neutral_yaw : 0.0;
  const currentNeutralPitch =
    mode === 'personalized' && activeProfile ? activeProfile.neutral_pitch : 0.0;
  const currentYawLimit =
    mode === 'personalized' && activeProfile ? activeProfile.yaw_limit : 25.0;
  const currentPitchDownLimit =
    mode === 'personalized' && activeProfile ? activeProfile.pitch_down_limit : 18.0;
  const currentPitchUpLimit =
    mode === 'personalized' && activeProfile ? activeProfile.pitch_up_limit : 22.0;

  useEffect(() => {
    soundManager.setMuted(isMuted);
  }, [isMuted]);

  // Acquire and release the camera only while webcam mode is selected.
  useEffect(() => {
    let cancelled = false;
    latestFaceRef.current = null;
    latestMeasurementsRef.current = null;
    setCameraError(null);

    if (sourceType !== 'camera') {
      stopCamera();
      setIsFaceDetected(true);
      return () => { cancelled = true; };
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('This browser does not provide camera access. Use a secure local connection and allow webcam access.');
      setCameraActive(false);
      return () => { cancelled = true; };
    }

    setCameraActive(false);
    setIsFaceDetected(false);
    modelFrameTimesRef.current = [];
    modelFpsRef.current = 0;
    setFps(0);
    void (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) throw new Error('Camera preview is unavailable.');
        video.srcObject = stream;
        await video.play();
        if (!cancelled) setCameraActive(true);
      } catch (error) {
        if (cancelled) return;
        const detail = error instanceof Error ? error.message : 'Camera access is unavailable.';
        setCameraError(`${detail} Allow webcam access and try again.`);
        setCameraActive(false);
      }
    })();

    return () => {
      cancelled = true;
      stopCamera();
    };
  }, [sourceType, cameraRetry]);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  };

  // Load the bundled model and run all face inference locally in this browser.
  useEffect(() => {
    if (!cameraActive) return;
    let cancelled = false;
    let mesh: FaceMeshInstance | null = null;

    void (async () => {
      try {
        const FaceMesh = await loadFaceMesh();
        if (cancelled) return;
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
          if (cancelled) return;
          const now = performance.now();
          modelFrameTimesRef.current = modelFrameTimesRef.current
            .filter((frameTime) => now - frameTime <= 1000);
          modelFrameTimesRef.current.push(now);
          modelFpsRef.current = modelFrameTimesRef.current.length > 1
            ? (modelFrameTimesRef.current.length - 1) * 1000 /
              (now - modelFrameTimesRef.current[0])
            : 0;
          setFps(Math.round(modelFpsRef.current * 10) / 10);
          const face = choosePrimaryFace(results.multiFaceLandmarks ?? []);
          const video = videoRef.current;
          latestFaceRef.current = face;
          latestMeasurementsRef.current = face && video
            ? measureFace(face, video.videoWidth || 640, video.videoHeight || 480)
            : null;
          setIsFaceDetected(Boolean(latestMeasurementsRef.current));
        });
        await mesh.initialize();
        if (cancelled) {
          void mesh.close();
          return;
        }
        faceMeshRef.current = mesh;
      } catch (error) {
        if (cancelled) return;
        const detail = error instanceof Error ? error.message : String(error);
        setCameraError(`Face tracking could not start: ${detail}`);
      }
    })();

    return () => {
      cancelled = true;
      faceMeshRef.current = null;
      latestFaceRef.current = null;
      latestMeasurementsRef.current = null;
      if (mesh) void mesh.close();
    };
  }, [cameraActive]);

  // Run scenario demos or consume the latest locally detected camera frame.
  useEffect(() => {
    if (!isPlaying) return;

    let animFrame: number;
    let lastTime = performance.now();
    let frameTimes: number[] = [];
    let scenarioTimer = 0;
    let closureStartTime: number | null = null;
    let distractionStartTime: number | null = null;
    let yawnStartTime: number | null = null;
    let blinkStartTime: number | null = null;
    let eyesWereClosed = false;
    let totalBlinkCount = 0;
    let blinkTimes: number[] = [];
    let closureAlerted = false;
    let distractionAlerted = false;
    let yawnAlerted = false;
    let lastAlertTime = 0;
    let lastUiUpdate = 0;
    let lastTelemetryUpdate = 0;
    let lastFpsUpdate = 0;

    const loop = (currentTime: number) => {
      const deltaSec = Math.min(0.25, Math.max(0, (currentTime - lastTime) / 1000.0));
      lastTime = currentTime;
      if (sourceType !== 'camera') scenarioTimer += deltaSec;

      if (sourceType === 'camera') {
        const video = videoRef.current;
        const mesh = faceMeshRef.current;
        if (
          cameraActive && video && mesh && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          currentTime - lastInferenceTimeRef.current >= 80 && !inferenceInFlightRef.current
        ) {
          lastInferenceTimeRef.current = currentTime;
          inferenceInFlightRef.current = true;
          void mesh.send({ image: video }).catch((error: unknown) => {
            const detail = error instanceof Error ? error.message : String(error);
            setCameraError(`Camera frame processing failed: ${detail}`);
          }).finally(() => {
            inferenceInFlightRef.current = false;
          });
        }
      }

      // Calculate running FPS
      frameTimes.push(currentTime);
      while (frameTimes.length > 0 && frameTimes[0] < currentTime - 1000) {
        frameTimes.shift();
      }
      const currentFps = frameTimes.length > 1 ? frameTimes.length : 30;
      if (sourceType !== 'camera' && currentTime - lastFpsUpdate >= 250) {
        setFps(Math.round(currentFps * 10) / 10);
        lastFpsUpdate = currentTime;
      }
      const measuredFps = sourceType === 'camera' ? modelFpsRef.current : currentFps;

      let curEar = 0.28;
      let curMar = 0.20;
      let curYaw = 0.0;
      let curPitch = -1.0;
      let curRoll = 0.0;
      let faceFound = sourceType !== 'camera';

      // Scenario Behaviors
      if (sourceType === 'camera') {
        const measurements = latestMeasurementsRef.current;
        faceFound = Boolean(measurements);
        if (measurements) {
          curEar = measurements.ear;
          curMar = measurements.mar;
          curYaw = measurements.yaw;
          curPitch = measurements.pitch;
          curRoll = measurements.roll;
        }
      } else if (sourceType === 'normal') {
        // Natural micro-variations and periodic blinks
        curEar = 0.28 + 0.015 * Math.sin(scenarioTimer * 2.0);
        curMar = 0.18 + 0.01 * Math.cos(scenarioTimer * 1.5);
        curYaw = 2.0 * Math.sin(scenarioTimer * 0.8);
        curPitch = -2.0 + 1.5 * Math.cos(scenarioTimer * 0.5);

        // Natural blink every 3.5 seconds for 0.15s
        if (scenarioTimer % 3.5 < 0.15) {
          curEar = 0.08;
        }
      } else if (sourceType === 'microsleep') {
        // Driver eyes close starting at t=2.0s and remain closed for 3.0s (> 1.5s rule)
        const cycle = scenarioTimer % 8.0;
        if (cycle >= 2.0 && cycle <= 5.2) {
          curEar = 0.06; // Fully closed
          curPitch = -6.0; // slight head drop
        } else {
          curEar = 0.27;
        }
      } else if (sourceType === 'distraction') {
        // Driver looks down and to the right at smartphone starting at t=1.5s for 3.5s (>2.0s rule)
        const cycle = scenarioTimer % 8.0;
        if (cycle >= 1.5 && cycle <= 5.5) {
          curYaw = 34.0; // Turned right
          curPitch = -22.0; // Look down at phone
          curEar = 0.25;
        } else {
          curYaw = 1.0;
          curPitch = -1.0;
          curEar = 0.28;
        }
      } else if (sourceType === 'yawn') {
        // Driver yawns starting at t=1.5s for 3.0s (> 2.2s rule)
        const cycle = scenarioTimer % 8.0;
        if (cycle >= 1.5 && cycle <= 4.8) {
          curMar = 0.68; // Wide mouth
          curEar = 0.28;
        } else {
          curMar = 0.19;
          curEar = 0.28;
        }
      } else if (sourceType === 'narrow_eyes') {
        // Natural narrow eyes / squinting driver (alert EAR ~0.23, dips to 0.195 in daylight)
        // Tests the core research hypothesis:
        // In STANDARD MODE (threshold 0.21), 0.195 causes a FALSE PROLONGED CLOSURE ALARM!
        // In PERSONALIZED MODE (calibrated threshold 0.175), it correctly recognizes driver is ALERT!
        const cycle = scenarioTimer % 8.0;
        if (cycle >= 1.5 && cycle <= 4.5) {
          curEar = 0.192; // Normal squint in daylight
        } else {
          curEar = 0.235;
        }
      }

      // 1. Evaluate Eye Closure
      let cDur = 0;
      const isClosed = faceFound && curEar < currentEarThreshold;
      if (isClosed) {
        if (closureStartTime === null) closureStartTime = currentTime;
        cDur = (currentTime - closureStartTime) / 1000.0;
      } else {
        closureStartTime = null;
        closureAlerted = false;
      }
      if (faceFound && isClosed && !eyesWereClosed) blinkStartTime = currentTime;
      if (faceFound && !isClosed && eyesWereClosed && blinkStartTime !== null) {
        const blinkDuration = (currentTime - blinkStartTime) / 1000;
        if (blinkDuration >= 0.05 && blinkDuration <= 0.7) {
          totalBlinkCount += 1;
          blinkTimes.push(currentTime);
        }
        blinkStartTime = null;
      }
      if (!faceFound) blinkStartTime = null;
      eyesWereClosed = faceFound && isClosed;
      blinkTimes = blinkTimes.filter((time) => currentTime - time <= 60_000);
      const blinkRateBpm = blinkTimes.length;

      // 2. Evaluate Distraction
      let dDur = 0;
      const deltaYaw = Math.abs(curYaw - currentNeutralYaw);
      const pitchDeviation = curPitch - currentNeutralPitch;
      const isDistracted = faceFound && (
        deltaYaw >= currentYawLimit ||
        pitchDeviation <= -currentPitchDownLimit ||
        pitchDeviation >= currentPitchUpLimit
      );
      if (isDistracted) {
        if (distractionStartTime === null) distractionStartTime = currentTime;
        dDur = (currentTime - distractionStartTime) / 1000.0;
      } else {
        distractionStartTime = null;
        distractionAlerted = false;
      }

      // 3. Evaluate Yawning
      let yDur = 0;
      const isYawning = faceFound && curMar >= 0.55;
      if (isYawning) {
        if (yawnStartTime === null) yawnStartTime = currentTime;
        yDur = (currentTime - yawnStartTime) / 1000.0;
      } else {
        yawnStartTime = null;
        yawnAlerted = false;
      }

      // Check Alerts
      let triggeredAlert: AlertEvent | null = null;
      const cooldownElapsed = currentTime - lastAlertTime > 3000;
      const alertTimestamp = Date.now();

      if (cDur >= 1.5 && !closureAlerted && cooldownElapsed) {
        triggeredAlert = {
          id: `alert_${alertTimestamp}`,
          category: 'prolonged_eye_closure',
          severity: 'critical',
          timestamp: alertTimestamp,
          duration: cDur,
          mode,
          triggerMetric: 'ear_duration',
          triggerValue: cDur,
          thresholdValue: 1.5,
          explanation: `Eyes continuously closed for ${cDur.toFixed(2)}s (EAR: ${curEar.toFixed(3)} < ${currentEarThreshold.toFixed(3)}). Exceeds safety threshold of 1.5s!`,
        };
        closureAlerted = true;
      } else if (dDur >= 2.0 && !distractionAlerted && cooldownElapsed) {
        triggeredAlert = {
          id: `alert_${alertTimestamp}`,
          category: 'sustained_distraction',
          severity: 'warning',
          timestamp: alertTimestamp,
          duration: dDur,
          mode,
          triggerMetric: 'head_pose_deviation',
          triggerValue: dDur,
          thresholdValue: 2.0,
          explanation: `Head pose remained outside its calibrated range for ${dDur.toFixed(1)}s (Yaw: ${curYaw.toFixed(1)}°, Pitch: ${curPitch.toFixed(1)}°).`,
        };
        distractionAlerted = true;
      } else if (yDur >= 2.2 && !yawnAlerted && cooldownElapsed) {
        triggeredAlert = {
          id: `alert_${alertTimestamp}`,
          category: 'possible_yawning',
          severity: 'warning',
          timestamp: alertTimestamp,
          duration: yDur,
          mode,
          triggerMetric: 'mar_duration',
          triggerValue: curMar,
          thresholdValue: 0.55,
          explanation: `Sustained wide mouth opening for ${yDur.toFixed(1)}s (MAR: ${curMar.toFixed(3)} >= 0.55). Possible yawning episode.`,
        };
        yawnAlerted = true;
      }

      if (triggeredAlert) {
        setActiveAlert(triggeredAlert);
        activeAlertRef.current = triggeredAlert;
        soundManager.playAlert(triggeredAlert.severity);
        lastAlertTime = currentTime;
        onAlertTriggeredRef.current(triggeredAlert);
      } else if (cDur === 0 && dDur === 0 && yDur === 0 && activeAlertRef.current) {
        // Clear active alert after recovery
        if (currentTime - lastAlertTime > 2500) {
          setActiveAlert(null);
          activeAlertRef.current = null;
        }
      }

      // Keep React updates at 10 Hz; detection and alert timing continue at camera speed.
      if (currentTime - lastUiUpdate >= 100 || triggeredAlert) {
        setEar(curEar);
        setMar(curMar);
        setYaw(curYaw);
        setPitch(curPitch);
        setRoll(curRoll);
        setClosureDuration(cDur);
        setDistractionDuration(dDur);
        setYawnDuration(yDur);
        setBpm(blinkRateBpm);
        setTotalBlinks(totalBlinkCount);
        setIsFaceDetected(faceFound);
        lastUiUpdate = currentTime;
      }

      // Emit telemetry point
      const point: TelemetryPoint = {
        timestamp: Date.now(),
        ear: curEar,
        mar: curMar,
        yaw: curYaw,
        pitch: curPitch,
        roll: curRoll,
        closureDuration: cDur,
        distractionDuration: dDur,
        yawnDuration: yDur,
        blinkRateBpm,
        totalBlinks: totalBlinkCount,
        fps: measuredFps,
        eventType: triggeredAlert ? triggeredAlert.category : 'sample',
        warningReason: triggeredAlert ? triggeredAlert.explanation : undefined,
      };
      if (currentTime - lastTelemetryUpdate >= 200 || triggeredAlert) {
        onTelemetryUpdateRef.current(point);
        lastTelemetryUpdate = currentTime;
      }

      drawCanvas(curEar, curMar, curYaw, curPitch, triggeredAlert !== null, isDistracted);

      animFrame = requestAnimationFrame(loop);
    };

    animFrame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animFrame);
  }, [
    isPlaying,
    sourceType,
    cameraActive,
    mode,
    currentEarThreshold,
    currentNeutralYaw,
    currentNeutralPitch,
    currentYawLimit,
    currentPitchDownLimit,
    currentPitchUpLimit,
  ]);

  // Canvas drawing for realistic landmark overlay
  const drawCanvas = (
    cEar: number,
    cMar: number,
    cYaw: number,
    cPitch: number,
    isAlertActive: boolean,
    isDistractedNow: boolean,
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (sourceType === 'camera') {
      const face = latestFaceRef.current;
      if (!face) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(cameraActive ? 'Looking for a face…' : 'Camera is not active', w / 2, h / 2);
        return;
      }

      const xs = face.map((point) => point.x * w);
      const ys = face.map((point) => point.y * h);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      ctx.beginPath();
      ctx.ellipse((minX + maxX) / 2, (minY + maxY) / 2, (maxX - minX) / 2, (maxY - minY) / 2, 0, 0, 2 * Math.PI);
      ctx.strokeStyle = isAlertActive ? 'rgba(239, 68, 68, 0.9)' : 'rgba(56, 189, 248, 0.7)';
      ctx.lineWidth = 2;
      ctx.stroke();

      const drawLandmarkPath = (indices: number[], color: string) => {
        ctx.beginPath();
        indices.forEach((index, pointIndex) => {
          const point = face[index];
          if (!point) return;
          const x = point.x * w;
          const y = point.y * h;
          if (pointIndex === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      };
      drawLandmarkPath(FACE_MESH_EYES.slice(0, 6), ear < currentEarThreshold ? '#ef4444' : '#38bdf8');
      drawLandmarkPath(FACE_MESH_EYES.slice(6), ear < currentEarThreshold ? '#ef4444' : '#38bdf8');
      drawLandmarkPath(FACE_MESH_MOUTH, mar >= 0.55 ? '#f59e0b' : '#38bdf8');

      const nose = face[1];
      if (nose) {
        const centerX = nose.x * w;
        const centerY = nose.y * h;
        const vectorX = centerX + (cYaw / 45) * 80;
        const vectorY = centerY + (cPitch / 40) * 80;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(vectorX, vectorY);
        ctx.strokeStyle = isDistractedNow ? '#ef4444' : '#10b981';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      return;
    }

    // Draw the synthetic driver cockpit preview.
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#0f172a');
    grad.addColorStop(1, '#020617');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Draw subtle vehicle cockpit windshield boundary
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(30, 20, w - 60, h - 40);

    // Face mesh wireframe centers
    const centerX = w / 2 + (cYaw / 45.0) * (w * 0.15);
    const centerY = h / 2 + (cPitch / 40.0) * (h * 0.12);
    const faceW = w * 0.28;
    const faceH = h * 0.44;

    // Face contour
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, faceW * 0.5, faceH * 0.5, 0, 0, 2 * Math.PI);
    ctx.strokeStyle = isAlertActive ? 'rgba(239, 68, 68, 0.8)' : 'rgba(56, 189, 248, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Eyes
    const eyeOffsetX = faceW * 0.22;
    const eyeOffsetY = -faceH * 0.12;
    const eyeW = faceW * 0.18;
    const eyeH = Math.max(3, eyeW * cEar * 2.2);

    // Left Eye
    ctx.beginPath();
    ctx.ellipse(centerX - eyeOffsetX, centerY + eyeOffsetY, eyeW * 0.5, eyeH * 0.5, 0, 0, 2 * Math.PI);
    ctx.strokeStyle = cEar < currentEarThreshold ? '#ef4444' : '#38bdf8';
    ctx.fillStyle = cEar < currentEarThreshold ? 'rgba(239, 68, 68, 0.3)' : 'rgba(56, 189, 248, 0.2)';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();

    // Right Eye
    ctx.beginPath();
    ctx.ellipse(centerX + eyeOffsetX, centerY + eyeOffsetY, eyeW * 0.5, eyeH * 0.5, 0, 0, 2 * Math.PI);
    ctx.strokeStyle = cEar < currentEarThreshold ? '#ef4444' : '#38bdf8';
    ctx.fillStyle = cEar < currentEarThreshold ? 'rgba(239, 68, 68, 0.3)' : 'rgba(56, 189, 248, 0.2)';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();

    // Pupils (if open)
    if (cEar >= currentEarThreshold) {
      ctx.fillStyle = '#0284c7';
      ctx.beginPath();
      ctx.arc(centerX - eyeOffsetX, centerY + eyeOffsetY, 3.5, 0, 2 * Math.PI);
      ctx.arc(centerX + eyeOffsetX, centerY + eyeOffsetY, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Mouth
    const mouthW = faceW * 0.32;
    const mouthH = Math.max(4, mouthW * cMar * 1.5);
    const mouthOffsetY = faceH * 0.24;

    ctx.beginPath();
    ctx.ellipse(centerX, centerY + mouthOffsetY, mouthW * 0.5, mouthH * 0.5, 0, 0, 2 * Math.PI);
    ctx.strokeStyle = cMar >= 0.55 ? '#f59e0b' : '#38bdf8';
    ctx.fillStyle = cMar >= 0.55 ? 'rgba(245, 158, 11, 0.3)' : 'rgba(56, 189, 248, 0.15)';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();

    // 3D Gaze / Head Pose Orientation Vector
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    const vecX = centerX + (cYaw / 45.0) * 80;
    const vecY = centerY + (cPitch / 40.0) * 80;
    ctx.lineTo(vecX, vecY);
    ctx.strokeStyle = isDistractedNow ? '#ef4444' : '#10b981';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Vector tip arrow
    ctx.fillStyle = isDistractedNow ? '#ef4444' : '#10b981';
    ctx.beginPath();
    ctx.arc(vecX, vecY, 5, 0, 2 * Math.PI);
    ctx.fill();
  };

  const isDistracted = isFaceDetected && (
    Math.abs(yaw - currentNeutralYaw) >= currentYawLimit ||
    pitch - currentNeutralPitch <= -currentPitchDownLimit ||
    pitch - currentNeutralPitch >= currentPitchUpLimit
  );
  const isEyeClosed = isFaceDetected && ear < currentEarThreshold;

  return (
    <div className="space-y-6">
      {/* Top Banner: Scenario Selector & Mode Explanation */}
      <div className="bg-slate-900/90 p-4 rounded-2xl border border-slate-800 shadow-xl flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Driving Scenario:
          </span>
          <div className="flex flex-wrap gap-1.5">
            {[
              { id: 'camera', label: 'Webcam (local AI)', icon: Camera },
              { id: 'normal', label: 'Normal Driving' },
              { id: 'microsleep', label: 'Microsleep (>1.5s)' },
              { id: 'distraction', label: 'Phone Distraction' },
              { id: 'yawn', label: 'Deep Yawn' },
              { id: 'narrow_eyes', label: 'Narrow Eyes / Squint (Hypothesis Test)' },
            ].map((sc) => (
              <button
                key={sc.id}
                onClick={() => setSourceType(sc.id as ScenarioType)}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  sourceType === sc.id
                    ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {sc.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center space-x-1 px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? 'Pause' : 'Resume'}</span>
          </button>
          <div className="px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-[11px] font-mono text-cyan-400">
            {fps.toFixed(1)} FPS
          </div>
        </div>
      </div>

      {cameraError && (
        <div className="p-3 rounded-xl bg-amber-950/60 border border-amber-800 text-amber-300 text-xs flex items-center space-x-2">
          <Info className="w-4 h-4 shrink-0" />
          <span>{cameraError}</span>
          {sourceType === 'camera' && (
            <button
              onClick={() => setCameraRetry((attempt) => attempt + 1)}
              className="ml-auto rounded-lg border border-amber-700 px-3 py-1 font-semibold text-amber-100 hover:bg-amber-900"
            >
              Try again
            </button>
          )}
        </div>
      )}

      {/* Main Monitoring View & HUD */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Visual Feed & Canvas Overlay (7 cols) */}
        <div className="lg:col-span-7 bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden relative shadow-2xl flex items-center justify-center min-h-[420px]">
          {sourceType === 'camera' && (
            <video
              ref={videoRef}
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]"
            />
          )}

          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className={`w-full h-full relative z-10 ${sourceType === 'camera' ? 'transform scale-x-[-1]' : ''}`}
          />

          {/* Active Alert Banner Overlay */}
          {activeAlert && (
            <div
              className={`absolute top-4 left-4 right-4 z-20 p-4 rounded-xl border backdrop-blur-md shadow-2xl transition-all animate-bounce ${
                activeAlert.severity === 'critical'
                  ? 'bg-rose-950/90 border-rose-600 text-rose-100 shadow-rose-900/40'
                  : 'bg-amber-950/90 border-amber-500 text-amber-100 shadow-amber-900/40'
              }`}
            >
              <div className="flex items-start space-x-3">
                <ShieldAlert className="w-6 h-6 shrink-0 text-white animate-pulse" />
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-black uppercase tracking-wider px-2 py-0.5 rounded bg-black/40">
                      {activeAlert.category.replace(/_/g, ' ')}
                    </span>
                    <span className="text-xs font-mono">
                      Triggered in {activeAlert.mode.toUpperCase()} MODE
                    </span>
                  </div>
                  <p className="mt-1 text-sm font-semibold">{activeAlert.explanation}</p>
                </div>
              </div>
            </div>
          )}

          {/* Bottom HUD bar on canvas */}
          <div className="absolute bottom-3 left-4 right-4 z-20 flex items-center justify-between text-xs font-mono text-slate-300 bg-slate-900/80 px-4 py-2 rounded-xl backdrop-blur-sm border border-slate-800/80">
            <div className="flex items-center space-x-4">
              <span className="flex items-center space-x-1.5">
                <span
                  className={`w-2 h-2 rounded-full ${
                    isFaceDetected ? 'bg-emerald-400' : 'bg-rose-500 animate-ping'
                  }`}
                />
                <span>{isFaceDetected ? 'FACE TRACKED' : 'SEARCHING...'}</span>
              </span>
              <span>EAR: {isFaceDetected ? ear.toFixed(3) : '—'}</span>
              <span>MAR: {isFaceDetected ? mar.toFixed(3) : '—'}</span>
            </div>
            <div className="flex items-center space-x-4">
              <span>YAW: {isFaceDetected ? `${yaw.toFixed(1)}°` : '—'}</span>
              <span>PITCH: {isFaceDetected ? `${pitch.toFixed(1)}°` : '—'}</span>
            </div>
          </div>
        </div>

        {/* Real-Time Gauges & Measurements HUD (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Card 1: Eye Aspect Ratio (EAR) & Closure Gauge */}
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Eye className="w-5 h-5 text-cyan-400" />
                <span className="text-sm font-semibold text-white">Eye Opening (EAR)</span>
              </div>
              <span
                className={`text-xs px-2 py-0.5 rounded font-bold uppercase ${
                  isEyeClosed
                    ? 'bg-rose-950 text-rose-400 border border-rose-800'
                    : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                }`}
              >
                {!isFaceDetected ? 'No Face' : isEyeClosed ? 'Eyes Closed' : 'Alert & Open'}
              </span>
            </div>

            <div className="flex items-baseline justify-between mb-2">
              <span className="text-3xl font-black font-mono tracking-tight text-white">
                {isFaceDetected ? ear.toFixed(3) : '—'}
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Threshold: <span className="text-cyan-400">{currentEarThreshold.toFixed(3)}</span>
                {mode === 'personalized' && ' (Adapted)'}
              </span>
            </div>

            {/* EAR Aperture Bar */}
            <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden border border-slate-800 relative">
              <div
                className={`h-full transition-all duration-150 ${
                  isEyeClosed ? 'bg-rose-500' : 'bg-gradient-to-r from-cyan-500 to-emerald-400'
                }`}
                style={{ width: `${isFaceDetected ? Math.min(100, (ear / 0.40) * 100) : 0}%` }}
              />
              {/* Threshold indicator notch */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-rose-400 z-10"
                style={{ left: `${(currentEarThreshold / 0.40) * 100}%` }}
                title={`Threshold: ${currentEarThreshold}`}
              />
            </div>

            {/* Closure Duration Alert Timer (> 1.5s rule) */}
            <div className="mt-3 flex items-center justify-between text-xs">
              <span className="text-slate-400 flex items-center space-x-1">
                <Clock className="w-3.5 h-3.5" />
                <span>Closure Duration:</span>
              </span>
              <span
                className={`font-mono font-bold ${
                  closureDuration >= 1.5
                    ? 'text-rose-400 animate-pulse text-sm'
                    : closureDuration > 0.5
                    ? 'text-amber-400'
                    : 'text-slate-400'
                }`}
              >
                {closureDuration.toFixed(2)}s / 1.50s max
              </span>
            </div>
          </div>

          {/* Card 2: Head Pose Compass & Distraction Gauge */}
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Compass className="w-5 h-5 text-indigo-400" />
                <span className="text-sm font-semibold text-white">Head Pose Orientation</span>
              </div>
              <span
                className={`text-xs px-2 py-0.5 rounded font-bold uppercase ${
                  isDistracted
                    ? 'bg-amber-950 text-amber-400 border border-amber-800'
                    : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                }`}
              >
                {!isFaceDetected ? 'No Face' : isDistracted ? 'Off Road' : 'Road Center'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-center my-2">
              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[11px] text-slate-400">Yaw (Left/Right)</div>
                <div className="text-xl font-bold font-mono text-white mt-0.5">
                  {!isFaceDetected ? '—' : yaw > 0 ? `+${yaw.toFixed(1)}°` : `${yaw.toFixed(1)}°`}
                </div>
                <div className="text-[10px] text-slate-500">Limit: ±{currentYawLimit.toFixed(0)}°</div>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[11px] text-slate-400">Pitch (Up/Down)</div>
                <div className="text-xl font-bold font-mono text-white mt-0.5">
                  {!isFaceDetected ? '—' : pitch > 0 ? `+${pitch.toFixed(1)}°` : `${pitch.toFixed(1)}°`}
                </div>
                <div className="text-[10px] text-slate-500">Limit: -18° / +22°</div>
              </div>
            </div>

            {/* Distraction Timer */}
            <div className="mt-3 flex items-center justify-between text-xs">
              <span className="text-slate-400 flex items-center space-x-1">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Distraction Timer:</span>
              </span>
              <span
                className={`font-mono font-bold ${
                  distractionDuration >= 2.0
                    ? 'text-amber-400 animate-pulse text-sm'
                    : 'text-slate-400'
                }`}
              >
                {distractionDuration.toFixed(2)}s / 2.00s limit
              </span>
            </div>
          </div>

          {/* Card 3: Mouth Opening (MAR) & Blink Rate */}
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-slate-400 font-medium">Mouth Opening (MAR)</div>
              <div className="text-2xl font-bold font-mono text-white mt-1">{isFaceDetected ? mar.toFixed(3) : '—'}</div>
              <div className="text-[11px] text-slate-500 mt-1">Yawn Limit: 0.550</div>
              {yawnDuration > 0 && (
                <div className="text-[11px] font-mono text-amber-400 mt-0.5 font-bold">
                  Yawn: {yawnDuration.toFixed(1)}s / 2.2s
                </div>
              )}
            </div>

            <div>
              <div className="text-xs text-slate-400 font-medium">Blink Frequency</div>
              <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">
                {bpm.toFixed(1)} <span className="text-xs font-normal text-slate-400">bpm</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1">Total: {totalBlinks} blinks</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
