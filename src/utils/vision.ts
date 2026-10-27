export interface FacePoint {
  x: number;
  y: number;
  z?: number;
}

export interface FaceMeshResults {
  multiFaceLandmarks: FacePoint[][];
}

export interface FaceMeshInstance {
  setOptions(options: {
    maxNumFaces: number;
    refineLandmarks: boolean;
    minDetectionConfidence: number;
    minTrackingConfidence: number;
  }): void;
  initialize(): Promise<void>;
  onResults(listener: (results: FaceMeshResults) => void): void;
  send(input: { image: HTMLVideoElement }): Promise<void>;
  close(): Promise<void>;
}

export interface FaceMeshConstructor {
  new (options: { locateFile: (file: string) => string }): FaceMeshInstance;
}

declare global {
  interface Window {
    FaceMesh?: FaceMeshConstructor;
  }
}

export interface FaceMeasurements {
  ear: number;
  mar: number;
  yaw: number;
  pitch: number;
  roll: number;
}

let faceMeshScriptPromise: Promise<FaceMeshConstructor> | null = null;

export function loadFaceMesh(): Promise<FaceMeshConstructor> {
  if (window.FaceMesh) return Promise.resolve(window.FaceMesh);
  if (faceMeshScriptPromise) return faceMeshScriptPromise;

  faceMeshScriptPromise = new Promise((resolve, reject) => {
    const existingScript = document.querySelector<HTMLScriptElement>('script[data-safer-shift-face-mesh]');
    const script = existingScript ?? document.createElement('script');
    const fail = () => {
      script.remove();
      faceMeshScriptPromise = null;
      reject(new Error('The local FaceMesh files could not be loaded. Run npm run build after npm install.'));
    };
    const ready = () => {
      if (window.FaceMesh) resolve(window.FaceMesh);
      else fail();
    };

    script.addEventListener('error', fail, { once: true });
    script.addEventListener('load', ready, { once: true });
    if (!existingScript) {
      script.dataset.saferShiftFaceMesh = 'true';
      script.src = '/mediapipe/face_mesh/face_mesh.js';
      document.head.appendChild(script);
    }
  });

  return faceMeshScriptPromise;
}

export function choosePrimaryFace(faces: FacePoint[][]): FacePoint[] | null {
  if (faces.length === 0) return null;

  let bestFace: FacePoint[] | null = null;
  let bestScore = -1;
  for (const face of faces) {
    if (face.length < 400) continue;
    const xs = face.map((point) => point.x);
    const ys = face.map((point) => point.y);
    const width = Math.max(0.0001, Math.max(...xs) - Math.min(...xs));
    const height = Math.max(0.0001, Math.max(...ys) - Math.min(...ys));
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
    const centerDistance = Math.hypot(centerX - 0.5, centerY - 0.5);
    const score = width * height * Math.max(0.1, 1 - centerDistance);
    if (score > bestScore) {
      bestFace = face;
      bestScore = score;
    }
  }
  return bestFace;
}

export function measureFace(face: FacePoint[], width: number, height: number): FaceMeasurements | null {
  if (face.length < 400) return null;
  const point = (index: number) => ({ x: face[index].x * width, y: face[index].y * height });
  const distance = (a: number, b: number) => {
    const first = point(a);
    const second = point(b);
    return Math.hypot(first.x - second.x, first.y - second.y);
  };
  const eyeRatio = (indices: number[]) => {
    const horizontal = distance(indices[0], indices[3]);
    if (horizontal < 1e-6) return 0;
    return (distance(indices[1], indices[5]) + distance(indices[2], indices[4])) / (2 * horizontal);
  };

  const ear = (eyeRatio([33, 160, 158, 133, 153, 144]) + eyeRatio([362, 385, 387, 263, 373, 380])) / 2;
  const mouthWidth = distance(78, 308);
  if (mouthWidth < 1e-6) return null;
  const mar = (
    distance(13, 14) + 0.5 * (distance(82, 87) + distance(312, 317))
  ) / (2 * mouthWidth);

  // Match the Python engine's geometric solvePnP fallback so both interfaces use
  // the same sign convention and approximately the same neutral pose scale.
  const nose = point(1);
  const leftEye = point(33);
  const rightEye = point(263);
  const leftMouth = point(61);
  const rightMouth = point(291);
  const eyeDx = rightEye.x - leftEye.x;
  const eyeDy = rightEye.y - leftEye.y;
  const eyeWidth = Math.hypot(eyeDx, eyeDy);
  const yaw = eyeWidth > 1e-4
    ? ((nose.x - (leftEye.x + rightEye.x) / 2) / (eyeWidth * 0.5)) * 45
    : 0;
  const eyeLineY = (leftEye.y + rightEye.y) / 2;
  const mouthLineY = (leftMouth.y + rightMouth.y) / 2;
  const faceHeight = mouthLineY - eyeLineY;
  const expectedNoseY = eyeLineY + 0.4 * faceHeight;
  const pitch = faceHeight > 1e-4 ? -((nose.y - expectedNoseY) / faceHeight) * 55 : 0;
  const roll = Math.atan2(eyeDy, eyeDx) * (180 / Math.PI);

  return {
    ear: Math.max(0, ear),
    mar: Math.max(0, mar),
    yaw: Math.max(-90, Math.min(90, yaw)),
    pitch: Math.max(-90, Math.min(90, pitch)),
    roll: Math.max(-90, Math.min(90, roll)),
  };
}

export const FACE_MESH_EYES = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380];
export const FACE_MESH_MOUTH = [78, 82, 13, 312, 308, 317, 14, 87];
