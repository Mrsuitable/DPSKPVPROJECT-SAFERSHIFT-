# SaferShift

SaferShift is a local driver-safety prototype with two ways to use the same monitoring ideas:

- A React dashboard served by an Express API. Its scenario controls demonstrate the temporal alerts; webcam mode uses MediaPipe FaceMesh in the browser to measure the live camera feed.
- A Python engine for real-time camera or video monitoring, calibration, telemetry, benchmarking, and a Streamlit dashboard.

Camera images stay in memory. The web monitor stores only numerical telemetry and safety events in `safer-shift/data/logs/`. Driver calibration profiles are stored locally in `safer-shift/data/profiles/`.

## Run the web dashboard

Prerequisites: Node.js 20.19+ or 22.12+ and Python 3.10+ with `pip`.

```bash
npm install
python -m pip install -r safer-shift/requirements.txt
npm run dev
```

Open `http://127.0.0.1:3000`. On Windows, use `py -3 -m pip install -r safer-shift/requirements.txt` if `python` is not on your PATH. The backend uses the `py -3` launcher on Windows and `python3` on Linux/macOS; set `SAFERSHIFT_PYTHON` to use a specific Python executable. Set `PORT` to change the local server port.

## Dependencies

The web app uses React, React DOM, TypeScript, Vite, Tailwind CSS, Express, tsx, Motion, Lucide React, and the local `@mediapipe/face_mesh` and `@mediapipe/camera_utils` packages. Exact versions are in `package.json`.

The Python engine requires `opencv-contrib-python`, MediaPipe, NumPy, pandas, scikit-learn, Streamlit, Plotly, and pytest. Exact minimum versions are in `safer-shift/requirements.txt`.

## Run the Python monitor

```bash
cd safer-shift
python -m pip install -r requirements.txt
python run_monitor.py --mode standard --source demo
```

Use `--source 0` for the default webcam, a video path for a recording, and `--headless` when no display is available. On Windows, use `py -3` in place of `python` if needed. Launch the separate telemetry dashboard with `streamlit run app.py`.

For a personalized session, calibrate a stationary driver profile first:

```bash
python run_monitor.py --mode personalized --profile driver_alex --calibrate --source 0
```

## Checks and benchmark

From the repository root:

```bash
npm run lint
npm run build
```

From `safer-shift/`:

```bash
python -m pytest tests -v
python evaluation/evaluate.py
```

The web dashboard can also run the Python test suite and regenerate the benchmark from its Benchmark tab. These actions require the Python dependencies above. The generated report is written to `safer-shift/reports/`.

## Files and their purposes

### Repository root

| File or folder | Purpose |
| --- | --- |
| `README.md` | Setup, operation, dependency, and project-file guide. |
| `package.json` | Web dependencies and `dev`, `build`, `lint`, and `clean` commands. |
| `package-lock.json` / `bun.lock` | Locked dependency trees for npm and Bun. |
| `server.ts` | Express API, profile/telemetry endpoints, Python command runner, and Vite development server. |
| `vite.config.ts` | Vite/React configuration and local MediaPipe asset copying. |
| `tsconfig.json` | TypeScript compiler settings. |
| `index.html` | Web application HTML entry point. |
| `metadata.json` | Application metadata and camera permission declaration. |
| `scripts/clean.mjs` | Cross-platform cleanup of generated web files. |
| `public/assets/aistudio/*` | Static asset exported with the original project scaffold. |

### React application: `src/`

| File | Purpose |
| --- | --- |
| `main.tsx` | React browser entry point. |
| `App.tsx` | Root view, navigation, profile state, and API integration. |
| `index.css` | Global styles and Tailwind imports. |
| `types/index.ts` | Shared types for alerts, profiles, telemetry, and views. |
| `utils/audio.ts` | Browser warning-sound helper. |
| `utils/vision.ts` | Local MediaPipe loader, primary-face selection, and EAR/MAR/head-pose calculations. |
| `components/Navbar.tsx` | Navigation and monitor mode controls. |
| `components/LiveMonitor.tsx` | Webcam monitor, scenario simulator, metrics, overlays, and alerts. |
| `components/CalibrationStudio.tsx` | Live-camera calibration and safe profile-threshold creation. |
| `components/TelemetryView.tsx` | Local telemetry history and event display. |
| `components/BenchmarkView.tsx` | Benchmark-report and Python-test controls. |
| `components/PrivacyView.tsx` | Local-data explanation and profile/log purge controls. |

### Python engine: `safer-shift/`

| File or folder | Purpose |
| --- | --- |
| `README.md` | Focused Python-engine usage guide. |
| `requirements.txt` | Required Python libraries. |
| `config.py` | Default thresholds, safety bounds, and timing configuration. |
| `run_monitor.py` | CLI monitor, camera/file/demo inputs, HUD, calibration, alerts, and CSV logging. |
| `app.py` | Streamlit dashboard for profiles, logs, and reports. |
| `test_chunk.bin` | Small bundled binary fixture from the project source. |
| `detector/__init__.py` | Detector package marker. |
| `detector/face_landmarks.py` | MediaPipe landmark wrapper and primary-driver face selection. |
| `detector/eye_metrics.py` | Eye Aspect Ratio and blink calculation. |
| `detector/mouth_metrics.py` | Mouth Aspect Ratio calculation. |
| `detector/head_pose.py` | Head yaw, pitch, and roll estimation. |
| `detector/temporal_engine.py` | Time-based detection rules for closures, distraction, yawning, and tracking loss. |
| `detector/alerts.py` | Alert categories, severity, and event data structures. |
| `detector/verify_detector.py` | Manual detector verification helper. |
| `personalization/__init__.py` | Personalization package marker. |
| `personalization/calibration.py` | Calibration validation and safety-bounded threshold calculation. |
| `personalization/profiles.py` | Local JSON driver-profile persistence and validation. |
| `telemetry/__init__.py` | Telemetry package marker. |
| `telemetry/event_logger.py` | Local CSV telemetry-event writer. |
| `telemetry/metrics.py` | Frame-rate and latency measurements. |
| `evaluation/__init__.py` | Evaluation package marker. |
| `evaluation/sample_annotations.csv` | Ground-truth labels for the bundled scenarios. |
| `evaluation/event_matching.py` | Predicted-alert to annotation matching. |
| `evaluation/evaluate.py` | Benchmark generation for standard and personalized modes. |
| `tests/__init__.py` | Test package marker. |
| `tests/test_eye_metrics.py` | EAR and blink unit tests. |
| `tests/test_temporal_engine.py` | Time-based alert unit tests. |
| `tests/test_calibration.py` | Calibration and safety-bound unit tests. |
| `tests/test_profiles.py` | Profile persistence and validation unit tests. |
| `tests/test_event_matching.py` | Evaluation matching unit tests. |
| `models/README.md` | Local-model folder notes. |
| `models/face_landmarker.task` | Local MediaPipe model asset used by the Python detector. |
| `data/profiles/` | Runtime-created local profile JSON files. |
| `data/logs/` | Runtime-created numerical telemetry CSV files. |
| `reports/` | Runtime-created benchmark report files. |

## AI tools and models used

- MediaPipe FaceMesh and FaceLandmarker are pre-trained machine-learning components used for facial landmark detection. The browser monitor uses the local FaceMesh package, and the Python monitor uses `models/face_landmarker.task`.
- OpenAI Codex was used to assist with implementation, debugging, and documentation updates in this workspace.
- `@google/genai` remains listed as an original scaffold dependency, but the current application does not import it or make Gemini API requests.

## Safety

SaferShift is a research and educational prototype. It does not provide a medical diagnosis or guarantee collision avoidance. Do not test alerts or induce drowsiness while driving.
