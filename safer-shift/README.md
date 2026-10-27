# SaferShift Python engine

This folder contains the local camera monitor, calibration and profile logic, CSV telemetry, Streamlit dashboard, and scenario benchmark. The React/Express dashboard and repository setup guide are in the parent directory: [../README.md](../README.md).

## Install

Use Python 3.10 or newer, then install the dependencies from this folder:

```bash
python -m pip install -r requirements.txt
```

On Windows, replace `python` with `py -3` if the Python launcher is used.

## Run the monitor

```bash
python run_monitor.py --source demo --mode standard
python run_monitor.py --source 0 --mode standard
python run_monitor.py --source /path/to/driver_video.mp4 --mode standard
python run_monitor.py --source 0 --mode personalized --profile driver_alex --calibrate
```

`--source demo` generates a synthetic feed. A camera index or video path uses that input; if it cannot be opened, the monitor reports the issue and falls back to the synthetic feed. Use `--headless` on a machine without a display. `opencv-contrib-python` supplies the OpenCV GUI used by the visible monitor.

## Dashboard and evaluation

```bash
streamlit run app.py
python -m pytest tests -v
python evaluation/evaluate.py
```

The Streamlit page reads local profiles, telemetry CSV files, and the latest report. The benchmark evaluates the bundled scenario data and writes `reports/benchmark_report.md` and `reports/benchmark_report.json`; its results reflect those scenarios, not a clinical or road-safety certification.

## Project map

- `run_monitor.py`: capture loop, HUD, and CLI.
- `detector/`: face landmarks, EAR/MAR calculations, head pose, temporal rules, and alerts.
- `personalization/`: calibration quality checks, bounded thresholds, and JSON profiles.
- `telemetry/`: local CSV event records and performance metrics.
- `evaluation/`: scenario annotations, event matching, and benchmark generation.
- `app.py`: Streamlit views for telemetry, profiles, and evaluation reports.
- `models/face_landmarker.task`: local MediaPipe model used by the Python detector.

Raw frames are processed in memory and never written. Profiles store derived thresholds; logs store numeric metrics and event details. SaferShift is a research prototype and must not be used as the only safety system while driving.
