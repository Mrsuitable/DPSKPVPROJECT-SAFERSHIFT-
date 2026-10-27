"""SaferShift Streamlit Dashboard.

Displays:
- Panel 1: Monitoring & Tracking Status (Standard vs Personalized, FPS, Alerts)
- Panel 2: Eye & Mouth Aspect Ratio (EAR & MAR time-series trends)
- Panel 3: Head Pose (Yaw & Pitch deviation from calibrated neutral)
- Panel 4: Event Analytics (Blinks, Yawns, Closures, Distractions)
- Panel 5: Explainable Alert History (with triggering measurements & explanations)
- Panel 6: Driver Profile Management (Inspection, Calibration Quality, Safe Deletion)
- Panel 7: Telemetry Export & Evaluation Benchmark Results
"""

import os
import sys
from pathlib import Path
import json

# Ensure project modules are discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st = None

from config import LOGS_DIR, REPORTS_DIR, PROFILES_DIR, STANDARD_CONFIG
from personalization.profiles import ProfileManager
from telemetry.event_logger import load_telemetry_records


def run_dashboard():
    if st is None:
        print("[!] Streamlit is not installed. Run 'pip install streamlit plotly pandas' to launch.")
        return

    st.set_page_config(
        page_title="SaferShift - Driver Safety Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .metric-card {
            background-color: #1e293b;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #334155;
        }
        .alert-critical {
            background-color: #7f1d1d;
            border-left: 5px solid #ef4444;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin-bottom: 0.5rem;
        }
        .alert-warning {
            background-color: #78350f;
            border-left: 5px solid #f59e0b;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ SaferShift: Driver Safety Intelligence")
    st.caption("Personalized, Privacy-Preserving Edge Fatigue & Distraction Monitor")

    # Sidebar: Session & Profile Selection
    st.sidebar.header("Monitoring Configuration")
    mode = st.sidebar.radio("Detection Mode", ["Standard Mode", "Personalized Mode"])
    profile_mgr = ProfileManager()
    profiles = profile_mgr.list_profiles()

    selected_profile = None
    if mode == "Personalized Mode":
        if profiles:
            p_names = {f"{p.driver_name} ({p.profile_id})": p.profile_id for p in profiles}
            choice = st.sidebar.selectbox("Active Driver Profile", list(p_names.keys()))
            selected_profile = profile_mgr.load_profile(p_names[choice])
        else:
            st.sidebar.warning("No calibrated profiles found. Create one in Profile Controls.")

    # Locate telemetry files
    log_files = sorted(list(LOGS_DIR.glob("telemetry_*.csv")), reverse=True)
    if not log_files:
        st.warning("No telemetry sessions found. Run `python run_monitor.py` to start logging.")
        records = []
    else:
        latest_file = log_files[0]
        st.sidebar.info(f"📁 Log Source: `{latest_file.name}`")
        records = load_telemetry_records(latest_file)

    # Convert records to DataFrame
    if records:
        df = pd.DataFrame([r.__dict__ for r in records])
    else:
        df = pd.DataFrame()

    # Panel 1: Monitoring Status
    st.subheader("1. System Health & Real-Time Monitoring Status")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Detection Mode", "Personalized" if mode == "Personalized Mode" else "Standard")
    with col2:
        calib_text = f"Quality: {int(selected_profile.calibration_quality * 100)}%" if selected_profile else "Universal Default"
        st.metric("Calibration Status", calib_text)
    with col3:
        latest_tracking = df["tracking_quality"].iloc[-1] if not df.empty else 0.0
        track_str = "Reliable (100%)" if latest_tracking > 0.8 else ("Degraded" if latest_tracking > 0 else "Disconnected")
        st.metric("Tracking Quality", track_str)
    with col4:
        measured_fps = df["measured_fps"].iloc[-1] if not df.empty else 0.0
        st.metric("Measured FPS", f"{measured_fps:.1f} FPS")
    with col5:
        critical_alerts = df[df["event_type"] == "prolonged_eye_closure"] if not df.empty else []
        st.metric("Prolonged Closures", len(critical_alerts))

    st.markdown("---")

    # Panel 2: EAR & MAR Trends
    st.subheader("2. Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR)")
    if not df.empty and len(df) > 1:
        ear_threshold = selected_profile.adapted_ear_threshold if selected_profile else STANDARD_CONFIG.EAR_THRESHOLD

        fig_ratios = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Eye Aspect Ratio (EAR)", "Mouth Aspect Ratio (MAR)"])

        fig_ratios.add_trace(
            go.Scatter(x=df["timestamp"], y=df["ear"], mode="lines", name="EAR", line=dict(color="#38bdf8", width=2)),
            row=1, col=1
        )
        fig_ratios.add_hline(
            y=ear_threshold, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Threshold ({ear_threshold:.3f})", row=1, col=1
        )

        fig_ratios.add_trace(
            go.Scatter(x=df["timestamp"], y=df["mar"], mode="lines", name="MAR", line=dict(color="#f59e0b", width=2)),
            row=2, col=1
        )
        fig_ratios.add_hline(
            y=STANDARD_CONFIG.MAR_THRESHOLD, line_dash="dash", line_color="#dc2626",
            annotation_text=f"Yawn Threshold ({STANDARD_CONFIG.MAR_THRESHOLD})", row=2, col=1
        )

        fig_ratios.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
        st.plotly_chart(fig_ratios, use_container_width=True)
    else:
        st.info("Waiting for telemetry data...")

    st.markdown("---")

    # Panel 3: Head Pose Dynamics
    st.subheader("3. Head Pose Orientation (Yaw & Pitch)")
    if not df.empty and len(df) > 1:
        neutral_yaw = selected_profile.neutral_yaw if selected_profile else 0.0
        neutral_pitch = selected_profile.neutral_pitch if selected_profile else 0.0

        fig_pose = go.Figure()
        fig_pose.add_trace(go.Scatter(x=df["timestamp"], y=df["yaw"], mode="lines", name="Yaw (°)", line=dict(color="#10b981")))
        fig_pose.add_trace(go.Scatter(x=df["timestamp"], y=df["pitch"], mode="lines", name="Pitch (°)", line=dict(color="#8b5cf6")))

        fig_pose.add_hline(y=neutral_yaw + 25.0, line_dash="dot", line_color="#f87171", annotation_text="Max Yaw Right")
        fig_pose.add_hline(y=neutral_yaw - 25.0, line_dash="dot", line_color="#f87171", annotation_text="Max Yaw Left")

        fig_pose.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark")
        st.plotly_chart(fig_pose, use_container_width=True)

    st.markdown("---")

    # Panel 4 & 5: Event Analytics & Explainable Alert History
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("4. Event Analytics")
        if not df.empty:
            total_blinks = int(df["total_blinks"].max()) if "total_blinks" in df else 0
            bpm = df["blink_rate_bpm"].iloc[-1] if "blink_rate_bpm" in df else 0.0
            yawn_events = len(df[df["event_type"] == "possible_yawning"])
            distractions = len(df[df["event_type"] == "sustained_distraction"])
            closures = len(df[df["event_type"] == "prolonged_eye_closure"])

            c1, c2 = st.columns(2)
            c1.metric("Blink Rate", f"{bpm:.1f} bpm")
            c1.metric("Total Blinks", total_blinks)
            c2.metric("Yawn Events", yawn_events)
            c2.metric("Distraction Alerts", distractions)

    with col_right:
        st.subheader("5. Explainable Alert History")
        if not df.empty:
            events_df = df[df["event_type"] != "sample"]
            if events_df.empty:
                st.success("No active safety warnings. Safe driving state maintained.")
            else:
                for _, row in events_df.tail(8).iterrows():
                    cls = "alert-critical" if row["event_type"] == "prolonged_eye_closure" else "alert-warning"
                    st.markdown(f"""
                        <div class="{cls}">
                            <strong>[{row['event_type'].upper()}]</strong> at {row['timestamp_iso'][11:19]} UTC<br>
                            <em>Reason:</em> {row['warning_reason']}<br>
                            <small>Mode: {row['mode']} | EAR: {row['ear']:.3f} | Yaw: {row['yaw']:.1f}° | Closure: {row['closure_duration']:.2f}s</small>
                        </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")

    # Panel 6: Profile Controls & Calibration Management
    st.subheader("6. Driver Profiles & Privacy Controls")
    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.write("#### Calibrated Profiles")
        if profiles:
            for p in profiles:
                with st.expander(f"👤 {p.driver_name} (ID: {p.profile_id})"):
                    st.json({
                        "adapted_ear_threshold": p.adapted_ear_threshold,
                        "baseline_ear_mean": p.baseline_ear_mean,
                        "neutral_yaw": p.neutral_yaw,
                        "neutral_pitch": p.neutral_pitch,
                        "calibration_quality": f"{int(p.calibration_quality * 100)}%",
                        "created_at": p.created_at,
                    })
                    if st.button(f"🗑️ Delete Profile '{p.profile_id}'", key=f"del_{p.profile_id}"):
                        profile_mgr.delete_profile(p.profile_id)
                        st.success(f"Profile {p.profile_id} permanently deleted.")
                        st.rerun()
        else:
            st.info("No profiles stored locally.")

    with p_col2:
        st.write("#### Privacy Guarantee")
        st.info("""
            **Local Biometric Protection:**
            - SaferShift does NOT record or store raw webcam footage.
            - Calibration files store only derived statistical thresholds (e.g. median EAR, neutral angles).
            - No facial recognition or identity tracking is performed.
            - Driver profiles can be purged permanently at any time.
        """)

    st.markdown("---")

    # Panel 7: Telemetry Export & Evaluation Benchmark
    st.subheader("7. Telemetry Export & Evaluation Benchmark")
    e_col1, e_col2 = st.columns([1, 1])

    with e_col1:
        if not df.empty:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Telemetry CSV",
                data=csv_data,
                file_name=latest_file.name if log_files else "safer_shift_telemetry.csv",
                mime="text/csv",
            )

    with e_col2:
        bench_json = REPORTS_DIR / "benchmark_report.json"
        if bench_json.exists():
            with open(bench_json, "r") as f:
                b_data = json.load(f)
            st.write(f"**Benchmark F1 Score:** Standard: `{b_data['standard']['f1_score']:.3f}` | Personalized: `{b_data['personalized']['f1_score']:.3f}`")
            st.write(f"**False Alarm Reduction:** `{b_data['false_alarm_reduction_pct']}%`")


if __name__ == "__main__":
    if st is not None:
        run_dashboard()
    else:
        print("[!] Streamlit is required to run the dashboard.")
