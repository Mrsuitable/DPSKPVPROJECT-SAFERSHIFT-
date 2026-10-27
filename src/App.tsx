import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LiveMonitor } from './components/LiveMonitor';
import { CalibrationStudio } from './components/CalibrationStudio';
import { TelemetryView } from './components/TelemetryView';
import { BenchmarkView } from './components/BenchmarkView';
import { PrivacyView } from './components/PrivacyView';
import { DetectionMode, DriverProfile, AlertEvent, TelemetryPoint } from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('monitor');
  const [mode, setMode] = useState<DetectionMode>('standard');
  const [activeProfile, setActiveProfile] = useState<DriverProfile | null>(null);
  const [profiles, setProfiles] = useState<DriverProfile[]>([]);
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryPoint[]>([]);
  const [alertHistory, setAlertHistory] = useState<AlertEvent[]>([]);
  const [isMuted, setIsMuted] = useState<boolean>(false);

  useEffect(() => {
    void fetchProfiles();
  }, []);

  const fetchProfiles = async () => {
    try {
      const res = await fetch('/api/profiles');
      if (!res.ok) throw new Error('Could not load saved driver profiles.');
      const data = await res.json();
      if (data.success && Array.isArray(data.profiles)) {
        setProfiles(data.profiles);
        setActiveProfile(data.profiles[0] ?? null);
      } else {
        setProfiles([]);
        setActiveProfile(null);
      }
    } catch {
      setProfiles([]);
      setActiveProfile(null);
    }
  };

  const handleProfileCreated = async (newProfile: DriverProfile) => {
    const response = await fetch('/api/profiles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newProfile),
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.error || 'Could not save the driver profile locally.');
    }
    const savedProfile = result.profile as DriverProfile;
    setProfiles((prev) => {
      const filtered = prev.filter((p) => p.profile_id !== savedProfile.profile_id);
      return [savedProfile, ...filtered];
    });
    setActiveProfile(savedProfile);
    setMode('personalized');
  };

  const handleProfileDeleted = async (profileId: string) => {
    const response = await fetch(`/api/profiles/${encodeURIComponent(profileId)}`, { method: 'DELETE' });
    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.error || 'Could not delete the driver profile.');
    }
    const remaining = profiles.filter((profile) => profile.profile_id !== profileId);
    setProfiles(remaining);
    if (activeProfile?.profile_id === profileId) {
      setActiveProfile(remaining[0] ?? null);
      if (remaining.length === 0) setMode('standard');
    }
  };

  const handlePurgeAllProfiles = async () => {
    const results = await Promise.all(profiles.map((profile) =>
      fetch(`/api/profiles/${encodeURIComponent(profile.profile_id)}`, { method: 'DELETE' }),
    ));
    if (results.some((response) => !response.ok)) {
      throw new Error('One or more profiles could not be deleted. Please retry.');
    }
    setProfiles([]);
    setActiveProfile(null);
    setMode('standard');
  };

  const handleTelemetryUpdate = (point: TelemetryPoint) => {
    setTelemetryHistory((prev) => {
      const next = [...prev, point];
      return next.length > 200 ? next.slice(-200) : next;
    });

    // Send periodic sample or alert to server
    if (point.eventType !== 'sample') {
      fetch('/api/logs/record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamp: point.timestamp / 1000,
          timestamp_iso: new Date(point.timestamp).toISOString(),
          event_type: point.eventType,
          mode,
          ear: point.ear,
          mar: point.mar,
          yaw: point.yaw,
          pitch: point.pitch,
          roll: point.roll,
          closure_duration: point.closureDuration,
          distraction_duration: point.distractionDuration,
          yawn_duration: point.yawnDuration,
          blink_rate_bpm: point.blinkRateBpm,
          total_blinks: point.totalBlinks,
          calibration_status: mode === 'personalized' ? 'calibrated' : 'standard',
          warning_reason: point.warningReason || '',
          tracking_quality: 1.0,
          measured_fps: point.fps,
        }),
      }).catch(() => {});
    }
  };

  const handleAlertTriggered = (alert: AlertEvent) => {
    setAlertHistory((prev) => [alert, ...prev.slice(0, 49)]);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        mode={mode}
        setMode={setMode}
        activeProfile={activeProfile}
        isMuted={isMuted}
        setIsMuted={setIsMuted}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'monitor' && (
          <LiveMonitor
            mode={mode}
            activeProfile={activeProfile}
            isMuted={isMuted}
            onTelemetryUpdate={handleTelemetryUpdate}
            onAlertTriggered={handleAlertTriggered}
          />
        )}

        {activeTab === 'calibration' && (
          <CalibrationStudio
            activeProfile={activeProfile}
            profiles={profiles}
            onProfileCreated={handleProfileCreated}
            onProfileDeleted={handleProfileDeleted}
            onSelectProfile={setActiveProfile}
          />
        )}

        {activeTab === 'telemetry' && (
          <TelemetryView
            telemetryHistory={telemetryHistory}
            alertHistory={alertHistory}
            mode={mode}
          />
        )}

        {activeTab === 'benchmark' && <BenchmarkView />}

        {activeTab === 'privacy' && (
          <PrivacyView
            profiles={profiles}
            onPurgeAllProfiles={handlePurgeAllProfiles}
          />
        )}
      </main>

      {/* Bottom Status Bar */}
      <footer className="bg-slate-900 border-t border-slate-800/80 py-3 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center space-x-3">
            <span className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="font-semibold text-slate-300">SaferShift Engine v1.0</span>
            </span>
            <span>•</span>
            <span>Offline Edge CPU Inference</span>
            <span>•</span>
            <span className="text-cyan-400 font-mono">
              Active Mode: {mode.toUpperCase()}
            </span>
          </div>

          <div className="flex items-center space-x-3 font-mono text-[11px]">
            <span>Prolonged Closure Rule: &gt;1.50s</span>
            <span>•</span>
            <span className="text-emerald-400">Zero Video Retention Active</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
