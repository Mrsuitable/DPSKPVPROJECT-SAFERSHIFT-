import React, { useState } from 'react';
import { Lock, Shield, Check, Trash2, AlertTriangle, EyeOff, Server, HardDrive } from 'lucide-react';
import { DriverProfile } from '../types';

interface PrivacyViewProps {
  profiles: DriverProfile[];
  onPurgeAllProfiles: () => Promise<void>;
}

export const PrivacyView: React.FC<PrivacyViewProps> = ({
  profiles,
  onPurgeAllProfiles,
}) => {
  const [purged, setPurged] = useState<boolean>(false);
  const [purgeError, setPurgeError] = useState<string | null>(null);

  const handlePurge = async () => {
    setPurgeError(null);
    try {
      await onPurgeAllProfiles();
      setPurged(true);
      setTimeout(() => setPurged(false), 4000);
    } catch (error) {
      setPurgeError(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="space-y-8">
      {/* Privacy Constitution Banner */}
      <div className="bg-gradient-to-r from-slate-900 to-emerald-950 p-6 rounded-3xl border border-emerald-800/40 shadow-2xl">
        <div className="max-w-3xl">
          <div className="flex items-center space-x-2 text-emerald-400 text-xs font-bold uppercase tracking-wider mb-2">
            <Lock className="w-4 h-4" />
            <span>Zero-Retention Privacy Architecture</span>
          </div>
          <h2 className="text-2xl font-black text-white tracking-tight">
            Privacy-Preserving Local Driver Monitoring
          </h2>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed">
            SaferShift is engineered from the ground up to respect driver privacy.
            Inference runs completely offline on the local CPU with zero cloud dependencies.
            Raw webcam imagery is processed ephemerally in memory and never written to disk.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl space-y-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-950 border border-emerald-800/60 flex items-center justify-center text-emerald-400">
            <EyeOff className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">No Raw Video Storage</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Webcam feeds and video frames exist solely in volatile RAM during landmark extraction.
            No video clips, image snapshots, or facial crops are recorded or transmitted.
          </p>
        </div>

        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl space-y-3">
          <div className="w-10 h-10 rounded-xl bg-cyan-950 border border-cyan-800/60 flex items-center justify-center text-cyan-400">
            <HardDrive className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">Only Derived Thresholds</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Driver profiles store only derived mathematical floating-point parameters (e.g., adapted EAR = 0.185).
            No facial recognition embeddings or identity-identifying biometrics are computed.
          </p>
        </div>

        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl space-y-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-950 border border-indigo-800/60 flex items-center justify-center text-indigo-400">
            <Server className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-white">Offline Edge Operation</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            All facial landmark analysis, solvePnP head pose estimation, and sliding-window temporal
            alert triggers execute locally on edge CPU hardware without cloud API calls.
          </p>
        </div>
      </div>

      {/* Audit Specification Matrix */}
      <div className="bg-slate-900/90 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-4">
        <h3 className="text-lg font-bold text-white">Data Processing & Storage Audit Matrix</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                <th className="py-2.5 px-3">Data Artifact</th>
                <th className="py-2.5 px-3">Processed Location</th>
                <th className="py-2.5 px-3">Persistent Storage</th>
                <th className="py-2.5 px-3">Retention Policy</th>
                <th className="py-2.5 px-3">Deletion Mechanism</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              <tr>
                <td className="py-2.5 px-3 font-sans font-semibold text-white">Webcam Video Stream</td>
                <td className="py-2.5 px-3 text-emerald-400">Local RAM Only</td>
                <td className="py-2.5 px-3 text-rose-400">NEVER (0 bytes)</td>
                <td className="py-2.5 px-3 text-slate-400 font-sans">Zero Retention (Immediate purge)</td>
                <td className="py-2.5 px-3 text-slate-400 font-sans">Automatic</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-sans font-semibold text-white">Facial Landmarks (468 pts)</td>
                <td className="py-2.5 px-3 text-emerald-400">Local CPU</td>
                <td className="py-2.5 px-3 text-rose-400">NEVER (0 bytes)</td>
                <td className="py-2.5 px-3 text-slate-400 font-sans">Discarded after EAR/MAR calculation</td>
                <td className="py-2.5 px-3 text-slate-400 font-sans">Automatic</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-sans font-semibold text-white">Driver Profile Settings</td>
                <td className="py-2.5 px-3 text-cyan-400">Local Device</td>
                <td className="py-2.5 px-3 text-cyan-300">data/profiles/*.json</td>
                <td className="py-2.5 px-3 text-slate-400 font-sans">Stores derived threshold statistics only</td>
                <td className="py-2.5 px-3 text-cyan-400 font-sans">User Delete / 1-Click Purge</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-sans font-semibold text-white">Safety Event Telemetry</td>
                <td className="py-2.5 px-3 text-cyan-400">Local Device</td>
                <td className="py-2.5 px-3 text-cyan-300">data/logs/*.csv</td>
                <td className="py-2.5 px-3 text-slate-400 font-sans">Numerical metrics & timestamps only</td>
                <td className="py-2.5 px-3 text-cyan-400 font-sans">Local file purge</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Immediate Biometric Purge Control */}
      <div className="bg-rose-950/40 p-6 rounded-2xl border border-rose-900/60 shadow-xl flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-rose-200 flex items-center space-x-2">
            <Trash2 className="w-5 h-5 text-rose-400" />
            <span>Driver Biometric Profile Purge</span>
          </h3>
          <p className="text-xs text-rose-300/80 mt-1 max-w-xl">
            Permanently delete all calibrated driver profiles from local disk.
            The system will immediately revert to the universal baseline (Standard Mode).
          </p>
        </div>

        <button
          onClick={handlePurge}
          className="px-4 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold shadow-lg shadow-rose-600/30 transition-all flex items-center space-x-2"
        >
          <Trash2 className="w-4 h-4" />
          <span>Purge All Calibrated Profiles ({profiles.length})</span>
        </button>
      </div>

      {purged && (
        <div className="p-4 rounded-xl bg-emerald-950/80 border border-emerald-800 text-emerald-300 text-xs flex items-center space-x-2">
          <Check className="w-4 h-4" />
          <span>All local driver profiles have been purged successfully.</span>
        </div>
      )}
      {purgeError && (
        <div role="alert" className="p-4 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-300 text-xs">
          {purgeError}
        </div>
      )}

      {/* Safety & Medical Disclaimer */}
      <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 text-xs text-slate-400 leading-relaxed flex items-start space-x-3">
        <AlertTriangle className="w-5 h-5 shrink-0 text-amber-400 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">Safety & Research Disclaimer: </span>
          SaferShift is an edge computer-vision assistive intelligence research prototype. It does
          not provide medical diagnoses of sleep apnea or narcolepsy, nor does it guarantee collision avoidance.
          Drivers must never deliberately test alerts or induce drowsiness in an actively moving vehicle.
        </div>
      </div>
    </div>
  );
};
