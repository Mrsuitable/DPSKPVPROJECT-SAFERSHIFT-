import React from 'react';
import { Shield, Activity, Sliders, BarChart3, FileCheck, Lock, Volume2, VolumeX, Eye } from 'lucide-react';
import { DetectionMode, DriverProfile } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  mode: DetectionMode;
  setMode: (mode: DetectionMode) => void;
  activeProfile: DriverProfile | null;
  isMuted: boolean;
  setIsMuted: (muted: boolean) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  mode,
  setMode,
  activeProfile,
  isMuted,
  setIsMuted,
}) => {
  const tabs = [
    { id: 'monitor', label: 'Live Monitor', icon: Activity },
    { id: 'calibration', label: 'Calibration Studio', icon: Sliders },
    { id: 'telemetry', label: 'Telemetry & Charts', icon: BarChart3 },
    { id: 'benchmark', label: 'Benchmark & Evaluation', icon: FileCheck },
    { id: 'privacy', label: 'Privacy & Architecture', icon: Lock },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight text-white">SaferShift</span>
                <span className="text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                  Edge AI
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Personalized Driver Safety Intelligence
              </p>
            </div>
          </div>

          {/* Mode Switcher & Controls */}
          <div className="flex items-center space-x-3">
            {/* Mode Switcher Button Group */}
            <div className="bg-slate-950 p-1 rounded-xl border border-slate-800 flex items-center shadow-inner">
              <button
                onClick={() => setMode('standard')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  mode === 'standard'
                    ? 'bg-slate-800 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="Universal baseline threshold (EAR = 0.21)"
              >
                Standard Mode
              </button>
              <button
                onClick={() => activeProfile && setMode('personalized')}
                disabled={!activeProfile}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center space-x-1.5 disabled:cursor-not-allowed disabled:opacity-40 ${
                  mode === 'personalized' && activeProfile
                    ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-md shadow-cyan-600/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title={activeProfile ? 'Calibrated adaptive threshold customized for driver' : 'Create or select a driver profile first'}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Personalized Mode</span>
              </button>
            </div>

            {/* Active profile badge if in personalized mode */}
            {mode === 'personalized' && (
              <div className="hidden lg:flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-cyan-950/80 border border-cyan-800/60 text-xs">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-slate-300">
                  {activeProfile ? activeProfile.driver_name : 'Default Profile'}
                </span>
                <span className="font-mono text-cyan-400">
                  (EAR: {activeProfile ? activeProfile.adapted_ear_threshold.toFixed(3) : '0.195'})
                </span>
              </div>
            )}

            {/* Audio Mute/Unmute */}
            <button
              onClick={() => setIsMuted(!isMuted)}
              className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition-colors"
              title={isMuted ? 'Unmute safety chimes' : 'Mute safety chimes'}
            >
              {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4 text-emerald-400" />}
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 border-t border-slate-800/80 pt-1 -mb-px overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3 py-2 text-xs sm:text-sm font-medium border-b-2 transition-all whitespace-nowrap ${
                  isActive
                    ? 'border-cyan-500 text-cyan-400 bg-slate-800/40'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
