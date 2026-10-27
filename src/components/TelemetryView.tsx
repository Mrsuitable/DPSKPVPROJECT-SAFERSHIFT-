import React, { useState } from 'react';
import { Download, FileText, BarChart2, ShieldAlert, Clock, ArrowDown } from 'lucide-react';
import { TelemetryPoint, AlertEvent } from '../types';

interface TelemetryViewProps {
  telemetryHistory: TelemetryPoint[];
  alertHistory: AlertEvent[];
  mode: string;
}

export const TelemetryView: React.FC<TelemetryViewProps> = ({
  telemetryHistory,
  alertHistory,
  mode,
}) => {
  const [filterCategory, setFilterCategory] = useState<string>('all');

  const downloadCSV = () => {
    if (telemetryHistory.length === 0) return;

    const headers = [
      'timestamp',
      'timestamp_iso',
      'event_type',
      'ear',
      'mar',
      'yaw',
      'pitch',
      'roll',
      'closure_duration',
      'distraction_duration',
      'yawn_duration',
      'blink_rate_bpm',
      'total_blinks',
      'fps',
      'warning_reason',
    ];

    const rows = telemetryHistory.map((p) => [
      p.timestamp,
      new Date(p.timestamp).toISOString(),
      p.eventType,
      p.ear.toFixed(4),
      p.mar.toFixed(4),
      p.yaw.toFixed(2),
      p.pitch.toFixed(2),
      p.roll.toFixed(2),
      p.closureDuration.toFixed(2),
      p.distractionDuration.toFixed(2),
      p.yawnDuration.toFixed(2),
      p.blinkRateBpm.toFixed(1),
      p.totalBlinks,
      p.fps.toFixed(1),
      `"${(p.warningReason || '').replace(/"/g, '""')}"`,
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `safer_shift_telemetry_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredAlerts =
    filterCategory === 'all'
      ? alertHistory
      : alertHistory.filter((a) => a.category === filterCategory);

  // SVG mini sparkline generator
  const renderSparkline = (
    data: number[],
    color: string,
    threshold?: number,
    minVal: number = 0,
    maxVal: number = 1
  ) => {
    if (data.length < 2) return null;
    const width = 600;
    const height = 120;
    const padding = 10;

    const pts = data.slice(-50);
    const step = (width - padding * 2) / (pts.length - 1);

    const points = pts
      .map((val, idx) => {
        const x = padding + idx * step;
        const norm = (val - minVal) / (maxVal - minVal);
        const y = height - padding - norm * (height - padding * 2);
        return `${x},${Math.max(padding, Math.min(height - padding, y))}`;
      })
      .join(' ');

    const threshY =
      threshold !== undefined
        ? height - padding - ((threshold - minVal) / (maxVal - minVal)) * (height - padding * 2)
        : null;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
        {/* Threshold dashed line */}
        {threshY !== null && (
          <line
            x1={padding}
            y1={threshY}
            x2={width - padding}
            y2={threshY}
            stroke="#ef4444"
            strokeDasharray="4 4"
            strokeWidth="1.5"
          />
        )}
        {/* Trend Polyline */}
        <polyline fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" points={points} />
      </svg>
    );
  };

  const earSeries = telemetryHistory.map((p) => p.ear);
  const marSeries = telemetryHistory.map((p) => p.mar);
  const yawSeries = telemetryHistory.map((p) => p.yaw);

  return (
    <div className="space-y-8">
      {/* Header and Export actions */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <BarChart2 className="w-5 h-5 text-cyan-400" />
            <span>Driver Telemetry & Event Analytics</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time biometric feature streaming and explainable warning logs
          </p>
        </div>

        <button
          onClick={downloadCSV}
          disabled={telemetryHistory.length === 0}
          className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold shadow-lg shadow-cyan-600/30 transition-all disabled:opacity-50"
        >
          <Download className="w-4 h-4" />
          <span>Export Local Telemetry (CSV)</span>
        </button>
      </div>

      {/* Time-Series Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* EAR Trend */}
        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-white">Eye Aspect Ratio (EAR) Trend</span>
            <span className="text-xs font-mono text-cyan-400">
              Latest: {telemetryHistory.length ? telemetryHistory[telemetryHistory.length - 1].ear.toFixed(3) : '0.000'}
            </span>
          </div>
          <div className="h-32 bg-slate-950 rounded-xl p-3 border border-slate-800 flex items-center">
            {renderSparkline(earSeries, '#38bdf8', 0.21, 0.0, 0.4)}
          </div>
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Dashed Red Line: Closure Threshold (0.21)</span>
            <span>Last 50 Samples</span>
          </div>
        </div>

        {/* MAR Trend */}
        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-white">Mouth Aspect Ratio (MAR) Trend</span>
            <span className="text-xs font-mono text-amber-400">
              Latest: {telemetryHistory.length ? telemetryHistory[telemetryHistory.length - 1].mar.toFixed(3) : '0.000'}
            </span>
          </div>
          <div className="h-32 bg-slate-950 rounded-xl p-3 border border-slate-800 flex items-center">
            {renderSparkline(marSeries, '#f59e0b', 0.55, 0.0, 0.8)}
          </div>
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Dashed Red Line: Yawn Threshold (0.55)</span>
            <span>Last 50 Samples</span>
          </div>
        </div>
      </div>

      {/* Explainable Alert History Table */}
      <div className="bg-slate-900/90 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-lg font-bold text-white flex items-center space-x-2">
              <ShieldAlert className="w-5 h-5 text-rose-400" />
              <span>Explainable Safety Alert History</span>
            </h3>
            <p className="text-xs text-slate-400">
              Auditable event records documenting the exact triggering evidence
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400">Filter:</span>
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-slate-300 focus:outline-none focus:border-cyan-500"
            >
              <option value="all">All Alerts</option>
              <option value="prolonged_eye_closure">Prolonged Eye Closure</option>
              <option value="sustained_distraction">Sustained Distraction</option>
              <option value="possible_yawning">Possible Yawning</option>
            </select>
          </div>
        </div>

        {filteredAlerts.length === 0 ? (
          <div className="text-center py-12 text-slate-500 text-xs">
            No safety alerts recorded in current session. Drive safely!
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Severity</th>
                  <th className="py-2.5 px-3">Event Category</th>
                  <th className="py-2.5 px-3">Duration</th>
                  <th className="py-2.5 px-3">Mode</th>
                  <th className="py-2.5 px-3">Trigger Measurement</th>
                  <th className="py-2.5 px-3">Explanation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {filteredAlerts.slice(-15).reverse().map((al) => (
                  <tr key={al.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2.5 px-3 text-slate-400">
                      {new Date(al.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          al.severity === 'critical'
                            ? 'bg-rose-950 text-rose-400 border border-rose-800'
                            : 'bg-amber-950 text-amber-400 border border-amber-800'
                        }`}
                      >
                        {al.severity}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-white">
                      {al.category.replace(/_/g, ' ')}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {al.duration.toFixed(2)}s
                    </td>
                    <td className="py-2.5 px-3 uppercase text-[11px] text-cyan-400">
                      {al.mode}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {al.triggerMetric} = {al.triggerValue.toFixed(3)} (th: {al.thresholdValue})
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 font-sans max-w-xs truncate" title={al.explanation}>
                      {al.explanation}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
