import React, { useState, useEffect } from 'react';
import {
  FileCheck,
  CheckCircle2,
  TrendingDown,
  Play,
  RotateCcw,
  ShieldCheck,
  AlertCircle,
  Terminal,
  Cpu,
} from 'lucide-react';
import { BenchmarkReport } from '../types';

export const BenchmarkView: React.FC = () => {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [testOutput, setTestOutput] = useState<string | null>(null);
  const [testsRunning, setTestsRunning] = useState<boolean>(false);
  const [testsSuccess, setTestsSuccess] = useState<boolean | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    fetchBenchmark();
  }, []);

  const fetchBenchmark = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await fetch('/api/benchmark');
      const data = await res.json();
      if (!res.ok || !data.success || !data.report) {
        throw new Error(data.error || 'The benchmark report is unavailable.');
      }
      setReport(data.report);
    } catch (error) {
      setReport(null);
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  };

  const runBenchmarkNow = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await fetch('/api/benchmark/run', { method: 'POST' });
      const data = await res.json();
      if (!res.ok || !data.success || !data.report) {
        throw new Error(data.error || 'Benchmark execution failed.');
      }
      setReport(data.report);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  };

  const runPytestsNow = async () => {
    setTestsRunning(true);
    setTestOutput('Executing pytest test suite on edge engine...');
    try {
      const res = await fetch('/api/tests/run', { method: 'POST' });
      const data = await res.json();
      setTestsSuccess(data.success);
      setTestOutput(data.output || 'Tests completed.');
    } catch (err: any) {
      setTestsSuccess(false);
      setTestOutput(`Execution error: ${err.message}`);
    } finally {
      setTestsRunning(false);
    }
  };

  const std = report?.standard;
  const pers = report?.personalized;
  const falseAlarmSummary = report
    ? report.false_alarm_reduction_pct > 0
      ? `reduced false alarms by ${report.false_alarm_reduction_pct.toFixed(1)}%`
      : report.false_alarm_reduction_pct < 0
        ? `increased false alarms by ${Math.abs(report.false_alarm_reduction_pct).toFixed(1)}%`
        : 'left false alarms unchanged'
    : null;

  return (
    <div className="space-y-8">
      {/* Research Question & Findings Banner */}
      <div className="bg-gradient-to-r from-slate-900 to-indigo-950 p-6 rounded-3xl border border-indigo-800/40 shadow-2xl">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="max-w-3xl">
            <div className="flex items-center space-x-2 text-indigo-400 text-xs font-bold uppercase tracking-wider mb-2">
              <FileCheck className="w-4 h-4" />
              <span>Comparative Benchmark & Hypothesis Evaluation</span>
            </div>
            <h2 className="text-2xl font-black text-white tracking-tight">
              Can Personalized Calibration Reduce False Alarms?
            </h2>
            <p className="mt-2 text-sm text-slate-300 leading-relaxed">
              {report ? (
                <>
                  In the current annotated scenarios, personalized calibration {falseAlarmSummary} and achieved{' '}
                  <strong> {(report.personalized.recall * 100).toFixed(1)}% recall</strong>.
                </>
              ) : 'Load or run the benchmark to see results from the current evaluation scenarios.'}
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={runBenchmarkNow}
              disabled={loading}
              className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold shadow-lg shadow-indigo-600/30 transition-all disabled:opacity-50"
            >
              <RotateCcw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              <span>{loading ? 'Evaluating...' : 'Re-run Benchmark'}</span>
            </button>
            <button
              onClick={runPytestsNow}
              disabled={testsRunning}
              className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold border border-slate-700 transition-all disabled:opacity-50"
            >
              <Terminal className="w-4 h-4" />
              <span>{testsRunning ? 'Running Pytest...' : 'Execute Unit Tests'}</span>
            </button>
          </div>
        </div>
      </div>

      {loadError && (
        <div role="alert" className="rounded-xl border border-amber-800 bg-amber-950/50 p-4 text-xs text-amber-200">
          Benchmark unavailable: {loadError}
        </div>
      )}

      {/* Live Pytest Output Terminal */}
      {testOutput && (
        <div className="bg-slate-950 rounded-2xl border border-slate-800 p-5 shadow-2xl space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="flex items-center space-x-2 text-slate-300 font-mono font-bold">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span>Automated Test Suite Execution Log (pytest)</span>
            </span>
            {testsSuccess !== null && (
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                  testsSuccess
                    ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                    : 'bg-rose-950 text-rose-400 border border-rose-800'
                }`}
              >
                {testsSuccess ? 'TESTS PASSED' : 'TEST FAILURES DETECTED'}
              </span>
            )}
          </div>
          <pre className="text-xs font-mono text-emerald-400/90 bg-black/60 p-4 rounded-xl overflow-x-auto max-h-56">
            {testOutput}
          </pre>
        </div>
      )}

      {/* Key Metric Headline Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl">
          <div className="text-xs text-slate-400 font-medium">False Alarm Reduction</div>
          <div className="text-3xl font-black font-mono text-emerald-400 mt-1 flex items-center space-x-1.5">
            <TrendingDown className="w-6 h-6 text-emerald-400" />
            <span>{report ? `${report.false_alarm_reduction_pct > 0 ? '-' : '+'}${Math.abs(report.false_alarm_reduction_pct).toFixed(1)}%` : '—'}</span>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Suppressed nuisance false alarms</div>
        </div>

        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl">
          <div className="text-xs text-slate-400 font-medium">Critical Fatigue Recall</div>
          <div className="text-3xl font-black font-mono text-cyan-400 mt-1 flex items-center space-x-1.5">
            <CheckCircle2 className="w-6 h-6 text-cyan-400" />
            <span>{pers ? `${(pers.recall * 100).toFixed(0)}%` : '—'}</span>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Recall on annotated events</div>
        </div>

        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl">
          <div className="text-xs text-slate-400 font-medium">Personalized F1-Score</div>
          <div className="text-3xl font-black font-mono text-indigo-400 mt-1">
            {pers?.f1_score.toFixed(3)}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            vs Standard: {std?.f1_score.toFixed(3)} (+{((pers?.f1_score || 0) - (std?.f1_score || 0)).toFixed(3)})
          </div>
        </div>

        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-xl">
          <div className="text-xs text-slate-400 font-medium">Annotated Events</div>
          <div className="text-3xl font-black font-mono text-white mt-1 flex items-center space-x-1.5">
            <Cpu className="w-6 h-6 text-indigo-400" />
            <span>{report ? report.ground_truth_count : '—'}</span>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Ground-truth events in this benchmark</div>
        </div>
      </div>

      {/* Side-by-side Comparative Table */}
      <div className="bg-slate-900/90 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center space-x-2">
          <FileCheck className="w-5 h-5 text-indigo-400" />
          <span>Fair Comparison: Standard vs Personalized Detection</span>
        </h3>
        <p className="text-xs text-slate-400">
          Both modes evaluated on identical test clips using the exact same temporal engine.
          Differences are attributable strictly to adaptive baseline calibration.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                <th className="py-3 px-4">Evaluation Metric</th>
                <th className="py-3 px-4 text-center">Standard Mode (Fixed EAR: 0.21)</th>
                <th className="py-3 px-4 text-center">Personalized Mode (Adaptive)</th>
                <th className="py-3 px-4 text-right">Delta / Improvement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-semibold text-white">True Positives (TP)</td>
                <td className="py-3 px-4 text-center text-slate-300">{std?.true_positives}</td>
                <td className="py-3 px-4 text-center text-cyan-400 font-bold">{pers?.true_positives}</td>
                <td className="py-3 px-4 text-right text-slate-400">{(pers?.true_positives ?? 0) - (std?.true_positives ?? 0)}</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-semibold text-white">False Positives (FP)</td>
                <td className="py-3 px-4 text-center text-rose-400">{std?.false_positives}</td>
                <td className="py-3 px-4 text-center text-emerald-400 font-bold">{pers?.false_positives}</td>
                <td className="py-3 px-4 text-right text-emerald-400">
                  {((pers?.false_positives || 0) - (std?.false_positives || 0))} (-{report?.false_alarm_reduction_pct}%)
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-semibold text-white">False Negatives (FN)</td>
                <td className="py-3 px-4 text-center text-slate-300">{std?.false_negatives}</td>
                <td className="py-3 px-4 text-center text-emerald-400 font-bold">{pers?.false_negatives}</td>
                <td className="py-3 px-4 text-right text-slate-400">{(pers?.false_negatives ?? 0) - (std?.false_negatives ?? 0)}</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-semibold text-white">Event Precision</td>
                <td className="py-3 px-4 text-center text-slate-300">{std?.precision.toFixed(3)}</td>
                <td className="py-3 px-4 text-center text-cyan-400 font-bold">{pers?.precision.toFixed(3)}</td>
                <td className="py-3 px-4 text-right text-emerald-400">
                  +{((pers?.precision || 0) - (std?.precision || 0)).toFixed(3)}
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-semibold text-white">Event Recall</td>
                <td className="py-3 px-4 text-center text-slate-300">{std?.recall.toFixed(3)}</td>
                <td className="py-3 px-4 text-center text-cyan-400 font-bold">{pers?.recall.toFixed(3)}</td>
                <td className="py-3 px-4 text-right text-slate-400">{(pers?.recall ?? 0).toFixed(3)}</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-semibold text-white">F1-Score</td>
                <td className="py-3 px-4 text-center text-slate-300">{std?.f1_score.toFixed(3)}</td>
                <td className="py-3 px-4 text-center text-indigo-400 font-bold">{pers?.f1_score.toFixed(3)}</td>
                <td className="py-3 px-4 text-right text-emerald-400">
                  +{((pers?.f1_score || 0) - (std?.f1_score || 0)).toFixed(3)}
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-semibold text-white">False Alarms / Minute</td>
                <td className="py-3 px-4 text-center text-rose-400">{std?.false_alarms_per_minute.toFixed(2)}</td>
                <td className="py-3 px-4 text-center text-emerald-400 font-bold">{pers?.false_alarms_per_minute.toFixed(2)}</td>
                <td className="py-3 px-4 text-right text-emerald-400">
                  {((pers?.false_alarms_per_minute || 0) - (std?.false_alarms_per_minute || 0)).toFixed(2)} / min
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-semibold text-white">Mean Detection Delay</td>
                <td className="py-3 px-4 text-center text-slate-300">{std?.mean_detection_delay_sec.toFixed(2)}s</td>
                <td className="py-3 px-4 text-center text-slate-300">{pers?.mean_detection_delay_sec.toFixed(2)}s</td>
                <td className="py-3 px-4 text-right text-slate-400">{((pers?.mean_detection_delay_sec ?? 0) - (std?.mean_detection_delay_sec ?? 0)).toFixed(2)}s</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
