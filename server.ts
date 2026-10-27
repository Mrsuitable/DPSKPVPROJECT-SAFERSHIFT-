import express from 'express';
import { spawn } from 'child_process';
import { createServer as createViteServer } from 'vite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const PROJECT_DIR = path.dirname(fileURLToPath(import.meta.url));
const BASE_DIR = path.join(PROJECT_DIR, 'safer-shift');
const PROFILES_DIR = path.join(BASE_DIR, 'data', 'profiles');
const LOGS_DIR = path.join(BASE_DIR, 'data', 'logs');
const REPORTS_DIR = path.join(BASE_DIR, 'reports');

interface PythonRunResult {
  exitCode: number;
  stdout: string;
  stderr: string;
}

function runPython(args: string[]): Promise<PythonRunResult> {
  const configuredPython = process.env.SAFERSHIFT_PYTHON;
  const executable = configuredPython || (process.platform === 'win32' ? 'py' : 'python3');
  const pythonArgs = process.platform === 'win32' && !configuredPython ? ['-3', ...args] : args;

  return new Promise((resolve) => {
    const child = spawn(executable, pythonArgs, {
      cwd: BASE_DIR,
      env: {
        ...process.env,
        PYTHONPATH: [BASE_DIR, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
        PYTHONIOENCODING: 'utf-8',
      },
      windowsHide: true,
    });
    let stdout = '';
    let stderr = '';

    child.stdout.setEncoding('utf-8').on('data', (chunk: string) => { stdout += chunk; });
    child.stderr.setEncoding('utf-8').on('data', (chunk: string) => { stderr += chunk; });
    child.on('error', (error) => {
      resolve({ exitCode: 1, stdout, stderr: `${stderr}${error.message}` });
    });
    child.on('close', (code) => {
      resolve({ exitCode: code ?? 1, stdout, stderr });
    });
  });
}

function readBenchmarkReport() {
  const jsonPath = path.join(REPORTS_DIR, 'benchmark_report.json');
  return JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  [PROFILES_DIR, LOGS_DIR, REPORTS_DIR].forEach((directory) => {
    fs.mkdirSync(directory, { recursive: true });
  });

  app.get('/api/benchmark', async (_req, res) => {
    const jsonPath = path.join(REPORTS_DIR, 'benchmark_report.json');
    if (fs.existsSync(jsonPath)) {
      try {
        return res.json({ success: true, report: readBenchmarkReport() });
      } catch (error) {
        return res.status(500).json({
          success: false,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    const result = await runPython(['evaluation/evaluate.py']);
    if (result.exitCode !== 0) {
      return res.status(500).json({
        success: false,
        error: result.stderr || 'Benchmark failed.',
        stdout: result.stdout,
      });
    }
    try {
      return res.json({ success: true, report: readBenchmarkReport(), stdout: result.stdout });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });

  app.post('/api/benchmark/run', async (_req, res) => {
    const result = await runPython(['evaluation/evaluate.py']);
    if (result.exitCode !== 0) {
      return res.status(500).json({
        success: false,
        error: result.stderr || 'Benchmark failed.',
        stdout: result.stdout,
      });
    }
    try {
      return res.json({ success: true, report: readBenchmarkReport(), stdout: result.stdout });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : String(error),
        stdout: result.stdout,
      });
    }
  });

  app.post('/api/tests/run', async (_req, res) => {
    const result = await runPython(['-m', 'pytest', 'tests', '-v']);
    return res.status(result.exitCode === 0 ? 200 : 500).json({
      success: result.exitCode === 0,
      exitCode: result.exitCode,
      output: [result.stdout, result.stderr].filter(Boolean).join('\n'),
    });
  });

  app.get('/api/profiles', (_req, res) => {
    try {
      const files = fs.readdirSync(PROFILES_DIR).filter((file) => file.endsWith('.json'));
      const profiles = files
        .map((file) => {
          try {
            return JSON.parse(fs.readFileSync(path.join(PROFILES_DIR, file), 'utf-8'));
          } catch {
            return null;
          }
        })
        .filter(Boolean);
      return res.json({ success: true, profiles });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });

  app.post('/api/profiles', (req, res) => {
    try {
      const profile = req.body;
      if (!profile || typeof profile.profile_id !== 'string' || !profile.profile_id.trim()) {
        return res.status(400).json({ success: false, error: 'Invalid profile data' });
      }
      const safeId = profile.profile_id.trim().replace(/[^a-zA-Z0-9_-]/g, '_');
      profile.profile_id = safeId;
      fs.writeFileSync(path.join(PROFILES_DIR, `${safeId}.json`), JSON.stringify(profile, null, 2), 'utf-8');
      return res.json({ success: true, profile });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });

  app.delete('/api/profiles/:id', (req, res) => {
    try {
      const safeId = req.params.id.replace(/[^a-zA-Z0-9_-]/g, '_');
      const filePath = path.join(PROFILES_DIR, `${safeId}.json`);
      if (!fs.existsSync(filePath)) {
        return res.json({ success: true, deleted: false });
      }
      fs.unlinkSync(filePath);
      return res.json({ success: true, deleted: true });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });

  app.get('/api/logs', (_req, res) => {
    try {
      const files = fs.readdirSync(LOGS_DIR).filter((file) => file.endsWith('.csv')).sort().reverse();
      const latestContent = files.length > 0
        ? fs.readFileSync(path.join(LOGS_DIR, files[0]), 'utf-8')
        : '';
      return res.json({ success: true, files, latestContent });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });

  app.post('/api/logs/record', (req, res) => {
    try {
      const record = req.body;
      if (!record || typeof record !== 'object' || Array.isArray(record)) {
        return res.status(400).json({ success: false, error: 'Invalid telemetry record' });
      }

      const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      const filePath = path.join(LOGS_DIR, `telemetry_web_${dateStr}.csv`);
      const headers = [
        'timestamp', 'timestamp_iso', 'event_type', 'mode', 'ear', 'mar', 'yaw', 'pitch', 'roll',
        'closure_duration', 'distraction_duration', 'yawn_duration', 'blink_rate_bpm', 'total_blinks',
        'calibration_status', 'warning_reason', 'tracking_quality', 'measured_fps',
      ];

      if (!fs.existsSync(filePath)) {
        fs.writeFileSync(filePath, `${headers.join(',')}\n`, 'utf-8');
      }

      const row = headers
        .map((header) => {
          const value = record[header] ?? '';
          return `"${String(value).replace(/"/g, '""')}"`;
        })
        .join(',');
      fs.appendFileSync(filePath, `${row}\n`, 'utf-8');
      return res.json({ success: true });
    } catch (error) {
      return res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });

  const vite = await createViteServer({
    root: PROJECT_DIR,
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);

  const port = Number(process.env.PORT || 3000);
  const host = process.env.HOST || '127.0.0.1';
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`Invalid PORT value: ${process.env.PORT}`);
  }
  app.listen(port, host, () => {
    console.log(`SaferShift Full-Stack Server active on http://${host}:${port}`);
  });
}

startServer().catch((error) => {
  console.error('Failed to start server:', error);
  process.exit(1);
});
