import { rmSync } from 'node:fs';
import path from 'node:path';

const projectDirectory = path.resolve(import.meta.dirname, '..');
for (const generatedPath of ['dist', 'server.js']) {
  rmSync(path.join(projectDirectory, generatedPath), { recursive: true, force: true });
}
