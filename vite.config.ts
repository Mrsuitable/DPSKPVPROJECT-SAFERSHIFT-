import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'path';
import {defineConfig} from 'vite';

const projectDirectory = import.meta.dirname;

function copyLocalFaceMeshAssets() {
  const packageDirectory = path.join(projectDirectory, 'node_modules', '@mediapipe', 'face_mesh');
  const publicDirectory = path.join(projectDirectory, 'public', 'mediapipe', 'face_mesh');
  if (!fs.existsSync(packageDirectory)) {
    throw new Error('MediaPipe FaceMesh is missing. Run npm install before starting or building SaferShift.');
  }

  fs.mkdirSync(publicDirectory, { recursive: true });
  for (const file of fs.readdirSync(packageDirectory)) {
    if (/^face_mesh.*\.(js|wasm|data|binarypb)$/.test(file)) {
      fs.copyFileSync(path.join(packageDirectory, file), path.join(publicDirectory, file));
    }
  }
}

export default defineConfig(() => {
  return {
    plugins: [
      {
        name: 'safer-shift-local-face-mesh-assets',
        configResolved: copyLocalFaceMeshAssets,
      },
      react(),
      tailwindcss(),
    ],
    resolve: {
      alias: {
        '@': path.resolve(projectDirectory, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
