import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Dev server proxies API calls to the FastAPI backend on :8001.
// Production builds are served by FastAPI itself (same origin), so no proxy.
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
