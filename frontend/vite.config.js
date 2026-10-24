import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/version': 'http://127.0.0.1:8000',
      '/system-status': 'http://127.0.0.1:8000',
      '/model-status': 'http://127.0.0.1:8000',
      '/ready': 'http://127.0.0.1:8000',
      '/pipeline': 'http://127.0.0.1:8000',
      '/enhance': 'http://127.0.0.1:8000',
      '/analyze': 'http://127.0.0.1:8000',
      '/evaluation': 'http://127.0.0.1:8000',
      '/reports': 'http://127.0.0.1:8000',
      '/chat': 'http://127.0.0.1:8000',
      '/generate-landing-video': 'http://127.0.0.1:8000',
    },
  },
})
