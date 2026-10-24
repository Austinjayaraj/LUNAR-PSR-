# Lunar PSR Terminal — Project Overview

## Overview
The **Lunar Permanently Shadowed Region (PSR) Terminal** is an end-to-end mission planning, deep restoration, and hazard assessment platform for lunar exploration. It transforms extremely low-signal-to-noise, low-light orbital imagery from ISRO's Chandrayaan-2 Orbiter High Resolution Camera (OHRC) into restored, high-resolution terrain representations, identifies surface hazards (craters, boulders, steep slopes), ranks candidate landing zones, and provides interactive 3D surface visualization.

## Key Capabilities
1. **LunarUNet Deep Restoration**:
   - Neural super-resolution and contrast enhancement specifically tuned for extreme shadow dynamics and photometric sensor noise.
   - 8.5M parameter custom Residual U-Net running on Apple Silicon MPS or CUDA.
2. **Multiscale Hazard Engine (V2)**:
   - Multiscale Sobel gradient analysis, morphological operations, and connected component filtering.
   - Three operational safety profiles: `balanced`, `conservative`, and `precision`.
3. **Automated Landing Zone (LZ) Ranking**:
   - Extraction of safe touchdown basins with clearance metrics, slope variance, and composite safety scoring.
4. **3D Interactive Terrain & Mission Viewport**:
   - WebGL / Three.js surface rendering with real-time normal maps, height displacement, lander touchdown simulation, rover navigation, and EVA inspection.
5. **Mission Reporting & n8n Orchestration**:
   - Selection-aware PDF and HTML report generation with embedded visual maps and metrics telemetry.
   - SQLite job persistence and webhook integration with n8n workflow engines.

## Tech Stack
- **Frontend**: React 19, Vite, Three.js, React Three Fiber, Lucide Icons, Zustand.
- **Backend**: FastAPI, Uvicorn, PyTorch, OpenCV, NumPy, SciPy, ReportLab.
- **Hardware Acceleration**: Apple Silicon MPS (Metal Performance Shaders) & CUDA.
- **Data**: ISRO Chandrayaan-2 OHRC (PDS4 standard).
