# Lunar PSR Terminal

**Lunar Permanently Shadowed Region (PSR) Terminal** is an advanced mission planning, deep neural image restoration, hazard assessment, and 3D terrain visualization system for lunar surface exploration.

---

## 1. What Lunar PSR Terminal Does
In the Moon's polar craters, permanently shadowed regions (PSRs) never receive direct sunlight. Optical cameras capturing these regions must rely on faint secondary scattering from surrounding crater rims, yielding extremely dark, noisy, low-signal imagery. 

The Lunar PSR Terminal ingests orbital imagery (including calibrated Chandrayaan-2 OHRC data), enhances signal-to-noise ratio and spatial fidelity using a custom deep neural network (`LunarUNet`), detects surface hazards (craters, boulders, steep slopes) using multiscale computer vision, identifies and scores candidate landing zones (LZs), renders interactive 3D terrain for mission inspection, and generates exportable mission intelligence reports.

---

## 2. Main Features
- **Deep Neural Restoration (`LunarUNet`)**: 8.5M parameter Residual U-Net trained on paired high-resolution and synthetic degraded lunar patches using combined L1, SSIM, and Sobel edge losses. Supports both full-image single-pass and seamless 2D linear-blended tiled inference.
- **Multiscale Hazard Engine (V2)**: Dual-kernel Sobel gradient analysis, morphological dilation, and connected component filtering with three operational safety profiles:
  - `balanced`: Optimal F1-score balance between precision and recall.
  - `conservative`: High recall mode maximizing detection of hazards for safety-critical landings.
  - `precision`: High precision mode minimizing false alarms for aggressive touchdown scenarios.
- **Landing Zone (LZ) Candidate Ranking**: Automatic extraction and ranking of contiguous safe touchdown regions evaluated on surface roughness, obstacle clearance radius, and slope variance.
- **Interactive 3D Surface Visualization**: Hardware-accelerated WebGL / Three.js viewport featuring:
  - Real-time terrain mesh generation with normal mapping and height displacement.
  - Lunar lander touchdown animation with thruster plume simulation.
  - Interactive lunar rover navigation with wheel physics.
  - Astronaut extravehicular activity (EVA) inspection mode.
  - Real-time video recording controller.
- **Automated Mission Intelligence Reports**: Durable, SQLite-backed asynchronous PDF and standalone HTML report generation with embedded base64 maps and telemetry.
- **Workflow Automation (n8n)**: Out-of-the-box webhook and worker orchestration templates.

---

## 3. Technology Stack
- **Frontend**: React 19, Three.js, React Three Fiber (`@react-three/fiber`), Drei (`@react-three/drei`), Zustand, Lucide React, Vite 8.
- **Backend**: FastAPI, Uvicorn, PyTorch (MPS / CUDA accelerated), OpenCV, NumPy, SciPy, ReportLab, Pydantic V2.
- **Machine Learning**: Custom PyTorch Residual U-Net, torchvision, scikit-image, scikit-learn.
- **Data Standards**: ISRO Chandrayaan-2 OHRC (PDS4 raster and XML label).

---

## 4. Repository Directory Structure
```text
lunarpsr/
├── frontend/                     # React 19 + Three.js application
│   ├── public/                   # 3D models (.glb), textures, presets, draco workers
│   ├── src/
│   │   ├── components/           # UI components organized by domain
│   │   │   ├── layout/           # MissionBar, LeftPanel, RightPanel, BottomBar, HUD
│   │   │   ├── analysis/         # AnalysisViewport, AnalysisReport, ImageComparison
│   │   │   ├── terrain/          # TerrainViewport, TerrainInspector, 3D scenes
│   │   │   ├── landing/          # LandingZoneCard, DescentSimulation
│   │   │   ├── common/           # ErrorBoundary, RecordingController
│   │   │   └── index.js          # Barrel component exports
│   │   ├── assets/               # 3D loaders and mesh definitions
│   │   ├── services/             # API client (api.js)
│   │   ├── state/                # Zustand state store (store.js)
│   │   ├── types/                # Protocol and telemetry types (types.js)
│   │   └── styles/               # Design system stylesheets (App.css, index.css)
│   ├── tests/                    # Vitest component test suites
│   ├── package.json
│   └── vite.config.js
├── backend/                      # FastAPI backend application
│   ├── app/
│   │   ├── main.py               # Main application entry point & routing
│   │   ├── api/                  # Route modules (report_routes.py)
│   │   ├── services/             # Core business logic
│   │   │   ├── restoration.py    # UNet single-pass & tiled inference
│   │   │   ├── hazard_detection.py # Multiscale hazard engine
│   │   │   ├── landing_zones.py  # LZ extraction & scoring
│   │   │   ├── report_service.py # SQLite job queue & state management
│   │   │   └── report_rendering.py # ReportLab PDF & HTML composition
│   │   ├── schemas/              # Pydantic request/response schemas
│   │   ├── core/                 # Configuration and device selection
│   │   └── utils/                # Image and cryptographic utilities
│   ├── tests/                    # Backend unit and integration tests
│   ├── requirements.txt          # Core backend dependencies
│   └── requirements-dev.txt      # Development dependencies
├── ml/                           # Machine learning research and pipelines
│   ├── src/                      # Model definitions, dataset loaders, loss functions
│   ├── training/                 # Model training scripts (train.py, train_v2.py)
│   ├── evaluation/               # Model evaluation scripts (evaluate_v2_final.py)
│   └── experiments/              # Hazard parameter sweeps and false negative analyses
├── data/                         # Datasets and imagery
│   ├── raw/                      # Raw PDS4 Chandrayaan-2 OHRC raster & XML
│   └── processed_pairs/          # 3,000 paired HR/LR patches & split.json
├── weights/                      # Trained model checkpoints
│   ├── lunar_unet_v2_best.pth    # Production checkpoint (Epoch 57, val_loss 0.1146)
│   └── training_history_v2.json  # Loss and metric training logs
├── evaluation_results/           # Benchmark tables, PR curves, and visual analysis
├── workflows/
│   └── n8n/                      # n8n report generation workflow JSON
├── docs/                         # Architecture, API, and setup documentation
├── scripts/                      # Hardware check and smoke test scripts
├── storage/                      # SQLite database storage (reports.sqlite3)
├── .gitignore
├── .dockerignore
└── README.md
```

---

## 5. Frontend Setup
```bash
cd frontend
npm install
```

---

## 6. Backend Setup
```bash
# Create and activate virtual environment (if not already active)
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r backend/requirements.txt

# (Optional) Install development dependencies for report verification
pip install -r backend/requirements-dev.txt

# Verify hardware acceleration (MPS/CUDA) and dependencies
python scripts/check_deps.py
```

---

## 7. Machine Learning Model Location
The trained model checkpoints are located in the `weights/` directory:
- `weights/lunar_unet_v2_best.pth` (34.3 MB): Production model loaded by the backend.
- `weights/lunar_unet_v2_final.pth`: Final epoch weights.
- `weights/lunar_unet_best.pth`: Baseline model weights.
- `weights/training_history_v2.json`: Training history metadata.

---

## 8. Dataset Location
Datasets are stored in the `data/` directory:
- `data/raw/`: Raw Chandrayaan-2 OHRC image (`ch2_ohr_ncp_20260102T1224107393_d_img_d18.img`, 1.21 GB) and XML label.
- `data/processed_pairs/`: 1,500 LR and 1,500 HR 256x256 image patches.
  - `data/processed_pairs/split.json`: Partitions into 900 training, 300 validation, and 300 held-out test patches.

---

## 9. How to Start the Application
### Option A: Start Both Services (Standard Workflow)
**Terminal 1 — Backend:**
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
*(Or the backward-compatible shortcut: `uvicorn api.main:app --host 127.0.0.1 --port 8000`)*

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 10. How to Run Tests
### Backend Tests (29 tests)
```bash
./venv/bin/python -m unittest discover -s tests
```
### Frontend Tests (6 tests)
```bash
cd frontend
npm test
```
### End-to-End Pipeline Smoke Test
```bash
./venv/bin/python scripts/verify_report_pipeline.py frontend/public/presets/crater_raw.png
```

---

## 11. How to Run Model Evaluation
Run the complete multi-metric benchmark against the 300 held-out test patches:
```bash
./venv/bin/python ml/evaluation/evaluate_v2_final.py
```
To run the full report generator benchmarking suite:
```bash
./venv/bin/python scripts/evaluate_lunar_unet_complete.py
```

---

## 12. API Documentation
When the backend is running, interactive API documentation is automatically accessible at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

Key endpoints include:
- `GET /health`: Health and hardware status.
- `GET /version`: System software version.
- `POST /pipeline`: Full restoration, hazard detection, and LZ ranking pipeline.
- `POST /enhance`: Image restoration only.
- `POST /reports/generate`: Asynchronous mission report generation.

---

## 13. How to Configure Environment Variables
Copy `.env.example` to `.env` in the repository root:
```bash
cp .env.example .env
```
Configuration variables:
```ini
# Optional Google API key for Gemini chat assistant and Veo landing simulation
GEMINI_API_KEY=your_key_here

# Directory for SQLite reports database and artifact storage (defaults to storage/reports)
REPORT_STORAGE_DIR=storage/reports

# Secret token required for n8n or internal orchestration endpoints
REPORT_INTERNAL_TOKEN=your_internal_secret_here

# (Optional) Webhook URL for external n8n workflow triggers
N8N_REPORT_WEBHOOK_URL=http://localhost:5678/webhook/lunar-report
```

---

## 14. How to Run n8n Workflows
1. Start your local n8n instance:
   ```bash
   npx n8n
   ```
2. In the n8n UI, navigate to **Workflows → Import from File** and select `workflows/n8n/lunar-analysis-report.n8n.json`.
3. Set the `REPORT_INTERNAL_TOKEN` header authentication credential matching the value in your backend `.env`.
4. Point your backend's `N8N_REPORT_WEBHOOK_URL` in `.env` to the webhook URL displayed in n8n.

---

## 15. Known Scientific Limitations
1. **Photometric Calibration**: The current deep restoration model is trained on simulated degradation models applied to real Chandrayaan-2 OHRC patches. While effective at high-frequency signal recovery, absolute radiometric calibration across extreme dynamic ranges requires in-situ cross-calibration.
2. **Monocular Slope Estimation**: Slope angles and boulder heights are inferred via shadow photometric properties and multiscale gradient analysis. True 3D topological verification requires stereoscopic imagery or LiDAR digital elevation models (DEMs).
3. **Safety Certification**: Candidate Landing Zone scores are algorithmic assessments based on 2D resolution limitations. They do not constitute certified flight guidance without cross-referencing multi-sensor radar and laser altimetry datasets.
