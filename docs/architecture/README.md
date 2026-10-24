# System Architecture — Lunar PSR Terminal

```mermaid
graph TD
    Client[React 19 Three.js Frontend] -->|HTTP /pipeline| API[FastAPI Backend]
    API -->|Raw Image Patch| Preproc[Preprocessing & Normalization]
    Preproc -->|Tensor| UNet[LunarUNet PyTorch Model]
    UNet -->|Restored Image| Postproc[Dynamic Range Scaling]
    Postproc -->|Restored Array| Hazard[V2 Multiscale Hazard Engine]
    Hazard -->|Hazard & Safe Masks| LZ[Landing Zone Candidate Extractor]
    LZ -->|Ranked Candidates| Telecom[Pipeline Telemetry & Response]
    Telecom --> Client
    Telecom -->|Optional Webhook| n8n[n8n Workflow Service]
    n8n -->|Async Render| Report[ReportLab PDF / HTML Engine]
    Report --> Storage[(SQLite & Filesystem Storage)]
```

## Modular Components
1. **Frontend (`frontend/`)**: Single-page application using Zustand for mission state and Three.js / React Three Fiber for 3D orbital-to-surface visualization.
2. **Backend (`backend/app/`)**: REST API exposing health endpoints, image inference, hazard masking, and report generation.
3. **ML Pipeline (`ml/src/`)**: PyTorch UNet implementation, loss functions, and evaluation metrics.
4. **Data Layer (`data/`)**: PDS4 raw calibration raster storage and high-resolution/low-resolution training pairs.
5. **Report Service (`backend/app/services/reports/`)**: SQLite durable state engine for background PDF/HTML export.
