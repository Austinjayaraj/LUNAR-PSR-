# API Reference — Lunar PSR Terminal

The FastAPI backend exposes the following endpoints on port `8000`:

## Health & System Status
- `GET /health`: Basic health check returning status, device type (`mps` or `cpu`), and model version.
- `GET /ready`: Readiness probe confirming model checkpoint is loaded in memory.
- `GET /version`: Current software and checkpoint build metadata.
- `GET /model-status`: Detailed PyTorch parameter count, device, and active checkpoint path.
- `GET /system-status`: End-to-end subsystem check (GPU, inference engine, report storage).

## Inference & Pipeline
- `POST /pipeline`:
  - **Input**: Multipart form with `file` (PNG/JPEG lunar image), optional `hazard_mode` (`balanced`, `conservative`, `precision`), and `tile_inference` flag.
  - **Output**: JSON payload with `input_info`, `restoration` (base64 image, PSNR, SSIM), `hazard_analysis` (danger map, safe region map, hazard count), `landing_zones` (ranked list of LZ candidates), and `processing_timeline`.
- `POST /enhance`: Lightweight restoration-only endpoint returning restored base64 image.
- `POST /analyze`: Hazard detection only on an input image.
- `POST /chat`: Mission assistant conversational endpoint using Gemini.
- `POST /generate-landing-video`: Generates landing simulation video via Google Veo.

## Report Management
- `POST /reports/generate`: Enqueues an asynchronous PDF/HTML report generation job.
- `GET /reports/{report_id}?access_token={token}`: Retrieves report generation status (`PENDING`, `READY`, `FAILED`).
- `GET /reports/{report_id}/download?access_token={token}`: Downloads finalized mission PDF report.
- `GET /reports/{report_id}/preview?access_token={token}`: Returns standalone rendered HTML report.
