# Backend API & Processing Pipeline — Lunar PSR Terminal

This directory houses the FastAPI production backend, deep neural network inference engine, hazard detection algorithms, landing zone candidate extraction, and PDF/HTML report generators.

## Architecture
- `app/`:
  - `main.py`: Primary FastAPI application entry point, CORS configuration, lifecycle hooks, and endpoint definitions.
  - `api/`: API subrouters (such as `report_routes.py` for asynchronous report jobs).
  - `services/`:
    - `restoration.py`: Tiled and single-pass neural network inference using `LunarUNet`.
    - `hazard_detection.py`: Multiscale edge and morphological hazard mask generators (`balanced`, `conservative`, `precision`).
    - `landing_zones.py`: Connected-component safe-region extraction and candidate ranking algorithms.
    - `report_service.py`: Durable SQLite job queue for asynchronous report creation.
    - `report_rendering.py`: Pure-Python ReportLab PDF and embedded HTML generator.
  - `schemas/`: Pydantic V2 data models for input validation and typed API responses.
  - `core/`: Environment settings, hardware acceleration detection (`mps` / `cuda`), and model configuration.
  - `utils/`: Image encoding, base64 formatting, hash calculations, and dimension validation.
- `tests/`: Backend test suites for unit and integration testing.
- `requirements.txt`: Python package requirements.
- `requirements-dev.txt`: Development dependencies.

## Running the Server
```bash
# Recommended command
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Legacy / backward-compatible command
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
API documentation is automatically available at `http://127.0.0.1:8000/docs`.
