# Operational & Diagnostic Scripts — Lunar PSR Terminal

This directory contains standalone execution, diagnostic, and smoke-testing utilities.

## Scripts
- `check_deps.py`: Deep diagnostic script checking Python dependencies and hardware acceleration (Apple Silicon MPS tensor operations).
- `verify_report_pipeline.py`: Comprehensive smoke test validating:
  - Real end-to-end local inference (restoration + hazard detection + landing zone ranking).
  - SQLite persistence and PDF/HTML report generation.
  - Failure isolation and error handling.
- `evaluate_lunar_unet_complete.py`: Comprehensive benchmark suite evaluating LunarUNet against 300 held-out test patches. Generates markdown and CSV metric tables.

## Execution
Run from project root:
```bash
# Check dependencies and MPS hardware acceleration
./venv/bin/python scripts/check_deps.py

# Run end-to-end pipeline smoke test
./venv/bin/python scripts/verify_report_pipeline.py frontend/public/presets/crater_raw.png
```
