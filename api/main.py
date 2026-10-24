"""
Backward compatibility wrapper for Lunar PSR Terminal API.
Points to the canonical implementation in backend.app.main.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app.main import *
from backend.app.main import (
    app,
    report_service,
    SOFTWARE_VERSION,
    device,
    model,
    pipeline,
    enhance_image,
    analyze_image,
    decode_uploaded_image,
    compute_hazard_mask,
    detect_landing_zones,
)
