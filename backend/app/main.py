import io
import os
import json
import time
import base64
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

import torch
import cv2
import numpy as np
from scipy import ndimage
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

try:
    from ml.src.model import LunarUNet
except ImportError:
    from src.model import LunarUNet
try:
    from backend.app.schemas import (
        LandingZoneCandidate,
        RestorationResult,
        HazardAnalysisResult,
        ImageStats,
        ProcessingTimeline,
        ModelInfo,
        TerrainMetrics,
        InputInfo,
        PipelineResult,
        HealthResponse,
        SystemStatus,
        ErrorResponse,
        ScoreType,
        DataSource,
    )
except ImportError:
    from api.schemas import (
        LandingZoneCandidate,
        RestorationResult,
        HazardAnalysisResult,
        ImageStats,
        ProcessingTimeline,
        ModelInfo,
        TerrainMetrics,
        InputInfo,
        PipelineResult,
        HealthResponse,
        SystemStatus,
        ErrorResponse,
        ScoreType,
        DataSource,
    )

# Load environment variables from .env file if it exists
def _find_repo_root():
    cur = os.path.abspath(os.path.dirname(__file__))
    while cur != os.path.dirname(cur):
        if os.path.exists(os.path.join(cur, 'weights')) and os.path.exists(os.path.join(cur, 'evaluation_results')):
            return cur
        cur = os.path.dirname(cur)
    return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

REPO_ROOT = _find_repo_root()
dotenv_path = os.path.join(REPO_ROOT, ".env")
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0].strip()] = parts[1].strip()

app = FastAPI(title="Lunar PSR Terminal API", version="2.0.0")

try:
    from backend.app.api.report_routes import install_reports
except ImportError:
    from api.report_routes import install_reports
report_service = install_reports(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Landing-Telemetry"],
)

# Chat models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    context: Optional[dict] = None

# Constants
SOFTWARE_VERSION = "2.2.0"
# Epoch-57 checkpoint: best val_loss=0.1146 from train_v2.py on real OHRC data
MODEL_PATH = os.path.join(REPO_ROOT, "weights", "lunar_unet_v2_best.pth")
MODEL_EPOCH = 57
STARTUP_TIME = time.time()

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load model
model = LunarUNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())

# Load V2 hazard config (three operating modes)
_V2_CFG_PATH = os.path.join(REPO_ROOT, "evaluation_results", "hazard_v2_config.json")
_V2_CONFIG = None
if os.path.exists(_V2_CFG_PATH):
    with open(_V2_CFG_PATH) as _f:
        _V2_CONFIG = json.load(_f)

HAZARD_METHOD = "oracle_multiscale_balanced"
HAZARD_CONFIG = None  # kept for legacy fallback path


def _hazard_oracle_core(pred_img: np.ndarray, big_sz: int) -> np.ndarray:
    """Reference-style pipeline on restored image. big_sz=0 means no large-scale kernel."""
    k_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr, _ = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shd = pred_img < max(25, thr * 0.5)
    th_s = cv2.morphologyEx(pred_img, cv2.MORPH_TOPHAT, k_t)
    bh_s = cv2.morphologyEx(pred_img, cv2.MORPH_BLACKHAT, k_t)
    mg = cv2.morphologyEx(pred_img, cv2.MORPH_GRADIENT, k_g)
    raw = shd | (th_s > 25) | (bh_s > 25) | (mg > 35)
    if big_sz > 0:
        k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big_sz, big_sz))
        th_l = cv2.morphologyEx(pred_img, cv2.MORPH_TOPHAT, k_big)
        bh_l = cv2.morphologyEx(pred_img, cv2.MORPH_BLACKHAT, k_big)
        raw = raw | (th_l > 20) | (bh_l > 20)
    return raw.astype(np.uint8)


def _apply_component_filter(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 1
    return out


def compute_hazard_mask(pred_img: np.ndarray, mode: str = "balanced") -> tuple:
    """
    Compute binary hazard mask using V2 oracle-core pipeline.
    mode: 'balanced' | 'conservative' | 'precision'

    Returns (hazard_mask, method_description).
    Hazard mask is the CORE hazard + controlled safety buffer.
    """
    if mode == "precision":
        core = _hazard_oracle_core(pred_img, big_sz=0)
        k_buf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = (cv2.dilate(core, k_buf, iterations=2) > 0).astype(np.uint8)
        mask = _apply_component_filter(dilated, 100)
        desc = "oracle-core(tophat7+blackhat7+morph_grad+otsu_shadow)+buf(9x9,2)+mc=100"
    elif mode == "conservative":
        core = _hazard_oracle_core(pred_img, big_sz=15)
        k_buf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = (cv2.dilate(core, k_buf, iterations=2) > 0).astype(np.uint8)
        mask = _apply_component_filter(dilated, 100)
        desc = "oracle-core(tophat7+11+15+blackhat+morph_grad+otsu_shadow)+buf(7x7,2)+mc=100"
    else:  # balanced (default)
        core = _hazard_oracle_core(pred_img, big_sz=11)
        k_buf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = (cv2.dilate(core, k_buf, iterations=2) > 0).astype(np.uint8)
        mask = _apply_component_filter(dilated, 50)
        desc = "oracle-core(tophat7+11+blackhat+morph_grad+otsu_shadow)+buf(7x7,2)+mc=50"
    return mask, desc


print(f"Advanced Lunar Processor Online")
print(f"Hardware: {device}")
print(f"Parameters: {total_params:,}")
print(f"Checkpoint: {MODEL_PATH} (epoch {MODEL_EPOCH})")
print(f"Hazard engine: V2 oracle-multiscale, 3 modes (balanced/conservative/precision)")


# === HELPER FUNCTIONS ===

def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def encode_image_to_base64(img: np.ndarray) -> str:
    """Encode numpy image to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def decode_uploaded_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image to grayscale numpy array."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def validate_image(img: np.ndarray):
    """Validate image constraints."""
    h, w = img.shape
    if h < 64 or w < 64:
        raise ValueError("Image too small (minimum 64x64)")
    if h > 4096 or w > 4096:
        raise ValueError("Image too large (maximum 4096x4096)")


def tiled_inference(img: np.ndarray, tile_size: int = 256, stride: int = 128) -> tuple:
    """
    Perform tiled U-Net inference with 2D linear blending.
    Returns (restored_image, tile_count, inference_time_ms).
    """
    h, w = img.shape

    # Pad image
    pad_h = (stride - (h % stride)) % stride + stride
    pad_w = (stride - (w % stride)) % stride + stride
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    # Prepare canvases
    out_canvas = np.zeros_like(padded_img, dtype=np.float32)
    weight_canvas = np.zeros_like(padded_img, dtype=np.float32)

    # Create 2D linear blend mask
    linear_1d = np.ones(tile_size, dtype=np.float32)
    edge_width = 32
    linear_1d[:edge_width] = np.linspace(0, 1, edge_width)
    linear_1d[-edge_width:] = np.linspace(1, 0, edge_width)
    linear_2d = np.outer(linear_1d, linear_1d)

    tile_count = 0
    inference_start = time.time()

    with torch.inference_mode():
        for y in range(0, padded_img.shape[0] - tile_size + 1, stride):
            for x in range(0, padded_img.shape[1] - tile_size + 1, stride):
                patch = padded_img[y:y+tile_size, x:x+tile_size]
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float() / 255.0
                tensor = tensor.to(device)

                enhanced_tensor = model(tensor)
                enhanced_patch = enhanced_tensor.squeeze().cpu().numpy() * 255.0

                out_canvas[y:y+tile_size, x:x+tile_size] += enhanced_patch * linear_2d
                weight_canvas[y:y+tile_size, x:x+tile_size] += linear_2d
                tile_count += 1

    inference_time_ms = (time.time() - inference_start) * 1000

    # Normalize by weights
    out_canvas = out_canvas / (weight_canvas + 1e-8)
    restored = out_canvas[:h, :w]
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    return restored, tile_count, inference_time_ms


def single_pass_inference(img: np.ndarray) -> tuple:
    """
    Perform single-pass U-Net inference for small images.
    Returns (restored_image, inference_time_ms).
    """
    h, w = img.shape

    # Resize to 256x256
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    inference_start = time.time()

    with torch.inference_mode():
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)
        enhanced_tensor = model(tensor)
        enhanced = enhanced_tensor.squeeze().cpu().numpy() * 255.0
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    inference_time_ms = (time.time() - inference_start) * 1000

    # Scale back to original size
    restored = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)

    return restored, inference_time_ms


def compute_image_stats(img: np.ndarray) -> ImageStats:
    """Compute no-reference image quality statistics."""
    # Noise level via Laplacian std
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    noise_level = float(np.std(laplacian))

    # Local contrast: std of local means
    kernel = np.ones((15, 15)) / 225
    local_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
    local_contrast = float(np.std(local_mean))

    # Edge density via Canny
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)

    # Brightness stats
    brightness_mean = float(np.mean(img))
    brightness_std = float(np.std(img))

    # Dynamic range
    dynamic_range = int(np.max(img) - np.min(img))

    # Saturation percentage (pixels at 0 or 255)
    saturated = np.sum((img == 0) | (img == 255))
    saturation_pct = float(saturated / img.size)

    return ImageStats(
        noise_level=noise_level,
        local_contrast=local_contrast,
        edge_density=edge_density,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        dynamic_range=dynamic_range,
        saturation_pct=saturation_pct,
    )


def detect_landing_zones(
    restored_img: np.ndarray,
    min_area: int = 500,
    hazard_mode: str = "balanced",
) -> tuple:
    """
    Detect landing zone candidates using V2 oracle-multiscale hazard engine.
    hazard_mode: 'balanced' | 'conservative' | 'precision'
    Returns (landing_zones, hazard_analysis_result).
    """
    h, w = restored_img.shape

    # Step 1: Compute hazard mask via V2 engine
    hazard_mask, hazard_method_desc = compute_hazard_mask(restored_img, mode=hazard_mode)
    danger_map = (hazard_mask * 255).astype(np.uint8)
    hazard_params = {"mode": hazard_mode, "method": "oracle_multiscale"}

    # Edge display: lightweight Canny for visual overlay only (not used in hazard decision)
    edges = cv2.Canny(restored_img, 40, 100)

    # Step 2: Create safe region mask
    safe_region = cv2.bitwise_not(danger_map)

    # Step 4: Find connected components in safe region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(safe_region, connectivity=8)

    # Step 5: Filter by minimum area and create candidates
    candidates = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_bbox = stats[i, cv2.CC_STAT_WIDTH]
            h_bbox = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            candidates.append({
                'center_x': int(cx),
                'center_y': int(cy),
                'area': int(area),
                'bbox': [int(x), int(y), int(w_bbox), int(h_bbox)],
            })

    # Step 6: Score candidates
    scored_candidates = []
    for cand in candidates:
        x, y, w_bbox, h_bbox = cand['bbox']
        cx, cy = cand['center_x'], cand['center_y']
        area = cand['area']

        # Extract ROI
        roi_danger = danger_map[y:y+h_bbox, x:x+w_bbox]
        roi_edges = edges[y:y+h_bbox, x:x+w_bbox]

        # Hazard density
        hazard_pixels = np.sum(roi_danger > 0)
        hazard_density = hazard_pixels / (w_bbox * h_bbox) if (w_bbox * h_bbox) > 0 else 1.0

        # Edge density
        edge_pixels = np.sum(roi_edges > 0)
        edge_density = edge_pixels / (w_bbox * h_bbox) if (w_bbox * h_bbox) > 0 else 1.0

        # Clearance: distance transform from nearest hazard
        dist_transform = ndimage.distance_transform_edt(safe_region)
        clearance = float(dist_transform[cy, cx])

        # Score calculation
        area_score = min(area / 10000.0, 1.0) * 30
        clearance_score = min(clearance / 50.0, 1.0) * 25
        hazard_penalty = hazard_density * 25
        edge_penalty = edge_density * 20

        raw_score = area_score + clearance_score - hazard_penalty - edge_penalty
        # Normalize to 0-100
        normalized_score = max(0, min(100, (raw_score / 50.0) * 100))

        scored_candidates.append({
            'center_x': cx,
            'center_y': cy,
            'area': area,
            'bbox': cand['bbox'],
            'score': normalized_score,
            'hazard_density': hazard_density,
            'edge_density': edge_density,
            'clearance': clearance,
            'score_breakdown': {
                'area_score': area_score,
                'clearance_score': clearance_score,
                'hazard_penalty': hazard_penalty,
                'edge_penalty': edge_penalty,
            },
            'scoring_weights': {
                'area': 30.0,
                'clearance': 25.0,
                'hazard': 25.0,
                'edge': 20.0,
            },
        })

    # Sort by score descending
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)

    # Assign IDs and ranks
    landing_zones = []
    for rank, cand in enumerate(scored_candidates, start=1):
        lz_id = f"LZ-{rank:02d}"

        # Generate selection reasons for top candidate
        selection_reasons = []
        if rank == 1:
            if cand['area'] > 5000:
                selection_reasons.append("Large usable area")
            if cand['hazard_density'] < 0.2:
                selection_reasons.append("Low hazard density")
            if cand['clearance'] > 30:
                selection_reasons.append("High clearance from obstacles")
            if cand['edge_density'] < 0.15:
                selection_reasons.append("Smooth terrain profile")

        landing_zones.append(LandingZoneCandidate(
            id=lz_id,
            center_x=cand['center_x'],
            center_y=cand['center_y'],
            area=cand['area'],
            score=cand['score'],
            score_type=ScoreType.HEURISTIC,
            hazard_density=cand['hazard_density'],
            clearance=cand['clearance'],
            edge_density=cand['edge_density'],
            bbox=cand['bbox'],
            rank=rank,
            status="selected" if rank == 1 else "candidate",
            score_breakdown=cand['score_breakdown'],
            scoring_weights=cand['scoring_weights'],
            selection_reasons=selection_reasons,
        ))

    # Compute overall hazard stats
    hazard_pixels = int(np.sum(danger_map > 0))
    safe_pixels = int(np.sum(safe_region > 0))
    total_pixels = int(h * w)
    hazard_percentage = float(hazard_pixels / total_pixels)
    safe_percentage = float(safe_pixels / total_pixels)
    edge_density_overall = float(np.sum(edges > 0) / total_pixels)

    # Create HazardAnalysisResult
    hazard_analysis = HazardAnalysisResult(
        edge_map_b64=encode_image_to_base64(edges),
        danger_map_b64=encode_image_to_base64(danger_map),
        safe_region_b64=encode_image_to_base64(safe_region),
        landing_zones_overlay_b64="",  # Will be filled below
        hazard_pixels=hazard_pixels,
        safe_pixels=safe_pixels,
        total_pixels=total_pixels,
        hazard_percentage=hazard_percentage,
        safe_percentage=safe_percentage,
        edge_density=edge_density_overall,
        canny_low=hazard_params.get("canny_low", 80),
        canny_high=hazard_params.get("canny_high", 200),
        dilation_kernel=hazard_params.get("kernel", 11),
        dilation_iterations=hazard_params.get("iter", 2),
        min_safe_area=min_area,
    )

    # Create landing zones overlay
    overlay = cv2.cvtColor(restored_img, cv2.COLOR_GRAY2BGR)

    # Draw danger map in red
    overlay[danger_map > 0] = [0, 0, 255]

    # Draw safe regions in green with transparency
    safe_overlay = overlay.copy()
    safe_overlay[safe_region > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, safe_overlay, 0.3, 0)

    # Draw landing zone candidates
    for lz in landing_zones:
        color = (255, 255, 255) if lz.rank == 1 else (0, 255, 255)  # White for selected, cyan for candidates
        thickness = 3 if lz.rank == 1 else 2

        # Draw circle
        cv2.circle(overlay, (lz.center_x, lz.center_y), 20, color, thickness)

        # Draw bounding box
        x, y, w_bbox, h_bbox = lz.bbox
        cv2.rectangle(overlay, (x, y), (x + w_bbox, y + h_bbox), color, thickness)

        # Draw ID label
        cv2.putText(overlay, lz.id, (lz.center_x - 20, lz.center_y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw score
        cv2.putText(overlay, f"{lz.score:.1f}", (lz.center_x - 20, lz.center_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    hazard_analysis.landing_zones_overlay_b64 = encode_image_to_base64(overlay)

    return landing_zones, hazard_analysis


# === ENDPOINTS ===

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="online",
        backend="fastapi",
        model="LunarUNet",
        device=str(device),
        version=SOFTWARE_VERSION,
    )


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    model_loaded = model is not None
    return {
        "ready": model_loaded,
        "model_loaded": model_loaded,
        "device": str(device),
    }


@app.get("/version")
async def version():
    """Version endpoint."""
    return {
        "version": SOFTWARE_VERSION,
        "model": "LunarUNet",
        "checkpoint": MODEL_PATH,
        "epoch": MODEL_EPOCH,
        "val_loss": 0.1146,
        "device": str(device),
        "hazard_engine": "oracle_multiscale_v2",
        "hazard_modes": ["balanced", "conservative", "precision"],
        "test_metrics": {
            "balanced":     {"precision": 70.21, "recall": 71.23, "f1": 70.71, "iou": 54.69},
            "conservative": {"precision": 62.64, "recall": 78.90, "f1": 69.83, "iou": 53.65},
            "precision":    {"precision": 75.39, "recall": 63.96, "f1": 69.21, "iou": 52.91},
        },
        "reference_type": "derived_photometric_reference (NOT human ground truth)",
    }


@app.get("/model-status", response_model=ModelInfo)
async def model_status():
    """Model status endpoint."""
    return ModelInfo(
        name="LunarUNet",
        architecture="U-Net with skip connections",
        device=str(device),
        input_channels=1,
        output_channels=1,
        parameter_count=total_params,
    )


@app.get("/system-status", response_model=SystemStatus)
async def system_status():
    """System status endpoint."""
    uptime = time.time() - STARTUP_TIME
    return SystemStatus(
        backend="fastapi",
        model_loaded=True,
        model_name="LunarUNet",
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
        version=SOFTWARE_VERSION,
        uptime_seconds=uptime,
    )


@app.get("/evaluation/latest")
async def get_latest_evaluation():
    """Returns the latest real model evaluation results."""
    eval_json_path = os.path.join(REPO_ROOT, "evaluation_results", "metrics.json")
    if not os.path.exists(eval_json_path):
        return {
            "status": "NOT_EVALUATED",
            "message": "Model evaluation has not been run yet. Run evaluate_model.py first.",
            "restoration": None,
            "hazard_detection": None,
        }
    with open(eval_json_path, "r") as f:
        data = json.load(f)
    return {
        "status": "EVALUATED",
        **data,
    }


@app.post("/pipeline", response_model=PipelineResult)
async def pipeline(
    file: UploadFile = File(...),
    hazard_mode: str = "balanced",
):
    """
    Main pipeline endpoint: restoration -> hazard analysis -> landing zone detection.
    hazard_mode: 'balanced' | 'conservative' | 'precision'
    """
    if hazard_mode not in ("balanced", "conservative", "precision"):
        hazard_mode = "balanced"
    pipeline_start = time.time()
    report_stages = {k: "SKIPPED" for k in ("image_received", "validation", "restoration", "hazard", "safe_region", "landing_zones")}
    active_report_stage = "image_received"
    mission_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    timeline = {
        'validation_ms': 0,
        'preprocessing_ms': 0,
        'inference_ms': 0,
        'blending_ms': 0,
        'hazard_analysis_ms': 0,
        'lz_detection_ms': 0,
        'scoring_ms': 0,
        'encoding_ms': 0,
        'total_ms': 0,
    }

    try:
        # Step 1: Validation
        validation_start = time.time()
        image_bytes = await file.read()
        report_stages["image_received"] = "COMPLETE"
        active_report_stage = "validation"
        file_size_bytes = len(image_bytes)
        content_hash = compute_sha256(image_bytes)

        img = decode_uploaded_image(image_bytes)
        validate_image(img)

        report_stages["validation"] = "COMPLETE"
        h, w = img.shape
        channels = 1

        timeline['validation_ms'] = (time.time() - validation_start) * 1000

        # Step 2: Input info
        input_info = InputInfo(
            filename=file.filename or "unknown",
            width=w,
            height=h,
            file_size_bytes=file_size_bytes,
            content_hash=content_hash,
            channels=channels,
        )

        active_report_stage = "restoration"
        # Step 3: U-Net Restoration
        preprocessing_start = time.time()
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        timeline['preprocessing_ms'] = (time.time() - preprocessing_start) * 1000

        if h > 300 and w > 300:
            # Large image: tiled inference
            tile_size = 256
            stride = 128
            restored, tile_count, inference_time_ms = tiled_inference(img_normalized, tile_size, stride)
            method = "tiled"
        else:
            # Small image: single pass
            restored, inference_time_ms = single_pass_inference(img_normalized)
            tile_size = 256
            stride = 0
            tile_count = 1
            method = "single-pass"

        timeline['inference_ms'] = inference_time_ms
        timeline['blending_ms'] = 0  # Included in inference

        overlap_pct = ((tile_size - stride) / tile_size * 100) if stride > 0 else 0

        restoration_result = RestorationResult(
            image_b64=encode_image_to_base64(restored),
            input_resolution=[w, h],
            output_resolution=[w, h],
            tile_size=tile_size,
            stride=stride,
            tile_count=tile_count,
            overlap_pct=overlap_pct,
            device=str(device),
            model_name="LunarUNet",
            inference_time_ms=inference_time_ms,
            total_time_ms=timeline['validation_ms'] + timeline['preprocessing_ms'] + inference_time_ms,
            method=method,
        )

        report_stages["restoration"] = "COMPLETE"
        # Step 4: Image statistics
        stats_start = time.time()
        image_stats = compute_image_stats(restored)
        stats_time = (time.time() - stats_start) * 1000

        # Step 5 & 6: Hazard detection and landing zone scoring
        active_report_stage = "hazard"
        hazard_start = time.time()
        hazard_mode_req = hazard_mode
        landing_zones, hazard_analysis = detect_landing_zones(
            restored, min_area=500, hazard_mode=hazard_mode_req)
        timeline['hazard_analysis_ms'] = (time.time() - hazard_start) * 1000
        timeline['lz_detection_ms'] = timeline['hazard_analysis_ms']  # Combined
        timeline['scoring_ms'] = timeline['hazard_analysis_ms']  # Combined

        report_stages.update(hazard="COMPLETE", safe_region="COMPLETE", landing_zones="COMPLETE")
        active_report_stage = "result_assembly"
        # Select top LZ
        selected_lz = landing_zones[0] if landing_zones else None

        # Step 7: Encoding (already done above)
        timeline['encoding_ms'] = stats_time

        # Step 8: Processing timeline
        timeline['total_ms'] = (time.time() - pipeline_start) * 1000

        processing_timeline = ProcessingTimeline(**timeline)

        # Step 9: Build PipelineResult
        terrain_metrics = TerrainMetrics(
            elevation_available=False,
            slope_available=False,
            roughness_available=False,
            dem_source=DataSource.UNAVAILABLE,
        )

        model_info = ModelInfo(
            name="LunarUNet",
            architecture="U-Net with skip connections",
            device=str(device),
            input_channels=1,
            output_channels=1,
            parameter_count=total_params,
        )

        scientific_limitations = [
            "Hazard detection is compared against a DERIVED PHOTOMETRIC REFERENCE, not human ground truth",
            "Hazard scores are HEURISTIC (edge/shadow/morphology-based) — not physical measurements",
            "No real-world elevation data (DEM) is used; slope/roughness are image-derived proxies",
            "Diameter estimates assume arbitrary scale factors — not calibrated km values",
            "Clearance calculations are 2D projections without slope analysis",
            "Water-ice prospectivity layer is UNAVAILABLE — no spectral/radar data ingested",
            "This tool is for research and demonstration purposes only",
            "Not validated for actual mission planning",
        ]

        result = PipelineResult(
            mission_id=mission_id,
            timestamp=timestamp,
            software_version=SOFTWARE_VERSION,
            input_info=input_info,
            restoration=restoration_result,
            hazard_analysis=hazard_analysis,
            landing_zones=landing_zones,
            selected_lz=selected_lz,
            image_stats=image_stats,
            terrain_metrics=terrain_metrics,
            processing_timeline=processing_timeline,
            model_info=model_info,
            scientific_limitations=scientific_limitations,
        )

        # Step 10: Build structured telemetry and return
        result_dict = result.model_dump()

        # V2 hazard metrics (test-set validated)
        _v2_mode_metrics = {
            "balanced":     {"test_precision": 70.21, "test_recall": 71.23, "test_f1": 70.71, "test_iou": 54.69},
            "conservative": {"test_precision": 62.64, "test_recall": 78.90, "test_f1": 69.83, "test_iou": 53.65},
            "precision":    {"test_precision": 75.39, "test_recall": 63.96, "test_f1": 69.21, "test_iou": 52.91},
        }

        # Structured telemetry block for frontend / Three.js mission UI
        result_dict["pipeline_telemetry"] = {
            "restoration": {
                "checkpoint": MODEL_PATH,
                "epoch": MODEL_EPOCH,
                "val_loss": 0.1146,
                "architecture": "LunarUNet (3-level encoder-decoder, skip connections)",
                "parameter_count": total_params,
                "training_data": "Chandrayaan-2 OHRC 0.25m lunar south-pole imagery",
            },
            "hazard": {
                "mask_available": True,
                "engine": "oracle_multiscale_v2",
                "mode": hazard_mode_req,
                "method_description": f"V2 oracle-multiscale ({hazard_mode_req}): Otsu-adaptive shadow + tophat/blackhat + morph-gradient on restored image + safety buffer",
                "reference_type": "derived_photometric_reference",
                "reference_description": (
                    "Hazard reference derived from HR telemetry via top-hat, black-hat, "
                    "shadow thresholding and morphological gradient. NOT human ground truth."
                ),
                "validated_test_metrics": _v2_mode_metrics.get(hazard_mode_req, {}),
                "available_modes": ["balanced", "conservative", "precision"],
            },
            "landing_zones": [
                {
                    "id": lz.id,
                    "centroid": {"x": lz.center_x, "y": lz.center_y},
                    "area_px": lz.area,
                    "hazard_density": round(lz.hazard_density, 4),
                    "clearance_px": round(lz.clearance, 1),
                    "edge_density": round(lz.edge_density, 4),
                    "heuristic_score": round(lz.score, 2),
                    "rank": lz.rank,
                    "status": lz.status,
                }
                for lz in landing_zones[:5]  # Top 5 only in telemetry
            ],
            "resource_layers": {
                "water_ice_prospectivity": {
                    "available": False,
                    "source": None,
                    "note": "Requires spectral/radar data — not available in current pipeline",
                },
                "thermal": {"available": False, "source": None},
                "illumination": {"available": False, "source": None},
                "radar": {"available": False, "source": None},
                "spectral": {"available": False, "source": None},
            },
        }

        # Reports consume a durable snapshot; failures here never invalidate analysis.
        try:
            analysis_token = report_service.save_analysis(
                result_dict, image_bytes=image_bytes,
                input_stats=compute_image_stats(img).model_dump(), stages=report_stages)
            result_dict["analysis_token"] = analysis_token
            if os.getenv("REPORT_AUTO_GENERATE", "true").lower() in ("true", "1", "yes"):
                job = report_service.create(mission_id, analysis_token)
                result_dict["report"] = report_service.metadata(job, include_token=True)
            else:
                result_dict["report"] = None
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Report persistence failed for %s", mission_id)
            result_dict["report"] = {"status": "FAILED", "error": "Analysis succeeded, but report storage is unavailable. Check backend storage configuration."}

        response = JSONResponse(content=result_dict)
        if selected_lz:
            telemetry = f"LZ={selected_lz.id},score={selected_lz.score:.1f},area={selected_lz.area}"
            response.headers["X-Landing-Telemetry"] = telemetry
            response.headers["Access-Control-Expose-Headers"] = "X-Landing-Telemetry"

        return response

    except Exception as e:
        report_stages[active_report_stage] = "FAILED"
        failure_info = {"mission_id": mission_id, "timestamp": timestamp,
                        "software_version": SOFTWARE_VERSION,
                        "input_info": {"filename": file.filename}}
        if "restoration_result" in locals():
            failure_info["restoration"] = restoration_result.model_dump()
        if "hazard_analysis" in locals():
            failure_info["hazard_analysis"] = hazard_analysis.model_dump()
        if "landing_zones" in locals():
            failure_info["landing_zones"] = [lz.model_dump() for lz in landing_zones]
        try:
            failure_token = report_service.save_analysis(failure_info, stages=report_stages,
                                                        error="Analysis failed at " + active_report_stage)
            detail = str(e) if isinstance(e, ValueError) else "Image analysis failed"
            failure_response = {"detail": detail, "analysis_id": mission_id, "analysis_token": failure_token}
        except Exception:
            failure_response = {"detail": "Image analysis failed; report storage unavailable"}
        import logging
        logging.getLogger(__name__).exception("Pipeline failed for %s", mission_id)
        return JSONResponse(status_code=400 if isinstance(e, ValueError) else 500, content=failure_response)


@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """
    Simplified enhancement endpoint: raw U-Net output without post-processing.
    """
    try:
        image_bytes = await file.read()
        img = decode_uploaded_image(image_bytes)
        h, w = img.shape

        # Normalize
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if h > 300 and w > 300:
            # Tiled inference
            restored, _, _ = tiled_inference(img, tile_size=256, stride=128)
        else:
            # Single pass
            restored, _ = single_pass_inference(img)

        # Return raw U-Net output as PNG
        _, encoded_img = cv2.imencode('.png', restored)
        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement error: {str(e)}")


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Crater analysis endpoint (kept from original, with HEURISTIC disclaimer).
    """
    try:
        image_bytes = await file.read()
        img_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        h, w = image.shape

        # Shadow density
        shadow_pixels = np.sum(image < 25)
        shadow_ratio = float(shadow_pixels) / (h * w)

        # Roughness
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_var = float(np.var(laplacian))
        roughness_index = float(np.clip(laplacian_var / 3500.0, 0.05, 0.95))

        # SNR
        mean_val = float(np.mean(image))
        std_val = float(np.std(image)) + 1e-6
        snr_db = float(10 * np.log10(mean_val / std_val)) if mean_val > 0 else 0.0

        # Albedo
        mean_albedo = float(mean_val / 255.0)

        # Crater detection via Hough
        resized = cv2.resize(image, (500, 500))
        blurred = cv2.GaussianBlur(resized, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=30, minRadius=30, maxRadius=240
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            main_circle = circles[0]
            cx = float(main_circle[0] * (w / 500.0))
            cy = float(main_circle[1] * (h / 500.0))
            r = float(main_circle[2] * (w / 500.0))
            detection_confidence = "HIGH"
        else:
            cx, cy, r = w / 2.0, h / 2.0, min(w, h) * 0.4
            detection_confidence = "LOW (ESTIMATED)"

        # Central peak
        crop_r = int(r * 0.15)
        x_start = max(0, int(cx - crop_r))
        x_end = min(w, int(cx + crop_r))
        y_start = max(0, int(cy - crop_r))
        y_end = min(h, int(cy + crop_r))

        central_crop = image[y_start:y_end, x_start:x_end]
        if central_crop.size > 0:
            _, _, _, maxLoc = cv2.minMaxLoc(central_crop)
            peak_x = float(x_start + maxLoc[0])
            peak_y = float(y_start + maxLoc[1])
            has_peak = True
        else:
            peak_x, peak_y = cx, cy
            has_peak = False

        # Diameter (HEURISTIC)
        estimated_diameter_km = float(r * 0.03)

        if estimated_diameter_km > 15.0:
            crater_type = "Complex Impact Crater"
            crater_desc = "Complex crater structure exceeding 15 km in diameter (HEURISTIC estimate)."
        else:
            crater_type = "Simple Bowl-Shaped Crater"
            crater_desc = "Simple bowl-shaped crater, under 15 km in diameter (HEURISTIC estimate)."

        # Narrative
        narrative_parts = []
        narrative_parts.append(f"Geological analysis identifies this as a {crater_type} ({estimated_diameter_km:.1f} km diameter, HEURISTIC).")

        if roughness_index > 0.5:
            narrative_parts.append("High topographic roughness detected.")
        else:
            narrative_parts.append("Relatively smooth, degraded topography.")

        if shadow_ratio > 0.3:
            narrative_parts.append(f"Significant shadow coverage ({shadow_ratio*100:.1f}%) indicates PSR presence.")
        else:
            narrative_parts.append(f"Low shadow coverage ({shadow_ratio*100:.1f}%).")

        if has_peak and estimated_diameter_km > 12.0:
            narrative_parts.append("Central peak uplift detected.")

        narrative = " ".join(narrative_parts)

        # Features
        features = [
            {
                "name": "Crater Rim Boundary",
                "x": cx, "y": cy, "radius": r,
                "desc": f"Outer circular limit. Confidence: {detection_confidence}."
            }
        ]

        if has_peak:
            features.append({
                "name": "Central Peak Uplift",
                "x": peak_x, "y": peak_y, "radius": r * 0.08,
                "desc": "Central peak from tectonic rebound."
            })

        features.append({
            "name": "Permanently Shadowed Zone",
            "x": cx - (r * 0.2), "y": cy + (r * 0.15), "radius": r * 0.35,
            "desc": f"Shadow concentration: {shadow_ratio*100:.1f}%."
        })

        return {
            "crater_type": crater_type,
            "description": crater_desc,
            "diameter_km": round(estimated_diameter_km, 2),
            "diameter_note": "HEURISTIC estimate based on arbitrary scale",
            "psr_coverage": f"{shadow_ratio * 100:.1f}%",
            "roughness_index": round(roughness_index, 3),
            "snr_db": f"{snr_db:.1f} dB",
            "mean_albedo": round(mean_albedo, 3),
            "features": features,
            "narrative": narrative
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Chat endpoint using Gemini (kept from original).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY is not configured in the backend environment."}

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
                "You are the Lunar PSR Geological Assistant, a premium built-in AI helper "
                "for the Lunar PSR Telemetry Enhancement Console. You help scientists, "
                "engineers, and operators process and understand Permanently Shadowed Regions "
                "(PSRs) on the Moon. You specialize in U-Net image enhancement, computer vision "
                "crater mapping, Topographic Roughness Index (TRI), shadow ratio calculations, "
                "and general lunar geology. Be highly precise, informative, helpful, and scientific."
            )
        )

        contents = []
        if req.context:
            ctx_summary = (
                f"Active Telemetry Context:\n"
                f"- Crater Type: {req.context.get('crater_type', 'Unknown')}\n"
                f"- Diameter: {req.context.get('diameter_km', 'N/A')} km\n"
                f"- PSR coverage ratio: {req.context.get('psr_coverage', 'N/A')}\n"
                f"- Topographic Roughness Index: {req.context.get('roughness_index', 'N/A')}\n"
                f"- Signal-to-Noise Ratio (SNR): {req.context.get('snr_db', 'N/A')}\n"
                f"- Albedo: {req.context.get('mean_albedo', 'N/A')}\n"
                f"- Narrative: {req.context.get('narrative', 'N/A')}"
            )
            contents.append({
                "role": "user",
                "parts": [f"[System Alert: The user is currently analyzing the following target:\n{ctx_summary}\nKeep this in mind when answering questions about the current image or target metrics.]"]
            })
            contents.append({
                "role": "model",
                "parts": ["Geological telemetry loaded. I am ready to answer queries relative to this target."]
            })

        for msg in req.history:
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [msg.content]})

        contents.append({"role": "user", "parts": [req.message]})

        response = model.generate_content(contents)
        return {"response": response.text}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"error": f"Gemini API Error: {str(e)}"}


@app.post("/generate-landing-video")
async def generate_landing_video(file: UploadFile = File(...)):
    """
    Generate landing video using Google Veo 3 (kept from original).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    try:
        image_bytes = await file.read()
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        _, jpeg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        jpeg_bytes_raw = jpeg_buf.tobytes()

        video_prompt = (
            "Cinematic short film, photoreal. "
            "A sleek silver lunar lander descends from the black void of space "
            "toward the moon surface shown in the reference image. "
            "Retro-rocket thrusters fire, kicking up bright plumes of regolith dust as the "
            "lander slows and touches down gently on the exact crater terrain visible in the image. "
            "Camera slowly zooms out to reveal the full landing site against a star-filled sky. "
            "No dialogue. Dramatic orchestral score. Ultra-realistic NASA-style footage."
        )

        from google import genai as genai_new
        from google.genai import types as genai_types

        client = genai_new.Client(api_key=api_key)

        image_ref = genai_types.Image(
            image_bytes=jpeg_bytes_raw,
            mime_type="image/jpeg"
        )

        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=video_prompt,
            image=image_ref,
            config=genai_types.GenerateVideosConfig(
                aspect_ratio="16:9",
                duration_seconds=8,
                number_of_videos=1,
            )
        )

        max_wait_seconds = 300
        poll_interval = 10
        elapsed = 0

        while not operation.done:
            if elapsed >= max_wait_seconds:
                raise HTTPException(status_code=504, detail="Video generation timed out.")
            time.sleep(poll_interval)
            elapsed += poll_interval
            operation = client.operations.get(operation)

        video_bytes = None
        if operation.response and hasattr(operation.response, "generated_videos"):
            for generated_video in operation.response.generated_videos:
                vid = generated_video.video
                if vid and vid.video_bytes:
                    video_bytes = vid.video_bytes
                    break
                elif vid and vid.uri:
                    import urllib.request
                    with urllib.request.urlopen(vid.uri) as resp:
                        video_bytes = resp.read()
                    break

        if not video_bytes:
            raise HTTPException(status_code=500, detail="Veo 3 did not return video bytes.")

        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        return JSONResponse(content={
            "status": "success",
            "video_b64": video_b64,
            "mime_type": "video/mp4"
        })

    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e)
        print(f"Veo 3 generation error: {err_msg}")
        import traceback
        traceback.print_exc()

        if "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg or "429" in err_msg:
            raise HTTPException(
                status_code=429,
                detail="Your API Key does not have quota for Veo 3.1 video generation."
            )
        raise HTTPException(status_code=500, detail=f"Veo 3 API error: {err_msg}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "online",
        "version": SOFTWARE_VERSION,
        "model": "LunarUNet",
        "device": str(device),
        "endpoints": [
            "/health", "/ready", "/version", "/model-status", "/system-status",
            "/pipeline", "/enhance", "/analyze", "/chat", "/generate-landing-video"
        ]
    }
