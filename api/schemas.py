from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ScoreType(str, Enum):
    HEURISTIC = "HEURISTIC"
    COMPUTED = "COMPUTED"
    SIMULATED = "SIMULATED"


class DataSource(str, Enum):
    COMPUTED = "COMPUTED"
    HEURISTIC = "HEURISTIC"
    SIMULATED = "SIMULATED"
    INPUT = "INPUT"
    UNAVAILABLE = "UNAVAILABLE"


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class LandingZoneCandidate(BaseModel):
    id: str
    center_x: int
    center_y: int
    area: int
    score: float
    score_type: ScoreType = ScoreType.HEURISTIC
    hazard_density: float
    clearance: float
    edge_density: float
    bbox: List[int] = Field(description="[x, y, w, h]")
    rank: int
    status: str = "candidate"
    score_breakdown: Dict[str, float] = {}
    scoring_weights: Dict[str, float] = {}
    selection_reasons: List[str] = []


class RestorationResult(BaseModel):
    image_b64: str
    input_resolution: List[int]
    output_resolution: List[int]
    tile_size: int
    stride: int
    tile_count: int
    overlap_pct: float
    device: str
    model_name: str
    inference_time_ms: float
    total_time_ms: float
    method: str = "tiled"


class HazardAnalysisResult(BaseModel):
    edge_map_b64: str
    danger_map_b64: str
    safe_region_b64: str
    landing_zones_overlay_b64: str
    hazard_pixels: int
    safe_pixels: int
    total_pixels: int
    hazard_percentage: float
    safe_percentage: float
    edge_density: float
    canny_low: int
    canny_high: int
    dilation_kernel: int
    dilation_iterations: int
    min_safe_area: int


class ImageStats(BaseModel):
    noise_level: float = Field(description="Estimated noise std via Laplacian method")
    local_contrast: float
    edge_density: float
    brightness_mean: float
    brightness_std: float
    dynamic_range: int
    saturation_pct: float = Field(description="Percentage of pixels at 0 or 255")


class ProcessingTimeline(BaseModel):
    validation_ms: float
    preprocessing_ms: float
    inference_ms: float
    blending_ms: float
    hazard_analysis_ms: float
    lz_detection_ms: float
    scoring_ms: float
    encoding_ms: float
    total_ms: float


class ModelInfo(BaseModel):
    name: str
    architecture: str
    device: str
    input_channels: int
    output_channels: int
    parameter_count: int


class TerrainMetrics(BaseModel):
    elevation_available: bool = False
    slope_available: bool = False
    roughness_available: bool = False
    dem_source: DataSource = DataSource.UNAVAILABLE


class InputInfo(BaseModel):
    filename: str
    width: int
    height: int
    file_size_bytes: int
    content_hash: str
    channels: int


class PipelineResult(BaseModel):
    mission_id: str
    timestamp: str
    software_version: str
    input_info: InputInfo
    restoration: RestorationResult
    hazard_analysis: HazardAnalysisResult
    landing_zones: List[LandingZoneCandidate]
    selected_lz: Optional[LandingZoneCandidate]
    image_stats: ImageStats
    terrain_metrics: TerrainMetrics
    processing_timeline: ProcessingTimeline
    model_info: ModelInfo
    scientific_limitations: List[str]
    pipeline_telemetry: Optional[Dict[str, Any]] = None
    analysis_token: Optional[str] = None
    report: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    device: str
    version: str


class SystemStatus(BaseModel):
    backend: str
    model_loaded: bool
    model_name: str
    device: str
    cuda_available: bool
    mps_available: bool
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
