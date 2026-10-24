/**
 * Mission state types — mirrors the backend pipeline_telemetry contract.
 *
 * These are JSDoc annotations (no TypeScript build step needed).
 * Every field maps to actual backend output. Fields marked UNAVAILABLE
 * are confirmed absent from the backend and must never be fabricated.
 *
 * Backend version: 2.2.0
 * Checkpoint: weights/lunar_unet_v2_best.pth (epoch 57, val_loss=0.1146)
 */

/**
 * @typedef {'IDLE'|'ORBIT'|'ANALYSIS'|'TERRAIN'|'LANDING'|'TOUCHDOWN'|'ROVER'|'EVA'|'SCIENCE'} MissionScene
 */

/**
 * @typedef {'PENDING'|'RUNNING'|'COMPLETE'|'FAILED'} StageStatus
 */

/**
 * @typedef {'balanced'|'conservative'|'precision'} HazardMode
 */

/**
 * @typedef {Object} PipelineStage
 * @property {'INGEST'|'VALIDATION'|'RESTORATION'|'HAZARD_ANALYSIS'|'LZ_DETECTION'|'TELEMETRY'} name
 * @property {StageStatus} status
 */

/**
 * @typedef {Object} RestorationResult
 * Source: backend /pipeline → restoration
 * @property {string|null} image_b64        - Base64 PNG of restored image
 * @property {number[]} input_resolution    - [width, height]
 * @property {number[]} output_resolution   - [width, height]
 * @property {number} inference_time_ms
 * @property {string} method                - 'tiled' | 'single-pass'
 * @property {string} model_name
 * @property {string} device
 */

/**
 * @typedef {Object} RestorationCheckpoint
 * Source: pipeline_telemetry.restoration
 * @property {string} checkpoint            - 'weights/lunar_unet_v2_best.pth'
 * @property {number} epoch                 - 57
 * @property {number} val_loss              - 0.1146
 * @property {string} architecture
 * @property {number} parameter_count
 * @property {string} training_data
 */

/**
 * @typedef {Object} HazardTelemetry
 * Source: pipeline_telemetry.hazard
 * @property {boolean} mask_available
 * @property {string} engine                - 'oracle_multiscale_v2'
 * @property {HazardMode} mode
 * @property {string} method_description
 * @property {'derived_photometric_reference'} reference_type
 * @property {string} reference_description
 * @property {Object} validated_test_metrics - {test_precision, test_recall, test_f1, test_iou}
 * @property {string[]} available_modes
 */

/**
 * @typedef {Object} LandingZoneTelemetry
 * Source: pipeline_telemetry.landing_zones[]
 * @property {string} id
 * @property {{x:number, y:number}} centroid
 * @property {number} area_px
 * @property {number} hazard_density
 * @property {number} clearance_px
 * @property {number} edge_density
 * @property {number} heuristic_score
 * @property {number} rank
 * @property {string} status
 */

/**
 * @typedef {Object} ResourceLayer
 * Source: pipeline_telemetry.resource_layers
 * @property {boolean} available            - Always false until backend provides data
 * @property {string|null} source
 * @property {string|null} note
 */

/**
 * @typedef {Object} ResourceLayers
 * @property {ResourceLayer} water_ice_prospectivity
 * @property {ResourceLayer} thermal
 * @property {ResourceLayer} illumination
 * @property {ResourceLayer} radar
 * @property {ResourceLayer} spectral
 */

/**
 * @typedef {Object} PipelineTelemetry
 * Source: backend /pipeline → pipeline_telemetry
 * @property {RestorationCheckpoint} restoration

















































/**
 * Validated test metrics for each hazard mode.
 * Source: evaluation_results/v2_final/v2_final_comparison.json
 * Reference: DERIVED PHOTOMETRIC REFERENCE (not human ground truth)
 */
export const HAZARD_MODE_METRICS = {
  balanced: {
    precision: 70.21,
    recall: 71.23,
    f1: 70.71,
    iou: 54.69,
    pred_frac: 48.3,
    description: 'Balanced precision/recall — calibrated to similar hazard prevalence as reference',
  },
  conservative: {
    precision: 62.64,
    recall: 78.90,
    f1: 69.83,
    iou: 53.65,
    pred_frac: 59.9,
    description: 'Higher recall — flags more potential hazards, accepting more false positives',
  },
  precision: {
    precision: 75.39,
    recall: 63.96,
    f1: 69.21,
    iou: 52.91,
    pred_frac: 40.3,
    description: 'Higher precision — fewer false positives, misses more potential hazards',
  },
};

/**
 * Ordered mission timeline steps.
 * Status is driven by missionScene in the store.
 */
export const MISSION_TIMELINE = [
  { id: 'IMAGE_RECEIVED',    label: 'IMAGE RECEIVED',       scene: null },
  { id: 'RESTORATION',       label: 'RESTORATION COMPLETE', scene: null },
  { id: 'HAZARD_ANALYSIS',   label: 'HAZARD ANALYSIS',      scene: null },
  { id: 'LZ_GENERATED',      label: 'LZ CANDIDATES',        scene: null },
  { id: 'LANDING',           label: 'LANDING',              scene: 'LANDING' },
  { id: 'TOUCHDOWN',         label: 'TOUCHDOWN',            scene: 'TOUCHDOWN' },
  { id: 'ROVER',             label: 'ROVER DEPLOYED',       scene: 'ROVER' },
  { id: 'EVA',               label: 'EVA',                  scene: 'EVA' },
  { id: 'SCIENCE',           label: 'SCIENCE LAYERS',       scene: 'SCIENCE' },
];

export const SCENE_ORDER = [
  'LANDING_PAGE',   // cinematic Moon — hero screen
  'MOON_INTERACTION', // Moon click → approach
  'MISSION_SETUP',  // mission brief + upload
  'ANALYSIS',       // pipeline running
  'ANALYSIS_RESULTS', // results display
  'TERRAIN',        // 3D terrain
  'LANDING',        // lander descent
  'TOUCHDOWN',      // touchdown moment
  'ROVER',          // rover exploration
  'EVA',            // astronaut EVA
  'SCIENCE',        // science inspection
];

/** Backwards-compat aliases used by existing store logic */
export const SCENE_LEGACY_MAP = {
  IDLE:    'LANDING_PAGE',
  ORBIT:   'MOON_INTERACTION',
  TERRAIN: 'TERRAIN',
};

export const TERRAIN_SIZE = 100;

/**
 * World space spans [-TERRAIN_SIZE/2, TERRAIN_SIZE/2] on X and Z axes.
 * World coordinates: wx = (u - 0.5) * TERRAIN_SIZE, wz = (v - 0.5) * TERRAIN_SIZE
 * where u = px_x / imageWidth, v = px_y / imageHeight (normalised image UV).
 */
export function getObservationWorldSpan(aspectRatio = 1) {
  if (aspectRatio >= 1) {
    return { spanX: TERRAIN_SIZE, spanZ: TERRAIN_SIZE / aspectRatio };
  } else {
    return { spanX: TERRAIN_SIZE * aspectRatio, spanZ: TERRAIN_SIZE };
  }
}

export function pxToWorldX(pxX, imgW, spanX = TERRAIN_SIZE) {
  return (pxX / imgW - 0.5) * spanX;
}

export function pxToWorldZ(pxY, imgH, spanZ = TERRAIN_SIZE) {
  return (pxY / imgH - 0.5) * spanZ;
}

export function worldToU(wx, spanX = TERRAIN_SIZE) {
  return wx / spanX + 0.5;
}

export function worldToV(wz, spanZ = TERRAIN_SIZE) {
  return wz / spanZ + 0.5;
}

