/**
 * Mission store — Zustand state for Lunar PSR Terminal.
 *
 * All state is driven by actual backend data or simulation.
 * Fields clearly marked SIMULATED are not real spacecraft telemetry.
 * Fields marked UNAVAILABLE have no backend data source yet.
 */
import { create } from 'zustand';
import { runPipeline, checkHealth } from './api';
import { SCENE_ORDER, TERRAIN_SIZE, getObservationWorldSpan, pxToWorldX, pxToWorldZ } from './types';

const INITIAL_STAGES = [
  { name: 'INGEST',         status: 'PENDING' },
  { name: 'VALIDATION',     status: 'PENDING' },
  { name: 'RESTORATION',    status: 'PENDING' },
  { name: 'HAZARD_ANALYSIS',status: 'PENDING' },
  { name: 'LZ_DETECTION',   status: 'PENDING' },
  { name: 'TELEMETRY',      status: 'PENDING' },
];

const INITIAL_TIMELINE = [
  { id: 'IMAGE_RECEIVED',  label: 'IMAGE RECEIVED',       status: 'pending', timestamp: null },
  { id: 'RESTORATION',     label: 'RESTORATION COMPLETE', status: 'pending', timestamp: null },
  { id: 'HAZARD_ANALYSIS', label: 'HAZARD ANALYSIS',      status: 'pending', timestamp: null },
  { id: 'LZ_GENERATED',    label: 'LZ CANDIDATES',        status: 'pending', timestamp: null },
  { id: 'LANDING',         label: 'LANDING',              status: 'pending', timestamp: null },
  { id: 'TOUCHDOWN',       label: 'TOUCHDOWN',            status: 'pending', timestamp: null },
  { id: 'ROVER',           label: 'ROVER DEPLOYED',       status: 'pending', timestamp: null },
  { id: 'EVA',             label: 'EVA',                  status: 'pending', timestamp: null },
  { id: 'SCIENCE',         label: 'SCIENCE LAYERS',       status: 'pending', timestamp: null },
];

function nowTimestamp() {
  return new Date().toISOString().split('T')[1].split('.')[0];
}

const useStore = create((set, get) => ({
  // ── Mission scene ─────────────────────────────────────────────────────────
  // Controls which major UI state is active.
  // See SCENE_ORDER in types.js for the full list.
  missionScene: 'LANDING_PAGE',

  missionId: null,
  missionStatus: 'IDLE', // 'IDLE' | 'PROCESSING' | 'COMPLETE' | 'ERROR'

  missionTimeline: INITIAL_TIMELINE,

  // ── Pipeline stages ───────────────────────────────────────────────────────
  pipelineStages: INITIAL_STAGES,

  // ── Input ─────────────────────────────────────────────────────────────────
  inputFile: null,
  inputInfo: null,   // from backend input_info

  // ── Hazard mode ───────────────────────────────────────────────────────────
  // Controls which V2 detector operating point is used.
  // Each mode has validated test metrics in types.js / HAZARD_MODE_METRICS.
  hazardMode: 'balanced', // 'balanced' | 'conservative' | 'precision'

  // ── Pipeline results (directly from backend) ──────────────────────────────
  restorationResult: null,   // backend: result.restoration
  hazardAnalysis: null,      // backend: result.hazard_analysis
  landingZones: [],           // backend: result.landing_zones[]
  selectedLzId: null,
  imageStats: null,           // backend: result.image_stats
  terrainMetrics: null,       // backend: result.terrain_metrics (all unavailable)
  processingTimeline: null,   // backend: result.processing_timeline
  modelInfo: null,            // backend: result.model_info
  scientificLimitations: [],  // backend: result.scientific_limitations[]

  // ── pipeline_telemetry (V2 structured data) ───────────────────────────────
  // Set from result.pipeline_telemetry — the primary source of truth for
  // restoration metadata, hazard engine info, and resource layer stubs.
  pipelineTelemetry: null,

  // ── Image URLs (decoded from base64 backend responses) ───────────────────
  rawImageUrl: null,       // local object URL from uploaded file
  restoredImageUrl: null,  // decoded from restoration.image_b64
  dangerMapUrl: null,      // decoded from hazard_analysis.danger_map_b64
  safeRegionUrl: null,     // decoded from hazard_analysis.safe_region_b64
  lzOverlayUrl: null,      // decoded from hazard_analysis.landing_zones_overlay_b64
  edgeMapUrl: null,        // decoded from hazard_analysis.edge_map_b64

  // ── System ────────────────────────────────────────────────────────────────
  backendStatus: 'checking', // 'checking' | 'online' | 'offline'
  systemStatus: null,

  // ── Events log ────────────────────────────────────────────────────────────
  events: [],

  addEvent: (text, type = 'info') => {
    set((s) => ({
      events: [
        ...s.events,
        { text, type, timestamp: nowTimestamp() },
      ].slice(-100), // keep last 100
    }));
  },

  // ── View state ────────────────────────────────────────────────────────────
  viewMode: '2D',          // '2D' | '3D'
  analysisLayer: 'RESTORED', // 'RAW'|'RESTORED'|'DANGER'|'SAFE_REGION'|'LANDING_ZONES'

  // ── Descent simulation (SIMULATED — not real telemetry) ───────────────────
  descentMode: false,
  descentPhase: 'IDLE',
  descentTelemetry: {
    altitude: 0,
    vVel: 0,
    hVel: 0,
    targetError: 0,
    time: 0,
  },

  // ── Terrain layer (3D visualization mode) ────────────────────────────────
  terrainLayer: 'NORMAL', // 'NORMAL'|'HAZARD'|'SAFE'|'LZ'

  // ── Surface operations (SIMULATED) ───────────────────────────────────────
  touchdownComplete: false,
  roverActive: false,
  roverPos: [5, 5],      // [x, z] world coords
  roverHeading: 0,       // radians
  astronautActive: false,
  astronautPos: [0, 0],
  astronautHeading: 0,
  sciencePanelOpen: false,

  // ── Recording ─────────────────────────────────────────────────────────────
  isRecording: false,
  recordingDuration: 0,
  recordingUrl: null,

  // ── Actions ───────────────────────────────────────────────────────────────

  setMissionScene: (scene) => {
    if (!SCENE_ORDER.includes(scene)) return;
    set({ missionScene: scene });
    get().addEvent(`Mission scene → ${scene}`, 'info');
  },

  // Moon clicked — begin approach animation, then mission setup
  startMission: () => {
    set({ missionScene: 'MOON_INTERACTION' });
    get().addEvent('Mission initiated — approaching lunar surface', 'info');
    // After approach animation, advance to setup
    setTimeout(() => {
      // Only advance if still in MOON_INTERACTION (user hasn't reset)
      if (get().missionScene === 'MOON_INTERACTION') {
        set({ missionScene: 'MISSION_SETUP' });
        get().addEvent('Lunar surface reached — upload observation', 'info');
      }
    }, 2800);
  },

  setHazardMode: (mode) => {
    if (!['balanced', 'conservative', 'precision'].includes(mode)) return;
    set({ hazardMode: mode });
    get().addEvent(`Hazard mode → ${mode.toUpperCase()}`, 'info');
  },

  setInputFile: (file) => {
    if (!file) {
      set({ inputFile: null, inputInfo: null, rawImageUrl: null });
      return;
    }

    const img = new Image();
    const url = URL.createObjectURL(file);
    img.src = url;

    img.onload = () => {
      set({
        inputFile: file,
        inputInfo: {
          filename: file.name,
          width: img.width,
          height: img.height,
          file_size_bytes: file.size,
          content_hash: `${file.name}-${file.size}`,
          channels: 1,
        },
        rawImageUrl: url,
      });
      get().addEvent(`Loaded: ${file.name} (${img.width}×${img.height})`, 'info');
      get()._markTimeline('IMAGE_RECEIVED', 'complete');
    };
  },

  updateStage: (stageName, status) => {
    set((s) => ({
      pipelineStages: s.pipelineStages.map((stage) =>
        stage.name === stageName ? { ...stage, status } : stage
      ),
    }));
  },

  _markTimeline: (id, status) => {
    set((s) => ({
      missionTimeline: s.missionTimeline.map((step) =>
        step.id === id
          ? { ...step, status, timestamp: status === 'complete' ? nowTimestamp() : step.timestamp }
          : step
      ),
    }));
  },

  runPipeline: async (file) => {
    const { addEvent, updateStage, _markTimeline, hazardMode } = get();

    set({
      missionStatus: 'PROCESSING',
      missionId: `MISSION-${Date.now()}`,
      pipelineStages: INITIAL_STAGES.map((s) => ({ ...s })),
    });

    addEvent('Pipeline initiated', 'info');

    // INGEST
    updateStage('INGEST', 'RUNNING');
    updateStage('VALIDATION', 'RUNNING');

    // Move to ANALYSIS scene when pipeline starts
    set({ missionScene: 'ANALYSIS' }); // analysis running

    try {
      // RESTORATION — this is where the network call happens
      updateStage('RESTORATION', 'RUNNING');
      addEvent(`Running neural restoration (epoch 57, mode=${hazardMode})…`, 'info');

      const result = await runPipeline(file, hazardMode);

      updateStage('INGEST', 'COMPLETE');
      updateStage('VALIDATION', 'COMPLETE');
      updateStage('RESTORATION', 'COMPLETE');
      addEvent('Restoration complete', 'success');
      _markTimeline('RESTORATION', 'complete');

      // HAZARD
      updateStage('HAZARD_ANALYSIS', 'RUNNING');
      updateStage('HAZARD_ANALYSIS', 'COMPLETE');
      addEvent(`Hazard analysis complete (${hazardMode})`, 'success');
      _markTimeline('HAZARD_ANALYSIS', 'complete');

      // LZ
      updateStage('LZ_DETECTION', 'RUNNING');
      updateStage('LZ_DETECTION', 'COMPLETE');
      const lzCount = (result.landing_zones || []).length;
      addEvent(`${lzCount} LZ candidates found`, 'success');
      _markTimeline('LZ_GENERATED', 'complete');

      // TELEMETRY
      updateStage('TELEMETRY', 'RUNNING');
      updateStage('TELEMETRY', 'COMPLETE');

      // Decode images
      const restoration = result.restoration || {};
      const hazard = result.hazard_analysis || {};
      const telemetry = result.pipeline_telemetry || null;

      set({
        missionId: result.mission_id || get().missionId,
        missionStatus: 'COMPLETE',
        missionScene: 'ANALYSIS_RESULTS',  // stay until user enters terrain

        restorationResult: restoration,
        hazardAnalysis: hazard,
        landingZones: result.landing_zones || [],
        selectedLzId: result.selected_lz ? result.selected_lz.id : null,
        imageStats: result.image_stats || {},
        terrainMetrics: result.terrain_metrics || {},
        processingTimeline: result.processing_timeline || {},
        modelInfo: result.model_info || {},
        scientificLimitations: result.scientific_limitations || [],
        inputInfo: result.input_info || get().inputInfo,
        pipelineTelemetry: telemetry,

        restoredImageUrl: restoration.image_b64
          ? `data:image/png;base64,${restoration.image_b64}`
          : null,
        edgeMapUrl: hazard.edge_map_b64
          ? `data:image/png;base64,${hazard.edge_map_b64}`
          : null,
        dangerMapUrl: hazard.danger_map_b64
          ? `data:image/png;base64,${hazard.danger_map_b64}`
          : null,
        safeRegionUrl: hazard.safe_region_b64
          ? `data:image/png;base64,${hazard.safe_region_b64}`
          : null,
        lzOverlayUrl: hazard.landing_zones_overlay_b64
          ? `data:image/png;base64,${hazard.landing_zones_overlay_b64}`
          : null,
      });

      if (telemetry) {
        const ckpt = telemetry.restoration?.checkpoint || '';
        const epoch = telemetry.restoration?.epoch || '';
        addEvent(`Checkpoint: ${ckpt} (epoch ${epoch})`, 'info');

        const hEngine = telemetry.hazard?.engine || 'unknown';
        const hMode = telemetry.hazard?.mode || hazardMode;
        const hMetrics = telemetry.hazard?.validated_test_metrics || {};
        addEvent(
          `Hazard engine: ${hEngine} [${hMode}] — F1=${hMetrics.test_f1?.toFixed(1)}% IoU=${hMetrics.test_iou?.toFixed(1)}%`,
          'info'
        );

        // Log resource layer status
        const rl = telemetry.resource_layers || {};
        const unavailable = Object.entries(rl)
          .filter(([, v]) => !v.available)
          .map(([k]) => k);
        if (unavailable.length) {
          addEvent(`Resource layers unavailable: ${unavailable.join(', ')}`, 'warn');
        }
      }

      addEvent('Pipeline complete — ready for terrain view', 'success');
    } catch (err) {
      console.error('Pipeline error:', err);
      set({ missionStatus: 'ERROR' });
      addEvent(`Pipeline failed: ${err.message}`, 'error');

      get().pipelineStages.forEach((stage) => {
        if (stage.status === 'RUNNING') {
          updateStage(stage.name, 'FAILED');
        }
      });
    }
  },

  selectLz: (id) => {
    set({ selectedLzId: id });
    get().addEvent(`LZ ${id} selected`, 'info');
  },

  startDescent: () => {
    const { selectedLzId, landingZones } = get();
    if (!selectedLzId || landingZones.length === 0) {
      get().addEvent('Cannot start descent: no LZ selected', 'error');
      return;
    }

    set({
      descentMode: true,
      descentPhase: 'APPROACH',
      missionScene: 'LANDING',
      descentTelemetry: { altitude: 65, vVel: -8, hVel: 0.5, targetError: 8.5, time: 0 },
    });

    get()._markTimeline('LANDING', 'active');
    get().addEvent('Descent simulation initiated (SIMULATED)', 'info');
  },

  updateDescentTelemetry: (telemetry) => set({ descentTelemetry: telemetry }),

  setDescentPhase: (phase) => {
    set({ descentPhase: phase });
    if (phase === 'TOUCHDOWN') {
      get()._markTimeline('LANDING', 'complete');
      get()._markTimeline('TOUCHDOWN', 'active');
      set({ missionScene: 'TOUCHDOWN' });
      get().addEvent('TOUCHDOWN CONFIRMED (SIMULATED)', 'success');
    }
  },

  stopDescent: () => {
    set({ descentMode: false, descentPhase: 'IDLE' });
    get().addEvent('Descent simulation ended', 'info');
  },

  setTerrainLayer: (layer) => set({ terrainLayer: layer }),

  confirmTouchdown: () => {
    // Idempotent — DescentSimulation's rAF can call this exactly once
    if (get().touchdownComplete) return;
    set({ touchdownComplete: true, missionScene: 'TOUCHDOWN' });
    get()._markTimeline('LANDING', 'complete');
    get()._markTimeline('TOUCHDOWN', 'active');
    get().addEvent('TOUCHDOWN CONFIRMED (SIMULATED)', 'success');
  },

  deployRover: () => {
    const { selectedLzId, landingZones, inputInfo } = get();
    const lz = landingZones.find((z) => z.id === selectedLzId);
    const w = inputInfo?.width || 256;
    const h = inputInfo?.height || 256;
    // Place rover near the selected LZ in terrain world space.
    // World coord = (pixelCoord/imageSize - 0.5) * TERRAIN_SIZE  (same as TerrainViewport)
    const rx = lz ? ((lz.center_x / w) - 0.5) * TERRAIN_SIZE + 5 : 5;
    const rz = lz ? ((lz.center_y / h) - 0.5) * TERRAIN_SIZE + 5 : 5;
    set({ missionScene: 'ROVER', roverActive: true, roverPos: [rx, rz] });
    get()._markTimeline('TOUCHDOWN', 'complete');
    get()._markTimeline('ROVER', 'active');
    get().addEvent('Rover deployment sequence initiated (SIMULATED)', 'info');
  },

  updateRover: (pos, heading) => set({ roverPos: pos, roverHeading: heading }),

  startEVA: () => {
    const { roverPos } = get();
    set({
      missionScene: 'EVA',
      astronautActive: true,
      astronautPos: [roverPos[0] - 3, roverPos[1]],
      astronautHeading: 0,
    });
    get()._markTimeline('ROVER', 'complete');
    get()._markTimeline('EVA', 'active');
    get().addEvent('EVA commenced (SIMULATED)', 'info');
  },

  updateAstronaut: (pos, heading) => set({ astronautPos: pos, astronautHeading: heading }),

  openScience: () => {
    set({ missionScene: 'SCIENCE', sciencePanelOpen: true });
    get()._markTimeline('EVA', 'complete');
    get()._markTimeline('SCIENCE', 'active');
    get().addEvent('Science layer panel opened', 'info');
  },

  enterTerrain: () => {
    set({ missionScene: 'TERRAIN', viewMode: '3D' });
    get()._markTimeline('LZ_GENERATED', 'complete');
    get().addEvent('Entering terrain view', 'success');
  },

  setViewMode: (mode) => set({ viewMode: mode }),
  setAnalysisLayer: (layer) => set({ analysisLayer: layer }),

  setRecording: (v) => set({ isRecording: v }),
  setRecordingUrl: (url) => set({ recordingUrl: url }),

  checkBackend: async () => {
    try {
      const health = await checkHealth();
      set({ backendStatus: 'online', systemStatus: health });
    } catch {
      set({ backendStatus: 'offline', systemStatus: null });
    }
  },

  reset: () => {
    // Revoke old object URLs to avoid memory leaks
    const { rawImageUrl } = get();
    if (rawImageUrl && rawImageUrl.startsWith('blob:')) {
      URL.revokeObjectURL(rawImageUrl);
    }

    set({
      missionScene: 'LANDING_PAGE',
      missionId: null,
      missionStatus: 'IDLE',
      missionTimeline: INITIAL_TIMELINE.map((s) => ({ ...s })),
      pipelineStages: INITIAL_STAGES.map((s) => ({ ...s })),
      inputFile: null,
      inputInfo: null,
      restorationResult: null,
      hazardAnalysis: null,
      selectedLzId: null,
      imageStats: null,
      terrainMetrics: null,
      processingTimeline: null,
      modelInfo: null,
      scientificLimitations: [],
      pipelineTelemetry: null,
      rawImageUrl: null,
      restoredImageUrl: null,
      dangerMapUrl: null,
      safeRegionUrl: null,
      lzOverlayUrl: null,
      edgeMapUrl: null,
      viewMode: '2D',
      analysisLayer: 'RESTORED',
      descentMode: false,
      descentPhase: 'IDLE',
      descentTelemetry: { altitude: 0, vVel: 0, hVel: 0, targetError: 0, time: 0 },
      terrainLayer: 'NORMAL',
      touchdownComplete: false,
      roverActive: false,
      roverPos: [5, 5],
      roverHeading: 0,
      astronautActive: false,
      astronautPos: [0, 0],
      astronautHeading: 0,
      sciencePanelOpen: false,
      isRecording: false,
      recordingDuration: 0,
      recordingUrl: null,
    });
    get().addEvent('Mission state reset to default', 'info');
  },
}));

export default useStore;