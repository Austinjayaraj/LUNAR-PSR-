/**
 * AnalysisViewport — Single Unified Mission Analysis (Scene 1).
 *
 * UNIFIED VISUALIZATION:
 *   - Combines Restored Lunar Image + Hazard Overlay + Safe Regions + Interactive LZ Reticles
 *     into ONE comprehensive scientific mission map.
 *   - No separate tabs.
 *   - No exposed BALANCED / CONSERVATIVE / PRECISION selector (defaults to validated V2 balanced mode).
 *   - Large image centerpiece filling the viewport.
 *   - Interactive LZ markers: clicking an LZ selects it as the landing target.
 *   - Bottom CTA: "ANALYSIS COMPLETE · [ ENTER SITE ]".
 */
import { useRef, useState, useCallback, useEffect } from 'react';
import { Upload, Play, RefreshCw, AlertTriangle, ChevronRight, Crosshair, Check } from 'lucide-react';
import useStore from '../store';

// ── Upload Dropzone (Mission Setup) ──────────────────────────────────────────

function UploadZone() {
  const setInputFile = useStore((s) => s.setInputFile);
  const [dragOver, setDragOver] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const inputRef = useRef(null);

  const VALID_TYPES = ['image/png', 'image/jpeg', 'image/tiff', 'image/bmp', 'image/webp'];
  const MAX_BYTES = 50 * 1024 * 1024; // 50 MB

  const validate = (file) => {
    if (!file) return 'No file selected.';
    if (!VALID_TYPES.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|tif|tiff|bmp)$/i))
      return `Unsupported format: ${file.type || file.name.split('.').pop()}`;
    if (file.size === 0) return 'File is empty.';
    if (file.size > MAX_BYTES) return `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max 50 MB.`;
    return null;
  };

  const accept = (file) => {
    const err = validate(file);
    if (err) { setValidationError(err); return; }
    setValidationError(null);
    setInputFile(file);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) accept(file);
  }, []);

  const onDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);
  const onChange = (e) => { const f = e.target.files[0]; if (f) accept(f); };

  const loadPresetImage = async (url, filename) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/png' });
      accept(file);
    } catch {
      setValidationError('Failed to load sample observation.');
    }
  };

  return (
    <div className="av-setup-container">
      <div className="av-mission-header">
        <div className="av-eyebrow">LUNAR SOUTH POLE</div>
        <h1 className="av-title">PERMANENTLY SHADOWED REGION</h1>
        <div className="av-subtitle">OBSERVATION ANALYSIS</div>
      </div>

      <div className="av-upload-wrap">
        <div
          className={`av-dropzone${dragOver ? ' av-dropzone--over' : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"
            className="hidden"
            onChange={onChange}
          />
          <Upload size={38} className="av-dropzone-icon" />
          <p className="av-dropzone-title">DROP LUNAR OBSERVATION</p>
          <span className="av-dropzone-hint">or</span>
          <div className="av-dropzone-btn">SELECT IMAGE</div>
          <p className="av-dropzone-note">CHANDRAYAAN-2 OHRC / LRO OPTICAL IMAGERY (0.25M)</p>
        </div>

        {validationError && (
          <div className="av-error-banner">
            <AlertTriangle size={13} />
            <span>{validationError}</span>
          </div>
        )}

        <div className="av-preset-loader">
          <span className="av-preset-label">— OR LOAD CALIBRATED TEST OBSERVATION —</span>
          <button
            className="av-preset-btn"
            onClick={(e) => {
              e.stopPropagation();
              loadPresetImage('/presets/crater_raw.png', 'shackleton_crater_raw.png');
            }}
          >
            SHACKLETON RIM / PSR CRATER (256 × 256)
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Observation Preview ──────────────────────────────────────────────────────

function ImagePreview() {
  const inputInfo     = useStore((s) => s.inputInfo);
  const rawImageUrl   = useStore((s) => s.rawImageUrl);
  const inputFile     = useStore((s) => s.inputFile);
  const setInputFile  = useStore((s) => s.setInputFile);
  const runPipeline   = useStore((s) => s.runPipeline);
  const setHazardMode = useStore((s) => s.setHazardMode);

  // Enforce validated balanced mode internally
  useEffect(() => {
    setHazardMode('balanced');
  }, [setHazardMode]);

  return (
    <div className="av-preview-wrap">
      <div className="av-preview-header">
        <div className="av-eyebrow">OBSERVATION STAGED</div>
        <h2 className="av-preview-title">LUNAR SOUTH POLE CANDIDATE</h2>
        <div className="av-preview-sub">LUNAR SITE ANALYSIS PIPELINE</div>
      </div>

      <div className="av-preview-img-frame">
        <img src={rawImageUrl} alt="Observation Preview" className="av-preview-img" />
        <div className="av-preview-badge">RAW OBSERVATION INPUT</div>
      </div>

      <div className="av-preview-meta">
        <div className="av-meta-item">
          <span className="av-meta-label">SOURCE</span>
          <span className="av-meta-value">{inputInfo?.filename || 'lunar_observation.png'}</span>
        </div>
        <div className="av-meta-item">
          <span className="av-meta-label">DIMENSIONS</span>
          <span className="av-meta-value">
            {inputInfo ? `${inputInfo.width} × ${inputInfo.height} px` : '256 × 256 px'}
          </span>
        </div>
        <div className="av-meta-item">
          <span className="av-meta-label">SENSOR</span>
          <span className="av-meta-value">OHRC 0.25M OPTICAL</span>
        </div>
      </div>

      <div className="av-preview-actions">
        <button
          className="av-run-btn"
          onClick={() => inputFile && runPipeline(inputFile)}
        >
          <Play size={15} />
          RUN SITE ANALYSIS
        </button>
        <button className="av-clear-btn" onClick={() => setInputFile(null)}>
          CHANGE IMAGE
        </button>
      </div>
    </div>
  );
}

// ── Processing State ─────────────────────────────────────────────────────────

function ProcessingState() {
  const pipelineStages = useStore((s) => s.pipelineStages);
  const rawImageUrl    = useStore((s) => s.rawImageUrl);
  const events         = useStore((s) => s.events);

  const STAGE_LABELS = {
    INGEST:          'Observation Ingestion',
    VALIDATION:      'Payload Validation',
    RESTORATION:     'U-Net Optical Restoration',
    HAZARD_ANALYSIS: 'V2 Hazard Detection',
    LZ_DETECTION:    'Landing Zone Detection',
    TELEMETRY:       'Telemetry Assembly',
  };

  const stageIcon = (status) => {
    if (status === 'COMPLETE') return <span className="ps-stage-icon ps-stage-icon--done">✓</span>;
    if (status === 'RUNNING')  return <span className="ps-stage-icon ps-stage-icon--run" />;
    if (status === 'FAILED')   return <span className="ps-stage-icon ps-stage-icon--fail">✕</span>;
    return <span className="ps-stage-icon ps-stage-icon--wait">○</span>;
  };

  const recentEvents = events.slice(-6);

  return (
    <div className="av-processing-wrap">
      <div className="av-scan-frame">
        {rawImageUrl && (
          <img src={rawImageUrl} alt="Processing observation" className="av-scan-img" />
        )}
        <div className="av-scan-overlay" aria-hidden="true">
          <div className="av-scan-line" />
          <div className="av-scan-grid" />
        </div>
        <div className="av-scan-badge">
          <RefreshCw size={11} className="rotating" />
          ANALYZING OBSERVATION
        </div>
      </div>

      <div className="ps-stages">
        {pipelineStages.map((stage) => (
          <div key={stage.name} className={`ps-stage ps-stage--${stage.status.toLowerCase()}`}>
            {stageIcon(stage.status)}
            <span className="ps-stage-name">{STAGE_LABELS[stage.name] || stage.name}</span>
          </div>
        ))}
      </div>

      <div className="ps-events">
        {recentEvents.map((ev, i) => (
          <div key={i} className={`ps-event ps-event--${ev.type}`}>
            <span className="ps-event-ts">[{ev.timestamp}]</span>
            <span className="ps-event-text">{ev.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Error State ───────────────────────────────────────────────────────────────

function ErrorState() {
  const events      = useStore((s) => s.events);
  const inputFile   = useStore((s) => s.inputFile);
  const runPipeline = useStore((s) => s.runPipeline);
  const reset       = useStore((s) => s.reset);

  const lastError = [...events].reverse().find((e) => e.type === 'error');

  return (
    <div className="av-error-state">
      <AlertTriangle size={32} className="av-error-icon" />
      <div className="av-error-title">OBSERVATION ANALYSIS FAILED</div>
      {lastError && (
        <div className="av-error-detail">{lastError.text.replace(/^Pipeline failed: /, '')}</div>
      )}
      <div className="av-error-actions">
        {inputFile && (
          <button className="av-run-btn" onClick={() => runPipeline(inputFile)}>
            <RefreshCw size={13} />
            RETRY ANALYSIS
          </button>
        )}
        <button className="av-clear-btn" onClick={reset}>
          RESET MISSION
        </button>
      </div>
    </div>
  );
}

// ── Unified Analysis Visualization ───────────────────────────────────────────

function UnifiedAnalysisView() {
  const restoredImageUrl   = useStore((s) => s.restoredImageUrl);
  const dangerMapUrl       = useStore((s) => s.dangerMapUrl);
  const safeRegionUrl      = useStore((s) => s.safeRegionUrl);
  const landingZones       = useStore((s) => s.landingZones);
  const selectedLzId       = useStore((s) => s.selectedLzId);
  const selectLz           = useStore((s) => s.selectLz);
  const inputInfo          = useStore((s) => s.inputInfo);
  const enterTerrain       = useStore((s) => s.enterTerrain);
  const processingTimeline = useStore((s) => s.processingTimeline);

  // Layer visibility toggles (all enabled by default into ONE unified visualization)
  const [showHazards, setShowHazards] = useState(true);
  const [showSafe, setShowSafe]       = useState(true);
  const [showLz, setShowLz]           = useState(true);

  // Auto-select first LZ if none selected
  useEffect(() => {
    if (!selectedLzId && landingZones.length > 0) {
      selectLz(landingZones[0].id);
    }
  }, [selectedLzId, landingZones, selectLz]);

  const w = inputInfo?.width || 256;
  const h = inputInfo?.height || 256;
  const totalTime = processingTimeline?.total_ms;

  return (
    <div className="av-unified-wrap">
      {/* Header */}
      <div className="av-unified-header">
        <div className="av-eyebrow">MISSION ANALYSIS</div>
        <h2 className="av-unified-title">LUNAR SOUTH POLE OBSERVATION</h2>
        <div className="av-unified-sub">
          U-NET OPTICAL RESTORATION + V2 HAZARD EVALUATION + LANDING CANDIDATES
        </div>
      </div>

      {/* Main Unified Centerpiece Map */}
      <div className="av-unified-canvas-container">
        {/* Layer Visibility Toggles in top-right */}
        <div className="av-layer-toggles">
          <button
            className={`av-layer-chip${showHazards ? ' active' : ''}`}
            onClick={() => setShowHazards(!showHazards)}
          >
            <span className="av-chip-dot av-chip-dot--hazard" />
            HAZARDS
          </button>
          <button
            className={`av-layer-chip${showSafe ? ' active' : ''}`}
            onClick={() => setShowSafe(!showSafe)}
          >
            <span className="av-chip-dot av-chip-dot--safe" />
            SAFE AREAS
          </button>
          <button
            className={`av-layer-chip${showLz ? ' active' : ''}`}
            onClick={() => setShowLz(!showLz)}
          >
            <span className="av-chip-dot av-chip-dot--lz" />
            LZ TARGETS
          </button>
        </div>

        {/* Base: Restored Lunar Image */}
        <img
          src={restoredImageUrl}
          alt="Restored lunar observation"
          className="av-unified-base-img"
        />

        {/* Overlay 1: Hazards (subtle danger shading) */}
        {showHazards && dangerMapUrl && (
          <img
            src={dangerMapUrl}
            alt="Hazard map overlay"
            className="av-unified-overlay av-unified-overlay--hazards"
          />
        )}

        {/* Overlay 2: Safe traversal regions */}
        {showSafe && safeRegionUrl && (
          <img
            src={safeRegionUrl}
            alt="Safe region overlay"
            className="av-unified-overlay av-unified-overlay--safe"
          />
        )}

        {/* Overlay 3: Interactive LZ Target Reticles */}
        {showLz && (
          <div className="av-unified-lz-layer">
            {landingZones.map((lz) => {
              const left = (lz.center_x / w) * 100;
              const top  = (lz.center_y / h) * 100;
              const isSelected = lz.id === selectedLzId;

              return (
                <div
                  key={lz.id}
                  className={`av-unified-lz-reticle${isSelected ? ' selected' : ''}`}
                  style={{ left: `${left}%`, top: `${top}%` }}
                  onClick={() => selectLz(lz.id)}
                  title={`${lz.id} — Score: ${lz.score?.toFixed(1)}`}
                >
                  {/* Glowing target ring */}
                  <div className="av-reticle-ring" />
                  <div className="av-reticle-center" />

                  {/* Reticle badge */}
                  <div className="av-reticle-badge">
                    <span className="av-reticle-id">{lz.id}</span>
                    <span className="av-reticle-score">{lz.score?.toFixed(0)}</span>
                    {isSelected && <span className="av-reticle-target-tag">TARGET</span>}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Map Corner Labels */}
        <div className="av-map-corner av-map-corner--tl">CHANDRAYAAN-2 OHRC 0.25M</div>
        <div className="av-map-corner av-map-corner--br">PERMANENTLY SHADOWED REGION</div>
      </div>

      {/* Bottom Mission CTA Bar */}
      <div className="av-unified-footer">
        <div className="av-footer-status">
          <span className="av-cta-dot" />
          <span className="av-footer-text">
            ANALYSIS COMPLETE · {landingZones.length} CANDIDATE LANDING ZONES
          </span>
          {selectedLzId && (
            <span className="av-footer-target">
              SELECTED TARGET: <strong>{selectedLzId}</strong>
            </span>
          )}
          {totalTime && <span className="av-footer-time">{(totalTime / 1000).toFixed(1)}s</span>}
        </div>

        <button className="av-enter-site-btn" onClick={enterTerrain}>
          <span>ENTER SITE</span>
          <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}

// ── Root AnalysisViewport Component ──────────────────────────────────────────

export default function AnalysisViewport() {
  const missionStatus = useStore((s) => s.missionStatus);
  const missionScene  = useStore((s) => s.missionScene);
  const inputFile     = useStore((s) => s.inputFile);

  const isSetup    = missionScene === 'MISSION_SETUP';
  const isResults  = missionScene === 'ANALYSIS_RESULTS' || missionStatus === 'COMPLETE';

  const showUpload     = isSetup && !inputFile && missionStatus !== 'PROCESSING';
  const showPreview    = isSetup && !!inputFile && missionStatus !== 'PROCESSING';
  const showProcessing = missionStatus === 'PROCESSING';
  const showError      = missionStatus === 'ERROR';
  const showUnified    = isResults && missionStatus === 'COMPLETE';

  return (
    <div className="analysis-viewport">
      {showUpload     && <UploadZone />}
      {showPreview    && <ImagePreview />}
      {showProcessing && <ProcessingState />}
      {showError      && <ErrorState />}
      {showUnified    && <UnifiedAnalysisView />}
    </div>
  );
}
