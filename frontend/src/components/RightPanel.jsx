import { ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import useStore from '../store';
import LandingZoneCard from './LandingZoneCard';

export default function RightPanel() {
  const [limitationsExpanded, setLimitationsExpanded] = useState(false);

  const landingZones = useStore((state) => state.landingZones);
  const selectedLzId = useStore((state) => state.selectedLzId);
  const descentMode = useStore((state) => state.descentMode);
  const descentTelemetry = useStore((state) => state.descentTelemetry);
  const descentPhase = useStore((state) => state.descentPhase);
  const scientificLimitations = useStore((state) => state.scientificLimitations);
  const hazardAnalysis = useStore((state) => state.hazardAnalysis);
  const startDescent = useStore((state) => state.startDescent);
  const stopDescent = useStore((state) => state.stopDescent);

  const selectedLz = landingZones.find((lz) => lz.id === selectedLzId);

  const getScoreColor = (score) => {
    if (score >= 70) return '#10b981';
    if (score >= 40) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="right-panel">
      {selectedLz && (
        <div className="panel-card">
          <h3 className="panel-card-title">SELECTED LANDING ZONE</h3>
          <div className="lz-details">
            <div className="lz-detail-row">
              <span className="lz-detail-label">ID</span>
              <span className="lz-detail-value">{selectedLz.id}</span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">CENTROID</span>
              <span className="lz-detail-value">
                ({selectedLz.center_x}, {selectedLz.center_y})
              </span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">AREA</span>
              <span className="lz-detail-value">{selectedLz.area} px</span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">SUITABILITY</span>
              <span className="lz-detail-value" style={{ color: getScoreColor(selectedLz.score) }}>
                {selectedLz.score.toFixed(1)} <span className="heuristic-label">HEURISTIC</span>
              </span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">HAZARD DENSITY</span>
              <span className="lz-detail-value">{(selectedLz.hazard_density * 100).toFixed(1)}%</span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">CLEARANCE</span>
              <span className="lz-detail-value">{selectedLz.clearance?.toFixed(1) || 'N/A'} px</span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">EDGE DENSITY</span>
              <span className="lz-detail-value">{(selectedLz.edge_density * 100).toFixed(1)}%</span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">RANK</span>
              <span className="lz-detail-value">#{selectedLz.rank}</span>
            </div>
          </div>

          {selectedLz.selection_reasons && selectedLz.selection_reasons.length > 0 && (
            <div className="lz-reasons">
              <h4 className="lz-reasons-title">WHY SELECTED</h4>
              {selectedLz.selection_reasons.map((reason, idx) => (
                <div key={idx} className="lz-reason">
                  <span className="lz-reason-indicator positive">+</span>
                  <span className="lz-reason-text">{reason}</span>
                </div>
              ))}
              <div className="lz-reason">
                <span className="lz-reason-indicator negative">-</span>
                <span className="lz-reason-text">Terrain slope unavailable (no DEM)</span>
              </div>
              <div className="lz-reason">
                <span className="lz-reason-indicator negative">-</span>
                <span className="lz-reason-text">Real elevation data unavailable</span>
              </div>
            </div>
          )}

          {!descentMode && (
            <button className="descent-button" onClick={startDescent}>
              START DESCENT SIMULATION
            </button>
          )}
          {descentMode && (
            <button className="descent-button stop" onClick={stopDescent}>
              STOP DESCENT
            </button>
          )}
        </div>
      )}

      {hazardAnalysis && Object.keys(hazardAnalysis).length > 0 && (
        <div className="panel-card">
          <h3 className="panel-card-title">HAZARD SUMMARY</h3>
          <div className="lz-details">
            <div className="lz-detail-row">
              <span className="lz-detail-label">HAZARD AREA</span>
              <span className="lz-detail-value" style={{ color: '#ef4444' }}>
                {(hazardAnalysis.hazard_percentage * 100).toFixed(1)}%
              </span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">SAFE AREA</span>
              <span className="lz-detail-value" style={{ color: '#10b981' }}>
                {(hazardAnalysis.safe_percentage * 100).toFixed(1)}%
              </span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">EDGE DENSITY</span>
              <span className="lz-detail-value">{(hazardAnalysis.edge_density * 100).toFixed(1)}%</span>
            </div>
            <div className="lz-detail-row">
              <span className="lz-detail-label">CANNY</span>
              <span className="lz-detail-value">{hazardAnalysis.canny_low}/{hazardAnalysis.canny_high}</span>
            </div>
          </div>
        </div>
      )}

      {landingZones.length > 0 && (
        <div className="panel-card">
          <h3 className="panel-card-title">LZ CANDIDATES ({landingZones.length})</h3>
          <div className="lz-candidates-list">
            {landingZones.slice(0, 10).map((lz) => (
              <LandingZoneCard key={lz.id} lz={lz} />
            ))}
          </div>
        </div>
      )}

      <div className="panel-card">
        <h3 className="panel-card-title">DESCENT TELEMETRY</h3>
        <div className="telemetry-grid">
          <div className="telemetry-item">
            <span className="telemetry-label">ALTITUDE</span>
            <span className="telemetry-value">
              {descentMode ? `${descentTelemetry.altitude.toFixed(1)} m` : 'N/A'}
            </span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">VERTICAL VEL</span>
            <span className="telemetry-value">
              {descentMode ? `${descentTelemetry.vVel.toFixed(2)} m/s` : 'N/A'}
            </span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">HORIZONTAL VEL</span>
            <span className="telemetry-value">
              {descentMode ? `${descentTelemetry.hVel.toFixed(2)} m/s` : 'N/A'}
            </span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">TARGET ERROR</span>
            <span className="telemetry-value">
              {descentMode ? `${descentTelemetry.targetError.toFixed(1)} m` : 'N/A'}
            </span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">PHASE</span>
            <span className="telemetry-value">{descentMode ? descentPhase : 'N/A'}</span>
          </div>
          <div className="telemetry-item">
            <span className="telemetry-label">TIME</span>
            <span className="telemetry-value">
              {descentMode ? `${descentTelemetry.time.toFixed(1)} s` : 'N/A'}
            </span>
          </div>
        </div>
        {descentMode && <div className="simulated-label">SIMULATED</div>}
      </div>

      <div className="panel-card">
        <h3 className="panel-card-title">TERRAIN METRICS</h3>
        <div className="terrain-metrics">
          <div className="terrain-metric">
            <span className="terrain-metric-label">ELEVATION</span>
            <span className="terrain-metric-value unavailable">UNAVAILABLE (NO DEM)</span>
          </div>
          <div className="terrain-metric">
            <span className="terrain-metric-label">SLOPE</span>
            <span className="terrain-metric-value unavailable">UNAVAILABLE</span>
          </div>
          <div className="terrain-metric">
            <span className="terrain-metric-label">ROUGHNESS</span>
            <span className="terrain-metric-value unavailable">UNAVAILABLE</span>
          </div>
          <div className="terrain-metric">
            <span className="terrain-metric-label">DEM SOURCE</span>
            <span className="terrain-metric-value unavailable">UNAVAILABLE</span>
          </div>
        </div>
      </div>

      {scientificLimitations.length > 0 && (
        <div className="panel-card">
          <h3
            className="panel-card-title clickable"
            onClick={() => setLimitationsExpanded(!limitationsExpanded)}
          >
            SCIENTIFIC LIMITATIONS
            {limitationsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </h3>
          {limitationsExpanded && (
            <div className="limitations-list">
              {scientificLimitations.map((limitation, idx) => (
                <div key={idx} className="limitation-item">
                  {limitation}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
