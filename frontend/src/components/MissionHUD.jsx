/**
 * MissionHUD — minimal persistent overlay strip.
 *
 * Visible once the mission is active (past LANDING_PAGE).
 * Shows only mission-critical context. Technical data lives in ResearchDrawer.
 */
import { useState, useEffect, useRef } from 'react';
import useStore from '../store';

function UtcClock() {
  const [t, setT] = useState('');
  useEffect(() => {
    const tick = () => {
      const n = new Date();
      setT(
        `${String(n.getUTCHours()).padStart(2,'0')}:` +
        `${String(n.getUTCMinutes()).padStart(2,'0')}:` +
        `${String(n.getUTCSeconds()).padStart(2,'0')} UTC`
      );
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return <span className="hud-clock">{t}</span>;
}

const SCENE_LABELS = {
  MISSION_SETUP:    'MISSION SETUP',
  ANALYSIS:         'RESTORING IMAGE',
  ANALYSIS_RESULTS: 'ANALYSIS COMPLETE',
  TERRAIN:          'LUNAR TERRAIN',
  LANDING:          'DESCENT',
  TOUCHDOWN:        'TOUCHDOWN',
  ROVER:            'ROVER OPS',
  EVA:              'EVA',
  SCIENCE:          'SCIENCE',
};

export default function MissionHUD() {
  const missionScene   = useStore((s) => s.missionScene);
  const missionId      = useStore((s) => s.missionId);
  const missionStatus  = useStore((s) => s.missionStatus);
  const backendStatus  = useStore((s) => s.backendStatus);
  const hazardMode     = useStore((s) => s.hazardMode);
  const selectedLzId   = useStore((s) => s.selectedLzId);
  const descentMode    = useStore((s) => s.descentMode);
  const descentTelemetry = useStore((s) => s.descentTelemetry);
  const touchdownComplete = useStore((s) => s.touchdownComplete);
  const isRecording    = useStore((s) => s.isRecording);
  const setRecording   = useStore((s) => s.setRecording);
  const recordingUrl   = useStore((s) => s.recordingUrl);
  const reset          = useStore((s) => s.reset);

  const hidden = missionScene === 'LANDING_PAGE' || missionScene === 'MOON_INTERACTION';
  if (hidden) return null;

  const sceneLabel = SCENE_LABELS[missionScene] || missionScene;

  return (
    <div className="mission-hud">
      {/* Top strip */}
      <div className="hud-top">
        <div className="hud-brand">LUNAR PSR TERMINAL</div>

        <div className="hud-status-row">
          {/* Backend dot */}
          <span className={`hud-dot hud-dot--${backendStatus === 'online' ? 'on' : 'off'}`} />
          <span className="hud-scene">{sceneLabel}</span>
          {missionId && (
            <span className="hud-id">{missionId.replace('MISSION-', '').slice(-6)}</span>
          )}
        </div>

        <div className="hud-right-row">
          {/* Active flight telemetry — only shown during descent */}
          {descentMode && !touchdownComplete && (
            <div className="hud-telemetry">
              <span className="hud-tel-item">
                <span className="hud-tel-label">ALT</span>
                <span className="hud-tel-value">{descentTelemetry.altitude.toFixed(0)} m</span>
              </span>
              <span className="hud-tel-item">
                <span className="hud-tel-label">VEL</span>
                <span className="hud-tel-value">{descentTelemetry.vVel.toFixed(1)} m/s</span>
              </span>
              {selectedLzId && (
                <span className="hud-tel-item">
                  <span className="hud-tel-label">TGT</span>
                  <span className="hud-tel-value" style={{ color: '#00f0ff' }}>{selectedLzId}</span>
                </span>
              )}
              <span className="hud-simulated">SIM</span>
            </div>
          )}

          {/* Hazard mode pill — visible in terrain+ */}
          {['TERRAIN','LANDING','TOUCHDOWN','ROVER','EVA','SCIENCE'].includes(missionScene) && (
            <span className="hud-mode-pill">{hazardMode.toUpperCase()}</span>
          )}

          {/* Recording indicator */}
          {isRecording && <span className="hud-rec">● REC</span>}
          {recordingUrl && !isRecording && (
            <a href={recordingUrl} download="mission.webm" className="hud-dl">↓ SAVE</a>
          )}

          <UtcClock />

          {/* Reset */}
          <button className="hud-reset" onClick={reset} title="Reset mission">✕</button>
        </div>
      </div>
    </div>
  );
}
