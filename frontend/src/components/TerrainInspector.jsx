/**
 * TerrainInspector — Phase 10 terrain science inspection system.
 *
 * Architecture:
 *   • A Three.js Raycaster inside the Canvas detects pointer position on
 *     the terrain mesh every frame, writing world coords to a shared ref
 *     (no React state per-frame).
 *   • A throttled useEffect (~10 Hz) flushes the ref to React state so
 *     the DOM overlay panel updates without causing render-per-frame.
 *   • Pixel lookup for hazard/safe uses a single off-screen <canvas> drawn
 *     once per image URL — never recomputed per frame.
 *   • All coordinate conversions reuse the helpers already exported from
 *     TerrainViewport: pxToWorld / worldToUV / sampleHeightWorld.
 *   • No new scientific computations. Every displayed value comes from the
 *     existing height map or the backend image data.
 *
 * Scientific labelling:
 *   Displacement  → IMAGE-DERIVED DISPLACEMENT (not real elevation)
 *   Hazard        → from backend danger_map (oracle_multiscale_v2)
 *   LZ score      → HEURISTIC
 *   Resources     → UNAVAILABLE
 */
import { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { TERRAIN_SIZE, getObservationWorldSpan, worldToU, worldToV, pxToWorldX, pxToWorldZ } from '../types';

// ─── Shared coordinate helpers (same as TerrainViewport) ─────────────────────
/** World X/Z → normalised UV [0,1] */
function worldToUV(w, span = TERRAIN_SIZE) { return w / span + 0.5; }

/** Pixel coord [0, dim] → world coordinate */
function pxToWorld(px, dim, span = TERRAIN_SIZE) { return (px / dim - 0.5) * span; }

/** UV [0,1] → pixel coord */
function uvToPx(uv, dim) { return Math.round(uv * (dim - 1)); }

/** Sample height-map at world position */
function sampleHeightWorld(hm, wx, wz) {
  if (!hm) return 0;
  if (typeof hm.sample === 'function') return hm.sample(wx, wz);
  const xi = Math.round(Math.min(1, Math.max(0, worldToUV(wx))) * (hm.res - 1));
  const yi = Math.round(Math.min(1, Math.max(0, worldToUV(wz))) * (hm.res - 1));
  return hm.map[yi * hm.res + xi] ?? 0;
}

// ─── Off-screen pixel sampler ─────────────────────────────────────────────────
/**
 * Build a sampler from a data-URI image: given normalised UV (u,v) returns
 * the R-channel value [0,255] at that pixel.
 *
 * danger_map encoding: 255 = hazard, 0 = safe
 * safe_region encoding: 255 = safe,  0 = hazard
 */
export function buildPixelSampler(imageUrl, onReady) {
  if (!imageUrl) return null;
  const img = new Image();
  img.onload = () => {
    const cv = document.createElement('canvas');
    cv.width  = img.naturalWidth  || 256;
    cv.height = img.naturalHeight || 256;
    const ctx = cv.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const w = cv.width, h = cv.height;

    const sample = (u, v) => {
      const x = Math.min(w - 1, Math.max(0, Math.round(u * (w - 1))));
      const y = Math.min(h - 1, Math.max(0, Math.round(v * (h - 1))));
      const px = ctx.getImageData(x, y, 1, 1).data;
      return px[0]; // R channel
    };

    onReady({ sample, w, h });
  };
  img.src = imageUrl;
  return null;
}

// ─── Local profile SVG ────────────────────────────────────────────────────────
const PROFILE_W = 160;
const PROFILE_H = 40;
const PROFILE_SAMPLES = 32;

function LocalProfileSVG({ hmRef, wx, wz }) {
  const points = useMemo(() => {
    const hm = hmRef?.current;
    if (!hm) return null;
    const span = 15; // world-units sample span
    const pts = [];
    for (let i = 0; i < PROFILE_SAMPLES; i++) {
      const t = i / (PROFILE_SAMPLES - 1);
      const sx = wx - span / 2 + t * span;
      const h = sampleHeightWorld(hm, sx, wz);
      pts.push({ t, h });
    }
    const mn = Math.min(...pts.map((p) => p.h));
    const mx = Math.max(...pts.map((p) => p.h));
    const range = Math.max(mx - mn, 0.1);
    return pts.map((p) => ({
      x: p.t * PROFILE_W,
      y: PROFILE_H - ((p.h - mn) / range) * (PROFILE_H - 4) - 2,
      norm: (p.h - mn) / range,
    }));
  }, [hmRef, wx, wz]);

  if (!points) return null;

  const polyline = points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  // selected position marker
  const cx  = PROFILE_W / 2;
  const midY = points[Math.floor(PROFILE_SAMPLES / 2)]?.y ?? PROFILE_H / 2;

  return (
    <svg
      width={PROFILE_W}
      height={PROFILE_H}
      className="insp-profile-svg"
      aria-label="Image-derived displacement profile"
    >
      <polyline points={polyline} fill="none" stroke="#00f0ff" strokeWidth="1" opacity="0.7" />
      <line x1={cx} y1={0} x2={cx} y2={PROFILE_H} stroke="#ff0055" strokeWidth="1" strokeDasharray="2,2" />
      <circle cx={cx} cy={midY} r="2" fill="#ff0055" />
    </svg>
  );
}

// ─── Inspection Raycaster ──────────────────────────────────────────────────
export function InspectionRaycaster({ meshRef, resultRef, active }) {
  const { raycaster, pointer, camera } = useThree();

  useFrame(() => {
    if (!active || !meshRef.current) return;
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(meshRef.current, false);
    if (intersects.length > 0) {
      const pt = intersects[0].point;
      resultRef.current = {
        valid: true,
        point: pt,
        wx: pt.x,
        wy: pt.y,
        wz: pt.z,
      };
    } else {
      if (resultRef.current) resultRef.current.valid = false;
    }
  });

  return null;
}

// ─── Inspection Crosshair ───────────────────────────────────────────────────
export function InspectionCrosshair({ resultRef, active }) {
  const crosshairRef = useRef();

  useFrame(() => {
    if (!crosshairRef.current) return;
    const r = resultRef.current;
    if (active && r && r.valid && r.point) {
      crosshairRef.current.visible = true;
      crosshairRef.current.position.set(r.wx, r.wy + 0.1, r.wz);
    } else {
      crosshairRef.current.visible = false;
    }
  });

  return (
    <group ref={crosshairRef} visible={false}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.8, 1.0, 32]} />
        <meshBasicMaterial color="#00f0ff" transparent opacity={0.8} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0.05, 0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshBasicMaterial color="#00f0ff" />
      </mesh>
    </group>
  );
}

// ─── Inspector DOM Panel ────────────────────────────────────────────────────
export default function TerrainInspectorPanel({
  resultRef,
  hmRef,
  dangerSamplerRef,
  safeSamplerRef,
  inspectionActive,
  setInspectionActive,
}) {
  const inputInfo        = useStore((s) => s.inputInfo);
  const pipelineTelemetry = useStore((s) => s.pipelineTelemetry);
  const hazardMode        = useStore((s) => s.hazardMode);
  const imageStats        = useStore((s) => s.imageStats);
  const landingZones      = useStore((s) => s.landingZones);
  const terrainLayer      = useStore((s) => s.terrainLayer);

  const imgW = inputInfo?.width  || 256;
  const imgH = inputInfo?.height || 256;

  // Current inspection data — throttled to ~10 Hz
  const [snap, setSnap] = useState(null);

  useEffect(() => {
    if (!inspectionActive) { setSnap(null); return; }
    const id = setInterval(() => {
      const r = resultRef.current;
      if (!r?.valid) return;
      const u = worldToUV(r.wx);
      const v = worldToUV(r.wz);
      const px_x = uvToPx(u, imgW);
      const px_y = uvToPx(v, imgH);

      const hm = hmRef?.current;
      const disp = hm ? (sampleHeightWorld(hm, r.wx, r.wz)) : null;
      const normBrightness = disp != null ? Math.min(1, Math.max(0, disp / 2.4)) : null; // DISP_SCALE=2.4

      // Hazard/safe lookup from off-screen samplers
      const dS = dangerSamplerRef?.current;
      const sS = safeSamplerRef?.current;
      const hazardVal = dS ? dS.sample(u, v) : null;
      const safeVal   = sS ? sS.sample(u, v) : null;
      const isHazard  = hazardVal != null ? hazardVal > 127 : null;
      const isSafe    = safeVal   != null ? safeVal   > 127 : null;

      // LZ candidate lookup — check if this point is within any LZ bbox
      const nearLz = landingZones.find((lz) => {
        if (!lz.bbox) return false;
        const [bx, by, bw, bh] = lz.bbox;
        return px_x >= bx && px_x <= bx + bw && px_y >= by && px_y <= by + bh;
      }) ?? null;

      setSnap({ wx: r.wx, wz: r.wz, wy: r.wy, u, v, px_x, px_y,
                disp, normBrightness, isHazard, isSafe, nearLz });
    }, 100); // 10 Hz
    return () => clearInterval(id);
  }, [inspectionActive, imgW, imgH]);

  // Pipeline trace from existing state
  const restMeta  = pipelineTelemetry?.restoration;
  const hazardTel = pipelineTelemetry?.hazard;

  return (
    <div className={`terrain-inspector${inspectionActive ? ' active' : ''}`}>
      {/* Toggle button */}
      <button
        className={`insp-toggle${inspectionActive ? ' insp-toggle--on' : ''}`}
        onClick={() => setInspectionActive((v) => !v)}
        title="Toggle image analysis scan mode"
      >
        {inspectionActive ? '◎ SCAN SITE: ACTIVE' : '◎ SCAN SITE'}
      </button>

      {inspectionActive && (
        <div className="insp-panel">
          <div className="insp-title">IMAGE ANALYSIS SCAN</div>

          {snap ? (
            <>
              {/* ── Location ── */}
              <div className="insp-section-label">LOCATION</div>
              <div className="insp-row">
                <span className="insp-key">PIXEL</span>
                <span className="insp-val">({snap.px_x}, {snap.px_y})</span>
              </div>
              <div className="insp-row">
                <span className="insp-key">UV</span>
                <span className="insp-val">({snap.u.toFixed(3)}, {snap.v.toFixed(3)})</span>
              </div>
              <div className="insp-row">
                <span className="insp-key">WORLD</span>
                <span className="insp-val">
                  ({snap.wx.toFixed(1)}, {snap.wz.toFixed(1)})
                </span>
              </div>

              {/* ── Displacement ── */}
              <div className="insp-section-label" style={{ marginTop: 8 }}>
                IMAGE-DERIVED DISPLACEMENT
              </div>
              <div className="insp-row">
                <span className="insp-key">VALUE</span>
                <span className="insp-val">
                  {snap.disp != null ? snap.disp.toFixed(2) : '—'}
                </span>
              </div>
              <div className="insp-row">
                <span className="insp-key">NORMALISED</span>
                <span className="insp-val">
                  {snap.normBrightness != null ? snap.normBrightness.toFixed(3) : '—'}
                </span>
              </div>
              <div className="insp-disclaimer">
                NOT REAL ELEVATION — image brightness → visual displacement
              </div>

              {/* ── Local profile ── */}
              <div className="insp-section-label" style={{ marginTop: 6 }}>
                IMAGE-DERIVED PROFILE (H SLICE)
              </div>
              <LocalProfileSVG hmRef={hmRef} wx={snap.wx} wz={snap.wz} />
              <div className="insp-disclaimer">NOT AN ELEVATION PROFILE</div>

              {/* ── Hazard status ── */}
              <div className="insp-section-label" style={{ marginTop: 8 }}>
                HAZARD STATUS
                <span className="insp-source">V2 oracle backend</span>
              </div>
              <div className="insp-row">
                <span className="insp-key">HAZARD</span>
                <span className={`insp-val insp-val--${snap.isHazard === true ? 'hazard' : snap.isHazard === false ? 'safe' : 'unknown'}`}>
                  {snap.isHazard === true ? 'DETECTED' : snap.isHazard === false ? 'NOT DETECTED' : 'UNAVAILABLE'}
                </span>
              </div>
              <div className="insp-row">
                <span className="insp-key">SAFE REGION</span>
                <span className={`insp-val insp-val--${snap.isSafe === true ? 'safe' : snap.isSafe === false ? 'hazard' : 'unknown'}`}>
                  {snap.isSafe === true ? 'YES' : snap.isSafe === false ? 'NO' : 'UNAVAILABLE'}
                </span>
              </div>
              <div className="insp-row">
                <span className="insp-key">MODE</span>
                <span className="insp-val">{hazardMode.toUpperCase()}</span>
              </div>

              {/* ── LZ status ── */}
              <div className="insp-section-label" style={{ marginTop: 8 }}>LANDING ZONE</div>
              {snap.nearLz ? (
                <>
                  <div className="insp-row">
                    <span className="insp-key">ZONE</span>
                    <span className="insp-val">{snap.nearLz.label || `LZ-${snap.nearLz.id}`}</span>
                  </div>
                  <div className="insp-row">
                    <span className="insp-key">SCORE</span>
                    <span className="insp-val">{snap.nearLz.score != null ? snap.nearLz.score.toFixed(2) : '—'}</span>
                  </div>
                </>
              ) : (
                <div className="insp-row">
                  <span className="insp-key">STATUS</span>
                  <span className="insp-val">OUTSIDE CANDIDATES</span>
                </div>
              )}
            </>
          ) : (
            <div className="insp-hint">Hover pointer over lunar surface to inspect telemetry</div>
          )}
        </div>
      )}
    </div>
  );
}