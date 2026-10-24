/**
 * SpaceScene — Photorealistic Cinematic Lunar Intro (Scene 0).
 *
 * Visual Architecture:
 *   - Photometrically calibrated NASA LROC global Moon globe
 *   - Smooth spherical geometry with ZERO spiky displacement distortion
 *   - ACES Filmic Tone Mapping with balanced exposure (no crushed shadows, no blown-out whites)
 *   - Collimated solar lighting with realistic terminator and subtle earthshine fill
 *   - Spacecraft orbital approach camera (starts at 68% viewport framing, approaches South Pole)
 *   - Interactive raycasting: Clicking the Moon initiates orbital approach to the South Pole
 */
import { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import useStore from '../store';
import { MoonAsset } from '../assets';

// ── Astronomical Star Field (Subtle & Sparse, Moon is the Hero) ───────────────
function StarField({ count = 750 }) {
  const points = useMemo(() => {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);
      const r     = 420 + Math.random() * 260;
      positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [count]);

  return (
    <points geometry={points}>
      <pointsMaterial
        color="#e2e8f0"
        size={0.38}
        sizeAttenuation
        transparent
        opacity={0.55}
      />
    </points>
  );
}

// ── Cinematic Camera & Orbital Approach Controller ────────────────────────────
function CinematicCameraController({ timelineProgress, targetPoint, isUserTransition }) {
  const { camera } = useThree();

  useFrame(() => {
    const p = Math.max(0, Math.min(1, timelineProgress.current));

    if (isUserTransition && targetPoint.current) {
      // User clicked the Moon: fly camera toward clicked surface point / South Pole
      const target = targetPoint.current;
      const endPos = new THREE.Vector3(target.x * 1.18, Math.min(target.y, -4.5) * 1.18, target.z * 1.18);
      camera.position.lerp(endPos, 0.05);
      camera.lookAt(target.x, target.y, target.z);
      return;
    }

    // Suggested 10–13.5s cinematic orbital sequence:
    // 0.0 – 0.55 (0–7s): Moon occupies 68% of viewport, slow orbital drift
    // 0.55 – 0.80 (7–10s): Camera slowly approaches Moon
    // 0.80 – 1.0 (10–13.5s): Camera pitches down and translates toward South Pole
    let x, y, z, lookY;

    if (p < 0.55) {
      const tNorm = p / 0.55;
      const ease = Math.sin((tNorm * Math.PI) / 2);
      x = Math.sin(tNorm * 0.3) * 2.2;
      y = THREE.MathUtils.lerp(2.2, 1.4, ease);
      z = THREE.MathUtils.lerp(50.0, 42.0, ease);
      lookY = THREE.MathUtils.lerp(0.0, -0.4, ease);
    } else if (p < 0.80) {
      const tNorm = (p - 0.55) / 0.25;
      const ease = tNorm * tNorm;
      x = THREE.MathUtils.lerp(Math.sin(0.3) * 2.2, 1.4, ease);
      y = THREE.MathUtils.lerp(1.4, 0.4, ease);
      z = THREE.MathUtils.lerp(42.0, 32.0, ease);
      lookY = THREE.MathUtils.lerp(-0.4, -1.5, ease);
    } else {
      // Transition toward Lunar South Pole region
      const tNorm = (p - 0.80) / 0.20;
      const ease = tNorm * tNorm;
      x = THREE.MathUtils.lerp(1.4, 0.3, ease);
      y = THREE.MathUtils.lerp(0.4, -3.8, ease);
      z = THREE.MathUtils.lerp(32.0, 22.0, ease);
      lookY = THREE.MathUtils.lerp(-1.5, -6.5, ease);
    }

    camera.position.set(x, y, z);
    camera.lookAt(0, lookY, 0);
  });

  return null;
}

// ── Realistic Sunlight & Earthshine Fill Controller ───────────────────────────
function CinematicSunlight({ timelineProgress }) {
  const lightRef = useRef();

  useFrame(() => {
    if (!lightRef.current) return;
    const p = Math.max(0, Math.min(1, timelineProgress.current));
    // 0–2.5 sec: sunlight builds smoothly from darkness to full intensity
    if (p < 0.18) {
      lightRef.current.intensity = THREE.MathUtils.lerp(0.3, 2.8, p / 0.18);
    } else {
      lightRef.current.intensity = 2.8;
    }
  });

  return (
    <>
      {/* ONE dominant Sun DirectionalLight creating a natural, gradual terminator */}
      <directionalLight
        ref={lightRef}
        position={[105, 20, 50]}
        intensity={2.8}
        color="#fffdf8"
      />

      {/* Deep space ambient fill: keeps dark side subtly visible without crushing blacks */}
      <ambientLight intensity={0.045} color="#141720" />

      {/* Subtle secondary earthshine scatter on dark hemisphere */}
      <directionalLight
        position={[-70, -20, -45]}
        intensity={0.16}
        color="#384050"
      />
    </>
  );
}

// ── Root SpaceScene Component ────────────────────────────────────────────────
export default function SpaceScene() {
  const missionScene    = useStore((s) => s.missionScene);
  const setMissionScene = useStore((s) => s.setMissionScene);
  const addEvent        = useStore((s) => s.addEvent);

  const timelineProgress = useRef(0);
  const targetPointRef   = useRef(null);
  const [isUserTransition, setIsUserTransition] = useState(false);
  const [stage, setStage] = useState('INTRO');
  const [fadeOpacity, setFadeOpacity] = useState(0);

  // Total automatic duration: 13.5 seconds
  const totalDuration = 13500;

  const triggerMissionTransition = useCallback((customPoint = null) => {
    if (targetPointRef.current) return;
    targetPointRef.current = customPoint || new THREE.Vector3(0, -13.5, 3.0);
    setIsUserTransition(true);
    setStage('INITIALIZING');
    addEvent('Approaching Lunar South Pole PSR target', 'info');

    setTimeout(() => {
      setFadeOpacity(1);
    }, 700);

    setTimeout(() => {
      setMissionScene('MISSION_SETUP');
    }, 1500);
  }, [addEvent, setMissionScene]);

  // Main automatic timeline (10–13.5s)
  useEffect(() => {
    if (missionScene !== 'LANDING_PAGE') return;

    let animId;
    let startTime = null;

    const tick = (now) => {
      if (!startTime) startTime = now;
      const elapsed = now - startTime;
      const p = Math.min(1, elapsed / totalDuration);
      timelineProgress.current = p;

      if (!targetPointRef.current) {
        if (elapsed < 2000) {
          setStage('INTRO');
        } else if (elapsed < 10000) {
          setStage('TITLE');
        } else if (elapsed < 12500) {
          setStage('INITIALIZING');
        } else {
          setStage('FADING');
        }

        const fadeStart = totalDuration - 1000;
        if (elapsed > fadeStart) {
          const fadeP = (elapsed - fadeStart) / 1000;
          setFadeOpacity(Math.min(1, fadeP));
        } else {
          setFadeOpacity(0);
        }
      }

      if (elapsed < totalDuration && !targetPointRef.current) {
        animId = requestAnimationFrame(tick);
      } else if (!targetPointRef.current) {
        triggerMissionTransition();
      }
    };

    animId = requestAnimationFrame(tick);

    const handleKey = (e) => {
      if (e.key === 'Escape' || e.key === ' ' || e.key === 'Enter') {
        triggerMissionTransition();
      }
    };
    window.addEventListener('keydown', handleKey);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('keydown', handleKey);
    };
  }, [missionScene, totalDuration, triggerMissionTransition]);

  if (missionScene !== 'LANDING_PAGE' && missionScene !== 'MOON_INTERACTION') {
    return null;
  }

  return (
    <div className="space-scene" aria-label="Interactive Lunar Globe">
      <Canvas
        className="space-canvas"
        camera={{ position: [0, 2.2, 50], fov: 38, near: 0.1, far: 1800 }}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.05,
          outputColorSpace: THREE.SRGBColorSpace,
        }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.05;
          gl.outputColorSpace = THREE.SRGBColorSpace;
        }}
      >
        <color attach="background" args={['#000003']} />

        <StarField count={750} />
        <CinematicSunlight timelineProgress={timelineProgress} />

        <MoonAsset
          onMoonClick={(pt) => triggerMissionTransition(pt)}
          isTransitioning={isUserTransition}
        />

        <CinematicCameraController
          timelineProgress={timelineProgress}
          targetPoint={targetPointRef}
          isUserTransition={isUserTransition}
        />
      </Canvas>

      {/* Restrained aerospace typography overlay */}
      <div className="cinematic-overlay" aria-live="polite">
        <div className={`cinematic-text-block${stage === 'TITLE' ? ' visible' : ''}`}>
          <h1 className="cinematic-title">LUNAR PSR TERMINAL</h1>
          <p className="cinematic-subtitle">AI-ASSISTED LUNAR SOUTH POLE EXPLORATION</p>
        </div>

        <div className={`cinematic-status-block${stage === 'INITIALIZING' ? ' visible' : ''}`}>
          <div className="cinematic-status-dot" />
          <span className="cinematic-status-text">MISSION INITIALIZING — APPROACHING SOUTH POLE</span>
        </div>
      </div>

      {/* Skip button in bottom corner */}
      <button
        className="cinematic-skip-btn"
        onClick={() => triggerMissionTransition()}
        title="Skip intro to mission analysis"
      >
        SKIP INTRO
      </button>

      {/* Match transition fade overlay */}
      <div
        className="cinematic-fade-overlay"
        style={{
          opacity: fadeOpacity,
          pointerEvents: fadeOpacity > 0.1 ? 'all' : 'none',
        }}
      />
    </div>
  );
}
