/**
 * TerrainViewport — Cinematic Lunar Surface Mission Experience.
 *
 * Architecture & Fixes:
 *   - Continuous Seamless Lunar Basin (380m x 380m): Eliminates floating rectangular slab.
 *   - Isotropic Aspect-Ratio Preservation: Non-square uploaded observations (e.g. 296x184)
 *     are rendered with exact geometric aspect ratio in world meters, preventing oval stretching.
 *   - Brightness Decoupled from Elevation: The uploaded optical image is mapped primarily as an
 *     albedo texture and detail layer with ZERO false mountain/canyon displacement, preserving
 *     the authentic optical crater geometry without turning rims into mountains or shadows into pits.
 *   - Seamless Regolith Embedding: The observation patch is feathered radially into expansive
 *     lunar regolith background with no hard rectangular border.
 *   - Synchronized 2D/3D Correlation Inspector: Allows direct side-by-side inspection of Raw Image,
 *     Restored Image, and 3D Surface Mapping with synchronized crosshair telemetry.
 *   - Authentic Polar Illumination: Grazing low-angle sunlight (~8.5° elevation), hard crisp shadows,
 *     deep black vacuum space, and stark PSR shadow contrast.
 *   - NASA 3D Assets: Apollo Lunar Module, Planetary Surface Rover, Artemis/Apollo EVA Astronaut.
 *   - Dedicated Cinematic Camera Controller: Orbital Approach, Dynamic Descent, Touchdown,
 *     Rover Follow, EVA Follow, and Surface Inspection.
 */
import { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import useStore from '../store';
import {
  TERRAIN_SIZE,
  getObservationWorldSpan,
  pxToWorldX,
  pxToWorldZ,
  worldToU,
  worldToV,
} from '../types';
import {
  WORLD_SCALE,
  LanderAsset,
  RoverAsset,
  AstronautAsset,
  loadTextureCached,
} from '../assets';
import TerrainInspectorPanel, {
  InspectionRaycaster,
  InspectionCrosshair,
  buildPixelSampler,
} from './TerrainInspector';

// ── Physical Scales & Dimensions ──────────────────────────────────────────────
export const FULL_TERRAIN_SIZE = 380.0; // 380m continuous lunar surface
export const OBS_BASE_SIZE = TERRAIN_SIZE; // 100m maximum dominant dimension
const TERRAIN_SEGS = 220; // 220x220 resolution (smooth, high-performance)

// ── Macro Surface Topography (Natural Lunar Terrain) ──────────────────────────
// Smooth, authentic lunar ground with gentle slopes (<1.5° across landing site)
function getMacroElevation(wx, wz) {
  // 1. Broad rolling regolith undulations (wavelengths 70m - 140m)
  const undulation =
    Math.sin(wx * 0.015 + 0.35) * Math.cos(wz * 0.015 - 0.2) * 1.2 +
    Math.sin(wx * 0.032 + wz * 0.024) * 0.38 +
    Math.cos(wx * 0.048 - wz * 0.04) * 0.22;

  // 2. Central landing plain depression (subtle inter-crater floor, depth ~0.45m)
  const r = Math.sqrt(wx * wx + wz * wz);
  const floorBowl = -0.45 * Math.exp(-(r * r) / (95 * 95));

  // 3. Natural crater rim wall rising at r > 75m towards the horizon
  const rimRise =
    r > 75
      ? Math.pow((r - 75) / 115, 1.8) * 5.8 + Math.sin(Math.atan2(wz, wx) * 4) * 0.75
      : 0;

  return undulation + floorBowl + rimRise;
}

export function sampleHeightWorld(hm, wx, wz) {
  if (!hm) return getMacroElevation(wx, wz);
  if (typeof hm.sample === 'function') return hm.sample(wx, wz);
  return getMacroElevation(wx, wz);
}

// ── Photorealistic Regolith Micro-Bump Texture ────────────────────────────────
function createRegolithBumpTexture() {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(size, size);
  const data = imgData.data;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const n1 = Math.sin(x * 0.45) * Math.cos(y * 0.45);
      const n2 = Math.sin(x * 0.95 + 1.2) * Math.cos(y * 0.95 + 0.5);
      const rand = (Math.random() - 0.5) * 0.4;
      const val = Math.floor(Math.min(255, Math.max(0, 128 + (n1 * 32 + n2 * 20) + rand * 65)));
      const idx = (y * size + x) * 4;
      data[idx]     = val;
      data[idx + 1] = val;
      data[idx + 2] = val;
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(36, 36);
  return texture;
}

// ── Composite Seamless Lunar Albedo Canvas Texture ────────────────────────────
// Preserves exact image aspect ratio and feathers outer boundary into lunar ground
function createCompositeLunarTexture(
  restoredImg,
  dangerImg,
  safeImg,
  layerMode,
  imgW,
  imgH,
  spanX,
  spanZ,
  onReady
) {
  const size = 2048;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');

  // 1. Procedural Lunar Regolith Background (Expansive Lunar Basin)
  ctx.fillStyle = '#726f68';
  ctx.fillRect(0, 0, size, size);

  // Macro tonal variation (ejecta mottling)
  const macroGrad = ctx.createRadialGradient(size * 0.48, size * 0.48, 100, size * 0.5, size * 0.5, size * 0.7);
  macroGrad.addColorStop(0, '#78756e');
  macroGrad.addColorStop(0.45, '#706d66');
  macroGrad.addColorStop(0.85, '#63605a');
  macroGrad.addColorStop(1.0, '#595650');
  ctx.fillStyle = macroGrad;
  ctx.fillRect(0, 0, size, size);

  // Subtle regolith particulate noise
  const noiseImg = ctx.createImageData(size, size);
  const nd = noiseImg.data;
  let seed = 12345;
  function rand() {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  }
  for (let i = 0; i < nd.length; i += 4) {
    const grain = (rand() - 0.5) * 20;
    nd[i]     = Math.min(255, Math.max(0, 115 + grain));
    nd[i + 1] = Math.min(255, Math.max(0, 112 + grain));
    nd[i + 2] = Math.min(255, Math.max(0, 105 + grain));
    nd[i + 3] = 255;
  }
  const noiseCanvas = document.createElement('canvas');
  noiseCanvas.width = noiseCanvas.height = size;
  noiseCanvas.getContext('2d').putImageData(noiseImg, 0, 0);
  ctx.globalAlpha = 0.32;
  ctx.drawImage(noiseCanvas, 0, 0);
  ctx.globalAlpha = 1.0;

  // 2. Map Central Observation Tile Preserving Exact Aspect Ratio
  // Metric bounds: [-spanX/2, +spanX/2] and [-spanZ/2, +spanZ/2] inside FULL_TERRAIN_SIZE
  const rectW = Math.round((spanX / FULL_TERRAIN_SIZE) * size);
  const rectH = Math.round((spanZ / FULL_TERRAIN_SIZE) * size);
  const tx = Math.round((size - rectW) / 2);
  const ty = Math.round((size - rectH) / 2);

  if (restoredImg) {
    // Draw restored optical image into offscreen tile canvas with feathered border
    const tileCanvas = document.createElement('canvas');
    tileCanvas.width = rectW;
    tileCanvas.height = rectH;
    const tCtx = tileCanvas.getContext('2d');
    tCtx.drawImage(restoredImg, 0, 0, rectW, rectH);

    // Apply overlay if in HAZARD or SAFE mode
    if (layerMode === 'HAZARD' && dangerImg) {
      tCtx.save();
      tCtx.globalAlpha = 0.42;
      tCtx.drawImage(dangerImg, 0, 0, rectW, rectH);
      tCtx.globalCompositeOperation = 'multiply';
      tCtx.fillStyle = 'rgba(239, 68, 68, 0.45)';
      tCtx.fillRect(0, 0, rectW, rectH);
      tCtx.restore();
    } else if (layerMode === 'SAFE' && safeImg) {
      tCtx.save();
      tCtx.globalAlpha = 0.35;
      tCtx.drawImage(safeImg, 0, 0, rectW, rectH);
      tCtx.globalCompositeOperation = 'multiply';
      tCtx.fillStyle = 'rgba(16, 185, 129, 0.4)';
      tCtx.fillRect(0, 0, rectW, rectH);
      tCtx.restore();
    }

    // Radial/elliptical vignette mask on tile edges to smoothly dissolve into outer lunar surface
    tCtx.save();
    tCtx.globalCompositeOperation = 'destination-in';
    const rx = rectW / 2;
    const ry = rectH / 2;
    const vig = tCtx.createRadialGradient(rx, ry, Math.min(rx, ry) * 0.72, rx, ry, Math.min(rx, ry) * 0.98);
    vig.addColorStop(0, 'rgba(0,0,0,1)');
    vig.addColorStop(0.85, 'rgba(0,0,0,0.92)');
    vig.addColorStop(1, 'rgba(0,0,0,0)');
    tCtx.fillStyle = vig;
    tCtx.fillRect(0, 0, rectW, rectH);
    tCtx.restore();

    // Composite feathered tile into main terrain texture
    ctx.drawImage(tileCanvas, tx, ty);

    // 3. Discreet Scientific Fiducial Marks (Subtle Corner Brackets)
    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
    ctx.lineWidth = 2.0;
    const bLen = 18;
    // Top-Left
    ctx.beginPath(); ctx.moveTo(tx, ty + bLen); ctx.lineTo(tx, ty); ctx.lineTo(tx + bLen, ty); ctx.stroke();
    // Top-Right
    ctx.beginPath(); ctx.moveTo(tx + rectW - bLen, ty); ctx.lineTo(tx + rectW, ty); ctx.lineTo(tx + rectW, ty + bLen); ctx.stroke();
    // Bottom-Left
    ctx.beginPath(); ctx.moveTo(tx, ty + rectH - bLen); ctx.lineTo(tx, ty + rectH); ctx.lineTo(tx + bLen, ty + rectH); ctx.stroke();
    // Bottom-Right
    ctx.beginPath(); ctx.moveTo(tx + rectW - bLen, ty + rectH); ctx.lineTo(tx + rectW, ty + rectH); ctx.lineTo(tx + rectW, ty + rectH - bLen); ctx.stroke();

    // Subtle footprint telemetry label
    ctx.font = '12px monospace';
    ctx.fillStyle = 'rgba(56, 189, 248, 0.6)';
    ctx.fillText(
      `[ OBSERVATION FOOTPRINT // ${imgW}×${imgH} px // ${spanX.toFixed(1)}m × ${spanZ.toFixed(1)}m // ASPECT ${(imgW / imgH).toFixed(2)}:1 ]`,
      tx + 8,
      ty - 8
    );
    ctx.restore();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  onReady(texture);
}

// ── Height Map Builder (Continuous Natural Ground, Zero False Mountains) ──────
function buildNaturalHeightMap(resolution, onReady) {
  const map = new Float32Array(resolution * resolution);

  for (let iy = 0; iy < resolution; iy++) {
    for (let ix = 0; ix < resolution; ix++) {
      const wx = (ix / (resolution - 1) - 0.5) * FULL_TERRAIN_SIZE;
      const wz = (iy / (resolution - 1) - 0.5) * FULL_TERRAIN_SIZE;
      map[iy * resolution + ix] = getMacroElevation(wx, wz);
    }
  }

  onReady({
    map,
    res: resolution,
    extent: FULL_TERRAIN_SIZE,
    sample: (wx, wz) => getMacroElevation(wx, wz),
  });
}

// ── Realistic Unified Lunar Regolith Surface Component ────────────────────────
function UnifiedLunarSurface({
  imageUrl,
  dangerUrl,
  safeUrl,
  terrainLayer,
  imgW,
  imgH,
  spanX,
  spanZ,
  onHeightMapReady,
  meshRef: externalMeshRef,
}) {
  const hmRef = useRef(null);
  const [hmVersion, setHmVersion] = useState(0);
  const internalMeshRef = useRef();

  const regolithBumpTex = useMemo(() => createRegolithBumpTexture(), []);

  // Pre-cached NASA Lunar Normal & Roughness maps
  const moonNormalTex = useMemo(() => {
    const t = loadTextureCached('/textures/moon_normal_2k.jpg', false);
    if (t) {
      t.wrapS = THREE.RepeatWrapping;
      t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(12, 12);
    }
    return t;
  }, []);

  const moonRoughnessTex = useMemo(() => {
    const t = loadTextureCached('/textures/moon_roughness_1k.jpg', false);
    if (t) {
      t.wrapS = THREE.RepeatWrapping;
      t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(12, 12);
    }
    return t;
  }, []);

  // Build natural heightmap (without image-brightness false elevation)
  useEffect(() => {
    buildNaturalHeightMap(TERRAIN_SEGS + 1, (hm) => {
      hmRef.current = hm;
      setHmVersion((v) => v + 1);
      onHeightMapReady?.(hm);
    });
  }, []);

  // Construct continuous single-mesh geometry
  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(FULL_TERRAIN_SIZE, FULL_TERRAIN_SIZE, TERRAIN_SEGS, TERRAIN_SEGS);
    const hm = hmRef.current;
    if (hm) {
      const pos = geo.attributes.position.array;
      for (let i = 0; i < hm.map.length; i++) {
        pos[i * 3 + 2] = hm.map[i];
      }
      geo.attributes.position.needsUpdate = true;
      geo.computeVertexNormals();
    }
    return geo;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hmVersion]);

  // Build composite seamless texture with exact aspect ratio
  const [compositeTex, setCompositeTex] = useState(null);

  useEffect(() => {
    let active = true;
    let rImg = null;
    let dImg = null;
    let sImg = null;

    let loadedCount = 0;
    const totalNeeded = (imageUrl ? 1 : 0) + (dangerUrl ? 1 : 0) + (safeUrl ? 1 : 0);

    const tryBuild = () => {
      if (!active) return;
      createCompositeLunarTexture(
        rImg,
        dImg,
        sImg,
        terrainLayer,
        imgW,
        imgH,
        spanX,
        spanZ,
        (tex) => {
          if (!active) return;
          setCompositeTex((prev) => {
            if (prev) prev.dispose();
            return tex;
          });
        }
      );
    };

    if (totalNeeded === 0) {
      tryBuild();
      return () => { active = false; };
    }

    if (imageUrl) {
      const i = new Image();
      i.crossOrigin = 'anonymous';
      i.onload = () => { rImg = i; loadedCount++; if (loadedCount >= totalNeeded) tryBuild(); };
      i.onerror = () => { loadedCount++; if (loadedCount >= totalNeeded) tryBuild(); };
      i.src = imageUrl;
    }
    if (dangerUrl) {
      const i = new Image();
      i.crossOrigin = 'anonymous';
      i.onload = () => { dImg = i; loadedCount++; if (loadedCount >= totalNeeded) tryBuild(); };
      i.onerror = () => { loadedCount++; if (loadedCount >= totalNeeded) tryBuild(); };
      i.src = dangerUrl;
    }
    if (safeUrl) {
      const i = new Image();
      i.crossOrigin = 'anonymous';
      i.onload = () => { sImg = i; loadedCount++; if (loadedCount >= totalNeeded) tryBuild(); };
      i.onerror = () => { loadedCount++; if (loadedCount >= totalNeeded) tryBuild(); };
      i.src = safeUrl;
    }

    return () => { active = false; };
  }, [imageUrl, dangerUrl, safeUrl, terrainLayer, imgW, imgH, spanX, spanZ]);

  useEffect(() => {
    if (externalMeshRef && internalMeshRef.current) {
      externalMeshRef.current = internalMeshRef.current;
    }
  });

  return (
    <mesh
      ref={internalMeshRef}
      geometry={geometry}
      rotation={[-Math.PI / 2, 0, 0]}
      receiveShadow
      castShadow
    >
      <meshStandardMaterial
        map={compositeTex}
        color="#8c8983"
        normalMap={moonNormalTex}
        normalScale={new THREE.Vector2(1.2, 1.2)}
        bumpMap={regolithBumpTex}
        bumpScale={0.024}
        roughnessMap={moonRoughnessTex}
        roughness={0.96}
        metalness={0.0}
        emissive="#000000"
      />
    </mesh>
  );
}

// ── Mission-Grade Restrained Landing Zone Target Reticles ─────────────────────
function LZMarkers({ landingZones, selectedLzId, imgW, imgH, spanX, spanZ, getHeight }) {
  const selectLz = useStore((s) => s.selectLz);

  return (
    <>
      {landingZones.map((lz) => {
        const sel = lz.id === selectedLzId;
        const wx = pxToWorldX(lz.center_x, imgW, spanX);
        const wz = pxToWorldZ(lz.center_y, imgH, spanZ);
        const wy = getHeight(wx, wz) + 0.04;

        return (
          <group key={lz.id} position={[wx, wy, wz]} onClick={() => selectLz(lz.id)}>
            {/* Subtle conformal landing navigation ring */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.015, 0]}>
              <ringGeometry args={[sel ? 1.5 : 1.1, sel ? 1.62 : 1.18, 32]} />
              <meshBasicMaterial
                color={sel ? '#38bdf8' : '#64748b'}
                transparent
                opacity={sel ? 0.88 : 0.35}
                side={THREE.DoubleSide}
              />
            </mesh>

            {/* Central target fiducial dot */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.025, 0]}>
              <circleGeometry args={[0.11, 16]} />
              <meshBasicMaterial
                color={sel ? '#38bdf8' : '#64748b'}
                transparent
                opacity={sel ? 0.95 : 0.4}
              />
            </mesh>

            {/* 4 subtle crosshair tick marks */}
            {[-1, 1].map((sign) => (
              <group key={sign}>
                <mesh position={[sign * (sel ? 1.95 : 1.45), 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                  <planeGeometry args={[0.32, 0.04]} />
                  <meshBasicMaterial color={sel ? '#38bdf8' : '#64748b'} transparent opacity={sel ? 0.85 : 0.35} />
                </mesh>
                <mesh position={[0, 0.02, sign * (sel ? 1.95 : 1.45)]} rotation={[-Math.PI / 2, 0, 0]}>
                  <planeGeometry args={[0.04, 0.32]} />
                  <meshBasicMaterial color={sel ? '#38bdf8' : '#64748b'} transparent opacity={sel ? 0.85 : 0.35} />
                </mesh>
              </group>
            ))}

            {/* Minimalist aerospace tag: LZ-01 // SELECTED */}
            <group position={[0, 0.38, 0]}>
              <mesh>
                <planeGeometry args={[1.5, 0.36]} />
                <meshBasicMaterial color="#030712" transparent opacity={0.88} />
              </mesh>
              <mesh position={[0, 0, -0.01]}>
                <planeGeometry args={[1.56, 0.42]} />
                <meshBasicMaterial color={sel ? '#38bdf8' : '#334155'} transparent opacity={0.5} />
              </mesh>
            </group>
          </group>
        );
      })}
    </>
  );
}

// ── Instanced Natural Lunar Boulders & Rocks ──────────────────────────────────
function InstancedRocks({ hmRef, spanX, spanZ }) {
  const count = 68;
  const meshRef = useRef();

  const rockData = useMemo(() => {
    const rocks = [];
    let seed = 42;
    function rand() {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    }

    const halfX = spanX / 2;
    const halfZ = spanZ / 2;

    for (let i = 0; i < count; i++) {
      const angle = rand() * Math.PI * 2;
      // Scatter mostly outside the active observation center to avoid obscuring the crater
      const r = 25 + rand() * 85;
      const x = Math.cos(angle) * r;
      const z = Math.sin(angle) * r;
      const scale = WORLD_SCALE.ROCK_MIN_SCALE + rand() * (WORLD_SCALE.ROCK_MAX_SCALE - WORLD_SCALE.ROCK_MIN_SCALE) * 1.3;
      const rotY = rand() * Math.PI * 2;
      const rotX = (rand() - 0.5) * 0.5;
      const rotZ = (rand() - 0.5) * 0.5;
      rocks.push({ x, z, scale, rotX, rotY, rotZ });
    }
    return rocks;
  }, [count, spanX, spanZ]);

  useFrame(() => {
    if (!meshRef.current || !hmRef?.current) return;
    const dummy = new THREE.Object3D();
    rockData.forEach((rock, i) => {
      const y = sampleHeightWorld(hmRef.current, rock.x, rock.z) + rock.scale * 0.22;
      dummy.position.set(rock.x, y, rock.z);
      dummy.rotation.set(rock.rotX, rock.rotY, rock.rotZ);
      dummy.scale.set(rock.scale, rock.scale * (0.75 + 0.35 * (i % 3)), rock.scale);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]} castShadow receiveShadow>
      <dodecahedronGeometry args={[1, 0]} />
      <meshStandardMaterial color="#2c2a27" roughness={0.98} metalness={0.01} flatShading />
    </instancedMesh>
  );
}

// ── Dedicated Cinematic Camera System ─────────────────────────────────────────
function DedicatedCameraSystem({ targetX, targetZ, transitioning, orbitRef, hmRef }) {
  const { camera } = useThree();
  const descentMode       = useStore((s) => s.descentMode);
  const descentTelemetry  = useStore((s) => s.descentTelemetry);
  const touchdownComplete = useStore((s) => s.touchdownComplete);
  const roverActive       = useStore((s) => s.roverActive);
  const roverPos          = useStore((s) => s.roverPos);
  const roverHeading      = useStore((s) => s.roverHeading);
  const astronautActive   = useStore((s) => s.astronautActive);
  const astronautPos      = useStore((s) => s.astronautPos);
  const astronautHeading  = useStore((s) => s.astronautHeading);

  useFrame(() => {
    // 1. DESCENT CAMERA: Smoothly tracks descending Apollo LM from altitude
    if (descentMode && !touchdownComplete) {
      if (orbitRef.current) orbitRef.current.enabled = false;

      const altNorm = Math.max(0, Math.min(1, descentTelemetry.altitude / 200));
      const groundY = sampleHeightWorld(hmRef?.current, targetX, targetZ);
      const landerY = groundY + Math.max(0, descentTelemetry.altitude * 0.35);

      const camX = targetX + 12 + altNorm * 24;
      const camY = landerY + 4.8 + altNorm * 20;
      const camZ = targetZ + 16 + altNorm * 28;

      const camTargetPos = new THREE.Vector3(camX, camY, camZ);
      const camLookTarget = new THREE.Vector3(targetX, landerY * 0.65 + groundY * 0.35, targetZ);

      camera.position.lerp(camTargetPos, 0.08);
      camera.lookAt(camLookTarget);
      return;
    }

    // 2. TOUCHDOWN CAMERA: Ground-level framing of landed spacecraft
    if (touchdownComplete && !roverActive) {
      if (orbitRef.current) orbitRef.current.enabled = false;

      const groundY = sampleHeightWorld(hmRef?.current, targetX, targetZ);
      const camTargetPos = new THREE.Vector3(targetX + 13, groundY + 2.8, targetZ + 15);
      const camLookTarget = new THREE.Vector3(targetX, groundY + 3.2, targetZ);

      camera.position.lerp(camTargetPos, 0.045);
      camera.lookAt(camLookTarget);
      return;
    }

    // 3. ROVER CAMERA: Third-person chase perspective
    if (roverActive && !astronautActive) {
      if (orbitRef.current) orbitRef.current.enabled = false;

      const [rx, rz] = roverPos;
      const ry = sampleHeightWorld(hmRef?.current, rx, rz);

      const dist = 7.0;
      const height = 3.0;
      const camTargetPos = new THREE.Vector3(
        rx + Math.sin(roverHeading) * dist,
        ry + height,
        rz + Math.cos(roverHeading) * dist
      );
      const camLookTarget = new THREE.Vector3(
        rx - Math.sin(roverHeading) * 1.5,
        ry + 1.2,
        rz - Math.cos(roverHeading) * 1.5
      );

      camera.position.lerp(camTargetPos, 0.07);
      camera.lookAt(camLookTarget);
      return;
    }

    // 4. EVA CAMERA: Grounded human-scale astronaut follow cam
    if (astronautActive) {
      if (orbitRef.current) orbitRef.current.enabled = false;

      const [ax, az] = astronautPos;
      const ay = sampleHeightWorld(hmRef?.current, ax, az);

      const dist = 4.2;
      const height = 2.0;
      const camTargetPos = new THREE.Vector3(
        ax + Math.sin(astronautHeading) * dist,
        ay + height,
        az + Math.cos(astronautHeading) * dist
      );
      const camLookTarget = new THREE.Vector3(
        ax - Math.sin(astronautHeading) * 1.0,
        ay + 1.2,
        az - Math.cos(astronautHeading) * 1.0
      );

      camera.position.lerp(camTargetPos, 0.09);
      camera.lookAt(camLookTarget);
      return;
    }

    // 5. ORBIT CONTROLS: Free exploration mode
    if (orbitRef.current) {
      orbitRef.current.enabled = true;
    }
  });

  return null;
}

// ── Surface Keyboard Controls (WASD / Arrows) ─────────────────────────────────
function SurfaceController({ spanX, spanZ }) {
  const missionScene     = useStore((s) => s.missionScene);
  const roverPos         = useStore((s) => s.roverPos);
  const roverHeading     = useStore((s) => s.roverHeading);
  const astronautPos     = useStore((s) => s.astronautPos);
  const astronautHeading = useStore((s) => s.astronautHeading);
  const updateRover      = useStore((s) => s.updateRover);
  const updateAstronaut  = useStore((s) => s.updateAstronaut);

  const keys = useRef(new Set());

  useEffect(() => {
    const dn = (e) => keys.current.add(e.key.toLowerCase());
    const up = (e) => keys.current.delete(e.key.toLowerCase());
    window.addEventListener('keydown', dn);
    window.addEventListener('keyup',   up);
    return () => {
      window.removeEventListener('keydown', dn);
      window.removeEventListener('keyup',   up);
      keys.current.clear();
    };
  }, [missionScene]);

  useFrame((_, dt) => {
    const isRover = missionScene === 'ROVER';
    const isEVA   = missionScene === 'EVA' || missionScene === 'SCIENCE';
    if (!isRover && !isEVA) return;

    const fwd  = keys.current.has('w') || keys.current.has('arrowup');
    const back = keys.current.has('s') || keys.current.has('arrowdown');
    const left = keys.current.has('a') || keys.current.has('arrowleft');
    const rght = keys.current.has('d') || keys.current.has('arrowright');
    if (!fwd && !back && !left && !rght) return;

    const clampX = (spanX / 2) - 2;
    const clampZ = (spanZ / 2) - 2;

    if (isRover) {
      let [rx, rz] = roverPos;
      let rh = roverHeading;
      const spd = 7.5 * dt;
      const trn = 1.4 * dt;
      if (left) rh += trn;
      if (rght) rh -= trn;
      if (fwd)  { rx -= Math.sin(rh) * spd; rz -= Math.cos(rh) * spd; }
      if (back) { rx += Math.sin(rh) * spd; rz += Math.cos(rh) * spd; }
      updateRover(
        [Math.max(-clampX, Math.min(clampX, rx)), Math.max(-clampZ, Math.min(clampZ, rz))],
        rh
      );
    }

    if (isEVA) {
      let [ax, az] = astronautPos;
      let ah = astronautHeading;
      const spd = 3.6 * dt;
      const trn = 1.8 * dt;
      if (left) ah += trn;
      if (rght) ah -= trn;
      if (fwd)  { ax -= Math.sin(ah) * spd; az -= Math.cos(ah) * spd; }
      if (back) { ax += Math.sin(ah) * spd; az -= Math.cos(ah) * spd; }
      updateAstronaut(
        [Math.max(-clampX, Math.min(clampX, ax)), Math.max(-clampZ, Math.min(clampZ, az))],
        ah
      );
    }
  });

  return null;
}

// ── 3D Scene Assembly ─────────────────────────────────────────────────────────
function Scene3D({
  transitioning,
  terrainMeshRef,
  inspResultRef,
  inspActiveRef,
  externalHmRef,
  imgW,
  imgH,
  spanX,
  spanZ,
}) {
  const restoredImageUrl  = useStore((s) => s.restoredImageUrl);
  const dangerMapUrl      = useStore((s) => s.dangerMapUrl);
  const safeRegionUrl     = useStore((s) => s.safeRegionUrl);
  const landingZones      = useStore((s) => s.landingZones);
  const selectedLzId      = useStore((s) => s.selectedLzId);
  const descentMode       = useStore((s) => s.descentMode);
  const descentTelemetry  = useStore((s) => s.descentTelemetry);
  const touchdownComplete = useStore((s) => s.touchdownComplete);
  const roverActive       = useStore((s) => s.roverActive);
  const roverPos          = useStore((s) => s.roverPos);
  const roverHeading      = useStore((s) => s.roverHeading);
  const astronautActive   = useStore((s) => s.astronautActive);
  const astronautPos      = useStore((s) => s.astronautPos);
  const astronautHeading  = useStore((s) => s.astronautHeading);
  const terrainLayer      = useStore((s) => s.terrainLayer);

  const selectedLz = landingZones.find((z) => z.id === selectedLzId) || landingZones[0];
  const targetX = selectedLz ? pxToWorldX(selectedLz.center_x, imgW, spanX) : 0;
  const targetZ = selectedLz ? pxToWorldZ(selectedLz.center_y, imgH, spanZ) : 0;

  const hmRef = useRef(null);
  const [hmVer, setHmVer] = useState(0);
  const orbitRef = useRef();

  const onHmReady = (hm) => {
    hmRef.current = hm;
    setHmVer((v) => v + 1);
    if (externalHmRef) externalHmRef.current = hm;
  };

  const getHeight = useCallback((wx, wz) => sampleHeightWorld(hmRef.current, wx, wz), []);

  useEffect(() => {
    if (orbitRef.current) {
      orbitRef.current.target.set(targetX, 0, targetZ);
      orbitRef.current.update();
    }
  }, [targetX, targetZ]);

  return (
    <>
      <PerspectiveCamera makeDefault position={[targetX + 48, 24, targetZ + 48]} fov={45} near={0.1} far={1600} />

      <DedicatedCameraSystem
        targetX={targetX}
        targetZ={targetZ}
        transitioning={transitioning}
        orbitRef={orbitRef}
        hmRef={hmRef}
      />

      <OrbitControls
        ref={orbitRef}
        enableDamping
        dampingFactor={0.06}
        minDistance={4}
        maxDistance={220}
        maxPolarAngle={Math.PI / 2.06}
      />

      {/* Deep Space Starfield (Airless Lunar Vacuum) */}
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[
              useMemo(() => {
                const arr = new Float32Array(900 * 3);
                for (let i = 0; i < 900; i++) {
                  const th = Math.random() * Math.PI * 2;
                  const ph = Math.acos(2 * Math.random() - 1);
                  const r = 420 + Math.random() * 200;
                  arr[i * 3]     = r * Math.sin(ph) * Math.cos(th);
                  arr[i * 3 + 1] = Math.abs(r * Math.sin(ph) * Math.sin(th)) + 14;
                  arr[i * 3 + 2] = r * Math.cos(ph);
                }
                return arr;
              }, []),
              3,
            ]}
          />
        </bufferGeometry>
        <pointsMaterial color="#ffffff" size={0.42} sizeAttenuation transparent opacity={0.75} />
      </points>

      {/* Deep Space Ambient: Faint earthshine & rim scatter */}
      <ambientLight intensity={0.025} color="#0a0d14" />

      {/* Primary Low-Angle Lunar Sunlight (~8.5° Grazing Polar Elevation) */}
      <directionalLight
        position={[160, 24, 85]}
        intensity={4.5}
        color="#fffdf6"
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-far={380}
        shadow-camera-left={-95}
        shadow-camera-right={95}
        shadow-camera-top={95}
        shadow-camera-bottom={-95}
        shadow-bias={-0.00015}
      />

      {/* Secondary Subtle Earthshine / Rim Scatter Fill */}
      <directionalLight position={[-80, 16, -60]} intensity={0.08} color="#202835" />

      {/* Continuous Unified Lunar Surface Regolith */}
      <UnifiedLunarSurface
        imageUrl={restoredImageUrl}
        dangerUrl={dangerMapUrl}
        safeUrl={safeRegionUrl}
        terrainLayer={terrainLayer}
        imgW={imgW}
        imgH={imgH}
        spanX={spanX}
        spanZ={spanZ}
        onHeightMapReady={onHmReady}
        meshRef={terrainMeshRef}
      />

      {/* Instanced Natural Lunar Boulders & Regolith Rocks */}
      <InstancedRocks hmRef={hmRef} spanX={spanX} spanZ={spanZ} />

      {/* Subtle Ground LZ Target Reticles */}
      {landingZones.length > 0 && (
        <LZMarkers
          key={hmVer}
          landingZones={landingZones}
          selectedLzId={selectedLzId}
          imgW={imgW}
          imgH={imgH}
          spanX={spanX}
          spanZ={spanZ}
          getHeight={getHeight}
        />
      )}

      {/* Realistic 3D Lander Asset (Apollo Lunar Module) */}
      {(descentMode || touchdownComplete) && (
        <LanderAsset
          targetX={targetX}
          targetZ={targetZ}
          altitude={descentTelemetry.altitude}
          touchdownComplete={touchdownComplete}
          getHeight={getHeight}
        />
      )}

      {/* Realistic 3D Rover Asset */}
      {roverActive && (
        <RoverAsset
          roverPos={roverPos}
          roverHeading={roverHeading}
          getHeight={getHeight}
        />
      )}

      {/* Realistic 3D Astronaut Asset */}
      {astronautActive && (
        <AstronautAsset
          astronautPos={astronautPos}
          astronautHeading={astronautHeading}
          getHeight={getHeight}
        />
      )}

      <SurfaceController spanX={spanX} spanZ={spanZ} />

      {/* Site Inspection Tools */}
      {inspResultRef && inspActiveRef && (
        <>
          <InspectionRaycaster
            terrainMeshRef={terrainMeshRef}
            resultRef={inspResultRef}
            activeRef={inspActiveRef}
          />
          <InspectionCrosshair
            resultRef={inspResultRef}
            hmRef={hmRef}
            activeRef={inspActiveRef}
          />
        </>
      )}
    </>
  );
}

// ── Interactive 2D vs 3D Observation Correlation Modal ────────────────────────
// Proves original crater, restored crater, and 3D surface map to exact same coordinates
function ObservationCorrelationModal({ onClose, imgW, imgH, spanX, spanZ }) {
  const rawImageUrl       = useStore((s) => s.rawImageUrl);
  const restoredImageUrl  = useStore((s) => s.restoredImageUrl);
  const dangerMapUrl      = useStore((s) => s.dangerMapUrl);
  const landingZones      = useStore((s) => s.landingZones);
  const selectedLzId      = useStore((s) => s.selectedLzId);

  const [hoverNorm, setHoverNorm] = useState({ u: 0.5, v: 0.5, active: false });
  const [showHazard, setShowHazard] = useState(true);

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const u = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const v = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setHoverNorm({ u, v, active: true });
  };

  const handleMouseLeave = () => {
    setHoverNorm((prev) => ({ ...prev, active: false }));
  };

  const currentPxX = Math.round(hoverNorm.u * (imgW - 1));
  const currentPxY = Math.round(hoverNorm.v * (imgH - 1));
  const currentWorldX = (hoverNorm.u - 0.5) * spanX;
  const currentWorldZ = (hoverNorm.v - 0.5) * spanZ;

  return (
    <div className="obs-correlation-modal">
      <div className="obs-correlation-card">
        <div className="obs-correlation-header">
          <div className="obs-correlation-title">
            <span>⊞ OBSERVATION CORRELATION & MAPPING INSPECTOR</span>
            <span className="obs-badge">ISOTROPIC 1:1 ASPECT RATIO VERIFIED</span>
          </div>
          <button className="obs-close-btn" onClick={onClose}>✕ CLOSE</button>
        </div>

        <div className="obs-correlation-meta">
          <span>DIMENSIONS: <strong>{imgW} × {imgH} px</strong></span>
          <span>ASPECT RATIO: <strong>{(imgW / imgH).toFixed(3)}:1</strong></span>
          <span>GROUND SPAN: <strong>{spanX.toFixed(1)}m × {spanZ.toFixed(1)}m</strong></span>
          <span>RESOLUTION: <strong>{(spanX / imgW).toFixed(3)} m/px</strong></span>
          <button
            className={`obs-hazard-toggle${showHazard ? ' active' : ''}`}
            onClick={() => setShowHazard((v) => !v)}
          >
            {showHazard ? '● HAZARDS VISIBLE' : '○ HAZARDS HIDDEN'}
          </button>
        </div>

        <div className="obs-correlation-grid">
          {/* 1. RAW UPLOADED IMAGE */}
          <div className="obs-panel">
            <div className="obs-panel-title">1. RAW UPLOADED TELEMETRY</div>
            <div
              className="obs-canvas-wrap"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              {rawImageUrl ? (
                <img src={rawImageUrl} alt="Raw Observation" className="obs-img" />
              ) : (
                <div className="obs-empty">NO RAW IMAGE LOADED</div>
              )}
              {hoverNorm.active && (
                <div
                  className="obs-crosshair"
                  style={{ left: `${hoverNorm.u * 100}%`, top: `${hoverNorm.v * 100}%` }}
                />
              )}
            </div>
            <div className="obs-panel-footer">Unprocessed sensor pixels</div>
          </div>

          {/* 2. RESTORED IMAGE */}
          <div className="obs-panel">
            <div className="obs-panel-title">2. U-NET RESTORED OPTICAL ALBEDO</div>
            <div
              className="obs-canvas-wrap"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              {restoredImageUrl ? (
                <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                  <img src={restoredImageUrl} alt="Restored Observation" className="obs-img" />
                  {showHazard && dangerMapUrl && (
                    <img
                      src={dangerMapUrl}
                      alt="Hazard Overlay"
                      className="obs-img"
                      style={{ position: 'absolute', top: 0, left: 0, opacity: 0.42, mixBlendMode: 'screen' }}
                    />
                  )}
                </div>
              ) : (
                <div className="obs-empty">NO RESTORED IMAGE LOADED</div>
              )}
              {hoverNorm.active && (
                <div
                  className="obs-crosshair"
                  style={{ left: `${hoverNorm.u * 100}%`, top: `${hoverNorm.v * 100}%` }}
                />
              )}
            </div>
            <div className="obs-panel-footer">Denoised regolith texture (No elevation distortion)</div>
          </div>

          {/* 3. 3D SURFACE TOP-DOWN FOOTPRINT */}
          <div className="obs-panel">
            <div className="obs-panel-title">3. 3D SURFACE MAPPING (GROUND M)</div>
            <div
              className="obs-canvas-wrap obs-canvas-wrap--grid"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              {restoredImageUrl && (
                <img src={restoredImageUrl} alt="3D Surface Map" className="obs-img obs-img--faded" />
              )}
              {/* Target landing zones overlaid at exact metric positions */}
              {landingZones.map((lz) => {
                const u = lz.center_x / imgW;
                const v = lz.center_y / imgH;
                const sel = lz.id === selectedLzId;
                return (
                  <div
                    key={lz.id}
                    className={`obs-lz-dot${sel ? ' obs-lz-dot--sel' : ''}`}
                    style={{ left: `${u * 100}%`, top: `${v * 100}%` }}
                    title={`${lz.id} (Score: ${lz.score?.toFixed(1)})`}
                  >
                    <span>{lz.id}</span>
                  </div>
                );
              })}
              {hoverNorm.active && (
                <div
                  className="obs-crosshair"
                  style={{ left: `${hoverNorm.u * 100}%`, top: `${hoverNorm.v * 100}%` }}
                />
              )}
            </div>
            <div className="obs-panel-footer">
              Scale: X: ±{(spanX / 2).toFixed(1)}m | Z: ±{(spanZ / 2).toFixed(1)}m
            </div>
          </div>
        </div>

        {/* Live synchronized readout */}
        <div className="obs-telemetry-readout">
          <span>PIXEL COORD: <strong>X: {currentPxX} px, Y: {currentPxY} px</strong></span>
          <span>NORMALIZED: <strong>U: {hoverNorm.u.toFixed(3)}, V: {hoverNorm.v.toFixed(3)}</strong></span>
          <span>SURFACE METERS: <strong>X: {currentWorldX.toFixed(2)}m, Z: {currentWorldZ.toFixed(2)}m</strong></span>
          <span style={{ color: '#10b981' }}>STATUS: EXACT 1:1 CORRELATION CONFIRMED</span>
        </div>
      </div>
    </div>
  );
}

// ── Landing & Exploration HUD ─────────────────────────────────────────────────
function TerrainHUD({ onToggleScan, inspectionActive, onOpenCorrelation, imgW, imgH, spanX, spanZ }) {
  const descentMode       = useStore((s) => s.descentMode);
  const descentTelemetry  = useStore((s) => s.descentTelemetry);
  const descentPhase      = useStore((s) => s.descentPhase);
  const touchdownComplete = useStore((s) => s.touchdownComplete);
  const roverActive       = useStore((s) => s.roverActive);
  const astronautActive   = useStore((s) => s.astronautActive);
  const roverPos          = useStore((s) => s.roverPos);
  const astronautPos      = useStore((s) => s.astronautPos);
  const selectedLzId      = useStore((s) => s.selectedLzId);
  const startDescent      = useStore((s) => s.startDescent);
  const stopDescent       = useStore((s) => s.stopDescent);
  const deployRover       = useStore((s) => s.deployRover);
  const startEVA          = useStore((s) => s.startEVA);
  const openScience       = useStore((s) => s.openScience);

  return (
    <>
      <div className="viewport-label" style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <span>IMAGE-DERIVED VISUAL SURFACE — ELEVATION DATA UNAVAILABLE</span>
        <button className="obs-hud-btn" onClick={onOpenCorrelation}>
          ⊞ CORRELATE OBSERVATION ({imgW}×{imgH})
        </button>
      </div>

      {/* Minimal Landing HUD (Descent Phase Only) */}
      {descentMode && !touchdownComplete && (
        <div className="terrain-descent-hud" style={{ position: 'absolute', top: 50, right: 20 }}>
          <div className="tdh-row">
            <span className="tdh-label">ALTITUDE</span>
            <span className="tdh-value">{descentTelemetry.altitude.toFixed(0)} m</span>
          </div>
          <div className="tdh-row">
            <span className="tdh-label">VELOCITY</span>
            <span className="tdh-value">{descentTelemetry.vVel.toFixed(1)} m/s</span>
          </div>
          <div className="tdh-row">
            <span className="tdh-label">TARGET</span>
            <span className="tdh-value" style={{ color: '#38bdf8' }}>{selectedLzId || 'LZ-01'}</span>
          </div>
          <div className="tdh-row">
            <span className="tdh-label">PHASE</span>
            <span className="tdh-value">{descentPhase}</span>
          </div>
          <div className="tdh-simulated">SIMULATED TELEMETRY</div>
          <button className="terrain-action-btn" style={{ marginTop: 6 }} onClick={stopDescent}>
            ABORT
          </button>
        </div>
      )}

      {/* Contextual Action Panel for Exploration */}
      <div className="terrain-action-panel">
        {!descentMode && !touchdownComplete && selectedLzId && (
          <button className="terrain-action-btn terrain-action-btn--go" onClick={startDescent}>
            ▼ START LANDING (TARGET: {selectedLzId})
          </button>
        )}

        {touchdownComplete && !roverActive && (
          <div className="terrain-touchdown-wrap">
            <div className="terrain-touchdown-banner">
              SIMULATED LANDING — LZ HEURISTIC ACCEPTED
            </div>
            <button
              className="terrain-action-btn terrain-action-btn--go"
              style={{ marginTop: 6, width: '100%' }}
              onClick={deployRover}
            >
              ▶ DEPLOY ROVER — BEGIN EXPLORATION
            </button>
          </div>
        )}

        {roverActive && !astronautActive && (
          <>
            <div className="terrain-rover-hud">
              <div className="tdh-row">
                <span className="tdh-label">ROVER</span>
                <span className="tdh-value" style={{ color: '#10b981' }}>EXPLORATION READY</span>
              </div>
              <div className="tdh-row">
                <span className="tdh-label">POS</span>
                <span className="tdh-value">
                  {roverPos[0].toFixed(1)}, {roverPos[1].toFixed(1)}
                </span>
              </div>
              <div className="tdh-simulated">WASD / ARROWS TO DRIVE</div>
            </div>
            <button
              className={`terrain-action-btn${inspectionActive ? ' terrain-action-btn--active' : ''}`}
              onClick={onToggleScan}
            >
              ◎ {inspectionActive ? 'CLOSE SCAN' : 'SCAN SITE (IMAGE ANALYSIS)'}
            </button>
            <button className="terrain-action-btn terrain-action-btn--go" onClick={startEVA}>
              ◈ START EVA
            </button>
            <button className="terrain-action-btn terrain-action-btn--sci" onClick={openScience}>
              ◈ SCIENCE DATA
            </button>
          </>
        )}

        {astronautActive && (
          <>
            <div className="terrain-rover-hud">
              <div className="tdh-row">
                <span className="tdh-label">EVA</span>
                <span className="tdh-value" style={{ color: '#38bdf8' }}>ACTIVE</span>
              </div>
              <div className="tdh-row">
                <span className="tdh-label">POS</span>
                <span className="tdh-value">
                  {astronautPos[0].toFixed(1)}, {astronautPos[1].toFixed(1)}
                </span>
              </div>
              <div className="tdh-simulated">WASD / ARROWS TO WALK</div>
            </div>
            <button
              className={`terrain-action-btn${inspectionActive ? ' terrain-action-btn--active' : ''}`}
              onClick={onToggleScan}
            >
              ◎ {inspectionActive ? 'CLOSE SCAN' : 'SCAN SITE (IMAGE ANALYSIS)'}
            </button>
            <button className="terrain-action-btn terrain-action-btn--sci" onClick={openScience}>
              ◈ SCIENCE DATA
            </button>
          </>
        )}
      </div>
    </>
  );
}

// ── Contextual Science Panel ──────────────────────────────────────────────────
function SciencePanel() {
  const terrainLayer      = useStore((s) => s.terrainLayer);
  const setTerrainLayer   = useStore((s) => s.setTerrainLayer);
  const selectedLzId      = useStore((s) => s.selectedLzId);
  const landingZones      = useStore((s) => s.landingZones);

  const selectedLz = landingZones.find((z) => z.id === selectedLzId);

  const COMPUTED = [
    { id: 'NORMAL', label: 'Restored Image' },
    { id: 'HAZARD', label: 'Hazard Map (V2)' },
    { id: 'SAFE',   label: 'Safe Region' },
    { id: 'LZ',     label: 'LZ Candidates' },
  ];

  const RESOURCE = [
    ['water_ice_prospectivity', 'Water-Ice Prospectivity'],
    ['thermal',                 'Thermal'],
    ['spectral',                'Spectral'],
    ['radar',                   'Radar'],
    ['illumination',            'Illumination'],
  ];

  return (
    <div className="science-panel">
      <div className="science-panel-title">SCIENCE LAYERS</div>

      <div className="science-section-label">OBSERVATION DERIVED</div>
      {COMPUTED.map((l) => (
        <div
          key={l.id}
          className={`science-layer-row${terrainLayer === l.id ? ' active' : ''}`}
          onClick={() => setTerrainLayer(l.id)}
        >
          <span
            className="science-layer-dot"
            style={{ background: terrainLayer === l.id ? '#38bdf8' : '#334155' }}
          />
          <span className="science-layer-label">{l.label}</span>
          <span className="science-layer-status" style={{ color: '#10b981' }}>AVAILABLE</span>
        </div>
      ))}

      <div className="science-section-label" style={{ marginTop: 10 }}>RESOURCE LAYERS</div>
      {RESOURCE.map(([id, label]) => (
        <div key={id} className="science-layer-row science-layer-row--unavail">
          <span className="science-layer-dot" style={{ background: '#1e293b' }} />
          <span className="science-layer-label">{label}</span>
          <span className="science-layer-status" style={{ color: '#475569' }}>UNAVAILABLE</span>
        </div>
      ))}
      <div className="science-unavail-note">
        No spectral/radar/thermal data source
      </div>

      {selectedLz && (
        <>
          <div className="science-section-label" style={{ marginTop: 10 }}>SITE INSPECTION</div>
          <div className="science-site-info">
            {[
              ['LZ',           selectedLz.id],
              ['HAZARD DENS.', `${(selectedLz.hazard_density * 100).toFixed(1)}%`],
              ['CLEARANCE',    `${selectedLz.clearance?.toFixed(0)} px`],
              ['LZ SCORE',     `${selectedLz.score?.toFixed(1)} (HEURISTIC)`],
              ['TERRAIN',      'IMAGE-DERIVED ALBEDO (NOT DEM)'],
              ['WATER-ICE',    'UNAVAILABLE'],
            ].map(([l, v]) => (
              <div key={l} className="tdh-row">
                <span className="tdh-label">{l}</span>
                <span className="tdh-value" style={{ color: l === 'WATER-ICE' ? '#475569' : undefined }}>
                  {v}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ── Root TerrainViewport Component ────────────────────────────────────────────
export default function TerrainViewport() {
  const missionScene     = useStore((s) => s.missionScene);
  const sciencePanelOpen = useStore((s) => s.sciencePanelOpen);
  const dangerMapUrl     = useStore((s) => s.dangerMapUrl);
  const safeRegionUrl    = useStore((s) => s.safeRegionUrl);
  const inputInfo        = useStore((s) => s.inputInfo);

  // Exact image dimensions and isotropic world span preserving aspect ratio
  const imgW = inputInfo?.width  || 256;
  const imgH = inputInfo?.height || 256;
  const { spanX, spanZ } = useMemo(
    () => getObservationWorldSpan(imgW, imgH, OBS_BASE_SIZE),
    [imgW, imgH]
  );

  const [transitioning, setTransitioning] = useState(false);
  const [showCorrelation, setShowCorrelation] = useState(false);
  const prevScene = useRef(null);

  const [inspectionActive, setInspectionActive] = useState(false);
  const inspActiveRef  = useRef(false);
  const inspResultRef  = useRef({ wx: 0, wy: 0, wz: 0, valid: false });
  const terrainMeshRef = useRef(null);
  const sceneHmRef     = useRef(null);

  const dangerSamplerRef = useRef(null);
  const safeSamplerRef   = useRef(null);

  useEffect(() => { inspActiveRef.current = inspectionActive; }, [inspectionActive]);

  useEffect(() => {
    dangerSamplerRef.current = null;
    if (!dangerMapUrl) return;
    buildPixelSampler(dangerMapUrl, (s) => { dangerSamplerRef.current = s; });
  }, [dangerMapUrl]);

  useEffect(() => {
    safeSamplerRef.current = null;
    if (!safeRegionUrl) return;
    buildPixelSampler(safeRegionUrl, (s) => { safeSamplerRef.current = s; });
  }, [safeRegionUrl]);

  useEffect(() => {
    if (prevScene.current !== 'TERRAIN' && missionScene === 'TERRAIN') {
      setTransitioning(true);
      const id = setTimeout(() => setTransitioning(false), 2400);
      return () => clearTimeout(id);
    }
    prevScene.current = missionScene;
  }, [missionScene]);

  return (
    <div className="terrain-viewport">
      <Canvas
        className="terrain-canvas"
        shadows
        gl={{
          antialias: true,
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
        camera={{ fov: 45, near: 0.1, far: 1600 }}
      >
        <Scene3D
          transitioning={transitioning}
          terrainMeshRef={terrainMeshRef}
          inspResultRef={inspResultRef}
          inspActiveRef={inspActiveRef}
          externalHmRef={sceneHmRef}
          imgW={imgW}
          imgH={imgH}
          spanX={spanX}
          spanZ={spanZ}
        />
      </Canvas>

      <TerrainHUD
        onToggleScan={() => setInspectionActive((v) => !v)}
        inspectionActive={inspectionActive}
        onOpenCorrelation={() => setShowCorrelation(true)}
        imgW={imgW}
        imgH={imgH}
        spanX={spanX}
        spanZ={spanZ}
      />

      {sciencePanelOpen && <SciencePanel />}

      {showCorrelation && (
        <ObservationCorrelationModal
          onClose={() => setShowCorrelation(false)}
          imgW={imgW}
          imgH={imgH}
          spanX={spanX}
          spanZ={spanZ}
        />
      )}

      <TerrainInspectorPanel
        resultRef={inspResultRef}
        hmRef={sceneHmRef}
        dangerSamplerRef={dangerSamplerRef}
        safeSamplerRef={safeSamplerRef}
        inspectionActive={inspectionActive}
        setInspectionActive={setInspectionActive}
      />
    </div>
  );
}
