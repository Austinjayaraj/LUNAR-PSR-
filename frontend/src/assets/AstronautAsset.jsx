/**
 * AstronautAsset — Photorealistic 3D Astronaut Component with Lunar Animation.
 *
 * Sourced from official NASA / Google Astronaut 3D EVA suit asset.
 * Features:
 *   - Accurate physical human scale (1.85m height) via WORLD_SCALE
 *   - Ground-normalized pivot (boots rest on surface at y = 0)
 *   - PBR materials with reflective gold visor
 *   - Procedural lunar gravity animation state system:
 *       IDLE: Subtle breathing sway & life-support backpack motion
 *       WALK: 1/6th gravity bounding stride with vertical hop, lateral sway & forward pitch
 *       TURN: Inertial rotation alignment
 *       INSPECT: Ground inspection tilt toward lunar regolith
 *   - Continuous terrain elevation following
 *   - Development fallback badge if asset fails to load
 */
import { useEffect, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { loadGLTF, cloneGLTFScene } from './AssetLoader';
import { WORLD_SCALE } from './scale';

const MODEL_URL = '/models/astronaut.glb';

export default function AstronautAsset({
  astronautPos = [0, 0],
  astronautHeading = 0,
  getHeight = () => 0,
  inspecting = false,
}) {
  const groupRef = useRef();
  const bodyRef = useRef();
  const [modelScene, setModelScene] = useState(null);
  const [loadError, setLoadError] = useState(null);

  // Animation & gait tracking
  const prevPosRef = useRef([...astronautPos]);
  const walkPhaseRef = useRef(0);
  const speedRef = useRef(0);
  const animStateRef = useRef('IDLE'); // 'IDLE' | 'WALK' | 'INSPECT' | 'TURN'
  const currentHeadingRef = useRef(astronautHeading);

  useEffect(() => {
    let active = true;
    loadGLTF(MODEL_URL)
      .then((gltf) => {
        if (!active) return;
        const cloned = cloneGLTFScene(gltf);

        // Normalize scale to WORLD_SCALE.ASTRONAUT_HEIGHT (1.85m)
        const bbox = new THREE.Box3().setFromObject(cloned);
        const size = new THREE.Vector3();
        bbox.getSize(size);

        const targetHeight = WORLD_SCALE.ASTRONAUT_HEIGHT;
        const scaleFactor = targetHeight / Math.max(0.1, size.y);
        cloned.scale.setScalar(scaleFactor);

        // Recalculate with scale applied
        bbox.setFromObject(cloned);
        const center = new THREE.Vector3();
        bbox.getCenter(center);

        // Pivot normalization: center horizontally, boots rest at Y = 0
        cloned.position.x = -center.x;
        cloned.position.z = -center.z;
        cloned.position.y = -bbox.min.y;

        // Configure shadows & PBR materials
        cloned.traverse((node) => {
          if (node.isMesh) {
            node.castShadow = true;
            node.receiveShadow = true;
            if (node.material) {
              node.material.envMapIntensity = 0.9;
              // Gold visor reflection enhancement
              if (node.material.name?.toLowerCase().includes('visor') || node.name?.toLowerCase().includes('visor')) {
                node.material.metalness = 0.95;
                node.material.roughness = 0.05;
                node.material.color = new THREE.Color('#d4af37');
              }
            }
          }
        });

        const container = new THREE.Group();
        container.add(cloned);
        setModelScene(container);
      })
      .catch((err) => {
        if (!active) return;
        setLoadError(err?.message || 'Failed to load astronaut model');
      });

    return () => {
      active = false;
    };
  }, []);

  useFrame((_, dt) => {
    if (!groupRef.current || !bodyRef.current) return;

    const [ax, az] = astronautPos;

    // 1. Calculate displacement and speed
    const dx = ax - prevPosRef.current[0];
    const dz = az - prevPosRef.current[1];
    const dist = Math.sqrt(dx * dx + dz * dz);
    const instantSpeed = dist / Math.max(0.001, dt);
    speedRef.current = THREE.MathUtils.lerp(speedRef.current, instantSpeed, 0.25);

    prevPosRef.current[0] = ax;
    prevPosRef.current[1] = az;

    // 2. Determine animation state
    const isMoving = dist > 0.005;
    const isTurning = Math.abs(astronautHeading - currentHeadingRef.current) > 0.05;

    if (inspecting) {
      animStateRef.current = 'INSPECT';
    } else if (isMoving) {
      animStateRef.current = 'WALK';
    } else if (isTurning) {
      animStateRef.current = 'TURN';
    } else {
      animStateRef.current = 'IDLE';
    }

    // Smooth heading alignment
    currentHeadingRef.current = THREE.MathUtils.lerp(
      currentHeadingRef.current,
      astronautHeading,
      0.18
    );

    // 3. Terrain following: Boots rest on sampled terrain elevation
    const groundY = getHeight(ax, az);

    // 4. Procedural Lunar Movement State Machine
    let hopY = 0;
    let pitch = 0;
    let roll = 0;
    let yawSway = 0;

    const state = animStateRef.current;

    if (state === 'WALK') {
      // Advance walk phase proportional to displacement (lunar stride ~1.4m per bound)
      walkPhaseRef.current += dist * 3.8;

      // Authentic 1/6g lunar vertical bounce
      hopY = Math.abs(Math.sin(walkPhaseRef.current * 1.8)) * 0.14;

      // Lateral weight shift from foot to foot
      roll = Math.sin(walkPhaseRef.current) * 0.055;

      // Forward pitch lean into stride
      pitch = -0.09;

      // Subtle torso yaw twist
      yawSway = Math.sin(walkPhaseRef.current) * 0.04;
    } else if (state === 'INSPECT') {
      // Upper body tilts forward and down to inspect regolith
      pitch = THREE.MathUtils.lerp(bodyRef.current.rotation.x, 0.28, 0.1);
      hopY = -0.04;
    } else if (state === 'TURN') {
      // Subtle heel-pivot bounce while turning
      const turnPhase = performance.now() * 0.006;
      hopY = Math.abs(Math.sin(turnPhase)) * 0.03;
      roll = Math.sin(turnPhase) * 0.02;
    } else {
      // IDLE: Subtle breathing sway & resting stance
      const idleTime = performance.now() * 0.0014;
      hopY = Math.sin(idleTime) * 0.012;
      pitch = Math.sin(idleTime * 0.7) * 0.01;
      roll = Math.cos(idleTime * 0.8) * 0.012;
    }

    // Apply procedural motion to body container
    bodyRef.current.position.y = hopY;
    bodyRef.current.rotation.x = THREE.MathUtils.lerp(bodyRef.current.rotation.x, pitch, 0.15);
    bodyRef.current.rotation.z = THREE.MathUtils.lerp(bodyRef.current.rotation.z, roll, 0.15);
    bodyRef.current.rotation.y = yawSway;

    // Position and orient astronaut root
    groupRef.current.position.set(ax, groundY, az);
    groupRef.current.rotation.y = currentHeadingRef.current;
  });

  if (loadError) {
    return (
      <group position={[astronautPos[0], getHeight(astronautPos[0], astronautPos[1]) + 0.4, astronautPos[1]]}>
        <mesh position={[0, 0.9, 0]}>
          <boxGeometry args={[1.6, 0.35, 0.1]} />
          <meshBasicMaterial color="#ef4444" />
        </mesh>
        <lineSegments position={[0, 0.9, 0]}>
          <edgesGeometry args={[new THREE.BoxGeometry(0.8, 1.85, 0.8)]} />
          <lineBasicMaterial color="#ef4444" />
        </lineSegments>
      </group>
    );
  }

  if (!modelScene) return null;

  return (
    <group ref={groupRef} position={[astronautPos[0], getHeight(astronautPos[0], astronautPos[1]), astronautPos[1]]}>
      <group ref={bodyRef}>
        <primitive object={modelScene} />
      </group>
    </group>
  );
}
