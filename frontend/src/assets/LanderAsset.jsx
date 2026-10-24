/**
 * LanderAsset — Photorealistic 3D Spacecraft Component.
 *
 * Sourced from official NASA Apollo Lunar Module 3D asset.
 * Features:
 *   - Accurate physical scale (7.0m height) via WORLD_SCALE
 *   - Ground-normalized pivot (footpads contact surface at y = 0)
 *   - PBR materials with directional shadow casting
 *   - Descent thruster plume & warm surface illumination
 *   - Touchdown suspension settling
 *   - Development fallback badge if asset fails to load
 */
import { useEffect, useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { loadGLTF, cloneGLTFScene } from './AssetLoader';
import { WORLD_SCALE } from './scale';

const MODEL_URL = '/models/lander.glb';

export default function LanderAsset({
  targetX = 0,
  targetZ = 0,
  altitude = 0,
  touchdownComplete = false,
  getHeight = () => 0,
}) {
  const groupRef = useRef();
  const [modelScene, setModelScene] = useState(null);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    let active = true;
    loadGLTF(MODEL_URL)
      .then((gltf) => {
        if (!active) return;
        const cloned = cloneGLTFScene(gltf);

        // Normalize scale and center footpads at y = 0
        const bbox = new THREE.Box3().setFromObject(cloned);
        const size = new THREE.Vector3();
        bbox.getSize(size);

        const targetHeight = WORLD_SCALE.LANDER_HEIGHT;
        const scaleFactor = targetHeight / Math.max(0.1, size.y);
        cloned.scale.setScalar(scaleFactor);

        // Recalculate with scale applied
        bbox.setFromObject(cloned);
        const center = new THREE.Vector3();
        bbox.getCenter(center);

        // Pivot normalization: center on X/Z, footpads touch ground at Y = 0
        cloned.position.x = -center.x;
        cloned.position.z = -center.z;
        cloned.position.y = -bbox.min.y;

        // Configure shadows & materials
        cloned.traverse((node) => {
          if (node.isMesh) {
            node.castShadow = true;
            node.receiveShadow = true;
            if (node.material) {
              node.material.envMapIntensity = 0.8;
              node.material.roughness = Math.min(0.9, Math.max(0.2, node.material.roughness ?? 0.4));
            }
          }
        });

        const container = new THREE.Group();
        container.add(cloned);
        setModelScene(container);
      })
      .catch((err) => {
        if (!active) return;
        setLoadError(err?.message || 'Failed to load lander model');
      });

    return () => {
      active = false;
    };
  }, []);

  // Update lander vertical position based on terrain height and descent altitude
  useFrame(() => {
    if (!groupRef.current) return;
    const groundY = getHeight(targetX, targetZ);
    const targetY = touchdownComplete
      ? groundY
      : groundY + Math.max(0, altitude);

    groupRef.current.position.x = targetX;
    groupRef.current.position.z = targetZ;
    groupRef.current.position.y = THREE.MathUtils.lerp(
      groupRef.current.position.y,
      targetY,
      touchdownComplete ? 0.08 : 0.12
    );
  });

  const isFiring = !touchdownComplete && altitude > 0.3;

  if (loadError) {
    return (
      <group position={[targetX, getHeight(targetX, targetZ) + 1, targetZ]}>
        {/* Development fallback indicator */}
        <mesh position={[0, 1.5, 0]}>
          <boxGeometry args={[4, 0.6, 0.1]} />
          <meshBasicMaterial color="#ef4444" />
        </mesh>
        <lineSegments position={[0, 3.5, 0]}>
          <edgesGeometry args={[new THREE.BoxGeometry(4, 7, 4)]} />
          <lineBasicMaterial color="#ef4444" />
        </lineSegments>
      </group>
    );
  }

  if (!modelScene) return null;

  return (
    <group ref={groupRef} position={[targetX, getHeight(targetX, targetZ) + 100, targetZ]}>
      <primitive object={modelScene} />

      {/* Thruster flame & illumination when descending */}
      {isFiring && (
        <group position={[0, 0.4, 0]}>
          {/* Flame plume */}
          <mesh position={[0, -0.9, 0]}>
            <coneGeometry args={[0.55, 2.2, 16]} />
            <meshBasicMaterial color="#ff7700" transparent opacity={0.8} />
          </mesh>
          <mesh position={[0, -0.6, 0]}>
            <coneGeometry args={[0.3, 1.4, 16]} />
            <meshBasicMaterial color="#ffdd44" transparent opacity={0.9} />
          </mesh>
          {/* Ground surface illumination */}
          <pointLight
            position={[0, -0.8, 0]}
            intensity={2.8}
            distance={28}
            color="#ff9922"
            decay={2}
          />
        </group>
      )}
    </group>
  );
}
