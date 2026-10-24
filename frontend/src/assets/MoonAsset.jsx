/**
 * MoonAsset — Photorealistic Global 3D Moon Globe.
 *
 * Implements a high-fidelity photographic & scientific visualization:
 *   - Perfectly spherical geometry (256x128 segments) with zero spiky displacement distortion
 *   - Photometrically calibrated NASA LROC albedo map with authentic maria / highland tones
 *   - Smooth tangent-space normal map derived from LOLA laser altimeter data (subtle 0.12 scale)
 *   - Restrained lunar regolith PBR material (roughness 0.94, metalness 0.0, desaturated gray)
 *   - Subtle South Pole PSR mission indicator
 *   - Raycasting interaction for seamless transition to mission setup
 */
import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { loadTextureCached } from './AssetLoader';

export default function MoonAsset({ onMoonClick, isTransitioning = false }) {
  const meshRef = useRef();

  const { albedoTex, normalTex } = useMemo(() => {
    // Calibrated LROC albedo (Maria #2A2A2A - Highlands #8A8A8A - Rays #BCBCBC)
    const albedo = loadTextureCached('/textures/moon_albedo_2k.jpg', true);
    if (albedo) {
      albedo.wrapS = THREE.RepeatWrapping;
      albedo.wrapT = THREE.ClampToEdgeWrapping;
      albedo.generateMipmaps = true;
      albedo.minFilter = THREE.LinearMipmapLinearFilter;
      albedo.magFilter = THREE.LinearFilter;
    }

    // Smooth LOLA laser altimeter normal map
    const normal = loadTextureCached('/textures/moon_normal_2k.jpg', false);
    if (normal) {
      normal.wrapS = THREE.RepeatWrapping;
      normal.wrapT = THREE.ClampToEdgeWrapping;
      normal.generateMipmaps = true;
      normal.minFilter = THREE.LinearMipmapLinearFilter;
      normal.magFilter = THREE.LinearFilter;
    }

    return { albedoTex: albedo, normalTex: normal };
  }, []);

  useFrame((_, dt) => {
    if (!meshRef.current) return;
    // Extremely slow, majestic lunar axial rotation (~0.18 deg/sec)
    meshRef.current.rotation.y += 0.0032 * dt;
  });

  const handlePointerDown = (e) => {
    e.stopPropagation();
    if (onMoonClick && !isTransitioning) {
      onMoonClick(e.point);
    }
  };

  return (
    <group>
      {/* Pristine spherical Moon geometry: zero vertex displacement for a true planetary silhouette */}
      <mesh
        ref={meshRef}
        position={[0, 0, 0]}
        onPointerDown={handlePointerDown}
      >
        <sphereGeometry args={[14, 256, 128]} />
        <meshStandardMaterial
          map={albedoTex}
          color="#b0ada6"
          normalMap={normalTex}
          normalScale={new THREE.Vector2(0.12, 0.12)}
          roughness={0.94}
          metalness={0.0}
        />
      </mesh>

      {/* Subtle South Pole PSR indicator target */}
      <mesh position={[0, -13.98, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.45, 0.58, 32]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.25} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}
