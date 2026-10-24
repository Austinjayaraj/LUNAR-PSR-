/**
 * RoverAsset — Photorealistic 3D Planetary Rover Component.
 *
 * Sourced from official NASA Mars 2020 / Planetary Rover 3D asset.
 * Features:
 *   - Accurate physical scale (3.0m length) via WORLD_SCALE
 *   - Ground-normalized pivot (wheels contact surface at y = 0)
 *   - Six wheels with physical rotation during driving
 *   - Suspension response & terrain slope following (pitch & roll)
 *   - Animated mast camera head with autonomous scanning
 *   - Deployment sequence animation (rolls out from lander & activates instruments)
 *   - Development fallback badge if asset fails to load
 */
import { useEffect, useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { loadGLTF, cloneGLTFScene } from './AssetLoader';
import { WORLD_SCALE } from './scale';

const MODEL_URL = '/models/rover.glb';

export default function RoverAsset({
  roverPos = [5, 5],
  roverHeading = 0,
  getHeight = () => 0,
  isDeploying = false,
  onDeploymentComplete,
}) {
  const groupRef = useRef();
  const chassisRef = useRef();
  const [modelScene, setModelScene] = useState(null);
  const [loadError, setLoadError] = useState(null);

  // References to dynamic model subnodes
  const wheelsNodeRef = useRef(null);
  const headNodeRef = useRef(null);
  const suspensionNodeRef = useRef(null);

  // Movement & physics tracking
  const prevPosRef = useRef([...roverPos]);
  const wheelAngleRef = useRef(0);
  const speedRef = useRef(0);
  const currentPitchRef = useRef(0);
  const currentRollRef = useRef(0);

  // Deployment animation state (0 -> 1)
  const deployProgressRef = useRef(isDeploying ? 0 : 1);
  const deployDoneRef = useRef(!isDeploying);

  useEffect(() => {
    let active = true;
    loadGLTF(MODEL_URL)
      .then((gltf) => {
        if (!active) return;
        const cloned = cloneGLTFScene(gltf);

        // Normalize scale to WORLD_SCALE.ROVER_LENGTH (3.0m)
        const bbox = new THREE.Box3().setFromObject(cloned);
        const size = new THREE.Vector3();
        bbox.getSize(size);

        // Rovers are typically longer in Z or X
        const maxHoriz = Math.max(size.x, size.z);
        const scaleFactor = WORLD_SCALE.ROVER_LENGTH / Math.max(0.1, maxHoriz);
        cloned.scale.setScalar(scaleFactor);

        // Recalculate with scale applied
        bbox.setFromObject(cloned);
        const center = new THREE.Vector3();
        bbox.getCenter(center);

        // Pivot normalization: center horizontally, bottom of wheels rests at Y = 0
        cloned.position.x = -center.x;
        cloned.position.z = -center.z;
        cloned.position.y = -bbox.min.y;

        // Find key subnodes for animation
        cloned.traverse((node) => {
          if (node.isMesh) {
            node.castShadow = true;
            node.receiveShadow = true;
            if (node.material) {
              node.material.roughness = Math.min(0.9, Math.max(0.3, node.material.roughness ?? 0.5));
            }
          }
          if (node.name === 'Wheels_objs') {
            wheelsNodeRef.current = node;
          }
          if (node.name === 'head') {
            headNodeRef.current = node;
          }
          if (node.name === 'suspension') {
            suspensionNodeRef.current = node;
          }
        });

        const container = new THREE.Group();
        container.add(cloned);
        setModelScene(container);
      })
      .catch((err) => {
        if (!active) return;
        setLoadError(err?.message || 'Failed to load rover model');
      });

    return () => {
      active = false;
    };
  }, []);

  useFrame((_, dt) => {
    if (!groupRef.current) return;

    const [rx, rz] = roverPos;

    // Handle deployment sequence (rolls out from lander)
    if (isDeploying && !deployDoneRef.current) {
      deployProgressRef.current = Math.min(1, deployProgressRef.current + dt * 0.45);
      const p = deployProgressRef.current;
      const ease = 1 - Math.pow(1 - p, 2);

      // Deploying: rover emerges and settles onto regolith
      const groundY = getHeight(rx, rz);
      const deployY = groundY + Math.sin((1 - ease) * Math.PI * 0.5) * 1.5;
      groupRef.current.position.set(rx, deployY, rz);
      groupRef.current.rotation.y = roverHeading;

      // Mast camera tilts up into operational pose
      if (headNodeRef.current) {
        headNodeRef.current.rotation.x = THREE.MathUtils.lerp(-0.6, 0, ease);
      }

      if (p >= 1) {
        deployDoneRef.current = true;
        onDeploymentComplete?.();
      }
      return;
    }

    // 1. Calculate horizontal displacement & speed
    const dx = rx - prevPosRef.current[0];
    const dz = rz - prevPosRef.current[1];
    const dist = Math.sqrt(dx * dx + dz * dz);
    const instantSpeed = dist / Math.max(0.001, dt);
    speedRef.current = THREE.MathUtils.lerp(speedRef.current, instantSpeed, 0.2);

    prevPosRef.current[0] = rx;
    prevPosRef.current[1] = rz;

    // 2. Wheel rotation: rotate wheel nodes proportional to distance traveled
    if (dist > 0.0001) {
      // Determine forward vs backward dot product against heading
      const forwardX = -Math.sin(roverHeading);
      const forwardZ = -Math.cos(roverHeading);
      const dot = dx * forwardX + dz * forwardZ;
      const direction = dot >= 0 ? 1 : -1;

      const deltaAngle = (dist / WORLD_SCALE.ROVER_WHEEL_RADIUS) * direction;
      wheelAngleRef.current += deltaAngle;

      if (wheelsNodeRef.current) {
        // Rotate wheels node around pitch axis
        wheelsNodeRef.current.rotation.x = wheelAngleRef.current;
      }
    }

    // 3. Terrain following: Sample terrain height at center and four perimeter points
    const centerY = getHeight(rx, rz);

    // Front / Back slope for pitch
    const frontX = rx - Math.sin(roverHeading) * 1.2;
    const frontZ = rz - Math.cos(roverHeading) * 1.2;
    const backX  = rx + Math.sin(roverHeading) * 1.2;
    const backZ  = rz + Math.cos(roverHeading) * 1.2;
    const frontY = getHeight(frontX, frontZ);
    const backY  = getHeight(backX, backZ);
    const targetPitch = Math.atan2(backY - frontY, 2.4);

    // Left / Right slope for roll
    const leftX  = rx - Math.cos(roverHeading) * 0.9;
    const leftZ  = rz + Math.sin(roverHeading) * 0.9;
    const rightX = rx + Math.cos(roverHeading) * 0.9;
    const rightZ = rz - Math.sin(roverHeading) * 0.9;
    const leftY  = getHeight(leftX, leftZ);
    const rightY = getHeight(rightX, rightZ);
    const targetRoll = Math.atan2(leftY - rightY, 1.8);

    // Smooth chassis attitude response with suspension dampening
    currentPitchRef.current = THREE.MathUtils.lerp(currentPitchRef.current, targetPitch, 0.12);
    currentRollRef.current  = THREE.MathUtils.lerp(currentRollRef.current, targetRoll, 0.12);

    // Subtle acceleration squat / braking dive
    const accelDive = (speedRef.current > 0.5 ? -0.02 : 0) * (dist > 0.01 ? 1 : 0);

    // Position rover root
    groupRef.current.position.set(rx, centerY, rz);
    groupRef.current.rotation.y = roverHeading;

    if (chassisRef.current) {
      chassisRef.current.rotation.x = currentPitchRef.current + accelDive;
      chassisRef.current.rotation.z = currentRollRef.current;
    }

    // 4. Autonomous mast camera scan when stationary
    if (headNodeRef.current && speedRef.current < 0.2) {
      const time = performance.now() * 0.0008;
      headNodeRef.current.rotation.y = Math.sin(time) * 0.35;
      headNodeRef.current.rotation.x = Math.sin(time * 0.7) * 0.08 - 0.05;
    }
  });

  if (loadError) {
    return (
      <group position={[roverPos[0], getHeight(roverPos[0], roverPos[1]) + 0.5, roverPos[1]]}>
        <mesh position={[0, 0.8, 0]}>
          <boxGeometry args={[3, 0.5, 0.1]} />
          <meshBasicMaterial color="#ef4444" />
        </mesh>
        <lineSegments position={[0, 1.2, 0]}>
          <edgesGeometry args={[new THREE.BoxGeometry(3, 2.2, 2.5)]} />
          <lineBasicMaterial color="#ef4444" />
        </lineSegments>
      </group>
    );
  }

  if (!modelScene) return null;

  return (
    <group ref={groupRef} position={[roverPos[0], getHeight(roverPos[0], roverPos[1]), roverPos[1]]}>
      <group ref={chassisRef}>
        <primitive object={modelScene} />
      </group>
    </group>
  );
}
