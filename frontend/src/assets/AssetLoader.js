/**
 * AssetLoader — Reusable GLTF & Texture asset caching and management system.
 *
 * Features:
 *   - Local DRACO decompression support (/draco/)
 *   - Model caching by URL to prevent redundant fetches and re-parses
 *   - Deep cloning via SkeletonUtils for independent instance state
 *   - Proper resource disposal
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js';

// Cache for parsed GLTF assets: url -> Promise<GLTF>
const gltfCache = new Map();
// Cache for loaded textures: url -> THREE.Texture
const textureCache = new Map();

let dracoLoaderInstance = null;

function getDracoLoader() {
  if (!dracoLoaderInstance) {
    dracoLoaderInstance = new DRACOLoader();
    dracoLoaderInstance.setDecoderPath('/draco/');
    dracoLoaderInstance.setDecoderConfig({ type: 'wasm' });
  }
  return dracoLoaderInstance;
}

/**
 * Load a GLTF / GLB model with DRACO support and in-memory caching.
 * @param {string} url - URL to the .glb / .gltf file
 * @returns {Promise<any>} Parsed GLTF object
 */
export function loadGLTF(url) {
  if (gltfCache.has(url)) {
    return gltfCache.get(url);
  }

  const promise = new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.setDRACOLoader(getDracoLoader());

    loader.load(
      url,
      (gltf) => {
        resolve(gltf);
      },
      undefined,
      (err) => {
        console.error(`[AssetLoader] Failed to load GLTF from "${url}":`, err);
        gltfCache.delete(url); // don't cache errors
        reject(err);
      }
    );
  });

  gltfCache.set(url, promise);
  return promise;
}

/**
 * Deep-clone a loaded GLTF scene so instances have isolated nodes/materials/bones.
 * @param {any} gltf - Loaded GLTF object
 * @returns {THREE.Group} Cloned root object
 */
export function cloneGLTFScene(gltf) {
  if (!gltf || !gltf.scene) return null;
  const clone = SkeletonUtils.clone(gltf.scene);
  return clone;
}

/**
 * Load and cache a Texture.
 * @param {string} url
 * @param {boolean} isSRGB
 * @returns {THREE.Texture}
 */
export function loadTextureCached(url, isSRGB = true) {
  if (!url) return null;
  if (textureCache.has(url)) {
    return textureCache.get(url);
  }

  const loader = new THREE.TextureLoader();
  const tex = loader.load(url);
  if (isSRGB) {
    tex.colorSpace = THREE.SRGBColorSpace;
  } else {
    tex.colorSpace = THREE.NoColorSpace;
  }
  textureCache.set(url, tex);
  return tex;
}

/**
 * Dispose cached textures and models.
 */
export function clearAssetCache() {
  textureCache.forEach((tex) => tex.dispose());
  textureCache.clear();
  gltfCache.clear();
  if (dracoLoaderInstance) {
    dracoLoaderInstance.dispose();
    dracoLoaderInstance = null;
  }
}
