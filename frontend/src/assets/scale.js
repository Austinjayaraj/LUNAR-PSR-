/**
 * WORLD_SCALE — Physical scale definitions for Lunar PSR Simulation.
 *
 * Metric standard: 1 Three.js world unit = 1.0 meter.
 *
 * Scale Hierarchy:
 *   ASTRONAUT (1.85m) < ROVER (3.0m) < LANDER (7.0m) < LOCAL SITE (100.0m)
 */
export const WORLD_SCALE = {
  // Local observation terrain patch: 100m x 100m
  TERRAIN_SIZE: 100.0,

  // Lunar Lander (Apollo Lunar Module spacecraft)
  // Height ~7.0m, diagonal footpad span ~9.4m, core stage ~4.2m
  LANDER_HEIGHT: 7.0,
  LANDER_DIAMETER: 9.4,

  // Rover (Planetary Surface Rover)
  // Length ~3.0m, width ~2.7m, height ~2.2m, wheel diameter ~0.52m
  ROVER_LENGTH: 3.0,
  ROVER_WIDTH: 2.7,
  ROVER_HEIGHT: 2.2,
  ROVER_WHEEL_RADIUS: 0.26,

  // Astronaut (Apollo / Artemis EVA Extravehicular Mobility Unit suit)
  // Height ~1.85m with helmet, PLSS life support backpack, and boots
  ASTRONAUT_HEIGHT: 1.85,
  ASTRONAUT_WIDTH: 0.85,

  // Rock sizes: 0.25m to 0.9m boulders (relative to 1.85m human scale)
  ROCK_MIN_SCALE: 0.25,
  ROCK_MAX_SCALE: 0.85,
};

export default WORLD_SCALE;
