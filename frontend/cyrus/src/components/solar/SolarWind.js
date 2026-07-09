import * as THREE from 'three';

/**

 * A recycling particle stream flowing outward from the sun, standing in for
 * measured solar wind speed/density. Particles are respawned near the sun's
 * surface and drift outward along a slightly turbulent radial path, fading
 * as they go. Speed and density are both telemetry-driven.
 *
 * createSolarWind(options) -> {
 *   group: THREE.Group,
 *   update(dt, telemetry),   // telemetry: { windSpeed (km/s), density (0-1) }
 *   dispose(),
 * }
 */
export function createSolarWind(options = {}) {
  const {
    count = 900,
    innerRadius = 3.2,
    outerRadius = 22,
    color = '#8ecbff',
  } = options;

  const positions = new Float32Array(count * 3);
  const velocities = new Float32Array(count); // radial speed scalar per particle
  const ages = new Float32Array(count); // 0..1 normalized life
  const dirs = new Float32Array(count * 3); // fixed outward direction per particle

  function respawn(i) {
    const dir = new THREE.Vector3(
      Math.random() * 2 - 1,
      Math.random() * 2 - 1,
      Math.random() * 2 - 1
    ).normalize();
    dirs[i * 3] = dir.x;
    dirs[i * 3 + 1] = dir.y;
    dirs[i * 3 + 2] = dir.z;

    const r = innerRadius * (0.95 + Math.random() * 0.1);
    positions[i * 3] = dir.x * r;
    positions[i * 3 + 1] = dir.y * r;
    positions[i * 3 + 2] = dir.z * r;

    velocities[i] = 0.85 + Math.random() * 0.3;
    ages[i] = Math.random() * 0.05; // stagger start
  }

  for (let i = 0; i < count; i++) respawn(i);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: new THREE.Color(color),
    size: 0.06,
    transparent: true,
    opacity: 0.75,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true,
  });

  const points = new THREE.Points(geometry, material);
  const group = new THREE.Group();
  group.add(points);

  let speedScale = 1;
  let densityScale = 1;

  function update(dt, telemetry = {}) {
    if (typeof telemetry.windSpeed === 'number') {
      // typical solar wind ~300-800 km/s -> map to a gentle animation multiplier
      speedScale = THREE.MathUtils.clamp(telemetry.windSpeed / 400, 0.5, 2.2);
    }
    if (typeof telemetry.density === 'number') {
      densityScale = THREE.MathUtils.clamp(telemetry.density, 0.15, 1);
    }
    material.opacity = 0.35 + densityScale * 0.5;

    const span = outerRadius - innerRadius;
    for (let i = 0; i < count; i++) {
      ages[i] += dt * 0.12 * velocities[i] * speedScale;
      if (ages[i] >= 1) {
        respawn(i);
        continue;
      }
      const r = innerRadius + span * ages[i];
      positions[i * 3] = dirs[i * 3] * r;
      positions[i * 3 + 1] = dirs[i * 3 + 1] * r;
      positions[i * 3 + 2] = dirs[i * 3 + 2] * r;
    }
    geometry.attributes.position.needsUpdate = true;
  }

  function dispose() {
    geometry.dispose();
    material.dispose();
  }

  return { group, update, dispose };
}
