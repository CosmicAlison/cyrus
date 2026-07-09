import * as THREE from 'three';

/**
 * An expanding, fading shell of light representing a coronal mass ejection.
 * Fire-and-forget: call trigger() with a source lat/lon (matching an active
 * region on the Sun) and it plays out a single burst automatically.
 *
 * createCMEBurst(options) -> {
 *   group: THREE.Group,
 *   update(dt),
 *   trigger(lat, lon, opts?),  // opts: { intensity 0-1, speed, color }
 *   dispose(),
 * }
 */
function latLonToVec3(lat, lon, r = 1) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -Math.sin(phi) * Math.cos(theta) * r,
    Math.cos(phi) * r,
    Math.sin(phi) * Math.sin(theta) * r
  );
}

export function createCMEBurst(options = {}) {
  const { sunRadius = 3, maxActive = 4 } = options;
  const group = new THREE.Group();
  const active = [];

  function trigger(lat, lon, opts = {}) {
    const { intensity = 0.7, speed = 3.2, color = '#ffcf8a' } = opts;
    if (active.length >= maxActive) {
      const oldest = active.shift();
      group.remove(oldest.mesh);
      oldest.mesh.geometry.dispose();
      oldest.mesh.material.dispose();
    }

    const dir = latLonToVec3(lat, lon, 1);
    const geo = new THREE.ConeGeometry(sunRadius * 0.55, sunRadius * 1.6, 24, 1, true);
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity: intensity * 0.55,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);

    // orient cone to point outward along dir, base near sun surface
    const up = new THREE.Vector3(0, 1, 0);
    mesh.quaternion.setFromUnitVectors(up, dir.clone().normalize());
    mesh.position.copy(dir.clone().multiplyScalar(sunRadius * 0.9));

    group.add(mesh);
    active.push({ mesh, mat, dir, age: 0, life: 2.4, speed, baseScale: 1 });
  }

  function update(dt) {
    for (let i = active.length - 1; i >= 0; i--) {
      const b = active[i];
      b.age += dt;
      const t = b.age / b.life;
      if (t >= 1) {
        group.remove(b.mesh);
        b.mesh.geometry.dispose();
        b.mesh.material.dispose();
        active.splice(i, 1);
        continue;
      }
      const scale = 1 + t * b.speed;
      b.mesh.scale.set(scale, 1 + t * b.speed * 1.4, scale);
      b.mesh.position.copy(b.dir.clone().multiplyScalar(sunRadius * (0.9 + t * b.speed * 0.8)));
      b.mat.opacity = (1 - t) * 0.55;
    }
  }

  function dispose() {
    active.forEach((b) => {
      b.mesh.geometry.dispose();
      b.mesh.material.dispose();
    });
    active.length = 0;
  }

  return { group, update, trigger, dispose };
}
