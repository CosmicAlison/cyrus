import * as THREE from 'three';

/**
 * A large orbiting ring (gauge) around the whole scene showing the current
 * Kp-index (geomagnetic activity, 0-9) as a glowing arc sweeping from calm
 * green through to storm red — an "aurora likelihood" readout.
 *
 * createKpDial(options) -> {
 *   group: THREE.Group,
 *   update(dt, telemetry),  // telemetry: { kpIndex: 0-9 }
 *   dispose(),
 * }
 */
const STOPS = [
  { kp: 0, color: new THREE.Color('#3ddc7a') },
  { kp: 4, color: new THREE.Color('#e8d84a') },
  { kp: 6, color: new THREE.Color('#ff9a3d') },
  { kp: 9, color: new THREE.Color('#ff3d3d') },
];

function colorForKp(kp) {
  for (let i = 0; i < STOPS.length - 1; i++) {
    const a = STOPS[i];
    const b = STOPS[i + 1];
    if (kp >= a.kp && kp <= b.kp) {
      const t = (kp - a.kp) / (b.kp - a.kp || 1);
      return a.color.clone().lerp(b.color, t);
    }
  }
  return STOPS[STOPS.length - 1].color.clone();
}

export function createKpDial(options = {}) {
  const { radius = 9, tubeThickness = 0.035, segments = 256 } = options;
  const group = new THREE.Group();

  // faint full track
  const trackGeo = new THREE.TorusGeometry(radius, tubeThickness * 0.6, 8, segments);
  const trackMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color('#3a4a66'),
    transparent: true,
    opacity: 0.25,
  });
  const track = new THREE.Mesh(trackGeo, trackMat);
  track.rotation.x = Math.PI / 2;
  group.add(track);

  // active arc: built from a partial torus, angle driven by Kp value
  let arcMesh = null;
  let currentKp = 0;
  let displayedKp = 0;

  function buildArc(kp) {
    if (arcMesh) {
      group.remove(arcMesh);
      arcMesh.geometry.dispose();
      arcMesh.material.dispose();
    }
    const frac = THREE.MathUtils.clamp(kp / 9, 0.02, 1);
    const arcLength = Math.PI * 2 * frac;
    const geo = new THREE.TorusGeometry(radius, tubeThickness, 10, Math.max(8, Math.floor(segments * frac)), arcLength);
    const mat = new THREE.MeshBasicMaterial({
      color: colorForKp(kp),
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    arcMesh = new THREE.Mesh(geo, mat);
    arcMesh.rotation.x = Math.PI / 2;
    arcMesh.rotation.z = -Math.PI / 2; // start at top
    group.add(arcMesh);
  }

  buildArc(0);

  function update(dt, telemetry = {}) {
    if (typeof telemetry.kpIndex === 'number') {
      currentKp = THREE.MathUtils.clamp(telemetry.kpIndex, 0, 9);
    }
    // smooth toward target, rebuild geometry only when it visibly changes
    displayedKp += (currentKp - displayedKp) * Math.min(dt * 1.5, 1);
    if (Math.abs(displayedKp - (buildArc.lastKp ?? -1)) > 0.05) {
      buildArc(displayedKp);
      buildArc.lastKp = displayedKp;
    }
    group.rotation.y += dt * 0.03;
  }

  function dispose() {
    trackGeo.dispose();
    trackMat.dispose();
    if (arcMesh) {
      arcMesh.geometry.dispose();
      arcMesh.material.dispose();
    }
  }

  return { group, update, dispose };
}
