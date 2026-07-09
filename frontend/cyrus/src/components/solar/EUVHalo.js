import * as THREE from 'three';

/**
 * Concentric glowing rings around the sun, each mapped to a real SDO/AIA-style
 * EUV wavelength band and its conventional false-color. Toggle a single
 * channel or show all at reduced opacity as an ambient "spectral" halo.
 *
 * Channels follow common EUV imaging conventions:
 *   171Å  -> teal/gold   (quiet corona / coronal loops)
 *   193Å  -> yellow-green (corona + hot flare plasma)
 *   211Å  -> violet       (active regions)
 *   304Å  -> orange-red   (chromosphere / prominences)
 *
 * createEUVHalo(options) -> {
 *   group: THREE.Group,
 *   update(dt),
 *   setChannel(channel), // '171' | '193' | '211' | '304' | 'all' | null
 *   dispose(),
 * }
 */
const CHANNELS = {
  171: { color: '#3ddbd9', radius: 3.4, tilt: 0.15 },
  193: { color: '#d9d43d', radius: 3.7, tilt: -0.22 },
  211: { color: '#a15bff', radius: 4.0, tilt: 0.34 },
  304: { color: '#ff5a36', radius: 4.35, tilt: -0.4 },
};

export function createEUVHalo(options = {}) {
  const { thickness = 0.045 } = options;
  const group = new THREE.Group();
  const rings = {};

  Object.entries(CHANNELS).forEach(([key, cfg]) => {
    const geo = new THREE.TorusGeometry(cfg.radius, thickness, 16, 128);
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(cfg.color),
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = Math.PI / 2 + cfg.tilt;
    mesh.userData.baseOpacity = 0.55;
    group.add(mesh);
    rings[key] = { mesh, mat, spin: (Math.random() - 0.5) * 0.05 };
  });

  let activeChannel = null;
  let fadeTargets = {};

  function setChannel(channel) {
    activeChannel = channel;
    Object.keys(rings).forEach((key) => {
      if (channel === 'all') fadeTargets[key] = 0.22;
      else if (channel === key || channel === Number(key)) fadeTargets[key] = 0.6;
      else fadeTargets[key] = 0;
    });
  }

  // default: show all faintly
  setChannel('all');

  function update(dt) {
    Object.entries(rings).forEach(([key, r]) => {
      r.mesh.rotation.z += r.spin * dt;
      const target = fadeTargets[key] ?? 0;
      r.mat.opacity += (target - r.mat.opacity) * Math.min(dt * 3, 1);
    });
  }

  function dispose() {
    Object.values(rings).forEach((r) => {
      r.mesh.geometry.dispose();
      r.mat.dispose();
    });
  }

  return { group, update, setChannel, dispose };
}
