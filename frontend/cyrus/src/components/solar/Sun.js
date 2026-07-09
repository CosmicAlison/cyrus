import * as THREE from 'three';

/**
 * Sun.js
 * Stylized, telemetry-driven sun. Not photoreal — a warm plasma sphere whose
 * surface turbulence, active-region hotspots, and flare flashes are all
 * driven by shader uniforms so updates never touch geometry.
 *
 * createSun(options) -> {
 *   group: THREE.Group,              // add this to your scene
 *   update(dt, telemetry),           // call every frame
 *   setActiveRegions(regions),       // regions: [{ lat, lon, intensity (0-1) }, ...] max 8
 *   triggerFlare(lat, lon, classStrength, durationSec), // classStrength 0-1
 *   dispose(),
 * }
 */

const MAX_AR = 8;

// Compact 3D simplex noise (Ashima Arts / Ian McEwan, public-domain-style GLSL utility
// used ubiquitously in shader work) — kept local so this file has zero extra deps.
const NOISE_GLSL = `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0);
  const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy));
  vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz);
  vec3 l=1.0-g;
  vec3 i1=min(g.xyz,l.zxy);
  vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx;
  vec3 x2=x0-i2+C.yyy;
  vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(
      i.z+vec4(0.0,i1.z,i2.z,1.0))
    +i.y+vec4(0.0,i1.y,i2.y,1.0))
    +i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857;
  vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z);
  vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy;
  vec4 y=y_*ns.x+ns.yyyy;
  vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy);
  vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0;
  vec4 s1=floor(b1)*2.0+1.0;
  vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
  vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x);
  vec3 p1=vec3(a0.zw,h.y);
  vec3 p2=vec3(a1.xy,h.z);
  vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);
  m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
float fbm(vec3 p){
  float v=0.0; float amp=0.55;
  for(int i=0;i<4;i++){ v+=amp*snoise(p); p*=2.02; amp*=0.5; }
  return v;
}
`;

const SUN_VERTEX = `
varying vec3 vNormal;
varying vec3 vWorldPos;
void main(){
  vNormal = normalize(normalMatrix * normal);
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const SUN_FRAGMENT = `
${NOISE_GLSL}
varying vec3 vNormal;
varying vec3 vWorldPos;

uniform float uTime;
uniform vec3 uCoreColor;
uniform vec3 uEdgeColor;
uniform vec3 uHotColor;
uniform vec3 uCameraPos;

uniform vec3 uARPos[${MAX_AR}];
uniform float uARIntensity[${MAX_AR}];

uniform vec3 uFlarePos;
uniform float uFlareIntensity;
uniform vec3 uFlareColor;

void main(){
  vec3 n = normalize(vNormal);

  vec3 p = n;
  float angle = uTime * 0.08;
  mat2 r = mat2(
      cos(angle), -sin(angle),
      sin(angle),  cos(angle)
  );

  p.xz = r * p.xz;

  float turbulence = fbm(p * 2.4);

  turbulence = turbulence * 0.5 + 0.5;

  vec3 color = mix(uCoreColor, uEdgeColor, turbulence);

  // fine granulation shimmer
  float grain = fbm(n * 9.0 - vec3(uTime * 0.1));
  color += uEdgeColor * (grain * 0.06);

  // limb brightening toward edge (fresnel), reversed vs realism on purpose: soft glow rim
  vec3 viewDir = normalize(uCameraPos - vWorldPos);
  float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 2.2);
  color += uEdgeColor * fresnel * 0.55;

  // active region hotspots
  float arGlow = 0.0;
  for (int i = 0; i < ${MAX_AR}; i++) {
    float d = distance(n, uARPos[i]);
    float glow = smoothstep(0.38, 0.0, d) * uARIntensity[i];
    float pulse =
    0.8 +
    0.2 *
    sin(
        uTime * 5.0 +
        float(i) * 2.0
    );

    arGlow += glow * pulse;
  }
  arGlow = clamp(arGlow, 0.0, 1.0);
  color = mix(color, uHotColor, arGlow * 0.85);

  // flare flash
  if (uFlareIntensity > 0.001) {
    float fd = distance(n, uFlarePos);
    float flareGlow = smoothstep(0.55, 0.0, fd) * uFlareIntensity;
    float flicker = 0.82 + 0.4 * sin(uTime * 46.0) + 0.1 * sin(uTime * 113.0);
    color += uFlareColor * flareGlow * flicker;
  }

  gl_FragColor = vec4(color, 1.0);
}
`;

const CORONA_VERTEX = `
varying vec3 vNormal;
varying vec3 vWorldPos;
void main(){
  vNormal = normalize(normalMatrix * normal);
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const CORONA_FRAGMENT = `
${NOISE_GLSL}
varying vec3 vNormal;
varying vec3 vWorldPos;
uniform float uTime;
uniform vec3 uColor;
uniform vec3 uCameraPos;
uniform float uOpacity;
uniform float uPower;

void main(){
  vec3 n = normalize(vNormal);
  vec3 viewDir = normalize(uCameraPos - vWorldPos);
  float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), uPower);
  float breathe = 0.85 + 0.15 * sin(uTime * 0.6 + fbm(n * 3.0 + uTime * 0.08) * 2.0);
  float alpha = fresnel * uOpacity * breathe;
  gl_FragColor = vec4(uColor, alpha);
}
`;

function latLonToVec3(lat, lon) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -Math.sin(phi) * Math.cos(theta),
    Math.cos(phi),
    Math.sin(phi) * Math.sin(theta)
  );
}

export function createSun(options = {}) {
  const {
    radius = 3,
    coreColor = '#ff7a1a',
    edgeColor = '#ffd27a',
    hotColor = '#fff3d6',
    flareColor = '#fff7e6',
    coronaColor = '#ffb066',
  } = options;

  const group = new THREE.Group();

  // --- core sun ---
  const sunGeo = new THREE.SphereGeometry(radius, 96, 96);
  const arPos = new Array(MAX_AR).fill(0).map(() => new THREE.Vector3(0, -999, 0));
  const arIntensity = new Array(MAX_AR).fill(0);

  const sunUniforms = {
    uTime: { value: 0 },
    uCoreColor: { value: new THREE.Color(coreColor) },
    uEdgeColor: { value: new THREE.Color(edgeColor) },
    uHotColor: { value: new THREE.Color(hotColor) },
    uCameraPos: { value: new THREE.Vector3() },
    uARPos: { value: arPos },
    uARIntensity: { value: arIntensity },
    uFlarePos: { value: new THREE.Vector3(0, -999, 0) },
    uFlareIntensity: { value: 0 },
    uFlareColor: { value: new THREE.Color(flareColor) },
  };

  const sunMat = new THREE.ShaderMaterial({
    vertexShader: SUN_VERTEX,
    fragmentShader: SUN_FRAGMENT,
    uniforms: sunUniforms,
  });

  const sunMesh = new THREE.Mesh(sunGeo, sunMat);
  group.add(sunMesh);

  // --- corona shells (layered, additive) ---
  const coronaLayers = [
    { scale: 1.12, opacity: 0.5, power: 2.0 },
    { scale: 1.28, opacity: 0.28, power: 2.8 },
    { scale: 1.55, opacity: 0.14, power: 3.4 },
  ].map(({ scale, opacity, power }) => {
    const geo = new THREE.SphereGeometry(radius * scale, 64, 64);
    const uniforms = {
      uTime: { value: 0 },
      uColor: { value: new THREE.Color(coronaColor) },
      uCameraPos: { value: new THREE.Vector3() },
      uOpacity: { value: opacity },
      uPower: { value: power },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader: CORONA_VERTEX,
      fragmentShader: CORONA_FRAGMENT,
      uniforms,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.FrontSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);
    return { mesh, uniforms };
  });

  // point light so nearby components (wind particles, CME) can react to sun color
  const light = new THREE.PointLight(new THREE.Color(coreColor), 2.2, radius * 40, 2);
  group.add(light);

  let flareTimer = 0;
  let flareDuration = 0;

  function setActiveRegions(regions = []) {
    const list = regions.slice(0, MAX_AR);
    for (let i = 0; i < MAX_AR; i++) {
      if (list[i]) {
        arPos[i].copy(latLonToVec3(list[i].lat, list[i].lon));
        arIntensity[i] = THREE.MathUtils.clamp(list[i].intensity ?? 0.5, 0, 1);
      } else {
        arIntensity[i] = 0;
      }
    }
  }

  function triggerFlare(lat, lon, classStrength = 0.6, durationSec = 1.6) {
    sunUniforms.uFlarePos.value.copy(latLonToVec3(lat, lon));
    flareDuration = durationSec;
    flareTimer = durationSec;
    sunUniforms.uFlareIntensity.value = THREE.MathUtils.clamp(classStrength, 0, 1);
  }

  function update(dt, telemetry = {}) {
    sunUniforms.uTime.value += dt;
    coronaLayers.forEach((l) => (l.uniforms.uTime.value += dt));

    if (telemetry.camera) {
      sunUniforms.uCameraPos.value.copy(telemetry.camera);
      coronaLayers.forEach((l) => l.uniforms.uCameraPos.value.copy(telemetry.camera));
    }

    if (flareTimer > 0) {
      flareTimer -= dt;
      const t = Math.max(flareTimer / flareDuration, 0);
      // quick attack, slow decay
      sunUniforms.uFlareIntensity.value = t;
      if (flareTimer <= 0) sunUniforms.uFlareIntensity.value = 0;
    }

    // solar wind speed subtly swells the corona
    if (typeof telemetry.windSpeed === 'number') {
      const norm = THREE.MathUtils.clamp((telemetry.windSpeed - 300) / 500, 0, 1);
      coronaLayers.forEach((l, i) => {
        l.mesh.scale.setScalar(1 + norm * 0.06 * (i + 1));
      });
    }
  }

  function dispose() {
    sunGeo.dispose();
    sunMat.dispose();
    coronaLayers.forEach((l) => {
      l.mesh.geometry.dispose();
      l.mesh.material.dispose();
    });
  }

  return { group, update, setActiveRegions, triggerFlare, dispose, radius };
}
