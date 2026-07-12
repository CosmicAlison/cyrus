import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { createSun } from './Sun';
import { createSolarWind } from './SolarWind';
import { createEUVHalo } from './EUVHalo';
import { createCMEBurst } from './CMEBurst';
import { createKpDial } from './KpDial';
import { useCyrusStream } from '../../hooks/useCyrusStream.tsx';
/**
 * SolarSystemScene
 * Drop this into your dashboard on top of / alongside your existing night-sky
 * background. It owns its own transparent WebGL canvas, so it composites
 * cleanly over other layers.
 *
 * Props (all optional, all live-updatable):
 *   activeRegions: [{ lat, lon, intensity }]        // up to 8
 *   windSpeed:     number (km/s)                    // ~300-800 typical
 *   windDensity:   number (0-1)
 *   kpIndex:       number (0-9)
 *   euvChannel:    '171' | '193' | '211' | '304' | 'all'
 *   flareEvent:    { lat, lon, classStrength (0-1), key } // bump `key` to re-fire
 *   cmeEvent:      { lat, lon, intensity (0-1), speed, key }
 *   className:     string (Tailwind classes for the wrapper div)
 */
export default function SolarSystemScene({
  activeRegions = [],
  windSpeed = 420,
  windDensity = 0.5,
  kpIndex = 2,
  euvChannel = 'all',
  flareEvent = null,
  cmeEvent = null,
  className = 'absolute inset-0 w-full h-full',
}) {
  const mountRef = useRef(null);
  const stateRef = useRef({});

  //one-time scene setup
  useEffect(() => {
    const mount = mountRef.current;
    const width = mount.clientWidth;
    const height = mount.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 200);
    camera.position.set(0, 2.2, 13);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0); // transparent so your night sky shows through
    mount.appendChild(renderer.domElement);

    const sun = createSun({ radius: 3 });
    const wind = createSolarWind({});
    const halo = createEUVHalo({});
    const cme = createCMEBurst({ sunRadius: 3 });
    const dial = createKpDial({ radius: 9 });

    scene.add(sun.group, wind.group, halo.group, cme.group, dial.group);

    const clock = new THREE.Clock();
    let frameId;

    function animate() {
      frameId = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      const s = stateRef.current;

      sun.update(dt, { camera: camera.position, windSpeed: s.windSpeed });
      wind.update(dt, { windSpeed: s.windSpeed, density: s.windDensity });
      halo.update(dt);
      cme.update(dt);
      dial.update(dt, { kpIndex: s.kpIndex });

      renderer.render(scene, camera);
    }
    animate();

    function handleResize() {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener('resize', handleResize);

    // expose imperative handles for the props-sync effect below
    stateRef.current.api = { sun, wind, halo, cme, dial };

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener('resize', handleResize);
      sun.dispose();
      wind.dispose();
      halo.dispose();
      cme.dispose();
      dial.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  // keep live telemetry values available to the animation loop
  useEffect(() => {
    stateRef.current.windSpeed = windSpeed;
    stateRef.current.windDensity = windDensity;
    stateRef.current.kpIndex = kpIndex;
  }, [windSpeed, windDensity, kpIndex]);

  //active regions
  useEffect(() => {
    stateRef.current.api?.sun.setActiveRegions(activeRegions);
  }, [activeRegions]);

  // EUV channel toggle
  useEffect(() => {
    stateRef.current.api?.halo.setChannel(euvChannel);
  }, [euvChannel]);

  //flare trigger (fires once per unique `key`)
  useEffect(() => {
    if (!flareEvent) return;
    stateRef.current.api?.sun.triggerFlare(
      flareEvent.lat,
      flareEvent.lon,
      flareEvent.classStrength ?? 0.6
    );
  }, [flareEvent?.key]);

  //CME trigger (fires once per unique `key`)
  useEffect(() => {
    if (!cmeEvent) return;
    stateRef.current.api?.cme.trigger(cmeEvent.lat, cmeEvent.lon, {
      intensity: cmeEvent.intensity,
      speed: cmeEvent.speed,
    });
  }, [cmeEvent?.key]);

  return <div ref={mountRef} className={className} />;
}
