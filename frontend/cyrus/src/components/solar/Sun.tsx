import { Sphere } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";

type Severity = "low" | "moderate" | "high" | "extreme";

interface SunProps {
  severity?: Severity;
  flareActive?: boolean;
}

export default function Sun({
  severity = "low",
  flareActive = false,
}: SunProps) {
  const coreRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    const pulse =
      1 +
      Math.sin(t * (severity === "extreme" ? 4 : 2)) *
        (severity === "extreme" ? 0.05 : 0.025);

    if (coreRef.current) {
      coreRef.current.scale.setScalar(pulse);
      coreRef.current.rotation.y += 0.0015;
    }

    if (glowRef.current) {
      glowRef.current.scale.setScalar(
        pulse * (flareActive ? 1.15 : 1.05)
      );

      const mat = glowRef.current.material as THREE.MeshBasicMaterial;

      mat.opacity =
        0.18 +
        Math.sin(t * 2.5) * 0.04 +
        (flareActive ? 0.08 : 0);
    }
  });

  const emissive =
    severity === "extreme"
      ? 10
      : severity === "high"
      ? 8
      : severity === "moderate"
      ? 6
      : 4;

  return (
    <group>

      <Sphere ref={glowRef} args={[2.9, 64, 64]}>
        <meshBasicMaterial
          color="#ff751f"
          transparent
          opacity={0.18}
          side={THREE.BackSide}
        />
      </Sphere>

      <Sphere args={[2.65, 128, 128]}>
        <meshStandardMaterial
          color="#ff8d2d"
          emissive="#ff5e00"
          emissiveIntensity={emissive * 0.6}
          roughness={1}
          metalness={0}
        />
      </Sphere>

      <Sphere ref={coreRef} args={[2.45, 128, 128]}>
        <meshStandardMaterial
          color="#ff7b47"
          emissive="#ff6b00"
          emissiveIntensity={emissive}
          roughness={0.9}
          metalness={0}
        />
      </Sphere>

      <Sphere args={[2.15, 128, 128]}>
        <meshBasicMaterial color="#a1450f" />
      </Sphere>

      <pointLight
        intensity={250}
        distance={60}
        decay={2}
        color="#ff9b42"
      />

    </group>
  );
}