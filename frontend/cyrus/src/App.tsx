import StarField from './components/background/StarField.tsx'
import './App.css'
import { Canvas } from '@react-three/fiber'
import Sun from './components/solar/Sun.tsx'

function App() {
  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <StarField />

      <Canvas
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 1,
        }}
        camera={{ position: [0, 0, 8], fov: 45 }}
      >
        <ambientLight intensity={0.15} />

        <Sun
          severity="moderate"
          flareActive={false}
        />
      </Canvas>
    </div>
  );
}

export default App
