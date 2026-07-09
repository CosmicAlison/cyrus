import StarField from './components/background/StarField.tsx'
import './App.css'
import SolarSystemScene from './components/solar/SolarSystemScene.jsx'

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
        <SolarSystemScene
        activeRegions={[
          { lat: 12, lon: -40, intensity: 0.8 },
          { lat: -25, lon: 60, intensity: 0.4 },
        ]}
        windSpeed={520}
        windDensity={0.6}
        kpIndex={5}
        euvChannel="193"
        />
    </div>
  );
}

export default App
