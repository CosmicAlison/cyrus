import StarField from './components/background/StarField.tsx'
import './App.css'
import CommandCenter from './components/command/CommandCenter.tsx';
import SolarSystemScene from './components/solar/SolarSystemScene.jsx'
import { useCyrusStream } from './hooks/useCyrusStream.tsx';
import { useState, useEffect} from 'react';
import SolarInfoOverlay from './components/solar/SolarInfoOverlay.tsx';

function App() {
  const apiUrl = import.meta.env.VITE_API_URL;
  const events = useCyrusStream("mock-x-flare-test-001", apiUrl);
  const [solarState, setSolarState] = useState({
    activeRegions: [
          //{ lat: 12, lon: -40, intensity: 0.8 },
          //{ lat: -25, lon: 60, intensity: 0.4 },
        ],
    euvChannel:"all",
    windSpeed: 120,
    windDensity: 0.5,
    kpIndex: 5,
    flareEvent: null,
    cmeEvent: null,
    executiveBrief: null as string | null,
    forecastTargetTime:null,
    solarTime:null
  });

  useEffect(() => {
    const event = events.at(-1);

    if (!event) return;

    console.log(event);


    if (event.type === "pipeline_complete") {

      const strength =
        event.data.flare_class?.startsWith("X")
          ? 1
          : event.data.flare_class?.startsWith("M")
          ? 0.7
          : 0.4;

      const lat = event.data.active_region_lat ?? 12;
      const lon = event.data.active_region_lon ?? 25;

      setSolarState(prev => ({
        ...prev,

        activeRegions: event.data.active_regions ?? prev.activeRegions,
        windSpeed: event.data.wind_speed ?? prev.windSpeed,
        windDensity: event.data.wind_density ?? prev.windDensity,

        euvChannel:
        event.data.flare_class?.startsWith("X") || event.data.flare_class?.startsWith("M")
          ? "304"
          : "171",

        flareEvent: {
          lat,
          lon,
          classStrength: strength,
          key: Date.now()
        },

        cmeEvent: {
          lat,
          lon,
          intensity: strength,
          speed: 1200,
          key: Date.now()
        },
        forecastTargetTime: event.data.forecast_target_time ?? prev.forecastTargetTime,
        solarTime: event.data.solar_time ?? prev.solarTime,
        executiveBrief: event.data.executive_brief ?? prev.executiveBrief,
        kpIndex: strength > 0.8 ? 8 : 5
      }));

    }

  }, [events]);
  
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
        {...solarState}
        />
      <SolarInfoOverlay
        windSpeed={solarState.windSpeed}
        windDensity={solarState.windDensity}
        kpIndex={solarState.kpIndex}
        euvChannel={solarState.euvChannel}
      />
      <CommandCenter events={events} executiveBrief={solarState.executiveBrief} solarTime={solarState.solarTime} forecastTargetTime={solarState.forecastTargetTime} />
    </div>
  );
}

export default App