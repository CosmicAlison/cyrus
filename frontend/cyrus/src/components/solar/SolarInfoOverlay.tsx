import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

type Props = {
  windSpeed: number;
  windDensity: number;
  kpIndex: number;
  euvChannel: string;
};

function kpDescription(kp: number): string {
  if (kp >= 7) return "Severe geomagnetic storm — aurora visible far from the poles";
  if (kp >= 5) return "Geomagnetic storm — aurora likely visible at mid latitudes";
  if (kp >= 4) return "Active — aurora possible at higher latitudes";
  return "Quiet — minimal geomagnetic disturbance";
}

function windDescription(speed: number): string {
  if (speed >= 700) return "Very fast — likely storm-driven";
  if (speed >= 500) return "Elevated — above typical background speed";
  return "Typical background solar wind";
}

export default function SolarInfoOverlay({ windSpeed, windDensity, kpIndex, euvChannel }: Props) {
  const [hovering, setHovering] = useState(false);

  return (
    <div
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 5,
      }}
    >
      <AnimatePresence>
        {hovering && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            style={{
              position: "absolute",
              right: "20px",
              top: "30px",
              width: "300px",
              padding: "18px",
              borderRadius: "20px",
              background: "rgba(10, 15, 30, 0.7)",
              backdropFilter: "blur(18px)",
              border: "1px solid rgba(255,255,255,0.12)",
              boxShadow: "0 0 40px rgba(255,120,30,0.12)",
              color: "white",
              fontSize: "12px",
              lineHeight: 1.5,
              pointerEvents: "none",
            }}
          >
            <div style={{ fontSize: "11px", letterSpacing: "2px", opacity: 0.6, marginBottom: "10px" }}>
              READING THE SUN
            </div>

            <div style={{ marginBottom: "12px" }}>
              <div style={{ fontWeight: 600, marginBottom: "2px" }}>Glowing rings</div>
              <div style={{ opacity: 0.8 }}>
                Concentric halos showing extreme-UV light at different wavelengths —
                each color reveals a different layer of the sun's atmosphere, from
                cooler coronal loops (teal/gold) to hot flare plasma (violet) and
                the chromosphere (orange-red). Currently showing:{" "}
                <strong>{euvChannel === "all" ? "all channels" : `${euvChannel}Å`}</strong>.
              </div>
            </div>

            <div style={{ marginBottom: "12px" }}>
              <div style={{ fontWeight: 600, marginBottom: "2px" }}>Outer arc (Kp dial)</div>
              <div style={{ opacity: 0.8 }}>
                Sweeps from green to red as geomagnetic activity rises — a real-time
                readout of how disturbed Earth's magnetic field is, and how likely
                aurora are visible right now.
              </div>
              <div style={{ marginTop: "4px", color: "#ffcf8b" }}>
                Kp {kpIndex} — {kpDescription(kpIndex)}
              </div>
            </div>

            <div style={{ marginBottom: "12px" }}>
              <div style={{ fontWeight: 600, marginBottom: "2px" }}>Outward particles</div>
              <div style={{ opacity: 0.8 }}>
                A live stream representing the solar wind — charged particles
                constantly flowing outward from the sun. Speed and density track
                real measurements.
              </div>
              <div style={{ marginTop: "4px", color: "#ffcf8b" }}>
                {windSpeed.toFixed(0)} km/s — {windDescription(windSpeed)}
                <br />
                Density: {(windDensity * 100).toFixed(0)}%
              </div>
            </div>

            <div style={{ opacity: 0.5, fontSize: "10px", marginTop: "10px" }}>
              Hover anywhere over the sun to view this panel
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}