import { motion } from "framer-motion";
import { useEffect, useState } from "react";

type LogEvent = {
  id: number;
  agent: string;
  message: string;
  severity: "info" | "warning" | "critical";
  time: string;
};

const mockLogs: LogEvent[] = [
  {
    id: 1,
    agent: "HELIO",
    message: "Solar wind velocity increased to 520 km/s",
    severity: "info",
    time: "15:42:01",
  },
  {
    id: 2,
    agent: "GRIDOPS",
    message: "Transformer vulnerability analysis complete",
    severity: "warning",
    time: "15:42:04",
  },
  {
    id: 3,
    agent: "SATOPS",
    message: "LEO satellite drag risk elevated",
    severity: "warning",
    time: "15:42:08",
  },
  {
    id: 4,
    agent: "COMMSOPS",
    message: "HF blackout probability calculated",
    severity: "critical",
    time: "15:42:11",
  },
];


export default function CommandCenter() {
  const [logs, setLogs] = useState<LogEvent[]>(mockLogs);

  // temporary simulation
  useEffect(() => {
    const interval = setInterval(() => {
      const newLog: LogEvent = {
        id: Date.now(),
        agent: "EXECUTIVE",
        message: "Monitoring global threat state",
        severity: "info",
        time: new Date().toLocaleTimeString(),
      };

      setLogs(prev => [...prev.slice(-5), newLog]);
    }, 6000);

    return () => clearInterval(interval);
  }, []);


  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      style={{
        position: "absolute",
        left: "32px",
        bottom: "32px",
        width: "380px",
        maxHeight: "320px",

        padding: "20px",

        borderRadius: "24px",

        background:
          "rgba(10, 15, 30, 0.55)",

        backdropFilter:
          "blur(18px)",

        border:
          "1px solid rgba(255,255,255,0.12)",

        boxShadow:
          "0 0 40px rgba(255,120,30,0.15)",

        color: "white",

        overflow: "hidden",

        zIndex: 10,
      }}
    >

      <div
        style={{
          display:"flex",
          justifyContent:"space-between",
          alignItems:"center",
          marginBottom:"16px"
        }}
      >

        <div>
          <div
            style={{
              fontSize:"12px",
              letterSpacing:"3px",
              opacity:0.6
            }}
          >
            CYRUS
          </div>

          <div
            style={{
              fontSize:"20px",
              fontWeight:600
            }}
          >
            Command Center
          </div>
        </div>


        <div
          style={{
            padding:"6px 12px",
            borderRadius:"999px",
            background:"rgba(80,255,120,0.15)",
            border:"1px solid rgba(80,255,120,0.3)",
            fontSize:"12px"
          }}
        >
          ONLINE
        </div>

      </div>


      <div
        style={{
          display:"flex",
          flexDirection:"column",
          gap:"10px",
          overflowY:"auto",
          maxHeight:"220px"
        }}
      >

        {logs.map(log => (

          <motion.div
            key={log.id}
            initial={{opacity:0,x:20}}
            animate={{opacity:1,x:0}}
            style={{
              padding:"10px",
              borderRadius:"12px",

              background:
                "rgba(255,255,255,0.05)",

              border:
                "1px solid rgba(255,255,255,0.06)"
            }}
          >

            <div
              style={{
                display:"flex",
                justifyContent:"space-between",
                fontSize:"11px",
                opacity:0.6
              }}
            >
              <span>{log.agent}</span>
              <span>{log.time}</span>
            </div>


            <div
              style={{
                marginTop:"5px",
                fontSize:"13px",
                color:
                  log.severity === "critical"
                    ? "#ff8b8b"
                    : log.severity === "warning"
                    ? "#ffd27a"
                    : "white"
              }}
            >
              {log.message}
            </div>


          </motion.div>

        ))}

      </div>

    </motion.div>
  );
}