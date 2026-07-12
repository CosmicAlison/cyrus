import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

type LogEvent = {
  id: number;
  agent: string;
  message: string;
  severity: "info" | "warning" | "critical";
  time: string;
};

type CyrusEvent = {
  type: string;
  data: Record<string, any>;
};

function severityFor(eventType: string, data: Record<string, any>): LogEvent["severity"] {
  if (eventType === "pipeline_complete") {
    if (data.severity === "extreme" || data.severity === "high") return "critical";
    if (data.severity === "moderate") return "warning";
  }
  return "info";
}

function agentFor(eventType: string): string {
  if (eventType === "pipeline_complete") return "EXECUTIVE";
  if (eventType === "telemetry") return "HELIO";
  return "SYSTEM";
}

function messageFor(eventType: string, data: Record<string, any>): string {
  if (eventType === "pipeline_complete") {
    return `Pipeline complete — ${data.flare_class ?? "unknown"} flare, severity ${data.severity ?? "unknown"}, ${data.total_actions ?? 0} actions taken`;
  }
  if (eventType === "telemetry") {
    return `Wind ${data.wind_speed?.toFixed?.(0) ?? data.wind_speed} km/s, density ${data.density}`;
  }
  return JSON.stringify(data);
}

function describeEvent(event: CyrusEvent): { agent: string; message: string; severity: LogEvent["severity"] } {
  const { type, data } = event;

  switch (type) {
    case "pipeline_complete":
      return {
        agent: "EXECUTIVE",
        message: `Pipeline complete — ${data.flare_class ?? "unknown"} flare, severity ${data.severity}, ${data.total_actions ?? 0} actions taken`,
        severity: data.severity === "extreme" || data.severity === "high" ? "critical" : data.severity === "moderate" ? "warning" : "info",
      };
    case "telemetry":
      return {
        agent: "HELIO",
        message: `Wind ${data.wind_speed?.toFixed?.(0) ?? data.wind_speed} km/s, density ${data.density}`,
        severity: "info",
      };
    case "agent_started":
      return {
        agent: data.agent?.toUpperCase() ?? "AGENT",
        message: "Starting analysis…",
        severity: "info",
      };
    case "agent_complete":
      return {
        agent: data.agent?.toUpperCase() ?? "AGENT",
        message: data.message ?? `${data.actions_count ?? 0} actions taken`,
        severity: data.status === "error" ? "critical" : data.status === "partial" ? "warning" : "info",
      };
    case "tool_call":
      return {
        agent: data.agent?.toUpperCase() ?? "AGENT",
        message: `${data.tool} → ${typeof data.result === "object" ? JSON.stringify(data.result).slice(0, 80) : data.result}`,
        severity: "info",
      };
    case "pipeline_started":
      return {
        agent: "EXECUTIVE",
        message: `Pipeline started for job ${data.job_id}`,
        severity: "info",
      };
    case "agent_error":
      return {
        agent: data.agent?.toUpperCase() ?? "AGENT",
        message: `Error with ${data.agent?.toUpperCase ?? "AGENT"}: ${typeof data.error === "string" ? data.error.slice(0, 150) : "unknown error"}`,
        severity: "critical",
      };
    default:
      // Unknown event type — still show it rather than silently dropping it
      return {
        agent: type?.toUpperCase() ?? "SYSTEM",
        message: JSON.stringify(data).slice(0, 120),
        severity: "info",
      };
  }
}


function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
}


export default function CommandCenter({
  events,
  executiveBrief,
  solarTime,
  forecastTargetTime
}: {
  events: CyrusEvent[];
  executiveBrief: string | null;
  solarTime: string | null;
  forecastTargetTime: string | null;
}) {
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const lastLength = useRef(0);

  useEffect(() => {
    if (events.length === lastLength.current) return;

    const newEvents = events.slice(lastLength.current);
    lastLength.current = events.length;

const newLogs: LogEvent[] = newEvents.map((event, i) => {
    const { agent, message, severity } = describeEvent(event);
    return {
      id: Date.now() + i,
      agent,
      message,
      severity,
      time: new Date().toLocaleTimeString(),
    };
  });

  setLogs(prev => [...prev, ...newLogs].slice(-12)); 
  }, [events]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      style={{
        position: "absolute",
        left: "15px",
        top: "30px",
        width: "380px",
        maxHeight: "500px",
        padding: "20px",
        borderRadius: "24px",
        background: "rgba(10, 15, 30, 0.55)",
        backdropFilter: "blur(18px)",
        border: "1px solid rgba(255,255,255,0.12)",
        boxShadow: "0 0 40px rgba(255,120,30,0.15)",
        color: "white",
        overflow: "hidden",
        zIndex: 10,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <div style={{ fontSize: "12px", letterSpacing: "3px", opacity: 0.6 }}>CYRUS</div>
          <div style={{ fontSize: "20px", fontWeight: 600 }}>Command Center</div>
        </div>
        <div
          style={{
            padding: "6px 12px",
            borderRadius: "999px",
            background: "rgba(80,255,120,0.15)",
            border: "1px solid rgba(80,255,120,0.3)",
            fontSize: "12px",
          }}
        >
          ONLINE
        </div>
      </div>

      {executiveBrief && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            marginBottom: "16px",
            padding: "12px",
            borderRadius: "14px",
            background: "rgba(255,120,30,0.1)",
            border: "1px solid rgba(255,120,30,0.25)",
            fontSize: "12px",
            lineHeight: 1.6,
            maxHeight: "180px",
            overflowY: "auto",
          }}
        >
          <div style={{ fontSize: "11px", letterSpacing: "2px", opacity: 0.6, marginBottom: "4px" }}>
            EXECUTIVE BRIEF
          </div>
          {forecastTargetTime && (
            <div style={{ fontSize: "11px", opacity: 0.75, marginBottom: "8px", fontStyle: "italic" }}>
              Predicted observable sun at {formatTime(forecastTargetTime)}
            </div>
          )}
          <div className="brief-markdown">{executiveBrief}</div>
          {solarTime && (
            <div style={{ fontSize: "10px", opacity: 0.4, marginTop: "8px" }}>
              Generated from observation at {formatTime(solarTime)}
            </div>
          )}
        </motion.div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: "10px", overflowY: "auto", maxHeight: "220px" }}>
        {logs.map(log => (
          <motion.div
            key={log.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            style={{
              padding: "10px",
              borderRadius: "12px",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.06)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", opacity: 0.6 }}>
              <span>{log.agent}</span>
              <span>{log.time}</span>
            </div>
            <div
              style={{
                marginTop: "5px",
                fontSize: "13px",
                color: log.severity === "critical" ? "#ff8b8b" : log.severity === "warning" ? "#ffd27a" : "white",
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