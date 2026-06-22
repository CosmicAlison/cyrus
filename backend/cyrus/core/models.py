from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ForecastRun(Base):
    """
    A single Surya inference job triggered by POST /api/forecast.
    Tracks the full lifecycle from request → Surya output → agent completion.
    """
    __tablename__ = "forecast_runs"

    id = Column(String(36), primary_key=True)           # UUID
    status = Column(String(32), nullable=False, default="pending")
    # pending | surya_running | surya_complete | agents_running | complete | failed

    forecast_start = Column(DateTime(timezone=True), nullable=False)
    forecast_end = Column(DateTime(timezone=True), nullable=False)
    rollout_steps = Column(Integer, nullable=False, default=12)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)

    # Relationships
    threat_events = relationship("ThreatEvent", back_populates="forecast_run")
    agent_actions = relationship("AgentAction", back_populates="forecast_run")
    mitigation_logs = relationship("MitigationLog", back_populates="forecast_run")


class ThreatEvent(Base):
    """
    Structured threat assessment produced by Agent 1 (HelioAnalyst)
    from a raw Surya forecast payload.
    """
    __tablename__ = "threat_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id = Column(String(36), ForeignKey("forecast_runs.id"), nullable=False)

    # Core threat signals
    severity = Column(String(16), nullable=False)       # low | moderate | high | extreme
    flare_class = Column(String(4), nullable=False)     # A/B/C/M/X
    flare_probability = Column(Float, nullable=False)
    euv_flux = Column(Float, nullable=True)
    magnetic_complexity = Column(Float, nullable=True)
    solar_wind_proxy = Column(Float, nullable=True)

    # Full ThreatPayload as JSON (for agent consumption)
    payload = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    forecast_run = relationship("ForecastRun", back_populates="threat_events")


class AgentAction(Base):
    """
    A single action taken by one of the 5 Cyrus agents.
    Each agent logs one or more actions per ThreatEvent.
    """
    __tablename__ = "agent_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id = Column(String(36), ForeignKey("forecast_runs.id"), nullable=False)
    threat_event_id = Column(Integer, ForeignKey("threat_events.id"), nullable=True)

    agent = Column(String(32), nullable=False)
    # helio_analyst | satops | gridops | commsops | commander

    action_type = Column(String(64), nullable=False)
    # e.g. "safe_mode_command", "load_reroute", "hf_advisory", "executive_brief"

    description = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)       # tool call inputs/outputs
    status = Column(String(16), nullable=False, default="success")  # success | error

    created_at = Column(DateTime(timezone=True), default=utcnow)

    forecast_run = relationship("ForecastRun", back_populates="agent_actions")


class MitigationLog(Base):
    """
    Final executive summary produced by Agent 5 (Commander).
    One per forecast run.
    """
    __tablename__ = "mitigation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id = Column(String(36), ForeignKey("forecast_runs.id"), nullable=False)

    executive_brief = Column(Text, nullable=False)
    actions_summary = Column(JSON, nullable=False)  # {satops: [...], gridops: [...], commsops: [...]}
    severity = Column(String(16), nullable=False)
    total_actions_taken = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    forecast_run = relationship("ForecastRun", back_populates="mitigation_logs")


class Satellite(Base):
    __tablename__ = "satellites"

    id = Column(String(16), primary_key=True)       # e.g. "SAT-A"
    name = Column(String(128), nullable=False)
    orbit_type = Column(String(16), nullable=False) # LEO / MEO / GEO
    altitude_km = Column(Float, nullable=False)
    inclination_deg = Column(Float, nullable=False)
    status = Column(String(32), nullable=False, default="nominal")
    # nominal | safe_mode | reorienting | thruster_burn | degraded

    last_command = Column(String(128), nullable=True)
    last_command_at = Column(DateTime(timezone=True), nullable=True)



class GridNode(Base):
    __tablename__ = "grid_nodes"

    id = Column(String(16), primary_key=True)       # e.g. "SUB-NE-01"
    name = Column(String(128), nullable=False)
    region = Column(String(64), nullable=False)
    node_type = Column(String(32), nullable=False)  # substation | transformer | generator
    capacity_mw = Column(Float, nullable=False)
    gic_vulnerability = Column(Float, nullable=False)  # 0-1 (higher = more exposed)
    status = Column(String(32), nullable=False, default="online")
    # online | decoupled | load_reduced | offline

    last_action = Column(String(128), nullable=True)
    last_action_at = Column(DateTime(timezone=True), nullable=True)


class FlightRoute(Base):
    __tablename__ = "flight_routes"

    id = Column(String(16), primary_key=True)       # e.g. "POLAR-01"
    flight_number = Column(String(16), nullable=False)
    origin = Column(String(4), nullable=False)      # ICAO code
    destination = Column(String(4), nullable=False)
    route_type = Column(String(32), nullable=False) # polar | non-polar | oceanic
    hf_dependency = Column(String(8), nullable=False)  # high | medium | low
    status = Column(String(32), nullable=False, default="nominal")
    advisory = Column(Text, nullable=True)
    advisory_issued_at = Column(DateTime(timezone=True), nullable=True)