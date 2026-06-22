from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class ActiveRegion(BaseModel):
    x: int
    y: int
    radius_pixels: int


class TimestepForecast(BaseModel):
    timestamp: str
    flare_probability: float = Field(ge=0.0, le=1.0)
    estimated_flare_class: Literal["X", "M", "C", "B", "A"]
    peak_intensity_per_channel: dict[str, float]
    delta_intensity_per_channel: dict[str, float]
    active_region: ActiveRegion
    euv_integrated_flux: float = Field(ge=0.0, le=1.0)
    magnetic_complexity: float = Field(ge=0.0)
    solar_wind_proxy: float


class ForecastSummary(BaseModel):
    max_flare_probability: float
    peak_flare_class: Literal["X", "M", "C", "B", "A"]
    peak_timestamp: str
    mean_euv_flux: float
    max_magnetic_complexity: float


class RawForecastPayload(BaseModel):
    """Published by surya_service → consumed by helio_worker (Agent 1)."""
    job_id: str
    forecast_start: str
    forecast_end: str
    timesteps: list[TimestepForecast]
    summary: ForecastSummary


# ── Agent 1 (HelioAnalyst) → cyrus.threats ───────────────────────────────────

SeverityLevel = Literal["low", "moderate", "high", "extreme"]

class ThreatPayload(BaseModel):
    """
    Structured threat assessment produced by Agent 1.
    Published to cyrus.threats — fan-out to Agents 2, 3, 4.
    """
    job_id: str
    threat_event_id: int
    severity: SeverityLevel
    flare_class: Literal["X", "M", "C", "B", "A"]
    flare_probability: float

    # Signals for downstream agents
    euv_impact: float = Field(ge=0.0, le=1.0, description="EUV flux — CommsOps severity driver")
    magnetic_complexity: float = Field(ge=0.0, description="Bz variance — CME / GridOps driver")
    solar_wind_speed_proxy: float = Field(description="HMI Doppler proxy — GIC driver")
    atmospheric_drag_risk: float = Field(ge=0.0, le=1.0, description="EUV heating — SatOps driver")

    # Active region info
    active_region_x: int
    active_region_y: int
    peak_timestamp: str

    # Natural language context for LLM agents
    analyst_summary: str = Field(description="Agent 1 plain-English assessment")

    # Which agents should act (set by LangGraph severity routing)
    activate_satops: bool
    activate_gridops: bool
    activate_commsops: bool

class AgentReport(BaseModel):
    """
    Published by each operational agent (2/3/4) after completing mitigation actions.
    Consumed by Agent 5 (Commander).
    """
    job_id: str
    agent: Literal["satops", "gridops", "commsops"]
    status: Literal["success", "partial", "skipped", "error"]
    actions_taken: list[dict[str, Any]] = Field(default_factory=list)
    summary: str
    completed_at: str


class ExecutiveBrief(BaseModel):
    """Final output of a complete Cyrus pipeline run."""
    job_id: str
    severity: SeverityLevel
    flare_class: str
    flare_probability: float

    satops_summary: str
    gridops_summary: str
    commsops_summary: str
    executive_brief: str

    total_actions: int
    completed_at: str

class ForecastRequest(BaseModel):
    start_datetime: datetime
    end_datetime: datetime
    rollout_steps: int = Field(default=12, ge=1, le=120)


class ForecastResponse(BaseModel):
    job_id: str
    status: str
    message: str