from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

SeverityLevel = Literal["low", "moderate", "high", "extreme"]


class FlareForecast(BaseModel):
    prediction: int
    probability: float = Field(ge=0.0, le=1.0)
    time_input: str
    time_target: str
    goes_class: str


class RegionCentroid(BaseModel):
    x: float
    y: float
    area_frac: float
    disk_proximity: float


class ARSnapshot(BaseModel):
    timestamp_input: str
    timestamp_target: str
    active_region_count: int
    centroids: list[RegionCentroid]
    total_area_frac: float
    dominant_region: RegionCentroid | None = None


class EUVForecast(BaseModel):
    time_input: str
    time_target: str
    integrated_flux: float
    soft_xray_flux: float
    thermospheric_flux: float
    he2_flux: float
    spectrum_mini: list[float] = Field(default_factory=list)


class WindForecast(BaseModel):
    time_input: str
    time_target: str
    speed_kms: float
    bz_gsm: float
    bx_gse: float
    by_gsm: float
    density: float
    cached: bool = False


class ThreatSummary(BaseModel):
    severity: SeverityLevel
    composite_risk: float
    flare_probability: float
    gic_risk: float
    atmospheric_drag_risk: float
    hf_blackout_risk: float
    bz_southward: bool
    bz_nT: float
    wind_speed_kms: float
    active_region_count: int
    activate_commsops: bool
    activate_satops: bool
    activate_gridops: bool


class RawForecastPayload(BaseModel):
    """Published by surya_service → consumed by helio_worker (Agent 1).
    Mirrors Aggregator.build()'s output shape directly — no adapter needed."""
    job_id: str
    solar_time: str
    real_time: str
    flare: FlareForecast
    ar_now: ARSnapshot
    ar_flare: ARSnapshot | None = None
    euv: EUVForecast
    wind: WindForecast
    threat_summary: ThreatSummary


# Agent 1 (HelioAnalyst) → cyrus.threats

class ThreatPayload(BaseModel):
    job_id: str
    threat_event_id: int
    severity: SeverityLevel
    flare_probability: float
    goes_class: str
    euv_impact: float
    magnetic_complexity: float
    solar_wind_speed_proxy: float
    atmospheric_drag_risk: float
    active_region_x: float
    active_region_y: float
    peak_timestamp: str
    analyst_summary: str
    activate_satops: bool
    activate_gridops: bool
    activate_commsops: bool

    @property
    def flare_class(self) -> str:
        return self.goes_class

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