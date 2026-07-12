"""
Agent 1 — The Heliophysics Analyst.

Consumes RawForecastPayload from cyrus.raw_forecast.
Interprets Surya's parsed .nc signals using domain knowledge.
Produces a ThreatPayload for downstream agents.
"""

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.config import settings
from core.schemas import RawForecastPayload, ThreatPayload

log = logging.getLogger(__name__)


class HelioAnalyst(BaseAgent):
    name = "helio_analyst"

    @property
    def system_prompt(self) -> str:
        return """You are a senior heliophysicist in NASA's Space Weather Operations Center.
You receive a pre-computed threat assessment from the Surya Foundation Model pipeline and
write a concise plain-English analyst summary for the executive dashboard. You may override
the pre-computed severity only if the raw signals clearly warrant it.

Output ONLY 1-2 sentence plain English summary of actions you took
"""

    def analyse(self, raw_payload: dict[str, Any], threat_event_id: int) -> ThreatPayload:
        parsed = RawForecastPayload.model_validate(raw_payload)
        ts = parsed.threat_summary

        user_message = self._build_prompt(parsed)
        log.info("[helio_analyst] Analysing forecast %s — severity(rule-based)=%s",
                  parsed.job_id, ts.severity)

        raw_response = self.invoke(user_message)

        if raw_response:
            analyst_summary = raw_response.strip()
        else:
            analyst_summary = self._fallback_analysis(parsed)["analyst_summary"]

        threat = ThreatPayload(
            job_id=parsed.job_id,
            threat_event_id=threat_event_id,

            severity=ts.severity,

            flare_probability=parsed.flare.probability,
            goes_class=parsed.flare.goes_class,

            euv_impact=parsed.euv.integrated_flux,
            magnetic_complexity=ts.gic_risk,
            solar_wind_speed_proxy=parsed.wind.speed_kms,

            atmospheric_drag_risk=ts.atmospheric_drag_risk,

            active_region_x=(
                parsed.ar_now.dominant_region.x
                if parsed.ar_now.dominant_region else 0.0
            ),
            active_region_y=(
                parsed.ar_now.dominant_region.y
                if parsed.ar_now.dominant_region else 0.0
            ),

            peak_timestamp=parsed.flare.time_target,

            analyst_summary=analyst_summary,

            activate_satops=ts.activate_satops,
            activate_gridops=ts.activate_gridops,
            activate_commsops=ts.activate_commsops,
        )
        log.info("[helio_analyst] severity=%s activate=[satops=%s gridops=%s commsops=%s]",
                  threat.severity, threat.activate_satops, threat.activate_gridops, threat.activate_commsops)
        return threat

    def _build_prompt(self, payload: RawForecastPayload) -> str:
        ts = payload.threat_summary
        return f"""
        You are a NASA space weather analyst.

        Summarise this Surya prediction for an executive dashboard.

        Rules:
        - Do not change severity.
        - Do not invent values.
        - Explain the operational impact.
        - Maximum 3 sentences.

        Forecast:
        Flare probability: {payload.flare.probability:.1%}
        GOES class: {payload.flare.goes_class}
        Severity: {ts.severity}

        Solar wind:
        Speed: {payload.wind.speed_kms} km/s
        Bz: {payload.wind.bz_gsm} nT

        Risks:
        GIC risk: {ts.gic_risk}
        HF blackout risk: {ts.hf_blackout_risk}
        Atmospheric drag risk: {ts.atmospheric_drag_risk}

        Explain what this means.
        """

    def _fallback_analysis(self, payload: RawForecastPayload) -> dict[str, Any]:
        ts = payload.threat_summary
        return {
            "severity": ts.severity,
            "analyst_summary": (
                f"Surya predicts flare probability {payload.flare.probability:.0%} "
                f"({payload.flare.goes_class or 'class n/a'}). "
                f"Composite risk {ts.composite_risk:.2f}, southward Bz: {ts.bz_southward}. "
                f"Protocols activated per threshold rules."
            ),
        }