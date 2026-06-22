"""
Agent 1 — The Heliophysics Analyst.

Consumes RawForecastPayload from cyrus.raw_forecast.
Interprets Surya's parsed .nc signals using domain knowledge.
Produces a ThreatPayload for downstream agents.
"""

import json
import logging
from typing import Any

from cyrus.agents.base_agent import BaseAgent
from cyrus.core.config import settings
from cyrus.core.schemas import RawForecastPayload, ThreatPayload

log = logging.getLogger(__name__)


class HelioAnalyst(BaseAgent):
    name = "helio_analyst"

    @property
    def system_prompt(self) -> str:
        return """You are a senior heliophysicist and space weather analyst working in NASA's
Space Weather Operations Center. You receive parsed telemetry from the Surya Foundation Model
(NASA/IBM) and produce structured threat assessments for operational response teams.

Your role:
1. Interpret solar observation data: AIA channel intensities, EUV flux, magnetic complexity (Bz),
   and solar wind proxies from HMI Doppler data.
2. Classify the threat severity and flare class based on the signals.
3. Determine which infrastructure sectors are at risk and require immediate action.
4. Write a concise plain-English analyst summary for the executive dashboard.

Output ONLY a valid JSON object matching this schema exactly — no markdown, no preamble:
{
  "severity": "low|moderate|high|extreme",
  "analyst_summary": "<2-3 sentence plain English assessment>",
  "activate_satops": true|false,
  "activate_gridops": true|false,
  "activate_commsops": true|false,
  "atmospheric_drag_risk": 0.0-1.0,
  "euv_impact": 0.0-1.0,
  "solar_wind_speed_proxy": -1.0 to 1.0
}

Severity thresholds:
- extreme: X-class flare (prob >= 0.85) AND magnetic complexity > 0.7
- high: M-class (prob >= 0.65) OR magnetic complexity > 0.6
- moderate: C-class (prob >= 0.45) OR EUV flux > 0.4
- low: anything below the above

Activation rules:
- activate_commsops: flare prob > 0.25 (HF blackout risk even for C-class)
- activate_satops: flare prob > 0.45 (atmospheric drag from EUV heating)
- activate_gridops: flare prob > 0.65 AND magnetic_complexity > 0.5 (GIC risk)

atmospheric_drag_risk: derived primarily from AIA 171/304 EUV flux prediction.
euv_impact: mean EUV integrated flux from the forecast summary.
solar_wind_speed_proxy: normalised HMI Doppler proxy."""

    def analyse(
        self,
        raw_payload: dict[str, Any],
        threat_event_id: int,
    ) -> ThreatPayload:
        """
        Run analysis on a raw forecast payload.
        Returns a validated ThreatPayload.
        """
        parsed = RawForecastPayload.model_validate(raw_payload)
        summary = parsed.summary
        peak_ts = parsed.timesteps[
            max(
                range(len(parsed.timesteps)),
                key=lambda i: parsed.timesteps[i].flare_probability,
            )
        ]

        user_message = self._build_prompt(parsed)
        log.info(
            "[helio_analyst] Analysing forecast %s — peak class: %s (%.2f)",
            parsed.job_id,
            summary.peak_flare_class,
            summary.max_flare_probability,
        )

        raw_response = self.invoke(user_message)

        try:
            llm_output = json.loads(raw_response.strip())
        except json.JSONDecodeError:
            log.warning("[helio_analyst] LLM returned non-JSON, using fallback")
            llm_output = self._fallback_analysis(parsed)

        threat = ThreatPayload(
            job_id=parsed.job_id,
            threat_event_id=threat_event_id,
            severity=llm_output["severity"],
            flare_class=summary.peak_flare_class,
            flare_probability=summary.max_flare_probability,
            euv_impact=llm_output.get("euv_impact", summary.mean_euv_flux),
            magnetic_complexity=summary.max_magnetic_complexity,
            solar_wind_speed_proxy=llm_output.get("solar_wind_speed_proxy", 0.0),
            atmospheric_drag_risk=llm_output.get("atmospheric_drag_risk", summary.mean_euv_flux),
            active_region_x=peak_ts.active_region.x,
            active_region_y=peak_ts.active_region.y,
            peak_timestamp=summary.peak_timestamp,
            analyst_summary=llm_output["analyst_summary"],
            activate_satops=llm_output["activate_satops"],
            activate_gridops=llm_output["activate_gridops"],
            activate_commsops=llm_output["activate_commsops"],
        )

        log.info(
            "[helio_analyst] Assessment: severity=%s activate=[satops=%s gridops=%s commsops=%s]",
            threat.severity,
            threat.activate_satops,
            threat.activate_gridops,
            threat.activate_commsops,
        )

        return threat

    def _build_prompt(self, payload: RawForecastPayload) -> str:
        s = payload.summary
        peak = max(payload.timesteps, key=lambda t: t.flare_probability)

        return f"""Surya Foundation Model Forecast Report
========================================
Job ID: {payload.job_id}
Forecast window: {payload.forecast_start} → {payload.forecast_end}
Rollout timesteps: {len(payload.timesteps)}

SUMMARY SIGNALS:
- Peak flare probability: {s.max_flare_probability:.3f}
- Estimated peak flare class: {s.peak_flare_class}
- Peak timestamp: {s.peak_timestamp}
- Mean EUV integrated flux: {s.mean_euv_flux:.3f}
- Max magnetic complexity (Bz variance): {s.max_magnetic_complexity:.3f}

PEAK TIMESTEP DETAIL ({peak.timestamp}):
- AIA 131 peak intensity: {peak.peak_intensity_per_channel.get('131', 0):.3f}
- AIA 171 peak intensity: {peak.peak_intensity_per_channel.get('171', 0):.3f}
- AIA 304 peak intensity: {peak.peak_intensity_per_channel.get('304', 0):.3f}
- AIA 94 peak intensity:  {peak.peak_intensity_per_channel.get('94', 0):.3f}
- Delta 131 (rate of change): {peak.delta_intensity_per_channel.get('131', 0):.3f}
- EUV integrated flux: {peak.euv_integrated_flux:.3f}
- Magnetic complexity: {peak.magnetic_complexity:.3f}
- Solar wind proxy (HMI Doppler): {peak.solar_wind_proxy:.3f}
- Active region location: ({peak.active_region.x}, {peak.active_region.y}), radius: {peak.active_region.radius_pixels}px

Produce your structured threat assessment JSON now."""

    def _fallback_analysis(self, payload: RawForecastPayload) -> dict[str, Any]:
        """Rule-based fallback if LLM output is malformed."""
        prob = payload.summary.max_flare_probability
        cls = payload.summary.peak_flare_class
        mag = payload.summary.max_magnetic_complexity

        if prob >= 0.85 and mag > 0.7:
            severity = "extreme"
        elif prob >= 0.65 or mag > 0.6:
            severity = "high"
        elif prob >= 0.45:
            severity = "moderate"
        else:
            severity = "low"

        return {
            "severity": severity,
            "analyst_summary": (
                f"Surya predicts a {cls}-class flare event with {prob:.0%} probability. "
                f"Magnetic complexity index: {mag:.2f}. "
                f"Infrastructure protection protocols have been activated based on threat thresholds."
            ),
            "activate_commsops": prob > settings.THRESHOLD_COMMSOPS,
            "activate_satops": prob > settings.THRESHOLD_SATOPS,
            "activate_gridops": prob > settings.THRESHOLD_GRIDOPS and mag > 0.5,
            "atmospheric_drag_risk": payload.summary.mean_euv_flux,
            "euv_impact": payload.summary.mean_euv_flux,
            "solar_wind_speed_proxy": 0.0,
        }