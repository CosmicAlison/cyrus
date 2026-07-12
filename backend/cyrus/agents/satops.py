"""
agents/satops.py
Agent 2 — SatOps Engineer.

Receives ThreatPayload, queries the satellite registry,
and issues safe mode / orientation / thruster commands as needed.
"""

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.schemas import ThreatPayload, AgentReport
from tools.satellite_tools import SATELLITE_TOOLS

log = logging.getLogger(__name__)


class SatOpsAgent(BaseAgent):
    name = "satops"

    def __init__(self) -> None:
        super().__init__(tools=SATELLITE_TOOLS)

    @property
    def system_prompt(self) -> str:
        return """You are the SatOps Engineer at Cyrus Space Weather Defense.
You manage a registry of Low Earth Orbit (LEO), MEO, and GEO satellites.

When given a space weather threat assessment, you must:
1. Query the satellite registry to understand which assets are at risk.
2. Prioritise LEO satellites — they face the greatest atmospheric drag risk from EUV heating.
3. Issue appropriate commands using your tools:
   - issue_safe_mode_command: for all LEO satellites when flare_probability > 0.45
   - adjust_orientation: to minimise drag profile (drag_minimise attitude)
   - schedule_thruster_burn: to raise orbit if atmospheric_drag_risk > 0.7
4. MEO/GEO satellites: issue safe mode if flare_class is M or X.

After taking all actions, respond with ONLY 1-2 sentence plain English summary of actions taken

Be decisive. During a space weather event, speed of response is critical."""

    def respond(self, threat: ThreatPayload) -> AgentReport:
        if not threat.activate_satops:
            log.info("[satops] Skipped — threat below SatOps threshold")
            return AgentReport(
                job_id=threat.job_id,
                agent="satops",
                status="skipped",
                actions_taken=[],
                summary="Threat level below SatOps activation threshold. No satellite commands issued.",
                completed_at=_now(),
            )

        prompt = f"""SPACE WEATHER THREAT ASSESSMENT — ACTION REQUIRED
==================================================
Job ID: {threat.job_id}
Severity: {threat.severity.upper()}
Flare class: {threat.flare_class}  |  Probability: {threat.flare_probability:.1%}
Atmospheric drag risk (EUV): {threat.atmospheric_drag_risk:.2f}
Peak event timestamp: {threat.peak_timestamp}

Analyst summary: {threat.analyst_summary}

Query the satellite registry and issue all appropriate protective commands now."""

        raw = self.invoke(prompt)


        return AgentReport(
            job_id=threat.job_id,
            agent="satops",
            status="success",
            actions_taken=self.tool_actions,
            summary=raw,
            completed_at=_now(),
        )


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()