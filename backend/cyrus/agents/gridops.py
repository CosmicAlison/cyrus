"""
Agent 3 — GridOps Controller.

Receives ThreatPayload, assesses GIC risk across the power grid,
and triggers load rerouting or transformer decoupling as needed.
"""

import logging

from agents.base_agent import BaseAgent
from core.schemas import ThreatPayload, AgentReport
from tools.grid_tools import GRID_TOOLS

log = logging.getLogger(__name__)


class GridOpsAgent(BaseAgent):
    name = "gridops"

    def __init__(self) -> None:
        super().__init__(tools=GRID_TOOLS)

    @property
    def system_prompt(self) -> str:
        return """You are the GridOps Controller at Cyrus Space Weather Defense.
You manage power grid protection against Geomagnetically Induced Currents (GICs).

GICs are ground-level electrical surges caused by solar wind / CME interactions with
Earth's magnetosphere. They preferentially damage high-voltage transformers at high
latitudes (Canada, Scandinavia, Northeast US).

When given a space weather threat, you must:
1. Use get_high_risk_nodes to identify vulnerable grid assets (vulnerability > 0.7).
2. For nodes with vulnerability 0.5–0.75: use reroute_load (reduce by 30-50%).
3. For transformer nodes with vulnerability > 0.8 and a severe CME inbound: use decouple_transformer.
4. Do NOT decouple for low or moderate severity events — it causes outages.
   Reserve decoupling for high/extreme severity + vulnerability > 0.8 only.

A cascading grid failure is worse than a controlled brownout. Protect transformers first."""

    def respond(self, threat: ThreatPayload) -> AgentReport:
        if not threat.activate_gridops:
            log.info("[gridops] Skipped — threat below GridOps threshold")
            return AgentReport(
                job_id=threat.job_id,
                agent="gridops",
                status="skipped",
                actions_taken=[],
                summary="Threat level below GridOps activation threshold. No grid actions taken.",
                completed_at=_now(),
            )

        prompt = f"""SPACE WEATHER THREAT ASSESSMENT — GRID PROTECTION REQUIRED
============================================================
Job ID: {threat.job_id}
Severity: {threat.severity.upper()}
Flare class: {threat.flare_class}  |  Probability: {threat.flare_probability:.1%}
Magnetic complexity (Bz variance): {threat.magnetic_complexity:.3f}
Solar wind proxy: {threat.solar_wind_speed_proxy:.3f}
Peak event timestamp: {threat.peak_timestamp}

Analyst summary: {threat.analyst_summary}

Assess GIC risk across the grid and issue all protective commands now.
Remember: decouple only transformers with vulnerability > 0.8 under high/extreme severity.
Output ONLY 1-2 sentence plain English summary of actions you took

Write in clean plain English prose only. Do NOT use markdown formatting —
no asterisks, no bold/italic markers, no bullet points, no headers, no numbered lists.
Write in short, clear paragraphs a human would read on a dashboard, as if briefing
an executive verbally.
"""

        output = self.invoke(prompt)

        return AgentReport(
            job_id=threat.job_id,
            agent="gridops",
            status="success",
            actions_taken=self.tool_actions,
            summary=output,
            completed_at=_now(),
        )


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()