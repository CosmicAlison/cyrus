"""
Agent 4 — CommsOps Specialist.

Receives ThreatPayload, assesses HF radio blackout and radiation risk,
and issues rerouting advisories / backup comms switches for at-risk flights.
"""

import logging

from agents.base_agent import BaseAgent
from core.schemas import ThreatPayload, AgentReport
from tools.comms_tools import COMMS_TOOLS

log = logging.getLogger(__name__)


class CommsOpsAgent(BaseAgent):
    name = "commsops"

    def __init__(self) -> None:
        super().__init__(tools=COMMS_TOOLS)

    @property
    def system_prompt(self) -> str:
        return """You are the CommsOps Specialist at Cyrus Space Weather Defense.
You protect aviation and signals infrastructure from solar flare impacts.

Solar flares cause two aviation hazards:
1. HF radio blackouts — X-ray flux ionises the D-layer, absorbing HF signals.
   Affects transpolar and oceanic routes that rely on HF as primary comms.
2. Radiation exposure — high-energy particles at polar latitudes exceed safe limits.
   Requires transpolar route avoidance.

When given a space weather threat, you must:
1. Use get_high_risk_routes('high') to find high-HF-dependency routes at immediate risk.
2. For C-class (prob > 0.25): issue_rerouting_advisory for all polar high-HF routes.
3. For M/X-class: additionally use switch_to_backup_band (SATCOM_L or SATCOM_KU)
   for all high-HF routes AND issue_rerouting_advisory for medium-HF polar routes.
4. Non-polar routes with low HF dependency: no action needed unless X-class.

Be specific about which routes received advisories and which switched to backup comms."""

    def respond(self, threat: ThreatPayload) -> AgentReport:
        if not threat.activate_commsops:
            log.info("[commsops] Skipped — threat below CommsOps threshold")
            return AgentReport(
                job_id=threat.job_id,
                agent="commsops",
                status="skipped",
                actions_taken=[],
                summary="Threat level below CommsOps activation threshold. No advisories issued.",
                completed_at=_now(),
            )

        prompt = f"""SPACE WEATHER THREAT ASSESSMENT — COMMS PROTECTION REQUIRED
=============================================================
Job ID: {threat.job_id}
Severity: {threat.severity.upper()}
Flare class: {threat.flare_class}  |  Probability: {threat.flare_probability:.1%}
EUV impact (ionospheric): {threat.euv_impact:.3f}
Peak event timestamp: {threat.peak_timestamp}

Analyst summary: {threat.analyst_summary}

Assess HF blackout risk for active flight routes and issue all advisories now.
For M/X-class events, also switch high-dependency routes to backup satellite comms.
Output ONLY 2-3 sentence plain English assessment and summary
"""

        raw = self.invoke(prompt)

        return AgentReport(
            job_id=threat.job_id,
            agent="commsops",
            status="success",
            actions_taken=self.tool_actions,
            summary=raw[:500],
            completed_at=_now(),
        )


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()