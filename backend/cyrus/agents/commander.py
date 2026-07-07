"""
Agent 5 — Executive Commander.

Collects AgentReports from SatOps, GridOps, and CommsOps,
synthesises them into a comprehensive executive brief, and
produces the final ExecutiveBrief for the dashboard.
"""

import json
import logging

from agents.base_agent import BaseAgent
from core.schemas import AgentReport, ExecutiveBrief, ThreatPayload

log = logging.getLogger(__name__)


class CommanderAgent(BaseAgent):
    name = "commander"

    def __init__(self) -> None:
        super().__init__(tools=None)

    @property
    def system_prompt(self) -> str:
        return """You are the Executive Commander of Cyrus Space Weather Defense.

You receive a threat assessment and the completed action reports from three
operational teams (SatOps, GridOps, CommsOps) and produce a final executive brief.

Your brief must:
1. Open with a one-sentence threat classification (flare class, severity, probability).
2. Summarise what each team did in 1-2 sentences each.
3. State the overall outcome: was the 2-hour window used effectively?
4. Note any gaps: skipped agents, partial actions, or residual risks.
5. Close with a recommended human review priority (IMMEDIATE / MONITOR / ROUTINE).

Tone: authoritative, concise, factual. No jargon. Written for a senior executive,
not a scientist. Total length: 150-250 words.

Respond with ONLY a JSON object:
{
  "executive_brief": "<full brief text>",
  "recommended_priority": "IMMEDIATE|MONITOR|ROUTINE"
}"""

    def synthesise(
        self,
        threat: ThreatPayload,
        reports: dict[str, AgentReport],
    ) -> ExecutiveBrief:
        satops = reports.get("satops")
        gridops = reports.get("gridops")
        commsops = reports.get("commsops")

        total_actions = sum(
            len(r.actions_taken) for r in [satops, gridops, commsops] if r
        )

        prompt = f"""CYRUS MISSION COMPLETE — EXECUTIVE SYNTHESIS REQUIRED
        ======================================================
        Job ID: {threat.job_id}
        Severity: {threat.severity.upper()}  |  Flare class: {threat.flare_class}
        Flare probability: {threat.flare_probability:.1%}
        Peak event: {threat.peak_timestamp}

        ANALYST ASSESSMENT:
        {threat.analyst_summary}

        SATOPS REPORT [{satops.status if satops else 'N/A'}]:
        {satops.summary if satops else 'No report received.'}
        Actions taken: {len(satops.actions_taken) if satops else 0}

        GRIDOPS REPORT [{gridops.status if gridops else 'N/A'}]:
        {gridops.summary if gridops else 'No report received.'}
        Actions taken: {len(gridops.actions_taken) if gridops else 0}

        COMMSOPS REPORT [{commsops.status if commsops else 'N/A'}]:
        {commsops.summary if commsops else 'No report received.'}
        Actions taken: {len(commsops.actions_taken) if commsops else 0}

        TOTAL ACTIONS ACROSS ALL TEAMS: {total_actions}

        Produce the executive brief now."""

        raw = self.invoke(prompt)

        try:
            output = json.loads(raw.strip())
            brief_text = output["executive_brief"]
        except (json.JSONDecodeError, KeyError):
            log.warning("[commander] Non-JSON response, using raw output")
            brief_text = raw[:1000]

        log.info(
            "[commander] Brief produced for job %s — %d total actions",
            threat.job_id,
            total_actions,
        )

        return ExecutiveBrief(
            job_id=threat.job_id,
            severity=threat.severity,
            flare_class=threat.flare_class,
            flare_probability=threat.flare_probability,
            satops_summary=satops.summary if satops else "No report",
            gridops_summary=gridops.summary if gridops else "No report",
            commsops_summary=commsops.summary if commsops else "No report",
            executive_brief=brief_text,
            total_actions=total_actions,
            completed_at=_now(),
        )


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()