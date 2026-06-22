"""
graph/nodes.py
Each Cyrus agent wrapped as a LangGraph node function.

Node functions receive the full CyrusState dict and return
a partial dict of keys they've updated.
"""

import logging
from typing import Any

from cyrus.agents.helio_analyst import HelioAnalyst
from cyrus.agents.satops import SatOpsAgent
from cyrus.agents.gridops import GridOpsAgent
from cyrus.agents.commsops import CommsOpsAgent
from cyrus.agents.commander import CommanderAgent
from cyrus.core.schemas import ThreatPayload, AgentReport

log = logging.getLogger(__name__)


_helio_analyst: HelioAnalyst | None = None
_satops_agent: SatOpsAgent | None = None
_gridops_agent: GridOpsAgent | None = None
_commsops_agent: CommsOpsAgent | None = None
_commander_agent: CommanderAgent | None = None


def get_helio_analyst() -> HelioAnalyst:
    global _helio_analyst
    if _helio_analyst is None:
        _helio_analyst = HelioAnalyst()
    return _helio_analyst


def get_satops() -> SatOpsAgent:
    global _satops_agent
    if _satops_agent is None:
        _satops_agent = SatOpsAgent()
    return _satops_agent


def get_gridops() -> GridOpsAgent:
    global _gridops_agent
    if _gridops_agent is None:
        _gridops_agent = GridOpsAgent()
    return _gridops_agent


def get_commsops() -> CommsOpsAgent:
    global _commsops_agent
    if _commsops_agent is None:
        _commsops_agent = CommsOpsAgent()
    return _commsops_agent


def get_commander() -> CommanderAgent:
    global _commander_agent
    if _commander_agent is None:
        _commander_agent = CommanderAgent()
    return _commander_agent


# ── Node functions ────────────────────────────────────────────────────────────

def helio_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Node 1: Interpret raw Surya forecast → ThreatPayload.
    Reads:  state["raw_forecast"], state["threat_event_id"]
    Writes: threat_payload, severity, activate_* flags
    """
    log.info("[graph] helio_analyst_node running")

    try:
        threat = get_helio_analyst().analyse(
            raw_payload=state["raw_forecast"],
            threat_event_id=state.get("threat_event_id", 0),
        )
        return {
            "threat_payload": threat.model_dump(),
            "severity": threat.severity,
            "activate_satops": threat.activate_satops,
            "activate_gridops": threat.activate_gridops,
            "activate_commsops": threat.activate_commsops,
            "error": None,
        }
    except Exception as exc:
        log.error("[graph] helio_analyst_node failed: %s", exc)
        return {"error": str(exc)}


def satops_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Node 2: SatOps — satellite safe mode and orbit commands.
    Reads:  state["threat_payload"], state["activate_satops"]
    Writes: satops_report
    """
    log.info("[graph] satops_node running")
    try:
        threat = ThreatPayload.model_validate(state["threat_payload"])
        report = get_satops().respond(threat)
        return {"satops_report": report.model_dump()}
    except Exception as exc:
        log.error("[graph] satops_node failed: %s", exc)
        return {
            "satops_report": _error_report("satops", state.get("threat_payload", {}).get("job_id", ""), str(exc))
        }


def gridops_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Node 3: GridOps — GIC load rerouting and transformer protection.
    Reads:  state["threat_payload"], state["activate_gridops"]
    Writes: gridops_report
    """
    log.info("[graph] gridops_node running")
    try:
        threat = ThreatPayload.model_validate(state["threat_payload"])
        report = get_gridops().respond(threat)
        return {"gridops_report": report.model_dump()}
    except Exception as exc:
        log.error("[graph] gridops_node failed: %s", exc)
        return {
            "gridops_report": _error_report("gridops", state.get("threat_payload", {}).get("job_id", ""), str(exc))
        }


def commsops_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Node 4: CommsOps — aviation HF advisories and backup comms.
    Reads:  state["threat_payload"], state["activate_commsops"]
    Writes: commsops_report
    """
    log.info("[graph] commsops_node running")
    try:
        threat = ThreatPayload.model_validate(state["threat_payload"])
        report = get_commsops().respond(threat)
        return {"commsops_report": report.model_dump()}
    except Exception as exc:
        log.error("[graph] commsops_node failed: %s", exc)
        return {
            "commsops_report": _error_report("commsops", state.get("threat_payload", {}).get("job_id", ""), str(exc))
        }


def commander_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Node 5: Commander — synthesise all reports → executive brief.
    Reads:  threat_payload, satops_report, gridops_report, commsops_report
    Writes: executive_brief
    """
    log.info("[graph] commander_node running")
    try:
        threat = ThreatPayload.model_validate(state["threat_payload"])

        reports: dict[str, AgentReport] = {}
        for agent in ["satops", "gridops", "commsops"]:
            key = f"{agent}_report"
            if state.get(key):
                reports[agent] = AgentReport.model_validate(state[key])

        brief = get_commander().synthesise(threat, reports)
        return {"executive_brief": brief.model_dump()}
    except Exception as exc:
        log.error("[graph] commander_node failed: %s", exc)
        return {"error": str(exc)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_report(agent: str, job_id: str, error: str) -> dict[str, Any]:
    from datetime import datetime, timezone
    return {
        "job_id": job_id,
        "agent": agent,
        "status": "error",
        "actions_taken": [],
        "summary": f"Agent failed with error: {error}",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }