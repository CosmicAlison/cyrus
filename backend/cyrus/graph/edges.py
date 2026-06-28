"""
Conditional routing functions for the Cyrus LangGraph pipeline.

After helio_analyst runs, the router fans out to whichever ops agents
are activated based on the threat severity thresholds.
The commander node always runs last once all active agents complete.
"""

from typing import Any, Literal


def route_after_analyst(
    state: dict[str, Any],
) -> list[Literal["satops", "gridops", "commsops", "commander"]]:
    """
    Fan-out routing after helio_analyst completes.
    Returns a list of nodes to activate in parallel.
    If no agents are activated (very low severity), goes straight to commander.
    """
    if state.get("error"):
        return ["commander"]  # Commander will note the failure

    targets = []
    if state.get("activate_satops"):
        targets.append("satops")
    if state.get("activate_gridops"):
        targets.append("gridops")
    if state.get("activate_commsops"):
        targets.append("commsops")

    # Always produce an executive brief, even if all agents were skipped
    if not targets:
        targets.append("commander")

    return targets


def route_to_commander(
    state: dict[str, Any],
) -> Literal["commander"]:
    """
    All ops agents converge here — always proceed to commander.
    Used as the edge from each ops node.
    """
    return "commander"