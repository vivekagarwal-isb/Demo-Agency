"""
Supervisor Agent — Controls LangGraph routing and maintains shared state.
Determines which agent runs next based on what has already been completed.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict

from agents.state import AgentState

log = logging.getLogger(__name__)

try:
    from config import AGENT_SEQUENCE
except ImportError:
    AGENT_SEQUENCE = [
        "data_agent", "feature_agent", "forecasting_agent",
        "outlier_agent", "opportunity_agent", "causal_agent",
        "digital_twin_agent", "shock_agent", "memory_agent",
        "explanation_agent", "reporting_agent", "END",
    ]


def supervisor(state: AgentState) -> AgentState:
    """
    Supervisor node: determines next agent to execute.
    Follows the linear AGENT_SEQUENCE, skipping completed agents.
    """
    completed = state.get("completed_agents", [])

    # Initialise run metadata on first call
    if not state.get("run_id"):
        state["run_id"] = str(uuid.uuid4())[:8]
        state["timestamp"] = datetime.utcnow().isoformat()
        state["errors"] = []
        state["completed_agents"] = []
        log.info(f"[Supervisor] New run: {state['run_id']} @ {state['timestamp']}")

    # Walk the sequence and pick the first not-yet-completed step
    for agent_name in AGENT_SEQUENCE:
        if agent_name == "END":
            log.info("[Supervisor] All agents completed → END")
            state["next_agent"] = "END"
            return state
        if agent_name not in completed:
            log.info(f"[Supervisor] Routing to → {agent_name}")
            state["next_agent"] = agent_name
            return state

    state["next_agent"] = "END"
    return state


def route(state: AgentState) -> str:
    """Conditional edge: returns the string key for the next node."""
    return state.get("next_agent", "END")
