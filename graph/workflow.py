"""
LangGraph Workflow — Orchestrates all 12 agents in the Farmers EA system.

Graph topology:
  START → supervisor → [agent_nodes] → supervisor (loop) → END

The supervisor determines the next agent using the completed_agents list.
Each agent node updates state and marks itself complete.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.state import AgentState
from agents.supervisor import supervisor, route

# Import all agent nodes
from agents.data_agent          import data_agent
from agents.feature_agent       import feature_agent
from agents.forecasting_agent   import forecasting_agent
from agents.outlier_agent       import outlier_agent
from agents.opportunity_agent   import opportunity_agent
from agents.causal_agent        import causal_agent
from agents.digital_twin_agent  import digital_twin_agent
from agents.shock_agent         import shock_agent
from agents.memory_agent        import memory_agent
from agents.explanation_agent   import explanation_agent
from agents.reporting_agent     import reporting_agent

log = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END, START
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    log.warning("LangGraph not installed — using sequential fallback runner.")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

AGENT_NODES = {
    "data_agent":         data_agent,
    "feature_agent":      feature_agent,
    "forecasting_agent":  forecasting_agent,
    "outlier_agent":      outlier_agent,
    "opportunity_agent":  opportunity_agent,
    "causal_agent":       causal_agent,
    "digital_twin_agent": digital_twin_agent,
    "shock_agent":        shock_agent,
    "memory_agent":       memory_agent,
    "explanation_agent":  explanation_agent,
    "reporting_agent":    reporting_agent,
}


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH WORKFLOW BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Topology:
      START → supervisor
      supervisor → [each agent] (conditional edge via route())
      [each agent] → supervisor
      supervisor → END (when next_agent == 'END')
    """
    if not LANGGRAPH_AVAILABLE:
        log.info("LangGraph unavailable — returning sequential runner.")
        return None

    builder = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor)
    for name, fn in AGENT_NODES.items():
        builder.add_node(name, fn)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("supervisor")

    # ── Conditional edges from supervisor ────────────────────────────────────
    routing_map = {name: name for name in AGENT_NODES}
    routing_map["END"] = END

    builder.add_conditional_edges(
        "supervisor",
        route,
        routing_map,
    )

    # ── Return edge: every agent routes back to supervisor ───────────────────
    for name in AGENT_NODES:
        builder.add_edge(name, "supervisor")

    graph = builder.compile()
    log.info(f"LangGraph compiled: {len(AGENT_NODES) + 1} nodes")
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL FALLBACK RUNNER (when LangGraph not installed)
# ─────────────────────────────────────────────────────────────────────────────

def sequential_run(initial_state: AgentState) -> AgentState:
    """Run agents sequentially without LangGraph dependency."""
    state = dict(initial_state)

    # Bootstrap run metadata
    import uuid
    from datetime import datetime
    state.setdefault("run_id", str(uuid.uuid4())[:8])
    state.setdefault("timestamp", datetime.utcnow().isoformat())
    state.setdefault("errors", [])
    state.setdefault("completed_agents", [])

    ordered = [
        "supervisor",
        *AGENT_NODES.keys(),
    ]

    log.info(f"[SequentialRunner] Starting run {state['run_id']} with {len(AGENT_NODES)} agents")

    for agent_name in AGENT_NODES:
        log.info(f"[SequentialRunner] Running → {agent_name}")
        try:
            fn = AGENT_NODES[agent_name]
            state = fn(state)
        except Exception as e:
            log.error(f"[SequentialRunner] Agent {agent_name} failed: {e}", exc_info=True)
            state["errors"].append(f"{agent_name}: {e}")

    state["next_agent"] = "END"
    log.info(f"[SequentialRunner] All agents done. Errors: {len(state['errors'])}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

_compiled_graph = None


def get_graph():
    """Singleton compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(
    shock_params: dict | None = None,
    forecast_horizon: int = 12,
) -> AgentState:
    """
    Public entry point: run the full multi-agent pipeline.

    Args:
        shock_params: dict with keys: conversion_delta, retention_delta,
                      pricing_delta, producer_count_delta, discount_delta
        forecast_horizon: months to forecast ahead

    Returns:
        Final AgentState with all analysis results.
    """
    try:
        from config import SHOCK_DEFAULTS
    except ImportError:
        SHOCK_DEFAULTS = {
            "conversion_delta": 0.0, "retention_delta": 0.0,
            "pricing_delta": 0.0, "producer_count_delta": 0.0,
            "discount_delta": 0.0,
        }

    initial: AgentState = {
        "shock_params":      shock_params or SHOCK_DEFAULTS,
        "forecast_horizon":  forecast_horizon,
        "completed_agents":  [],
        "errors":            [],
        "next_agent":        "data_agent",
        "run_id":            "",
        "timestamp":         "",
    }

    graph = get_graph()

    if graph is not None:
        log.info("[Pipeline] Running via LangGraph …")
        final_state = graph.invoke(initial)
    else:
        log.info("[Pipeline] Running via sequential fallback …")
        final_state = sequential_run(initial)

    log.info(
        f"[Pipeline] Complete. "
        f"Agents run: {len(final_state.get('completed_agents', []))}. "
        f"Errors: {len(final_state.get('errors', []))}."
    )
    return final_state
