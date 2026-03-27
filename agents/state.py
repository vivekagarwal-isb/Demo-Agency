"""
Shared LangGraph State for the Farmers EA Multi-Agent System.
All agents read from and write to this TypedDict.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd


class AgentState(TypedDict, total=False):
    # ── Data ────────────────────────────────────────────────────────────────
    agents_df:          Optional[Any]   # pd.DataFrame: agent metadata
    policies_df:        Optional[Any]   # pd.DataFrame: policy records (150K rows)
    agent_month_df:     Optional[Any]   # pd.DataFrame: agent×month KPIs
    state_month_df:     Optional[Any]   # pd.DataFrame: state×month KPIs
    data_quality:       Optional[Dict]  # validation results

    # ── Features ────────────────────────────────────────────────────────────
    features_df:        Optional[Any]   # enriched feature matrix
    feature_importance: Optional[Dict]  # top features by importance

    # ── Forecasting ─────────────────────────────────────────────────────────
    forecasts:          Optional[Dict]  # {metric: pd.DataFrame with forecast}
    forecast_accuracy:  Optional[Dict]  # MAPE / RMSE per metric
    forecast_horizon:   int             # months ahead

    # ── Outlier Detection ───────────────────────────────────────────────────
    outliers_df:        Optional[Any]   # flagged outlier rows
    outlier_summary:    Optional[Dict]  # counts and descriptions

    # ── Opportunity Ranking ─────────────────────────────────────────────────
    opportunities_df:   Optional[Any]   # ranked opportunity table
    opportunity_summary: Optional[Dict]

    # ── Causal Inference ────────────────────────────────────────────────────
    causal_graph:       Optional[Dict]  # DAG adjacency representation
    ate_results:        Optional[Dict]  # Average Treatment Effects
    cate_results:       Optional[Dict]  # Conditional Treatment Effects
    causal_summary:     Optional[str]   # Human-readable causal insights

    # ── Digital Twin ────────────────────────────────────────────────────────
    twin_state:         Optional[Dict]  # current twin entity states
    twin_simulation:    Optional[Any]   # pd.DataFrame: simulated trajectory
    twin_summary:       Optional[str]

    # ── Shock Simulation ────────────────────────────────────────────────────
    shock_params:       Optional[Dict]  # slider values
    shock_results:      Optional[Dict]  # {scenario: impact_df}
    shock_summary:      Optional[str]

    # ── Memory & Learning ───────────────────────────────────────────────────
    memory_retrieved:   Optional[List[Dict]]  # relevant past memories
    learning_update:    Optional[Dict]        # weight/rule adjustments
    prediction_log:     Optional[List[Dict]]  # stored for future feedback

    # ── Explanations ────────────────────────────────────────────────────────
    shap_values:        Optional[Dict]  # {agent_id: shap_array}
    explanations:       Optional[List[str]]   # human-readable SHAP insights

    # ── Report ──────────────────────────────────────────────────────────────
    report:             Optional[str]   # final markdown report
    insights:           Optional[List[str]]   # bullet-point AI insights

    # ── Routing ─────────────────────────────────────────────────────────────
    next_agent:         str             # supervisor routing target
    completed_agents:   List[str]       # track executed agents
    errors:             List[str]       # accumulated errors
    run_id:             str             # unique run identifier
    timestamp:          str             # ISO timestamp
