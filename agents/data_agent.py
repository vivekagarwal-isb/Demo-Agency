"""
Data Agent — Ingestion, validation, and loading of synthetic datasets.
Checks data quality: completeness, schema, distributions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from agents.state import AgentState

log = logging.getLogger(__name__)

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = Path("data")

REQUIRED_POLICY_COLS = [
    "policy_id", "agent_id", "state", "district", "producer_id",
    "policy_type", "issue_date", "status", "gwp", "premium_annual",
    "is_new_business",
]

REQUIRED_AGENT_MONTH_COLS = [
    "agent_id", "year", "month", "year_month", "gwp", "pif",
    "new_business_count", "renewal_count", "cancellation_count",
    "renewal_rate", "cancellation_rate",
]


def _validate_schema(df: pd.DataFrame, required_cols: list, name: str) -> list[str]:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return [f"[DataAgent] {name} missing columns: {missing}"]
    return []


def _validate_no_nulls(df: pd.DataFrame, critical_cols: list, name: str) -> list[str]:
    errors = []
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                errors.append(f"[DataAgent] {name}.{col} has {null_count} nulls")
    return errors


def _compute_data_quality(
    policies_df: pd.DataFrame,
    agent_month_df: pd.DataFrame,
) -> Dict:
    quality = {}

    # Policies
    quality["policies_rows"]         = len(policies_df)
    quality["policies_completeness"]  = 1 - policies_df.isna().mean().mean()
    quality["gwp_negative"]           = int((policies_df["gwp"] < 0).sum())
    quality["gwp_zero"]               = int((policies_df["gwp"] == 0).sum())
    quality["premium_median"]         = float(policies_df["gwp"].median())
    quality["anomaly_rate"]           = float(policies_df["is_anomaly"].mean())
    quality["status_distribution"]    = policies_df["status"].value_counts().to_dict()
    quality["policy_types"]           = policies_df["policy_type"].value_counts().to_dict()
    quality["date_range"]             = {
        "min": str(policies_df["issue_date"].min()),
        "max": str(policies_df["issue_date"].max()),
    }

    # Agent-month
    quality["agent_month_rows"]       = len(agent_month_df)
    quality["n_unique_agents"]        = int(agent_month_df["agent_id"].nunique())
    quality["n_months"]               = int(agent_month_df["year_month"].nunique())
    quality["gwp_total"]              = float(agent_month_df["gwp"].sum())
    quality["pif_total"]              = float(agent_month_df["pif"].sum())
    quality["avg_renewal_rate"]       = float(agent_month_df["renewal_rate"].mean())
    quality["avg_cancellation_rate"]  = float(agent_month_df["cancellation_rate"].mean())

    return quality


def data_agent(state: AgentState) -> AgentState:
    """Load and validate all data assets."""
    log.info("[DataAgent] Starting data ingestion …")
    errors = list(state.get("errors", []))

    try:
        # ── Load data files ──────────────────────────────────────────────────
        agents_path       = DATA_DIR / "agents.parquet"
        policies_path     = DATA_DIR / "policies.parquet"
        agent_month_path  = DATA_DIR / "agent_month.parquet"
        state_month_path  = DATA_DIR / "state_month.parquet"

        # Auto-generate if missing
        for p in [agents_path, policies_path, agent_month_path, state_month_path]:
            if not p.exists():
                log.warning(f"[DataAgent] {p} not found — generating synthetic data …")
                import generate_data
                generate_data.main()
                break

        agents_df      = pd.read_parquet(agents_path)
        policies_df    = pd.read_parquet(policies_path)
        agent_month_df = pd.read_parquet(agent_month_path)
        state_month_df = pd.read_parquet(state_month_path)

        # Ensure datetime columns
        for col in ["issue_date", "effective_date", "expiration_date"]:
            if col in policies_df.columns:
                policies_df[col] = pd.to_datetime(policies_df[col])
        if "year_month_dt" in agent_month_df.columns:
            agent_month_df["year_month_dt"] = pd.to_datetime(agent_month_df["year_month_dt"])
        if "year_month_dt" in state_month_df.columns:
            state_month_df["year_month_dt"] = pd.to_datetime(state_month_df["year_month_dt"])

        log.info(f"[DataAgent] Loaded: policies={len(policies_df):,}  "
                 f"agent_month={len(agent_month_df):,}  "
                 f"state_month={len(state_month_df):,}")

        # ── Schema validation ────────────────────────────────────────────────
        errors += _validate_schema(policies_df,    REQUIRED_POLICY_COLS,      "policies")
        errors += _validate_schema(agent_month_df, REQUIRED_AGENT_MONTH_COLS, "agent_month")

        # ── Null validation ──────────────────────────────────────────────────
        errors += _validate_no_nulls(policies_df, ["gwp", "agent_id", "status"], "policies")

        # ── Business rule checks ─────────────────────────────────────────────
        neg_gwp = (agent_month_df["gwp"] < 0).sum()
        if neg_gwp > 0:
            errors.append(f"[DataAgent] {neg_gwp} rows have negative GWP in agent_month")

        # ── Quality metrics ──────────────────────────────────────────────────
        quality = _compute_data_quality(policies_df, agent_month_df)

        log.info(f"[DataAgent] Quality: {quality['policies_completeness']:.1%} complete, "
                 f"GWP total=${quality['gwp_total']:,.0f}, "
                 f"anomaly rate={quality['anomaly_rate']:.1%}")

        state["agents_df"]      = agents_df
        state["policies_df"]    = policies_df
        state["agent_month_df"] = agent_month_df
        state["state_month_df"] = state_month_df
        state["data_quality"]   = quality

    except Exception as e:
        err_msg = f"[DataAgent] FATAL: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("data_agent")
    state["completed_agents"] = completed
    log.info("[DataAgent] Done.")
    return state
