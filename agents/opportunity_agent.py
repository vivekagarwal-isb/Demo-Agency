"""
Opportunity Agent — Ranks agents/districts/states by growth potential.

Scoring dimensions:
1. Opportunity Gap (vs peer average)
2. Momentum (recent growth trajectory)
3. Untapped producer capacity
4. Conversion improvement headroom
5. Retention leakage (lost GWP from cancellations)
6. Market share headroom

Output: ranked opportunity table with estimated $ upside.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from agents.state import AgentState

log = logging.getLogger(__name__)


def _score_opportunity_gap(df: pd.DataFrame) -> pd.Series:
    """Normalised opportunity gap vs district peers."""
    gap = df.get("opportunity_gap_pct", pd.Series(0, index=df.index))
    return gap.clip(0).rank(pct=True)


def _score_momentum(df: pd.DataFrame) -> pd.Series:
    """Agents with positive momentum are already growing — boost further."""
    mom = df.get("momentum", pd.Series(1, index=df.index))
    return mom.rank(pct=True)


def _score_conversion_headroom(df: pd.DataFrame) -> pd.Series:
    """
    Lower conversion → more headroom. Ranked so that low converters score higher.
    """
    conv = df.get("conversion_rate", pd.Series(0.3, index=df.index))
    # Invert: low conversion = high opportunity
    return (1 - conv).clip(0).rank(pct=True)


def _score_retention_leakage(df: pd.DataFrame) -> pd.Series:
    """High cancellation rate = leakage = opportunity to plug."""
    can = df.get("cancellation_rate", pd.Series(0.05, index=df.index))
    return can.clip(0).rank(pct=True)


def _score_producer_capacity(df: pd.DataFrame) -> pd.Series:
    """Agents with fewer NB per producer have room to grow producer productivity."""
    nb_pp = df.get("nb_per_producer", pd.Series(1, index=df.index))
    return (1 / (nb_pp + 1)).rank(pct=True)


def _estimate_gwp_upside(df: pd.DataFrame, scores: pd.DataFrame) -> pd.Series:
    """
    Upside GWP estimate:
    avg_premium × policy_count × composite_score × improvement_factor
    """
    avg_prem  = df.get("avg_premium", pd.Series(1400, index=df.index))
    pol_count = df.get("policy_count", df.get("pif", pd.Series(100, index=df.index)))
    composite = scores["composite_score"]
    # Assume 10% improvement in policy count per unit of composite opportunity
    upside = avg_prem * pol_count * composite * 0.10
    return upside.clip(lower=0)


def _classify_opportunity(score: float) -> str:
    if score >= 0.85:   return "★★★ High Priority"
    elif score >= 0.65: return "★★  Medium Priority"
    elif score >= 0.40: return "★   Low Priority"
    else:               return "    Maintain"


def opportunity_agent(state: AgentState) -> AgentState:
    log.info("[OpportunityAgent] Ranking opportunities …")
    errors = list(state.get("errors", []))

    try:
        df = (state.get("features_df") or state.get("agent_month_df")).copy()

        if "year_month_dt" not in df.columns:
            df["year_month_dt"] = pd.to_datetime(df["year_month"])

        # Use the most recent month for each agent
        latest = df.sort_values("year_month_dt").groupby("agent_id").last().reset_index()

        # ── Individual dimension scores ──────────────────────────────────────
        scores = pd.DataFrame(index=latest.index)
        scores["gap_score"]        = _score_opportunity_gap(latest).values
        scores["momentum_score"]   = _score_momentum(latest).values
        scores["conv_score"]       = _score_conversion_headroom(latest).values
        scores["leakage_score"]    = _score_retention_leakage(latest).values
        scores["capacity_score"]   = _score_producer_capacity(latest).values

        # ── Composite weighted score ─────────────────────────────────────────
        weights = {
            "gap_score":      0.30,
            "leakage_score":  0.25,
            "conv_score":     0.20,
            "momentum_score": 0.15,
            "capacity_score": 0.10,
        }
        scores["composite_score"] = sum(
            scores[k] * w for k, w in weights.items()
        )

        # ── Upside estimation ────────────────────────────────────────────────
        scores["gwp_upside_est"] = _estimate_gwp_upside(latest, scores).values

        # ── Build output table ───────────────────────────────────────────────
        opp_df = latest[[
            "agent_id", "agent_name", "state", "district",
            "gwp", "pif", "renewal_rate", "cancellation_rate",
            "conversion_rate", "n_producers",
        ]].copy()

        for col in scores.columns:
            opp_df[col] = scores[col].values

        opp_df["opportunity_class"] = opp_df["composite_score"].apply(_classify_opportunity)
        opp_df["rank"] = opp_df["composite_score"].rank(ascending=False, method="min").astype(int)
        opp_df = opp_df.sort_values("rank")

        # ── District & State roll-ups ─────────────────────────────────────────
        district_opp = (
            opp_df.groupby(["state", "district"])
            .agg(
                agent_count=("agent_id", "count"),
                total_gwp=("gwp", "sum"),
                avg_composite=("composite_score", "mean"),
                total_upside=("gwp_upside_est", "sum"),
            )
            .reset_index()
            .sort_values("avg_composite", ascending=False)
        )

        state_opp = (
            opp_df.groupby("state")
            .agg(
                agent_count=("agent_id", "count"),
                total_gwp=("gwp", "sum"),
                avg_composite=("composite_score", "mean"),
                total_upside=("gwp_upside_est", "sum"),
            )
            .reset_index()
            .sort_values("avg_composite", ascending=False)
        )

        # ── Summary ──────────────────────────────────────────────────────────
        high_pri = opp_df[opp_df["opportunity_class"].str.startswith("★★★")]
        summary = {
            "total_agents_scored": len(opp_df),
            "high_priority_count": len(high_pri),
            "total_gwp_upside":    float(opp_df["gwp_upside_est"].sum()),
            "top_10_agents":       opp_df.head(10)[[
                "agent_id", "agent_name", "state", "composite_score", "gwp_upside_est"
            ]].to_dict("records"),
            "top_states":          state_opp.head(5)[["state", "total_upside"]].to_dict("records"),
            "top_districts":       district_opp.head(10)[["district", "state", "total_upside"]].to_dict("records"),
        }

        log.info(f"[OpportunityAgent] {len(high_pri)} high-priority agents, "
                 f"total upside: ${summary['total_gwp_upside']:,.0f}")

        state["opportunities_df"]   = opp_df
        state["opportunity_summary"] = summary

    except Exception as e:
        err_msg = f"[OpportunityAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        state["opportunities_df"]   = pd.DataFrame()
        state["opportunity_summary"] = {}

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("opportunity_agent")
    state["completed_agents"] = completed
    log.info("[OpportunityAgent] Done.")
    return state
