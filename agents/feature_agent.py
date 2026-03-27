"""
Feature Engineering Agent — Derives enriched KPIs and ML features.

Computed features:
- Rolling averages (3m, 6m, 12m) for GWP, PIF, NB, Renewals
- YoY growth rates
- Agent tier classification (based on GWP percentile)
- Market share within district
- Producer efficiency metrics
- Momentum scores
- Lag features for ML models
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from agents.state import AgentState

log = logging.getLogger(__name__)


def _rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    """Add rolling mean/std features for a column, grouped by agent."""
    for w in windows:
        df[f"{col}_rolling_{w}m_mean"] = (
            df.groupby("agent_id")[col]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"{col}_rolling_{w}m_std"] = (
            df.groupby("agent_id")[col]
            .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        )
    return df


def _yoy_growth(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Year-over-year growth rate."""
    df = df.sort_values(["agent_id", "year", "month"])
    df[f"{col}_yoy_growth"] = (
        df.groupby(["agent_id", "month"])[col]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return df


def _lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag{lag}"] = (
            df.groupby("agent_id")[col].shift(lag).fillna(method="bfill")
        )
    return df


def _agent_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Classify agents into tiers based on trailing 3-month GWP."""
    trailing = (
        df.groupby("agent_id")["gwp"]
        .apply(lambda x: x.tail(3).mean())
        .rename("trailing_gwp")
    )
    percentiles = trailing.quantile([0.25, 0.50, 0.75, 0.90])
    def tier(v):
        if v >= percentiles[0.90]:  return "Platinum"
        elif v >= percentiles[0.75]: return "Gold"
        elif v >= percentiles[0.50]: return "Silver"
        elif v >= percentiles[0.25]: return "Bronze"
        else: return "Developing"
    trailing_df = trailing.reset_index()
    trailing_df["agent_tier"] = trailing_df["trailing_gwp"].apply(tier)
    return df.merge(trailing_df[["agent_id", "agent_tier"]], on="agent_id", how="left")


def _market_share(df: pd.DataFrame) -> pd.DataFrame:
    """Each agent's share of GWP within district-month."""
    district_total = (
        df.groupby(["district", "year_month"])["gwp"]
        .transform("sum")
        .replace(0, np.nan)
    )
    df["market_share_district"] = df["gwp"] / district_total
    return df


def _producer_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """GWP and PIF per producer, per agent."""
    df["gwp_per_producer"]  = df["gwp"] / (df["n_producers"].clip(lower=1))
    df["pif_per_producer"]  = df["pif"] / (df["n_producers"].clip(lower=1))
    df["nb_per_producer"]   = df["new_business_count"] / (df["n_producers"].clip(lower=1))
    return df


def _momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum = weighted sum of recent growth signals.
    Higher = accelerating, Lower = decelerating.
    """
    df["momentum"] = (
        0.4 * df.get("gwp_rolling_3m_mean", df["gwp"]) / (df["gwp"].mean() + 1e-9) +
        0.3 * df["renewal_rate"] +
        0.3 * (1 - df["cancellation_rate"])
    )
    df["momentum"] = df["momentum"].clip(0, 3)
    return df


def _opportunity_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opportunity gap = district average GWP - agent GWP (per producer).
    Positive = underperforming vs peers → opportunity.
    """
    district_avg = (
        df.groupby(["district", "year_month"])["gwp_per_producer"]
        .transform("mean")
    )
    df["opportunity_gap"] = district_avg - df["gwp_per_producer"]
    df["opportunity_gap_pct"] = df["opportunity_gap"] / (district_avg.clip(lower=1))
    return df


def feature_agent(state: AgentState) -> AgentState:
    """Feature engineering on agent-month dataframe."""
    log.info("[FeatureAgent] Starting feature engineering …")
    errors = list(state.get("errors", []))

    try:
        df = state["agent_month_df"].copy()
        df = df.sort_values(["agent_id", "year_month_dt"]).reset_index(drop=True)

        # ── Rolling features ─────────────────────────────────────────────────
        for col in ["gwp", "pif", "new_business_count", "renewal_count"]:
            df = _rolling_features(df, col, windows=[3, 6, 12])

        # ── Lag features ─────────────────────────────────────────────────────
        for col in ["gwp", "pif"]:
            df = _lag_features(df, col, lags=[1, 3, 6])

        # ── YoY growth ───────────────────────────────────────────────────────
        for col in ["gwp", "pif"]:
            df = _yoy_growth(df, col)

        # ── Structural features ───────────────────────────────────────────────
        df = _agent_tier(df)
        df = _market_share(df)
        df = _producer_efficiency(df)
        df = _momentum_score(df)
        df = _opportunity_gap(df)

        # ── Seasonal dummies ─────────────────────────────────────────────────
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["is_q4"]     = (df["month"].isin([10, 11, 12])).astype(int)

        # ── Composite scores ─────────────────────────────────────────────────
        df["health_score"] = (
            0.35 * df["renewal_rate"].clip(0, 1) +
            0.25 * (1 - df["cancellation_rate"].clip(0, 1)) +
            0.20 * df["nb_rate"].clip(0, 1) +
            0.20 * df["quality_score"].clip(0, 1)
        )

        # Fill NaN from rolling/lag
        df = df.fillna(0)

        n_features = len(df.columns)
        log.info(f"[FeatureAgent] Feature matrix: {df.shape}  ({n_features} columns)")

        # Feature importance proxy (variance-based)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        variances    = df[numeric_cols].var().sort_values(ascending=False)
        feature_importance = variances.head(20).to_dict()

        state["features_df"]        = df
        state["feature_importance"] = feature_importance

    except Exception as e:
        err_msg = f"[FeatureAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        # Fallback: pass through original
        state["features_df"]        = state.get("agent_month_df")
        state["feature_importance"] = {}

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("feature_agent")
    state["completed_agents"] = completed
    log.info("[FeatureAgent] Done.")
    return state
