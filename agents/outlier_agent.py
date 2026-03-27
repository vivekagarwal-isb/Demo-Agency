"""
Outlier Detection Agent — Identifies anomalous agents, policies, and time periods.

Methods:
1. Isolation Forest (multivariate)
2. Z-score (univariate, per metric)
3. IQR fence (robust)
4. Model-based deviation (actual vs rolling expected)

Flags:
- Premium outliers (unusually high/low)
- Volume outliers (GWP/PIF spike or crash)
- Behavioural outliers (sudden change in NB/renewal/cancellation mix)
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from agents.state import AgentState

log = logging.getLogger(__name__)

try:
    from config import OUTLIER_CONFIG
except ImportError:
    OUTLIER_CONFIG = {"contamination": 0.03, "z_score_threshold": 3.0, "iqr_multiplier": 1.5}


OUTLIER_FEATURES = [
    "gwp", "pif", "new_business_count", "renewal_count",
    "cancellation_count", "avg_premium", "renewal_rate",
    "cancellation_rate", "nb_rate",
]


def _isolation_forest_flags(df: pd.DataFrame) -> np.ndarray:
    """Return boolean mask of outliers via Isolation Forest."""
    feature_cols = [c for c in OUTLIER_FEATURES if c in df.columns]
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = IsolationForest(
        contamination=OUTLIER_CONFIG["contamination"],
        random_state=42,
        n_estimators=100,
    )
    preds = clf.fit_predict(X_scaled)
    return preds == -1  # True = outlier


def _zscore_flags(df: pd.DataFrame, col: str, threshold: float) -> np.ndarray:
    vals = df[col].fillna(df[col].median()).values
    z = np.abs((vals - vals.mean()) / (vals.std() + 1e-9))
    return z > threshold


def _iqr_flags(df: pd.DataFrame, col: str, multiplier: float) -> np.ndarray:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (df[col] < lo) | (df[col] > hi)


def _model_deviation(df: pd.DataFrame, col: str, window: int = 3) -> pd.DataFrame:
    """Flag rows where actual deviates >2σ from rolling expected."""
    df = df.copy().sort_values(["agent_id", "year_month_dt"])
    roll_mean = df.groupby("agent_id")[col].transform(
        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
    )
    roll_std  = df.groupby("agent_id")[col].transform(
        lambda x: x.rolling(window, min_periods=1).std().shift(1).fillna(1)
    )
    deviation = (df[col] - roll_mean).abs() / (roll_std + 1e-9)
    return deviation


def outlier_agent(state: AgentState) -> AgentState:
    log.info("[OutlierAgent] Starting outlier detection …")
    errors = list(state.get("errors", []))

    try:
        df = (state.get("features_df") or state.get("agent_month_df")).copy()

        if "year_month_dt" not in df.columns:
            df["year_month_dt"] = pd.to_datetime(df["year_month"])

        # ── Method 1: Isolation Forest ────────────────────────────────────────
        df["if_outlier"] = _isolation_forest_flags(df)

        # ── Method 2: Z-score on GWP ─────────────────────────────────────────
        df["zscore_gwp_outlier"]  = _zscore_flags(df, "gwp",  OUTLIER_CONFIG["z_score_threshold"])
        df["zscore_pif_outlier"]  = _zscore_flags(df, "pif",  OUTLIER_CONFIG["z_score_threshold"])
        df["zscore_can_outlier"]  = _zscore_flags(df, "cancellation_rate",
                                                   OUTLIER_CONFIG["z_score_threshold"])

        # ── Method 3: IQR on premium ─────────────────────────────────────────
        if "avg_premium" in df.columns:
            df["iqr_premium_outlier"] = _iqr_flags(
                df, "avg_premium", OUTLIER_CONFIG["iqr_multiplier"]
            )
        else:
            df["iqr_premium_outlier"] = False

        # ── Method 4: Model deviation ─────────────────────────────────────────
        for col in ["gwp", "pif"]:
            df[f"{col}_deviation"] = _model_deviation(df, col)
        df["model_deviation_outlier"] = (
            (df["gwp_deviation"] > 2.5) | (df["pif_deviation"] > 2.5)
        )

        # ── Composite flag ───────────────────────────────────────────────────
        df["is_outlier"] = (
            df["if_outlier"] |
            df["zscore_gwp_outlier"] |
            df["zscore_can_outlier"] |
            df["model_deviation_outlier"]
        )

        df["outlier_score"] = (
            df["if_outlier"].astype(int) * 3 +
            df["zscore_gwp_outlier"].astype(int) * 2 +
            df["zscore_can_outlier"].astype(int) * 2 +
            df["iqr_premium_outlier"].astype(int) * 1 +
            df["model_deviation_outlier"].astype(int) * 2
        )

        # ── Outlier type classification ───────────────────────────────────────
        conditions = [
            (df["gwp"] > df["gwp"].quantile(0.97)) & df["is_outlier"],
            (df["gwp"] < df["gwp"].quantile(0.03)) & df["is_outlier"],
            (df["cancellation_rate"] > df["cancellation_rate"].quantile(0.97)) & df["is_outlier"],
            (df["new_business_count"] > df["new_business_count"].quantile(0.97)) & df["is_outlier"],
        ]
        choices = ["GWP Spike", "GWP Crash", "High Cancellation", "NB Surge"]
        df["outlier_type"] = np.select(conditions, choices, default="Multivariate")
        df.loc[~df["is_outlier"], "outlier_type"] = "Normal"

        outliers_df = df[df["is_outlier"]].copy()
        log.info(f"[OutlierAgent] Flagged {len(outliers_df):,} outlier rows "
                 f"({len(outliers_df)/len(df):.1%} of agent-months)")

        outlier_summary = {
            "total_flagged": int(df["is_outlier"].sum()),
            "pct_flagged": float(df["is_outlier"].mean()),
            "by_type": outliers_df["outlier_type"].value_counts().to_dict(),
            "top_agents": (
                outliers_df.groupby("agent_id")["outlier_score"]
                .sum().nlargest(10).to_dict()
            ),
            "top_states": (
                outliers_df.groupby("state")["outlier_score"]
                .sum().nlargest(5).to_dict()
            ),
            "gwp_at_risk": float(outliers_df[outliers_df["outlier_type"] == "GWP Crash"]["gwp"].sum()),
        }

        state["outliers_df"]    = outliers_df
        state["features_df"]    = df          # updated with outlier flags
        state["outlier_summary"] = outlier_summary

    except Exception as e:
        err_msg = f"[OutlierAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        state["outliers_df"]    = pd.DataFrame()
        state["outlier_summary"] = {}

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("outlier_agent")
    state["completed_agents"] = completed
    log.info("[OutlierAgent] Done.")
    return state
