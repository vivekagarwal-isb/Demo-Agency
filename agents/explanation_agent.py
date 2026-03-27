"""
Explanation Agent — Generates SHAP-based feature importance explanations
and integrates causal findings into human-readable narratives.

Outputs:
- Global SHAP feature importance (what drives GWP at portfolio level)
- Local SHAP (why a specific agent is an outlier / opportunity)
- Causal-enriched explanations ("X CAUSES Y, not just correlated")
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from agents.state import AgentState

log = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning("SHAP not installed — using permutation importance fallback.")


FEATURE_COLS = [
    "renewal_rate", "cancellation_rate", "conversion_rate",
    "n_producers", "quality_score", "tenure_years",
    "nb_rate", "avg_premium", "avg_discount",
    "sin_month", "cos_month",
    "gwp_rolling_3m_mean", "gwp_rolling_6m_mean",
    "pif_rolling_3m_mean", "momentum",
    "market_share_district", "gwp_per_producer",
    "opportunity_gap_pct", "health_score",
]


def _prepare_features(df: pd.DataFrame, target: str = "gwp") -> tuple:
    """Prepare feature matrix X and target y."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        return None, None, None

    sub = df[available + [target]].dropna()
    if len(sub) < 50:
        return None, None, None

    X = sub[available].values
    y = sub[target].values
    return X, y, available


def _shap_global(model, X: np.ndarray, feature_names: List[str]) -> Dict:
    """Global SHAP feature importance."""
    if not SHAP_AVAILABLE:
        return {}
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:500])  # sample for speed
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(feature_names, mean_abs.tolist()))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        log.debug(f"SHAP global failed: {e}")
        return {}


def _permutation_importance(model, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str]) -> Dict:
    """Fallback: permutation importance."""
    base_score = np.mean((model.predict(X) - y) ** 2)
    importances = {}
    rng = np.random.default_rng(42)
    for i, name in enumerate(feature_names):
        X_perm = X.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_score = np.mean((model.predict(X_perm) - y) ** 2)
        importances[name] = max(0, perm_score - base_score)
    total = sum(importances.values()) or 1
    return {k: v / total for k, v in
            sorted(importances.items(), key=lambda x: x[1], reverse=True)}


def _local_shap(model, X_local: np.ndarray, feature_names: List[str]) -> Optional[Dict]:
    """Local SHAP for a single instance."""
    if not SHAP_AVAILABLE or X_local is None:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_local.reshape(1, -1))
        if sv is not None:
            return dict(zip(feature_names, sv[0].tolist()))
    except Exception:
        pass
    return None


def _build_narratives(
    feature_importance: Dict,
    ate_results: Dict,
    outlier_summary: Dict,
    opportunity_summary: Dict,
) -> List[str]:
    """Compose human-readable AI insight bullets."""
    insights = []

    # Top drivers
    top_features = list(feature_importance.items())[:5]
    if top_features:
        top_str = ", ".join(f"**{k}** ({v:.3f})" for k, v in top_features)
        insights.append(f"🔍 Top GWP drivers (SHAP): {top_str}")

    # Causal insights
    for label, result in ate_results.items():
        ate = result.get("ate")
        p   = result.get("p_value")
        if ate is not None and p is not None and p < 0.05:
            direction = "increases" if ate > 0 else "decreases"
            insights.append(
                f"⚡ CAUSAL: {label} — 1-unit increase "
                f"{direction} GWP by ${abs(ate):,.0f} (p={p:.3f})"
            )

    # Outliers
    if outlier_summary:
        n_flagged = outlier_summary.get("total_flagged", 0)
        gwp_risk  = outlier_summary.get("gwp_at_risk", 0)
        insights.append(
            f"⚠️ {n_flagged:,} agent-months flagged as outliers. "
            f"GWP at risk (crash outliers): ${gwp_risk:,.0f}"
        )
        by_type = outlier_summary.get("by_type", {})
        if by_type:
            top_type = max(by_type, key=by_type.get)
            insights.append(f"   Most common outlier type: **{top_type}** ({by_type[top_type]} cases)")

    # Opportunity
    if opportunity_summary:
        upside     = opportunity_summary.get("total_gwp_upside", 0)
        high_pri   = opportunity_summary.get("high_priority_count", 0)
        top_agents = opportunity_summary.get("top_10_agents", [])
        insights.append(
            f"💰 Total addressable GWP opportunity: **${upside:,.0f}** "
            f"across {high_pri} high-priority agents"
        )
        if top_agents:
            top_agent = top_agents[0]
            insights.append(
                f"   Top opportunity agent: **{top_agent.get('agent_name', 'N/A')}** "
                f"({top_agent.get('state', 'N/A')}) — "
                f"${top_agent.get('gwp_upside_est', 0):,.0f} upside"
            )

    return insights


def explanation_agent(state: AgentState) -> AgentState:
    log.info("[ExplanationAgent] Generating SHAP explanations …")
    errors = list(state.get("errors", []))

    try:
        df = (state.get("features_df") or state.get("agent_month_df")).copy()

        # ── Train a GBM model for SHAP ────────────────────────────────────────
        X, y, feat_names = _prepare_features(df, target="gwp")
        shap_global_importance = {}

        if X is not None and len(X) >= 50:
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
            scaler  = StandardScaler()
            X_sc    = scaler.fit_transform(X)
            model.fit(X_sc, y)

            # Global SHAP
            shap_global_importance = _shap_global(model, X_sc, feat_names)
            if not shap_global_importance:
                shap_global_importance = _permutation_importance(model, X_sc, y, feat_names)

            log.info(f"[ExplanationAgent] Top feature: "
                     f"{list(shap_global_importance.keys())[0] if shap_global_importance else 'N/A'}")

            # Local SHAP for top-10 outliers
            outliers_df = state.get("outliers_df")
            local_shap_map = {}
            if outliers_df is not None and len(outliers_df) > 0:
                top_outliers = outliers_df.nlargest(min(10, len(outliers_df)), "outlier_score")
                for _, row in top_outliers.iterrows():
                    feat_vals = [row.get(f, 0) for f in feat_names]
                    X_loc = scaler.transform(np.array([feat_vals]))
                    local = _local_shap(model, X_loc[0], feat_names)
                    if local:
                        local_shap_map[row.get("agent_id", "unknown")] = local

            state["shap_values"] = {
                "global_importance": shap_global_importance,
                "local": local_shap_map,
            }
        else:
            state["shap_values"] = {"global_importance": {}, "local": {}}

        # ── Narrative insights ────────────────────────────────────────────────
        insights = _build_narratives(
            feature_importance=shap_global_importance,
            ate_results=state.get("ate_results", {}),
            outlier_summary=state.get("outlier_summary", {}),
            opportunity_summary=state.get("opportunity_summary", {}),
        )
        state["explanations"] = insights
        log.info(f"[ExplanationAgent] Generated {len(insights)} insight bullets.")

    except Exception as e:
        err_msg = f"[ExplanationAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        state["shap_values"]  = {}
        state["explanations"] = ["Explanation generation encountered an error."]

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("explanation_agent")
    state["completed_agents"] = completed
    log.info("[ExplanationAgent] Done.")
    return state
