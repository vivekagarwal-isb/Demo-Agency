"""
Shock Simulation Agent — Applies user-defined lever shocks to Topline.

Levers (from Streamlit sliders):
- Conversion rate:    ±30%
- Retention rate:     ±20%
- Pricing:            ±15%
- Producer count:     ±40%
- Discount:           ±20%

Each shock is applied to the agent-month data and the impact on
GWP / PIF is calculated using multiplicative adjustment rules
informed by the causal estimates.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from agents.state import AgentState

log = logging.getLogger(__name__)

try:
    from config import SHOCK_DEFAULTS, TWIN_CONFIG
except ImportError:
    SHOCK_DEFAULTS = {
        "conversion_delta": 0.0,
        "retention_delta": 0.0,
        "pricing_delta": 0.0,
        "producer_count_delta": 0.0,
        "discount_delta": 0.0,
    }
    TWIN_CONFIG = {
        "pricing_sensitivity": -0.8,
        "discount_sensitivity": 0.3,
        "retention_gwp_multiplier": 1.2,
    }


def _apply_shock(
    df: pd.DataFrame,
    conversion_delta: float = 0.0,
    retention_delta: float = 0.0,
    pricing_delta: float = 0.0,
    producer_count_delta: float = 0.0,
    discount_delta: float = 0.0,
) -> pd.DataFrame:
    """
    Apply shocks to agent-month data and recompute KPIs.

    Shock mechanics:
      conversion_delta: +0.10 → 10% more new business policies
      retention_delta:  +0.05 → 5% fewer cancellations
      pricing_delta:    +0.05 → 5% higher average premium (reduces conversion slightly)
      producer_count_delta: +0.10 → 10% more producers → proportional NB increase
      discount_delta:   +0.05 → 5% more discount → higher conversion but lower premium
    """
    shocked = df.copy()

    # ── New Business Impact ──────────────────────────────────────────────────
    # More conversion + more producers → more new policies
    nb_multiplier = (
        (1 + conversion_delta) *
        (1 + producer_count_delta) *
        (1 + TWIN_CONFIG["discount_sensitivity"] * discount_delta) *
        (1 + TWIN_CONFIG["pricing_sensitivity"] * pricing_delta * 0.5)
    )
    nb_multiplier = max(0.1, nb_multiplier)
    shocked["new_business_count"] = (shocked["new_business_count"] * nb_multiplier).clip(lower=0)
    shocked["new_business_gwp"]   = (shocked["new_business_gwp"] * nb_multiplier).clip(lower=0)

    # ── Renewal / Retention Impact ───────────────────────────────────────────
    # Higher retention → fewer cancellations → more renewals
    eff_retention    = (shocked["renewal_rate"] + retention_delta).clip(0, 0.99)
    eff_cancellation = (shocked["cancellation_rate"] - 0.5 * retention_delta).clip(0.001, 0.5)
    shocked["renewal_rate"]      = eff_retention
    shocked["cancellation_rate"] = eff_cancellation
    renewal_multiplier = eff_retention / (shocked["renewal_rate"].replace(0, 0.01))
    shocked["renewal_count"]  = (shocked["renewal_count"] * renewal_multiplier).clip(lower=0)

    # ── Premium Impact ───────────────────────────────────────────────────────
    # Pricing up → higher premium on each policy, but also reduces conversion
    premium_multiplier = (1 + pricing_delta) * (1 - 0.05 * discount_delta)
    premium_multiplier = max(0.5, premium_multiplier)
    shocked["avg_premium"] = (shocked["avg_premium"] * premium_multiplier).clip(lower=100)

    # ── PIF Impact ───────────────────────────────────────────────────────────
    pif_delta_rate = (
        + conversion_delta * 0.4
        + retention_delta  * 0.6
        - pricing_delta    * 0.2
        + producer_count_delta * 0.3
        + discount_delta   * 0.15
    )
    shocked["pif"] = (shocked["pif"] * (1 + pif_delta_rate)).clip(lower=0)

    # ── GWP Recomputation ────────────────────────────────────────────────────
    gwp_delta_rate = (
        + conversion_delta  * 0.30 * (1 + TWIN_CONFIG["discount_sensitivity"] * discount_delta)
        + retention_delta   * TWIN_CONFIG["retention_gwp_multiplier"] * 0.20
        + pricing_delta     * 0.40                     # price directly boosts GWP
        + producer_count_delta * 0.25
        - discount_delta    * 0.05                     # slight premium erosion
    )
    shocked["gwp"] = (shocked["gwp"] * (1 + gwp_delta_rate)).clip(lower=0)

    return shocked


def _compute_impact(baseline_df: pd.DataFrame, shocked_df: pd.DataFrame) -> Dict:
    """Compute $ and % impact of shock vs baseline."""
    base_gwp = baseline_df["gwp"].sum()
    shk_gwp  = shocked_df["gwp"].sum()
    base_pif = baseline_df["pif"].sum()
    shk_pif  = shocked_df["pif"].sum()

    return {
        "baseline_gwp":    float(base_gwp),
        "shocked_gwp":     float(shk_gwp),
        "gwp_delta":       float(shk_gwp - base_gwp),
        "gwp_delta_pct":   float((shk_gwp - base_gwp) / (base_gwp + 1e-9)),
        "baseline_pif":    float(base_pif),
        "shocked_pif":     float(shk_pif),
        "pif_delta":       float(shk_pif - base_pif),
        "pif_delta_pct":   float((shk_pif - base_pif) / (base_pif + 1e-9)),
        "nb_delta":        float(shocked_df["new_business_count"].sum() - baseline_df["new_business_count"].sum()),
        "renewal_delta":   float(shocked_df["renewal_count"].sum() - baseline_df["renewal_count"].sum()),
    }


def shock_agent(state: AgentState) -> AgentState:
    log.info("[ShockAgent] Running scenario simulations …")
    errors = list(state.get("errors", []))

    try:
        df = (state.get("features_df") or state.get("agent_month_df")).copy()
        shock_params = state.get("shock_params") or SHOCK_DEFAULTS

        # ── User-defined scenario ─────────────────────────────────────────────
        user_shocked = _apply_shock(
            df,
            conversion_delta     = shock_params.get("conversion_delta", 0.0),
            retention_delta      = shock_params.get("retention_delta", 0.0),
            pricing_delta        = shock_params.get("pricing_delta", 0.0),
            producer_count_delta = shock_params.get("producer_count_delta", 0.0),
            discount_delta       = shock_params.get("discount_delta", 0.0),
        )
        user_impact = _compute_impact(df, user_shocked)

        # ── Pre-defined scenarios ─────────────────────────────────────────────
        scenarios = {
            "baseline": {
                "df": df.copy(),
                "params": SHOCK_DEFAULTS,
                "impact": {
                    "baseline_gwp": float(df["gwp"].sum()),
                    "shocked_gwp":  float(df["gwp"].sum()),
                    "gwp_delta": 0.0, "gwp_delta_pct": 0.0,
                    "baseline_pif": float(df["pif"].sum()),
                    "shocked_pif":  float(df["pif"].sum()),
                    "pif_delta": 0.0, "pif_delta_pct": 0.0,
                    "nb_delta": 0.0, "renewal_delta": 0.0,
                },
            },
            "high_growth": {
                "df": _apply_shock(df, conversion_delta=0.15, retention_delta=0.10,
                                   producer_count_delta=0.20),
                "params": {"conversion_delta": 0.15, "retention_delta": 0.10,
                           "producer_count_delta": 0.20},
                "impact": None,
            },
            "pricing_hardening": {
                "df": _apply_shock(df, pricing_delta=0.10, conversion_delta=-0.08),
                "params": {"pricing_delta": 0.10, "conversion_delta": -0.08},
                "impact": None,
            },
            "competitive_pressure": {
                "df": _apply_shock(df, conversion_delta=-0.15, retention_delta=-0.05,
                                   discount_delta=0.10),
                "params": {"conversion_delta": -0.15, "retention_delta": -0.05,
                           "discount_delta": 0.10},
                "impact": None,
            },
            "producer_expansion": {
                "df": _apply_shock(df, producer_count_delta=0.30, conversion_delta=0.05),
                "params": {"producer_count_delta": 0.30, "conversion_delta": 0.05},
                "impact": None,
            },
            "user_scenario": {
                "df": user_shocked,
                "params": shock_params,
                "impact": user_impact,
            },
        }

        # Compute impacts for pre-defined scenarios
        for name, sc in scenarios.items():
            if sc["impact"] is None:
                sc["impact"] = _compute_impact(df, sc["df"])

        # ── State-level scenario comparison ──────────────────────────────────
        state_comparison = {}
        for sc_name, sc in scenarios.items():
            if sc_name == "baseline":
                continue
            state_comp = (
                sc["df"].groupby("state")["gwp"].sum()
                - df.groupby("state")["gwp"].sum()
            ).rename(f"{sc_name}_gwp_delta")
            state_comparison[sc_name] = state_comp.to_dict()

        shock_results = {
            "scenarios": {
                k: {
                    "params": v["params"],
                    "impact": v["impact"],
                }
                for k, v in scenarios.items()
            },
            "state_comparison": state_comparison,
        }

        # Build scenario summary text
        lines = ["## Shock Simulation Results\n"]
        for sc_name, sc in scenarios.items():
            imp = sc["impact"]
            lines.append(
                f"**{sc_name.replace('_',' ').title()}**: "
                f"GWP {imp['gwp_delta_pct']:+.1%} (${imp['gwp_delta']:+,.0f}) | "
                f"PIF {imp['pif_delta_pct']:+.1%} ({imp['pif_delta']:+,.0f})"
            )
        shock_summary = "\n".join(lines)

        log.info(f"[ShockAgent] User scenario: GWP delta = "
                 f"{user_impact['gwp_delta_pct']:+.1%}, "
                 f"${user_impact['gwp_delta']:+,.0f}")

        state["shock_results"] = shock_results
        state["shock_summary"] = shock_summary

    except Exception as e:
        err_msg = f"[ShockAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        state["shock_results"] = {}
        state["shock_summary"] = "Shock simulation encountered an error."

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("shock_agent")
    state["completed_agents"] = completed
    log.info("[ShockAgent] Done.")
    return state
