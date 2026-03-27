"""
Memory & Learning Agent — Stores predictions, retrieves relevant context,
implements feedback loop, and applies bias correction.

Implements:
1. Store current run's predictions in SQLite
2. Retrieve similar historical contexts via FAISS
3. Compute model bias from past errors
4. Apply bias correction to forecasts
5. Incremental learning: adjust forecast multipliers based on systematic errors
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from agents.state import AgentState
from memory.store import MemoryManager

log = logging.getLogger(__name__)


def _extract_context(state: AgentState) -> Dict:
    """Build a context dict from current state for FAISS lookup."""
    features_df = state.get("features_df") or state.get("agent_month_df")
    quality = state.get("data_quality", {})

    ctx = {
        "gwp":               float(quality.get("gwp_total", 0)),
        "pif":               float(quality.get("pif_total", 0)),
        "n_agents":          int(quality.get("n_unique_agents", 0)),
        "avg_renewal_rate":  float(quality.get("avg_renewal_rate", 0.82)),
        "avg_cancellation":  float(quality.get("avg_cancellation_rate", 0.05)),
    }

    if features_df is not None and len(features_df) > 0:
        latest = (
            features_df.sort_values("year_month_dt" if "year_month_dt" in features_df.columns
                                    else "year_month")
            .tail(100)
        )
        for col in ["gwp", "pif", "renewal_rate", "cancellation_rate",
                    "conversion_rate", "n_producers", "avg_premium"]:
            if col in latest.columns:
                ctx[col] = float(latest[col].mean())

    return ctx


def _store_forecast_predictions(
    mm: MemoryManager,
    run_id: str,
    forecasts: Dict,
) -> List[str]:
    """Store forecast predictions in memory for future feedback."""
    pred_ids = []
    for series_name, fc_df in forecasts.items():
        if fc_df is None or len(fc_df) == 0:
            continue
        metric = "gwp" if "gwp" in series_name else "pif"
        fc_col = f"{metric}_forecast"
        if fc_col not in fc_df.columns:
            continue
        for _, row in fc_df.iterrows():
            pid = mm.store_prediction(
                run_id=run_id,
                agent="forecasting_agent",
                metric=series_name,
                period=str(row.get("ds", "unknown")),
                entity_id=series_name.split("_")[0],
                predicted=float(row[fc_col]),
                meta={"series": series_name},
            )
            pred_ids.append(pid)
    return pred_ids


def _apply_learned_corrections(
    mm: MemoryManager,
    forecasts: Dict,
) -> Dict:
    """Apply bias corrections from historical errors to forecasts."""
    corrected = {}
    for series_name, fc_df in forecasts.items():
        if fc_df is None or len(fc_df) == 0:
            corrected[series_name] = fc_df
            continue

        metric  = "gwp" if "gwp" in series_name else "pif"
        fc_col  = f"{metric}_forecast"
        bias    = mm.get_bias(series_name)

        if fc_col in fc_df.columns and abs(bias) > 0.001:
            fc_copy = fc_df.copy()
            correction_factor = 1 - bias * mm.learning_rate
            fc_copy[fc_col] = (fc_copy[fc_col] * correction_factor).clip(lower=0)
            corrected[series_name] = fc_copy
            log.debug(f"[MemoryAgent] Bias correction for {series_name}: "
                      f"bias={bias:.4f}, factor={correction_factor:.4f}")
        else:
            corrected[series_name] = fc_df

    return corrected


def _simulate_feedback(
    mm: MemoryManager,
    forecasts: Dict,
    features_df: pd.DataFrame,
) -> Dict:
    """
    Simulate the feedback loop by comparing the most recent month's
    'forecast' (from the previous run) against 'actuals' (known data).
    In production, this would use truly held-out data.
    """
    feedback_stats = {}

    if features_df is None or len(features_df) == 0:
        return feedback_stats

    # Use last 3 months of actuals as simulated "outcomes"
    recent = (
        features_df.sort_values("year_month_dt" if "year_month_dt" in features_df.columns
                                 else "year_month")
        .tail(3)
    )

    for metric in ["gwp", "pif"]:
        key = f"portfolio_{metric}"
        if key not in forecasts or len(forecasts[key]) == 0:
            continue

        fc_col = f"{metric}_forecast"
        if fc_col not in forecasts[key].columns:
            continue

        actual_mean   = float(recent[metric].mean()) if metric in recent.columns else 0
        forecast_mean = float(forecasts[key][fc_col].mean())

        if actual_mean > 0:
            error_pct = (forecast_mean - actual_mean) / actual_mean
            feedback_stats[metric] = {
                "actual_mean":   actual_mean,
                "forecast_mean": forecast_mean,
                "error_pct":     error_pct,
                "bias_direction": "over" if error_pct > 0 else "under",
            }

            # Update bias correction
            mm.sql.compute_and_store_bias(key)

    return feedback_stats


def memory_agent(state: AgentState) -> AgentState:
    log.info("[MemoryAgent] Running memory operations …")
    errors = list(state.get("errors", []))

    try:
        mm      = MemoryManager.get_instance()
        run_id  = state.get("run_id", "unknown")

        # ── 1. Retrieve similar contexts ─────────────────────────────────────
        context = _extract_context(state)
        similar = mm.retrieve_similar(context, k=5)
        log.info(f"[MemoryAgent] Retrieved {len(similar)} similar past contexts")

        # ── 2. Store current forecasts ────────────────────────────────────────
        forecasts = state.get("forecasts", {})
        pred_ids  = _store_forecast_predictions(mm, run_id, forecasts)
        log.info(f"[MemoryAgent] Stored {len(pred_ids)} forecast predictions")

        # ── 3. Apply bias corrections from historical learning ────────────────
        if forecasts:
            corrected_forecasts = _apply_learned_corrections(mm, forecasts)
            state["forecasts"] = corrected_forecasts

        # ── 4. Simulate feedback loop (actual vs predicted) ───────────────────
        features_df = state.get("features_df") or state.get("agent_month_df")
        feedback    = _simulate_feedback(mm, state.get("forecasts", {}), features_df)
        log.info(f"[MemoryAgent] Feedback loop: {feedback}")

        # ── 5. Store current context in FAISS for future retrieval ────────────
        memory_payload = {
            "run_id":            run_id,
            "gwp_total":         context.get("gwp"),
            "pif_total":         context.get("pif"),
            "feedback":          feedback,
            "n_forecasts":       len(forecasts),
            "outlier_pct":       state.get("outlier_summary", {}).get("pct_flagged"),
            "top_opportunity":   (state.get("opportunity_summary", {})
                                  .get("top_10_agents", [{}])[0]
                                  .get("composite_score") if state.get("opportunity_summary") else None),
        }
        mm.store_context(context, memory_payload)

        # ── 6. Build learning update ──────────────────────────────────────────
        error_summary = mm.get_error_summary()
        learning_update = {
            "bias_corrections": {
                m: mm.get_bias(m) for m in ["portfolio_gwp", "portfolio_pif"]
            },
            "historical_mape":  error_summary,
            "feedback":         feedback,
            "n_memories":       mm.faiss.size(),
        }

        # ── 7. Log decision ───────────────────────────────────────────────────
        mm.store_decision(
            run_id=run_id,
            decision="Memory retrieval and bias correction applied",
            context=f"GWP={context.get('gwp', 0):,.0f}, "
                    f"PIF={context.get('pif', 0):,.0f}",
        )

        state["memory_retrieved"] = similar
        state["learning_update"]  = learning_update
        state["prediction_log"]   = [{"id": p} for p in pred_ids[:20]]

    except Exception as e:
        err_msg = f"[MemoryAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        state["memory_retrieved"] = []
        state["learning_update"]  = {}

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("memory_agent")
    state["completed_agents"] = completed
    log.info("[MemoryAgent] Done.")
    return state
