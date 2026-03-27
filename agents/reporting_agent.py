"""
Reporting Agent — Synthesises all agent outputs into a structured report.

Produces:
1. Executive summary (markdown)
2. KPI scorecard
3. Top findings per section
4. Actionable recommendations
5. Saves report to disk
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from agents.state import AgentState

log = logging.getLogger(__name__)

try:
    from config import OUTPUTS_DIR, KPI_CONFIG
except ImportError:
    OUTPUTS_DIR = Path("outputs")
    KPI_CONFIG = {
        "gwp_target_annual": 150_000_000,
        "pif_target": 85_000,
        "nb_rate_target": 0.25,
        "renewal_rate_target": 0.85,
        "cancellation_rate_target": 0.05,
    }

OUTPUTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI SCORECARD
# ─────────────────────────────────────────────────────────────────────────────

def _build_scorecard(state: AgentState) -> str:
    quality  = state.get("data_quality", {})
    features = state.get("features_df")

    gwp_total   = quality.get("gwp_total", 0)
    pif_total   = quality.get("pif_total", 0)
    n_agents    = quality.get("n_unique_agents", 0)
    renewal_rate = quality.get("avg_renewal_rate", 0)
    cancel_rate  = quality.get("avg_cancellation_rate", 0)

    gwp_target   = KPI_CONFIG["gwp_target_annual"]
    pif_target   = KPI_CONFIG["pif_target"]
    gwp_attainm  = gwp_total / (gwp_target + 1e-9) * 100
    pif_attainm  = pif_total / (pif_target + 1e-9) * 100

    def status(val, target, higher_is_better=True):
        diff = val - target if higher_is_better else target - val
        if diff >= 0:  return "✅"
        if diff >= -abs(target) * 0.1: return "⚠️"
        return "❌"

    lines = [
        "## KPI Scorecard\n",
        f"| Metric | Actual | Target | Status |",
        f"|--------|--------|--------|--------|",
        f"| Total GWP | ${gwp_total:,.0f} | ${gwp_target:,.0f} | "
        f"{status(gwp_total, gwp_target)} |",
        f"| PIF | {pif_total:,.0f} | {pif_target:,.0f} | "
        f"{status(pif_total, pif_target)} |",
        f"| Renewal Rate | {renewal_rate:.1%} | {KPI_CONFIG['renewal_rate_target']:.1%} | "
        f"{status(renewal_rate, KPI_CONFIG['renewal_rate_target'])} |",
        f"| Cancellation Rate | {cancel_rate:.1%} | {KPI_CONFIG['cancellation_rate_target']:.1%} | "
        f"{status(cancel_rate, KPI_CONFIG['cancellation_rate_target'], higher_is_better=False)} |",
        f"| Active Agents | {n_agents} | — | ℹ️ |",
        f"| GWP Attainment | {gwp_attainm:.1f}% | 100% | "
        f"{'✅' if gwp_attainm >= 100 else '⚠️' if gwp_attainm >= 90 else '❌'} |",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST SECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_forecast_section(state: AgentState) -> str:
    forecasts = state.get("forecasts", {})
    accuracy  = state.get("forecast_accuracy", {})
    horizon   = state.get("forecast_horizon", 12)

    lines = [f"\n## Forecast ({horizon}-Month Horizon)\n"]

    for key in ["portfolio_gwp", "portfolio_pif"]:
        if key not in forecasts or len(forecasts[key]) == 0:
            continue
        fc  = forecasts[key]
        metric = "gwp" if "gwp" in key else "pif"
        fc_col = f"{metric}_forecast"
        if fc_col not in fc.columns:
            continue
        total_fcast = fc[fc_col].sum()
        avg_fcast   = fc[fc_col].mean()
        acc  = accuracy.get(key, {})
        mape = acc.get("mape")
        mape_str = f"{mape:.1%}" if mape is not None else "N/A"

        lines.append(f"**{key.replace('_', ' ').title()}**")
        lines.append(f"- Forecast sum: ${total_fcast:,.0f}")
        lines.append(f"- Monthly avg:  ${avg_fcast:,.0f}")
        lines.append(f"- Model MAPE:   {mape_str}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER SECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_outlier_section(state: AgentState) -> str:
    summary = state.get("outlier_summary", {})
    if not summary:
        return "\n## Outlier Detection\nNo outlier data available."

    lines = ["\n## Outlier Detection\n"]
    lines.append(f"- **Total flagged**: {summary.get('total_flagged', 0):,} agent-months "
                 f"({summary.get('pct_flagged', 0):.1%})")
    lines.append(f"- **GWP at risk** (crash outliers): ${summary.get('gwp_at_risk', 0):,.0f}")

    by_type = summary.get("by_type", {})
    if by_type:
        lines.append("\n**By Type:**")
        for t, cnt in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {t}: {cnt}")

    top_agents = summary.get("top_agents", {})
    if top_agents:
        lines.append("\n**Top Anomalous Agents:**")
        for aid, score in list(top_agents.items())[:5]:
            lines.append(f"  - {aid}: score={score:.0f}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OPPORTUNITY SECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_opportunity_section(state: AgentState) -> str:
    summary = state.get("opportunity_summary", {})
    if not summary:
        return "\n## Opportunity Analysis\nNo opportunity data available."

    lines = ["\n## Opportunity Analysis\n"]
    lines.append(f"- **Total GWP Upside**: ${summary.get('total_gwp_upside', 0):,.0f}")
    lines.append(f"- **High Priority Agents**: {summary.get('high_priority_count', 0)}")

    top_10 = summary.get("top_10_agents", [])
    if top_10:
        lines.append("\n**Top 10 Opportunity Agents:**")
        lines.append("| Rank | Agent | State | Score | GWP Upside |")
        lines.append("|------|-------|-------|-------|------------|")
        for i, ag in enumerate(top_10[:10], 1):
            lines.append(
                f"| {i} | {ag.get('agent_name', 'N/A')} | {ag.get('state', 'N/A')} | "
                f"{ag.get('composite_score', 0):.3f} | ${ag.get('gwp_upside_est', 0):,.0f} |"
            )

    top_states = summary.get("top_states", [])
    if top_states:
        lines.append("\n**Top Opportunity States:**")
        for st in top_states:
            lines.append(f"  - {st.get('state', 'N/A')}: ${st.get('total_upside', 0):,.0f}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL SECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_causal_section(state: AgentState) -> str:
    summary = state.get("causal_summary", "")
    ate     = state.get("ate_results", {})

    lines = ["\n## Causal Inference\n"]
    if summary:
        lines.append(summary)

    if ate:
        lines.append("\n**Average Treatment Effects (ATE):**")
        for label, res in ate.items():
            a = res.get("ate")
            p = res.get("p_value")
            if a is not None:
                lines.append(
                    f"- {label}: ATE=${a:+,.2f}"
                    + (f"  (p={p:.3f})" if p is not None else "")
                )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SHOCK SECTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_shock_section(state: AgentState) -> str:
    summary = state.get("shock_summary", "")
    results = state.get("shock_results", {})
    if not results:
        return "\n## Scenario Analysis\nNo scenario data available."

    lines = ["\n## Scenario Analysis\n"]
    if summary:
        lines.append(summary)

    scenarios = results.get("scenarios", {})
    if scenarios:
        lines.append("\n**Scenario Impact Summary:**")
        lines.append("| Scenario | GWP Delta | PIF Delta |")
        lines.append("|----------|-----------|-----------|")
        for name, sc in scenarios.items():
            imp = sc.get("impact", {})
            lines.append(
                f"| {name.replace('_', ' ').title()} | "
                f"{imp.get('gwp_delta_pct', 0):+.1%} (${imp.get('gwp_delta', 0):+,.0f}) | "
                f"{imp.get('pif_delta_pct', 0):+.1%} ({imp.get('pif_delta', 0):+,.0f}) |"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_recommendations(state: AgentState) -> str:
    insights    = state.get("explanations", [])
    ate_results = state.get("ate_results", {})
    opp_summary = state.get("opportunity_summary", {})
    twin_sum    = state.get("twin_summary", "")

    recs = ["\n## AI-Generated Recommendations\n"]

    # From causal findings
    ret_ate = ate_results.get("Retention → GWP", {})
    if ret_ate.get("ate") and ret_ate.get("p_value", 1) < 0.05:
        recs.append(
            f"1. **Retention Program**: A 5% improvement in retention rate is "
            f"causally estimated to add ${ret_ate['ate'] * 5:,.0f} to GWP. "
            "Prioritise renewal outreach for agents with high cancellation rates."
        )

    conv_ate = ate_results.get("Conversion → GWP", {})
    if conv_ate.get("ate") and conv_ate.get("p_value", 1) < 0.05:
        recs.append(
            f"2. **Conversion Uplift**: Improving conversion by 10% is estimated "
            f"to add ${conv_ate['ate'] * 10:,.0f} to GWP. "
            "Focus on training for low-converting agents."
        )

    # From opportunity analysis
    total_upside = opp_summary.get("total_gwp_upside", 0)
    high_pri     = opp_summary.get("high_priority_count", 0)
    if total_upside > 0:
        recs.append(
            f"3. **Priority Agents**: {high_pri} agents represent ${total_upside:,.0f} "
            "in untapped GWP opportunity. Deploy district managers for targeted coaching."
        )

    top_opp_states = opp_summary.get("top_states", [])
    if top_opp_states:
        top_st = top_opp_states[0]
        recs.append(
            f"4. **Geographic Focus**: **{top_st.get('state', 'N/A')}** has the highest "
            f"upside (${top_st.get('total_upside', 0):,.0f}). "
            "Consider state-level initiatives."
        )

    # From digital twin
    if "Optimistic" in twin_sum:
        recs.append(
            "5. **Digital Twin Insight**: The optimistic scenario (retention +5%, "
            "producers +10%) shows meaningful GWP uplift. "
            "Model suggests producer expansion is a high-leverage lever."
        )

    # Generic
    recs.extend([
        "6. **Outlier Follow-up**: Schedule reviews for agents flagged as 'GWP Crash' outliers.",
        "7. **Memory Loop**: Enable monthly actuals ingestion to continuously "
        "reduce forecast bias via the built-in feedback loop.",
    ])

    return "\n".join(recs)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────────────────────────────────────

def reporting_agent(state: AgentState) -> AgentState:
    log.info("[ReportingAgent] Generating final report …")
    errors = list(state.get("errors", []))

    try:
        run_id    = state.get("run_id", "N/A")
        timestamp = state.get("timestamp", datetime.utcnow().isoformat())
        quality   = state.get("data_quality", {})

        header = (
            f"# Farmers EA Topline Analytics — Executive Report\n\n"
            f"**Run ID**: {run_id}  |  "
            f"**Generated**: {timestamp}  |  "
            f"**Agents Completed**: {len(state.get('completed_agents', []))}\n\n"
            f"---\n"
        )

        sections = [
            header,
            "## Executive Summary\n",
            f"This report covers **{quality.get('n_unique_agents', 0)} Exclusive Agents** "
            f"across **{quality.get('policies_rows', 0):,} policy records**. "
            f"Total GWP in the analysis period: **${quality.get('gwp_total', 0):,.0f}**.\n",
            _build_scorecard(state),
            _build_forecast_section(state),
            _build_outlier_section(state),
            _build_opportunity_section(state),
            _build_causal_section(state),
            _build_shock_section(state),
            _build_recommendations(state),
            "\n---\n*Generated by Farmers EA Multi-Agent Analytics System*"
        ]

        full_report = "\n\n".join(s for s in sections if s)

        # Save report
        report_path = OUTPUTS_DIR / f"report_{run_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, "w") as f:
            f.write(full_report)

        # Save state summary as JSON
        json_summary = {
            "run_id":         run_id,
            "timestamp":      timestamp,
            "gwp_total":      quality.get("gwp_total"),
            "pif_total":      quality.get("pif_total"),
            "n_agents":       quality.get("n_unique_agents"),
            "n_forecasts":    len(state.get("forecasts", {})),
            "n_outliers":     state.get("outlier_summary", {}).get("total_flagged"),
            "gwp_upside":     state.get("opportunity_summary", {}).get("total_gwp_upside"),
            "n_insights":     len(state.get("explanations", [])),
            "errors":         state.get("errors", []),
        }
        json_path = OUTPUTS_DIR / f"summary_{run_id}.json"
        with open(json_path, "w") as f:
            json.dump(json_summary, f, indent=2, default=str)

        log.info(f"[ReportingAgent] Report saved: {report_path}")

        state["report"]   = full_report
        state["insights"] = state.get("explanations", [])

    except Exception as e:
        err_msg = f"[ReportingAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        state["report"]   = f"Report generation failed: {e}"
        state["insights"] = []

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("reporting_agent")
    state["completed_agents"] = completed
    log.info("[ReportingAgent] Done.")
    return state
