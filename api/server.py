"""
FastAPI Server — Farmers EA Topline Multi-Agent Analytics System.

Endpoints:
  GET  /                          → Serve enterprise dashboard HTML
  POST /api/pipeline/run          → Trigger full multi-agent pipeline
  GET  /api/pipeline/status       → Current run status
  GET  /api/data/kpis             → Portfolio KPI tiles
  GET  /api/data/portfolio-trend  → Monthly GWP + PIF time series
  GET  /api/data/state-map        → State-level GWP aggregates
  GET  /api/data/forecasts        → 12-month GWP/PIF forecast
  GET  /api/data/outliers         → Outlier agent-month records
  GET  /api/data/opportunities    → Ranked opportunity table
  GET  /api/data/leaderboard      → Top 20 agents by composite score
  GET  /api/data/causal           → ATE / CATE results + graph
  GET  /api/data/twin             → Digital twin simulation scenarios
  POST /api/simulate/shock        → Re-run shock agent with custom params
  GET  /api/data/shap             → SHAP feature importance
  GET  /api/report                → Full markdown report
  WS   /ws/pipeline               → Real-time pipeline event stream
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Farmers EA Analytics API",
    description="Multi-Agent Topline Analytics for Farmers Exclusive Agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── In-memory pipeline state cache ──────────────────────────────────────────
_pipeline_cache: Dict[str, Any] = {}
_pipeline_status = {
    "state":    "idle",        # idle | running | complete | error
    "progress": 0,
    "current_agent": None,
    "run_id":   None,
    "error":    None,
    "started_at": None,
    "completed_at": None,
}
_ws_clients: List[WebSocket] = []


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ShockParams(BaseModel):
    conversion_delta:     float = 0.0   # ±0.30
    retention_delta:      float = 0.0   # ±0.20
    pricing_delta:        float = 0.0   # ±0.15
    producer_count_delta: float = 0.0   # ±0.40
    discount_delta:       float = 0.0   # ±0.20


class PipelineRunRequest(BaseModel):
    shock_params:     Optional[Dict] = None
    forecast_horizon: int = 12


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET BROADCAST
# ─────────────────────────────────────────────────────────────────────────────

async def _broadcast(msg: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _sync_broadcast(msg: dict):
    """Thread-safe broadcast from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast(msg), loop)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER (background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_thread(shock_params: dict, horizon: int):
    """Execute the multi-agent pipeline in a background thread."""
    global _pipeline_cache, _pipeline_status

    _pipeline_status.update({
        "state": "running",
        "progress": 0,
        "current_agent": "Initialising",
        "started_at": datetime.utcnow().isoformat(),
        "error": None,
    })
    _sync_broadcast({"type": "status", "data": _pipeline_status})

    try:
        from graph.workflow import run_pipeline

        # Monkey-patch agent progress reporting
        _agent_order = [
            "data_agent", "feature_agent", "forecasting_agent",
            "outlier_agent", "opportunity_agent", "causal_agent",
            "digital_twin_agent", "shock_agent", "memory_agent",
            "explanation_agent", "reporting_agent",
        ]

        # Wrap each agent to emit progress events
        import agents.data_agent as da
        import agents.feature_agent as fa
        import agents.forecasting_agent as fca
        import agents.outlier_agent as oa
        import agents.opportunity_agent as oppa
        import agents.causal_agent as ca
        import agents.digital_twin_agent as dta
        import agents.shock_agent as sha
        import agents.memory_agent as ma
        import agents.explanation_agent as ea
        import agents.reporting_agent as ra

        agent_modules = [da, fa, fca, oa, oppa, ca, dta, sha, ma, ea, ra]
        agent_fns     = [
            "data_agent", "feature_agent", "forecasting_agent",
            "outlier_agent", "opportunity_agent", "causal_agent",
            "digital_twin_agent", "shock_agent", "memory_agent",
            "explanation_agent", "reporting_agent",
        ]
        total = len(agent_fns)

        originals = {}
        for idx, (mod, fn_name) in enumerate(zip(agent_modules, agent_fns)):
            orig = getattr(mod, fn_name)
            originals[fn_name] = orig

            def make_wrapper(fn, name, step):
                def wrapper(state):
                    pct = int((step / total) * 100)
                    _pipeline_status.update({
                        "current_agent": name.replace("_", " ").title(),
                        "progress": pct,
                    })
                    _sync_broadcast({
                        "type": "progress",
                        "agent": name,
                        "progress": pct,
                        "message": f"Running {name.replace('_', ' ').title()} …",
                    })
                    result = fn(state)
                    return result
                return wrapper

            setattr(mod, fn_name, make_wrapper(orig, fn_name, idx + 1))

        # Run
        final_state = run_pipeline(shock_params=shock_params, forecast_horizon=horizon)

        # Restore originals
        for idx, (mod, fn_name) in enumerate(zip(agent_modules, agent_fns)):
            setattr(mod, fn_name, originals[fn_name])

        # Cache results
        _pipeline_cache = _serialise_state(final_state)
        _pipeline_status.update({
            "state":        "complete",
            "progress":     100,
            "current_agent": "Done",
            "run_id":        final_state.get("run_id"),
            "completed_at":  datetime.utcnow().isoformat(),
        })
        _sync_broadcast({"type": "complete", "run_id": final_state.get("run_id")})
        log.info(f"[Server] Pipeline complete: {final_state.get('run_id')}")

    except Exception as e:
        log.error(f"[Server] Pipeline error: {e}", exc_info=True)
        _pipeline_status.update({"state": "error", "error": str(e)})
        _sync_broadcast({"type": "error", "message": str(e)})


def _serialise_state(state: dict) -> dict:
    """Convert DataFrames and numpy types to JSON-safe structures."""
    out = {}
    for k, v in state.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.where(pd.notnull(v), None).to_dict("records")
        elif isinstance(v, (np.integer, np.int64)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _serialise_dict(v)
        else:
            out[k] = v
    return out


def _serialise_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.where(pd.notnull(v), None).to_dict("records")
        elif isinstance(v, (np.integer, np.int64)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _serialise_dict(v)
        else:
            out[k] = v
    return out


def _require_cache():
    """Raise 503 if pipeline hasn't run yet."""
    if not _pipeline_cache:
        raise HTTPException(
            status_code=503,
            detail="Pipeline has not run yet. POST /api/pipeline/run first.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the enterprise dashboard HTML."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found. Check static/index.html</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


# ── Pipeline control ─────────────────────────────────────────────────────────

@app.post("/api/pipeline/run")
async def run_pipeline_endpoint(
    req: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    if _pipeline_status["state"] == "running":
        return JSONResponse({"message": "Pipeline already running", "status": _pipeline_status})

    background_tasks.add_task(
        _run_pipeline_thread,
        req.shock_params or {},
        req.forecast_horizon,
    )
    return {"message": "Pipeline started", "status": "running"}


@app.get("/api/pipeline/status")
async def pipeline_status():
    return _pipeline_status


# ── KPIs ─────────────────────────────────────────────────────────────────────

@app.get("/api/data/kpis")
async def get_kpis():
    _require_cache()
    q = _pipeline_cache.get("data_quality", {})
    opp = _pipeline_cache.get("opportunity_summary", {})
    outlier = _pipeline_cache.get("outlier_summary", {})
    acc = _pipeline_cache.get("forecast_accuracy", {})

    gwp   = q.get("gwp_total", 0) or 0
    pif   = q.get("pif_total", 0) or 0
    mape  = (acc.get("portfolio_gwp", {}) or {}).get("mape")
    upside = opp.get("total_gwp_upside", 0) or 0

    try:
        from config import KPI_CONFIG
        gwp_target = KPI_CONFIG["gwp_target_annual"]
        pif_target = KPI_CONFIG["pif_target"]
    except ImportError:
        gwp_target, pif_target = 150_000_000, 85_000

    return {
        "gwp":              round(float(gwp), 2),
        "gwp_target":       gwp_target,
        "gwp_attainment":   round(float(gwp) / gwp_target * 100, 1),
        "pif":              round(float(pif), 2),
        "pif_target":       pif_target,
        "pif_attainment":   round(float(pif) / pif_target * 100, 1),
        "n_agents":         q.get("n_unique_agents", 0),
        "n_policies":       q.get("policies_rows", 0),
        "renewal_rate":     round(float(q.get("avg_renewal_rate", 0) or 0), 4),
        "cancellation_rate": round(float(q.get("avg_cancellation_rate", 0) or 0), 4),
        "gwp_upside":       round(float(upside), 2),
        "high_priority_agents": opp.get("high_priority_count", 0),
        "outliers_flagged": outlier.get("total_flagged", 0),
        "forecast_mape":    round(float(mape), 4) if mape else None,
        "anomaly_rate":     round(float(q.get("anomaly_rate", 0) or 0), 4),
    }


# ── Portfolio trend ──────────────────────────────────────────────────────────

@app.get("/api/data/portfolio-trend")
async def get_portfolio_trend():
    _require_cache()
    agent_month = _pipeline_cache.get("agent_month_df", [])
    if not agent_month:
        return []

    df = pd.DataFrame(agent_month)
    if df.empty:
        return []

    monthly = (
        df.groupby("year_month")
        .agg(
            gwp=("gwp", "sum"),
            pif=("pif", "sum"),
            new_business=("new_business_count", "sum"),
            renewals=("renewal_count", "sum"),
            cancellations=("cancellation_count", "sum"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    monthly["gwp"] = monthly["gwp"].round(2)
    return monthly.replace({np.nan: None}).to_dict("records")


# ── State map ────────────────────────────────────────────────────────────────

@app.get("/api/data/state-map")
async def get_state_map():
    _require_cache()
    state_month = _pipeline_cache.get("state_month_df", [])
    if not state_month:
        return []

    df = pd.DataFrame(state_month)
    if df.empty:
        return []

    state_agg = (
        df.groupby("state")
        .agg(
            total_gwp=("gwp", "sum"),
            total_pif=("pif", "sum"),
            avg_cancellation=("cancellation_count", "mean"),
            agent_count=("agent_count", "mean"),
        )
        .reset_index()
    )
    state_agg["total_gwp"] = state_agg["total_gwp"].round(2)
    return state_agg.replace({np.nan: None}).to_dict("records")


# ── Forecasts ────────────────────────────────────────────────────────────────

@app.get("/api/data/forecasts")
async def get_forecasts():
    _require_cache()
    forecasts_raw = _pipeline_cache.get("forecasts", {})
    result = {}
    for k, v in forecasts_raw.items():
        if isinstance(v, list):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = v
    return result


# ── Outliers ─────────────────────────────────────────────────────────────────

@app.get("/api/data/outliers")
async def get_outliers():
    _require_cache()
    outliers = _pipeline_cache.get("outliers_df", [])
    summary  = _pipeline_cache.get("outlier_summary", {})
    df = pd.DataFrame(outliers) if outliers else pd.DataFrame()

    if not df.empty:
        cols = ["agent_id", "agent_name", "state", "district",
                "year_month", "gwp", "pif", "cancellation_rate",
                "outlier_type", "outlier_score"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].nlargest(200, "outlier_score") if "outlier_score" in df.columns else df.head(200)

    return {
        "records": df.replace({np.nan: None}).to_dict("records"),
        "summary": summary,
    }


# ── Opportunities ────────────────────────────────────────────────────────────

@app.get("/api/data/opportunities")
async def get_opportunities():
    _require_cache()
    opp = _pipeline_cache.get("opportunities_df", [])
    summary = _pipeline_cache.get("opportunity_summary", {})
    df = pd.DataFrame(opp) if opp else pd.DataFrame()

    if not df.empty:
        cols = ["agent_id", "agent_name", "state", "district",
                "gwp", "pif", "composite_score", "gwp_upside_est",
                "opportunity_class", "rank",
                "gap_score", "momentum_score", "conv_score",
                "leakage_score", "capacity_score"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].head(100)

    return {
        "records": df.replace({np.nan: None}).to_dict("records"),
        "summary": summary,
    }


# ── Leaderboard ──────────────────────────────────────────────────────────────

@app.get("/api/data/leaderboard")
async def get_leaderboard():
    _require_cache()
    opp = _pipeline_cache.get("opportunities_df", [])
    df = pd.DataFrame(opp) if opp else pd.DataFrame()

    if df.empty:
        return []

    cols = ["rank", "agent_id", "agent_name", "state", "district",
            "gwp", "composite_score", "gwp_upside_est", "opportunity_class"]
    cols = [c for c in cols if c in df.columns]
    top = df.sort_values("rank").head(20)[cols] if "rank" in df.columns else df.head(20)[cols]
    return top.replace({np.nan: None}).to_dict("records")


# ── Causal results ───────────────────────────────────────────────────────────

@app.get("/api/data/causal")
async def get_causal():
    _require_cache()
    return {
        "ate_results":   _pipeline_cache.get("ate_results", {}),
        "cate_results":  _pipeline_cache.get("cate_results", {}),
        "causal_graph":  _pipeline_cache.get("causal_graph", {}),
        "causal_summary": _pipeline_cache.get("causal_summary", ""),
    }


# ── Digital Twin ─────────────────────────────────────────────────────────────

@app.get("/api/data/twin")
async def get_twin():
    _require_cache()
    twin_sim = _pipeline_cache.get("twin_simulation", [])
    df = pd.DataFrame(twin_sim) if twin_sim else pd.DataFrame()

    if df.empty:
        return {"scenarios": [], "summary": _pipeline_cache.get("twin_summary", "")}

    if "month" in df.columns:
        df["month"] = df["month"].astype(str)

    monthly = (
        df.groupby(["scenario", "month"])
        .agg(total_gwp=("total_gwp", "sum"), total_pif=("total_pif", "sum"))
        .reset_index()
        .sort_values(["scenario", "month"])
    )

    return {
        "scenarios": monthly.replace({np.nan: None}).to_dict("records"),
        "summary":   _pipeline_cache.get("twin_summary", ""),
    }


# ── Shock simulation ─────────────────────────────────────────────────────────

@app.post("/api/simulate/shock")
async def simulate_shock(params: ShockParams):
    """Re-run shock agent with custom parameters. Fast (no full pipeline)."""
    if not _pipeline_cache:
        raise HTTPException(status_code=503, detail="Run pipeline first.")

    from agents.shock_agent import shock_agent
    from agents.state import AgentState

    state: AgentState = {
        "agent_month_df":  pd.DataFrame(_pipeline_cache.get("agent_month_df", [])),
        "features_df":     pd.DataFrame(_pipeline_cache.get("features_df", []) or
                                        _pipeline_cache.get("agent_month_df", [])),
        "shock_params": {
            "conversion_delta":     params.conversion_delta,
            "retention_delta":      params.retention_delta,
            "pricing_delta":        params.pricing_delta,
            "producer_count_delta": params.producer_count_delta,
            "discount_delta":       params.discount_delta,
        },
        "completed_agents": [],
        "errors": [],
        "next_agent": "shock_agent",
        "run_id": "shock_sim",
        "timestamp": datetime.utcnow().isoformat(),
    }

    result = shock_agent(state)
    shock_results = _serialise_dict(result.get("shock_results", {}))
    return {
        "shock_results": shock_results,
        "shock_summary": result.get("shock_summary", ""),
    }


# ── SHAP feature importance ──────────────────────────────────────────────────

@app.get("/api/data/shap")
async def get_shap():
    _require_cache()
    sv = _pipeline_cache.get("shap_values", {})
    gi = sv.get("global_importance", {}) if isinstance(sv, dict) else {}
    top = sorted(gi.items(), key=lambda x: x[1], reverse=True)[:20]
    return {
        "global_importance": [{"feature": k, "importance": round(float(v), 6)} for k, v in top],
        "insights":          _pipeline_cache.get("explanations", []),
    }


# ── Insights ─────────────────────────────────────────────────────────────────

@app.get("/api/data/insights")
async def get_insights():
    _require_cache()
    return {
        "insights":          _pipeline_cache.get("explanations", []),
        "learning_update":   _pipeline_cache.get("learning_update", {}),
        "memory_retrieved":  (_pipeline_cache.get("memory_retrieved") or [])[:5],
    }


# ── Full report ──────────────────────────────────────────────────────────────

@app.get("/api/report")
async def get_report():
    _require_cache()
    return {"report": _pipeline_cache.get("report", "No report available.")}


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/pipeline")
async def ws_pipeline(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    # Send current status on connect
    await websocket.send_json({"type": "status", "data": _pipeline_status})
    try:
        while True:
            await asyncio.sleep(30)  # keep alive
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP: auto-generate data if not present
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        from config import DATA_DIR
        data_dir = DATA_DIR
    except ImportError:
        data_dir = ROOT / "data"

    if not (data_dir / "policies.parquet").exists():
        log.info("[Server] Generating synthetic data on startup …")
        import generate_data
        generate_data.main()
        log.info("[Server] Data generation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
