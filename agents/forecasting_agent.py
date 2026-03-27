"""
Forecasting Agent — Time-series prediction of GWP and PIF.

Models:
1. Prophet (trend + seasonality + holidays)
2. XGBoost (lag-based regression)
Ensemble: weighted average of both.

Outputs:
- 12-month ahead forecast per aggregate level (portfolio, state)
- Accuracy metrics on held-out period (MAPE, RMSE)
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from agents.state import AgentState

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    from config import FORECAST_CONFIG
except ImportError:
    FORECAST_CONFIG = {
        "horizon_months": 12,
        "prophet_changepoint_prior_scale": 0.05,
        "xgb_n_estimators": 200,
        "xgb_max_depth": 6,
        "xgb_learning_rate": 0.05,
        "train_test_split": 0.8,
    }

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    log.warning("Prophet not installed — using XGBoost only.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    log.warning("XGBoost not installed — using trend extrapolation.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build time-series at portfolio level
# ─────────────────────────────────────────────────────────────────────────────

def _portfolio_ts(df: pd.DataFrame) -> pd.DataFrame:
    ts = (
        df.groupby("year_month_dt")
        .agg(gwp=("gwp", "sum"), pif=("pif", "sum"))
        .reset_index()
        .rename(columns={"year_month_dt": "ds"})
        .sort_values("ds")
    )
    return ts


def _state_ts(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    result = {}
    for state, grp in df.groupby("state"):
        ts = (
            grp.groupby("year_month_dt")
            .agg(gwp=("gwp", "sum"), pif=("pif", "sum"))
            .reset_index()
            .rename(columns={"year_month_dt": "ds"})
            .sort_values("ds")
        )
        result[state] = ts
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PROPHET FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def _prophet_forecast(ts: pd.DataFrame, metric: str, horizon: int) -> pd.DataFrame:
    if not PROPHET_AVAILABLE or len(ts) < 12:
        return pd.DataFrame()

    prophet_df = ts[["ds", metric]].rename(columns={metric: "y"})
    model = Prophet(
        changepoint_prior_scale=FORECAST_CONFIG["prophet_changepoint_prior_scale"],
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
    forecast = forecast.rename(columns={
        "yhat": f"{metric}_prophet",
        "yhat_lower": f"{metric}_prophet_lower",
        "yhat_upper": f"{metric}_prophet_upper",
    })
    return forecast


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def _build_xgb_features(ts: pd.DataFrame, metric: str, n_lags: int = 6) -> pd.DataFrame:
    df = ts.copy()
    df["month_num"]  = df["ds"].dt.month
    df["year_num"]   = df["ds"].dt.year
    df["sin_month"]  = np.sin(2 * np.pi * df["month_num"] / 12)
    df["cos_month"]  = np.cos(2 * np.pi * df["month_num"] / 12)
    df["trend"]      = np.arange(len(df))
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df[metric].shift(lag)
    df = df.dropna()
    return df


def _xgb_forecast(ts: pd.DataFrame, metric: str, horizon: int) -> pd.DataFrame:
    if not XGB_AVAILABLE or len(ts) < 15:
        return pd.DataFrame()

    feat_df = _build_xgb_features(ts, metric)
    if len(feat_df) < 10:
        return pd.DataFrame()

    feature_cols = ["month_num", "year_num", "sin_month", "cos_month", "trend"] + \
                   [f"lag_{i}" for i in range(1, 7)]
    feature_cols = [c for c in feature_cols if c in feat_df.columns]

    split = int(len(feat_df) * FORECAST_CONFIG["train_test_split"])
    X_train, y_train = feat_df[feature_cols].iloc[:split], feat_df[metric].iloc[:split]

    model = xgb.XGBRegressor(
        n_estimators=FORECAST_CONFIG["xgb_n_estimators"],
        max_depth=FORECAST_CONFIG["xgb_max_depth"],
        learning_rate=FORECAST_CONFIG["xgb_learning_rate"],
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)

    # Recursive multi-step forecast
    last_vals = list(ts[metric].values[-6:])
    last_date = ts["ds"].max()
    preds = []
    for step in range(horizon):
        next_date = last_date + pd.DateOffset(months=step + 1)
        row = {
            "month_num":  next_date.month,
            "year_num":   next_date.year,
            "sin_month":  np.sin(2 * np.pi * next_date.month / 12),
            "cos_month":  np.cos(2 * np.pi * next_date.month / 12),
            "trend":      len(ts) + step,
        }
        for i, lag in enumerate(range(1, 7)):
            idx = len(last_vals) - lag
            row[f"lag_{lag}"] = last_vals[idx] if idx >= 0 else last_vals[0]
        x = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(x)[0])
        preds.append({"ds": next_date, f"{metric}_xgb": max(0, pred)})
        last_vals.append(pred)

    return pd.DataFrame(preds)


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE TREND FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _trend_forecast(ts: pd.DataFrame, metric: str, horizon: int) -> pd.DataFrame:
    """Linear trend extrapolation as last-resort fallback."""
    y = ts[metric].values
    x = np.arange(len(y))
    if len(y) > 1:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0, y[0] if len(y) else 0

    last_date = ts["ds"].max()
    rows = []
    for step in range(1, horizon + 1):
        pred = max(0, intercept + slope * (len(y) + step))
        rows.append({"ds": last_date + pd.DateOffset(months=step),
                     f"{metric}_trend": pred})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    if len(actual) == 0 or len(predicted) == 0:
        return {"mape": None, "rmse": None}
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))) if mask.sum() > 0 else None
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return {"mape": mape, "rmse": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def _ensemble(prophet_df: pd.DataFrame, xgb_df: pd.DataFrame,
              trend_df: pd.DataFrame, metric: str, horizon: int) -> pd.DataFrame:
    """Combine forecasts: prefer XGB+Prophet ensemble, fallback to trend."""
    if len(prophet_df) == 0 and len(xgb_df) == 0:
        return trend_df.rename(columns={f"{metric}_trend": f"{metric}_forecast"})

    frames = []
    if len(prophet_df) > 0:
        frames.append(prophet_df.set_index("ds")[f"{metric}_prophet"])
    if len(xgb_df) > 0:
        frames.append(xgb_df.set_index("ds")[f"{metric}_xgb"])

    combined = pd.concat(frames, axis=1)
    combined[f"{metric}_forecast"] = combined.mean(axis=1)
    combined = combined.reset_index()
    combined = combined[["ds", f"{metric}_forecast"]].tail(horizon)

    # Add uncertainty bounds (±10% as approximate CI)
    combined[f"{metric}_lower"] = combined[f"{metric}_forecast"] * 0.90
    combined[f"{metric}_upper"] = combined[f"{metric}_forecast"] * 1.10

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────────────────────────────────────

def forecasting_agent(state: AgentState) -> AgentState:
    log.info("[ForecastingAgent] Starting …")
    errors = list(state.get("errors", []))
    horizon = FORECAST_CONFIG["horizon_months"]

    try:
        df = state.get("features_df") or state.get("agent_month_df")
        df = df.copy()

        forecasts: Dict = {}
        accuracy:  Dict = {}

        # Ensure datetime
        if "year_month_dt" not in df.columns:
            df["year_month_dt"] = pd.to_datetime(df["year_month"])

        # ── Portfolio-level forecast ─────────────────────────────────────────
        portfolio_ts = _portfolio_ts(df)
        log.info(f"[ForecastingAgent] Portfolio TS: {len(portfolio_ts)} months")

        for metric in ["gwp", "pif"]:
            prophet_fc  = _prophet_forecast(portfolio_ts, metric, horizon)
            xgb_fc      = _xgb_forecast(portfolio_ts, metric, horizon)
            trend_fc    = _trend_forecast(portfolio_ts, metric, horizon)
            ensemble_fc = _ensemble(prophet_fc, xgb_fc, trend_fc, metric, horizon)
            forecasts[f"portfolio_{metric}"] = ensemble_fc

            # Accuracy on training data (last-6m holdout)
            if len(portfolio_ts) >= 12:
                holdout = portfolio_ts.tail(6)
                prophet_hold = _prophet_forecast(
                    portfolio_ts.iloc[:-6], metric, 6
                )
                if len(prophet_hold) == 6:
                    acc = _accuracy(
                        holdout[metric].values,
                        prophet_hold[f"{metric}_prophet"].values,
                    )
                    accuracy[f"portfolio_{metric}"] = acc
                    log.info(f"[ForecastingAgent] {metric} accuracy: "
                             f"MAPE={acc['mape']:.2%}, RMSE={acc['rmse']:,.0f}" if acc['mape'] else
                             f"[ForecastingAgent] {metric} accuracy: insufficient data")

        # ── State-level forecast ─────────────────────────────────────────────
        state_ts_dict = _state_ts(df)
        for st_name, ts in state_ts_dict.items():
            if len(ts) < 6:
                continue
            for metric in ["gwp", "pif"]:
                xgb_fc   = _xgb_forecast(ts, metric, horizon)
                trend_fc = _trend_forecast(ts, metric, horizon)
                ens_fc   = _ensemble(pd.DataFrame(), xgb_fc, trend_fc, metric, horizon)
                forecasts[f"state_{st_name}_{metric}"] = ens_fc

        log.info(f"[ForecastingAgent] Generated {len(forecasts)} forecast series.")

        state["forecasts"]       = forecasts
        state["forecast_accuracy"] = accuracy
        state["forecast_horizon"]  = horizon

    except Exception as e:
        err_msg = f"[ForecastingAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        state["forecasts"]       = {}
        state["forecast_accuracy"] = {}

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("forecasting_agent")
    state["completed_agents"] = completed
    log.info("[ForecastingAgent] Done.")
    return state
