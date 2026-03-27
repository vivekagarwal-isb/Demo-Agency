"""
Central configuration for Farmers EA Topline Multi-Agent Analytics System.
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = BASE_DIR / "memory_store"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in [DATA_DIR, MEMORY_DIR, OUTPUTS_DIR]:
    d.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
DATA_CONFIG = {
    "n_policies": 150_000,          # total policy records
    "n_agents": 500,                 # Exclusive Agents
    "n_producers_per_agent": 4,      # avg producers under each EA
    "n_states": 12,
    "n_districts_per_state": 8,
    "date_start": "2021-01-01",
    "date_end": "2023-12-31",
    "anomaly_rate": 0.03,            # 3% anomalies
    "random_seed": 42,
}

STATES = [
    "California", "Texas", "Florida", "New York", "Illinois",
    "Ohio", "Georgia", "Arizona", "Colorado", "Washington",
    "Nevada", "Oregon",
]

POLICY_TYPES = ["Auto", "Home", "Umbrella", "Life", "Commercial"]

POLICY_TYPE_WEIGHTS = [0.40, 0.30, 0.10, 0.12, 0.08]

# ─────────────────────────────────────────────
# KPI TARGETS
# ─────────────────────────────────────────────
KPI_CONFIG = {
    "gwp_target_annual": 150_000_000,   # $150M total GWP target
    "pif_target": 85_000,               # Policies in Force target
    "nb_rate_target": 0.25,             # 25% new business rate
    "renewal_rate_target": 0.85,        # 85% renewal rate
    "cancellation_rate_target": 0.05,   # 5% cancellation rate
}

# ─────────────────────────────────────────────
# FORECASTING
# ─────────────────────────────────────────────
FORECAST_CONFIG = {
    "horizon_months": 12,
    "prophet_changepoint_prior_scale": 0.05,
    "xgb_n_estimators": 200,
    "xgb_max_depth": 6,
    "xgb_learning_rate": 0.05,
    "train_test_split": 0.8,
}

# ─────────────────────────────────────────────
# OUTLIER DETECTION
# ─────────────────────────────────────────────
OUTLIER_CONFIG = {
    "contamination": 0.03,
    "z_score_threshold": 3.0,
    "iqr_multiplier": 1.5,
}

# ─────────────────────────────────────────────
# CAUSAL INFERENCE
# ─────────────────────────────────────────────
CAUSAL_CONFIG = {
    "treatment_vars": ["retention_rate", "conversion_rate", "discount_rate", "producer_count"],
    "outcome_vars": ["gwp", "pif"],
    "n_bootstrap": 100,
    "alpha": 0.05,
}

# ─────────────────────────────────────────────
# DIGITAL TWIN
# ─────────────────────────────────────────────
TWIN_CONFIG = {
    "simulation_steps": 24,          # months to simulate
    "pricing_sensitivity": -0.8,     # price elasticity
    "competition_sensitivity": -0.4,
    "discount_sensitivity": 0.3,
    "retention_gwp_multiplier": 1.2,
}

# ─────────────────────────────────────────────
# MEMORY / LEARNING
# ─────────────────────────────────────────────
MEMORY_CONFIG = {
    "db_path": str(MEMORY_DIR / "memory.db"),
    "faiss_index_path": str(MEMORY_DIR / "faiss_index"),
    "embedding_dim": 64,
    "max_memory_entries": 10_000,
    "learning_rate": 0.01,
    "feedback_window_months": 3,
}

# ─────────────────────────────────────────────
# SHOCK SIMULATION DEFAULTS
# ─────────────────────────────────────────────
SHOCK_DEFAULTS = {
    "conversion_delta": 0.0,      # ±30%
    "retention_delta": 0.0,       # ±20%
    "pricing_delta": 0.0,         # ±15%
    "producer_count_delta": 0.0,  # ±40%
    "discount_delta": 0.0,        # ±20%
}

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
DASHBOARD_CONFIG = {
    "title": "Farmers EA Topline Analytics",
    "theme": "dark",
    "primary_color": "#E8290B",    # Farmers red
    "secondary_color": "#1A3C5E",  # Farmers navy
    "accent_color": "#F5A623",     # Gold accent
    "background": "#0D1117",
    "surface": "#161B22",
    "card_bg": "#21262D",
}

# ─────────────────────────────────────────────
# AGENT ROUTING
# ─────────────────────────────────────────────
AGENT_SEQUENCE = [
    "data_agent",
    "feature_agent",
    "forecasting_agent",
    "outlier_agent",
    "opportunity_agent",
    "causal_agent",
    "digital_twin_agent",
    "shock_agent",
    "memory_agent",
    "explanation_agent",
    "reporting_agent",
    "END",
]
