"""
Synthetic Data Generator for Farmers EA Topline Analytics.

Generates 150,000+ individual policy records with:
- Realistic premium distributions (log-normal)
- Poisson count processes
- Beta-distributed conversion/retention rates
- Seasonality + trend + 3% anomalies
- Hierarchical structure: State → District → Agent → Producer
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

try:
    from config import DATA_CONFIG, STATES, POLICY_TYPES, POLICY_TYPE_WEIGHTS, DATA_DIR
except ImportError:
    from pathlib import Path
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)
    DATA_CONFIG = {
        "n_policies": 150_000,
        "n_agents": 500,
        "n_producers_per_agent": 4,
        "n_states": 12,
        "n_districts_per_state": 8,
        "date_start": "2021-01-01",
        "date_end": "2023-12-31",
        "anomaly_rate": 0.03,
        "random_seed": 42,
    }
    STATES = ["California", "Texas", "Florida", "New York", "Illinois",
              "Ohio", "Georgia", "Arizona", "Colorado", "Washington", "Nevada", "Oregon"]
    POLICY_TYPES = ["Auto", "Home", "Umbrella", "Life", "Commercial"]
    POLICY_TYPE_WEIGHTS = [0.40, 0.30, 0.10, 0.12, 0.08]


# ─────────────────────────────────────────────────────────────────────────────
# AGENT / HIERARCHY METADATA
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_hierarchy(seed: int = 42) -> pd.DataFrame:
    """Build State → District → Agent → Producer hierarchy."""
    rng = np.random.default_rng(seed)
    n_agents = DATA_CONFIG["n_agents"]

    records = []
    agent_id = 1
    for state in STATES:
        n_districts = DATA_CONFIG["n_districts_per_state"]
        for district_num in range(1, n_districts + 1):
            district = f"{state[:3].upper()}-D{district_num:02d}"
            # agents per district varies
            n_agents_in_district = max(1, int(rng.poisson(n_agents / (len(STATES) * n_districts))))
            for _ in range(n_agents_in_district):
                if agent_id > n_agents:
                    break
                n_producers = max(1, int(rng.poisson(DATA_CONFIG["n_producers_per_agent"])))
                # Agent profile parameters (drawn once, fixed per agent)
                records.append({
                    "agent_id": f"EA-{agent_id:04d}",
                    "agent_name": f"Agent_{agent_id:04d}",
                    "state": state,
                    "district": district,
                    "n_producers": n_producers,
                    # Latent quality score drives agent performance
                    "quality_score": float(rng.beta(5, 2)),           # 0–1, higher = better
                    "tenure_years": float(rng.uniform(1, 20)),
                    "base_conversion_rate": float(rng.beta(3, 7)),    # ~30%
                    "base_retention_rate": float(rng.beta(8, 2)),     # ~80%
                    "base_cancellation_rate": float(rng.beta(2, 20)), # ~9%
                    "avg_premium_multiplier": float(rng.lognormal(0, 0.3)),  # relative to base
                })
                agent_id += 1
            if agent_id > n_agents:
                break
        if agent_id > n_agents:
            break

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL FACTOR
# ─────────────────────────────────────────────────────────────────────────────

def seasonal_factor(month: int, amplitude: float = 0.15) -> float:
    """Insurance seasonality: peaks in spring (home renewals) and fall (auto)."""
    return 1.0 + amplitude * np.sin(2 * np.pi * (month - 3) / 12)


def trend_factor(date: pd.Timestamp, start: pd.Timestamp, growth_rate: float = 0.08) -> float:
    """Annual linear growth trend."""
    years_elapsed = (date - start).days / 365.25
    return 1.0 + growth_rate * years_elapsed


# ─────────────────────────────────────────────────────────────────────────────
# POLICY RECORD GENERATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_PREMIUMS = {
    "Auto":       1_400,
    "Home":       1_800,
    "Umbrella":     450,
    "Life":       1_200,
    "Commercial": 5_500,
}


def generate_policies(agents_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_policies = DATA_CONFIG["n_policies"]
    date_start = pd.Timestamp(DATA_CONFIG["date_start"])
    date_end   = pd.Timestamp(DATA_CONFIG["date_end"])
    anomaly_rate = DATA_CONFIG["anomaly_rate"]

    log.info(f"Generating {n_policies:,} policy records …")

    # Sample agents with probability proportional to n_producers (bigger agencies write more)
    agent_weights = agents_df["n_producers"].values.astype(float)
    agent_weights /= agent_weights.sum()

    # Pre-sample all random draws for performance
    chosen_agent_indices = rng.choice(len(agents_df), size=n_policies, p=agent_weights)
    policy_types = rng.choice(POLICY_TYPES, size=n_policies, p=POLICY_TYPE_WEIGHTS)

    # Issue dates: uniform across date range, with slight seasonality bias
    total_days = (date_end - date_start).days
    raw_offsets = rng.integers(0, total_days, size=n_policies)
    issue_dates = [date_start + pd.Timedelta(days=int(d)) for d in raw_offsets]

    # Policy term lengths (months)
    term_months = rng.choice([6, 12, 24], size=n_policies, p=[0.20, 0.70, 0.10])

    # Status: Active / Renewed / Cancelled / Lapsed
    status_outcomes = ["Active", "Renewed", "Cancelled", "Lapsed"]

    records = []
    policy_counter = 1

    for i in tqdm(range(n_policies), desc="Policies", unit="pol", mininterval=2.0):
        agent_row = agents_df.iloc[chosen_agent_indices[i]]
        ptype     = policy_types[i]
        iss_date  = issue_dates[i]
        months    = int(term_months[i])

        # Date features
        sf  = seasonal_factor(iss_date.month)
        tf  = trend_factor(iss_date, date_start)
        eff_date = iss_date
        exp_date = iss_date + pd.DateOffset(months=months)

        # Is new business vs renewal?
        is_new_business = bool(rng.random() > 0.60)  # ~40% renewals in book

        # Premium: log-normal around base, adjusted for agent quality & seasonality
        base = BASE_PREMIUMS[ptype]
        agent_mult = float(agent_row["avg_premium_multiplier"])
        premium_annual = float(
            rng.lognormal(
                np.log(base * agent_mult * sf * tf),
                sigma=0.20,
            )
        )
        premium_annual = max(100.0, premium_annual)

        # Effective premium for term
        premium_earned = premium_annual * months / 12.0

        # Retention / cancellation probabilities
        base_ret  = float(agent_row["base_retention_rate"])
        base_can  = float(agent_row["base_cancellation_rate"])

        # Status assignment
        p_renew = base_ret * (0.9 + 0.1 * rng.random())
        p_cancel = base_can * (0.9 + 0.1 * rng.random())
        p_lapse  = max(0, 1 - p_renew - p_cancel - 0.55)
        p_active = max(0, 1 - p_renew - p_cancel - p_lapse)

        probs = np.array([p_active, p_renew, p_cancel, p_lapse])
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()
        status = rng.choice(status_outcomes, p=probs)

        # Producer assignment
        n_prod = int(agent_row["n_producers"])
        producer_id = f"{agent_row['agent_id']}-P{rng.integers(1, n_prod + 1):02d}"

        # Conversion (for new business quotes)
        conv_rate = float(agent_row["base_conversion_rate"]) * sf

        # Anomaly injection (3%)
        is_anomaly = bool(rng.random() < anomaly_rate)
        if is_anomaly:
            # Either extreme premium or extreme volume signal
            anomaly_type = rng.choice(["premium_spike", "premium_crash", "status_flip"])
            if anomaly_type == "premium_spike":
                premium_earned *= rng.uniform(3.0, 8.0)
            elif anomaly_type == "premium_crash":
                premium_earned *= rng.uniform(0.05, 0.20)
            else:
                status = rng.choice(["Cancelled", "Lapsed"])
        else:
            anomaly_type = None

        # Customer demographics
        cust_age        = int(rng.integers(22, 75))
        years_with_co   = float(rng.exponential(4.0))
        credit_score    = int(np.clip(rng.normal(700, 80), 300, 850))
        discount_pct    = float(np.clip(rng.beta(2, 10), 0, 0.30))

        records.append({
            "policy_id":          f"POL-{policy_counter:07d}",
            "agent_id":           agent_row["agent_id"],
            "agent_name":         agent_row["agent_name"],
            "state":              agent_row["state"],
            "district":           agent_row["district"],
            "producer_id":        producer_id,
            "policy_type":        ptype,
            "issue_date":         iss_date,
            "effective_date":     eff_date,
            "expiration_date":    exp_date,
            "term_months":        months,
            "status":             status,
            "is_new_business":    is_new_business,
            "premium_annual":     round(premium_annual, 2),
            "premium_earned":     round(premium_earned, 2),
            "gwp":                round(premium_earned, 2),      # GWP = earned premium
            "customer_age":       cust_age,
            "years_with_company": round(years_with_co, 1),
            "credit_score":       credit_score,
            "discount_pct":       round(discount_pct, 4),
            "conversion_rate":    round(conv_rate, 4),
            "is_anomaly":         is_anomaly,
            "anomaly_type":       anomaly_type,
            "year":               iss_date.year,
            "month":              iss_date.month,
            "quarter":            iss_date.quarter,
            "year_month":         iss_date.strftime("%Y-%m"),
        })

        policy_counter += 1

    df = pd.DataFrame(records)
    log.info(f"Generated {len(df):,} policy records. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# AGENT-MONTH AGGREGATE (for time-series analytics)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_agent_month(policies_df: pd.DataFrame, agents_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate policy records to agent × month level KPIs."""
    log.info("Aggregating to agent-month level …")

    df = policies_df.copy()
    df["is_active"] = df["status"].isin(["Active", "Renewed"]).astype(int)
    df["is_cancelled"] = (df["status"] == "Cancelled").astype(int)
    df["is_lapsed"]    = (df["status"] == "Lapsed").astype(int)
    df["is_renewal"]   = ((df["status"] == "Renewed") & (~df["is_new_business"])).astype(int)

    agg = (
        df.groupby(["agent_id", "year", "month", "year_month"])
        .agg(
            gwp                = ("gwp",            "sum"),
            pif                = ("is_active",       "sum"),
            new_business_gwp   = ("gwp",             lambda x: x[df.loc[x.index, "is_new_business"]].sum()),
            new_business_count = ("is_new_business", "sum"),
            renewal_count      = ("is_renewal",      "sum"),
            cancellation_count = ("is_cancelled",    "sum"),
            lapse_count        = ("is_lapsed",       "sum"),
            policy_count       = ("policy_id",       "count"),
            avg_premium        = ("gwp",             "mean"),
            avg_discount       = ("discount_pct",    "mean"),
            avg_credit_score   = ("credit_score",    "mean"),
            anomaly_count      = ("is_anomaly",      "sum"),
        )
        .reset_index()
    )

    # Merge agent metadata
    agent_meta = agents_df[[
        "agent_id", "state", "district", "n_producers",
        "quality_score", "tenure_years",
        "base_conversion_rate", "base_retention_rate",
    ]]
    agg = agg.merge(agent_meta, on="agent_id", how="left")

    # Derived KPIs
    agg["renewal_rate"]      = agg["renewal_count"] / (agg["renewal_count"] + agg["new_business_count"] + 1e-9)
    agg["cancellation_rate"] = agg["cancellation_count"] / (agg["policy_count"] + 1e-9)
    agg["conversion_rate"]   = agg["base_conversion_rate"]   # from agent profile
    agg["retention_rate"]    = agg["base_retention_rate"]
    agg["nb_rate"]           = agg["new_business_count"] / (agg["policy_count"] + 1e-9)
    agg["gwp_per_producer"]  = agg["gwp"] / (agg["n_producers"] + 1e-9)
    agg["year_month_dt"]     = pd.to_datetime(agg["year_month"])

    agg = agg.sort_values(["agent_id", "year_month_dt"]).reset_index(drop=True)

    log.info(f"Agent-month aggregate shape: {agg.shape}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# STATE-MONTH AGGREGATE
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_state_month(agent_month_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate agent-month to state-month level."""
    agg = (
        agent_month_df.groupby(["state", "year", "month", "year_month"])
        .agg(
            gwp                = ("gwp",                "sum"),
            pif                = ("pif",                "sum"),
            new_business_gwp   = ("new_business_gwp",   "sum"),
            new_business_count = ("new_business_count", "sum"),
            renewal_count      = ("renewal_count",      "sum"),
            cancellation_count = ("cancellation_count", "sum"),
            agent_count        = ("agent_id",           "nunique"),
            avg_premium        = ("avg_premium",        "mean"),
            avg_conversion     = ("conversion_rate",    "mean"),
            avg_retention      = ("retention_rate",     "mean"),
            anomaly_count      = ("anomaly_count",      "sum"),
        )
        .reset_index()
    )
    agg["year_month_dt"] = pd.to_datetime(agg["year_month"])
    return agg.sort_values(["state", "year_month_dt"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    seed = DATA_CONFIG["random_seed"]
    log.info("=" * 60)
    log.info("Farmers EA Analytics — Synthetic Data Generation")
    log.info("=" * 60)

    # 1. Build agent hierarchy
    log.info("Building agent hierarchy …")
    agents_df = build_agent_hierarchy(seed=seed)
    log.info(f"  Agents: {len(agents_df):,} across {agents_df['state'].nunique()} states, "
             f"{agents_df['district'].nunique()} districts")

    # 2. Generate policy records
    policies_df = generate_policies(agents_df, seed=seed)

    # 3. Aggregate
    agent_month_df = aggregate_agent_month(policies_df, agents_df)
    state_month_df = aggregate_state_month(agent_month_df)

    # 4. Save
    out = DATA_DIR
    agents_path       = out / "agents.parquet"
    policies_path     = out / "policies.parquet"
    agent_month_path  = out / "agent_month.parquet"
    state_month_path  = out / "state_month.parquet"

    agents_df.to_parquet(agents_path, index=False)
    policies_df.to_parquet(policies_path, index=False)
    agent_month_df.to_parquet(agent_month_path, index=False)
    state_month_df.to_parquet(state_month_path, index=False)

    log.info("\n✅ Data saved:")
    log.info(f"   {agents_path}       → {len(agents_df):,} rows")
    log.info(f"   {policies_path}     → {len(policies_df):,} rows")
    log.info(f"   {agent_month_path}  → {len(agent_month_df):,} rows")
    log.info(f"   {state_month_path}  → {len(state_month_df):,} rows")

    # 5. Summary stats
    log.info("\n📊 Summary Statistics:")
    log.info(f"   Total GWP:          ${policies_df['gwp'].sum():>14,.0f}")
    log.info(f"   Total Policies:     {len(policies_df):>14,}")
    log.info(f"   Avg Annual Premium: ${policies_df['premium_annual'].mean():>14,.2f}")
    log.info(f"   Anomalies Injected: {policies_df['is_anomaly'].sum():>14,} "
             f"({policies_df['is_anomaly'].mean():.1%})")
    log.info(f"   New Business:       {policies_df['is_new_business'].mean():.1%}")
    log.info(f"   Status Distribution:\n{policies_df['status'].value_counts(normalize=True).to_string()}")

    return agents_df, policies_df, agent_month_df, state_month_df


if __name__ == "__main__":
    main()
