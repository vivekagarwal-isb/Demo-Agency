"""
Digital Twin Agent — Builds and runs a simulation model of the agency network.

Architecture:
  State → District → Agent → Producer (hierarchical entities)

Each entity has:
- Behaviour parameters (sensitivity to pricing, competition, discount)
- State variables (GWP, PIF, conversion, retention)
- Transition rules (monthly update equations)

Simulation capabilities:
- Macro shocks (market downturn, rate hardening)
- Policy changes (new product, producer incentives)
- Competitive pressure (price cuts by competitors)

Output: 24-month simulated GWP/PIF trajectory under baseline + scenarios.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agents.state import AgentState

log = logging.getLogger(__name__)

try:
    from config import TWIN_CONFIG
except ImportError:
    TWIN_CONFIG = {
        "simulation_steps": 24,
        "pricing_sensitivity": -0.8,
        "competition_sensitivity": -0.4,
        "discount_sensitivity": 0.3,
        "retention_gwp_multiplier": 1.2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProducerEntity:
    producer_id: str
    agent_id: str
    base_new_policies_month: float = 2.0
    conversion_rate: float = 0.30
    price_sensitivity: float = -0.8
    discount_sensitivity: float = 0.25

    def monthly_new_policies(self, pricing_shock: float = 0.0,
                              discount_shock: float = 0.0,
                              competition_shock: float = 0.0) -> float:
        multiplier = (
            1.0
            + self.price_sensitivity * pricing_shock
            + self.discount_sensitivity * discount_shock
            + TWIN_CONFIG["competition_sensitivity"] * competition_shock
        )
        multiplier = max(0.1, multiplier)
        rate = self.conversion_rate * multiplier
        return max(0, np.random.poisson(self.base_new_policies_month * rate))


@dataclass
class AgentEntity:
    agent_id: str
    district: str
    state: str
    producers: List[ProducerEntity] = field(default_factory=list)
    base_gwp_month: float = 50_000.0
    retention_rate: float = 0.85
    cancellation_rate: float = 0.05
    avg_premium: float = 1_400.0
    quality_score: float = 0.7
    pif: float = 200.0

    def step(self, pricing_shock: float = 0.0, discount_shock: float = 0.0,
             competition_shock: float = 0.0, retention_shock: float = 0.0,
             producer_count_shock: float = 0.0) -> Dict:
        """Advance agent state by one month."""
        effective_retention = np.clip(
            self.retention_rate + retention_shock, 0.0, 1.0
        )
        effective_cancellation = np.clip(
            self.cancellation_rate - 0.5 * retention_shock, 0.001, 0.5
        )

        # New business from producers
        n_active_producers = max(
            1, int(len(self.producers) * (1 + producer_count_shock))
        )
        new_policies = 0
        for prod in self.producers[:n_active_producers]:
            new_policies += prod.monthly_new_policies(
                pricing_shock, discount_shock, competition_shock
            )

        # PIF dynamics
        renewed = self.pif * effective_retention
        cancelled = self.pif * effective_cancellation
        self.pif = max(0, renewed - cancelled + new_policies)

        # GWP
        renewal_gwp     = renewed * self.avg_premium / 12.0
        new_gwp         = new_policies * self.avg_premium
        cancellation_adj = cancelled * self.avg_premium / 12.0
        gwp_month = max(0, renewal_gwp + new_gwp - cancellation_adj)

        # Pricing adjustment on avg premium
        price_adj = self.avg_premium * (1 + pricing_shock * 0.5)
        gwp_month *= (price_adj / self.avg_premium)

        return {
            "agent_id":    self.agent_id,
            "gwp":         gwp_month,
            "pif":         self.pif,
            "new_policies": new_policies,
            "renewed":     renewed,
            "cancelled":   cancelled,
        }


@dataclass
class DistrictEntity:
    district: str
    state: str
    agents: List[AgentEntity] = field(default_factory=list)

    def step(self, **shocks) -> List[Dict]:
        return [agent.step(**shocks) for agent in self.agents]


@dataclass
class StateEntity:
    state: str
    districts: List[DistrictEntity] = field(default_factory=list)

    def step(self, **shocks) -> List[Dict]:
        results = []
        for d in self.districts:
            results.extend(d.step(**shocks))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# DIGITAL TWIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class DigitalTwin:
    """
    Hierarchical agent network simulation.
    Built from historical data; simulates forward under any shock scenario.
    """

    def __init__(self):
        self.states: List[StateEntity] = []
        self.rng = np.random.default_rng(42)

    def build_from_data(self, agents_df: pd.DataFrame, agent_month_df: pd.DataFrame) -> None:
        """Initialise twin entities from historical data."""
        log.info("[DigitalTwin] Building twin from data …")

        # Get last-month state per agent
        last = (
            agent_month_df.sort_values("year_month_dt")
            .groupby("agent_id").last()
            .reset_index()
        )
        last_map = last.set_index("agent_id").to_dict("index")

        state_map: Dict[str, StateEntity] = {}

        for _, agent_row in agents_df.iterrows():
            aid     = agent_row["agent_id"]
            st      = agent_row["state"]
            dist    = agent_row["district"]
            n_prod  = int(agent_row["n_producers"])

            hist    = last_map.get(aid, {})
            base_gwp= float(hist.get("gwp", 40_000))
            pif     = float(hist.get("pif", 150))
            ret     = float(hist.get("renewal_rate", 0.82))
            can     = float(hist.get("cancellation_rate", 0.05))
            avg_p   = float(hist.get("avg_premium", 1_400))
            q_score = float(agent_row.get("quality_score", 0.7))

            producers = [
                ProducerEntity(
                    producer_id=f"{aid}-P{i:02d}",
                    agent_id=aid,
                    base_new_policies_month=float(self.rng.lognormal(0.5, 0.4)),
                    conversion_rate=float(agent_row.get("base_conversion_rate", 0.30)),
                    price_sensitivity=float(self.rng.uniform(-1.2, -0.4)),
                    discount_sensitivity=float(self.rng.uniform(0.1, 0.5)),
                )
                for i in range(1, n_prod + 1)
            ]

            agent_entity = AgentEntity(
                agent_id=aid,
                district=dist,
                state=st,
                producers=producers,
                base_gwp_month=base_gwp,
                retention_rate=ret,
                cancellation_rate=can,
                avg_premium=avg_p,
                quality_score=q_score,
                pif=pif,
            )

            if st not in state_map:
                state_map[st] = StateEntity(state=st, districts=[])

            # Find or create district
            dist_found = None
            for d in state_map[st].districts:
                if d.district == dist:
                    dist_found = d
                    break
            if dist_found is None:
                dist_found = DistrictEntity(district=dist, state=st, agents=[])
                state_map[st].districts.append(dist_found)
            dist_found.agents.append(agent_entity)

        self.states = list(state_map.values())
        n_agents = sum(len(d.agents) for s in self.states for d in s.districts)
        log.info(f"[DigitalTwin] Built: {len(self.states)} states, {n_agents} agents")

    def simulate(
        self,
        n_steps: int = 24,
        pricing_shock: float = 0.0,
        discount_shock: float = 0.0,
        competition_shock: float = 0.0,
        retention_shock: float = 0.0,
        producer_count_shock: float = 0.0,
        scenario_name: str = "baseline",
    ) -> pd.DataFrame:
        """Run forward simulation and return time-series results."""
        records = []
        base_date = pd.Timestamp("2024-01-01")

        # Save/restore state for reproducibility (shallow copy of pif)
        initial_pif = {
            agent.agent_id: agent.pif
            for state in self.states
            for dist in state.districts
            for agent in dist.agents
        }

        for step in range(n_steps):
            month_date = base_date + pd.DateOffset(months=step)
            # Add slight seasonality to shocks
            seasonal_mult = 1 + 0.1 * np.sin(2 * np.pi * month_date.month / 12)

            for state_entity in self.states:
                step_results = state_entity.step(
                    pricing_shock=pricing_shock * seasonal_mult,
                    discount_shock=discount_shock,
                    competition_shock=competition_shock,
                    retention_shock=retention_shock,
                    producer_count_shock=producer_count_shock,
                )
                for r in step_results:
                    r["month"]    = month_date
                    r["step"]     = step
                    r["scenario"] = scenario_name
                    r["state"]    = state_entity.state
                    records.append(r)

        # Restore initial PIF
        for state in self.states:
            for dist in state.districts:
                for agent in dist.agents:
                    if agent.agent_id in initial_pif:
                        agent.pif = initial_pif[agent.agent_id]

        df = pd.DataFrame(records)
        return df

    def get_twin_state(self) -> Dict:
        """Snapshot of current entity states."""
        snapshot = {}
        for state in self.states:
            snapshot[state.state] = {}
            for dist in state.districts:
                snapshot[state.state][dist.district] = {
                    agent.agent_id: {
                        "pif": agent.pif,
                        "retention_rate": agent.retention_rate,
                        "cancellation_rate": agent.cancellation_rate,
                        "avg_premium": agent.avg_premium,
                        "n_producers": len(agent.producers),
                    }
                    for agent in dist.agents
                }
        return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# AGENT NODE
# ─────────────────────────────────────────────────────────────────────────────

def digital_twin_agent(state: AgentState) -> AgentState:
    log.info("[DigitalTwinAgent] Building and running digital twin …")
    errors = list(state.get("errors", []))

    try:
        agents_df      = state.get("agents_df")
        agent_month_df = state.get("agent_month_df")

        twin = DigitalTwin()
        twin.build_from_data(agents_df, agent_month_df)

        # ── Baseline simulation ───────────────────────────────────────────────
        baseline_df = twin.simulate(
            n_steps=TWIN_CONFIG["simulation_steps"],
            scenario_name="baseline",
        )

        # ── Optimistic scenario (better retention, more producers) ────────────
        optimistic_df = twin.simulate(
            n_steps=TWIN_CONFIG["simulation_steps"],
            retention_shock=+0.05,
            producer_count_shock=+0.10,
            scenario_name="optimistic",
        )

        # ── Stress scenario (competition + price pressure) ────────────────────
        stress_df = twin.simulate(
            n_steps=TWIN_CONFIG["simulation_steps"],
            pricing_shock=+0.05,
            competition_shock=+0.15,
            retention_shock=-0.03,
            scenario_name="stress",
        )

        sim_all = pd.concat([baseline_df, optimistic_df, stress_df], ignore_index=True)

        # Monthly roll-up
        monthly_sim = (
            sim_all.groupby(["scenario", "month"])
            .agg(total_gwp=("gwp", "sum"), total_pif=("pif", "sum"),
                 new_policies=("new_policies", "sum"))
            .reset_index()
        )

        twin_state   = twin.get_twin_state()
        twin_summary = _build_summary(monthly_sim)

        log.info(f"[DigitalTwinAgent] Simulation complete: {len(sim_all):,} rows")

        state["twin_state"]      = twin_state
        state["twin_simulation"] = monthly_sim
        state["twin_summary"]    = twin_summary

    except Exception as e:
        err_msg = f"[DigitalTwinAgent] ERROR: {e}"
        log.error(err_msg, exc_info=True)
        errors.append(err_msg)
        state["twin_state"]      = {}
        state["twin_simulation"] = pd.DataFrame()
        state["twin_summary"]    = "Digital twin encountered an error."

    state["errors"] = errors
    completed = list(state.get("completed_agents", []))
    completed.append("digital_twin_agent")
    state["completed_agents"] = completed
    log.info("[DigitalTwinAgent] Done.")
    return state


def _build_summary(monthly_sim: pd.DataFrame) -> str:
    lines = ["## Digital Twin Simulation Summary\n"]
    for scenario in monthly_sim["scenario"].unique():
        sc_df = monthly_sim[monthly_sim["scenario"] == scenario]
        total_gwp = sc_df["total_gwp"].sum()
        final_pif = sc_df["total_pif"].iloc[-1]
        lines.append(f"**{scenario.title()}**: Total GWP=${total_gwp:,.0f} | "
                     f"Final PIF={final_pif:,.0f}")
    return "\n".join(lines)
