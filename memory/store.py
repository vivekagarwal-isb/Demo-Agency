"""
Memory Store for Farmers EA Multi-Agent System.

Implements:
- SQLite: structured storage for predictions, outcomes, errors
- FAISS: vector similarity search for context-aware retrieval
- Feedback loop: compare predicted vs actual, compute error metrics
- Incremental learning: update model bias corrections
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    log.warning("faiss-cpu not installed. Vector search disabled.")

try:
    from config import MEMORY_CONFIG
except ImportError:
    MEMORY_CONFIG = {
        "db_path": "memory_store/memory.db",
        "faiss_index_path": "memory_store/faiss_index",
        "embedding_dim": 64,
        "max_memory_entries": 10_000,
        "learning_rate": 0.01,
        "feedback_window_months": 3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE MEMORY DATABASE
# ─────────────────────────────────────────────────────────────────────────────

class SQLiteMemoryDB:
    """Structured memory: predictions, actuals, errors, decisions."""

    DDL = """
    CREATE TABLE IF NOT EXISTS predictions (
        id          TEXT PRIMARY KEY,
        run_id      TEXT,
        timestamp   TEXT,
        agent       TEXT,
        metric      TEXT,
        period      TEXT,
        entity_id   TEXT,
        predicted   REAL,
        actual      REAL,
        error_pct   REAL,
        meta_json   TEXT
    );

    CREATE TABLE IF NOT EXISTS decisions (
        id          TEXT PRIMARY KEY,
        run_id      TEXT,
        timestamp   TEXT,
        decision    TEXT,
        context     TEXT,
        outcome     TEXT
    );

    CREATE TABLE IF NOT EXISTS model_biases (
        metric      TEXT PRIMARY KEY,
        bias        REAL,
        n_samples   INTEGER,
        last_updated TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_pred_metric  ON predictions(metric);
    CREATE INDEX IF NOT EXISTS idx_pred_agent   ON predictions(entity_id);
    CREATE INDEX IF NOT EXISTS idx_pred_period  ON predictions(period);
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.DDL)
        self.conn.commit()
        log.info(f"SQLite memory DB initialised at {db_path}")

    # ── Predictions ──────────────────────────────────────────────────────────

    def store_prediction(
        self,
        run_id: str,
        agent: str,
        metric: str,
        period: str,
        entity_id: str,
        predicted: float,
        meta: Optional[Dict] = None,
    ) -> str:
        rec_id = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO predictions
               (id, run_id, timestamp, agent, metric, period, entity_id, predicted, meta_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                rec_id, run_id, datetime.utcnow().isoformat(),
                agent, metric, period, entity_id, predicted,
                json.dumps(meta or {}),
            ),
        )
        self.conn.commit()
        return rec_id

    def update_actual(self, prediction_id: str, actual: float) -> None:
        predicted = self.conn.execute(
            "SELECT predicted FROM predictions WHERE id=?", (prediction_id,)
        ).fetchone()
        if predicted:
            error_pct = (predicted[0] - actual) / (abs(actual) + 1e-9)
            self.conn.execute(
                "UPDATE predictions SET actual=?, error_pct=? WHERE id=?",
                (actual, error_pct, prediction_id),
            )
            self.conn.commit()

    def get_recent_predictions(self, metric: str, n: int = 50) -> List[Dict]:
        rows = self.conn.execute(
            """SELECT * FROM predictions WHERE metric=? ORDER BY timestamp DESC LIMIT ?""",
            (metric, n),
        ).fetchall()
        cols = [c[0] for c in self.conn.execute(
            "SELECT * FROM predictions LIMIT 0"
        ).description or []]
        if not cols:
            cols = ["id","run_id","timestamp","agent","metric","period",
                    "entity_id","predicted","actual","error_pct","meta_json"]
        return [dict(zip(cols, r)) for r in rows]

    # ── Bias Correction ──────────────────────────────────────────────────────

    def compute_and_store_bias(self, metric: str) -> float:
        """Compute mean prediction error for a metric (for bias correction)."""
        rows = self.conn.execute(
            """SELECT error_pct FROM predictions
               WHERE metric=? AND actual IS NOT NULL""",
            (metric,),
        ).fetchall()
        if not rows:
            return 0.0
        errors = [r[0] for r in rows if r[0] is not None]
        if not errors:
            return 0.0
        bias = float(np.mean(errors))
        n    = len(errors)
        self.conn.execute(
            """INSERT OR REPLACE INTO model_biases (metric, bias, n_samples, last_updated)
               VALUES (?,?,?,?)""",
            (metric, bias, n, datetime.utcnow().isoformat()),
        )
        self.conn.commit()
        log.info(f"Bias for {metric}: {bias:.4f} (n={n})")
        return bias

    def get_bias(self, metric: str) -> float:
        row = self.conn.execute(
            "SELECT bias FROM model_biases WHERE metric=?", (metric,)
        ).fetchone()
        return float(row[0]) if row else 0.0

    # ── Decisions ────────────────────────────────────────────────────────────

    def store_decision(self, run_id: str, decision: str, context: str, outcome: str = "") -> None:
        self.conn.execute(
            "INSERT INTO decisions (id,run_id,timestamp,decision,context,outcome) VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), run_id, datetime.utcnow().isoformat(), decision, context, outcome),
        )
        self.conn.commit()

    def get_error_summary(self) -> Dict[str, float]:
        rows = self.conn.execute(
            """SELECT metric, AVG(ABS(error_pct)), COUNT(*) FROM predictions
               WHERE actual IS NOT NULL GROUP BY metric"""
        ).fetchall()
        return {r[0]: {"mape": r[1], "n": r[2]} for r in rows}

    def close(self):
        self.conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# FAISS VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

class FAISSMemoryStore:
    """
    Vector memory for semantic similarity retrieval.
    Embeds context as a fixed-dim numeric vector and retrieves
    nearest-neighbour memories at inference time.
    """

    def __init__(self, dim: int = 64, index_path: Optional[str] = None):
        self.dim = dim
        self.index_path = index_path
        self.metadata: List[Dict] = []

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
            log.info(f"FAISS index initialised (dim={dim})")
        else:
            self.index = None
            log.warning("FAISS unavailable – using fallback linear store")

        if index_path and FAISS_AVAILABLE:
            self._load()

    # ── Vector helpers ───────────────────────────────────────────────────────

    def _context_to_vector(self, context: Dict) -> np.ndarray:
        """
        Deterministic numeric embedding from a context dict.
        Uses a fixed mapping of known feature keys → vector positions.
        """
        keys = [
            "gwp", "pif", "nb_rate", "renewal_rate", "cancellation_rate",
            "conversion_rate", "retention_rate", "n_producers", "tenure_years",
            "quality_score", "avg_premium", "avg_discount",
        ]
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, k in enumerate(keys[:self.dim]):
            val = context.get(k, 0.0)
            if val is None:
                val = 0.0
            # Normalize rough scales
            vec[i] = float(val) / 1e5 if k == "gwp" else float(val)

        # Fill remaining dims with hash-based noise for uniqueness
        for j in range(len(keys), self.dim):
            vec[j] = hash(str(context)) % 1000 / 1000.0

        return vec.reshape(1, -1)

    # ── Store / Retrieve ─────────────────────────────────────────────────────

    def store(self, context: Dict, memory: Dict) -> None:
        vec = self._context_to_vector(context)
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(vec)
        self.metadata.append({"context": context, "memory": memory,
                               "timestamp": datetime.utcnow().isoformat()})

        if len(self.metadata) % 500 == 0:
            self._save()

    def retrieve(self, context: Dict, k: int = 5) -> List[Dict]:
        if not self.metadata:
            return []

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            vec = self._context_to_vector(context)
            k_eff = min(k, self.index.ntotal)
            _distances, indices = self.index.search(vec, k_eff)
            return [self.metadata[i]["memory"] for i in indices[0] if i < len(self.metadata)]
        else:
            # Fallback: return most recent k
            return [m["memory"] for m in self.metadata[-k:]]

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        if self.index_path is None:
            return
        p = Path(self.index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(p) + ".faiss")
        meta_path = str(p) + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata[-5000:], f)  # keep last 5000

    def _load(self):
        p = Path(self.index_path)
        if FAISS_AVAILABLE and (str(p) + ".faiss") and Path(str(p) + ".faiss").exists():
            self.index = faiss.read_index(str(p) + ".faiss")
        meta_path = str(p) + "_meta.json"
        if Path(meta_path).exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
            log.info(f"Loaded {len(self.metadata)} memories from {meta_path}")

    def size(self) -> int:
        return len(self.metadata)


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED MEMORY MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """Single interface for all memory operations."""

    _instance: Optional["MemoryManager"] = None

    def __init__(self):
        cfg = MEMORY_CONFIG
        self.sql  = SQLiteMemoryDB(cfg["db_path"])
        self.faiss = FAISSMemoryStore(
            dim=cfg["embedding_dim"],
            index_path=cfg["faiss_index_path"],
        )
        self.learning_rate = cfg["learning_rate"]

    @classmethod
    def get_instance(cls) -> "MemoryManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Delegate to sub-stores
    def store_prediction(self, **kwargs) -> str:
        return self.sql.store_prediction(**kwargs)

    def store_context(self, context: Dict, memory: Dict) -> None:
        self.faiss.store(context, memory)

    def retrieve_similar(self, context: Dict, k: int = 5) -> List[Dict]:
        return self.faiss.retrieve(context, k)

    def update_actual(self, prediction_id: str, actual: float) -> None:
        self.sql.update_actual(prediction_id, actual)

    def compute_bias(self, metric: str) -> float:
        return self.sql.compute_and_store_bias(metric)

    def get_bias(self, metric: str) -> float:
        return self.sql.get_bias(metric)

    def get_error_summary(self) -> Dict:
        return self.sql.get_error_summary()

    def store_decision(self, run_id: str, decision: str, context: str) -> None:
        self.sql.store_decision(run_id, decision, context)

    def get_recent_predictions(self, metric: str, n: int = 50) -> List[Dict]:
        return self.sql.get_recent_predictions(metric, n)

    def apply_bias_correction(self, metric: str, predicted: float) -> float:
        bias = self.get_bias(metric)
        corrected = predicted * (1 - bias * self.learning_rate)
        return corrected
