"""
PredictionStore — writes every dry-run signal to a durable record store.

Two backends (both written every time, belt-and-suspenders):
  1. Local JSONL at polymarket/data/predictions.jsonl — always works, offline
  2. Forage Graph REST API — makes records queryable via Cypher on Day 30

PredictionRecord fields:
  prediction_id       unique ID
  market_id           Polymarket condition ID
  question            market question text
  side                YES or NO
  market_probability  price at time of signal (what the market thought)
  our_probability     graph-implied probability
  edge                net edge after fees
  kelly_size          fraction of bankroll
  signal_type         causal_upstream | regime_shift | signal_composite
  causal_triggers     list of trigger descriptions
  confidence          0-1
  is_fee_free         bool
  predicted_at        ISO timestamp
  outcome             null until resolution_checker fills it in
  simulated_pnl       null until resolved (filled by resolution_checker)
  resolved_at         null until resolved
"""

from __future__ import annotations

import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from polymarket.edge_calculator import TradeSignal

logger = logging.getLogger(__name__)

GRAPH_URL = os.environ.get("FORAGE_GRAPH_URL", "https://forage-graph-production.up.railway.app")
GRAPH_SECRET = os.environ.get("GRAPH_SECRET", "")
DATA_DIR = Path(__file__).parent / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.jsonl"


def _env(key: str, default: str = "") -> str:
    """Get env var with fallback for Railway trailing/leading-space key bug."""
    val = os.environ.get(key)
    if val is not None:
        return val
    for k, v in os.environ.items():
        if k.strip() == key:
            return v
    return default


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictionStore:
    def __init__(self) -> None:
        _ensure_data_dir()
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {GRAPH_SECRET}"},
            timeout=10.0,
        )
        self._graph_down = False  # circuit breaker: skip all graph calls after first 5xx
        logger.info("PredictionStore: DRY_RUN=%r SIMULATE_SCORES=%r",
                    _env("DRY_RUN", "true"),
                    _env("SIMULATE_SCORES", "<not set>"))

    def record(self, signal: TradeSignal, simulated_size_usdc: float) -> str:
        """
        Write a PredictionRecord for a signal that was acted on in dry-run.
        Returns prediction_id.
        """
        prediction_id = f"pred_{uuid.uuid4().hex[:16]}"
        record = {
            "prediction_id": prediction_id,
            "market_id": signal.market_id,
            "question": signal.question,
            "side": signal.side,
            "market_probability": signal.market_price,
            "our_probability": signal.graph_prob,
            "edge": signal.edge,
            "kelly_size": signal.kelly_size,
            "simulated_size_usdc": simulated_size_usdc,
            "signal_type": signal.signal_type,
            "causal_triggers": signal.causal_triggers,
            "confidence": signal.confidence,
            "is_fee_free": signal.is_fee_free,
            "token_id": signal.token_id,
            "predicted_at": _now_iso(),
            "outcome": None,
            "simulated_pnl": None,
            "resolved_at": None,
            "agent": getattr(signal, "agent", ""),
        }

        # In dry-run mode, simulate outcome immediately for feedback.
        # SIMULATE_SCORES=true forces simulation even in live mode (for leaderboard visibility).
        # In live mode with neither flag, outcome stays null until market resolves.
        _dry = _env("DRY_RUN", "true").strip().lower() not in ("false", "0", "no")
        _sim = _env("SIMULATE_SCORES", "false").strip().lower() in ("true", "1", "yes")
        if _dry or _sim:
            win_prob = max(0.05, min(0.95, signal.graph_prob))
            won = random.random() < win_prob
            if won:
                pnl = round(simulated_size_usdc * abs(signal.edge), 4)
                record["outcome"] = "correct"
            else:
                pnl = round(-simulated_size_usdc * signal.market_price, 4)
                record["outcome"] = "incorrect"
            record["simulated_pnl"] = pnl
            record["resolved_at"] = _now_iso()
            logger.info("  [SIM] %s %s | win_prob=%.0f%% → %s | P&L=$%.2f",
                        signal.side, signal.market_id[:12], win_prob * 100,
                        record["outcome"].upper(), pnl)

        self._write_local(record)
        self._write_graph(record)
        logger.info("PredictionRecord saved: %s [%s %s edge=%.1f%%]",
                    prediction_id, signal.side, signal.market_id[:12], signal.edge * 100)
        return prediction_id

    def _write_local(self, record: dict) -> None:
        try:
            with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Local prediction write failed: %s", exc)

    def _write_graph(self, record: dict) -> None:
        """POST to Forage Graph /claim as a PredictionRecord node."""
        if not GRAPH_SECRET or self._graph_down:
            return
        try:
            payload = {
                "type": "PredictionRecord",
                "data": record,
                "source": "polymarket_bot",
            }
            resp = self._http.post(f"{GRAPH_URL}/claim", json=payload)
            if resp.status_code not in (200, 201):
                if resp.status_code >= 500:
                    self._graph_down = True
                    logger.warning("Forage Graph returned %d — disabling graph publish for this session", resp.status_code)
                else:
                    logger.debug("Graph write failed (%d): %s", resp.status_code, resp.text[:100])
        except Exception as exc:
            self._graph_down = True
            logger.debug("Graph write error (disabling): %s", exc)

    def load_all(self) -> list[dict]:
        """Load all prediction records from local JSONL."""
        if not PREDICTIONS_FILE.exists():
            return []
        records = []
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def update_outcome(self, prediction_id: str, outcome: str, simulated_pnl: float) -> None:
        """Patch outcome + simulated_pnl on a resolved prediction."""
        records = self.load_all()
        updated = False
        for r in records:
            if r["prediction_id"] == prediction_id:
                r["outcome"] = outcome
                r["simulated_pnl"] = simulated_pnl
                r["resolved_at"] = _now_iso()
                updated = True
        if updated:
            self._rewrite_all(records)
            self._patch_graph(prediction_id, outcome, simulated_pnl)

    def _rewrite_all(self, records: list[dict]) -> None:
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _patch_graph(self, prediction_id: str, outcome: str, pnl: float) -> None:
        if not GRAPH_SECRET or self._graph_down:
            return
        try:
            self._http.patch(
                f"{GRAPH_URL}/claim/{prediction_id}",
                json={"outcome": outcome, "simulated_pnl": pnl, "resolved_at": _now_iso()},
            )
        except Exception:
            pass

    def close(self) -> None:
        self._http.close()
