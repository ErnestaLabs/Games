"""
ForageSignalSource — pulls entity signals from the Forage Graph API.

Replaces Polymarket market fetcher for IG-native execution.
Queries the /signal endpoint for recent entity events and converts
them into the signal format that agents understand.

Signal format (compatible with analyze_market dict):
  {
    "market_id":    str,       # entity ID from graph
    "question":     str,       # human-readable event description
    "entity_name":  str,
    "entity_type":  str,       # company / person / macro / political / regulatory
    "signal_text":  str,       # raw signal description
    "direction":    str,       # bullish / bearish / neutral
    "confidence":   float,
    "source":       str,
    "tokens":       [],        # empty — no Polymarket tokens
    "market_price": float,     # 0.5 placeholder (no binary market)
  }
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FORAGE_GRAPH_URL = os.environ.get(
    "FORAGE_GRAPH_URL", "https://forage-graph-production.up.railway.app"
)
GRAPH_API_SECRET = os.environ.get("GRAPH_API_SECRET", "")

# How many entities to pull per cycle
SIGNAL_LIMIT = int(os.environ.get("FORAGE_SIGNAL_LIMIT", "30"))


class ForageSignalSource:
    """Polls the Forage Graph for recent entity signals and emits IG-compatible market dicts."""

    def __init__(
        self,
        graph_url: str = FORAGE_GRAPH_URL,
        api_secret: str = GRAPH_API_SECRET,
    ) -> None:
        self._url = graph_url.rstrip("/")
        self._secret = api_secret
        self._http = httpx.Client(timeout=15.0)
        self._headers = {"Authorization": f"Bearer {self._secret}"}

    def fetch_signals(self) -> list[dict]:
        """
        Fetch recent entity signals from the graph.
        Returns list of signal dicts compatible with agent analyze_market().
        """
        signals: list[dict] = []

        # 1. Recent entity signals from /signal endpoint
        signals.extend(self._fetch_recent_signals())

        # 2. High-causal-weight entities from /query
        signals.extend(self._fetch_high_causal_entities())

        # Deduplicate by market_id
        seen: set[str] = set()
        unique: list[dict] = []
        for s in signals:
            mid = s.get("market_id", "")
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(s)

        logger.info("[ForageSignalSource] %d unique signals fetched", len(unique))
        return unique

    def _fetch_recent_signals(self) -> list[dict]:
        """GET /signal — recent signals emitted by the graph."""
        if not self._secret:
            logger.debug("[ForageSignalSource] No GRAPH_API_SECRET — skipping /signal")
            return []
        try:
            resp = self._http.get(
                f"{self._url}/signal",
                headers=self._headers,
                params={"limit": SIGNAL_LIMIT, "order": "desc"},
            )
            if resp.status_code == 200:
                data = resp.json()
                items = data if isinstance(data, list) else (data.get("signals") or data.get("data") or [])
                return [self._normalise_signal(s) for s in items if s]
            logger.debug("[ForageSignalSource] /signal returned %d", resp.status_code)
        except Exception as exc:
            logger.warning("[ForageSignalSource] /signal error: %s", exc)
        return []

    def _fetch_high_causal_entities(self) -> list[dict]:
        """
        POST /query — cypher query for entities with recent high-weight causal connections.
        Falls back gracefully if FalkorDB not available.
        """
        if not self._secret:
            return []
        query = (
            "MATCH (e:Entity)-[r:CAUSES]->(t:Entity) "
            "WHERE r.weight > 0.6 "
            "RETURN e.id AS id, e.name AS name, e.type AS type, "
            "       r.weight AS weight, r.description AS description, "
            "       t.name AS target "
            "ORDER BY r.weight DESC "
            f"LIMIT {SIGNAL_LIMIT}"
        )
        try:
            resp = self._http.post(
                f"{self._url}/query",
                headers={**self._headers, "Content-Type": "application/json"},
                json={"query": query},
            )
            if resp.status_code == 200:
                data = resp.json()
                rows = data if isinstance(data, list) else (data.get("results") or data.get("data") or [])
                return [self._normalise_graph_row(r) for r in rows if r]
            logger.debug("[ForageSignalSource] /query returned %d", resp.status_code)
        except Exception as exc:
            logger.warning("[ForageSignalSource] /query error: %s", exc)
        return []

    @staticmethod
    def _normalise_signal(raw: dict) -> dict:
        """Convert a /signal endpoint record to agent-compatible format."""
        entity = raw.get("entity") or {}
        entity_name = (
            entity.get("name")
            or raw.get("entity_name")
            or raw.get("name")
            or "Unknown Entity"
        )
        entity_type = (
            entity.get("type")
            or raw.get("entity_type")
            or raw.get("type")
            or "unknown"
        )
        signal_text = (
            raw.get("description")
            or raw.get("signal")
            or raw.get("text")
            or ""
        )
        direction = (
            raw.get("direction")
            or raw.get("sentiment")
            or "neutral"
        )
        return {
            "market_id":    raw.get("id") or raw.get("signal_id") or f"fg_{entity_name[:16]}",
            "question":     f"{entity_name}: {signal_text}"[:200],
            "entity_name":  entity_name,
            "entity_type":  entity_type,
            "signal_text":  signal_text,
            "direction":    direction,
            "confidence":   float(raw.get("confidence") or raw.get("weight") or 0.5),
            "source":       raw.get("source") or "forage_graph",
            "tokens":       [],
            "market_price": 0.5,
            "is_fee_free":  False,
            "tick_size":    "0.01",
            "min_order_size": 1.0,
        }

    @staticmethod
    def _normalise_graph_row(row: dict) -> dict:
        """Convert a /query causal row to agent-compatible format."""
        entity_name = row.get("name") or "Unknown"
        target      = row.get("target") or ""
        description = row.get("description") or f"{entity_name} → {target}"
        weight      = float(row.get("weight") or 0.6)
        return {
            "market_id":    row.get("id") or f"fg_{entity_name[:16]}_{target[:8]}",
            "question":     f"{entity_name} causal impact on {target}"[:200],
            "entity_name":  entity_name,
            "entity_type":  row.get("type") or "entity",
            "signal_text":  description,
            "direction":    "neutral",
            "confidence":   min(0.9, weight),
            "source":       "forage_causal_graph",
            "tokens":       [],
            "market_price": 0.5,
            "is_fee_free":  False,
            "tick_size":    "0.01",
            "min_order_size": 1.0,
        }

    def close(self) -> None:
        self._http.close()
