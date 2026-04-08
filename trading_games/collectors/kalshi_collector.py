"""
KalshiCollector — open markets + order-book snapshots from Kalshi trade API.

Pushes to Forage Graph:
  PredictionMarket  — all open markets with YES mid-price
  OddsSnapshot      — price checkpoint per run
  Market            — series/category metadata

No auth required for public market data.
"""
from __future__ import annotations

import logging
import os
import time

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

KALSHI_LIMIT = int(os.environ.get("KALSHI_MARKET_LIMIT", "200"))


class KalshiCollector(BaseCollector):
    source_name = "kalshi_collector"

    def collect(self) -> list[dict]:
        nodes: list[dict] = []
        ts = self._ts()

        markets = self._fetch_markets()
        series_seen: set[str] = set()

        for m in markets:
            ticker  = m.get("ticker") or m.get("market_id") or m.get("id") or ""
            title   = m.get("title") or m.get("question") or ""
            series  = m.get("series_ticker") or ""
            price   = self._yes_mid(m)
            vol     = float(m.get("volume") or 0)
            cat     = m.get("category") or ""

            if not ticker:
                continue

            nodes.append({
                "id":          f"kalshi_market_{ticker}",
                "type":        "PredictionMarket",
                "name":        title[:200],
                "venue":       "kalshi",
                "ticker":      ticker,
                "yes_price":   price,
                "volume":      vol,
                "category":    cat,
                "close_time":  m.get("close_time") or "",
                "result":      m.get("result") or "",
                "source":      self.source_name,
            })

            if price is not None:
                nodes.append({
                    "id":           f"kalshi_snap_{ticker}_{ts}",
                    "type":         "OddsSnapshot",
                    "market_id":    f"kalshi_market_{ticker}",
                    "venue":        "kalshi",
                    "yes_price":    price,
                    "volume":       vol,
                    "timestamp_ms": ts,
                    "source":       self.source_name,
                })

            if series and series not in series_seen:
                series_seen.add(series)
                nodes.append({
                    "id":          f"kalshi_series_{series}",
                    "type":        "Market",
                    "name":        series,
                    "venue":       "kalshi",
                    "category":    cat,
                    "source":      self.source_name,
                })

        logger.info(
            "[kalshi] markets=%d series=%d total_nodes=%d",
            len(markets), len(series_seen), len(nodes),
        )
        return nodes

    def _fetch_markets(self) -> list[dict]:
        try:
            resp = self._http.get(
                f"{KALSHI_BASE}/markets",
                params={"limit": KALSHI_LIMIT, "status": "open"},
            )
            if resp.status_code == 200:
                return resp.json().get("markets") or []
            logger.warning("[kalshi] markets fetch: %d %s", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.warning("[kalshi] markets fetch error: %s", exc)
        return []

    @staticmethod
    def _yes_mid(m: dict) -> float | None:
        ask = m.get("yes_ask")
        bid = m.get("yes_bid")
        if ask is not None and bid is not None:
            try:
                return (float(ask) + float(bid)) / 2 / 100   # Kalshi prices in cents
            except Exception:
                pass
        rp = m.get("result")
        if rp is not None:
            try:
                return float(rp) / 100
            except Exception:
                pass
        return None
