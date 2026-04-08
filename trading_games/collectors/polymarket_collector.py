"""
PolymarketCollector — Gamma API market snapshots + top-wallet activity.

Pushes to Forage Graph:
  PredictionMarket  — every active market with YES price + volume
  OddsSnapshot      — time-series price checkpoint
  Source            — high-volume wallet addresses (smart-money intel)

No auth required; Gamma API is public.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

GAMMA_API   = "https://gamma-api.polymarket.com"
DATA_API    = "https://data-api.polymarket.com"

# How many markets and wallets to pull per run
MARKET_LIMIT  = int(os.environ.get("PM_MARKET_LIMIT",  "150"))
WALLET_LIMIT  = int(os.environ.get("PM_WALLET_LIMIT",  "50"))


class PolymarketCollector(BaseCollector):
    source_name = "polymarket_collector"

    def collect(self) -> list[dict]:
        nodes: list[dict] = []
        ts = self._ts()

        markets = self._fetch_markets()
        for m in markets:
            mid   = m.get("conditionId") or m.get("id") or ""
            q     = m.get("question") or m.get("title") or ""
            price = self._yes_price(m)
            vol   = float(m.get("volume") or m.get("volume24hr") or 0)
            liq   = float(m.get("liquidity") or 0)

            if not mid:
                continue

            nodes.append({
                "id":          f"pm_market_{mid}",
                "type":        "PredictionMarket",
                "name":        q[:200],
                "venue":       "polymarket",
                "yes_price":   price,
                "volume":      vol,
                "liquidity":   liq,
                "end_date":    m.get("endDate") or m.get("end_date") or "",
                "category":    m.get("category") or "",
                "active":      m.get("active", True),
                "source":      self.source_name,
            })

            # OddsSnapshot — price at this instant
            if price is not None:
                nodes.append({
                    "id":          f"pm_snap_{mid}_{ts}",
                    "type":        "OddsSnapshot",
                    "market_id":   f"pm_market_{mid}",
                    "venue":       "polymarket",
                    "yes_price":   price,
                    "volume":      vol,
                    "timestamp_ms": ts,
                    "source":      self.source_name,
                })

        # Top wallets — smart money tracking
        wallets = self._fetch_top_wallets()
        for w in wallets:
            addr = w.get("proxyWallet") or w.get("address") or ""
            if not addr:
                continue
            nodes.append({
                "id":          f"pm_wallet_{addr}",
                "type":        "Source",
                "name":        addr,
                "source_type": "polymarket_wallet",
                "pnl":         float(w.get("pnl") or 0),
                "volume":      float(w.get("volume") or 0),
                "trades":      int(w.get("tradesCount") or 0),
                "source":      self.source_name,
            })

        logger.info(
            "[polymarket] markets=%d wallet_sources=%d total_nodes=%d",
            len(markets), len(wallets), len(nodes),
        )
        return nodes

    # ── Fetchers ──────────────────────────────────────────────────────────────

    def _fetch_markets(self) -> list[dict]:
        try:
            resp = self._http.get(
                f"{GAMMA_API}/markets",
                params={
                    "active":    "true",
                    "closed":    "false",
                    "limit":     MARKET_LIMIT,
                    "order":     "volume24hr",
                    "ascending": "false",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else (data.get("markets") or [])
            logger.warning("[polymarket] markets fetch: %d", resp.status_code)
        except Exception as exc:
            logger.warning("[polymarket] markets fetch error: %s", exc)
        return []

    def _fetch_top_wallets(self) -> list[dict]:
        """Derive top wallets from recent high-volume trades (leaderboard endpoint gone)."""
        try:
            resp = self._http.get(
                f"{DATA_API}/trades",
                params={"limit": 500, "taker_side": "BUY"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                trades = resp.json()
                if isinstance(trades, dict):
                    trades = trades.get("data") or []
                # Aggregate USDC volume per wallet, return top WALLET_LIMIT
                vol: dict[str, float] = {}
                for t in trades:
                    addr = t.get("maker") or t.get("proxyWallet") or ""
                    if addr:
                        vol[addr] = vol.get(addr, 0) + float(t.get("usdcSize") or 0)
                top = sorted(vol.items(), key=lambda x: -x[1])[:WALLET_LIMIT]
                return [{"address": a, "volume": v} for a, v in top]
            logger.debug("[polymarket] wallets fetch: %d", resp.status_code)
        except Exception as exc:
            logger.debug("[polymarket] wallets fetch error: %s", exc)
        return []

    @staticmethod
    def _yes_price(m: dict) -> float | None:
        for tok in (m.get("tokens") or []):
            if tok.get("outcome", "").upper() == "YES":
                p = tok.get("price")
                if p is not None:
                    return float(p)
        for f in ("lastTradePrice", "bestAsk"):
            val = m.get(f)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    pass
        return None
