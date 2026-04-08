"""
OnchainCollector — Polygon on-chain data for Polymarket smart-money tracking.

Pushes to Forage Graph:
  Source      — wallets with high recent PM volume (via data-api leaderboard)
  Trade       — resolved market settlements (on-chain outcomes)
  Narrative   — USDC flow signals (large transfers correlated with PM moves)

Uses public Polygon RPC + Polymarket data-api (no auth required).
Keeps calls lightweight: leaderboard + recent trades, no full node sync.
"""
from __future__ import annotations

import logging
import os
import time

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

DATA_API         = "https://data-api.polymarket.com"
POLYGON_RPC      = os.environ.get("POLYGON_RPC", "https://polygon-rpc.com")

TOP_TRADERS_LIMIT = int(os.environ.get("ONCHAIN_TOP_TRADERS", "30"))
RECENT_TRADES_LIMIT = int(os.environ.get("ONCHAIN_RECENT_TRADES", "100"))


class OnchainCollector(BaseCollector):
    source_name = "onchain_collector"

    def collect(self) -> list[dict]:
        nodes: list[dict] = []
        ts = self._ts()

        # Top leaderboard traders (smart money)
        leaders = self._fetch_leaderboard()
        for w in leaders:
            addr = w.get("proxyWallet") or w.get("address") or ""
            if not addr:
                continue
            nodes.append({
                "id":          f"wallet_{addr}",
                "type":        "Source",
                "name":        addr,
                "source_type": "polygon_wallet",
                "pnl_usdc":    float(w.get("pnl") or 0),
                "volume_usdc": float(w.get("volume") or 0),
                "trades":      int(w.get("tradesCount") or 0),
                "rank":        int(w.get("rank") or 0),
                "timestamp_ms": ts,
                "source":      self.source_name,
            })

        # Recent resolved trades — outcome signals for graph
        trades = self._fetch_recent_trades()
        for t in trades:
            trade_id = t.get("id") or t.get("transactionHash") or ""
            if not trade_id:
                continue
            nodes.append({
                "id":          f"onchain_trade_{trade_id[:32]}",
                "type":        "Trade",
                "venue":       "polymarket_onchain",
                "market_id":   t.get("market") or t.get("conditionId") or "",
                "wallet":      t.get("proxyWallet") or t.get("maker") or "",
                "outcome":     t.get("outcome") or t.get("side") or "",
                "size_usdc":   float(t.get("size") or t.get("usdcSize") or 0),
                "price":       float(t.get("price") or 0),
                "tx_hash":     t.get("transactionHash") or "",
                "timestamp_ms": ts,
                "source":      self.source_name,
            })

        logger.info(
            "[onchain] leaders=%d recent_trades=%d total_nodes=%d",
            len(leaders), len(trades), len(nodes),
        )
        return nodes

    def _fetch_leaderboard(self) -> list[dict]:
        """Derive top traders from recent trades volume (leaderboard endpoint is gone)."""
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
                vol: dict[str, float] = {}
                for t in trades:
                    addr = t.get("maker") or t.get("proxyWallet") or ""
                    if addr:
                        vol[addr] = vol.get(addr, 0) + float(t.get("usdcSize") or 0)
                top = sorted(vol.items(), key=lambda x: -x[1])[:TOP_TRADERS_LIMIT]
                return [{"proxyWallet": a, "volume": v} for a, v in top]
            logger.debug("[onchain] top traders: %d", resp.status_code)
        except Exception as exc:
            logger.debug("[onchain] top traders error: %s", exc)
        return []

    def _fetch_recent_trades(self) -> list[dict]:
        try:
            resp = self._http.get(
                f"{DATA_API}/trades",
                params={"limit": RECENT_TRADES_LIMIT},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else (data.get("data") or [])
            logger.debug("[onchain] trades: %d", resp.status_code)
        except Exception as exc:
            logger.debug("[onchain] trades error: %s", exc)
        return []
