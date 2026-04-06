"""
HIP-4 Arbitrage — Hyperliquid vs Polymarket spread capture.

HIP-4 is Hyperliquid's prediction market protocol. When the same outcome
trades on both Hyperliquid and Polymarket with a spread > HIP4_MIN_SPREAD (3%),
we can:
  - Buy the cheaper side on one exchange
  - Sell (or already hold) the more expensive side on the other

This is a convergence trade: both prices should resolve at the same value (0 or 1).

Requirements:
  - HYPERLIQUID_API_URL env var
  - POLYGON_PRIVATE_KEY for Polymarket execution
  - Hyperliquid API key for HL execution (HYPERLIQUID_API_KEY)

Min spread to trade: HIP4_MIN_SPREAD (default 3%)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from trading_games.config import (
    HYPERLIQUID_API_URL, HIP4_MIN_SPREAD, CLOB_HOST, DRY_RUN,
    FORAGE_GRAPH_URL, GRAPH_API_SECRET,
)

logger = logging.getLogger(__name__)

HYPERLIQUID_API_KEY = os.environ.get("HYPERLIQUID_API_KEY", "")
_GRAPH_HEADERS = {"Authorization": f"Bearer {GRAPH_API_SECRET}"} if GRAPH_API_SECRET else {}


@dataclass
class HIP4Signal:
    market_id: str          # Polymarket condition_id
    question: str
    hl_market_id: str       # Hyperliquid market ID
    poly_price_yes: float   # Polymarket YES price
    hl_price_yes: float     # Hyperliquid YES price
    spread: float           # abs(poly - hl)
    trade_side: str         # "BUY_POLY_SELL_HL" or "BUY_HL_SELL_POLY"
    edge: float             # net edge after fees


def _fetch_hl_markets() -> list[dict]:
    """Fetch active Hyperliquid prediction markets."""
    try:
        resp = httpx.post(
            HYPERLIQUID_API_URL,
            json={"type": "predictionMarketsInfo"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json() if isinstance(resp.json(), list) else []
    except Exception as exc:
        logger.debug("Hyperliquid market fetch failed: %s", exc)
    return []


def _poly_price(market_id: str) -> float | None:
    """Get current YES price for a Polymarket market."""
    try:
        resp = httpx.get(f"{CLOB_HOST}/markets/{market_id}", timeout=8.0)
        if resp.status_code == 200:
            tokens = resp.json().get("tokens") or []
            for t in tokens:
                if (t.get("outcome") or "").upper() == "YES":
                    return float(t.get("price") or 0.0)
    except Exception:
        pass
    return None


def _hl_price(hl_market_id: str) -> float | None:
    """Get current YES price for a Hyperliquid prediction market."""
    try:
        resp = httpx.post(
            HYPERLIQUID_API_URL,
            json={"type": "predictionMarketPrice", "marketId": hl_market_id},
            timeout=8.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("yes") or data.get("price") or 0.0)
    except Exception:
        pass
    return None


def _match_markets(poly_markets: list[dict], hl_markets: list[dict]) -> list[tuple]:
    """
    Match Polymarket markets to Hyperliquid markets by keyword similarity.
    Returns list of (poly_market, hl_market) tuples.
    """
    matches = []
    for pm in poly_markets:
        q_lower = (pm.get("question") or "").lower()
        for hm in hl_markets:
            hq = (hm.get("name") or hm.get("question") or "").lower()
            # Simple overlap score
            pm_words = set(w for w in q_lower.split() if len(w) > 4)
            hl_words = set(w for w in hq.split() if len(w) > 4)
            overlap  = len(pm_words & hl_words)
            if overlap >= 3:
                matches.append((pm, hm, overlap))

    matches.sort(key=lambda x: x[2], reverse=True)
    return [(m[0], m[1]) for m in matches[:20]]


def scan_hip4_opportunities(poly_markets: list[dict]) -> list[HIP4Signal]:
    """
    Scan for HIP-4 arbitrage between Polymarket and Hyperliquid.
    Returns list of signals above HIP4_MIN_SPREAD.
    """
    hl_markets = _fetch_hl_markets()
    if not hl_markets:
        logger.debug("No Hyperliquid markets fetched")
        return []

    matched = _match_markets(poly_markets, hl_markets)
    signals: list[HIP4Signal] = []

    for poly_mkt, hl_mkt in matched:
        market_id = poly_mkt.get("condition_id") or poly_mkt.get("market_id") or ""
        hl_id     = hl_mkt.get("id") or hl_mkt.get("marketId") or ""
        if not market_id or not hl_id:
            continue

        poly_yes = _poly_price(market_id)
        hl_yes   = _hl_price(hl_id)

        if poly_yes is None or hl_yes is None:
            continue

        spread = abs(poly_yes - hl_yes)
        if spread < HIP4_MIN_SPREAD:
            continue

        # Fee estimate: ~2% Polymarket + ~1% HL = ~3% total
        net_edge = spread - 0.03
        if net_edge < 0.005:
            continue

        if poly_yes < hl_yes:
            trade_side = "BUY_POLY_SELL_HL"
        else:
            trade_side = "BUY_HL_SELL_POLY"

        signal = HIP4Signal(
            market_id=market_id,
            question=poly_mkt.get("question", ""),
            hl_market_id=hl_id,
            poly_price_yes=poly_yes,
            hl_price_yes=hl_yes,
            spread=spread,
            trade_side=trade_side,
            edge=net_edge,
        )
        signals.append(signal)

        logger.info(
            "[HIP4] ARB: poly=%.3f hl=%.3f spread=%.1f%% net_edge=%.1f%% — %s",
            poly_yes, hl_yes, spread * 100, net_edge * 100, trade_side,
        )

    # Write signals to graph
    if signals and GRAPH_API_SECRET:
        try:
            http = httpx.Client(headers=_GRAPH_HEADERS, timeout=8.0)
            http.post(f"{FORAGE_GRAPH_URL}/signal", json={
                "entity": "trading_games:hip4_arb",
                "metric": "signals_found",
                "value": float(len(signals)),
            })
            http.close()
        except Exception:
            pass

    return sorted(signals, key=lambda s: s.edge, reverse=True)
