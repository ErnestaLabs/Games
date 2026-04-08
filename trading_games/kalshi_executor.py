"""
KalshiClient — read-only price feed from Kalshi for cross-platform arb detection.

No Kalshi account needed. The public market API requires no auth for reads.
Use to find Kalshi/Polymarket price divergences, then execute on Polymarket
via Railway EU West (Ireland IP — not geoblocked).

Settlement divergence arb:
  Kalshi settles on AP call (0–4h post-event).
  Polymarket settles 24–48h later.
  When Kalshi shows result=yes/no but Polymarket still trading at 0.85–0.97,
  buy the near-certain side on Polymarket from Railway.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiClient:
    """
    Read-only Kalshi price client. No account needed — public API.
    Used to detect price divergences vs Polymarket and settlement gaps.
    All execution happens on Polymarket via Railway Ireland.
    """

    def __init__(self) -> None:
        self._http = httpx.Client(timeout=15.0)

    def get_open_markets(self, query: str = "", limit: int = 50) -> list[dict]:
        """Fetch open Kalshi markets. No auth required."""
        try:
            params: dict = {"limit": limit, "status": "open"}
            if query:
                params["search"] = query
            resp = self._http.get(f"{KALSHI_BASE}/markets", params=params)
            if resp.status_code == 200:
                return resp.json().get("markets") or []
        except Exception as exc:
            logger.debug("Kalshi open markets fetch failed: %s", exc)
        return []

    def get_settled_markets(self, limit: int = 50) -> list[dict]:
        """
        Fetch recently settled Kalshi markets.
        These may still be trading on Polymarket — settlement divergence arb window.
        """
        try:
            resp = self._http.get(
                f"{KALSHI_BASE}/markets",
                params={"limit": limit, "status": "finalized"},
            )
            if resp.status_code == 200:
                markets = resp.json().get("markets") or []
                return [
                    {
                        "ticker": m.get("ticker"),
                        "title": m.get("title"),
                        "result": m.get("result"),      # "yes" or "no"
                        "yes_price": 1.0 if m.get("result") == "yes" else 0.0,
                    }
                    for m in markets if m.get("result") in ("yes", "no")
                ]
        except Exception as exc:
            logger.debug("Kalshi settled markets fetch failed: %s", exc)
        return []

    def find_divergence(
        self, poly_question: str, poly_yes_price: float, min_spread: float = 0.05
    ) -> Optional[dict]:
        """
        Search for a Kalshi market matching this Polymarket question.
        Returns divergence info if spread > min_spread (default 5%).
        """
        keywords = " ".join(w for w in poly_question.split() if len(w) > 4)[:60]
        markets = self.get_open_markets(query=keywords, limit=10)
        for m in markets:
            yes_ask = m.get("yes_ask") or m.get("yes_price") or 0.0
            if yes_ask <= 0:
                continue
            spread = abs(float(yes_ask) / 100.0 - poly_yes_price)
            if spread >= min_spread:
                logger.info(
                    "DIVERGENCE: Kalshi=%s %.2f vs Poly %.2f (spread=%.2f)",
                    m.get("ticker"), yes_ask / 100.0, poly_yes_price, spread,
                )
                return {
                    "kalshi_ticker": m.get("ticker"),
                    "kalshi_title": m.get("title"),
                    "kalshi_yes_price": float(yes_ask) / 100.0,
                    "poly_yes_price": poly_yes_price,
                    "spread": spread,
                }
        return None

    def close(self) -> None:
        self._http.close()


# Keep old name as alias so agent_runner.py import doesn't break
KalshiExecutor = KalshiClient
