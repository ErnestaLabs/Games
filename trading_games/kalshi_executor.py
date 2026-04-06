"""
KalshiExecutor — UK-legal trade execution via Kalshi REST API.

Kalshi is fully regulated in the US and unrestricted from the UK.
Uses email/password login for demo accounts, or API key for production.

API base: https://trading-api.kalshi.com/trade-api/v2
Auth:     POST /login  → returns token  (or X-Mode: demo for sandbox)

Market matching:
  Given a Polymarket market question, find the equivalent Kalshi market
  by keyword search. Execute on Kalshi first; fall back to Polymarket CLOB.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

KALSHI_BASE   = os.environ.get("KALSHI_API_URL", "https://trading-api.kalshi.com/trade-api/v2")
KALSHI_EMAIL  = os.environ.get("KALSHI_EMAIL", "")
KALSHI_PASSWORD = os.environ.get("KALSHI_PASSWORD", "")
KALSHI_API_KEY  = os.environ.get("KALSHI_API_KEY", "")   # alternative: API key auth
KALSHI_DEMO   = os.environ.get("KALSHI_DEMO", "false").lower() not in ("false", "0", "no")


@dataclass
class KalshiResult:
    success: bool
    order_id: Optional[str]
    platform: str
    side: str
    size_usd: float
    price: float
    reason: str = ""


class KalshiExecutor:
    """
    Executes trades on Kalshi. Falls back gracefully when no matching market.
    """

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._http = httpx.Client(timeout=15.0)
        self._login()

    def _login(self) -> None:
        """Login to Kalshi and cache bearer token."""
        if KALSHI_API_KEY:
            # API key auth — set as header directly
            self._token = KALSHI_API_KEY
            logger.info("KalshiExecutor: using API key auth")
            return
        if not KALSHI_EMAIL or not KALSHI_PASSWORD:
            logger.warning("KalshiExecutor: no credentials set — disabled")
            return
        try:
            resp = self._http.post(
                f"{KALSHI_BASE}/login",
                json={"email": KALSHI_EMAIL, "password": KALSHI_PASSWORD},
                headers={"X-Mode": "demo"} if KALSHI_DEMO else {},
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data.get("token") or data.get("access_token")
            logger.info("KalshiExecutor: logged in (demo=%s)", KALSHI_DEMO)
        except Exception as exc:
            logger.error("KalshiExecutor login failed: %s", exc)

    @property
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if KALSHI_DEMO:
            h["X-Mode"] = "demo"
        return h

    @property
    def available(self) -> bool:
        return bool(self._token)

    # ── Market search ─────────────────────────────────────────────────────

    def find_market(self, question: str, limit: int = 5) -> Optional[dict]:
        """Find a Kalshi market matching the given question keywords."""
        if not self.available:
            return None
        keywords = " ".join(w for w in question.split() if len(w) > 4)[:60]
        try:
            resp = self._http.get(
                f"{KALSHI_BASE}/markets",
                headers=self._headers,
                params={"limit": limit, "status": "open", "search": keywords},
            )
            if resp.status_code == 200:
                markets = resp.json().get("markets") or []
                if markets:
                    # Return the most liquid match
                    return max(markets, key=lambda m: m.get("volume", 0), default=None)
        except Exception as exc:
            logger.debug("Kalshi market search failed: %s", exc)
        return None

    # ── Balance ──────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Return available USD balance on Kalshi."""
        if not self.available:
            return 0.0
        try:
            resp = self._http.get(f"{KALSHI_BASE}/portfolio/balance", headers=self._headers)
            if resp.status_code == 200:
                data = resp.json()
                cents = data.get("balance") or data.get("available_balance") or 0
                return float(cents) / 100.0  # Kalshi uses cents
        except Exception as exc:
            logger.debug("Kalshi balance fetch failed: %s", exc)
        return 0.0

    # ── Order placement ──────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        size_usd: float,
        price: float,
        dry_run: bool = True,
    ) -> KalshiResult:
        """
        Place a limit order on Kalshi.

        ticker: Kalshi market ticker (e.g. "INXD-23JAN4175")
        side:   "yes" or "no"
        size_usd: dollar amount (converted to cents internally)
        price:  probability 0.01–0.99
        """
        if not self.available:
            return KalshiResult(False, None, "kalshi", side, size_usd, price, "not authenticated")

        count = max(1, int(size_usd / price))  # shares = $ / price
        price_cents = int(round(price * 100))

        logger.info(
            "%s KALSHI ORDER: %s %s @ %d¢ | count=%d ($%.2f)",
            "DRY_RUN" if dry_run else "LIVE",
            ticker, side.upper(), price_cents, count, size_usd,
        )

        if dry_run:
            fake_id = f"dry_kalshi_{ticker[:8]}_{int(datetime.now(timezone.utc).timestamp())}"
            return KalshiResult(True, fake_id, "kalshi", side, size_usd, price, "DRY_RUN")

        try:
            resp = self._http.post(
                f"{KALSHI_BASE}/portfolio/orders",
                headers=self._headers,
                json={
                    "ticker": ticker,
                    "action": "buy",
                    "side": side.lower(),
                    "type": "limit",
                    "count": count,
                    "yes_price" if side.lower() == "yes" else "no_price": price_cents,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            order = data.get("order") or {}
            order_id = order.get("order_id") or order.get("id")
            return KalshiResult(True, order_id, "kalshi", side, size_usd, price)
        except Exception as exc:
            logger.error("Kalshi order failed: %s", exc)
            return KalshiResult(False, None, "kalshi", side, size_usd, price, str(exc))

    # ── Settlement divergence scanner ────────────────────────────────────

    def scan_settled_markets(self) -> list[dict]:
        """
        Find markets that Kalshi has settled but are still trading on Polymarket.
        These represent risk-free arbitrage windows.
        Returns list of {ticker, result, settle_time, kalshi_price}.
        """
        if not self.available:
            return []
        try:
            resp = self._http.get(
                f"{KALSHI_BASE}/markets",
                headers=self._headers,
                params={"status": "finalized", "limit": 50},
            )
            if resp.status_code == 200:
                markets = resp.json().get("markets") or []
                settled = []
                for m in markets:
                    if m.get("result") in ("yes", "no"):
                        settled.append({
                            "ticker": m.get("ticker"),
                            "title": m.get("title"),
                            "result": m.get("result"),
                            "settle_price": 1.0 if m.get("result") == "yes" else 0.0,
                        })
                return settled
        except Exception as exc:
            logger.debug("Kalshi settled market scan failed: %s", exc)
        return []

    def close(self) -> None:
        self._http.close()
