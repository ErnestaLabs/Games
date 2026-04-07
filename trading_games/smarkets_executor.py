"""
SmarketsExecutor — betting exchange execution via Smarkets REST API v3.

UK-regulated prediction market exchange.
Auth: Bearer token (SMARKETS_API_KEY env var).

Env vars:
  SMARKETS_API_KEY  — your Smarkets API token (already in .env)

Flow:
  1. Search events by type (political, financial, current-affairs)
  2. Find matching market + contract for the signal question
  3. Place BUY (back) or SELL (lay) order
  4. Log result

Smarkets order model:
  - quantity: stake in pennies (GBP × 100)
  - price: odds in basis points (100 = 1.00, 150 = 1.50, 200 = 2.00)
    Smarkets uses Betfair-style decimal odds × 100
  - side: "buy" (back = bet it happens) or "sell" (lay = bet it doesn't)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SMARKETS_BASE = "https://api.smarkets.com/v3"

SMARKETS_API_KEY = os.environ.get("SMARKETS_API_KEY", "")

MIN_STAKE_GBP = float(os.environ.get("SMARKETS_MIN_STAKE", "2.0"))
MAX_STAKE_GBP = float(os.environ.get("SMARKETS_MAX_STAKE", "50.0"))

# Event type IDs to search for prediction markets
# politics, current-affairs, finance, entertainment
SMARKETS_EVENT_TYPES = os.environ.get("SMARKETS_EVENT_TYPES", "politics,current-affairs,finance")


@dataclass
class SmarketsOrder:
    order_id: str
    event_name: str
    market_name: str
    contract_name: str
    side: str          # "buy" or "sell"
    stake_gbp: float
    price_bp: int      # basis points (200 = 2.00 = evens)
    status: str
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def decimal_odds(self) -> float:
        return self.price_bp / 100.0


@dataclass
class SmarketsResult:
    success: bool
    order_id: str | None
    event_name: str
    market_name: str
    contract_name: str
    side: str
    stake_gbp: float
    price_bp: int
    reason: str = ""
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def decimal_odds(self) -> float:
        return self.price_bp / 100.0


class SmarketsExecutor:
    def __init__(self, api_key: str = SMARKETS_API_KEY) -> None:
        self._api_key = api_key
        self._http = httpx.Client(timeout=20.0)
        self._open_orders: dict[str, SmarketsOrder] = {}
        self._log: list[SmarketsResult] = []

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self._api_key)

    # ── Market search ─────────────────────────────────────────────────────────

    def search_events(self, query: str, limit: int = 10) -> list[dict]:
        """Search Smarkets events by name across prediction market types."""
        if not self.is_configured():
            logger.warning("SMARKETS_API_KEY not set")
            return []
        try:
            resp = self._http.get(
                f"{SMARKETS_BASE}/events/",
                headers=self._headers(),
                params={
                    "state": "upcoming,live",
                    "limit": limit,
                    "q": query[:80],
                    "type": SMARKETS_EVENT_TYPES,
                },
            )
            if resp.status_code == 200:
                return resp.json().get("events") or []
            logger.warning("Smarkets event search %s: %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("Smarkets event search error: %s", exc)
        return []

    def get_markets(self, event_id: str) -> list[dict]:
        """Get markets for an event."""
        try:
            resp = self._http.get(
                f"{SMARKETS_BASE}/events/{event_id}/markets/",
                headers=self._headers(),
            )
            if resp.status_code == 200:
                return resp.json().get("markets") or []
        except Exception as exc:
            logger.warning("Smarkets get_markets error: %s", exc)
        return []

    def get_quotes(self, market_id: str) -> dict:
        """Get best bid/ask prices for all contracts in a market."""
        try:
            resp = self._http.get(
                f"{SMARKETS_BASE}/markets/{market_id}/quotes/",
                headers=self._headers(),
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.warning("Smarkets get_quotes error: %s", exc)
        return {}

    def _find_contract(
        self, event: dict, question: str, side: str
    ) -> tuple[str, str, str, int] | None:
        """
        Find (market_id, contract_id, contract_name, best_price_bp).
        Returns None if no tradeable contract found.
        """
        event_id = event.get("id") or event.get("uuid", "")
        markets = self.get_markets(str(event_id))

        q_words = set(w for w in question.lower().split() if len(w) > 3)
        sm_side = "buy" if side.upper() in ("YES", "BUY") else "sell"

        for market in markets:
            if market.get("state") not in ("open", "live"):
                continue
            market_id = str(market.get("id") or market.get("uuid", ""))
            quotes_data = self.get_quotes(market_id)
            contracts = market.get("contracts") or []

            for contract in contracts:
                c_name = (contract.get("name") or "").lower()
                c_id   = str(contract.get("id") or contract.get("uuid", ""))
                # For YES/BUY signals, prefer "Yes"/"Win"/"True" contracts
                is_yes_contract = any(w in c_name for w in ("yes", "true", "win", "happen"))
                if sm_side == "sell":
                    is_yes_contract = not is_yes_contract  # reverse for lay

                if not is_yes_contract and not q_words.intersection(c_name.split()):
                    continue

                # Get best price from quotes
                contract_quotes = (quotes_data.get("quotes") or {}).get(c_id, {})
                ask_prices = contract_quotes.get("buy" if sm_side == "buy" else "sell", [])
                if not ask_prices:
                    continue
                # Smarkets prices in basis points (100 = evens = 2.0 decimal)
                best_bp = max(ask_prices, key=lambda p: p.get("price", 0)) if sm_side == "buy" \
                          else min(ask_prices, key=lambda p: p.get("price", 999))
                price_bp = int(best_bp.get("price", 0))
                if price_bp <= 100:  # invalid (≤1.00 decimal = impossible)
                    continue

                return market_id, c_id, contract.get("name", c_id), price_bp

        return None

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute_from_signal(
        self,
        question: str,
        side: str,
        size_usdc: float,
        edge: float = 0.0,
        event_id: str = "",
    ) -> SmarketsResult:
        """Find Smarkets event/contract for this question and place an order."""
        if not self.is_configured():
            return SmarketsResult(
                success=False, order_id=None,
                event_name=question, market_name="", contract_name="",
                side=side, stake_gbp=0.0, price_bp=0,
                reason="SMARKETS_API_KEY not set",
            )

        stake_gbp = round(min(max(size_usdc, MIN_STAKE_GBP), MAX_STAKE_GBP), 2)

        if event_id:
            events = [{"id": event_id, "name": question}]
        else:
            events = self.search_events(question)

        if not events:
            logger.info("Smarkets: no event found for '%s'", question[:60])
            return SmarketsResult(
                success=False, order_id=None,
                event_name=question, market_name="", contract_name="",
                side=side, stake_gbp=stake_gbp, price_bp=0,
                reason=f"No Smarkets event for: {question[:60]}",
            )

        event = events[0]
        result_info = self._find_contract(event, question, side)

        if not result_info:
            return SmarketsResult(
                success=False, order_id=None,
                event_name=event.get("name", ""), market_name="", contract_name="",
                side=side, stake_gbp=stake_gbp, price_bp=0,
                reason="No tradeable contract/price found",
            )

        market_id, contract_id, contract_name, price_bp = result_info
        sm_side = "buy" if side.upper() in ("YES", "BUY") else "sell"

        logger.info(
            "SMARKETS ORDER: %s %s @ %.2f | stake=£%.2f | edge=%.1f%%",
            sm_side.upper(), contract_name[:40], price_bp / 100.0, stake_gbp, edge * 100,
        )

        return self._place_order(
            market_id=market_id,
            contract_id=contract_id,
            sm_side=sm_side,
            stake_gbp=stake_gbp,
            price_bp=price_bp,
            event_name=event.get("name", ""),
            contract_name=contract_name,
        )

    def _place_order(
        self,
        market_id: str,
        contract_id: str,
        sm_side: str,
        stake_gbp: float,
        price_bp: int,
        event_name: str,
        contract_name: str,
    ) -> SmarketsResult:
        # Smarkets quantity is in pence (GBP × 100)
        quantity_pence = int(stake_gbp * 100)

        try:
            resp = self._http.post(
                f"{SMARKETS_BASE}/orders/",
                headers=self._headers(),
                json={
                    "market_id": market_id,
                    "contract_id": contract_id,
                    "side": sm_side,
                    "price": price_bp,
                    "quantity": quantity_pence,
                    "type": "limit",
                },
            )
            data = resp.json()
            if resp.status_code in (200, 201):
                order = (data.get("orders") or [data])[0] if isinstance(data, dict) else data
                order_id = str(order.get("id") or order.get("uuid", ""))
                status   = order.get("status", "")
                success  = status in ("matched", "open", "partially_matched") or bool(order_id)

                if success:
                    self._open_orders[order_id] = SmarketsOrder(
                        order_id=order_id, event_name=event_name,
                        market_name=market_id, contract_name=contract_name,
                        side=sm_side, stake_gbp=stake_gbp,
                        price_bp=price_bp, status=status,
                    )
                    logger.info(
                        "SMARKETS FILLED: %s %s @ %.2f | id=%s | status=%s",
                        sm_side.upper(), contract_name[:30],
                        price_bp / 100.0, order_id, status,
                    )
                else:
                    logger.warning("Smarkets order not matched: %s | %s", status, data)

                result = SmarketsResult(
                    success=success, order_id=order_id or None,
                    event_name=event_name, market_name=market_id,
                    contract_name=contract_name, side=sm_side,
                    stake_gbp=stake_gbp, price_bp=price_bp,
                    reason="" if success else f"{status}: {data}",
                )
            else:
                reason = data.get("message") or data.get("errors") or resp.text[:200]
                logger.error("Smarkets order failed: %s %s", resp.status_code, reason)
                result = SmarketsResult(
                    success=False, order_id=None,
                    event_name=event_name, market_name=market_id,
                    contract_name=contract_name, side=sm_side,
                    stake_gbp=stake_gbp, price_bp=price_bp,
                    reason=f"HTTP {resp.status_code}: {reason}",
                )
        except Exception as exc:
            logger.error("Smarkets execution error: %s", exc)
            result = SmarketsResult(
                success=False, order_id=None,
                event_name=event_name, market_name=market_id,
                contract_name=contract_name, side=sm_side,
                stake_gbp=stake_gbp, price_bp=price_bp, reason=str(exc),
            )

        self._log.append(result)
        return result

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            resp = self._http.delete(
                f"{SMARKETS_BASE}/orders/{order_id}/",
                headers=self._headers(),
            )
            if resp.status_code in (200, 204):
                self._open_orders.pop(order_id, None)
                logger.info("Smarkets cancelled order %s", order_id)
                return True
            logger.warning("Smarkets cancel failed: %s", resp.status_code)
        except Exception as exc:
            logger.error("Smarkets cancel error: %s", exc)
        return False

    def status_summary(self) -> dict:
        return {
            "configured": self.is_configured(),
            "open_orders": len(self._open_orders),
            "total_executions": len(self._log),
            "successful": sum(1 for r in self._log if r.success),
        }

    def close(self) -> None:
        self._http.close()
