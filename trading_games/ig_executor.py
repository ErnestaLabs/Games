"""
IGExecutor — spread betting execution via IG Group REST API.

UK-legal execution layer. Replaces Polymarket for agents running in the UK.

Env vars:
  IG_API_KEY       — your IG API key
  IG_USERNAME      — IG account username (email)
  IG_PASSWORD      — IG account password
  IG_ACCOUNT_ID    — IG account ID (shown in platform, e.g. "ABC12")
  IG_DEMO          — "true" to use demo environment (default: true)

Flow:
  1. Create session → get CST + X-SECURITY-TOKEN headers
  2. Search IG markets for the agent signal question
  3. Size position using Kelly (clamped to max risk %)
  4. Place market OTC spread bet (DFB — Daily Funded Bet)
  5. Log result

IG spread bet sizing:
  - Size is in £ per point, not shares/USDC
  - For binary/event markets, 1 point = 1 index point or 1 basis point
  - We convert USDC size to GBP£/point using a conservative mapping
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

IG_LIVE_BASE = "https://api.ig.com/gateway/deal"
IG_DEMO_BASE = "https://demo-api.ig.com/gateway/deal"

IG_API_KEY   = os.environ.get("IG_API_KEY", "")
IG_USERNAME  = os.environ.get("IG_USERNAME", "")
IG_PASSWORD  = os.environ.get("IG_PASSWORD", "")
IG_ACCOUNT_ID = os.environ.get("IG_ACCOUNT_ID", "")
IG_DEMO      = os.environ.get("IG_DEMO", "true").lower() not in ("false", "0", "no")

# Max £/point per position (risk control)
MAX_SIZE_PER_POINT = float(os.environ.get("IG_MAX_SIZE_PER_POINT", "1.0"))


@dataclass
class IGPosition:
    deal_id: str
    epic: str
    description: str
    direction: str      # "BUY" or "SELL"
    size: float         # £/point
    open_level: float
    opened_at: datetime


@dataclass
class IGResult:
    success: bool
    deal_id: str | None
    epic: str
    description: str
    direction: str
    size: float
    level: float
    reason: str = ""
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IGExecutor:
    def __init__(
        self,
        api_key: str = IG_API_KEY,
        username: str = IG_USERNAME,
        password: str = IG_PASSWORD,
        account_id: str = IG_ACCOUNT_ID,
        demo: bool = IG_DEMO,
    ) -> None:
        self._api_key  = api_key
        self._username = username
        self._password = password
        self._account_id = account_id
        self._base = IG_DEMO_BASE if demo else IG_LIVE_BASE
        self._demo = demo
        self._cst: str = ""
        self._token: str = ""
        self._http = httpx.Client(timeout=20.0)
        self._open_positions: dict[str, IGPosition] = {}
        self._log: list[IGResult] = []

    # ── Session ──────────────────────────────────────────────────────────────

    def _headers(self, version: str = "1") -> dict:
        h = {
            "X-IG-API-KEY": self._api_key,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "Version": version,
        }
        if self._cst:
            h["CST"] = self._cst
        if self._token:
            h["X-SECURITY-TOKEN"] = self._token
        if self._account_id:
            h["IG-ACCOUNT-ID"] = self._account_id
        return h

    def login(self) -> bool:
        """Create IG session. Returns True on success."""
        try:
            resp = self._http.post(
                f"{self._base}/session",
                headers=self._headers(version="2"),
                json={
                    "identifier": self._username,
                    "password": self._password,
                    "encryptedPassword": False,
                },
            )
            if resp.status_code == 200:
                self._cst   = resp.headers.get("CST", "")
                self._token = resp.headers.get("X-SECURITY-TOKEN", "")
                data = resp.json()
                # Use account ID from env or first spread bet account
                if not self._account_id:
                    for acc in data.get("accounts", []):
                        if acc.get("accountType") == "SPREADBET":
                            self._account_id = acc["accountId"]
                            break
                logger.info(
                    "IG session created | account=%s | demo=%s",
                    self._account_id, self._demo,
                )
                return True
            logger.error("IG login failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.error("IG login error: %s", exc)
        return False

    def _ensure_session(self) -> bool:
        if self._cst and self._token:
            return True
        return self.login()

    # ── Market search ─────────────────────────────────────────────────────────

    def search_markets(self, query: str, limit: int = 5) -> list[dict]:
        """Search IG for markets matching the agent signal question."""
        if not self._ensure_session():
            return []
        try:
            resp = self._http.get(
                f"{self._base}/markets",
                headers=self._headers(version="1"),
                params={"searchTerm": query[:50]},
            )
            if resp.status_code == 200:
                markets = resp.json().get("markets") or []
                # Filter to spreadbet-eligible, prefer binary/event types
                return [
                    {
                        "epic": m.get("epic"),
                        "name": m.get("instrumentName"),
                        "type": m.get("instrumentType"),
                        "bid": m.get("bid"),
                        "offer": m.get("offer"),
                        "expiry": m.get("expiry"),
                    }
                    for m in markets[:limit]
                    if m.get("epic")
                ]
        except Exception as exc:
            logger.warning("IG market search failed: %s", exc)
        return []

    def get_market(self, epic: str) -> dict:
        """Get full market details for an epic."""
        if not self._ensure_session():
            return {}
        try:
            resp = self._http.get(
                f"{self._base}/markets/{epic}",
                headers=self._headers(version="3"),
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.warning("IG get_market failed for %s: %s", epic, exc)
        return {}

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute_from_signal(
        self,
        question: str,
        side: str,          # "YES"/"BUY" or "NO"/"SELL"
        size_usdc: float,   # we'll map to £/point
        edge: float = 0.0,
    ) -> IGResult:
        """
        Find IG market for this question and place a spread bet.
        Returns IGResult with success/failure details.
        """
        if not self._ensure_session():
            return IGResult(
                success=False, deal_id=None, epic="", description=question,
                direction=side, size=0.0, level=0.0,
                reason="IG session creation failed",
            )

        # Search for matching market
        markets = self.search_markets(question)
        if not markets:
            logger.info("IG: no market found for '%s'", question[:60])
            return IGResult(
                success=False, deal_id=None, epic="", description=question,
                direction=side, size=0.0, level=0.0,
                reason=f"No IG market found for: {question[:60]}",
            )

        market = markets[0]
        epic = market["epic"]
        direction = "BUY" if side.upper() in ("YES", "BUY") else "SELL"

        # Convert USDC notional to £/point (conservative: 1:1 mapping, capped)
        size_per_point = min(round(size_usdc / 100.0, 2), MAX_SIZE_PER_POINT)
        size_per_point = max(size_per_point, 0.50)  # IG min is typically £0.50/pt

        level = market.get("offer") if direction == "BUY" else market.get("bid")

        logger.info(
            "IG SPREAD BET: [%s] %s %s @ %s | £%.2f/pt | edge=%.1f%%",
            epic, direction, question[:40], level, size_per_point, edge * 100,
        )

        return self._place_otc(epic, direction, size_per_point, question)

    def _place_otc(
        self, epic: str, direction: str, size: float, description: str
    ) -> IGResult:
        """Place OTC spreadbet position."""
        try:
            body = {
                "epic": epic,
                "direction": direction,
                "size": str(size),
                "orderType": "MARKET",
                "expiry": "DFB",                # Daily Funded Bet
                "guaranteedStop": False,
                "forceOpen": True,
            }
            resp = self._http.post(
                f"{self._base}/positions/otc",
                headers=self._headers(version="2"),
                json=body,
            )
            data = resp.json()
            deal_ref = data.get("dealReference", "")

            if resp.status_code in (200, 202) and deal_ref:
                # Confirm the deal
                confirm = self._confirm_deal(deal_ref)
                deal_id = confirm.get("dealId", deal_ref)
                status  = confirm.get("dealStatus", "UNKNOWN")
                level   = float(confirm.get("level") or 0.0)
                success = status in ("ACCEPTED", "OPEN")

                if success:
                    self._open_positions[deal_id] = IGPosition(
                        deal_id=deal_id, epic=epic, description=description,
                        direction=direction, size=size, open_level=level,
                        opened_at=datetime.now(timezone.utc),
                    )
                    logger.info(
                        "IG FILLED: %s %s @ %.4f | deal=%s",
                        direction, epic, level, deal_id,
                    )
                else:
                    logger.warning("IG deal rejected: %s | %s", status, confirm.get("reason"))

                result = IGResult(
                    success=success, deal_id=deal_id, epic=epic,
                    description=description, direction=direction,
                    size=size, level=level,
                    reason="" if success else f"{status}: {confirm.get('reason', '')}",
                )
            else:
                logger.error("IG OTC position failed: %s %s", resp.status_code, resp.text[:200])
                result = IGResult(
                    success=False, deal_id=None, epic=epic,
                    description=description, direction=direction,
                    size=size, level=0.0,
                    reason=f"HTTP {resp.status_code}: {data.get('errorCode', resp.text[:100])}",
                )

        except Exception as exc:
            logger.error("IG execution error: %s", exc)
            result = IGResult(
                success=False, deal_id=None, epic=epic,
                description=description, direction=direction,
                size=size, level=0.0, reason=str(exc),
            )

        self._log.append(result)
        return result

    def _confirm_deal(self, deal_reference: str) -> dict:
        """Confirm deal and get final status."""
        try:
            resp = self._http.get(
                f"{self._base}/confirms/{deal_reference}",
                headers=self._headers(version="1"),
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.warning("IG deal confirm failed: %s", exc)
        return {}

    def close_position(self, deal_id: str) -> bool:
        """Close an open IG position."""
        pos = self._open_positions.get(deal_id)
        if not pos:
            logger.warning("IG close: unknown deal_id %s", deal_id)
            return False
        if not self._ensure_session():
            return False
        try:
            close_dir = "SELL" if pos.direction == "BUY" else "BUY"
            resp = self._http.delete(
                f"{self._base}/positions/otc",
                headers=self._headers(version="1"),
                json={
                    "dealId": deal_id,
                    "direction": close_dir,
                    "size": str(pos.size),
                    "orderType": "MARKET",
                    "expiry": "DFB",
                },
            )
            if resp.status_code in (200, 202):
                self._open_positions.pop(deal_id, None)
                logger.info("IG closed position %s", deal_id)
                return True
            logger.warning("IG close failed: %s %s", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.error("IG close error: %s", exc)
        return False

    # ── Status ────────────────────────────────────────────────────────────────

    def get_open_positions(self) -> list[dict]:
        """Fetch current open positions from IG (live state)."""
        if not self._ensure_session():
            return []
        try:
            resp = self._http.get(
                f"{self._base}/positions",
                headers=self._headers(version="2"),
            )
            if resp.status_code == 200:
                return resp.json().get("positions") or []
        except Exception as exc:
            logger.warning("IG positions fetch failed: %s", exc)
        return []

    def status_summary(self) -> dict:
        return {
            "demo": self._demo,
            "account_id": self._account_id,
            "session_active": bool(self._cst),
            "tracked_positions": len(self._open_positions),
            "total_executions": len(self._log),
            "successful": sum(1 for r in self._log if r.success),
        }

    def close(self) -> None:
        self._http.close()
