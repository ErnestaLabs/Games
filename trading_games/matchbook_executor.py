"""
MatchbookExecutor — betting exchange execution via Matchbook REST API.

UK-regulated exchange. Auth: username/password → session-token (no API key).
Session token is short-lived and refreshed automatically.

Env vars:
  MATCHBOOK_USERNAME  — your Matchbook login email
  MATCHBOOK_PASSWORD  — your Matchbook password

Flow:
  1. POST /bpapi/rest/security/session → get session-token
  2. Search events for the signal question (sport-ids=15 = politics/specials)
  3. Find matching market + runner
  4. Place BACK (BUY) or LAY (SELL) order
  5. Log result, refresh session when expired
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MATCHBOOK_BASE  = "https://api.matchbook.com"
MATCHBOOK_BPAPI = f"{MATCHBOOK_BASE}/bpapi/rest"
MATCHBOOK_EDGE  = f"{MATCHBOOK_BASE}/edge/rest"

MATCHBOOK_USERNAME = os.environ.get("MATCHBOOK_USERNAME", "")
MATCHBOOK_PASSWORD = os.environ.get("MATCHBOOK_PASSWORD", "")

# Sport IDs: 15 = Politics, 12 = Specials/Other, use both for prediction markets
PREDICTION_SPORT_IDS = os.environ.get("MATCHBOOK_SPORT_IDS", "15,12")

MIN_STAKE_GBP = float(os.environ.get("MATCHBOOK_MIN_STAKE", "2.0"))    # £2 min
MAX_STAKE_GBP = float(os.environ.get("MATCHBOOK_MAX_STAKE", "50.0"))   # £50 max


@dataclass
class MatchbookOrder:
    bet_id: str
    event_name: str
    market_name: str
    runner_name: str
    side: str          # "back" or "lay"
    stake: float       # GBP
    odds: float        # decimal odds
    status: str
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MatchbookResult:
    success: bool
    bet_id: str | None
    event_name: str
    market_name: str
    runner_name: str
    side: str
    stake: float
    odds: float
    reason: str = ""
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MatchbookExecutor:
    def __init__(
        self,
        username: str = MATCHBOOK_USERNAME,
        password: str = MATCHBOOK_PASSWORD,
    ) -> None:
        self._username = username
        self._password = password
        self._session_token: str = ""
        self._session_expires: float = 0.0
        self._http = httpx.Client(timeout=20.0)
        self._open_orders: dict[str, MatchbookOrder] = {}
        self._log: list[MatchbookResult] = []
        self._account_locked: bool = False  # set True on 423 — stops all retries

    # ── Session ──────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._session_token:
            h["session-token"] = self._session_token
        return h

    def login(self) -> bool:
        """Authenticate with username/password, store session token."""
        if self._account_locked:
            return False
        if not (self._username and self._password):
            logger.warning("MATCHBOOK_USERNAME/PASSWORD not set")
            return False
        try:
            resp = self._http.post(
                f"{MATCHBOOK_BPAPI}/security/session",
                headers={"Content-Type": "application/json"},
                json={"username": self._username, "password": self._password},
            )
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("session-token") or resp.headers.get("session-token", "")
                if token:
                    self._session_token   = token
                    self._session_expires = time.time() + 3600  # 1h conservative TTL
                    logger.info("Matchbook session created | user=%s", self._username)
                    return True
                logger.error("Matchbook login: no session-token in response")
            elif resp.status_code == 423:
                # Account locked (Code 1301) — permanent error, stop retrying
                self._account_locked = True
                logger.warning(
                    "Matchbook account LOCKED (423 Code 1301) — "
                    "contact Matchbook support to unlock. All Matchbook execution disabled for this session."
                )
            else:
                logger.error("Matchbook login failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.error("Matchbook login error: %s", exc)
        return False

    def _ensure_session(self) -> bool:
        if self._account_locked:
            return False
        if self._session_token and time.time() < self._session_expires:
            return True
        return self.login()

    # ── Market search ─────────────────────────────────────────────────────────

    def search_events(self, query: str, sport_ids: str = PREDICTION_SPORT_IDS) -> list[dict]:
        """Search Matchbook events by name substring."""
        if not self._ensure_session():
            return []
        try:
            params: dict[str, Any] = {
                "status": "open",
                "per-page": 20,
            }
            if sport_ids:
                params["sport-ids"] = sport_ids

            resp = self._http.get(
                f"{MATCHBOOK_EDGE}/events",
                headers=self._headers(),
                params=params,
            )
            if resp.status_code == 200:
                events = resp.json().get("events") or []
                q = query.lower()
                # Fuzzy filter: keep events whose name shares words with query
                query_words = set(w for w in q.split() if len(w) > 3)
                matched = []
                for ev in events:
                    name = (ev.get("name") or "").lower()
                    if query_words and any(w in name for w in query_words):
                        matched.append(ev)
                    elif q[:20] in name:
                        matched.append(ev)
                return matched[:5]
        except Exception as exc:
            logger.warning("Matchbook event search failed: %s", exc)
        return []

    def _find_runner(
        self, event: dict, question: str, side: str
    ) -> tuple[str, str, str, float] | None:
        """
        Find (market_id, runner_id, runner_name, best_odds) for this question.
        Returns None if no suitable runner found.
        """
        for market in (event.get("markets") or []):
            if market.get("status") != "open":
                continue
            m_name = market.get("name") or ""
            for runner in (market.get("runners") or []):
                if runner.get("status") != "open":
                    continue
                r_name = runner.get("name") or ""
                # For YES/BUY signals, look for a YES/WIN runner with back prices
                prices = runner.get("prices") or []
                target_side = "back" if side.upper() in ("YES", "BUY") else "lay"
                side_prices = [p for p in prices if p.get("side") == target_side]
                if not side_prices:
                    continue
                best = max(side_prices, key=lambda p: p.get("odds", 0) if target_side == "back"
                           else -p.get("odds", 999))
                odds = float(best.get("odds", 0))
                if odds <= 1.0:
                    continue
                return str(market["id"]), str(runner["id"]), r_name, odds
        return None

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute_from_signal(
        self,
        question: str,
        side: str,
        size_usdc: float,
        edge: float = 0.0,
        event_id: str = "",
    ) -> MatchbookResult:
        """Find Matchbook event/runner for this question and place a back/lay bet."""
        if not self._ensure_session():
            return MatchbookResult(
                success=False, bet_id=None,
                event_name=question, market_name="", runner_name="",
                side=side, stake=0.0, odds=0.0,
                reason="Matchbook session creation failed",
            )

        # Convert USDC → GBP stake (1:1 approx, then clamp)
        stake_gbp = round(min(max(size_usdc, MIN_STAKE_GBP), MAX_STAKE_GBP), 2)

        # Find event
        if event_id:
            events = self._get_event(event_id)
        else:
            events = self.search_events(question)

        if not events:
            logger.info("Matchbook: no event found for '%s'", question[:60])
            return MatchbookResult(
                success=False, bet_id=None,
                event_name=question, market_name="", runner_name="",
                side=side, stake=stake_gbp, odds=0.0,
                reason=f"No Matchbook event for: {question[:60]}",
            )

        event = events[0] if isinstance(events, list) else events
        runner_info = self._find_runner(event, question, side)

        if not runner_info:
            logger.info("Matchbook: no runner found in event '%s'", event.get("name", ""))
            return MatchbookResult(
                success=False, bet_id=None,
                event_name=event.get("name", ""), market_name="", runner_name="",
                side=side, stake=stake_gbp, odds=0.0,
                reason="No suitable runner/price found",
            )

        market_id, runner_id, runner_name, odds = runner_info
        mb_side = "back" if side.upper() in ("YES", "BUY") else "lay"

        logger.info(
            "MATCHBOOK BET: %s %s @ %.3f | stake=£%.2f | edge=%.1f%%",
            mb_side.upper(), runner_name[:40], odds, stake_gbp, edge * 100,
        )

        return self._place_bet(
            event_id=str(event["id"]),
            market_id=market_id,
            runner_id=runner_id,
            side=mb_side,
            stake=stake_gbp,
            odds=odds,
            event_name=event.get("name", ""),
            runner_name=runner_name,
        )

    def _get_event(self, event_id: str) -> list[dict]:
        try:
            resp = self._http.get(
                f"{MATCHBOOK_EDGE}/events/{event_id}",
                headers=self._headers(),
            )
            if resp.status_code == 200:
                return [resp.json()]
        except Exception as exc:
            logger.warning("Matchbook get_event failed: %s", exc)
        return []

    def _place_bet(
        self,
        event_id: str,
        market_id: str,
        runner_id: str,
        side: str,
        stake: float,
        odds: float,
        event_name: str,
        runner_name: str,
    ) -> MatchbookResult:
        try:
            resp = self._http.post(
                f"{MATCHBOOK_EDGE}/orders",
                headers=self._headers(),
                json={
                    "odds": odds,
                    "stake": stake,
                    "side": side,
                    "event-id": int(event_id),
                    "market-id": int(market_id),
                    "runner-id": int(runner_id),
                    "keep-in-play": False,
                    "exchange-type": "back-lay",
                    "currency": "GBP",
                },
            )
            data = resp.json()
            if resp.status_code in (200, 201):
                orders = data.get("orders") or [data]
                order = orders[0] if orders else {}
                bet_id = str(order.get("id", ""))
                status = order.get("status", "")
                success = status in ("matched", "open", "partially-matched") or bool(bet_id)

                if success:
                    self._open_orders[bet_id] = MatchbookOrder(
                        bet_id=bet_id, event_name=event_name, market_name=market_id,
                        runner_name=runner_name, side=side,
                        stake=stake, odds=odds, status=status,
                    )
                    logger.info(
                        "MATCHBOOK FILLED: %s %s @ %.3f | bet_id=%s | status=%s",
                        side.upper(), runner_name[:30], odds, bet_id, status,
                    )
                else:
                    logger.warning("Matchbook bet not filled: %s | %s", status, data)

                result = MatchbookResult(
                    success=success, bet_id=bet_id or None,
                    event_name=event_name, market_name=market_id,
                    runner_name=runner_name, side=side,
                    stake=stake, odds=odds,
                    reason="" if success else f"{status}: {data}",
                )
            else:
                reason = data.get("errors") or resp.text[:200]
                logger.error("Matchbook bet failed: %s %s", resp.status_code, reason)
                result = MatchbookResult(
                    success=False, bet_id=None,
                    event_name=event_name, market_name=market_id,
                    runner_name=runner_name, side=side,
                    stake=stake, odds=odds,
                    reason=f"HTTP {resp.status_code}: {reason}",
                )
        except Exception as exc:
            logger.error("Matchbook execution error: %s", exc)
            result = MatchbookResult(
                success=False, bet_id=None,
                event_name=event_name, market_name=market_id,
                runner_name=runner_name, side=side,
                stake=stake, odds=0.0, reason=str(exc),
            )

        self._log.append(result)
        return result

    def logout(self) -> None:
        try:
            self._http.delete(
                f"{MATCHBOOK_BPAPI}/security/session",
                headers=self._headers(),
            )
        except Exception:
            pass
        self._session_token = ""

    def status_summary(self) -> dict:
        return {
            "session_active": bool(self._session_token),
            "open_orders": len(self._open_orders),
            "total_executions": len(self._log),
            "successful": sum(1 for r in self._log if r.success),
        }

    def close(self) -> None:
        self.logout()
        self._http.close()
