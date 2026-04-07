"""
MatchbookCollector — live event prices from Matchbook betting exchange.

Pushes to Forage Graph:
  PredictionMarket  — open events with market name
  OddsSnapshot      — best back/lay price per runner
  Instrument        — runners as tradeable instruments

Auth: requires active Matchbook session (username/password).
Reuses a shared session; re-authenticates on expiry.
"""
from __future__ import annotations

import logging
import os
import time

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

MATCHBOOK_EDGE = "https://api.matchbook.com/edge/rest"
MATCHBOOK_BPAPI = "https://api.matchbook.com/bpapi/rest"

MATCHBOOK_USERNAME = os.environ.get("MATCHBOOK_USERNAME", "")
MATCHBOOK_PASSWORD = os.environ.get("MATCHBOOK_PASSWORD", "")
MATCHBOOK_SPORT_IDS = os.environ.get("MATCHBOOK_SPORT_IDS", "15,12")   # politics, specials

EVENTS_LIMIT = int(os.environ.get("MATCHBOOK_EVENTS_LIMIT", "50"))


class MatchbookCollector(BaseCollector):
    source_name = "matchbook_collector"

    def __init__(self) -> None:
        super().__init__()
        self._session_token = ""
        self._session_expires = 0.0

    def collect(self) -> list[dict]:
        if not (MATCHBOOK_USERNAME and MATCHBOOK_PASSWORD):
            logger.debug("[matchbook] credentials not set — skipping")
            return []

        if not self._ensure_session():
            logger.warning("[matchbook] session unavailable — skipping run")
            return []

        nodes: list[dict] = []
        ts = self._ts()
        events = self._fetch_events()

        for ev in events:
            ev_id   = str(ev.get("id") or "")
            ev_name = ev.get("name") or ""
            if not ev_id:
                continue

            nodes.append({
                "id":      f"mb_event_{ev_id}",
                "type":    "PredictionMarket",
                "name":    ev_name[:200],
                "venue":   "matchbook",
                "event_id": ev_id,
                "sport":   str(ev.get("sport-id") or ""),
                "status":  ev.get("status") or "",
                "source":  self.source_name,
            })

            for market in (ev.get("markets") or []):
                if market.get("status") != "open":
                    continue
                m_id   = str(market.get("id") or "")
                m_name = market.get("name") or ""

                for runner in (market.get("runners") or []):
                    if runner.get("status") != "open":
                        continue
                    r_id   = str(runner.get("id") or "")
                    r_name = runner.get("name") or ""
                    prices = runner.get("prices") or []

                    best_back = self._best_price(prices, "back")
                    best_lay  = self._best_price(prices, "lay")

                    nodes.append({
                        "id":        f"mb_runner_{r_id}",
                        "type":      "Instrument",
                        "name":      f"{ev_name} / {m_name} / {r_name}",
                        "venue":     "matchbook",
                        "event_id":  ev_id,
                        "market_id": m_id,
                        "runner_id": r_id,
                        "best_back": best_back,
                        "best_lay":  best_lay,
                        "source":    self.source_name,
                    })

                    if best_back is not None:
                        nodes.append({
                            "id":           f"mb_snap_{r_id}_{ts}",
                            "type":         "OddsSnapshot",
                            "market_id":    f"mb_runner_{r_id}",
                            "venue":        "matchbook",
                            "best_back":    best_back,
                            "best_lay":     best_lay,
                            "timestamp_ms": ts,
                            "source":       self.source_name,
                        })

        logger.info("[matchbook] events=%d total_nodes=%d", len(events), len(nodes))
        return nodes

    # ── Session ───────────────────────────────────────────────────────────────

    def _ensure_session(self) -> bool:
        if self._session_token and time.time() < self._session_expires:
            return True
        try:
            resp = self._http.post(
                f"{MATCHBOOK_BPAPI}/security/session",
                headers={"Content-Type": "application/json"},
                json={"username": MATCHBOOK_USERNAME, "password": MATCHBOOK_PASSWORD},
            )
            if resp.status_code == 200:
                token = resp.json().get("session-token") or resp.headers.get("session-token", "")
                if token:
                    self._session_token   = token
                    self._session_expires = time.time() + 3600
                    return True
            logger.warning("[matchbook] login failed: %d %s", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.warning("[matchbook] login error: %s", exc)
        return False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._session_token:
            h["session-token"] = self._session_token
        return h

    # ── Data fetch ────────────────────────────────────────────────────────────

    def _fetch_events(self) -> list[dict]:
        try:
            params: dict = {"status": "open", "per-page": EVENTS_LIMIT, "include-markets": "true", "include-runners": "true", "include-prices": "true"}
            if MATCHBOOK_SPORT_IDS:
                params["sport-ids"] = MATCHBOOK_SPORT_IDS
            resp = self._http.get(f"{MATCHBOOK_EDGE}/events", headers=self._headers(), params=params)
            if resp.status_code == 200:
                return resp.json().get("events") or []
            logger.warning("[matchbook] events fetch: %d", resp.status_code)
        except Exception as exc:
            logger.warning("[matchbook] events fetch error: %s", exc)
        return []

    @staticmethod
    def _best_price(prices: list[dict], side: str) -> float | None:
        side_prices = [p for p in prices if p.get("side") == side]
        if not side_prices:
            return None
        try:
            if side == "back":
                return max(float(p["odds"]) for p in side_prices if p.get("odds"))
            else:
                return min(float(p["odds"]) for p in side_prices if p.get("odds"))
        except Exception:
            return None
