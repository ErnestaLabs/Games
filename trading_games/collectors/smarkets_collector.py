"""
SmarketsCollector — events, markets and quotes from Smarkets prediction exchange.

Pushes to Forage Graph:
  PredictionMarket  — open events
  Instrument        — contracts (YES/NO outcomes)
  OddsSnapshot      — best bid/ask per contract in decimal odds

Auth: Bearer SMARKETS_API_KEY.
Prices stored in basis points internally; converted to decimal for graph.
"""
from __future__ import annotations

import logging
import os
import time

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

SMARKETS_BASE = "https://api.smarkets.com/v3"
SMARKETS_API_KEY = os.environ.get("SMARKETS_API_KEY", "")
SMARKETS_EVENT_TYPES = os.environ.get("SMARKETS_EVENT_TYPES", "politics,current-affairs,finance")
SMARKETS_EVENTS_LIMIT = int(os.environ.get("SMARKETS_EVENTS_LIMIT", "50"))


class SmarketsCollector(BaseCollector):
    source_name = "smarkets_collector"

    def collect(self) -> list[dict]:
        if not SMARKETS_API_KEY:
            logger.debug("[smarkets] API key not set — skipping")
            return []

        nodes: list[dict] = []
        ts = self._ts()
        events = self._fetch_events()

        for ev in events:
            ev_id   = str(ev.get("id") or ev.get("uuid") or "")
            ev_name = ev.get("name") or ev.get("full_name") or ""
            if not ev_id:
                continue

            nodes.append({
                "id":       f"sm_event_{ev_id}",
                "type":     "PredictionMarket",
                "name":     ev_name[:200],
                "venue":    "smarkets",
                "event_id": ev_id,
                "type_slug": ev.get("type") or "",
                "source":   self.source_name,
            })

            markets = self._fetch_markets(ev_id)
            for mkt in markets:
                if mkt.get("state") not in ("open", "live"):
                    continue
                mkt_id = str(mkt.get("id") or mkt.get("uuid") or "")
                if not mkt_id:
                    continue
                quotes_data = self._fetch_quotes(mkt_id)

                for contract in (mkt.get("contracts") or []):
                    c_id   = str(contract.get("id") or contract.get("uuid") or "")
                    c_name = contract.get("name") or ""
                    if not c_id:
                        continue

                    c_quotes = (quotes_data.get("quotes") or {}).get(c_id, {})
                    best_buy  = self._best_bp(c_quotes.get("buy",  []), "buy")
                    best_sell = self._best_bp(c_quotes.get("sell", []), "sell")
                    buy_dec   = best_buy  / 100 if best_buy  else None
                    sell_dec  = best_sell / 100 if best_sell else None

                    nodes.append({
                        "id":        f"sm_contract_{c_id}",
                        "type":      "Instrument",
                        "name":      f"{ev_name} / {mkt.get('name','')} / {c_name}",
                        "venue":     "smarkets",
                        "event_id":  ev_id,
                        "market_id": mkt_id,
                        "contract_id": c_id,
                        "best_buy_dec":  buy_dec,
                        "best_sell_dec": sell_dec,
                        "source":    self.source_name,
                    })

                    if buy_dec is not None:
                        nodes.append({
                            "id":           f"sm_snap_{c_id}_{ts}",
                            "type":         "OddsSnapshot",
                            "market_id":    f"sm_contract_{c_id}",
                            "venue":        "smarkets",
                            "best_buy_dec": buy_dec,
                            "best_sell_dec": sell_dec,
                            "timestamp_ms": ts,
                            "source":       self.source_name,
                        })

        logger.info("[smarkets] events=%d total_nodes=%d", len(events), len(nodes))
        return nodes

    # ── API calls ─────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {SMARKETS_API_KEY}",
            "Accept": "application/json",
        }

    def _fetch_events(self) -> list[dict]:
        try:
            resp = self._http.get(
                f"{SMARKETS_BASE}/events/",
                headers=self._headers(),
                params={"state": "upcoming,live", "limit": SMARKETS_EVENTS_LIMIT, "type": SMARKETS_EVENT_TYPES},
            )
            if resp.status_code == 200:
                return resp.json().get("events") or []
            logger.warning("[smarkets] events fetch: %d %s", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.warning("[smarkets] events error: %s", exc)
        return []

    def _fetch_markets(self, event_id: str) -> list[dict]:
        try:
            resp = self._http.get(f"{SMARKETS_BASE}/events/{event_id}/markets/", headers=self._headers())
            if resp.status_code == 200:
                return resp.json().get("markets") or []
        except Exception as exc:
            logger.warning("[smarkets] markets error: %s", exc)
        return []

    def _fetch_quotes(self, market_id: str) -> dict:
        try:
            resp = self._http.get(f"{SMARKETS_BASE}/markets/{market_id}/quotes/", headers=self._headers())
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.warning("[smarkets] quotes error: %s", exc)
        return {}

    @staticmethod
    def _best_bp(price_list: list, side: str) -> int | None:
        if not price_list:
            return None
        try:
            if side == "buy":
                return max(int(p.get("price", 0)) for p in price_list if p.get("price"))
            else:
                return min(int(p.get("price", 999999)) for p in price_list if p.get("price"))
        except Exception:
            return None
