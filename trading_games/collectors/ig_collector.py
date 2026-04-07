"""
IGCollector — live prices and account positions from IG Group REST API.

Pushes to Forage Graph:
  Instrument        — tradeable epics with current bid/offer
  PriceSnapshot     — OHLC / bid-offer snapshots
  Trade             — open positions on the account (read-only mirror)

Auth: CST + X-SECURITY-TOKEN headers obtained via /session.
Re-authenticates on 401 / token expiry.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

IG_BASE = "https://api.ig.com/gateway/deal"
IG_DEMO_BASE = "https://api.ig.com/gateway/deal"   # same URL, demo flag in header

IG_API_KEY    = os.environ.get("IG_API_KEY",    "")
IG_USERNAME   = os.environ.get("IG_USERNAME",   "")
IG_PASSWORD   = os.environ.get("IG_PASSWORD",   "")
IG_ACCOUNT_ID = os.environ.get("IG_ACCOUNT_ID", "")
IG_DEMO       = os.environ.get("IG_DEMO", "true").lower() not in ("false", "0", "no")

# Epics to monitor (from config.py ig_epic_mapper, we watch a curated list)
DEFAULT_EPICS = os.environ.get("IG_WATCH_EPICS", "IX.D.SPTRD.DAILY.IP,IX.D.FTSE.DAILY.IP,CS.D.GBPUSD.TODAY.IP,CS.D.CRUDE.MONTH2.IP,CS.D.GOLD.MONTH3.IP").split(",")


class IGCollector(BaseCollector):
    source_name = "ig_collector"

    def __init__(self) -> None:
        super().__init__()
        self._cst   = ""
        self._token = ""
        self._token_expires = 0.0

    def collect(self) -> list[dict]:
        if not (IG_API_KEY and IG_USERNAME and IG_PASSWORD):
            logger.debug("[ig] credentials not set — skipping")
            return []

        if not self._ensure_session():
            logger.warning("[ig] session unavailable — skipping")
            return []

        nodes: list[dict] = []
        ts = self._ts()

        # Price snapshots for watched epics
        for epic in DEFAULT_EPICS:
            epic = epic.strip()
            if not epic:
                continue
            info = self._fetch_market(epic)
            if not info:
                continue

            snap = info.get("snapshot") or {}
            inst = info.get("instrument") or {}
            name = inst.get("name") or inst.get("epic") or epic
            bid  = snap.get("bid")
            offer = snap.get("offer")
            mid   = ((float(bid) + float(offer)) / 2) if (bid and offer) else None

            nodes.append({
                "id":      f"ig_epic_{epic}",
                "type":    "Instrument",
                "name":    name,
                "venue":   "ig",
                "epic":    epic,
                "bid":     float(bid)  if bid   else None,
                "offer":   float(offer) if offer else None,
                "mid":     mid,
                "type_ig": inst.get("type") or "",
                "currency": inst.get("currencies", [{}])[0].get("code") if inst.get("currencies") else "",
                "source":  self.source_name,
            })

            if mid is not None:
                nodes.append({
                    "id":           f"ig_snap_{epic}_{ts}",
                    "type":         "PriceSnapshot",
                    "instrument_id": f"ig_epic_{epic}",
                    "venue":        "ig",
                    "bid":          float(bid)   if bid   else None,
                    "offer":        float(offer) if offer else None,
                    "mid":          mid,
                    "net_change":   snap.get("netChange"),
                    "pct_change":   snap.get("percentageChange"),
                    "timestamp_ms": ts,
                    "source":       self.source_name,
                })

        # Open positions — passive intel on what's live
        positions = self._fetch_positions()
        for pos in positions:
            p   = pos.get("position") or {}
            mkt = pos.get("market")   or {}
            deal_id = p.get("dealId") or ""
            if not deal_id:
                continue
            nodes.append({
                "id":          f"ig_position_{deal_id}",
                "type":        "Trade",
                "venue":       "ig",
                "deal_id":     deal_id,
                "epic":        mkt.get("epic") or "",
                "direction":   p.get("direction") or "",
                "size":        float(p.get("dealSize") or 0),
                "open_level":  float(p.get("openLevel") or 0),
                "pnl":         float(p.get("upl") or 0),
                "currency":    p.get("currency") or "",
                "source":      self.source_name,
            })

        logger.info("[ig] epics=%d positions=%d total_nodes=%d", len(DEFAULT_EPICS), len(positions), len(nodes))
        return nodes

    # ── Session ───────────────────────────────────────────────────────────────

    def _ensure_session(self) -> bool:
        if self._cst and self._token and time.time() < self._token_expires:
            return True
        try:
            resp = self._http.post(
                f"{IG_BASE}/session",
                headers={
                    "X-IG-API-KEY":   IG_API_KEY,
                    "Content-Type":   "application/json",
                    "Accept":         "application/json; charset=UTF-8",
                    "Version":        "2",
                },
                json={"identifier": IG_USERNAME, "password": IG_PASSWORD, "encryptedPassword": False},
            )
            if resp.status_code == 200:
                self._cst   = resp.headers.get("CST", "")
                self._token = resp.headers.get("X-SECURITY-TOKEN", "")
                self._token_expires = time.time() + 3600
                logger.info("[ig] session created | CST=%s…", self._cst[:8])
                return True
            logger.warning("[ig] login failed: %d %s", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.warning("[ig] login error: %s", exc)
        return False

    def _headers(self) -> dict:
        return {
            "X-IG-API-KEY":      IG_API_KEY,
            "CST":               self._cst,
            "X-SECURITY-TOKEN":  self._token,
            "Accept":            "application/json; charset=UTF-8",
            "Version":           "1",
        }

    def _fetch_market(self, epic: str) -> dict | None:
        try:
            resp = self._http.get(f"{IG_BASE}/markets/{epic}", headers=self._headers())
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                self._cst = ""   # force re-auth next call
            logger.debug("[ig] market %s: %d", epic, resp.status_code)
        except Exception as exc:
            logger.warning("[ig] market fetch %s: %s", epic, exc)
        return None

    def _fetch_positions(self) -> list[dict]:
        try:
            resp = self._http.get(f"{IG_BASE}/positions/otc", headers=self._headers())
            if resp.status_code == 200:
                return resp.json().get("positions") or []
        except Exception as exc:
            logger.warning("[ig] positions fetch: %s", exc)
        return []
