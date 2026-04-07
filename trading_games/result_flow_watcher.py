"""
ResultFlow_Watcher — outcome resolver and P&L tracker.

Monitors resolved Polymarket and Kalshi markets and pushes:
  - Resolved market outcomes as Signal nodes (ground truth)
  - Agent P&L updates as Trade nodes (win/loss realised)
  - Closed IG positions with realised P&L

This closes the feedback loop: predictions → trade → outcome → learning.

Schedule:
  - Resolved markets check  every 300 s (5 min)
  - IG closed positions     every 60  s

Run standalone:
  python -m trading_games.result_flow_watcher

Or imported and started in a thread from agent_runner.py.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("result_flow_watcher")

FORAGE_GRAPH_URL = os.environ.get("FORAGE_GRAPH_URL", "https://forage-graph-production.up.railway.app")
GRAPH_API_SECRET = os.environ.get("GRAPH_API_SECRET", "")

GAMMA_API    = "https://gamma-api.polymarket.com"
KALSHI_BASE  = "https://trading-api.kalshi.com/trade-api/v2"
IG_BASE      = "https://api.ig.com/gateway/deal"

IG_API_KEY    = os.environ.get("IG_API_KEY",    "")
IG_USERNAME   = os.environ.get("IG_USERNAME",   "")
IG_PASSWORD   = os.environ.get("IG_PASSWORD",   "")

RESOLVE_INTERVAL_S  = int(os.environ.get("RESULT_RESOLVE_INTERVAL", "300"))
IG_POSITIONS_INTERVAL_S = int(os.environ.get("RESULT_IG_INTERVAL", "60"))


class ResultFlowWatcher:
    def __init__(self) -> None:
        self._http        = httpx.Client(timeout=15.0)
        self._stop_event  = threading.Event()
        self._run_count   = 0
        self._ig_cst      = ""
        self._ig_token    = ""
        self._ig_token_exp = 0.0

    def start(self) -> None:
        logger.info("ResultFlow_Watcher starting")
        t_resolve  = threading.Thread(target=self._resolve_loop, daemon=True, name="ResultFlow_Resolve")
        t_ig       = threading.Thread(target=self._ig_loop,      daemon=True, name="ResultFlow_IG")
        t_resolve.start()
        t_ig.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._http.close()

    # ── Resolution loop ───────────────────────────────────────────────────────

    def _resolve_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                nodes = []
                nodes += self._fetch_resolved_pm()
                nodes += self._fetch_resolved_kalshi()
                if nodes:
                    self._push(nodes)
                self._run_count += 1
                logger.info("ResultFlow resolve run #%d | nodes=%d", self._run_count, len(nodes))
            except Exception as exc:
                logger.error("ResultFlow resolve error: %s", exc)
            self._stop_event.wait(RESOLVE_INTERVAL_S)

    def _fetch_resolved_pm(self) -> list[dict]:
        """Fetch recently resolved Polymarket markets."""
        nodes: list[dict] = []
        ts = int(time.time() * 1000)
        try:
            resp = self._http.get(
                f"{GAMMA_API}/markets",
                params={"closed": "true", "limit": 50, "order": "updatedAt", "ascending": "false"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return []
            markets = resp.json()
            if isinstance(markets, dict):
                markets = markets.get("markets") or []
            for m in markets:
                mid     = m.get("conditionId") or m.get("id") or ""
                q       = m.get("question") or m.get("title") or ""
                outcome = m.get("winnerOutcome") or m.get("resolvedOutcome") or ""
                if not mid or not outcome:
                    continue
                nodes.append({
                    "id":          f"pm_result_{mid}",
                    "type":        "Signal",
                    "name":        f"RESOLVED: {q[:150]}",
                    "signal_type": "market_resolution",
                    "venue":       "polymarket",
                    "market_id":   mid,
                    "outcome":     outcome,
                    "question":    q[:300],
                    "confidence":  1.0,
                    "timestamp_ms": ts,
                    "source":      "result_flow_watcher",
                })
        except Exception as exc:
            logger.warning("ResultFlow PM resolve error: %s", exc)
        return nodes

    def _fetch_resolved_kalshi(self) -> list[dict]:
        """Fetch recently settled Kalshi markets."""
        nodes: list[dict] = []
        ts = int(time.time() * 1000)
        try:
            resp = self._http.get(
                f"{KALSHI_BASE}/markets",
                params={"limit": 50, "status": "finalized"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return []
            for m in (resp.json().get("markets") or []):
                ticker  = m.get("ticker") or m.get("id") or ""
                title   = m.get("title") or ""
                result  = m.get("result") or ""
                if not ticker or not result:
                    continue
                nodes.append({
                    "id":          f"kalshi_result_{ticker}",
                    "type":        "Signal",
                    "name":        f"RESOLVED: {title[:150]}",
                    "signal_type": "market_resolution",
                    "venue":       "kalshi",
                    "ticker":      ticker,
                    "outcome":     result,
                    "question":    title[:300],
                    "confidence":  1.0,
                    "timestamp_ms": ts,
                    "source":      "result_flow_watcher",
                })
        except Exception as exc:
            logger.warning("ResultFlow Kalshi resolve error: %s", exc)
        return nodes

    # ── IG closed positions loop ──────────────────────────────────────────────

    def _ig_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if IG_API_KEY and IG_USERNAME and IG_PASSWORD:
                    if self._ensure_ig_session():
                        nodes = self._fetch_ig_closed()
                        if nodes:
                            self._push(nodes)
                            logger.info("ResultFlow IG | pushed %d closed-position nodes", len(nodes))
            except Exception as exc:
                logger.error("ResultFlow IG error: %s", exc)
            self._stop_event.wait(IG_POSITIONS_INTERVAL_S)

    def _ensure_ig_session(self) -> bool:
        if self._ig_cst and self._ig_token and time.time() < self._ig_token_exp:
            return True
        try:
            resp = self._http.post(
                f"{IG_BASE}/session",
                headers={"X-IG-API-KEY": IG_API_KEY, "Content-Type": "application/json",
                         "Accept": "application/json; charset=UTF-8", "Version": "2"},
                json={"identifier": IG_USERNAME, "password": IG_PASSWORD, "encryptedPassword": False},
            )
            if resp.status_code == 200:
                self._ig_cst   = resp.headers.get("CST", "")
                self._ig_token = resp.headers.get("X-SECURITY-TOKEN", "")
                self._ig_token_exp = time.time() + 3600
                return True
        except Exception as exc:
            logger.warning("ResultFlow IG login error: %s", exc)
        return False

    def _ig_headers(self) -> dict:
        return {
            "X-IG-API-KEY":     IG_API_KEY,
            "CST":              self._ig_cst,
            "X-SECURITY-TOKEN": self._ig_token,
            "Accept":           "application/json; charset=UTF-8",
            "Version":          "1",
        }

    def _fetch_ig_closed(self) -> list[dict]:
        """Fetch closed/realised IG positions from activity history."""
        nodes: list[dict] = []
        ts = int(time.time() * 1000)
        try:
            resp = self._http.get(
                f"{IG_BASE}/history/activity",
                headers={**self._ig_headers(), "Version": "3"},
                params={"pageSize": 20},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return []
            for act in (resp.json().get("activities") or []):
                if act.get("type") not in ("POSITION", "TRADE"):
                    continue
                detail = act.get("details") or {}
                deal_id = act.get("dealId") or act.get("transactionId") or ""
                if not deal_id:
                    continue
                nodes.append({
                    "id":          f"ig_closed_{deal_id}",
                    "type":        "Trade",
                    "venue":       "ig",
                    "status":      "closed",
                    "deal_id":     deal_id,
                    "epic":        detail.get("epic") or act.get("epic") or "",
                    "direction":   detail.get("direction") or "",
                    "size":        float(detail.get("size") or 0),
                    "open_level":  float(detail.get("openLevel") or 0),
                    "close_level": float(detail.get("closeLevel") or 0),
                    "pnl":         float(detail.get("profit") or 0),
                    "currency":    detail.get("currency") or "",
                    "timestamp_ms": ts,
                    "source":      "result_flow_watcher",
                })
        except Exception as exc:
            logger.warning("ResultFlow IG closed positions error: %s", exc)
        return nodes

    # ── Graph push ────────────────────────────────────────────────────────────

    def _push(self, nodes: list[dict]) -> None:
        if not GRAPH_API_SECRET:
            return
        try:
            resp = self._http.post(
                f"{FORAGE_GRAPH_URL}/ingest/bulk",
                headers={"Authorization": f"Bearer {GRAPH_API_SECRET}", "Content-Type": "application/json"},
                json={"nodes": nodes, "source": "result_flow_watcher"},
                timeout=12.0,
            )
            logger.debug("ResultFlow push %d nodes | status=%d", len(nodes), resp.status_code)
        except Exception as exc:
            logger.warning("ResultFlow graph push failed: %s", exc)

    def status(self) -> dict:
        return {"runs": self._run_count, "ig_session_active": bool(self._ig_cst)}


def main() -> None:
    watcher = ResultFlowWatcher()
    watcher.start()
    try:
        while True:
            time.sleep(60)
            logger.info("ResultFlow status: %s", watcher.status())
    except KeyboardInterrupt:
        watcher.stop()


if __name__ == "__main__":
    main()
