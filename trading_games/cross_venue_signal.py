"""
CrossVenueSignal — Polymarket vs Kalshi price divergence detector.

Path B implementation: use PM + Kalshi as live intel feeds, map divergences
to IG spread bets on the underlying event.

Pattern: same event priced differently across venues encodes:
  Kalshi (US professionals, regulated) vs Polymarket (global/crypto crowd)
  Divergence = sentiment split → directional signal for IG

Divergences to exploit:
  1. Fed/macro: PM > Kalshi = global more bullish on cut → long risk (S&P, NASDAQ)
  2. PM < Kalshi = US pros more bullish → ??? (less actionable)
  3. Any event with PM/Kalshi gap > THRESHOLD → trade the correlated IG instrument

All API calls are read-only. No Polymarket or Kalshi account required.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Iterator

import httpx

FORAGE_GRAPH_URL = os.environ.get("FORAGE_GRAPH_URL", "https://forage-graph-production.up.railway.app")
GRAPH_API_SECRET = os.environ.get("GRAPH_API_SECRET", "")

logger = logging.getLogger(__name__)

DIVERGENCE_THRESHOLD = float(os.environ.get("DIVERGENCE_THRESHOLD", "0.04"))   # 4 cents
BOND_THRESHOLD       = float(os.environ.get("BOND_THRESHOLD", "0.97"))          # near-certain

POLYMARKET_GAMMA  = "https://gamma-api.polymarket.com"
KALSHI_BASE       = "https://trading-api.kalshi.com/trade-api/v2"
POLYMARKET_DATA   = "https://data-api.polymarket.com"

# Keywords that link a market topic to an IG instrument category + direction bias
_TOPIC_MAP: list[tuple[list[str], str, str, str]] = [
    # (topic_keywords, ig_category, ig_key, direction_when_pm_higher)
    (["fed rate", "federal reserve", "fomc", "interest rate cut", "rate cut"],
     "indices", "sp500", "BUY"),     # PM more bullish on cut = long risk
    (["fed hike", "rate hike", "hawkish fed"],
     "indices", "sp500", "SELL"),
    (["bitcoin", "btc", "crypto"],
     "indices", "nasdaq", "BUY"),    # crypto bullishness bleeds into NASDAQ
    (["trump", "harris", "democrat", "republican", "us election"],
     "forex", "gbpusd", "SELL"),     # US political risk → GBP/USD neutral/sell
    (["uk election", "keir", "labour", "tory"],
     "forex", "gbpusd", "SELL"),
    (["ukraine", "russia", "war", "nato", "sanctions"],
     "commodities", "oil", "BUY"),   # geopolitical → long oil
    (["oil", "opec", "crude"],
     "commodities", "oil", "BUY"),
    (["gold", "safe haven", "inflation"],
     "commodities", "gold", "BUY"),
    (["recession", "gdp", "economic slowdown"],
     "indices", "ftse", "SELL"),
    (["earnings", "quarterly results", "profit"],
     "indices", "ftse", "BUY"),      # positive earnings = risk-on
]


@dataclass
class DivergenceSignal:
    market_id: str
    question: str
    pm_price: float
    kalshi_price: float
    divergence: float           # PM - Kalshi (positive = PM more bullish)
    ig_category: str
    ig_key: str
    ig_epic: str
    direction: str              # BUY / SELL
    confidence: float
    signal_type: str            # "divergence" / "bond"
    rationale: str

    def to_market_dict(self) -> dict:
        """Convert to standard market dict for agent analyze_market()."""
        return {
            "market_id":    self.market_id,
            "question":     self.question,
            "entity_name":  self.question[:60],
            "entity_type":  "prediction_market_signal",
            "signal_text":  self.rationale,
            "direction":    "bullish" if self.direction == "BUY" else "bearish",
            "confidence":   self.confidence,
            "source":       "cross_venue_divergence",
            "tokens":       [],
            "market_price": self.pm_price,
            "is_fee_free":  False,
            "tick_size":    "0.01",
            "min_order_size": 1.0,
            # IG-specific extras
            "_ig_epic":     self.ig_epic,
            "_ig_direction": self.direction,
            "_signal_type": self.signal_type,
        }


class CrossVenueSignalDetector:
    """
    Detects price divergences between Polymarket and Kalshi on matching markets.
    Emits DivergenceSignal objects for IG execution.
    """

    def __init__(self) -> None:
        self._http = httpx.Client(timeout=12.0)

    # ── Data fetchers ─────────────────────────────────────────────────────────

    def _get_pm_markets(self, limit: int = 100) -> list[dict]:
        """Fetch active Polymarket markets with price > 0 and < 1."""
        try:
            resp = self._http.get(
                f"{POLYMARKET_GAMMA}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "order": "volume24hr",
                    "ascending": "false",
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else (data.get("markets") or [])
        except Exception as exc:
            logger.warning("[CrossVenue] PM fetch failed: %s", exc)
        return []

    def _get_kalshi_markets(self, limit: int = 100) -> list[dict]:
        """Fetch open Kalshi markets."""
        try:
            resp = self._http.get(
                f"{KALSHI_BASE}/markets",
                params={"limit": limit, "status": "open"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                return resp.json().get("markets") or []
        except Exception as exc:
            logger.warning("[CrossVenue] Kalshi fetch failed: %s", exc)
        return []

    # ── Matching ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_yes_price(market: dict) -> float | None:
        """Extract YES/true price from a PM or Kalshi market record."""
        # Polymarket
        for tok in (market.get("tokens") or []):
            if tok.get("outcome", "").upper() == "YES":
                p = tok.get("price")
                if p is not None:
                    return float(p)
        # Polymarket flat fields
        for f in ("lastTradePrice", "bestAsk", "outcomePrices"):
            val = market.get(f)
            if val is not None:
                try:
                    prices = [float(x) for x in (val if isinstance(val, list) else [val])]
                    return prices[0]
                except Exception:
                    pass
        # Kalshi
        yes_ask = market.get("yes_ask")
        yes_bid = market.get("yes_bid")
        if yes_ask and yes_bid:
            try:
                return (float(yes_ask) + float(yes_bid)) / 2 / 100  # Kalshi prices in cents
            except Exception:
                pass
        result_price = market.get("result")
        if result_price is not None:
            try:
                return float(result_price) / 100
            except Exception:
                pass
        return None

    @staticmethod
    def _topic_match(question: str) -> tuple[str, str, str] | None:
        """Return (ig_category, ig_key, direction_when_pm_higher) or None."""
        q = question.lower()
        for keywords, category, key, direction in _TOPIC_MAP:
            if any(kw in q for kw in keywords):
                return category, key, direction
        return None

    @staticmethod
    def _fuzzy_match(q1: str, q2: str) -> bool:
        """Very loose topic match between two market questions."""
        words1 = set(re.findall(r"\b\w{4,}\b", q1.lower()))
        words2 = set(re.findall(r"\b\w{4,}\b", q2.lower()))
        overlap = len(words1 & words2)
        return overlap >= 3

    # ── Main detector ─────────────────────────────────────────────────────────

    def detect(self) -> list[DivergenceSignal]:
        """
        Pull PM + Kalshi, find matching markets, compute divergences.
        Returns list of actionable DivergenceSignal objects.
        """
        from trading_games.ig_epic_mapper import INDICES, FOREX, COMMODITIES
        _epic_tables = {"indices": INDICES, "forex": FOREX, "commodities": COMMODITIES}

        pm_markets     = self._get_pm_markets()
        kalshi_markets = self._get_kalshi_markets()

        logger.info(
            "[CrossVenue] PM=%d markets | Kalshi=%d markets",
            len(pm_markets), len(kalshi_markets),
        )

        signals: list[DivergenceSignal] = []

        # High-probability "bond" signals from PM alone (>BOND_THRESHOLD)
        for pm in pm_markets:
            q     = pm.get("question") or pm.get("title") or ""
            mid   = pm.get("conditionId") or pm.get("id") or ""
            price = self._extract_yes_price(pm)
            if price is None:
                continue

            if price >= BOND_THRESHOLD:
                topic = self._topic_match(q)
                if topic:
                    cat, key, direction = topic
                    epic_info = _epic_tables.get(cat, {}).get(key, {})
                    epic = epic_info.get("epic", "")
                    if not epic:
                        continue
                    signals.append(DivergenceSignal(
                        market_id=mid,
                        question=q,
                        pm_price=price,
                        kalshi_price=0.0,
                        divergence=price - 0.5,
                        ig_category=cat,
                        ig_key=key,
                        ig_epic=epic,
                        direction=direction,
                        confidence=price * 0.85,   # scale back — PM alone
                        signal_type="bond",
                        rationale=(
                            f"PM near-certain at {price:.2f} | "
                            f"topic→IG {cat}/{key} | direction={direction}"
                        ),
                    ))

        # Cross-venue divergence signals
        for pm in pm_markets:
            q_pm  = pm.get("question") or pm.get("title") or ""
            mid   = pm.get("conditionId") or pm.get("id") or ""
            p_pm  = self._extract_yes_price(pm)
            if p_pm is None:
                continue

            for km in kalshi_markets:
                q_k = km.get("title") or km.get("question") or ""
                if not self._fuzzy_match(q_pm, q_k):
                    continue
                p_k = self._extract_yes_price(km)
                if p_k is None:
                    continue

                div = p_pm - p_k
                if abs(div) < DIVERGENCE_THRESHOLD:
                    continue

                topic = self._topic_match(q_pm) or self._topic_match(q_k)
                if not topic:
                    continue

                cat, key, dir_when_pm_higher = topic
                epic_info = _epic_tables.get(cat, {}).get(key, {})
                epic = epic_info.get("epic", "")
                if not epic:
                    continue

                # PM > Kalshi = global more bullish; PM < Kalshi = US pros more bullish
                direction = dir_when_pm_higher if div > 0 else (
                    "SELL" if dir_when_pm_higher == "BUY" else "BUY"
                )
                confidence = min(0.85, 0.5 + abs(div) * 4)

                signals.append(DivergenceSignal(
                    market_id=mid,
                    question=q_pm,
                    pm_price=p_pm,
                    kalshi_price=p_k,
                    divergence=div,
                    ig_category=cat,
                    ig_key=key,
                    ig_epic=epic,
                    direction=direction,
                    confidence=confidence,
                    signal_type="divergence",
                    rationale=(
                        f"PM={p_pm:.2f} Kalshi={p_k:.2f} gap={div:+.2f} | "
                        f"topic→IG {cat}/{key} | direction={direction}"
                    ),
                ))
                break   # one Kalshi match per PM market is enough

        # Sort by confidence desc, deduplicate by epic
        seen_epics: dict[str, DivergenceSignal] = {}
        for s in sorted(signals, key=lambda x: -x.confidence):
            if s.ig_epic not in seen_epics:
                seen_epics[s.ig_epic] = s

        result = list(seen_epics.values())
        logger.info("[CrossVenue] %d actionable signals", len(result))

        # Push everything — PM markets, Kalshi markets, divergence signals — to graph
        self._push_to_graph(pm_markets, kalshi_markets, result)

        return result

    def _push_to_graph(
        self,
        pm_markets: list[dict],
        kalshi_markets: list[dict],
        signals: list["DivergenceSignal"],
    ) -> None:
        if not GRAPH_API_SECRET:
            return
        nodes: list[dict] = []

        # Polymarket market nodes
        for m in pm_markets[:50]:
            q   = m.get("question") or m.get("title") or ""
            mid = m.get("conditionId") or m.get("id") or ""
            if not mid:
                continue
            price = self._extract_yes_price(m)
            nodes.append({
                "id":       f"pm_market_{mid}",
                "type":     "PredictionMarket",
                "name":     q[:200],
                "venue":    "polymarket",
                "yes_price": price,
                "volume":   m.get("volume") or m.get("volume24hr") or 0,
                "end_date": m.get("endDate") or m.get("end_date") or "",
                "source":   "polymarket_api",
            })

        # Kalshi market nodes
        for m in kalshi_markets[:50]:
            q   = m.get("title") or m.get("question") or ""
            mid = m.get("ticker") or m.get("market_id") or m.get("id") or ""
            if not mid:
                continue
            price = self._extract_yes_price(m)
            nodes.append({
                "id":       f"kalshi_market_{mid}",
                "type":     "PredictionMarket",
                "name":     q[:200],
                "venue":    "kalshi",
                "yes_price": price,
                "volume":   m.get("volume") or 0,
                "close_time": m.get("close_time") or "",
                "source":   "kalshi_api",
            })

        # Cross-venue divergence signal nodes
        for s in signals:
            nodes.append({
                "id":          f"crossvenue_{s.market_id}",
                "type":        "Signal",
                "name":        f"Cross-venue: {s.question[:100]}",
                "signal_type": s.signal_type,
                "pm_price":    s.pm_price,
                "kalshi_price": s.kalshi_price,
                "divergence":  s.divergence,
                "ig_epic":     s.ig_epic,
                "direction":   s.direction,
                "confidence":  s.confidence,
                "rationale":   s.rationale,
                "source":      "cross_venue_detector",
            })

        if not nodes:
            return
        try:
            resp = self._http.post(
                f"{FORAGE_GRAPH_URL}/ingest/bulk",
                headers={"Authorization": f"Bearer {GRAPH_API_SECRET}", "Content-Type": "application/json"},
                json={"nodes": nodes, "source": "cross_venue_signal"},
                timeout=10.0,
            )
            logger.info("[CrossVenue] Pushed %d nodes to graph | status=%d", len(nodes), resp.status_code)
        except Exception as exc:
            logger.warning("[CrossVenue] Graph push failed: %s", exc)

    def close(self) -> None:
        self._http.close()
