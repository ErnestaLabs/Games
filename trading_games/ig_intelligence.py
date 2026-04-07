"""
IGIntelligence — IG Group data ingestion + trade-idea engine.

Polling loop:
  1. Fetch IG quotes, news, and economic calendar via REST
  2. Normalise into structured graph nodes (Instrument, PriceSnapshot,
     Event, CalendarEvent, NewsItem, Signal)
  3. Feed enriched nodes to the Forage Reality Graph via /ingest/bulk
  4. Run pattern detection → emit IGTradeIdea objects

Patterns detected:
  A. Economic release surprise: actual vs forecast divergence → lag trade
  B. Signal alignment: Autochartist/PIA + macro news → directional conviction
  C. Geo/energy shock: oil, gold, FX under-reaction relative to graph patterns

Output: list[IGTradeIdea] — structured BUY/SELL proposals with entry, stop, rationale.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from trading_games.ig_executor import IGExecutor
from trading_games.ig_epic_mapper import INDICES, FOREX, COMMODITIES

logger = logging.getLogger(__name__)

FORAGE_GRAPH_URL = os.environ.get("FORAGE_GRAPH_URL", "https://forage-graph-production.up.railway.app")
GRAPH_API_SECRET = os.environ.get("GRAPH_API_SECRET", "")

# Watchlist: the tight universe we trade
WATCHLIST: dict[str, dict] = {
    **INDICES,
    **FOREX,
    **COMMODITIES,
    # Thematic
    "ai_index":     {"epic": "IX.D.AIINDX.DAILY.IP",  "name": "AI Index"},
    "crypto_stocks":{"epic": "IX.D.CRYPST.DAILY.IP",  "name": "Crypto Stocks Index"},
}

# Min % surprise to flag an economic release lag trade
ECON_SURPRISE_THRESHOLD = float(os.environ.get("ECON_SURPRISE_PCT", "0.3"))
# Min % price move still needed for lag trade to be valid (not already priced)
LAG_MAX_PRICED_PCT       = float(os.environ.get("LAG_MAX_PRICED_PCT", "0.8"))


@dataclass
class PriceSnapshot:
    epic: str
    name: str
    bid: float
    offer: float
    change_pct: float
    high: float
    low: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mid(self) -> float:
        return (self.bid + self.offer) / 2


@dataclass
class CalendarEntry:
    event_time: str
    country: str
    importance: str   # HIGH / MEDIUM / LOW
    event_name: str
    actual: str
    forecast: str
    previous: str
    surprise_pct: float = 0.0   # computed: (actual - forecast) / abs(forecast)


@dataclass
class NewsItem:
    headline: str
    source: str
    url: str
    published: str
    epics: list[str] = field(default_factory=list)


@dataclass
class IGTradeIdea:
    direction: str          # BUY / SELL
    epic: str
    instrument_name: str
    entry_note: str         # "market now at X"
    stop_distance_pts: float
    target_pts: float | None
    confidence: float
    rationale: str
    source_events: list[str] = field(default_factory=list)
    pattern: str = ""       # "lag_trade" / "signal_alignment" / "geo_shock"

    def to_market_dict(self) -> dict:
        return {
            "market_id":    f"ig_idea_{self.epic}",
            "question":     f"{self.direction} {self.instrument_name}: {self.rationale[:100]}",
            "entity_name":  self.instrument_name,
            "entity_type":  "ig_trade_idea",
            "signal_text":  self.rationale,
            "direction":    "bullish" if self.direction == "BUY" else "bearish",
            "confidence":   self.confidence,
            "source":       "ig_intelligence",
            "tokens":       [],
            "market_price": 0.5,
            "_ig_epic":     self.epic,
            "_ig_direction": self.direction,
            "_signal_type": self.pattern,
            "is_fee_free":  False,
            "tick_size":    "0.01",
            "min_order_size": 1.0,
            "edge": self.confidence - 0.5,
        }


class IGIntelligence:
    """
    Fetches and structures IG market data; detects trade patterns.
    """

    def __init__(self, ig: IGExecutor) -> None:
        self._ig = ig
        self._http = httpx.Client(timeout=12.0)
        self._graph_headers = {"Authorization": f"Bearer {GRAPH_API_SECRET}"}
        self._prices: dict[str, PriceSnapshot] = {}
        self._calendar: list[CalendarEntry] = []
        self._news: list[NewsItem] = []

    # ── Data ingestion ────────────────────────────────────────────────────────

    def refresh_prices(self) -> None:
        """Fetch current quotes for all watchlist instruments."""
        if not self._ig._ensure_session():
            logger.warning("[IGIntel] Session not available — skipping price refresh")
            return

        epics = ",".join(info["epic"] for info in WATCHLIST.values())
        try:
            resp = self._ig._http.get(
                f"{self._ig._base}/markets",
                headers=self._ig._headers(version="2"),
                params={"epics": epics, "filter": "TRADEABLE_MARKETS"},
            )
            if resp.status_code == 200:
                markets = resp.json().get("marketDetails") or []
                for m in markets:
                    snapshot = m.get("snapshot") or {}
                    inst     = m.get("instrument") or {}
                    epic     = inst.get("epic") or m.get("epic", "")
                    name     = inst.get("name") or epic
                    bid      = float(snapshot.get("bid") or 0)
                    offer    = float(snapshot.get("offer") or 0)
                    chg_pct  = float(snapshot.get("percentageChange") or 0)
                    high     = float(snapshot.get("high") or 0)
                    low      = float(snapshot.get("low") or 0)
                    if epic:
                        self._prices[epic] = PriceSnapshot(
                            epic=epic, name=name, bid=bid, offer=offer,
                            change_pct=chg_pct, high=high, low=low,
                        )
                logger.info("[IGIntel] %d price snapshots refreshed", len(self._prices))
        except Exception as exc:
            logger.warning("[IGIntel] Price refresh failed: %s", exc)

    def refresh_news(self, count: int = 20) -> None:
        """Fetch latest IG news items."""
        if not self._ig._ensure_session():
            return
        try:
            resp = self._ig._http.get(
                f"{self._ig._base}/news",
                headers=self._ig._headers(version="3"),
                params={"pageSize": count},
            )
            if resp.status_code == 200:
                items = resp.json().get("news") or []
                self._news = [
                    NewsItem(
                        headline=n.get("headline", ""),
                        source=n.get("source", "IG"),
                        url=n.get("url") or "",
                        published=n.get("lastUpdated") or "",
                        epics=n.get("relatedEpics") or [],
                    )
                    for n in items
                ]
                logger.info("[IGIntel] %d news items fetched", len(self._news))
        except Exception as exc:
            logger.debug("[IGIntel] News fetch failed: %s", exc)

    def refresh_calendar(self) -> None:
        """
        Fetch economic calendar from IG.
        IG's REST calendar endpoint: GET /calendars/economic
        Falls back gracefully if not available on this account tier.
        """
        if not self._ig._ensure_session():
            return
        try:
            resp = self._ig._http.get(
                f"{self._ig._base}/calendars/economic",
                headers=self._ig._headers(version="1"),
            )
            if resp.status_code == 200:
                events = resp.json().get("events") or []
                parsed = []
                for e in events:
                    actual   = e.get("actual",   "")
                    forecast = e.get("forecast",  "")
                    surprise = _compute_surprise(actual, forecast)
                    parsed.append(CalendarEntry(
                        event_time  = e.get("eventDate", ""),
                        country     = e.get("country", ""),
                        importance  = e.get("importance", "LOW"),
                        event_name  = e.get("event", ""),
                        actual      = actual,
                        forecast    = forecast,
                        previous    = e.get("previous", ""),
                        surprise_pct= surprise,
                    ))
                self._calendar = parsed
                logger.info("[IGIntel] %d calendar events loaded", len(self._calendar))
            else:
                logger.debug("[IGIntel] Calendar endpoint returned %d", resp.status_code)
        except Exception as exc:
            logger.debug("[IGIntel] Calendar fetch failed: %s", exc)

    def push_to_graph(self) -> None:
        """Bulk-push current prices + news + calendar to Forage Graph /ingest/bulk."""
        if not GRAPH_API_SECRET:
            return

        nodes: list[dict] = []

        # Instrument + PriceSnapshot nodes
        for epic, snap in self._prices.items():
            nodes.append({
                "id": f"ig_instrument_{epic}",
                "type": "Instrument",
                "name": snap.name,
                "epic": epic,
                "source": "ig_group",
            })
            nodes.append({
                "id": f"ig_price_{epic}_{int(snap.timestamp.timestamp())}",
                "type": "PriceSnapshot",
                "instrument_id": f"ig_instrument_{epic}",
                "bid": snap.bid,
                "offer": snap.offer,
                "mid": snap.mid,
                "change_pct": snap.change_pct,
                "high": snap.high,
                "low": snap.low,
                "timestamp": snap.timestamp.isoformat(),
            })

        # CalendarEvent nodes
        for cal in self._calendar:
            if cal.importance in ("HIGH", "MEDIUM"):
                nodes.append({
                    "id": f"ig_cal_{cal.event_name[:40]}_{cal.event_time}",
                    "type": "CalendarEvent",
                    "name": cal.event_name,
                    "country": cal.country,
                    "importance": cal.importance,
                    "event_time": cal.event_time,
                    "actual": cal.actual,
                    "forecast": cal.forecast,
                    "previous": cal.previous,
                    "surprise_pct": cal.surprise_pct,
                    "source": "ig_calendar",
                })

        # NewsItem nodes
        for news in self._news[:10]:
            if news.headline:
                nodes.append({
                    "id": f"ig_news_{hash(news.headline) & 0xFFFFFF}",
                    "type": "NewsItem",
                    "headline": news.headline,
                    "source": news.source,
                    "url": news.url,
                    "published": news.published,
                    "related_epics": news.epics,
                })

        if not nodes:
            return
        try:
            resp = self._http.post(
                f"{FORAGE_GRAPH_URL}/ingest/bulk",
                headers={**self._graph_headers, "Content-Type": "application/json"},
                json={"nodes": nodes, "source": "ig_intelligence"},
            )
            logger.info("[IGIntel] Pushed %d nodes to graph | status=%d", len(nodes), resp.status_code)
        except Exception as exc:
            logger.warning("[IGIntel] Graph push failed: %s", exc)

    # ── Pattern detection ─────────────────────────────────────────────────────

    def detect_trade_ideas(self) -> list[IGTradeIdea]:
        """
        Run all pattern detectors and return prioritised trade ideas.
        At most 3 ideas are returned per cycle to avoid spam.
        """
        ideas: list[IGTradeIdea] = []
        ideas.extend(self._detect_lag_trades())
        ideas.extend(self._detect_signal_alignment())
        ideas.extend(self._detect_geo_shock())

        # Sort by confidence, return top 3
        ideas.sort(key=lambda x: -x.confidence)
        return ideas[:3]

    def _detect_lag_trades(self) -> list[IGTradeIdea]:
        """
        Economic release surprise: actual vs forecast → instrument hasn't priced it yet.
        """
        ideas: list[IGTradeIdea] = []
        for cal in self._calendar:
            if cal.importance != "HIGH":
                continue
            if abs(cal.surprise_pct) < ECON_SURPRISE_THRESHOLD:
                continue

            # Determine affected instruments
            affected = _calendar_to_instruments(cal)
            for epic, name, direction_when_positive in affected:
                snap = self._prices.get(epic)
                if not snap:
                    continue

                # Only flag if price hasn't moved much yet
                if abs(snap.change_pct) > LAG_MAX_PRICED_PCT:
                    continue

                direction = direction_when_positive if cal.surprise_pct > 0 else (
                    "SELL" if direction_when_positive == "BUY" else "BUY"
                )
                stop_pts = _default_stop(epic)
                confidence = min(0.80, 0.55 + abs(cal.surprise_pct) * 0.1)

                ideas.append(IGTradeIdea(
                    direction=direction,
                    epic=epic,
                    instrument_name=name,
                    entry_note=f"market now {snap.mid:.2f}",
                    stop_distance_pts=stop_pts,
                    target_pts=stop_pts * 1.5,
                    confidence=confidence,
                    rationale=(
                        f"{cal.event_name}: actual={cal.actual} vs forecast={cal.forecast} "
                        f"(surprise={cal.surprise_pct:+.1%}) | {name} change so far={snap.change_pct:+.2f}%"
                    ),
                    source_events=[cal.event_name],
                    pattern="lag_trade",
                ))

        return ideas

    def _detect_signal_alignment(self) -> list[IGTradeIdea]:
        """
        When multiple signals (news headlines + price momentum) align on the same instrument,
        emit a high-conviction idea.
        """
        ideas: list[IGTradeIdea] = []
        # Build epic → sentiment score from news
        epic_score: dict[str, float] = {}
        for news in self._news:
            sentiment = _news_sentiment(news.headline)
            for epic in news.epics:
                epic_score[epic] = epic_score.get(epic, 0) + sentiment

        for epic, score in epic_score.items():
            if abs(score) < 1.5:
                continue
            snap = self._prices.get(epic)
            name = snap.name if snap else epic
            direction = "BUY" if score > 0 else "SELL"
            stop_pts  = _default_stop(epic)
            confidence = min(0.75, 0.5 + abs(score) * 0.08)

            ideas.append(IGTradeIdea(
                direction=direction,
                epic=epic,
                instrument_name=name,
                entry_note=f"market now {snap.mid:.2f}" if snap else "price unavailable",
                stop_distance_pts=stop_pts,
                target_pts=stop_pts * 1.2,
                confidence=confidence,
                rationale=(
                    f"News sentiment alignment on {name}: score={score:+.1f} | "
                    f"{len([n for n in self._news if epic in n.epics])} news items"
                ),
                source_events=[n.headline[:60] for n in self._news if epic in n.epics][:3],
                pattern="signal_alignment",
            ))

        return ideas

    def _detect_geo_shock(self) -> list[IGTradeIdea]:
        """
        Geo/energy shock keywords in news → oil, gold, FX under-reaction.
        """
        geo_keywords = [
            "strait of hormuz", "opec", "sanctions", "military strike",
            "oil terminal", "pipeline", "ukraine war", "gaza", "taiwan",
            "energy supply", "iranian", "russian oil",
        ]
        energy_ideas: list[IGTradeIdea] = []

        for news in self._news:
            h = news.headline.lower()
            if not any(kw in h for kw in geo_keywords):
                continue

            # Geo shock → long oil and gold
            for epic_key, direction in [("oil", "BUY"), ("gold", "BUY")]:
                epic_info = COMMODITIES.get(epic_key, {})
                epic = epic_info.get("epic", "")
                name = epic_info.get("name", epic_key)
                if not epic:
                    continue
                snap = self._prices.get(epic)
                if snap and abs(snap.change_pct) > 1.5:
                    continue   # already priced

                stop_pts = _default_stop(epic)
                energy_ideas.append(IGTradeIdea(
                    direction=direction,
                    epic=epic,
                    instrument_name=name,
                    entry_note=f"market now {snap.mid:.2f}" if snap else "",
                    stop_distance_pts=stop_pts,
                    target_pts=stop_pts * 2.0,
                    confidence=0.62,
                    rationale=f"Geo/energy shock: '{news.headline[:80]}'",
                    source_events=[news.headline[:80]],
                    pattern="geo_shock",
                ))

        return energy_ideas[:2]

    # ── Full cycle ────────────────────────────────────────────────────────────

    def run_cycle(self) -> list[IGTradeIdea]:
        """Refresh all data, push to graph, return trade ideas."""
        self.refresh_prices()
        self.refresh_news()
        self.refresh_calendar()
        self.push_to_graph()
        ideas = self.detect_trade_ideas()
        for idea in ideas:
            logger.info(
                "[IGIntel] IDEA: %s %s | confidence=%.0f%% | pattern=%s | %s",
                idea.direction, idea.instrument_name, idea.confidence * 100,
                idea.pattern, idea.rationale[:80],
            )
        return ideas

    def close(self) -> None:
        self._http.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_surprise(actual: str, forecast: str) -> float:
    """(actual - forecast) / abs(forecast), or 0 if unparseable."""
    try:
        a = float(actual.strip().replace("%", "").replace(",", ""))
        f = float(forecast.strip().replace("%", "").replace(",", ""))
        if f == 0:
            return 0.0
        return (a - f) / abs(f)
    except Exception:
        return 0.0


def _calendar_to_instruments(cal: CalendarEntry) -> list[tuple[str, str, str]]:
    """Return list of (epic, name, direction_when_positive_surprise)."""
    name = cal.event_name.lower()
    country = cal.country.lower()
    results: list[tuple[str, str, str]] = []

    if country in ("us", "united states", "usa") and any(
        kw in name for kw in ["gdp", "employment", "nfp", "payroll", "cpi", "inflation",
                               "durable goods", "retail sales", "ism", "pmi"]
    ):
        results.append((INDICES["sp500"]["epic"], "US 500", "BUY"))
        results.append((INDICES["nasdaq"]["epic"], "NASDAQ 100", "BUY"))
        results.append((FOREX["gbpusd"]["epic"], "GBP/USD", "SELL"))   # USD strength on surprise

    if country in ("uk", "united kingdom", "gb") and any(
        kw in name for kw in ["gdp", "inflation", "cpi", "employment", "boe", "rate", "pmi"]
    ):
        results.append((INDICES["ftse"]["epic"], "FTSE 100", "BUY"))
        results.append((FOREX["gbpusd"]["epic"], "GBP/USD", "BUY"))

    if country in ("eu", "eurozone", "europe") and any(
        kw in name for kw in ["gdp", "inflation", "cpi", "ecb", "rate", "pmi", "unemployment"]
    ):
        results.append((INDICES["dax"]["epic"], "DAX 40", "BUY"))
        results.append((FOREX["eurusd"]["epic"], "EUR/USD", "BUY"))

    return results


def _news_sentiment(headline: str) -> float:
    """Quick keyword sentiment: +1 bullish, -1 bearish, 0 neutral."""
    h = headline.lower()
    bullish = ["beats", "surges", "rally", "gains", "record high", "strong", "upgrade",
               "outperforms", "growth", "deal", "approval", "positive"]
    bearish = ["misses", "plunges", "falls", "drops", "warning", "downgrade", "loss",
               "weaker", "contraction", "crisis", "negative", "concern", "risk"]
    score = sum(1 for w in bullish if w in h) - sum(1 for w in bearish if w in h)
    return float(score)


def _default_stop(epic: str) -> float:
    """Conservative stop distance in points based on instrument type."""
    if "FTSE" in epic or "DAX" in epic:
        return 80.0
    if "SP" in epic or "NASDAQ" in epic or "DOW" in epic:
        return 60.0
    if "USD" in epic or "GBP" in epic or "EUR" in epic:
        return 30.0   # forex pips equivalent
    if "LCO" in epic or "NG" in epic:
        return 40.0   # oil/gas
    if "GOLD" in epic:
        return 15.0
    return 50.0
