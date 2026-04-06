"""
MarketMapper — bridges Polymarket markets to Reality Graph causal entities.

Flow:
  scan_markets() → fetch all active Polymarket markets
  map_market()   → extract keywords → POST /query → score entity matches
  get_causal_context() → POST /causal_parents + /causal_children + /regime + /signals
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class CausalLink:
    entity_id: str
    entity_name: str
    mechanism: str
    causal_weight: float
    last_updated: datetime


@dataclass
class SignalReading:
    metric: str
    value: float
    recorded_at: datetime
    direction: str  # rising | falling | stable


@dataclass
class CausalContext:
    upstream_entities: list[CausalLink] = field(default_factory=list)
    downstream_entities: list[CausalLink] = field(default_factory=list)
    regime: str | None = None
    active_signals: list[SignalReading] = field(default_factory=list)


@dataclass
class MappedMarket:
    market_id: str
    question: str
    category: str
    token_id_yes: str
    token_id_no: str
    end_date: datetime
    current_yes_price: float
    current_no_price: float
    liquidity_usd: float
    volume_24h: float
    tick_size: str
    min_order_size: float
    fee_schedule: dict
    graph_entities: list[str]
    entity_match_confidence: float
    causal_context: CausalContext | None
    is_fee_free: bool
    mapped_at: datetime


# ── Keyword extraction ───────────────────────────────────────────────────────

# Common words to ignore when searching graph
_STOPWORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "have",
    "has", "had", "do", "does", "did", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "by", "with", "from", "or", "and",
    "if", "any", "before", "after", "when", "than", "more", "less",
    "what", "which", "who", "how", "yes", "no", "win", "lose", "reach",
    "price", "market", "bet", "predict", "outcome", "end",
}

# Entity types that are relevant to prediction markets
_RELEVANT_TYPES = {
    "company", "person", "country", "organization", "crypto", "currency",
    "index", "commodity", "political_party", "government", "central_bank",
    "sports_team", "event", "technology", "regulator",
}


def _extract_keywords(question: str) -> list[str]:
    """Extract searchable keywords from a market question."""
    # Remove punctuation except hyphens within words
    clean = re.sub(r"[^\w\s\-]", " ", question)
    # Capitalised words and runs of caps are likely proper nouns
    tokens = clean.split()
    keywords = []
    for tok in tokens:
        stripped = tok.strip("-").lower()
        if stripped in _STOPWORDS or len(stripped) < 3:
            continue
        keywords.append(tok.strip("-"))
    # Also try 2-word phrases for org names
    for i in range(len(tokens) - 1):
        pair = f"{tokens[i]} {tokens[i+1]}"
        if not any(w.lower() in _STOPWORDS for w in tokens[i:i+2]):
            keywords.append(pair)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for k in keywords:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            unique.append(k)
    return unique[:12]  # cap at 12 to stay within graph rate limits


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _score_entity(entity: dict, keyword: str, question: str) -> float:
    """Score a graph entity match for a given market question."""
    name = entity.get("name", "") or entity.get("label", "")
    etype = (entity.get("type") or entity.get("entity_type") or "").lower()
    props = entity.get("properties", {})

    # Name similarity to keyword
    name_score = _name_similarity(name, keyword)

    # Bonus if entity type is relevant
    type_bonus = 0.15 if any(t in etype for t in _RELEVANT_TYPES) else 0.0

    # Bonus if entity name appears in full question
    question_bonus = 0.20 if name.lower() in question.lower() else 0.0

    # Signal freshness bonus (recent signals = more relevant)
    freshness = props.get("funding_recency") or props.get("signal_freshness")
    freshness_bonus = 0.0
    if freshness and isinstance(freshness, (int, float)):
        freshness_bonus = min(0.10, 30.0 / max(float(freshness), 1.0) * 0.10)

    return min(1.0, name_score + type_bonus + question_bonus + freshness_bonus)


# ── MarketMapper ─────────────────────────────────────────────────────────────

class MarketMapper:
    def __init__(
        self,
        clob_client: Any,           # py_clob_client.ClobClient
        graph_url: str,
        graph_secret: str,
        min_confidence: float = 0.50,
    ) -> None:
        self._clob = clob_client
        self._graph_url = graph_url.rstrip("/")
        self._graph_secret = graph_secret
        self._min_confidence = min_confidence
        self._mapped: list[MappedMarket] = []
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {graph_secret}"},
            timeout=20.0,
        )

    # ── Graph calls ──────────────────────────────────────────────────────

    async def _graph_query(self, keyword: str) -> list[dict]:
        try:
            resp = await self._http.post(
                f"{self._graph_url}/query",
                json={"query": keyword, "limit": 5},
            )
            data = resp.json()
            return data.get("nodes") or data.get("results") or []
        except Exception as exc:
            logger.debug("Graph query failed for '%s': %s", keyword, exc)
            return []

    async def _causal_parents(self, entity_id: str, min_weight: float = 0.0) -> list[dict]:
        try:
            resp = await self._http.post(
                f"{self._graph_url}/causal_parents",
                json={"entity_id": entity_id, "min_weight": min_weight, "limit": 10},
            )
            return resp.json().get("parents") or resp.json().get("results") or []
        except Exception:
            return []

    async def _causal_children(self, entity_id: str, min_weight: float = 0.0) -> list[dict]:
        try:
            resp = await self._http.post(
                f"{self._graph_url}/causal_children",
                json={"entity_id": entity_id, "min_weight": min_weight, "limit": 10},
            )
            return resp.json().get("children") or resp.json().get("results") or []
        except Exception:
            return []

    async def _get_regime(self, entity_id: str) -> str | None:
        try:
            resp = await self._http.post(
                f"{self._graph_url}/regime",
                json={"entity_id": entity_id},
            )
            return resp.json().get("regime")
        except Exception:
            return None

    async def _get_signals(self, entity_id: str) -> list[dict]:
        try:
            resp = await self._http.get(f"{self._graph_url}/signals/{entity_id}")
            return resp.json().get("signals") or []
        except Exception:
            return []

    # ── Entity matching ──────────────────────────────────────────────────

    async def _find_entities(self, question: str) -> tuple[list[str], float]:
        """Returns (entity_ids, confidence)."""
        keywords = _extract_keywords(question)
        if not keywords:
            return [], 0.0

        # Query graph for each keyword concurrently (batched to avoid hammering)
        best_matches: dict[str, tuple[dict, float]] = {}  # entity_id -> (entity, score)

        for kw in keywords:
            results = await self._graph_query(kw)
            for entity in results:
                eid = str(entity.get("id") or entity.get("entity_id") or "")
                if not eid:
                    continue
                score = _score_entity(entity, kw, question)
                if eid not in best_matches or score > best_matches[eid][1]:
                    best_matches[eid] = (entity, score)
            await asyncio.sleep(0.05)  # mild throttle

        if not best_matches:
            return [], 0.0

        # Sort by score, take top 3 above threshold
        ranked = sorted(best_matches.values(), key=lambda x: x[1], reverse=True)
        top = [(e, s) for e, s in ranked if s >= self._min_confidence][:3]

        if not top:
            return [], 0.0

        entity_ids = [str(e.get("id") or e.get("entity_id", "")) for e, _ in top]
        confidence = top[0][1]  # confidence = best match score
        return entity_ids, confidence

    # ── Causal context ───────────────────────────────────────────────────

    async def get_causal_context(self, entity_ids: list[str]) -> CausalContext:
        if not entity_ids:
            return CausalContext()

        primary_id = entity_ids[0]

        parents_raw, children_raw, regime, signals_raw = await asyncio.gather(
            self._causal_parents(primary_id),
            self._causal_children(primary_id),
            self._get_regime(primary_id),
            self._get_signals(primary_id),
        )

        def _parse_link(raw: dict) -> CausalLink:
            ts_raw = raw.get("last_updated") or raw.get("updated_at") or ""
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
            except ValueError:
                ts = datetime.now(timezone.utc)
            return CausalLink(
                entity_id=str(raw.get("id") or raw.get("entity_id", "")),
                entity_name=raw.get("name") or raw.get("label") or "",
                mechanism=raw.get("mechanism") or raw.get("relationship", ""),
                causal_weight=float(raw.get("causal_weight") or raw.get("weight") or 0.0),
                last_updated=ts,
            )

        def _parse_signal(raw: dict) -> SignalReading:
            ts_raw = raw.get("recorded_at") or raw.get("timestamp") or ""
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
            except ValueError:
                ts = datetime.now(timezone.utc)
            return SignalReading(
                metric=raw.get("metric") or raw.get("type") or "",
                value=float(raw.get("value") or 0.0),
                recorded_at=ts,
                direction=raw.get("direction") or "stable",
            )

        return CausalContext(
            upstream_entities=[_parse_link(r) for r in parents_raw],
            downstream_entities=[_parse_link(r) for r in children_raw],
            regime=regime,
            active_signals=[_parse_signal(s) for s in signals_raw],
        )

    # ── Market mapping ───────────────────────────────────────────────────

    async def map_market(self, market: dict) -> MappedMarket | None:
        question = market.get("question") or market.get("description") or ""
        if not question:
            return None

        # Extract token IDs (YES/NO outcomes)
        tokens = market.get("tokens") or []
        token_yes = next((t.get("token_id", "") for t in tokens if t.get("outcome", "").upper() == "YES"), "")
        token_no = next((t.get("token_id", "") for t in tokens if t.get("outcome", "").upper() == "NO"), "")
        if not token_yes:
            token_yes = tokens[0].get("token_id", "") if tokens else ""
        if not token_no:
            token_no = tokens[1].get("token_id", "") if len(tokens) > 1 else ""

        # Parse end date
        end_raw = market.get("end_date_iso") or market.get("endDate") or ""
        try:
            end_date = datetime.fromisoformat(end_raw.replace("Z", "+00:00")) if end_raw else datetime.now(timezone.utc)
        except ValueError:
            end_date = datetime.now(timezone.utc)

        # Prices
        yes_price = float(market.get("best_ask") or market.get("lastTradePrice") or 0.5)
        no_price = round(1.0 - yes_price, 4)

        # Fee-free detection: geopolitical / world events categories
        category = (market.get("category") or "").lower()
        fee_schedule = market.get("feeSchedule") or market.get("fee_schedule") or {}
        is_fee_free = any(c in category for c in ("geopolit", "world event", "international"))

        # Entity matching
        entity_ids, confidence = await self._find_entities(question)

        causal_context = None
        if entity_ids:
            causal_context = await self.get_causal_context(entity_ids)

        return MappedMarket(
            market_id=market.get("condition_id") or market.get("conditionId") or "",
            question=question,
            category=category,
            token_id_yes=token_yes,
            token_id_no=token_no,
            end_date=end_date,
            current_yes_price=yes_price,
            current_no_price=no_price,
            liquidity_usd=float(market.get("liquidityAmt") or market.get("liquidity") or 0.0),
            volume_24h=float(market.get("volume24hr") or market.get("volume") or 0.0),
            tick_size=str(market.get("minimum_tick_size") or market.get("tickSize") or "0.01"),
            min_order_size=float(market.get("min_order_size") or market.get("minOrderSize") or 1.0),
            fee_schedule=fee_schedule,
            graph_entities=entity_ids,
            entity_match_confidence=confidence,
            causal_context=causal_context,
            is_fee_free=is_fee_free,
            mapped_at=datetime.now(timezone.utc),
        )

    async def scan_markets(self) -> list[MappedMarket]:
        """Fetch all active markets and map them to graph entities."""
        logger.info("Scanning Polymarket markets...")
        raw_markets: list[dict] = []

        # Paginate through all markets
        next_cursor = "MA=="
        while True:
            try:
                resp = self._clob.get_markets(next_cursor=next_cursor)
                batch = resp.get("data") or []
                raw_markets.extend(batch)
                next_cursor = resp.get("next_cursor") or ""
                if not next_cursor or next_cursor == "LTE=":
                    break
                await asyncio.sleep(0.2)
            except Exception as exc:
                logger.error("Failed to fetch markets: %s", exc)
                break

        logger.info("Fetched %d raw markets", len(raw_markets))

        # Map markets concurrently in chunks of 20
        mapped: list[MappedMarket] = []
        chunk_size = 20
        for i in range(0, len(raw_markets), chunk_size):
            chunk = raw_markets[i:i + chunk_size]
            results = await asyncio.gather(
                *[self.map_market(m) for m in chunk],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, MappedMarket):
                    mapped.append(r)
                elif isinstance(r, Exception):
                    logger.debug("map_market error: %s", r)
            await asyncio.sleep(0.5)

        self._mapped = mapped
        logger.info("Mapped %d markets to graph entities", len(mapped))
        return mapped

    async def refresh(self) -> None:
        """Refresh market list and mappings (call every 30 minutes)."""
        logger.info("Refreshing market mappings...")
        await self.scan_markets()

    @property
    def mapped_markets(self) -> list[MappedMarket]:
        return self._mapped

    async def close(self) -> None:
        await self._http.aclose()
