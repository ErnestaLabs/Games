"""
IGEpicMapper — maps Forage entity signal types to IG Group spread-bet epics.

Epic codes are stable DFB (Daily Funded Bet) instruments on IG's platform.
Demo and live use the same epic codes.

Signal category → IG instrument mapping:
  macro / central_bank / employment  → Indices + Forex
  corporate / earnings / M&A         → Individual shares
  political / election / legislation  → Indices + Forex
  regulatory / FDA / FCA / SEC       → Shares + Sector indices
  geopolitical / conflict / sanctions → Commodities + Forex
"""
from __future__ import annotations

import re

# ── Epic tables ───────────────────────────────────────────────────────────────

# Indices (DFB = rolling daily)
INDICES: dict[str, dict] = {
    "ftse":     {"epic": "IX.D.FTSE.DAILY.IP",   "name": "FTSE 100",   "direction_hint": "neutral"},
    "sp500":    {"epic": "IX.D.SPTRD.DAILY.IP",  "name": "S&P 500",    "direction_hint": "neutral"},
    "dax":      {"epic": "IX.D.DAX.DAILY.IP",    "name": "DAX 40",     "direction_hint": "neutral"},
    "nasdaq":   {"epic": "IX.D.NASDAQ.DAILY.IP", "name": "NASDAQ 100", "direction_hint": "neutral"},
    "dow":      {"epic": "IX.D.DOW.DAILY.IP",    "name": "Dow Jones",  "direction_hint": "neutral"},
}

# Forex (DFB rolling)
FOREX: dict[str, dict] = {
    "gbpusd":   {"epic": "CS.D.GBPUSD.TODAY.IP", "name": "GBP/USD",  "direction_hint": "neutral"},
    "eurusd":   {"epic": "CS.D.EURUSD.TODAY.IP", "name": "EUR/USD",  "direction_hint": "neutral"},
    "usdjpy":   {"epic": "CS.D.USDJPY.TODAY.IP", "name": "USD/JPY",  "direction_hint": "neutral"},
    "gbpeur":   {"epic": "CS.D.GBPEUR.TODAY.IP", "name": "GBP/EUR",  "direction_hint": "neutral"},
}

# Commodities
COMMODITIES: dict[str, dict] = {
    "oil":      {"epic": "CC.D.LCO.USS.IP",      "name": "Brent Crude Oil", "direction_hint": "neutral"},
    "gold":     {"epic": "CS.D.CFDGOLD.CFD.IP",  "name": "Gold",            "direction_hint": "neutral"},
    "natgas":   {"epic": "CC.D.NG.USS.IP",        "name": "Natural Gas",     "direction_hint": "neutral"},
}

# Keyword → epic routing
_KEYWORD_ROUTES: list[tuple[list[str], str, str]] = [
    # (keywords, market_category, key)
    (["ftse", "uk stock", "uk market", "london stock", "british company", "fca", "ofsted",
      "ukplanning", "planning permission", "ofgem", "competition and markets", "cma"],
     "indices", "ftse"),

    (["federal reserve", "fed ", "fomc", "us interest rate", "powell", "inflation",
      "us employment", "non-farm", "nfp", "us gdp", "us election", "trump", "us congress"],
     "indices", "sp500"),

    (["dax", "german", "germany", "eurozone", "ecb", "european central bank", "draghi",
      "lagarde", "eu regulation", "european commission"],
     "indices", "dax"),

    (["nasdaq", "tech stock", "tech sector", "ai company", "semiconductor", "apple", "microsoft",
      "google", "alphabet", "meta", "nvidia", "amazon", "openai"],
     "indices", "nasdaq"),

    (["sterling", "gbp", "pound", "bank of england", "boe ", "bailey ", "uk inflation",
      "uk interest rate", "uk budget", "chancellor", "rishi", "keir", "rachel reeves"],
     "forex", "gbpusd"),

    (["euro ", "eur ", "ecb", "european central bank", "eurozone inflation", "germany gdp",
      "france ", "italy ", "spain ", "eu election"],
     "forex", "eurusd"),

    (["yen", "jpy", "bank of japan", "boj ", "japan interest", "nikkei"],
     "forex", "usdjpy"),

    (["oil", "opec", "crude", "brent", "wti", "energy sanction", "russia oil",
      "iran oil", "saudi", "aramco"],
     "commodities", "oil"),

    (["gold", "safe haven", "flight to quality", "war ", "conflict ", "sanctions"],
     "commodities", "gold"),
]

_CATEGORY_MAP = {
    "indices": INDICES,
    "forex": FOREX,
    "commodities": COMMODITIES,
}


def map_signal_to_epics(
    entity_name: str,
    entity_type: str,
    signal_text: str,
    direction: str = "neutral",     # "bullish" / "bearish" / "neutral"
    max_results: int = 2,
) -> list[dict]:
    """
    Map a Forage entity signal to the best-matching IG epics.

    Returns list of dicts:
      {
        "epic": str,
        "name": str,
        "category": str,   # indices / forex / commodities
        "direction": str,  # BUY / SELL / neutral
        "confidence": float,
        "rationale": str,
      }
    """
    combined = (f"{entity_name} {entity_type} {signal_text}").lower()
    results: list[dict] = []

    for keywords, category, key in _KEYWORD_ROUTES:
        score = sum(1 for kw in keywords if kw in combined)
        if score == 0:
            continue
        market = _CATEGORY_MAP[category][key]
        ig_direction = _resolve_direction(direction, category, entity_type)
        results.append({
            "epic": market["epic"],
            "name": market["name"],
            "category": category,
            "direction": ig_direction,
            "confidence": min(0.95, 0.5 + score * 0.1),
            "rationale": f"keyword_score={score} | entity={entity_type} | signal={signal_text[:60]}",
        })

    # De-duplicate by epic, keep highest confidence
    seen: dict[str, dict] = {}
    for r in sorted(results, key=lambda x: -x["confidence"]):
        if r["epic"] not in seen:
            seen[r["epic"]] = r

    return list(seen.values())[:max_results] or _default_fallback(direction)


def _resolve_direction(signal_direction: str, category: str, entity_type: str) -> str:
    """
    Convert a semantic direction (bullish/bearish/positive/negative) to BUY/SELL.
    For indices: bullish entity news = BUY index, bearish = SELL index.
    For forex: depends on which currency is affected.
    """
    d = signal_direction.lower()
    if d in ("bullish", "positive", "buy", "yes"):
        return "BUY"
    if d in ("bearish", "negative", "sell", "no"):
        return "SELL"
    return "BUY"   # default to long if neutral (trend-following bias)


def _default_fallback(direction: str) -> list[dict]:
    """Fallback to FTSE when no keyword matches (most UK-relevant)."""
    return [{
        "epic": INDICES["ftse"]["epic"],
        "name": INDICES["ftse"]["name"],
        "category": "indices",
        "direction": _resolve_direction(direction, "indices", "unknown"),
        "confidence": 0.35,
        "rationale": "fallback: no keyword match → FTSE 100 default",
    }]
