"""
News Bolt — rapid news-driven market entry.

Strategy:
  Monitor Forage web intelligence for breaking news matching open Polymarket markets.
  When news hits that has clear directional implications, enter BEFORE the market
  reprices (markets typically take 5-20 minutes to adjust).

  Signal quality: recency + relevance + sentiment clarity.
  Min edge after fees: 6% (news moves fast, must move fast too).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from trading_games.base_agent import BaseAgent

logger = logging.getLogger(__name__)

BOLT_MIN_EDGE   = float(os.environ.get("BOLT_MIN_EDGE", "0.06"))
BOLT_MAX_NEWS_AGE_MINS = int(os.environ.get("BOLT_MAX_NEWS_AGE_MINS", "30"))

_SYSTEM = (
    "You are News Bolt, an AI agent in The Trading Games. "
    "You catch news before Polymarket prices it in. Speed is your edge. "
    "Terse, decisive, no hedging. MoltLaunch agent. Token: $BOLT. Forage MCP."
)

_SENTIMENT_PROMPT = (
    "Rate the directional implication of this news for the YES side of the market.\n"
    "News: {news}\nMarket: {question}\n\n"
    "Reply with ONLY: STRONG_YES | MILD_YES | NEUTRAL | MILD_NO | STRONG_NO\n"
    "Then one sentence reason."
)

_SENTIMENT_TO_PROB_BOOST = {
    "STRONG_YES": 0.18,
    "MILD_YES":   0.07,
    "NEUTRAL":    0.00,
    "MILD_NO":   -0.07,
    "STRONG_NO": -0.18,
}


class NewsBoltAgent(BaseAgent):
    name         = "news_bolt"
    display_name = "News Bolt"
    token        = "$BOLT"
    description  = "Rapid news-driven entry before markets reprice"

    def analyze_market(self, market: dict) -> Optional[dict]:
        market_id = market.get("condition_id") or market.get("market_id") or ""
        question  = market.get("question") or ""
        if not market_id or not question:
            return None

        # Extract key terms for news search
        keywords = " ".join(w for w in question.split() if len(w) > 3)[:80]

        # Search for recent news via Forage
        news_results = self._forage_tool("search_web", {"query": f"{keywords} news today breaking"})
        if not news_results:
            return None

        # Get the top 2 results
        top_news = []
        for item in news_results[:3]:
            if isinstance(item, dict):
                snippet = item.get("snippet") or item.get("description") or item.get("text") or ""
                if snippet:
                    top_news.append(snippet[:300])

        if not top_news:
            return None

        news_block = "\n\n".join(top_news[:2])

        # Ask Claude to rate the directional implication
        raw = self.think_high(
            _SYSTEM,
            _SENTIMENT_PROMPT.format(news=news_block[:400], question=question[:200]),
            max_tokens=80,
        )
        if not raw:
            return None

        # Parse sentiment
        sentiment = "NEUTRAL"
        for s in _SENTIMENT_TO_PROB_BOOST:
            if s in raw.upper():
                sentiment = s
                break

        prob_boost = _SENTIMENT_TO_PROB_BOOST[sentiment]
        if abs(prob_boost) < 0.05:
            return None

        # Get market price
        tokens    = market.get("tokens") or []
        yes_price = 0.5
        for t in tokens:
            if (t.get("outcome") or "").upper() == "YES":
                yes_price = float(t.get("price") or 0.5)
                break

        side = "YES" if prob_boost > 0 else "NO"
        mkt_price = yes_price if side == "YES" else (1 - yes_price)
        our_prob  = min(0.95, mkt_price + abs(prob_boost))
        edge      = our_prob - mkt_price - 0.02  # minus 2% fee

        if edge < BOLT_MIN_EDGE:
            return None

        logger.info(
            "[NewsBolt] Signal: %s | sentiment=%s | edge=%.1f%%",
            market_id[:16], sentiment, edge * 100,
        )

        return {
            "market_id":  market_id,
            "question":   question,
            "side":       side,
            "market_price": mkt_price,
            "graph_prob": our_prob,
            "edge":       edge,
            "confidence": 0.70 if "STRONG" in sentiment else 0.55,
            "signal_type": "news_catalyst",
            "causal_triggers": [f"sentiment={sentiment}: {raw[:80]}"],
            "agent": self.name,
        }

    def generate_post(self, context: dict) -> str:
        day    = context.get("day", 1)
        signal = context.get("best_signal")

        if not signal:
            prompt = (
                f"Day {day} of The Trading Games. News Bolt scanned headlines but found "
                "no clear directional catalysts priced into Polymarket markets today. "
                "Post about the discipline of not trading on ambiguous news. "
                "Under 120 words. No hashtags. "
                "End: *News Bolt ($BOLT) — before the market catches up. Forage MCP.*"
            )
        else:
            prompt = (
                f"Day {day} of The Trading Games. "
                f"Caught a news catalyst: '{signal['causal_triggers'][0][:120]}'\n"
                f"Market direction: {signal['side']} | Edge: {signal['edge']:.1%}\n\n"
                "Post as News Bolt. Explain the speed edge — why markets lag on breaking news "
                "and how graph intelligence closes that gap. Under 150 words. "
                "End: *News Bolt ($BOLT) — before the market catches up. Forage MCP.*"
            )

        return self.think_high(_SYSTEM, prompt, max_tokens=280)
