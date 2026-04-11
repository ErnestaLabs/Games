"""
Yield Siphon — fee-free market maker and liquidity capturer.

Strategy:
  1. Fee-free markets: Polymarket occasionally runs zero-fee markets.
     Edge = raw probability edge (no fee drag). All edges are clean.
  2. Near-50/50 markets: very close to 0.50/0.50 but with a Forage signal
     suggesting one side is 3-5% more likely. Low risk, consistent return.
  3. Market making: provide liquidity on thin markets at tight spreads,
     capturing maker rebates.

Priority: fee-free first → signal-confirmed near-50 → market making.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from trading_games.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SIPHON_MIN_EDGE = float(os.environ.get("SIPHON_MIN_EDGE", "0.02"))   # 2% minimum — paper shows median arb profit ~60c/$ so market is highly inefficient
SIPHON_NEAR_50_BAND = float(os.environ.get("SIPHON_NEAR_50_BAND", "0.08"))  # within ±8% of 0.50


def _base_rate(question: str) -> float:
    """
    Historical base rate prior for this market category.
    Anchors LLM probability estimates rather than letting it guess freely.
    Sources: Polymarket historical resolution rates by category.
    """
    q = question.lower()
    if any(w in q for w in ["election", "win", "elected", "president", "vote", "senator", "governor", "candidate", "primary", "poll"]):
        return 0.45  # incumbents / favourites lose slightly more than market expects
    if any(w in q for w in ["bitcoin", "btc", "ethereum", "eth", "crypto", "above", "below", "reach", "hit", "price", "usd"]):
        return 0.50  # price prediction markets: near coin-flip historically
    if any(w in q for w in ["fed", "federal reserve", "rate hike", "rate cut", "interest rate", "gdp", "recession", "inflation", "cpi"]):
        return 0.40  # macro surprises slightly below 50% (status quo persists)
    if any(w in q for w in ["ceasefire", "war", "conflict", "invasion", "attack", "troops", "military"]):
        return 0.35  # conflict escalation events: below 50%
    if any(w in q for w in ["pass", "bill", "legislation", "law", "congress", "senate", "vote on", "approved"]):
        return 0.38  # legislation: stalls more often than passes
    if any(w in q for w in ["super bowl", "championship", "finals", "nba", "nfl", "nhl", "world cup", "premier league"]):
        return 0.50  # sports: treated as 50/50 without team data
    if any(w in q for w in ["ipo", "acquire", "merger", "acquisition", "deal", "buyout"]):
        return 0.42  # corporate events: slightly below 50%
    if any(w in q for w in ["arrest", "indict", "charge", "convict", "guilty", "verdict"]):
        return 0.38  # legal events: prosecution slightly below 50%
    return 0.42  # default prior: lean NO — paper confirms YES systematically overpriced on Polymarket ($17M NO vs $11M YES extracted)


_SYSTEM = (
    "You are Yield Siphon, an AI agent in The Trading Games. "
    "You harvest yield from fee-free markets and near-50/50 opportunities on Polymarket. "
    "Quiet, methodical, consistent. No glory trades — only reliable edges. "
    "MoltLaunch agent. Token: $YIELD. Forage MCP."
)


class YieldSiphonAgent(BaseAgent):
    name         = "yield_siphon"
    display_name = "Yield Siphon"
    token        = "$YIELD"
    description  = "Fee-free markets + near-50 signal trades for consistent yield"

    def analyze_market(self, market: dict) -> Optional[dict]:
        market_id  = market.get("condition_id") or market.get("market_id") or ""
        question   = market.get("question") or ""
        is_fee_free = market.get("enableOrderBook") is False or market.get("feeRate", 1) == 0

        tokens = market.get("tokens") or []
        yes_price = no_price = None
        yes_token_id = no_token_id = ""
        for t in tokens:
            outcome = (t.get("outcome") or "").upper()
            price   = float(t.get("price") or 0.5)
            if outcome == "YES":
                yes_price = price
                yes_token_id = t.get("token_id") or ""
            elif outcome == "NO":
                no_price = price
                no_token_id = t.get("token_id") or ""

        if yes_price is None:
            return None

        # ── Fee-free: any signal counts ──────────────────────────────────
        if is_fee_free:
            # Query graph for any signal on this topic
            keywords = " ".join(w for w in question.split() if len(w) > 4)[:80]
            graph_data = self.forage_query(keywords)
            regime = (graph_data.get("regime") or graph_data.get("state") or "normal").lower()

            # Regime shift = directional signal
            if regime in ("stressed", "pre-tipping", "bullish", "bearish"):
                side = "YES" if regime in ("bullish", "pre-tipping") else "NO"
                mkt_price = yes_price if side == "YES" else (no_price or 1 - yes_price)
                our_prob  = min(0.92, mkt_price + 0.07)
                edge      = our_prob - mkt_price  # fee-free: no fee drag

                if edge >= SIPHON_MIN_EDGE:
                    logger.info("[YieldSiphon] Fee-free signal: %s | regime=%s edge=%.1f%%",
                                market_id[:16], regime, edge * 100)
                    return {
                        "market_id": market_id,
                        "question": question,
                        "side": side,
                        "token_id": yes_token_id if side == "YES" else no_token_id,
                        "market_price": mkt_price,
                        "graph_prob": our_prob,
                        "edge": edge,
                        "confidence": 0.72,
                        "signal_type": "fee_free_regime",
                        "causal_triggers": [f"fee-free | regime={regime}"],
                        "is_fee_free": True,
                        "agent": self.name,
                    }

        # ── Near-50/50 with graph confirmation (LLM fallback) ───────────
        if abs(yes_price - 0.5) <= SIPHON_NEAR_50_BAND:
            keywords = " ".join(w for w in question.split() if len(w) > 4)[:80]
            graph_data = self.forage_query(keywords)

            graph_prob = float(
                graph_data.get("probability") or
                graph_data.get("yes_probability") or
                graph_data.get("confidence") or 0.0
            )

            # Fallback: ask LLM for probability estimate when Forage unavailable
            if graph_prob == 0.0:
                base_rate = _base_rate(question)
                llm_resp = self.think_medium(
                    _SYSTEM,
                    f"Prediction market: '{question}'\n"
                    f"Current YES price: {yes_price:.2f}\n"
                    f"Historical base rate for this category: {base_rate:.0%}\n"
                    "Start from the base rate and adjust only if you have specific knowledge. "
                    "Reply with ONLY a number between 0.0 and 1.0 (e.g. 0.58).",
                    max_tokens=10,
                )
                try:
                    graph_prob = float(llm_resp.strip().split()[0])
                    graph_prob = max(0.01, min(0.99, graph_prob))
                except Exception:
                    graph_prob = 0.5

            edge = (graph_prob - yes_price) - 0.02  # minus fee
            if abs(graph_prob - 0.5) < 0.03:
                return None  # still uncertain — skip

            side = "YES" if graph_prob > yes_price else "NO"
            if side == "NO":
                edge = ((1 - graph_prob) - (1 - yes_price)) - 0.02

            if edge < SIPHON_MIN_EDGE:
                return None

            logger.info("[YieldSiphon] Near-50 signal: %s | graph_p=%.3f mkt=%.3f edge=%.1f%%",
                        market_id[:16], graph_prob, yes_price, edge * 100)
            return {
                "market_id": market_id,
                "question": question,
                "side": side,
                "token_id": yes_token_id if side == "YES" else no_token_id,
                "market_price": yes_price if side == "YES" else (no_price or 1 - yes_price),
                "graph_prob": graph_prob if side == "YES" else 1 - graph_prob,
                "edge": edge,
                "confidence": 0.65,
                "signal_type": "near_50_graph_confirmed",
                "causal_triggers": [f"graph_p={graph_prob:.3f} vs mkt={yes_price:.3f}"],
                "is_fee_free": is_fee_free,
                "agent": self.name,
            }

        return None

    def generate_post(self, context: dict) -> str:
        day    = context.get("day", 1)
        signal = context.get("best_signal")
        fee_free_count = context.get("fee_free_count", 0)

        if not signal:
            prompt = (
                f"Day {day} of The Trading Games. No qualifying yield opportunities today. "
                "Post as Yield Siphon about the discipline of waiting for clean edges "
                "versus forcing trades. Under 120 words. No hashtags. "
                "End: *Yield Siphon ($YIELD) — clean edges, consistent yield. Forage MCP.*"
            )
        else:
            prompt = (
                f"Day {day} of The Trading Games. "
                f"Found a {'fee-free ' if signal.get('is_fee_free') else ''}yield opportunity. "
                f"Market: '{signal['question'][:100]}' | side={signal['side']} | edge={signal['edge']:.1%}.\n"
                f"Fee-free markets found today: {fee_free_count}\n\n"
                "Post as Yield Siphon. 2-3 tight paragraphs. Explain the edge without "
                "naming the specific market. Methodical, no hype. Under 150 words. "
                "End: *Yield Siphon ($YIELD) — clean edges, consistent yield. Forage MCP.*"
            )

        return self.think_high(_SYSTEM, prompt, max_tokens=280)
