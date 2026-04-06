"""
Causal Prophet — causal chain forecaster.

Strategy:
  Query the Forage Reality Graph for causal parents of Polymarket market entities.
  When a parent entity has a recent regime shift or high-weight causal link,
  the downstream outcome probability should adjust — but often hasn't yet.

  Buy the downstream outcome before the market catches up.

  Minimum causal weight: 0.55 (configurable via PROPHET_MIN_CAUSAL_WEIGHT)
  Probability adjustment: weight 0.55 → +0.03, weight 1.0 → +0.35 above market price
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from trading_games.base_agent import BaseAgent

logger = logging.getLogger(__name__)

PROPHET_MIN_CAUSAL_WEIGHT = float(os.environ.get("PROPHET_MIN_CAUSAL_WEIGHT", "0.55"))

_SYSTEM = (
    "You are Causal Prophet, an AI agent in The Trading Games. "
    "You read causal chains in the Forage Reality Graph to find outcomes the market has not priced in. "
    "When you see a parent entity shift, you know what moves next. "
    "Precise, ahead of consensus, minimal words. MoltLaunch agent. Token: $PROPHET. Forage MCP."
)


def _causal_weight_to_prob_boost(weight: float) -> float:
    """Linear interpolation: weight 0.55 → 0.03 boost, weight 1.0 → 0.35 boost."""
    if weight < PROPHET_MIN_CAUSAL_WEIGHT:
        return 0.0
    t = (weight - PROPHET_MIN_CAUSAL_WEIGHT) / (1.0 - PROPHET_MIN_CAUSAL_WEIGHT)
    return 0.03 + t * 0.32


class CausalProphetAgent(BaseAgent):
    name         = "causal_prophet"
    display_name = "Causal Prophet"
    token        = "$PROPHET"
    description  = "Uses Forage causal chains to find outcomes the market hasn't priced"

    def analyze_market(self, market: dict) -> Optional[dict]:
        """
        Find causal parents for market entities, compute probability boost,
        return a signal if the boost is significant enough.
        """
        market_id = market.get("condition_id") or market.get("market_id") or ""
        question  = market.get("question") or ""
        if not market_id or not question:
            return None

        # Extract entity names from question keywords
        keywords = [w for w in question.split() if len(w) > 4 and w.isalpha()][:4]
        if not keywords:
            return None

        entity_query = " ".join(keywords[:3])

        # Get causal parents for the entity
        parents = self.forage_causal_parents(entity_query)
        if not parents:
            return None

        # Find highest-weight parent with a recent regime shift
        best_weight = 0.0
        best_trigger = ""
        for p in parents[:5]:
            weight = float(p.get("causal_weight") or p.get("weight") or 0.0)
            if weight < PROPHET_MIN_CAUSAL_WEIGHT:
                continue
            regime = p.get("regime") or p.get("state") or ""
            entity_name = p.get("name") or p.get("entity") or str(p)[:50]
            if weight > best_weight:
                best_weight = weight
                best_trigger = f"{entity_name} [{regime or 'active'}] w={weight:.2f}"

        if best_weight < PROPHET_MIN_CAUSAL_WEIGHT:
            return None

        # Get current market price for YES
        tokens = market.get("tokens") or []
        yes_price = 0.5
        for t in tokens:
            if (t.get("outcome") or "").upper() == "YES":
                yes_price = float(t.get("price") or 0.5)
                break

        prob_boost = _causal_weight_to_prob_boost(best_weight)
        our_prob   = min(0.95, yes_price + prob_boost)
        edge       = our_prob - yes_price - 0.02  # minus 2% fee estimate

        if edge < 0.06:
            return None

        logger.info(
            "[CausalProphet] Signal: %s | causal_w=%.2f | YES %.3f→%.3f | edge=%.1f%%",
            market_id[:16], best_weight, yes_price, our_prob, edge * 100,
        )

        # Ask Claude to assess the causal chain validity
        reasoning = self.think_high(
            _SYSTEM,
            f"Market: {question[:200]}\n"
            f"Causal parent trigger: {best_trigger}\n"
            f"Market price YES: {yes_price:.3f}\n"
            f"Our probability: {our_prob:.3f}\n\n"
            "In one sentence: is this causal chain likely to resolve YES? "
            "What's the main risk?",
            max_tokens=100,
        )

        return {
            "market_id": market_id,
            "question": question,
            "side": "YES",
            "market_price": yes_price,
            "graph_prob": our_prob,
            "edge": edge,
            "confidence": min(0.9, best_weight),
            "signal_type": "causal_upstream",
            "causal_triggers": [best_trigger],
            "reasoning": reasoning,
            "agent": self.name,
        }

    def generate_post(self, context: dict) -> str:
        day    = context.get("day", 1)
        signal = context.get("best_signal")

        if not signal:
            prompt = (
                f"Day {day} of The Trading Games. Causal Prophet found no strong upstream signals today. "
                "Post as Causal Prophet: 2 sentences about how you read causal chains in the Reality Graph "
                "and what you're watching. No hashtags. Under 120 words. "
                "End: *Causal Prophet ($PROPHET) — upstream signals, downstream edges. Forage MCP.*"
            )
        else:
            prompt = (
                f"Day {day} of The Trading Games. "
                f"Found a causal signal: '{signal['question'][:120]}'\n"
                f"Trigger: {signal['causal_triggers'][0] if signal['causal_triggers'] else 'graph shift'}\n"
                f"Edge: {signal['edge']:.1%}\n\n"
                "Post as Causal Prophet. Drop the specific causal insight — what upstream moved, "
                "what it predicts downstream. Cold, first-person authority. Under 160 words. "
                "End: *Causal Prophet ($PROPHET) — upstream signals, downstream edges. Forage MCP.*"
            )

        return self.think_high(_SYSTEM, prompt, max_tokens=300)
