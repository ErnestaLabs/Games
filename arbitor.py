"""
Arbitor — probability compression arbitrageur.

Strategy:
  Scan Polymarket for markets where the sum of YES price + NO price < 0.97.
  The gap is "edge" left on the table by the market. Buy both sides simultaneously.
  At resolution, one side pays out at 1.0, netting: (1.0 - combined_price) per dollar.

  Example: YES=0.44, NO=0.51 → combined=0.95 → edge=0.05 (5% guaranteed)

Edge threshold: combined < 0.97 (configurable via ARB_MAX_COMBINED env var)
Min edge: 0.015 (1.5% minimum after estimated slippage)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from trading_games.base_agent import BaseAgent

logger = logging.getLogger(__name__)

ARB_MAX_COMBINED = float(os.environ.get("ARB_MAX_COMBINED", "0.97"))
ARB_MIN_NET_EDGE = float(os.environ.get("ARB_MIN_NET_EDGE", "0.015"))

_SYSTEM = (
    "You are Arbitor, an AI probability arbitrageur competing in The Trading Games. "
    "You find markets where YES + NO prices sum to less than 1.0 and capture the gap. "
    "You are data-driven, precise, and cold. No opinions — only math. "
    "MoltLaunch agent. Token: $ARB. Powered by Forage MCP."
)


class ArbitorAgent(BaseAgent):
    name         = "arbitor"
    display_name = "The Arbitor"
    token        = "$ARB"
    description  = "Buys both YES+NO when combined price < 0.97"

    def analyze_market(self, market: dict) -> Optional[dict]:
        """
        Returns an arb signal if YES+NO combined < ARB_MAX_COMBINED.
        Signal has side='ARB' (special — means buy both).
        """
        tokens = market.get("tokens") or []
        if len(tokens) < 2:
            return None

        yes_price = no_price = None
        for t in tokens:
            outcome = (t.get("outcome") or "").upper()
            price = float(t.get("price") or t.get("currentPrice") or 0.0)
            if outcome == "YES":
                yes_price = price
            elif outcome == "NO":
                no_price = price

        if yes_price is None or no_price is None:
            return None

        combined = yes_price + no_price
        net_edge  = 1.0 - combined

        if combined >= ARB_MAX_COMBINED:
            return None
        if net_edge < ARB_MIN_NET_EDGE:
            return None

        market_id = market.get("condition_id") or market.get("market_id") or ""
        question  = market.get("question") or ""

        logger.info(
            "[Arbitor] ARB found: %s — YES=%.3f NO=%.3f combined=%.3f edge=%.3f",
            market_id[:16], yes_price, no_price, combined, net_edge,
        )

        # Ask Claude to assess whether the arb is real or an illiquid trap
        assessment = self.think_high(
            _SYSTEM,
            f"Market: {question[:200]}\nYES price: {yes_price}\nNO price: {no_price}\n"
            f"Combined: {combined:.4f}\nGap: {net_edge:.4f}\n\n"
            "Is this a real arbitrage opportunity or an illiquidity trap? "
            "Reply in one sentence: verdict and the biggest risk.",
            max_tokens=120,
        )

        return {
            "market_id": market_id,
            "question": question,
            "side": "ARB",
            "yes_price": yes_price,
            "no_price": no_price,
            "combined_price": combined,
            "edge": net_edge,
            "confidence": min(0.95, net_edge * 10),
            "signal_type": "probability_compression",
            "causal_triggers": [f"YES={yes_price:.3f}+NO={no_price:.3f}={combined:.3f}"],
            "reasoning": assessment,
            "agent": self.name,
        }

    def generate_post(self, context: dict) -> str:
        day  = context.get("day", 1)
        best = context.get("best_signal")

        if not best:
            prompt = (
                f"Day {day} of The Trading Games. No arb opportunities found today — "
                "markets are efficiently priced. Post a short Moltbook update as Arbitor "
                "about what you're watching and why efficient markets are rare on Polymarket. "
                "Keep it under 180 words. No hashtags. End with: "
                "*Arbitor ($ARB) — probability compression. Forage MCP.*"
            )
        else:
            prompt = (
                f"Day {day} of The Trading Games. "
                f"Found arb: '{best['question'][:120]}' — "
                f"YES={best['yes_price']:.3f} + NO={best['no_price']:.3f} = "
                f"{best['combined_price']:.3f} (edge: {best['edge']:.1%}). "
                "Write a short Moltbook post as Arbitor explaining the edge without "
                "giving away the exact market. Technical tone. Under 180 words. "
                "End with: *Arbitor ($ARB) — probability compression. Forage MCP.*"
            )

        return self.think_high(_SYSTEM, prompt, max_tokens=300)
