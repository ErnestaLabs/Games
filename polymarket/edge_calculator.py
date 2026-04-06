"""
EdgeCalculator — identifies causal mispricings and calculates trade edge.

The edge comes from the Reality Graph's causal relationship data.
When an upstream entity (A) triggers with high causal_weight to market entity (B),
the market for B is likely mispriced because the crowd hasn't connected the dots yet.

Edge calculation:
  1. Causal signal strength  — upstream entity in pre_tipping regime + high weight
  2. Market implied probability — current YES price
  3. Graph-implied probability  — derived from causal chain signal strengths
  4. Edge = graph_prob - market_prob (positive = market underpricing YES)
  5. Kelly size = edge / (1 - market_prob) × kelly_fraction
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Literal

from polymarket.market_mapper import MappedMarket, CausalContext, CausalLink

logger = logging.getLogger(__name__)

# ── Config from env ──────────────────────────────────────────────────────────

MIN_EDGE_THRESHOLD = float(os.environ.get("MIN_EDGE_THRESHOLD", "0.08"))
MIN_CAUSAL_WEIGHT = float(os.environ.get("MIN_CAUSAL_WEIGHT", "0.55"))
MIN_LIQUIDITY_USD = float(os.environ.get("MIN_LIQUIDITY_USD", "500"))
KELLY_FRACTION = float(os.environ.get("KELLY_FRACTION", "0.25"))
MAX_DAYS_TO_EXPIRY = int(os.environ.get("MAX_DAYS_TO_EXPIRY", "90"))


# ── Dataclasses ──────────────────────────────────────────────────────────────

Side = Literal["YES", "NO"]


@dataclass
class TradeSignal:
    market_id: str
    question: str
    side: Side                  # YES or NO
    market_price: float         # current market price (0-1)
    graph_prob: float           # graph-implied probability
    edge: float                 # graph_prob - market_price (signed)
    kelly_size: float           # fraction of bankroll (pre-max-position clamp)
    causal_triggers: list[str]  # human-readable trigger descriptions
    confidence: float           # 0-1 overall confidence in the signal
    is_fee_free: bool
    token_id: str
    tick_size: str
    min_order_size: float
    fee_schedule: dict
    signal_type: str            # "causal_upstream" | "regime_shift" | "signal_composite"
    detected_at: datetime = None

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now(timezone.utc)


# ── Probability inference ─────────────────────────────────────────────────────

def _regime_boost(regime: str | None) -> float:
    """Pre-tipping regime means the causal chain is about to fire."""
    boosts = {
        "pre_tipping": 0.15,
        "post_event": -0.10,
        "normal": 0.0,
    }
    return boosts.get(regime or "normal", 0.0)


def _signal_direction_boost(signals: list) -> float:
    """
    Rising signals on upstream entities increase the probability the linked
    market outcome will resolve YES.
    """
    if not signals:
        return 0.0
    direction_scores = {"rising": 0.08, "stable": 0.0, "falling": -0.06}
    total = sum(direction_scores.get(s.direction, 0.0) for s in signals)
    return max(-0.15, min(0.15, total / max(len(signals), 1)))


def _causal_chain_prob(
    links: list[CausalLink],
    min_weight: float,
    regime: str | None,
    signals: list,
) -> float | None:
    """
    Derive a probability adjustment from the causal chain.
    Returns None if no significant signal found.
    """
    strong_links = [l for l in links if l.causal_weight >= min_weight]
    if not strong_links:
        return None

    # Base: weighted average of causal weights
    avg_weight = sum(l.causal_weight for l in strong_links) / len(strong_links)

    # Scale to probability: weight 1.0 = ~85% implied prob, weight 0.55 = ~60%
    # Linear interpolation: 0.55 → 0.58, 1.0 → 0.85
    base_prob = 0.58 + (avg_weight - MIN_CAUSAL_WEIGHT) / (1.0 - MIN_CAUSAL_WEIGHT) * 0.27

    # Apply regime and signal adjustments
    regime_adj = _regime_boost(regime)
    signal_adj = _signal_direction_boost(signals)

    prob = base_prob + regime_adj + signal_adj
    return max(0.05, min(0.95, prob))


def _is_expiry_valid(market: MappedMarket) -> bool:
    """Skip markets expiring too soon (< 1 day) or too far (> MAX_DAYS_TO_EXPIRY)."""
    now = datetime.now(timezone.utc)
    end = market.end_date
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    delta = end - now
    return timedelta(days=1) <= delta <= timedelta(days=MAX_DAYS_TO_EXPIRY)


def _get_effective_fee(market: MappedMarket, side: Side) -> float:
    """Extract the fee rate for this market from its fee_schedule. Never hardcoded."""
    if market.is_fee_free:
        return 0.0
    schedule = market.fee_schedule
    if not schedule:
        return 0.0
    # Fee schedule shape varies — handle common patterns
    rate = (
        schedule.get("makerFee")
        or schedule.get("takerFee")
        or schedule.get("fee")
        or schedule.get("rate")
        or 0.0
    )
    return float(rate)


# ── EdgeCalculator ───────────────────────────────────────────────────────────

class EdgeCalculator:
    def __init__(
        self,
        min_edge: float = MIN_EDGE_THRESHOLD,
        min_causal_weight: float = MIN_CAUSAL_WEIGHT,
        min_liquidity: float = MIN_LIQUIDITY_USD,
        kelly_fraction: float = KELLY_FRACTION,
    ) -> None:
        self.min_edge = min_edge
        self.min_causal_weight = min_causal_weight
        self.min_liquidity = min_liquidity
        self.kelly_fraction = kelly_fraction

    def _kelly_size(self, prob: float, price: float) -> float:
        """
        Full Kelly: (prob - (1-prob) * price / (1-price)) / 1  ... simplified:
        Kelly fraction = (edge) / (1 - price) × fraction
        """
        if price <= 0 or price >= 1:
            return 0.0
        edge = prob - price
        kelly_full = edge / (1.0 - price)
        return max(0.0, kelly_full * self.kelly_fraction)

    def evaluate_market(self, market: MappedMarket) -> TradeSignal | None:
        """
        Returns a TradeSignal if a significant edge exists, else None.
        """
        # Hard filters
        if market.liquidity_usd < self.min_liquidity:
            return None
        if market.entity_match_confidence < 0.5:
            return None
        if not _is_expiry_valid(market):
            return None
        if not market.causal_context:
            return None

        ctx = market.causal_context
        triggers: list[str] = []

        # ── Upstream causal signal ───────────────────────────────────────
        upstream_strong = [
            l for l in ctx.upstream_entities
            if l.causal_weight >= self.min_causal_weight
        ]

        upstream_prob = None
        if upstream_strong:
            upstream_prob = _causal_chain_prob(
                upstream_strong,
                self.min_causal_weight,
                ctx.regime,
                ctx.active_signals,
            )
            for link in upstream_strong[:3]:
                triggers.append(
                    f"{link.entity_name} → [{link.mechanism}] (weight={link.causal_weight:.2f})"
                )

        # ── Downstream risk signal (inverted) ───────────────────────────
        downstream_strong = [
            l for l in ctx.downstream_entities
            if l.causal_weight >= self.min_causal_weight
        ]

        # ── Regime shift signal ──────────────────────────────────────────
        regime_boost = _regime_boost(ctx.regime)
        if ctx.regime == "pre_tipping":
            triggers.append(f"Regime: pre_tipping (+{regime_boost:.0%})")

        # ── Signal composite ─────────────────────────────────────────────
        signal_adj = _signal_direction_boost(ctx.active_signals)
        for sig in ctx.active_signals[:2]:
            if sig.direction != "stable" and abs(sig.value) > 0.1:
                triggers.append(f"Signal: {sig.metric}={sig.value:.2f} ({sig.direction})")

        # ── Pick best probability estimate ───────────────────────────────
        if upstream_prob is None:
            # No upstream causal signal — check regime/signal only
            if abs(regime_boost) < 0.05 and abs(signal_adj) < 0.05:
                return None
            base_yes = market.current_yes_price
            graph_prob_yes = max(0.05, min(0.95, base_yes + regime_boost + signal_adj))
        else:
            graph_prob_yes = upstream_prob

        # Decide side: trade YES if underpriced, NO if overpriced
        yes_edge = graph_prob_yes - market.current_yes_price
        no_edge = (1.0 - graph_prob_yes) - market.current_no_price

        if abs(yes_edge) >= abs(no_edge):
            side: Side = "YES"
            edge = yes_edge
            market_price = market.current_yes_price
            token_id = market.token_id_yes
        else:
            side = "NO"
            edge = no_edge
            market_price = market.current_no_price
            token_id = market.token_id_no

        # Adjust for fees
        fee = _get_effective_fee(market, side)
        net_edge = edge - fee

        if net_edge < self.min_edge:
            return None

        # Kelly size
        graph_prob_side = graph_prob_yes if side == "YES" else (1.0 - graph_prob_yes)
        kelly = self._kelly_size(graph_prob_side, market_price)

        if kelly <= 0:
            return None

        # Confidence = entity match confidence × avg causal weight
        avg_cw = (
            sum(l.causal_weight for l in upstream_strong) / max(len(upstream_strong), 1)
            if upstream_strong else 0.5
        )
        confidence = market.entity_match_confidence * avg_cw

        signal_type = (
            "causal_upstream" if upstream_strong else
            "regime_shift" if ctx.regime == "pre_tipping" else
            "signal_composite"
        )

        return TradeSignal(
            market_id=market.market_id,
            question=market.question,
            side=side,
            market_price=market_price,
            graph_prob=graph_prob_side,
            edge=net_edge,
            kelly_size=kelly,
            causal_triggers=triggers,
            confidence=confidence,
            is_fee_free=market.is_fee_free,
            token_id=token_id,
            tick_size=market.tick_size,
            min_order_size=market.min_order_size,
            fee_schedule=market.fee_schedule,
            signal_type=signal_type,
        )

    def rank_signals(self, markets: list[MappedMarket]) -> list[TradeSignal]:
        """
        Evaluate all mapped markets, return ranked TradeSignals.
        Fee-free markets rank higher at equal edge.
        """
        signals: list[TradeSignal] = []
        for market in markets:
            try:
                sig = self.evaluate_market(market)
                if sig:
                    signals.append(sig)
            except Exception as exc:
                logger.debug("Edge eval error for %s: %s", market.market_id, exc)

        # Sort: fee-free first, then by edge × confidence
        signals.sort(
            key=lambda s: (s.is_fee_free, s.edge * s.confidence),
            reverse=True,
        )
        logger.info(
            "EdgeCalculator: %d/%d markets have edge above threshold",
            len(signals), len(markets),
        )
        return signals
