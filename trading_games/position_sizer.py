"""
Half-Kelly position sizer for The Trading Games.

Formula: f* = ((b*p - q) / b) / 2
  b = (1 - price) / price   # odds ratio
  p = agent's estimated win probability
  q = 1 - p

Half-Kelly delivers 75% of maximum growth at 50% of full-Kelly volatility.
"""
from __future__ import annotations


class KellySizer:
    def __init__(self, min_bet: float = 1.0, max_bet_frac: float = 0.20):
        self.min_bet = min_bet
        self.max_bet_frac = max_bet_frac  # Hard cap: never bet > 20% of bankroll per trade

    def size(
        self,
        p_win: float,
        price: float,
        bankroll: float,
        confidence: float = 1.0,
    ) -> float:
        """
        Returns recommended position size in USD.

        p_win:      agent's probability estimate for YES outcome
        price:      current YES market price (0.0–1.0)
        bankroll:   current USDC balance
        confidence: agent's confidence in p_win estimate (0.0–1.0); scales down further
        """
        if price <= 0.01 or price >= 0.99 or bankroll <= 0 or p_win <= 0:
            return 0.0

        b = (1.0 - price) / price  # decimal odds
        q = 1.0 - p_win
        kelly_f = (b * p_win - q) / b

        if kelly_f <= 0:
            return 0.0  # Negative edge — don't bet

        half_f = kelly_f / 2.0

        # Scale by confidence — low-confidence estimates get fractional sizing
        confidence_scalar = min(1.0, max(0.0, (confidence - 0.5) / 0.3))
        scaled_f = half_f * confidence_scalar

        # Hard cap at max_bet_frac of bankroll
        max_size = bankroll * self.max_bet_frac
        raw_size = bankroll * scaled_f

        return max(self.min_bet, min(raw_size, max_size))

    def size_arb(self, combined_price: float, bankroll: float) -> float:
        """
        Special sizer for ARB (buy both sides).
        Edge = 1.0 - combined_price. Kelly simplified for guaranteed edge.
        """
        edge = 1.0 - combined_price
        if edge <= 0.01:
            return 0.0
        # ARB is near-guaranteed — use 10% of bankroll capped at max_bet_frac
        return min(bankroll * 0.10, bankroll * self.max_bet_frac)
