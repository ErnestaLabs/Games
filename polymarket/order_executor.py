"""
OrderExecutor — risk-managed order execution for Polymarket.

Risk controls (all from env vars):
  MAX_POSITION_SIZE_PCT  — max % of bankroll per position
  MAX_DAILY_LOSS_PCT     — halt if daily P&L drops below this
  MAX_CONCURRENT_POSITIONS — max open positions at once
  DRY_RUN                — if True, log only, no real orders (default: True)

Order flow:
  1. Risk checks (bankroll, daily loss, position count, position size)
  2. Size order using Kelly (clamped to MAX_POSITION_SIZE_PCT)
  3. DRY_RUN check
  4. Place FOK order via ClobClient (market order pattern)
  5. Log result, update position tracker
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Any

from polymarket.edge_calculator import TradeSignal

logger = logging.getLogger(__name__)

# ── Risk config ───────────────────────────────────────────────────────────────

MAX_POSITION_SIZE_PCT = float(os.environ.get("MAX_POSITION_SIZE_PCT", "5")) / 100.0
MAX_DAILY_LOSS_PCT = float(os.environ.get("MAX_DAILY_LOSS_PCT", "10")) / 100.0
MAX_CONCURRENT_POSITIONS = int(os.environ.get("MAX_CONCURRENT_POSITIONS", "10"))
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() not in ("false", "0", "no")


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Position:
    market_id: str
    question: str
    side: str
    token_id: str
    size_usdc: float
    entry_price: float
    order_id: str
    opened_at: datetime
    signal: TradeSignal


@dataclass
class ExecutionResult:
    success: bool
    dry_run: bool
    order_id: str | None
    market_id: str
    side: str
    size_usdc: float
    price: float
    reason: str = ""
    executed_at: datetime = None

    def __post_init__(self):
        if self.executed_at is None:
            self.executed_at = datetime.now(timezone.utc)


@dataclass
class RiskState:
    bankroll_usdc: float
    daily_realized_pnl: float = 0.0
    daily_date: date = field(default_factory=date.today)
    open_positions: dict[str, Position] = field(default_factory=dict)

    def reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_date = today

    @property
    def available_bankroll(self) -> float:
        locked = sum(p.size_usdc for p in self.open_positions.values())
        return max(0.0, self.bankroll_usdc - locked)

    @property
    def daily_loss_pct(self) -> float:
        if self.bankroll_usdc <= 0:
            return 0.0
        return -self.daily_realized_pnl / self.bankroll_usdc


# ── OrderExecutor ────────────────────────────────────────────────────────────

class OrderExecutor:
    def __init__(
        self,
        clob_client: Any,     # py_clob_client.ClobClient
        initial_bankroll: float,
        dry_run: bool = DRY_RUN,
    ) -> None:
        self._clob = clob_client
        self._dry_run = dry_run
        self._risk = RiskState(bankroll_usdc=initial_bankroll)
        self._execution_log: list[ExecutionResult] = []

        if self._dry_run:
            logger.warning("DRY_RUN=True — no real orders will be placed")

    # ── Risk checks ──────────────────────────────────────────────────────

    def _check_risk(self, signal: TradeSignal, size_usdc: float) -> tuple[bool, str]:
        self._risk.reset_daily_if_needed()

        # Already in this market
        if signal.market_id in self._risk.open_positions:
            return False, "Already have an open position in this market"

        # Max concurrent positions
        if len(self._risk.open_positions) >= MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions reached ({MAX_CONCURRENT_POSITIONS})"

        # Daily loss halt
        if self._risk.daily_loss_pct >= MAX_DAILY_LOSS_PCT:
            return False, f"Daily loss halt triggered ({self._risk.daily_loss_pct:.1%} >= {MAX_DAILY_LOSS_PCT:.1%})"

        # Insufficient bankroll
        if size_usdc > self._risk.available_bankroll:
            return False, f"Insufficient available bankroll (need ${size_usdc:.2f}, have ${self._risk.available_bankroll:.2f})"

        # Min order size
        if size_usdc < signal.min_order_size:
            return False, f"Order size ${size_usdc:.2f} below market minimum ${signal.min_order_size}"

        return True, ""

    def _compute_size(self, signal: TradeSignal) -> float:
        """Apply Kelly fraction then clamp to MAX_POSITION_SIZE_PCT of bankroll."""
        kelly_usdc = signal.kelly_size * self._risk.available_bankroll
        max_usdc = MAX_POSITION_SIZE_PCT * self._risk.bankroll_usdc
        return round(min(kelly_usdc, max_usdc), 2)

    def _snap_price(self, price: float, tick_size: str) -> float:
        """Round price to market tick size."""
        try:
            tick = float(tick_size)
        except (ValueError, TypeError):
            tick = 0.01
        if tick <= 0:
            return round(price, 2)
        snapped = round(round(price / tick) * tick, 10)
        return max(0.01, min(0.99, snapped))

    # ── Execution ────────────────────────────────────────────────────────

    def execute(self, signal: TradeSignal) -> ExecutionResult:
        if not signal.token_id:
            logger.warning(
                "SKIPPED [%s] — empty token_id (market not on CLOB yet)",
                signal.market_id[:16],
            )
            return ExecutionResult(
                success=False,
                dry_run=self._dry_run,
                order_id=None,
                market_id=signal.market_id,
                side=signal.side,
                size_usdc=0.0,
                price=signal.market_price,
                reason="Empty token_id — market not tradeable on CLOB",
            )

        size_usdc = self._compute_size(signal)
        ok, reason = self._check_risk(signal, size_usdc)

        if not ok:
            result = ExecutionResult(
                success=False,
                dry_run=self._dry_run,
                order_id=None,
                market_id=signal.market_id,
                side=signal.side,
                size_usdc=size_usdc,
                price=signal.market_price,
                reason=reason,
            )
            logger.info(
                "SKIPPED [%s] %s @ %.3f — %s",
                signal.market_id[:12], signal.side, signal.market_price, reason
            )
            self._execution_log.append(result)
            return result

        # Snap price to tick size (slight premium for FOK fill)
        tick = float(signal.tick_size or "0.01")
        if signal.side == "YES":
            price = self._snap_price(signal.market_price + tick, signal.tick_size)
        else:
            price = self._snap_price(signal.market_price + tick, signal.tick_size)

        logger.info(
            "%s ORDER: [%s] %s @ %.3f | size=$%.2f | edge=%.1f%% | triggers: %s",
            "DRY_RUN" if self._dry_run else "LIVE",
            signal.market_id[:12],
            signal.side,
            price,
            size_usdc,
            signal.edge * 100,
            "; ".join(signal.causal_triggers[:2]),
        )

        if self._dry_run:
            fake_order_id = f"dry_{signal.market_id[:8]}_{int(datetime.now(timezone.utc).timestamp())}"
            result = ExecutionResult(
                success=True,
                dry_run=True,
                order_id=fake_order_id,
                market_id=signal.market_id,
                side=signal.side,
                size_usdc=size_usdc,
                price=price,
                reason="DRY_RUN",
            )
            self._record_position(signal, size_usdc, price, fake_order_id)
            self._execution_log.append(result)
            return result

        # ── Live execution ───────────────────────────────────────────────
        try:
            from py_clob_client.clob_types import OrderArgs

            order_args = OrderArgs(
                token_id=signal.token_id,
                price=price,
                size=size_usdc / price,  # convert USDC to shares
                side=signal.side,
            )
            resp = self._clob.create_order(order_args)
            order_id = resp.get("orderID") or resp.get("order_id") or ""
            success = bool(order_id) or resp.get("success", False)

            result = ExecutionResult(
                success=success,
                dry_run=False,
                order_id=order_id,
                market_id=signal.market_id,
                side=signal.side,
                size_usdc=size_usdc,
                price=price,
                reason="" if success else str(resp),
            )
            if success:
                self._record_position(signal, size_usdc, price, order_id)
                logger.info("ORDER FILLED: %s %s @ %.3f | id=%s", signal.side, signal.market_id[:12], price, order_id)
            else:
                logger.warning("ORDER FAILED: %s", resp)

        except Exception as exc:
            logger.error("Order execution error: %s", exc)
            result = ExecutionResult(
                success=False,
                dry_run=False,
                order_id=None,
                market_id=signal.market_id,
                side=signal.side,
                size_usdc=size_usdc,
                price=price,
                reason=str(exc),
            )

        self._execution_log.append(result)
        return result

    def _record_position(
        self, signal: TradeSignal, size_usdc: float, price: float, order_id: str
    ) -> None:
        self._risk.open_positions[signal.market_id] = Position(
            market_id=signal.market_id,
            question=signal.question,
            side=signal.side,
            token_id=signal.token_id,
            size_usdc=size_usdc,
            entry_price=price,
            order_id=order_id,
            opened_at=datetime.now(timezone.utc),
            signal=signal,
        )

    def close_position(self, market_id: str, exit_price: float) -> None:
        pos = self._risk.open_positions.pop(market_id, None)
        if pos:
            pnl = (exit_price - pos.entry_price) * (pos.size_usdc / pos.entry_price)
            if pos.side == "NO":
                pnl = -pnl
            self._risk.daily_realized_pnl += pnl
            logger.info(
                "CLOSED [%s] P&L: $%.2f (entry=%.3f exit=%.3f)",
                market_id[:12], pnl, pos.entry_price, exit_price,
            )

    # ── Status ───────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> dict[str, Position]:
        return dict(self._risk.open_positions)

    @property
    def daily_pnl(self) -> float:
        return self._risk.daily_realized_pnl

    @property
    def bankroll(self) -> float:
        return self._risk.bankroll_usdc

    def status_summary(self) -> dict:
        return {
            "dry_run": self._dry_run,
            "bankroll_usdc": self._risk.bankroll_usdc,
            "available_usdc": self._risk.available_bankroll,
            "open_positions": len(self._risk.open_positions),
            "daily_pnl_usdc": self._risk.daily_realized_pnl,
            "daily_loss_pct": f"{self._risk.daily_loss_pct:.1%}",
            "total_executions": len(self._execution_log),
            "successful_orders": sum(1 for r in self._execution_log if r.success),
        }
