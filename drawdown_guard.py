"""
Drawdown circuit breaker for The Trading Games.
Prevents tournament elimination via blowup — survival is the primary fitness function.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

DAILY_DRAWDOWN_LIMIT   = float(os.environ.get("DD_DAILY_PCT",   "0.08"))   # 8%
SESSION_DRAWDOWN_LIMIT = float(os.environ.get("DD_SESSION_PCT",  "0.15"))  # 15%
KILL_SWITCH_LIMIT      = float(os.environ.get("DD_KILL_PCT",     "0.25"))  # 25%
WIN_RATE_MIN           = float(os.environ.get("DD_WIN_RATE_MIN", "0.40"))  # 40% over 50 trades
WIN_RATE_WINDOW        = int(os.environ.get("DD_WIN_RATE_WINDOW", "50"))


class GuardAction(str, Enum):
    NORMAL      = "NORMAL"
    REDUCE_SIZE = "REDUCE_SIZE"   # Kelly × 0.25
    PAUSE       = "PAUSE"         # Skip trading this tick
    KILL        = "KILL"          # Full stop — alert orchestrator


@dataclass
class DrawdownGuard:
    agent_name: str
    starting_bankroll: float

    _session_high: float = field(init=False)
    _day_open: float = field(init=False)
    _day_str: str = field(default="", init=False)
    _recent_outcomes: list[int] = field(default_factory=list, init=False)  # 1=win, 0=loss
    _killed_at: float = field(default=0.0, init=False)

    def __post_init__(self):
        self._session_high = self.starting_bankroll
        self._day_open = self.starting_bankroll

    def check(self, current_balance: float) -> GuardAction:
        """Call on every tick before placing orders."""
        # Reset daily high at day boundary
        today = time.strftime("%Y-%m-%d")
        if today != self._day_str:
            self._day_open = current_balance
            self._day_str = today

        self._session_high = max(self._session_high, current_balance)

        daily_dd = (self._day_open - current_balance) / max(self._day_open, 1)
        session_dd = (self._session_high - current_balance) / max(self._session_high, 1)

        if session_dd >= KILL_SWITCH_LIMIT:
            if self._killed_at == 0.0:
                self._killed_at = time.time()
                logger.critical(
                    "[%s] KILL SWITCH: session drawdown %.1f%% — trading halted",
                    self.agent_name, session_dd * 100,
                )
            return GuardAction.KILL

        if daily_dd >= DAILY_DRAWDOWN_LIMIT or session_dd >= SESSION_DRAWDOWN_LIMIT:
            logger.warning(
                "[%s] PAUSE: daily_dd=%.1f%% session_dd=%.1f%%",
                self.agent_name, daily_dd * 100, session_dd * 100,
            )
            return GuardAction.PAUSE

        # Check win rate decay over recent N trades
        if len(self._recent_outcomes) >= WIN_RATE_WINDOW:
            win_rate = sum(self._recent_outcomes[-WIN_RATE_WINDOW:]) / WIN_RATE_WINDOW
            if win_rate < WIN_RATE_MIN:
                logger.warning(
                    "[%s] REDUCE_SIZE: win_rate=%.1f%% over last %d trades",
                    self.agent_name, win_rate * 100, WIN_RATE_WINDOW,
                )
                return GuardAction.REDUCE_SIZE

        return GuardAction.NORMAL

    def record_outcome(self, won: bool) -> None:
        self._recent_outcomes.append(1 if won else 0)
        if len(self._recent_outcomes) > WIN_RATE_WINDOW * 2:
            self._recent_outcomes = self._recent_outcomes[-WIN_RATE_WINDOW:]

    def kelly_scalar(self, action: GuardAction) -> float:
        """Returns Kelly multiplier for the current guard action."""
        return {
            GuardAction.NORMAL:      1.0,
            GuardAction.REDUCE_SIZE: 0.25,
            GuardAction.PAUSE:       0.0,
            GuardAction.KILL:        0.0,
        }[action]
