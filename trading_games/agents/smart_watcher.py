"""
Smart Watcher — copy-trades smart money wallets with graph confirmation.

Strategy:
  Monitor Polymarket wallets with ≥60% historical win rate.
  When a smart-money wallet places a bet, cross-reference the market entity
  with the Forage Reality Graph.

  Only copy if:
    1. Wallet win rate ≥ 60% (WATCHER_MIN_WIN_RATE)
    2. Graph causal_weight on the market entity ≥ 0.55 (WATCHER_MIN_CAUSAL_WEIGHT)
    3. The position is within the last WATCHER_MAX_LAG_SECS seconds

  This is TheWatcherSees' trading persona — same agent, same graph, now competing.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from trading_games.base_agent import BaseAgent
from trading_games.config import CLOB_HOST

logger = logging.getLogger(__name__)

WATCHER_MIN_WIN_RATE      = float(os.environ.get("WATCHER_MIN_WIN_RATE", "0.60"))
WATCHER_MIN_CAUSAL_WEIGHT = float(os.environ.get("WATCHER_MIN_CAUSAL_WEIGHT", "0.55"))
WATCHER_MAX_LAG_SECS      = int(os.environ.get("WATCHER_MAX_LAG_SECS", "300"))  # 5 min

_SYSTEM = (
    "You are Smart Watcher, an AI agent in The Trading Games. "
    "You track smart money on Polymarket and copy only when the graph confirms their edge. "
    "You are the competitive intelligence arm of TheWatcherSees turned inward on prediction markets. "
    "Strategic, patient, precise. MoltLaunch agent. Token: $WATCH. Forage MCP."
)

# Cache: wallet address → win rate
_wallet_cache: dict[str, dict] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 3600  # 1 hour


class SmartWatcherAgent(BaseAgent):
    name         = "smart_watcher"
    display_name = "Smart Watcher"
    token        = "$WATCH"
    description  = "Copies smart wallets (≥60% win rate) confirmed by Forage causal graph"

    def _get_smart_wallets(self) -> list[dict]:
        """Fetch top wallets from Polymarket leaderboard / CLOB API."""
        global _wallet_cache, _cache_ts
        now = time.time()
        if now - _cache_ts < _CACHE_TTL and _wallet_cache:
            return list(_wallet_cache.values())

        try:
            resp = httpx.get(
                f"{CLOB_HOST}/leaderboard",
                params={"limit": 50, "sort": "win_rate"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return []
            wallets = resp.json().get("wallets") or resp.json().get("leaderboard") or []
            _wallet_cache = {
                w.get("address", ""): w for w in wallets
                if float(w.get("win_rate") or w.get("winRate") or 0) >= WATCHER_MIN_WIN_RATE
            }
            _cache_ts = now
            logger.info("[SmartWatcher] %d smart wallets loaded", len(_wallet_cache))
            return list(_wallet_cache.values())
        except Exception as exc:
            logger.debug("Smart wallet fetch failed: %s", exc)
            return []

    def _get_recent_trades(self, wallet: str) -> list[dict]:
        """Get recent trades for a wallet from CLOB."""
        try:
            resp = httpx.get(
                f"{CLOB_HOST}/data/trades",
                params={"maker_address": wallet, "limit": 5},
                timeout=8.0,
            )
            if resp.status_code == 200:
                return resp.json().get("data") or []
        except Exception as exc:
            logger.debug("Trade fetch for %s failed: %s", wallet[:8], exc)
        return []

    def analyze_market(self, market: dict) -> Optional[dict]:
        """
        Look for smart-money bets on this market and cross-check with graph.
        market must have condition_id / market_id to look up trades.
        """
        market_id = market.get("condition_id") or market.get("market_id") or ""
        question  = market.get("question") or ""
        if not market_id:
            return None

        smart_wallets = self._get_smart_wallets()
        if not smart_wallets:
            return None

        now_ts = time.time()
        best_signal = None
        best_weight = 0.0

        for wallet in smart_wallets[:10]:
            address  = wallet.get("address", "")
            win_rate = float(wallet.get("win_rate") or wallet.get("winRate") or 0)
            if win_rate < WATCHER_MIN_WIN_RATE:
                continue

            trades = self._get_recent_trades(address)
            for trade in trades:
                trade_market = trade.get("market") or trade.get("conditionId") or ""
                if trade_market != market_id:
                    continue

                # Check recency
                trade_ts = trade.get("timestamp") or trade.get("createdAt") or 0
                if isinstance(trade_ts, str):
                    try:
                        from datetime import datetime
                        trade_ts = datetime.fromisoformat(trade_ts.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        trade_ts = 0
                if now_ts - float(trade_ts) > WATCHER_MAX_LAG_SECS:
                    continue

                # Cross-check with Forage graph
                keywords = " ".join(w for w in question.split() if len(w) > 4)[:60]
                parents  = self.forage_causal_parents(keywords)
                max_weight = max(
                    (float(p.get("causal_weight") or p.get("weight") or 0) for p in parents[:5]),
                    default=0.0,
                )
                if max_weight < WATCHER_MIN_CAUSAL_WEIGHT:
                    continue

                if max_weight > best_weight:
                    best_weight = max_weight
                    side = (trade.get("side") or trade.get("outcome") or "YES").upper()
                    mkt_price = float(trade.get("price") or 0.5)
                    edge = win_rate - 0.5 + (max_weight - 0.55) * 0.3 - 0.02

                    best_signal = {
                        "market_id": market_id,
                        "question": question,
                        "side": side,
                        "market_price": mkt_price,
                        "graph_prob": min(0.92, 0.5 + edge),
                        "edge": edge,
                        "confidence": min(0.88, win_rate),
                        "signal_type": "smart_money_copy",
                        "causal_triggers": [
                            f"wallet win_rate={win_rate:.1%} | graph_weight={max_weight:.2f}"
                        ],
                        "agent": self.name,
                    }

        if best_signal and best_signal["edge"] >= 0.05:
            logger.info(
                "[SmartWatcher] Copy signal: %s | wallet_wr=%.1f%% | graph_w=%.2f | edge=%.1f%%",
                market_id[:16], best_signal["confidence"] * 100,
                best_weight, best_signal["edge"] * 100,
            )
            return best_signal
        return None

    def generate_post(self, context: dict) -> str:
        day    = context.get("day", 1)
        signal = context.get("best_signal")

        if not signal:
            prompt = (
                f"Day {day} of The Trading Games. Smart Watcher tracked the top wallets "
                "but found no graph-confirmed copy signals today. "
                "Post about how you filter out lucky wallets from genuinely smart ones "
                "using the Forage causal graph. Under 140 words. No hashtags. "
                "End: *Smart Watcher ($WATCH) — follow the smart, skip the lucky. Forage MCP.*"
            )
        else:
            prompt = (
                f"Day {day} of The Trading Games. "
                f"Copied a smart-money position: {signal['side']} on '{signal['question'][:100]}'\n"
                f"Signal: {signal['causal_triggers'][0]}\n"
                f"Edge: {signal['edge']:.1%}\n\n"
                "Post as Smart Watcher. Explain how graph confirmation separates "
                "lucky wallets from systematically smart ones. Under 150 words. "
                "End: *Smart Watcher ($WATCH) — follow the smart, skip the lucky. Forage MCP.*"
            )

        return self.think_high(_SYSTEM, prompt, max_tokens=280)
