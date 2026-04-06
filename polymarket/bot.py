"""
Polymarket Intelligence Bot — Reality Graph Edge System

Entry point. Orchestrates the full cycle:
  1. Build ClobClient (L1 → derive L2)
  2. Scan + map markets to Reality Graph entities
  3. Calculate edge for each mapped market
  4. Execute trades on signals above threshold (DRY_RUN=true by default)
  5. Sleep, refresh, repeat

Run:
  DRY_RUN=true POLYGON_PRIVATE_KEY=<key> python polymarket/bot.py

Env vars (all optional — safe defaults apply):
  POLYGON_PRIVATE_KEY     — wallet private key (required for live trading)
  GRAPH_SECRET            — Reality Graph auth token
  DRY_RUN                 — true/false (default: true)
  MAX_POSITION_SIZE_PCT   — default 5
  MAX_DAILY_LOSS_PCT      — default 10
  MIN_EDGE_THRESHOLD      — default 0.08
  MIN_LIQUIDITY_USD       — default 500
  KELLY_FRACTION          — default 0.25
  MAX_CONCURRENT_POSITIONS — default 10
  MIN_CAUSAL_WEIGHT       — default 0.55
  SCAN_INTERVAL_SECONDS   — default 1800 (30 minutes)
  TOP_SIGNALS_PER_CYCLE   — default 5 (max signals to act on per cycle)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CLOB_HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID = int(os.environ.get("CHAIN_ID", "137"))
GRAPH_URL = os.environ.get("FORAGE_GRAPH_URL", "https://forage-graph-production.up.railway.app")
GRAPH_SECRET = os.environ.get("GRAPH_SECRET", "")
POLYGON_PRIVATE_KEY = os.environ.get("POLYGON_PRIVATE_KEY", "")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() not in ("false", "0", "no")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL_SECONDS", "1800"))
TOP_SIGNALS_PER_CYCLE = int(os.environ.get("TOP_SIGNALS_PER_CYCLE", "5"))
INITIAL_BANKROLL = float(os.environ.get("INITIAL_BANKROLL_USDC", "1000"))


# ── Client setup ──────────────────────────────────────────────────────────────

def _build_clob_client():
    """
    Build py-clob-client ClobClient with L1 → L2 auth.
    Returns None if private key not set (DRY_RUN mode without key is supported
    for market scanning only — order placement will be skipped).
    """
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON

        if not POLYGON_PRIVATE_KEY:
            logger.warning("POLYGON_PRIVATE_KEY not set — market scanning only, no order placement")
            # Build read-only client (no key)
            return ClobClient(host=CLOB_HOST, chain_id=CHAIN_ID)

        # L1 auth → derive L2 creds
        client = ClobClient(
            host=CLOB_HOST,
            key=POLYGON_PRIVATE_KEY,
            chain_id=CHAIN_ID,
        )
        # Derive and store L2 API credentials
        client.set_api_creds(client.create_or_derive_api_creds())
        logger.info("ClobClient initialised (L2 auth active)")
        return client

    except ImportError:
        logger.error("py-clob-client not installed. Run: pip install py-clob-client")
        sys.exit(1)
    except Exception as exc:
        logger.error("ClobClient init failed: %s", exc)
        if DRY_RUN:
            logger.warning("DRY_RUN=True, continuing without live client")
            return None
        sys.exit(1)


def _get_bankroll(clob_client) -> float:
    """Fetch USDC balance from Polygon wallet."""
    if not clob_client or not POLYGON_PRIVATE_KEY:
        logger.info("Using configured INITIAL_BANKROLL_USDC=$%.2f", INITIAL_BANKROLL)
        return INITIAL_BANKROLL
    try:
        bal = clob_client.get_balance()
        usdc = float(bal.get("USDC") or bal.get("usdc") or INITIAL_BANKROLL)
        logger.info("Wallet USDC balance: $%.2f", usdc)
        return usdc
    except Exception as exc:
        logger.warning("Could not fetch wallet balance: %s — using $%.2f", exc, INITIAL_BANKROLL)
        return INITIAL_BANKROLL


# ── Main bot loop ─────────────────────────────────────────────────────────────

class PolymarketBot:
    def __init__(self, clob_client, bankroll: float) -> None:
        from polymarket.market_mapper import MarketMapper
        from polymarket.edge_calculator import EdgeCalculator
        from polymarket.order_executor import OrderExecutor
        from polymarket.prediction_store import PredictionStore

        self._mapper = MarketMapper(
            clob_client=clob_client,
            graph_url=GRAPH_URL,
            graph_secret=GRAPH_SECRET,
        )
        self._edge = EdgeCalculator()
        self._executor = OrderExecutor(
            clob_client=clob_client,
            initial_bankroll=bankroll,
            dry_run=DRY_RUN,
        )
        self._predictions = PredictionStore()
        self._cycle = 0

    async def run_cycle(self) -> None:
        self._cycle += 1
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        logger.info("=== Cycle %d [%s] ===", self._cycle, now)

        # Refresh market mappings
        markets = await self._mapper.scan_markets()
        if not markets:
            logger.warning("No markets mapped — check graph connectivity")
            return

        # Rank signals
        signals = self._edge.rank_signals(markets)
        if not signals:
            logger.info("No signals above threshold this cycle")
        else:
            logger.info("%d signals found — acting on top %d", len(signals), TOP_SIGNALS_PER_CYCLE)

        # Execute top N signals and record every one
        for sig in signals[:TOP_SIGNALS_PER_CYCLE]:
            logger.info(
                "SIGNAL: [%s] %s | price=%.3f | graph_prob=%.3f | edge=%.1f%% | confidence=%.2f",
                sig.market_id[:16],
                sig.side,
                sig.market_price,
                sig.graph_prob,
                sig.edge * 100,
                sig.confidence,
            )
            for trigger in sig.causal_triggers:
                logger.info("  Trigger: %s", trigger)
            result = self._executor.execute(sig)
            if result.success:
                self._predictions.record(sig, result.size_usdc)

        # Status summary
        status = self._executor.status_summary()
        logger.info(
            "Status: open=%d | bankroll=$%.2f | daily_pnl=$%.2f | loss_pct=%s",
            status["open_positions"],
            status["bankroll_usdc"],
            status["daily_pnl_usdc"],
            status["daily_loss_pct"],
        )

    async def run_forever(self) -> None:
        logger.info("=" * 60)
        logger.info("POLYMARKET INTELLIGENCE BOT — Reality Graph Edge")
        logger.info("=" * 60)
        logger.info("DRY_RUN:         %s", DRY_RUN)
        logger.info("Graph URL:       %s", GRAPH_URL)
        logger.info("Scan interval:   %ds", SCAN_INTERVAL)
        logger.info("Top signals/cycle: %d", TOP_SIGNALS_PER_CYCLE)
        logger.info("Bankroll:        $%.2f USDC", self._executor.bankroll)
        logger.info("=" * 60)

        if DRY_RUN:
            logger.warning("*** DRY_RUN MODE — no real orders will be placed ***")
            logger.warning("*** Set DRY_RUN=false to enable live trading ***")

        while True:
            try:
                await self.run_cycle()
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)
            logger.info("Next cycle in %ds...", SCAN_INTERVAL)
            await asyncio.sleep(SCAN_INTERVAL)

    async def close(self) -> None:
        await self._mapper.close()
        self._predictions.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not DRY_RUN and not POLYGON_PRIVATE_KEY:
        logger.error("POLYGON_PRIVATE_KEY required for live trading. Set DRY_RUN=true to test without key.")
        sys.exit(1)

    clob_client = _build_clob_client()
    bankroll = _get_bankroll(clob_client)
    bot = PolymarketBot(clob_client=clob_client, bankroll=bankroll)

    try:
        await bot.run_forever()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
