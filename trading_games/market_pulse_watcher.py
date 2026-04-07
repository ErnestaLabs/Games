"""
MarketPulse_Watcher — always-on market data orchestrator.

Runs all market-side collectors in a thread pool on a fixed schedule:
  - PolymarketCollector  every 60 s
  - KalshiCollector      every 60 s
  - MatchbookCollector   every 90 s
  - SmarketsCollector    every 90 s
  - IGCollector          every 30 s  (fastest-moving prices)
  - OnchainCollector     every 120 s

All output is pushed directly to Forage Graph by each collector.
This watcher only manages scheduling + health stats.

Run standalone:
  python -m trading_games.market_pulse_watcher

Or imported and started in a thread from agent_runner.py.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from trading_games.collectors.polymarket_collector  import PolymarketCollector
from trading_games.collectors.kalshi_collector      import KalshiCollector
from trading_games.collectors.matchbook_collector   import MatchbookCollector
from trading_games.collectors.smarkets_collector    import SmarketsCollector
from trading_games.collectors.ig_collector          import IGCollector
from trading_games.collectors.onchain_collector     import OnchainCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("market_pulse_watcher")


@dataclass
class CollectorSchedule:
    name:        str
    collector:   object
    interval_s:  int
    last_run:    float = 0.0
    run_count:   int   = 0
    error_count: int   = 0


class MarketPulseWatcher:
    """
    Manages and schedules all market-side data collectors.
    Thread-safe: each collector runs in its own daemon thread slot,
    but only one invocation per collector runs at a time (lock per collector).
    """

    def __init__(self) -> None:
        self._schedules: list[CollectorSchedule] = [
            CollectorSchedule("ig",          IGCollector(),          interval_s=30),
            CollectorSchedule("polymarket",   PolymarketCollector(),  interval_s=60),
            CollectorSchedule("kalshi",       KalshiCollector(),      interval_s=60),
            CollectorSchedule("matchbook",    MatchbookCollector(),   interval_s=90),
            CollectorSchedule("smarkets",     SmarketsCollector(),    interval_s=90),
            CollectorSchedule("onchain",      OnchainCollector(),     interval_s=120),
        ]
        self._locks: dict[str, threading.Lock] = {
            s.name: threading.Lock() for s in self._schedules
        }
        self._stop_event = threading.Event()
        self._executor_threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start background scheduling loop."""
        logger.info("MarketPulse_Watcher starting | %d collectors", len(self._schedules))
        t = threading.Thread(target=self._loop, daemon=True, name="MarketPulseWatcher")
        t.start()
        self._executor_threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for s in self._schedules:
            try:
                s.collector.close()
            except Exception:
                pass
        logger.info("MarketPulse_Watcher stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            for sched in self._schedules:
                if now - sched.last_run >= sched.interval_s:
                    sched.last_run = now
                    self._dispatch(sched)
            time.sleep(5)

    def _dispatch(self, sched: CollectorSchedule) -> None:
        """Fire collector in a separate daemon thread (non-blocking)."""
        lock = self._locks[sched.name]
        if not lock.acquire(blocking=False):
            logger.debug("[%s] still running — skipping this tick", sched.name)
            return

        def _run() -> None:
            try:
                pushed = sched.collector.run_once()
                sched.run_count += 1
                logger.debug("[%s] run #%d pushed=%d", sched.name, sched.run_count, pushed)
            except Exception as exc:
                sched.error_count += 1
                logger.error("[%s] collector error: %s", sched.name, exc)
            finally:
                lock.release()

        t = threading.Thread(target=_run, daemon=True, name=f"collector_{sched.name}")
        t.start()

    def status(self) -> dict:
        return {
            s.name: {"runs": s.run_count, "errors": s.error_count, "interval_s": s.interval_s}
            for s in self._schedules
        }


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    watcher = MarketPulseWatcher()
    watcher.start()
    try:
        while True:
            time.sleep(60)
            logger.info("MarketPulse status: %s", watcher.status())
    except KeyboardInterrupt:
        watcher.stop()


if __name__ == "__main__":
    main()
