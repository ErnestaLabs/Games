"""
ResolutionChecker — runs daily, resolves PredictionRecords against Polymarket outcomes.

Flow:
  1. Load all unresolved PredictionRecords from local JSONL
  2. For each: fetch market from Polymarket CLOB API
  3. If market is resolved: determine correct side, compute simulated P&L
  4. Update record: outcome = "correct" | "incorrect", simulated_pnl = float
  5. Write back to JSONL + patch graph node

Simulated P&L:
  If correct: simulated_pnl = simulated_size_usdc × (1/entry_price - 1)
  If incorrect: simulated_pnl = -simulated_size_usdc

Run:
  python polymarket/resolution_checker.py           # one-shot
  python polymarket/resolution_checker.py --loop    # run daily at midnight UTC
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import httpx

from polymarket.prediction_store import PredictionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLOB_HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com")


class ResolutionChecker:
    def __init__(self) -> None:
        self._store = PredictionStore()
        self._http = httpx.Client(timeout=15.0)

    def _fetch_market(self, market_id: str) -> dict | None:
        try:
            resp = self._http.get(f"{CLOB_HOST}/markets/{market_id}")
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            logger.debug("Fetch market %s failed: %s", market_id, exc)
        return None

    def _is_resolved(self, market: dict) -> bool:
        """Market is resolved when closed=True and a winning token is set."""
        closed = market.get("closed") or market.get("active") is False
        resolved = market.get("resolved") or market.get("resolutionTime") or market.get("resolvedBy")
        return bool(closed and resolved)

    def _winning_side(self, market: dict, our_side: str) -> str | None:
        """
        Returns "YES" or "NO" based on which token resolved at 1.0.
        Returns None if market not yet resolved or ambiguous.
        """
        tokens = market.get("tokens") or []
        for token in tokens:
            winner = token.get("winner") or token.get("winnerToken")
            outcome = (token.get("outcome") or "").upper()
            if winner and outcome in ("YES", "NO"):
                return outcome
        # Alt: check resolution price
        for token in tokens:
            price = float(token.get("price") or token.get("currentPrice") or 0.0)
            outcome = (token.get("outcome") or "").upper()
            if price >= 0.99 and outcome in ("YES", "NO"):
                return outcome
        return None

    def _compute_pnl(self, record: dict, correct: bool) -> float:
        size = float(record.get("simulated_size_usdc") or 5.0)
        entry_price = float(record.get("market_probability") or 0.5)
        if correct:
            if entry_price <= 0 or entry_price >= 1:
                return 0.0
            # Profit = size × (1/price - 1) — e.g. 0.30 price → 2.33× return
            return round(size * (1.0 / entry_price - 1.0), 4)
        else:
            return round(-size, 4)

    def check_all(self) -> dict:
        records = self._store.load_all()
        unresolved = [r for r in records if r.get("outcome") is None]

        if not unresolved:
            logger.info("No unresolved predictions to check")
            return {"checked": 0, "resolved": 0, "correct": 0, "incorrect": 0}

        logger.info("Checking %d unresolved predictions...", len(unresolved))
        stats = {"checked": len(unresolved), "resolved": 0, "correct": 0, "incorrect": 0}

        for rec in unresolved:
            market_id = rec["market_id"]
            market = self._fetch_market(market_id)
            if not market:
                continue

            if not self._is_resolved(market):
                logger.debug("Market %s not yet resolved", market_id[:16])
                continue

            winning_side = self._winning_side(market, rec["side"])
            if not winning_side:
                logger.debug("Market %s resolved but winner unclear", market_id[:16])
                continue

            correct = winning_side == rec["side"]
            outcome = "correct" if correct else "incorrect"
            pnl = self._compute_pnl(rec, correct)

            self._store.update_outcome(rec["prediction_id"], outcome, pnl)
            stats["resolved"] += 1
            stats["correct" if correct else "incorrect"] += 1

            logger.info(
                "[%s] %s %s → winner=%s | %s | simulated_pnl=$%.2f",
                market_id[:16],
                rec["side"],
                f"@ {rec['market_probability']:.3f}",
                winning_side,
                outcome.upper(),
                pnl,
            )
            time.sleep(0.2)  # gentle rate limit

        logger.info(
            "Resolution check complete: %d resolved, %d correct, %d incorrect",
            stats["resolved"], stats["correct"], stats["incorrect"],
        )
        return stats

    def print_summary(self) -> None:
        """Print running accuracy + simulated P&L from all resolved records."""
        records = self._store.load_all()
        resolved = [r for r in records if r.get("outcome") is not None]
        unresolved = [r for r in records if r.get("outcome") is None]

        if not resolved:
            print(f"\nNo resolved predictions yet. {len(unresolved)} pending.\n")
            return

        correct = [r for r in resolved if r["outcome"] == "correct"]
        accuracy = len(correct) / len(resolved)
        total_pnl = sum(r.get("simulated_pnl") or 0.0 for r in resolved)
        avg_edge = sum(abs(r.get("edge") or 0.0) for r in resolved) / len(resolved)

        print(f"""
╔══════════════════════════════════════════════════╗
║         POLYMARKET BOT — 30-DAY VALIDATION       ║
╠══════════════════════════════════════════════════╣
║ Total predictions:    {len(records):>6}                    ║
║ Resolved:             {len(resolved):>6}                    ║
║ Pending:              {len(unresolved):>6}                    ║
║                                                  ║
║ Accuracy:             {accuracy:>6.1%}                    ║
║ Simulated P&L:        ${total_pnl:>+8.2f}                  ║
║ Avg edge taken:       {avg_edge:>6.1%}                    ║
╠══════════════════════════════════════════════════╣
║ GO LIVE?  accuracy > 55% AND P&L > 0             ║
║           {'✓ YES — flip DRY_RUN=false' if accuracy > 0.55 and total_pnl > 0 else '✗ NO  — tune edge_calculator.py first':<42} ║
╚══════════════════════════════════════════════════╝
""")

    def close(self) -> None:
        self._store.close()
        self._http.close()


def _seconds_until_midnight_utc() -> float:
    now = datetime.now(timezone.utc)
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    tomorrow += timedelta(days=1)
    return (tomorrow - now).total_seconds()


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket resolution checker")
    parser.add_argument("--loop", action="store_true", help="Run daily at midnight UTC")
    parser.add_argument("--summary", action="store_true", help="Print accuracy summary only")
    args = parser.parse_args()

    checker = ResolutionChecker()

    if args.summary:
        checker.print_summary()
        checker.close()
        return

    if args.loop:
        logger.info("Resolution checker running daily at midnight UTC")
        while True:
            checker.check_all()
            checker.print_summary()
            wait = _seconds_until_midnight_utc()
            logger.info("Next check in %.0fh", wait / 3600)
            time.sleep(wait)
    else:
        checker.check_all()
        checker.print_summary()
        checker.close()


if __name__ == "__main__":
    main()
