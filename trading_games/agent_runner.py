"""
Trading Games Agent Runner — orchestrates all 5 agents for 30 days.

Each agent runs its own analysis loop. The runner:
  1. Fetches Polymarket markets every SCAN_INTERVAL seconds
  2. Passes each market to all 5 agents in parallel
  3. Collects signals, routes to PredictionStore (dry-run) or OrderExecutor (live)
  4. Scores daily and publishes leaderboard

Run:
  python -m trading_games.agent_runner           # daemon mode
  python -m trading_games.agent_runner --once    # single scan and exit
  python -m trading_games.agent_runner --score   # print standings and exit
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, date

import httpx
from dotenv import load_dotenv

# Load trading.env (clean KEY=VALUE file) from CWD (Forage_Landing root).
# override=True ensures our values beat any stale vars already in the shell env.
load_dotenv(dotenv_path=os.path.join(os.getcwd(), "trading.env"), override=True)

from trading_games.config import (
    CLOB_HOST, DRY_RUN, STARTING_BANKROLL_USDC, SCORING_INTERVAL_SECS,
    GAME_START_DATE, GAME_DAYS, POLYGON_PRIVATE_KEY,
    KALSHI_API_KEY, KALSHI_EMAIL, KALSHI_PASSWORD,
)
from trading_games.scoring_engine import ScoringEngine
from trading_games.agents.arbitor       import ArbitorAgent
from trading_games.agents.causal_prophet import CausalProphetAgent
from trading_games.agents.yield_siphon  import YieldSiphonAgent
from trading_games.agents.news_bolt     import NewsBoltAgent
from trading_games.agents.smart_watcher import SmartWatcherAgent
from polymarket.prediction_store import PredictionStore
from polymarket.order_executor import OrderExecutor
from trading_games.kalshi_executor import KalshiExecutor

try:
    from trading_games.betfair_executor import BetfairExecutor
    from trading_games.ig_executor import IGClient
except ImportError:
    BetfairExecutor = None  # type: ignore[assignment,misc]
    IGClient = None  # type: ignore[assignment,misc]

try:
    from trading_games.matchbook_executor import MatchbookExecutor
except ImportError:
    MatchbookExecutor = None  # type: ignore[assignment,misc]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "600"))   # 10 minutes
MARKET_PAGE_SIZE = int(os.environ.get("MARKET_PAGE_SIZE", "50"))


# ── ClobClient setup ─────────────────────────────────────────────────────────

def _build_clob_client():
    """Build authenticated ClobClient. Returns None if key not set (read-only)."""
    try:
        from py_clob_client.client import ClobClient
        relayer_key = os.environ.get("POLYMARKET_RELAYER_API_KEY", "")
        if not POLYGON_PRIVATE_KEY:
            logger.warning("POLYGON_PRIVATE_KEY not set — read-only mode, no order placement")
            return ClobClient(host=CLOB_HOST, chain_id=137)
        client = ClobClient(
            host=CLOB_HOST,
            key=POLYGON_PRIVATE_KEY,
            chain_id=137,
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        logger.info("ClobClient initialised with L2 auth (live trading ready)")
        return client
    except ImportError:
        logger.warning("py-clob-client not installed — market data only")
        return None
    except Exception as exc:
        logger.error("ClobClient init failed: %s", exc)
        return None


def _get_balance(clob_client) -> float:
    if not clob_client or not POLYGON_PRIVATE_KEY:
        return STARTING_BANKROLL_USDC
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        # asset_type=1 = USDC (collateral), token_id=None
        result = clob_client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        raw = result.get("balance") or result.get("allowance") or "0"
        # USDC has 6 decimals on Polygon
        usdc = float(raw) / 1e6
        logger.info("Polygon wallet USDC balance: $%.2f", usdc)
        return usdc  # return real balance — 0.0 disables Polymarket executor
    except Exception as exc:
        logger.warning("Balance fetch failed (%s) — using $%.2f", exc, STARTING_BANKROLL_USDC)
        return STARTING_BANKROLL_USDC


# ── Market fetcher ────────────────────────────────────────────────────────────

def _normalise_market(m: dict) -> dict:
    """
    Normalise Gamma API market to the format agents expect.
    Ensures m['tokens'] = [{"outcome": "YES", "price": float, "token_id": str}, ...]
    Gamma API returns outcomePrices as a JSON-encoded string, not a list.
    """
    import json as _json
    if m.get("tokens") and isinstance(m["tokens"], list) and m["tokens"]:
        return m  # Already normalised (CLOB format)

    # Parse outcomePrices — may be a JSON string or a list
    raw_prices = m.get("outcomePrices") or "[]"
    if isinstance(raw_prices, str):
        try:
            raw_prices = _json.loads(raw_prices)
        except Exception:
            raw_prices = []

    outcomes = m.get("outcomes") or ["Yes", "No"]
    if isinstance(outcomes, str):
        try:
            outcomes = _json.loads(outcomes)
        except Exception:
            outcomes = ["Yes", "No"]

    token_ids = m.get("clobTokenIds") or ["", ""]
    if isinstance(token_ids, str):
        try:
            token_ids = _json.loads(token_ids)
        except Exception:
            token_ids = []

    tokens = []
    for i, outcome in enumerate(outcomes):
        tokens.append({
            "outcome": str(outcome).upper(),
            "price": float(raw_prices[i]) if i < len(raw_prices) else 0.0,
            "token_id": token_ids[i] if i < len(token_ids) else "",
        })
    m["tokens"] = tokens
    m.setdefault("condition_id", m.get("conditionId", ""))
    m.setdefault("market_id", m.get("id", ""))
    return m


def fetch_markets(limit: int = MARKET_PAGE_SIZE) -> list[dict]:
    """
    Fetch active, non-closed Polymarket markets ordered by volume (highest first).
    Uses Gamma API for current market discovery; CLOB API as fallback.
    Normalises all markets to a common token/price structure.
    """
    try:
        resp = httpx.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": limit,
                    "order": "volume", "ascending": "false"},
            timeout=15.0,
        )
        if resp.status_code == 200:
            markets = resp.json()
            if isinstance(markets, list) and markets:
                return [_normalise_market(m) for m in markets]
    except Exception as exc:
        logger.debug("Gamma API failed: %s", exc)

    # Fallback to CLOB API
    try:
        resp = httpx.get(
            f"{CLOB_HOST}/markets",
            params={"active": "true", "closed": "false", "limit": limit},
            timeout=15.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            raw = data.get("data") or data.get("markets") or []
            return [_normalise_market(m) for m in raw]
    except Exception as exc:
        logger.warning("Market fetch failed: %s", exc)
    return []


# ── Signal processing ────────────────────────────────────────────────────────

def _process_agent(agent, market: dict) -> dict | None:
    """Run one agent against one market. Returns enriched signal or None."""
    try:
        signal = agent.analyze_market(market)
        if signal:
            signal["agent"] = agent.name
            signal.setdefault("token_id", "")
            signal.setdefault("tick_size", "0.01")
            signal.setdefault("min_order_size", 1.0)
            signal.setdefault("is_fee_free", False)
            signal.setdefault("signal_type", "composite")
            signal.setdefault("causal_triggers", [])
        return signal
    except Exception as exc:
        logger.debug("[%s] analyze_market error: %s", agent.name, exc)
        return None


def scan_once(
    agents: list,
    store: PredictionStore,
    engine: ScoringEngine,
    executor: OrderExecutor | None = None,
    betfair_exec=None,
    ig_exec=None,
    matchbook_exec=None,
) -> int:
    """One scan: fetch markets, run all agents, record signals, execute live orders."""
    markets = fetch_markets()
    if not markets:
        logger.warning("No markets fetched — check CLOB_HOST or network")
        return 0

    logger.info("Scanning %d markets with %d agents [DRY_RUN=%s]...",
                len(markets), len(agents), DRY_RUN)
    total_signals = 0
    signals_list = []

    for market in markets:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(_process_agent, agent, market): agent for agent in agents}
            for future in as_completed(futures):
                signal = future.result()
                if not signal:
                    continue

                from polymarket.edge_calculator import TradeSignal
                ts = TradeSignal(
                    market_id=signal["market_id"],
                    question=signal["question"],
                    side=signal.get("side", "YES"),
                    token_id=signal.get("token_id", ""),
                    tick_size=signal.get("tick_size", "0.01"),
                    market_price=signal.get("market_price", 0.5),
                    graph_prob=signal.get("graph_prob", 0.5),
                    edge=signal.get("edge", 0.0),
                    kelly_size=signal.get("edge", 0.0) * 0.25,
                    min_order_size=signal.get("min_order_size", 1.0),
                    signal_type=signal.get("signal_type", "composite"),
                    causal_triggers=signal.get("causal_triggers", []),
                    confidence=signal.get("confidence", 0.5),
                    is_fee_free=signal.get("is_fee_free", False),
                    fee_schedule=signal.get("fee_schedule", {}),
                )

                # Always record for scoring + UK venue routing
                if executor:
                    result = executor.execute(ts)
                    size = result.size_usdc if result.success else 10.0
                    store.record(ts, simulated_size_usdc=size)
                else:
                    store.record(ts, simulated_size_usdc=10.0)

                # Route to UK/EU execution venues regardless of Polymarket result
                total_signals += 1
                signals_list.append(ts)

    logger.info("Scan complete: %d signals acted on", total_signals)

    # Route signals to UK execution venues
    if ig_exec is not None:
        # Always run crypto momentum scanner — independent of Polymarket signals
        ig_exec.scan_crypto_direct()

    if signals_list:
        if betfair_exec is not None:
            betfair_exec.scan_and_execute(signals_list)
        if ig_exec is not None:
            ig_exec.scan_and_execute(signals_list)
        if matchbook_exec is not None:
            for ts in signals_list:
                try:
                    matchbook_exec.execute_from_signal(
                        question=ts.question,
                        side=ts.side,
                        size_usdc=max(2.0, ts.kelly_size * 20.0),  # min £2
                        edge=ts.edge,
                    )
                except Exception as exc:
                    logger.debug("Matchbook execute error: %s", exc)

    return total_signals


# ── Daily scoring ─────────────────────────────────────────────────────────────

def _day_index() -> int:
    return max(0, (date.today() - GAME_START_DATE).days)


def _generate_social_posts(agents: list, engine: ScoringEngine) -> None:
    """Generate and log a social post for each agent based on today's standings."""
    day     = _day_index() + 1
    rankings = engine.rank_agents()
    rank_map = {r["agent"]: r for r in rankings}

    for agent in agents:
        stats = rank_map.get(agent.name, {})
        # Find best signal today (rough proxy: highest edge prediction)
        preds  = engine._load_predictions()
        my_preds = [p for p in preds if p.get("agent") == agent.name and p.get("edge", 0) > 0.05]
        best   = max(my_preds, key=lambda p: p.get("edge", 0), default=None)
        context = {
            "day": day,
            "best_signal": best,
            "rank": stats.get("rank", "?"),
            "score": stats.get("score", 0),
        }
        post = agent.generate_post(context)
        if post:
            logger.info("[%s] Social post draft:\n%s", agent.name, post[:200])


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_forever(
    agents: list,
    store: PredictionStore,
    engine: ScoringEngine,
    executor: OrderExecutor | None = None,
    betfair_exec=None,
    ig_exec=None,
    matchbook_exec=None,
) -> None:
    day_today = _day_index()
    last_score_ts = 0.0

    logger.info("The Trading Games — Day %d/%d | DRY_RUN=%s", day_today + 1, GAME_DAYS, DRY_RUN)
    for a in agents:
        logger.info("  %s | %s | wallet=%s", a.display_name, a.token, a.wallet_address[:10])

    while True:
        now_ts = time.time()

        scan_once(agents, store, engine, executor,
                  betfair_exec=betfair_exec, ig_exec=ig_exec, matchbook_exec=matchbook_exec)

        # Daily scoring at midnight UTC
        current_day = _day_index()
        if current_day != day_today or (now_ts - last_score_ts) >= SCORING_INTERVAL_SECS:
            engine.score_and_publish(current_day)
            last_score_ts = now_ts
            day_today = current_day

        if current_day >= GAME_DAYS:
            logger.info("The Trading Games — Day 30 complete!")
            from trading_games.termination_ceremony import run_ceremony
            run_ceremony(agents, engine)
            break

        logger.info("Next scan in %ds", SCAN_INTERVAL)
        time.sleep(SCAN_INTERVAL)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="The Trading Games agent runner")
    parser.add_argument("--once",  action="store_true", help="Single scan and exit")
    parser.add_argument("--score", action="store_true", help="Print standings and exit")
    args = parser.parse_args()

    # ── Wallet + ClobClient ───────────────────────────────────────────────
    clob   = _build_clob_client()
    bankroll = _get_balance(clob)

    if not DRY_RUN:
        if not POLYGON_PRIVATE_KEY:
            logger.error("POLYGON_PRIVATE_KEY required for live trading. Set it in .env then rerun.")
            return
        logger.info("LIVE TRADING ACTIVE — bankroll=$%.2f USDC | wallet=%s",
                    bankroll, (clob.get_address() if hasattr(clob, 'get_address') else "N/A"))
    else:
        logger.info("DRY_RUN=True — no real orders (set DRY_RUN=false to go live)")

    # Only attempt Polymarket execution if wallet has funds.
    # Signals always route to IG + Betfair regardless.
    poly_balance = bankroll if POLYGON_PRIVATE_KEY else 0.0
    if poly_balance > 1.0:
        executor = OrderExecutor(
            clob_client=clob,
            initial_bankroll=bankroll,
            dry_run=DRY_RUN,
        )
        logger.info("Polymarket executor active — $%.2f USDC", poly_balance)
    else:
        executor = None
        logger.info("Polymarket wallet empty — execution via IG + Betfair only")

    # Kalshi — read-only price feed for divergence detection (no account needed)
    kalshi = KalshiExecutor()
    logger.info("Kalshi price feed active — monitoring for Poly/Kalshi divergences")

    # UK execution venues
    betfair = BetfairExecutor() if BetfairExecutor is not None else None
    ig = IGClient() if IGClient is not None else None
    matchbook = MatchbookExecutor() if MatchbookExecutor is not None else None

    if betfair is not None:
        if betfair._authenticated:
            logger.info("Betfair Exchange connected — UK execution active")
        else:
            logger.warning("Betfair not authenticated — check BETFAIR_USERNAME/PASSWORD/APP_KEY")
    else:
        logger.warning("BetfairExecutor not available — betfair_executor.py not yet installed")
    if ig is not None:
        if ig._authenticated:
            logger.info("IG Group connected — spread bet execution active")
        else:
            logger.warning("IG not authenticated — check IG_API_KEY/IG_USERNAME/IG_PASSWORD")
    else:
        logger.warning("IGClient not available — ig_executor.py not yet installed")
    if matchbook is not None:
        if matchbook._ensure_session():
            logger.info("Matchbook connected — UK exchange execution active")
        elif matchbook._account_locked:
            logger.warning("Matchbook account LOCKED — execution disabled (contact Matchbook support)")
            matchbook = None  # don't pass it downstream, avoids per-signal noise
        else:
            logger.warning("Matchbook not authenticated — check MATCHBOOK_USERNAME/PASSWORD")
    else:
        logger.warning("MatchbookExecutor not available")

    agents  = [
        ArbitorAgent(),
        CausalProphetAgent(),
        YieldSiphonAgent(),
        NewsBoltAgent(),
        SmartWatcherAgent(),
    ]
    store   = PredictionStore()
    engine  = ScoringEngine(agents)

    try:
        if args.score:
            engine.print_standings()
        elif args.once:
            scan_once(agents, store, engine, executor,
                      betfair_exec=betfair, ig_exec=ig, matchbook_exec=matchbook)
            engine.print_standings()
        else:
            run_forever(agents, store, engine, executor,
                        betfair_exec=betfair, ig_exec=ig, matchbook_exec=matchbook)
    finally:
        store.close()
        engine.close()
        for a in agents:
            a.close()
        if betfair is not None:
            betfair.close()
        if ig is not None:
            ig.close()
        if matchbook is not None:
            matchbook.close()


if __name__ == "__main__":
    main()
