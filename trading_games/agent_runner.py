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

from trading_games.config import (
    DRY_RUN, STARTING_BANKROLL_USDC, SCORING_INTERVAL_SECS,
    GAME_START_DATE, GAME_DAYS,
    KALSHI_API_KEY, KALSHI_EMAIL, KALSHI_PASSWORD,
    IG_API_KEY, IG_USERNAME, IG_PASSWORD, IG_ACCOUNT_ID, IG_DEMO,
    MATCHBOOK_USERNAME, MATCHBOOK_PASSWORD,
    SMARKETS_API_KEY,
    FORAGE_GRAPH_URL, GRAPH_API_SECRET,
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
from trading_games.ig_executor import IGExecutor
from trading_games.matchbook_executor import MatchbookExecutor
from trading_games.smarkets_executor import SmarketsExecutor
from trading_games.forage_signal_source import ForageSignalSource
from trading_games.cross_venue_signal import CrossVenueSignalDetector
from trading_games.ig_intelligence import IGIntelligence
from trading_games.market_pulse_watcher import MarketPulseWatcher
from trading_games.news_flow_watcher    import NewsFlowWatcher
from trading_games.result_flow_watcher  import ResultFlowWatcher

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
        return usdc if usdc > 0 else STARTING_BANKROLL_USDC
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

    token_ids = m.get("clobTokenIds") or []
    if isinstance(token_ids, str):
        try:
            token_ids = _json.loads(token_ids)
        except Exception:
            token_ids = []

    # Token IDs are not needed for IG spread-bet execution (UK-only path)

    if not token_ids:
        token_ids = ["", ""]

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
    Fetch active Polymarket markets from Gamma API (read-only intel feed).
    CLOB API is not used — UK execution goes through IG spread betting only.
    """
    # Gamma API — public, read-only, not geoblocked, no auth required
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

    return []


# ── Signal processing ────────────────────────────────────────────────────────

def _process_agent(agent, market: dict) -> dict | None:
    """Run one agent against one market. Returns enriched signal or None."""
    try:
        signal = agent.analyze_market(market)
        if signal:
            signal["agent"] = agent.name
            # Populate token_id from market tokens if agent didn't set it
            if not signal.get("token_id") and market.get("tokens"):
                side = signal.get("side", "YES").upper()
                for tok in market["tokens"]:
                    if tok.get("outcome", "").upper() == side:
                        signal["token_id"] = tok.get("token_id", "")
                        break
            signal.setdefault("token_id", "")
            signal.setdefault("tick_size", "0.01")
            signal.setdefault("min_order_size", 1.0)
            signal.setdefault("is_fee_free", False)
            signal.setdefault("signal_type", "composite")
            signal.setdefault("causal_triggers", [])
        return signal
    except Exception as exc:
        logger.warning("[%s] analyze_market error: %s", agent.name, exc)
        return None


def _execute_ig_from_market(
    ig: "IGExecutor",
    ts: "TradeSignal",
    market: dict,
    store: "PredictionStore",
) -> bool:
    """
    Execute a signal on IG. If the market dict has a pre-mapped epic (_ig_epic),
    use it directly. Otherwise fall back to keyword search.
    Returns True if position was opened.
    """
    from trading_games.ig_epic_mapper import map_signal_to_epics

    size_usdc = ts.kelly_size * 100.0
    direction = market.get("_ig_direction") or ts.side
    epic      = market.get("_ig_epic", "")

    if not epic:
        # Map via entity/question keywords
        entity_name = market.get("entity_name", ts.question[:40])
        entity_type = market.get("entity_type", "unknown")
        signal_text = market.get("signal_text", ts.question)
        mapped = map_signal_to_epics(entity_name, entity_type, signal_text, direction)
        if mapped:
            epic      = mapped[0]["epic"]
            direction = mapped[0]["direction"]

    if not epic:
        return False

    # Auto stop: 2% of a typical index level (e.g. FTSE 8000 × 0.02 = 160 pts)
    # For forex: 200 pips. For commodities: 3% movement. All conservative.
    stop_distance = float(os.environ.get("IG_STOP_DISTANCE", "0"))  # 0 = no stop (demo safe)

    ig_result = ig.execute_from_signal(
        question=ts.question,
        side=direction,
        size_usdc=size_usdc,
        edge=ts.edge,
        epic=epic,
        stop_distance=stop_distance if stop_distance > 0 else None,
    )
    if ig_result.success:
        store.record(ts, simulated_size_usdc=size_usdc)
        logger.info(
            "IG FILLED: [%s] %s %s | £%.2f/pt | epic=%s | deal=%s",
            ts.agent, direction, ts.question[:40], size_usdc / 100, epic,
            ig_result.deal_id or "demo",
        )
        return True
    return False


def _push_trade_to_graph(venue: str, result: object, signal_context: dict) -> None:
    """Push any executed trade from any venue to the Forage Graph as a Trade node."""
    if not GRAPH_API_SECRET:
        return
    try:
        node = {
            "id":         f"trade_{venue}_{int(time.time()*1000)}",
            "type":       "Trade",
            "venue":      venue,
            "success":    getattr(result, "success", False),
            "direction":  getattr(result, "side", None) or getattr(result, "direction", ""),
            "question":   signal_context.get("question", ""),
            "agent":      signal_context.get("agent", ""),
            "edge":       signal_context.get("edge", 0.0),
            "confidence": signal_context.get("confidence", 0.0),
            "source":     f"{venue}_executor",
        }
        # Venue-specific fields
        if venue == "ig":
            node.update({
                "epic":    getattr(result, "epic", ""),
                "deal_id": getattr(result, "deal_id", ""),
                "size":    getattr(result, "size", 0.0),
                "level":   getattr(result, "level", 0.0),
            })
        elif venue == "matchbook":
            node.update({
                "bet_id":       getattr(result, "bet_id", ""),
                "event_name":   getattr(result, "event_name", ""),
                "runner_name":  getattr(result, "runner_name", ""),
                "odds":         getattr(result, "odds", 0.0),
                "stake_gbp":    getattr(result, "stake", 0.0),
            })
        elif venue == "smarkets":
            node.update({
                "order_id":     getattr(result, "order_id", ""),
                "event_name":   getattr(result, "event_name", ""),
                "contract":     getattr(result, "contract_name", ""),
                "price_bp":     getattr(result, "price_bp", 0),
                "stake_gbp":    getattr(result, "stake_gbp", 0.0),
            })
        httpx.post(
            f"{FORAGE_GRAPH_URL}/ingest/bulk",
            headers={"Authorization": f"Bearer {GRAPH_API_SECRET}", "Content-Type": "application/json"},
            json={"nodes": [node], "source": f"{venue}_trade"},
            timeout=6.0,
        )
    except Exception as exc:
        logger.debug("Graph trade push failed (%s): %s", venue, exc)


def scan_once(
    agents: list,
    store: "PredictionStore",
    engine: "ScoringEngine",
    executor: "OrderExecutor | None" = None,
    ig: "IGExecutor | None" = None,
    cross_venue: "CrossVenueSignalDetector | None" = None,
    forage_source: "ForageSignalSource | None" = None,
    ig_intel: "IGIntelligence | None" = None,
    matchbook: "MatchbookExecutor | None" = None,
    smarkets: "SmarketsExecutor | None" = None,
) -> int:
    """
    One scan: fetch signal sources, run all agents, record and execute.

    Signal priority for IG mode:
      1. IG Intelligence: IG prices + calendar + news → pattern-detected ideas (highest quality)
      2. Cross-venue divergence (PM vs Kalshi)
      3. Forage entity signals — causal intelligence
      4. Polymarket markets — baseline intel
    """
    # Build market list from all sources
    markets: list[dict] = []

    if ig:
        # IG Intelligence: pattern-detected trade ideas from IG's own data firehose
        if ig_intel:
            try:
                ideas = ig_intel.run_cycle()
                markets.extend([idea.to_market_dict() for idea in ideas])
                logger.info("[Scan] IG intelligence ideas: %d", len(ideas))
            except Exception as exc:
                logger.warning("[Scan] IG intel cycle failed: %s", exc)

        # Cross-venue divergence (Polymarket vs Kalshi)
        if cross_venue:
            try:
                cv_signals = cross_venue.detect()
                markets.extend([s.to_market_dict() for s in cv_signals])
                logger.info("[Scan] Cross-venue signals: %d", len(cv_signals))
            except Exception as exc:
                logger.warning("[Scan] Cross-venue detect failed: %s", exc)

        if forage_source:
            try:
                fg_signals = forage_source.fetch_signals()
                markets.extend(fg_signals)
                logger.info("[Scan] Forage signals: %d", len(fg_signals))
            except Exception as exc:
                logger.warning("[Scan] Forage fetch failed: %s", exc)

    # Always include Polymarket markets (intel + Polymarket execution fallback)
    pm_markets = fetch_markets()
    markets.extend(pm_markets)
    logger.info("[Scan] Polymarket markets: %d | total input signals: %d",
                len(pm_markets), len(markets))

    if not markets:
        logger.warning("No markets/signals fetched — check network")
        return 0

    logger.info("Scanning %d signals with %d agents [DRY_RUN=%s | IG=%s]...",
                len(markets), len(agents), DRY_RUN, ig is not None)
    total_signals = 0

    # ── Direct IG execution for pre-mapped signals ────────────────────────────
    # IG Intelligence ideas and cross-venue signals already have epic + direction.
    # Agents won't recognize these dicts, so execute them directly here.
    if ig:
        for market in markets:
            epic = market.get("_ig_epic", "")
            if not epic:
                continue
            confidence = float(market.get("confidence", 0))
            if confidence < 0.55:
                continue
            edge = float(market.get("edge", confidence - 0.5))
            if edge < 0.05:
                continue
            direction = market.get("_ig_direction", "BUY")
            stop_distance = float(os.environ.get("IG_STOP_DISTANCE", "0")) or None
            size_usdc = max(edge * 0.25 * 100.0, 0.50)   # Kelly-sized, min £0.50/pt equivalent

            ig_result = ig.execute_from_signal(
                question=market.get("question", "")[:120],
                side=direction,
                size_usdc=size_usdc,
                edge=edge,
                epic=epic,
                stop_distance=stop_distance,
            )
            _push_trade_to_graph("ig", ig_result, {
                "question": market.get("question", ""),
                "agent": "direct_ig",
                "edge": edge,
                "confidence": confidence,
                "market_id": market.get("market_id", ""),
            })
            if ig_result.success:
                total_signals += 1
                logger.info(
                    "IG DIRECT: %s %s | confidence=%.0f%% | pattern=%s | deal=%s",
                    direction, epic, confidence * 100,
                    market.get("_signal_type", "?"),
                    ig_result.deal_id or "demo",
                )

    # ── Agent-driven execution (Polymarket + Forage entity signals) ───────────
    for market in markets:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(_process_agent, agent, market): agent for agent in agents}
            for future in as_completed(futures):
                signal = future.result()
                if not signal:
                    continue

                try:
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
                        fee_schedule=signal.get("fee_schedule", {"maker": 0.0, "taker": 0.02}),
                        agent=signal.get("agent", ""),
                    )
                except Exception as ts_err:
                    logger.warning("TradeSignal build failed for %s: %s | signal=%s",
                                   signal.get("market_id","?")[:16], ts_err, list(signal.keys()))
                    continue

                acted = False
                ctx = {
                    "question":   ts.question,
                    "agent":      ts.agent,
                    "edge":       ts.edge,
                    "confidence": ts.confidence,
                    "market_id":  ts.market_id,
                }

                # IG spread betting — primary UK execution
                if ig and ts.edge >= 0.05:
                    acted = _execute_ig_from_market(ig, ts, market, store)
                    if acted:
                        total_signals += 1

                # Matchbook — UK prediction market exchange (YES/NO events)
                if not acted and matchbook and ts.edge >= 0.05:
                    try:
                        mb_result = matchbook.execute_from_signal(
                            question=ts.question, side=ts.side,
                            size_usdc=ts.kelly_size * 100.0, edge=ts.edge,
                        )
                        _push_trade_to_graph("matchbook", mb_result, ctx)
                        if mb_result.success:
                            store.record(ts, simulated_size_usdc=mb_result.stake)
                            total_signals += 1
                            acted = True
                    except Exception as exc:
                        logger.warning("Matchbook execute error: %s", exc)

                # Smarkets — UK prediction market exchange
                if not acted and smarkets and ts.edge >= 0.05:
                    try:
                        sm_result = smarkets.execute_from_signal(
                            question=ts.question, side=ts.side,
                            size_usdc=ts.kelly_size * 100.0, edge=ts.edge,
                        )
                        _push_trade_to_graph("smarkets", sm_result, ctx)
                        if sm_result.success:
                            store.record(ts, simulated_size_usdc=sm_result.stake_gbp)
                            total_signals += 1
                            acted = True
                    except Exception as exc:
                        logger.warning("Smarkets execute error: %s", exc)

                if not acted:
                    store.record(ts, simulated_size_usdc=10.0)
                    total_signals += 1

    logger.info("Scan complete: %d signals acted on", total_signals)
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
    store: "PredictionStore",
    engine: "ScoringEngine",
    executor: "OrderExecutor | None" = None,
    ig: "IGExecutor | None" = None,
    cross_venue: "CrossVenueSignalDetector | None" = None,
    forage_source: "ForageSignalSource | None" = None,
    ig_intel: "IGIntelligence | None" = None,
    matchbook: "MatchbookExecutor | None" = None,
    smarkets: "SmarketsExecutor | None" = None,
) -> None:
    day_today = _day_index()
    last_score_ts = 0.0

    logger.info("The Trading Games — Day %d/%d | DRY_RUN=%s", day_today + 1, GAME_DAYS, DRY_RUN)
    for a in agents:
        logger.info("  %s | %s | wallet=%s", a.display_name, a.token, a.wallet_address[:10])

    while True:
        now_ts = time.time()

        scan_once(agents, store, engine, executor, ig, cross_venue, forage_source, ig_intel, matchbook, smarkets)

        # Daily scoring at midnight UTC
        current_day = _day_index()
        if current_day != day_today or (now_ts - last_score_ts) >= SCORING_INTERVAL_SECS:
            engine.score_and_publish(current_day)
            _generate_social_posts(agents, engine)
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

    # UK execution is via IG spread betting — Polymarket CLOB is geoblocked.
    # ClobClient is not initialised; executor is None (IG handles all live trades).
    bankroll = STARTING_BANKROLL_USDC
    executor = None
    logger.info("Polymarket CLOB disabled (UK geoblocked) — execution via IG only | bankroll=$%.2f", bankroll)

    # Env-var diagnostic — logs presence (not values) of key secrets
    logger.info(
        "ENV DIAGNOSTIC | IG_API_KEY=%s IG_USERNAME=%s IG_PASSWORD=%s | STARTING_BANKROLL=%s",
        "SET" if os.environ.get("IG_API_KEY") else "MISSING",
        "SET" if os.environ.get("IG_USERNAME") else "MISSING",
        "SET" if os.environ.get("IG_PASSWORD") else "MISSING",
        os.environ.get("STARTING_BANKROLL", "NOT_SET"),
    )

    # Env-var diagnostic — logs presence (not values) of key secrets
    _ig_key_present  = bool(os.environ.get("IG_API_KEY"))
    _ig_user_present = bool(os.environ.get("IG_USERNAME"))
    _ig_pass_present = bool(os.environ.get("IG_PASSWORD"))
    logger.info(
        "ENV DIAGNOSTIC | IG_API_KEY=%s IG_USERNAME=%s IG_PASSWORD=%s | STARTING_BANKROLL=%s",
        "SET" if _ig_key_present  else "MISSING",
        "SET" if _ig_user_present else "MISSING",
        "SET" if _ig_pass_present else "MISSING",
        os.environ.get("STARTING_BANKROLL", "NOT_SET"),
    )

    # Kalshi — read-only price feed for divergence detection (no account needed)
    kalshi = KalshiExecutor()
    logger.info("Kalshi price feed active — monitoring for Poly/Kalshi divergences")

    # IG Group — UK spread betting execution (live if IG_API_KEY set)
    ig: IGExecutor | None = None
    if IG_API_KEY and IG_USERNAME and IG_PASSWORD:
        ig = IGExecutor(
            api_key=IG_API_KEY,
            username=IG_USERNAME,
            password=IG_PASSWORD,
            account_id=IG_ACCOUNT_ID,
            demo=IG_DEMO,
        )
        if ig.login():
            logger.info("IG spread betting active | demo=%s | %s", IG_DEMO, ig.status_summary())
        else:
            logger.warning("IG login failed — spread betting disabled")
            ig = None
    else:
        logger.info("IG_API_KEY/IG_USERNAME/IG_PASSWORD not set — spread betting inactive")

    # Matchbook — UK betting exchange (session-token auth)
    matchbook: MatchbookExecutor | None = None
    if MATCHBOOK_USERNAME and MATCHBOOK_PASSWORD:
        matchbook = MatchbookExecutor(username=MATCHBOOK_USERNAME, password=MATCHBOOK_PASSWORD)
        if matchbook.login():
            logger.info("Matchbook exchange active | %s", matchbook.status_summary())
        else:
            logger.warning("Matchbook login failed — exchange disabled")
            matchbook = None
    else:
        logger.info("MATCHBOOK_USERNAME/PASSWORD not set — Matchbook inactive")

    # Smarkets — UK prediction market exchange (Bearer token auth)
    smarkets: SmarketsExecutor | None = None
    if SMARKETS_API_KEY:
        smarkets = SmarketsExecutor(api_key=SMARKETS_API_KEY)
        logger.info("Smarkets exchange active | %s", smarkets.status_summary())
    else:
        logger.info("SMARKETS_API_KEY not set — Smarkets inactive")

    # ── Data flock — always-on background collectors ──────────────────────────
    market_pulse = MarketPulseWatcher()
    market_pulse.start()
    logger.info("MarketPulse_Watcher started — IG/PM/Kalshi/Matchbook/Smarkets/Onchain collectors active")

    news_flow = NewsFlowWatcher()
    news_flow.start()
    logger.info("NewsFlow_Watcher started — RSS feeds live")

    result_flow = ResultFlowWatcher()
    result_flow.start()
    logger.info("ResultFlow_Watcher started — tracking market resolutions + closed positions")

    # Cross-venue divergence detector (Polymarket vs Kalshi — read-only, no auth)
    cross_venue = CrossVenueSignalDetector()
    logger.info("Cross-venue signal detector active (PM vs Kalshi)")

    # Forage Graph signal source
    forage_source = ForageSignalSource()
    logger.info("Forage signal source active | url=%s", forage_source._url)

    # IG Intelligence: prices + calendar + news → pattern-detected trade ideas
    ig_intel: IGIntelligence | None = None
    if ig:
        ig_intel = IGIntelligence(ig)
        logger.info("IG Intelligence active — price/calendar/news pattern detection")

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
            scan_once(agents, store, engine, executor, ig, cross_venue, forage_source, ig_intel, matchbook, smarkets)
            engine.print_standings()
        else:
            run_forever(agents, store, engine, executor, ig, cross_venue, forage_source, ig_intel, matchbook, smarkets)
    finally:
        market_pulse.stop()
        news_flow.stop()
        result_flow.stop()
        store.close()
        engine.close()
        if ig:
            ig.close()
        if ig_intel:
            ig_intel.close()
        if matchbook:
            matchbook.close()
        if smarkets:
            smarkets.close()
        cross_venue.close()
        forage_source.close()
        for a in agents:
            a.close()


if __name__ == "__main__":
    main()
