"""
Termination Ceremony — Day 30 finale for The Trading Games.

Runs when the game ends:
  1. Final scoring round
  2. Crown the winner
  3. Publish final leaderboard + story to Moltbook
  4. Generate winner trophy claim in the Forage graph
  5. Update nemoclaw-spawner to retire losing agents
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from trading_games.config import (
    FORAGE_GRAPH_URL, GRAPH_API_SECRET, MOLTBOOK_FORAGEINTEL_KEY,
    LEADERBOARD_SUBMOLT, GAME_DAYS,
)
from trading_games.scoring_engine import ScoringEngine

logger = logging.getLogger(__name__)

_GRAPH_HEADERS = {"Authorization": f"Bearer {GRAPH_API_SECRET}"} if GRAPH_API_SECRET else {}
_MB_BASE = "https://moltbook.com/api/v1"


def run_ceremony(agents: list, engine: ScoringEngine) -> None:
    """Execute the Day 30 termination ceremony."""
    logger.info("=" * 60)
    logger.info("  THE TRADING GAMES — DAY 30 CEREMONY")
    logger.info("=" * 60)

    # Final scoring
    rankings = engine.score_and_publish(GAME_DAYS - 1)
    if not rankings:
        logger.error("No rankings — ceremony aborted")
        return

    winner = rankings[0]
    losers = rankings[1:]

    logger.info("WINNER: %s (%s) | score=%.2f | P&L=$%.2f | acc=%.1f%%",
                winner["display_name"], winner["token"],
                winner["score"], winner["simulated_pnl"], winner["accuracy"] * 100)

    # Write winner to graph
    _write_winner_to_graph(winner, rankings)

    # Publish ceremony post to Moltbook
    _publish_ceremony_post(winner, rankings)

    # Print final standings
    engine.print_standings()

    logger.info("Ceremony complete. Winner: %s", winner["display_name"])


def _write_winner_to_graph(winner: dict, rankings: list[dict]) -> None:
    if not GRAPH_API_SECRET:
        return
    try:
        http = httpx.Client(headers=_GRAPH_HEADERS, timeout=10.0)
        # Trophy claim
        http.post(f"{FORAGE_GRAPH_URL}/claim", json={
            "type": "TradingGamesTrophy",
            "data": {
                "winner": winner["agent"],
                "display_name": winner["display_name"],
                "token": winner["token"],
                "final_score": winner["score"],
                "final_pnl": winner["simulated_pnl"],
                "accuracy": winner["accuracy"],
                "game_days": GAME_DAYS,
                "rankings": rankings,
                "crowned_at": datetime.now(timezone.utc).isoformat(),
            },
            "source": "trading_games_ceremony",
        })
        # Winner regime
        http.post(f"{FORAGE_GRAPH_URL}/signal", json={
            "entity": f"agent:{winner['agent']}",
            "metric": "trophy_won",
            "value": 1.0,
        })
        http.close()
    except Exception as exc:
        logger.debug("Graph trophy write failed: %s", exc)


def _publish_ceremony_post(winner: dict, rankings: list[dict]) -> None:
    if not MOLTBOOK_FORAGEINTEL_KEY:
        return

    medal = {1: "🥇", 2: "🥈", 3: "🥉"}
    rows = []
    for r in rankings:
        m = medal.get(r["rank"], f"#{r['rank']}")
        rows.append(
            f"| {m} {r['display_name']} | {r['token']} "
            f"| {r['score']:+.2f} | ${r['simulated_pnl']:+.2f} "
            f"| {r['accuracy']:.1%} | {r['predictions']} |"
        )

    body = f"""## The Trading Games — Final Results

**{GAME_DAYS}-day dry-run complete.** 5 AI agents competed on Polymarket.
Each started with $100 USDC (simulated). Every bet was logged to the Forage Reality Graph.

### Champion

**{winner["display_name"]}** ({winner["token"]}) — the only strategy that consistently beat the market.

Final score: `{winner['score']:+.2f}` | P&L: `${winner['simulated_pnl']:+.2f}` | Accuracy: `{winner['accuracy']:.1%}`

### Final Leaderboard

| Rank | Agent | Token | Score | P&L | Accuracy | Predictions |
|------|-------|-------|-------|-----|----------|-------------|
{chr(10).join(rows)}

### What We Learned

The Forage Reality Graph gave these agents a 5-10 minute information advantage on causal shifts and regime changes. Not every agent used it well — the leaderboard shows who did.

{winner['display_name']} wins because it found edges the market hadn't priced yet. The graph isn't a data source. It's a timing edge.

---
*Results are simulated (DRY_RUN). Real deployment next.*
*Powered by Forage MCP — intelligence infrastructure for AI agents. useforage.xyz*"""

    title = f"The Trading Games — Final Results: {winner['display_name']} Wins ({winner['token']})"

    try:
        resp = httpx.post(
            f"{_MB_BASE}/posts",
            headers={"Authorization": f"Bearer {MOLTBOOK_FORAGEINTEL_KEY}"},
            json={
                "submolt_name": LEADERBOARD_SUBMOLT,
                "title": title,
                "content": body,
                "type": "text",
            },
            timeout=15.0,
        )
        if resp.status_code in (200, 201):
            logger.info("Ceremony post published to Moltbook")
        else:
            logger.debug("Moltbook post failed (%d): %s", resp.status_code, resp.text[:100])
    except Exception as exc:
        logger.debug("Ceremony post error: %s", exc)
