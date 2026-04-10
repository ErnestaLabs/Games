"""
ScoringEngine — daily leaderboard for The Trading Games.

Scores each agent on:
  - Simulated P&L (primary rank)
  - Prediction accuracy (tie-breaker)
  - Sharpness (Brier score)
  - Signal count (volume)

Publishes daily leaderboard to:
  - Forage Reality Graph (claim node)
  - Moltbook r/reality-games submolt
  - Graph signal per agent: score, rank, pnl

Run:
  from trading_games.scoring_engine import ScoringEngine
  engine = ScoringEngine(agents)
  engine.score_and_publish(day_index)
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from trading_games.config import (
    FORAGE_GRAPH_URL, GRAPH_API_SECRET, GAME_START_DATE, GAME_DAYS,
    MOLTBOOK_FORAGEINTEL_KEY, LEADERBOARD_SUBMOLT,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "polymarket" / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.jsonl"
SCORES_FILE = DATA_DIR / "trading_games_scores.jsonl"

_GRAPH_HEADERS = {"Authorization": f"Bearer {GRAPH_API_SECRET}"} if GRAPH_API_SECRET else {}
_MB_BASE = "https://moltbook.com/api/v1"


class ScoringEngine:
    def __init__(self, agents: list[Any]) -> None:
        self._agents = agents
        self._http = httpx.Client(headers=_GRAPH_HEADERS, timeout=12.0)
        self._graph_down = False  # circuit breaker: skip after first 5xx

    # ── Scoring ───────────────────────────────────────────────────────────

    def _load_predictions(self) -> list[dict]:
        if not PREDICTIONS_FILE.exists():
            return []
        records = []
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _agent_stats(self, agent_name: str, predictions: list[dict]) -> dict:
        """Compute per-agent scoring stats from resolved predictions."""
        mine = [p for p in predictions if p.get("agent") == agent_name]
        resolved = [p for p in mine if p.get("outcome") is not None]

        if not resolved:
            return {
                "predictions": len(mine),
                "resolved": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "simulated_pnl": 0.0,
                "brier_score": 1.0,
                "score": 0.0,
            }

        correct = [r for r in resolved if r["outcome"] == "correct"]
        accuracy = len(correct) / len(resolved)
        total_pnl = sum(r.get("simulated_pnl") or 0.0 for r in resolved)

        # Brier score: mean squared error of probability forecasts
        brier = 0.0
        brier_n = 0
        for r in resolved:
            prob = r.get("our_probability") or r.get("market_probability") or 0.5
            actual = 1.0 if r["outcome"] == "correct" else 0.0
            brier += (prob - actual) ** 2
            brier_n += 1
        brier_score = brier / brier_n if brier_n else 1.0

        # Composite score: P&L (primary) + accuracy bonus
        score = total_pnl + (accuracy - 0.5) * 10  # ±5 bonus for accuracy vs baseline

        return {
            "predictions": len(mine),
            "resolved": len(resolved),
            "correct": len(correct),
            "incorrect": len(resolved) - len(correct),
            "accuracy": round(accuracy, 4),
            "simulated_pnl": round(total_pnl, 4),
            "brier_score": round(brier_score, 4),
            "score": round(score, 4),
        }

    def rank_agents(self, predictions: list[dict] | None = None) -> list[dict]:
        """Return agents sorted by composite score descending."""
        if predictions is None:
            predictions = self._load_predictions()

        rankings = []
        for agent in self._agents:
            stats = self._agent_stats(agent.name, predictions)
            row = {
                **agent.to_score_record(),
                **stats,
            }
            rankings.append(row)

        rankings.sort(key=lambda r: r["score"], reverse=True)
        for i, row in enumerate(rankings):
            row["rank"] = i + 1

        return rankings

    # ── Graph publish ─────────────────────────────────────────────────────

    def _publish_to_graph(self, day: int, rankings: list[dict]) -> None:
        if not GRAPH_API_SECRET or self._graph_down:
            return
        try:
            resp = self._http.post(
                f"{FORAGE_GRAPH_URL}/claim",
                json={
                    "type": "TradingGamesLeaderboard",
                    "data": {
                        "day": day,
                        "game_start": GAME_START_DATE.isoformat(),
                        "rankings": rankings,
                        "published_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "source": "trading_games_scoring",
                },
            )
            if resp.status_code >= 500:
                self._graph_down = True
                logger.warning("Forage Graph %d — disabling graph publish for this session", resp.status_code)
                return
            # Per-agent signals
            for row in rankings:
                if self._graph_down:
                    break
                entity = f"agent:{row['agent']}"
                for metric, val in [("rank", row["rank"]), ("score", row["score"]), ("pnl_usd", row["simulated_pnl"])]:
                    r = self._http.post(
                        f"{FORAGE_GRAPH_URL}/signal",
                        json={"entity": entity, "metric": metric, "value": val},
                    )
                    if r.status_code >= 500:
                        self._graph_down = True
                        break
        except Exception as exc:
            self._graph_down = True
            logger.debug("Graph leaderboard publish failed (disabling): %s", exc)

    # ── Moltbook publish ──────────────────────────────────────────────────

    def _leaderboard_text(self, day: int, rankings: list[dict]) -> tuple[str, str]:
        """Return (title, body) for Moltbook post."""
        title = f"The Trading Games — Day {day}/{GAME_DAYS} Leaderboard"

        lines = [
            f"**Day {day} of {GAME_DAYS}** — 5 AI agents trading live on Polymarket.\n",
            "| Rank | Agent | Token | Score | P&L | Accuracy | Predictions |",
            "|------|-------|-------|-------|-----|----------|-------------|",
        ]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}
        for row in rankings:
            r = row["rank"]
            m = medal.get(r, f"#{r}")
            lines.append(
                f"| {m} | {row['display_name']} | {row['token']} "
                f"| {row['score']:+.2f} | ${row['simulated_pnl']:+.2f} "
                f"| {row['accuracy']:.1%} | {row['predictions']} |"
            )

        lines += [
            "",
            f"*Leaderboard updates daily. Day {GAME_DAYS} winner keeps the crown forever.*",
            "",
            "*Powered by Forage MCP — live intelligence for AI agents. useforage.xyz*",
        ]

        return title, "\n".join(lines)

    def _publish_to_moltbook(self, day: int, rankings: list[dict]) -> None:
        return  # disabled — Moltbook auth returns 403
        title, body = self._leaderboard_text(day, rankings)
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
                logger.info("Moltbook leaderboard posted: Day %d", day)
            else:
                logger.debug("Moltbook post failed (%d): %s", resp.status_code, resp.text[:120])
        except Exception as exc:
            logger.debug("Moltbook publish error: %s", exc)

    # ── Main entry ────────────────────────────────────────────────────────

    def score_and_publish(self, day_index: int) -> list[dict]:
        """Compute rankings for this day and publish. Returns rankings list."""
        day = day_index + 1  # human-readable day number
        predictions = self._load_predictions()
        rankings = self.rank_agents(predictions)

        logger.info("=== Day %d Leaderboard ===", day)
        for row in rankings:
            logger.info(
                "  #%d %s (%s) — score=%.2f | P&L=$%.2f | acc=%.1f%% | preds=%d",
                row["rank"], row["display_name"], row["token"],
                row["score"], row["simulated_pnl"],
                row["accuracy"] * 100, row["predictions"],
            )

        self._publish_to_graph(day, rankings)
        self._publish_to_moltbook(day, rankings)
        self._save_scores(day, rankings)
        return rankings

    def _save_scores(self, day: int, rankings: list[dict]) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SCORES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "day": day,
                "rankings": rankings,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")

    def print_standings(self) -> None:
        """Pretty-print current standings to stdout."""
        rankings = self.rank_agents()
        from trading_games.config import GAME_DAYS
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"{'THE TRADING GAMES -- STANDINGS':^62}")
        print(sep)
        medal = {1: "[1]", 2: "[2]", 3: "[3]"}
        for row in rankings:
            m = medal.get(row["rank"], f"[{row['rank']}]")
            print(
                f"{m} {row['display_name']:<20} {row['token']:<8} "
                f"score={row['score']:>+7.2f}  P&L=${row['simulated_pnl']:>+7.2f}  "
                f"acc={row['accuracy']:.0%}"
            )
        print(sep + "\n")

    def close(self) -> None:
        self._http.close()
