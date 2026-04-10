"""
BaseAgent — shared base class for all 5 Trading Games agents.

Each agent subclass implements:
  analyze_market(market: MappedMarket) -> TradeSignal | None
  generate_post(context: dict) -> str

Everything else (graph writes, LLM routing, wallet derivation, scoring hooks)
is handled here.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import httpx

from trading_games.config import (
    FORAGE_GRAPH_URL, GRAPH_API_SECRET, APIFY_TOKEN, FORAGE_ENDPOINT,
    MASTER_MNEMONIC, AGENT_WALLET_INDICES, AGENT_MOLTLAUNCH_IDS,
    DRY_RUN, STARTING_BANKROLL_USDC,
)
from trading_games.llm_router import Priority, llm

logger = logging.getLogger(__name__)

_GRAPH_HEADERS = {"Authorization": f"Bearer {GRAPH_API_SECRET}"} if GRAPH_API_SECRET else {}

# Circuit breaker: stop calling MCP after 3 consecutive 406/failures per process
_mcp_fail_count: int = 0
_MCP_FAIL_LIMIT: int = 3


class BaseAgent(ABC):
    """
    Abstract base for a Trading Games agent.

    Subclasses must set:
      name          str  — e.g. "arbitor"
      display_name  str  — e.g. "The Arbitor"
      token         str  — e.g. "$ARB"
      description   str  — one-line strategy description

    And implement:
      analyze_market(market) -> TradeSignal | None
      generate_post(context) -> str
    """

    name: str = ""
    display_name: str = ""
    token: str = ""
    description: str = ""

    def __init__(self) -> None:
        self._http = httpx.Client(headers=_GRAPH_HEADERS, timeout=12.0)
        self._bankroll = STARTING_BANKROLL_USDC
        self._wallet_index = AGENT_WALLET_INDICES.get(self.name, 0)
        self._moltlaunch_id = AGENT_MOLTLAUNCH_IDS.get(self.name, "")
        self._wallet_address: str = self._derive_wallet()
        logger.info(
            "[%s] Init | wallet[%d]=%s | DRY_RUN=%s",
            self.name, self._wallet_index,
            self._wallet_address[:10] + "...", DRY_RUN,
        )

    # ── Wallet derivation ─────────────────────────────────────────────────

    def _derive_wallet(self) -> str:
        """Derive Polygon wallet from MASTER_MNEMONIC via BIP44 m/44'/137'/0'/0/{index}."""
        if not MASTER_MNEMONIC:
            return f"0x{'0' * 40}"  # placeholder in dry-run / no-mnemonic mode
        try:
            from eth_account import Account
            Account.enable_unaudited_hdwallet_features()
            path = f"m/44'/137'/0'/0/{self._wallet_index}"
            account = Account.from_mnemonic(MASTER_MNEMONIC, account_path=path)
            return account.address
        except Exception as exc:
            logger.debug("Wallet derivation failed: %s", exc)
            return f"0x{'0' * 40}"

    @property
    def wallet_address(self) -> str:
        return self._wallet_address

    # ── LLM shortcuts ─────────────────────────────────────────────────────

    def think_critical(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        return llm(Priority.CRITICAL, system, prompt, max_tokens)

    def think_high(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        return llm(Priority.HIGH, system, prompt, max_tokens)

    def think_medium(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        return llm(Priority.MEDIUM, system, prompt, max_tokens)

    def think_low(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        return llm(Priority.LOW, system, prompt, max_tokens)

    # ── Forage Graph writes ───────────────────────────────────────────────

    def graph_signal(self, entity: str, metric: str, value: float) -> None:
        """Write a signal to the Forage Reality Graph."""
        if not GRAPH_API_SECRET:
            return
        try:
            self._http.post(
                f"{FORAGE_GRAPH_URL}/signal",
                json={"entity": entity, "metric": metric, "value": value},
            )
        except Exception as exc:
            logger.debug("graph_signal failed: %s", exc)

    def graph_claim(self, entity: str, claim_type: str, data: dict) -> None:
        """Write a claim node to the Forage Reality Graph."""
        if not GRAPH_API_SECRET:
            return
        try:
            self._http.post(
                f"{FORAGE_GRAPH_URL}/claim",
                json={
                    "type": claim_type,
                    "data": {**data, "agent": self.name, "wallet": self._wallet_address},
                    "source": f"trading_games_{self.name}",
                },
            )
        except Exception as exc:
            logger.debug("graph_claim failed: %s", exc)

    def graph_ingest(self, entities: list[dict]) -> None:
        """Bulk-ingest entities into the graph."""
        if not GRAPH_API_SECRET or not entities:
            return
        try:
            self._http.post(
                f"{FORAGE_GRAPH_URL}/ingest/bulk",
                json={"entities": entities},
            )
        except Exception as exc:
            logger.debug("graph_ingest failed: %s", exc)

    # ── Forage MCP ────────────────────────────────────────────────────────

    def forage_query(self, query: str) -> dict:
        """Query the Forage Knowledge Graph via MCP."""
        global _mcp_fail_count
        if not APIFY_TOKEN or _mcp_fail_count >= _MCP_FAIL_LIMIT:
            return {}
        try:
            resp = httpx.post(
                f"{FORAGE_ENDPOINT}/mcp",
                headers={
                    "Authorization": f"Bearer {APIFY_TOKEN}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {
                        "name": "query_knowledge",
                        "arguments": {"query": query},
                    },
                },
                timeout=20.0,
            )
            if resp.status_code == 406:
                _mcp_fail_count += 1
                return {}
            _mcp_fail_count = 0
            result = resp.json().get("result", {})
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            logger.debug("forage_query failed: %s", exc)
            return {}

    def forage_causal_parents(self, entity_id: str) -> list[dict]:
        return self._forage_tool("get_causal_parents", {"entity_id": entity_id})

    def forage_causal_children(self, entity_id: str) -> list[dict]:
        return self._forage_tool("get_causal_children", {"entity_id": entity_id})

    def _forage_tool(self, tool: str, args: dict) -> list:
        global _mcp_fail_count
        if not APIFY_TOKEN or _mcp_fail_count >= _MCP_FAIL_LIMIT:
            return []
        try:
            resp = httpx.post(
                f"{FORAGE_ENDPOINT}/mcp",
                headers={
                    "Authorization": f"Bearer {APIFY_TOKEN}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": tool, "arguments": args},
                },
                timeout=20.0,
            )
            if resp.status_code == 406:
                _mcp_fail_count += 1
                if _mcp_fail_count >= _MCP_FAIL_LIMIT:
                    logger.warning(
                        "Forage MCP returning 406 — disabling for this session. "
                        "Check FORAGE_ENDPOINT or Apify actor status."
                    )
                return []
            _mcp_fail_count = 0  # reset on success
            result = resp.json().get("result", [])
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.debug("forage_tool %s failed: %s", tool, exc)
            return []

    # ── Bankroll management ───────────────────────────────────────────────

    @property
    def bankroll(self) -> float:
        return self._bankroll

    def update_bankroll(self, pnl: float) -> None:
        self._bankroll = max(0.0, self._bankroll + pnl)
        self.graph_signal(f"agent:{self.name}", "bankroll_usdc", self._bankroll)

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def analyze_market(self, market: dict) -> Optional[dict]:
        """
        Evaluate a market and return a trade signal dict or None.
        Signal dict keys: market_id, question, side, confidence, edge, reasoning
        """

    @abstractmethod
    def generate_post(self, context: dict) -> str:
        """
        Generate a social post (Moltbook / Twitter) for a given context.
        context keys vary by subclass; at minimum includes 'day' and 'signal'.
        """

    # ── Scoring data ──────────────────────────────────────────────────────

    def to_score_record(self) -> dict:
        return {
            "agent": self.name,
            "display_name": self.display_name,
            "token": self.token,
            "moltlaunch_id": self._moltlaunch_id,
            "bankroll_usdc": self._bankroll,
            "wallet": self._wallet_address,
        }

    def close(self) -> None:
        self._http.close()
