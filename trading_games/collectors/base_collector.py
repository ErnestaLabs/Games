"""
BaseCollector — shared plumbing for every data flock collector.

Each collector subclass:
  1. Fetches raw data from one external source
  2. Normalises into graph node dicts
  3. Pushes to Forage Graph /ingest/bulk
  4. Returns the count of nodes pushed

All I/O is synchronous (httpx). Collectors run inside a thread pool
managed by MarketPulse_Watcher / NewsFlow_Watcher / ResultFlow_Watcher.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FORAGE_GRAPH_URL = os.environ.get("FORAGE_GRAPH_URL", "https://forage-graph-production.up.railway.app")
GRAPH_API_SECRET = os.environ.get("GRAPH_API_SECRET", "")

_BULK_LIMIT = 200   # nodes per /ingest/bulk call


class BaseCollector(ABC):
    """Abstract base for all data flock collectors."""

    source_name: str = "unknown"

    def __init__(self) -> None:
        self._http = httpx.Client(timeout=15.0)
        self._push_count = 0

    # ── Subclass interface ────────────────────────────────────────────────────

    @abstractmethod
    def collect(self) -> list[dict]:
        """
        Fetch raw data and return a list of graph node dicts.
        Each dict must have at least {"id": str, "type": str, "name": str}.
        """

    # ── Graph push ────────────────────────────────────────────────────────────

    def push_to_graph(self, nodes: list[dict]) -> int:
        """
        Push nodes to Forage Graph in batches.
        Returns total node count pushed successfully.
        """
        if not nodes or not GRAPH_API_SECRET:
            return 0

        pushed = 0
        for i in range(0, len(nodes), _BULK_LIMIT):
            batch = nodes[i : i + _BULK_LIMIT]
            try:
                resp = self._http.post(
                    f"{FORAGE_GRAPH_URL}/ingest/bulk",
                    headers={
                        "Authorization": f"Bearer {GRAPH_API_SECRET}",
                        "Content-Type": "application/json",
                    },
                    json={"nodes": batch, "source": self.source_name},
                    timeout=12.0,
                )
                if resp.status_code in (200, 201, 204):
                    pushed += len(batch)
                    logger.debug(
                        "[%s] pushed %d nodes | status=%d",
                        self.source_name, len(batch), resp.status_code,
                    )
                else:
                    logger.warning(
                        "[%s] graph push failed: %d %s",
                        self.source_name, resp.status_code, resp.text[:120],
                    )
            except Exception as exc:
                logger.warning("[%s] graph push error: %s", self.source_name, exc)

        self._push_count += pushed
        return pushed

    # ── Run helper ────────────────────────────────────────────────────────────

    def run_once(self) -> int:
        """Collect + push. Returns node count. Safe to call from thread pool."""
        t0 = time.monotonic()
        try:
            nodes = self.collect()
        except Exception as exc:
            logger.error("[%s] collect() raised: %s", self.source_name, exc)
            return 0
        pushed = self.push_to_graph(nodes)
        elapsed = time.monotonic() - t0
        logger.info(
            "[%s] run_once | collected=%d pushed=%d elapsed=%.2fs",
            self.source_name, len(nodes), pushed, elapsed,
        )
        return pushed

    def close(self) -> None:
        self._http.close()

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ts() -> int:
        """Unix timestamp in milliseconds for unique node IDs."""
        return int(time.time() * 1000)
