"""
NewsFlow_Watcher — always-on news + narrative monitor.

Runs the NewsCollector on a tight loop and tracks hot topics.
When a keyword cluster reaches threshold, it:
  1. Emits a priority alert to the Forage Graph as a Signal node
  2. Logs so agent_runner agents can see breaking news in graph queries

Schedule:
  - NewsCollector  every 120 s (news moves fast but RSS updates ~2 min)

Run standalone:
  python -m trading_games.news_flow_watcher

Or imported and started in a thread from agent_runner.py.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict

import httpx

from trading_games.collectors.news_collector import NewsCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("news_flow_watcher")

FORAGE_GRAPH_URL = os.environ.get("FORAGE_GRAPH_URL", "https://forage-graph-production.up.railway.app")
GRAPH_API_SECRET = os.environ.get("GRAPH_API_SECRET", "")

NEWS_INTERVAL_S  = int(os.environ.get("NEWS_INTERVAL", "120"))
# Minimum articles on a topic before emitting a Signal node
SIGNAL_THRESHOLD = int(os.environ.get("NEWS_SIGNAL_THRESHOLD", "3"))


class NewsFlowWatcher:
    def __init__(self) -> None:
        self._collector   = NewsCollector()
        self._http        = httpx.Client(timeout=10.0)
        self._stop_event  = threading.Event()
        self._lock        = threading.Lock()
        self._run_count   = 0
        self._error_count = 0
        # Rolling keyword hit counts across last N runs
        self._keyword_counts: defaultdict[str, int] = defaultdict(int)

    def start(self) -> None:
        logger.info("NewsFlow_Watcher starting | interval=%ds", NEWS_INTERVAL_S)
        t = threading.Thread(target=self._loop, daemon=True, name="NewsFlowWatcher")
        t.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._collector.close()
        self._http.close()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                try:
                    nodes = self._collector.collect()
                    pushed = self._collector.push_to_graph(nodes)
                    self._run_count += 1

                    # Extract keyword counts from Narrative nodes
                    new_counts: defaultdict[str, int] = defaultdict(int)
                    for n in nodes:
                        if n.get("type") == "Narrative":
                            kw    = n.get("keyword") or ""
                            count = int(n.get("article_count") or 0)
                            if kw and count >= SIGNAL_THRESHOLD:
                                new_counts[kw] += count

                    # Emit Signal nodes for breaking clusters
                    for kw, count in new_counts.items():
                        if count > self._keyword_counts.get(kw, 0):
                            self._emit_signal(kw, count)
                    self._keyword_counts.update(new_counts)

                    logger.info("NewsFlow run #%d | nodes=%d pushed=%d", self._run_count, len(nodes), pushed)
                except Exception as exc:
                    self._error_count += 1
                    logger.error("NewsFlow error: %s", exc)

            self._stop_event.wait(NEWS_INTERVAL_S)

    def _emit_signal(self, keyword: str, count: int) -> None:
        """Push a Signal node to the graph for a breaking news cluster."""
        if not GRAPH_API_SECRET:
            return
        ts = int(time.time() * 1000)
        node = {
            "id":           f"news_signal_{keyword.replace(' ', '_')}_{ts // 60_000}",
            "type":         "Signal",
            "name":         f"Breaking cluster: {keyword}",
            "signal_type":  "news_cluster",
            "keyword":      keyword,
            "article_count": count,
            "confidence":   min(0.9, 0.5 + count * 0.08),
            "timestamp_ms": ts,
            "source":       "news_flow_watcher",
        }
        try:
            resp = self._http.post(
                f"{FORAGE_GRAPH_URL}/ingest/bulk",
                headers={"Authorization": f"Bearer {GRAPH_API_SECRET}", "Content-Type": "application/json"},
                json={"nodes": [node], "source": "news_flow_watcher"},
            )
            logger.info("NewsFlow signal emitted: '%s' count=%d status=%d", keyword, count, resp.status_code)
        except Exception as exc:
            logger.warning("NewsFlow signal push failed: %s", exc)

    def status(self) -> dict:
        return {
            "runs": self._run_count,
            "errors": self._error_count,
            "hot_keywords": dict(sorted(self._keyword_counts.items(), key=lambda x: -x[1])[:10]),
        }


def main() -> None:
    watcher = NewsFlowWatcher()
    watcher.start()
    try:
        while True:
            time.sleep(60)
            logger.info("NewsFlow status: %s", watcher.status())
    except KeyboardInterrupt:
        watcher.stop()


if __name__ == "__main__":
    main()
