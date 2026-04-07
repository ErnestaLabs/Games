"""
NewsCollector — RSS/Atom feed monitor + headline ingestion.

Pushes to Forage Graph:
  Source    — each article/item as a Source node with text + URL
  Narrative — grouped topic threads when multiple articles share keywords

Feeds are public RSS. No auth required. Parse with stdlib xml.etree.
Keywords trigger priority flagging for the watcher agents.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

# Priority RSS feeds — mix of finance, politics, prediction-market adjacent
DEFAULT_FEEDS: list[str] = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.skynews.com/feeds/rss/politics.xml",
    "https://www.theguardian.com/politics/rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
]

EXTRA_FEEDS = [f.strip() for f in os.environ.get("NEWS_FEEDS", "").split(",") if f.strip()]
ALL_FEEDS = DEFAULT_FEEDS + EXTRA_FEEDS

# Keywords that flag an article as high-priority
PRIORITY_KEYWORDS = [
    "fed rate", "fomc", "interest rate", "rate cut", "rate hike",
    "trump", "election", "ukraine", "russia", "nato",
    "bitcoin", "crypto", "ethereum", "recession", "gdp",
    "opec", "oil", "gold", "inflation", "cpi", "pce",
    "polymarket", "kalshi", "prediction market",
]

MAX_ITEMS_PER_FEED = int(os.environ.get("NEWS_MAX_ITEMS", "20"))


@dataclass
class Article:
    url:       str
    title:     str
    summary:   str
    published: str
    feed:      str
    priority:  bool = False
    keywords:  list[str] = field(default_factory=list)

    @property
    def uid(self) -> str:
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]


class NewsCollector(BaseCollector):
    source_name = "news_collector"

    def collect(self) -> list[dict]:
        nodes: list[dict] = []
        ts = self._ts()
        articles = self._fetch_all_feeds()

        # Group by keyword for narrative stitching
        keyword_groups: dict[str, list[Article]] = {}

        for art in articles:
            nodes.append({
                "id":         f"news_{art.uid}",
                "type":       "Source",
                "name":       art.title[:200],
                "url":        art.url,
                "text":       art.summary[:500],
                "published":  art.published,
                "feed":       art.feed,
                "priority":   art.priority,
                "keywords":   art.keywords,
                "timestamp_ms": ts,
                "source":     self.source_name,
            })

            for kw in art.keywords:
                keyword_groups.setdefault(kw, []).append(art)

        # Narrative nodes for hot topics (≥3 articles same keyword)
        for kw, arts in keyword_groups.items():
            if len(arts) < 3:
                continue
            nodes.append({
                "id":       f"narrative_{kw.replace(' ', '_')}_{ts // 3_600_000}",
                "type":     "Narrative",
                "name":     f"Topic cluster: {kw}",
                "keyword":  kw,
                "article_count": len(arts),
                "headlines": [a.title[:100] for a in arts[:5]],
                "timestamp_ms": ts,
                "source":   self.source_name,
            })

        priority_count = sum(1 for a in articles if a.priority)
        logger.info(
            "[news] feeds=%d articles=%d priority=%d nodes=%d",
            len(ALL_FEEDS), len(articles), priority_count, len(nodes),
        )
        return nodes

    # ── Feed parsing ──────────────────────────────────────────────────────────

    def _fetch_all_feeds(self) -> list[Article]:
        articles: list[Article] = []
        for feed_url in ALL_FEEDS:
            try:
                resp = self._http.get(feed_url, timeout=10.0,
                                      headers={"User-Agent": "ForageFlock/1.0"})
                if resp.status_code == 200:
                    arts = self._parse_feed(resp.text, feed_url)
                    articles.extend(arts)
                else:
                    logger.debug("[news] feed %s: %d", feed_url, resp.status_code)
            except Exception as exc:
                logger.debug("[news] feed error %s: %s", feed_url, exc)
        return articles

    def _parse_feed(self, xml_text: str, feed_url: str) -> list[Article]:
        articles: list[Article] = []
        try:
            root = ET.fromstring(xml_text)
            # RSS 2.0
            items = root.findall(".//item")
            # Atom fallback
            if not items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                items = root.findall(".//atom:entry", ns)

            for item in items[:MAX_ITEMS_PER_FEED]:
                title   = self._tag(item, ["title"]) or ""
                url     = self._tag(item, ["link", "guid"]) or ""
                summary = self._tag(item, ["description", "summary", "content"]) or ""
                pubdate = self._tag(item, ["pubDate", "published", "updated"]) or ""

                # Strip HTML from summary
                summary = re.sub(r"<[^>]+>", " ", summary).strip()[:600]

                matched = [kw for kw in PRIORITY_KEYWORDS
                           if kw in (title + " " + summary).lower()]
                articles.append(Article(
                    url=url, title=title, summary=summary,
                    published=pubdate, feed=feed_url,
                    priority=bool(matched), keywords=matched,
                ))
        except Exception as exc:
            logger.debug("[news] parse error (%s): %s", feed_url, exc)
        return articles

    @staticmethod
    def _tag(elem: ET.Element, tags: list[str]) -> str | None:
        for t in tags:
            node = elem.find(t)
            if node is not None:
                text = (node.text or "").strip()
                if text:
                    return text
            # Check attribs (Atom <link href=...>)
            for child in elem:
                if child.tag.endswith(t) or child.tag == t:
                    href = child.get("href")
                    if href:
                        return href
                    text = (child.text or "").strip()
                    if text:
                        return text
        return None
