"""
Oracle — the meta-intelligence engine at the top of the org.

Mandate: continuously aggregate everything from the Reality Graph and every
agent, detect non-obvious patterns, and emit Revelation nodes that the rest
of the org must react to.

"Continuously" means:
  - Fixed cadence loop (ORACLE_INTERVAL_S, default 300 s / 5 min)
  - Event-triggered early wakeup (large mispricings, regime shifts, big news,
    drawdowns, new signals)

Loop per cycle:
  1. INGEST  — snapshot graph (Narratives, Signals, Trades, OddsSnapshots,
               PredictionMarkets, PriceSnapshots, Regimes, Sources)
  2. DETECT  — find mispricings, regime shifts, emerging alpha, failing agents,
               new tools
  3. SYNTHESIZE — Claude (claude-sonnet-4-6) turns detections into revelations
  4. PUBLISH — write Revelation nodes + edges to graph; log structured tasks
  5. MONITOR — track which revelations led to action (future cycle check)

Run standalone:
  python -m trading_games.oracle

Or imported + started from agent_runner.py.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from trading_games.oracle_team import OracleTeam, TeamOutput, Proposal, Critique

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oracle")

# ── Config ────────────────────────────────────────────────────────────────────

FORAGE_GRAPH_URL  = os.environ.get("FORAGE_GRAPH_URL",  "https://forage-graph-production.up.railway.app")
GRAPH_API_SECRET  = os.environ.get("GRAPH_API_SECRET",  "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

ORACLE_INTERVAL_S   = int(os.environ.get("ORACLE_INTERVAL",  "300"))   # 5 min default
ORACLE_DAILY_S      = int(os.environ.get("ORACLE_DAILY",     "86400"))  # deep daily pass
ORACLE_MAX_NODES    = int(os.environ.get("ORACLE_MAX_NODES", "400"))    # graph nodes to ingest per cycle
ORACLE_MODEL        = os.environ.get("ORACLE_MODEL", "claude-sonnet-4-6")

# Revelation urgency levels
URGENCY_CRITICAL = "critical"
URGENCY_HIGH     = "high"
URGENCY_MEDIUM   = "medium"
URGENCY_LOW      = "low"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Revelation:
    revelation_id: str
    title:         str
    description:   str
    domains:       list[str]          # trading, hiring, research, infra, sim, data
    confidence:    float              # 0–1
    urgency:       str                # critical / high / medium / low
    evidence:      list[str]          # node IDs or brief quotes
    actions:       list[dict]         # [{target: str, instruction: str}]
    cycle:         int
    timestamp_ms:  int = field(default_factory=lambda: int(time.time() * 1000))

    def to_graph_node(self) -> dict:
        return {
            "id":           self.revelation_id,
            "type":         "Revelation",
            "name":         self.title,
            "description":  self.description,
            "domains":      self.domains,
            "confidence":   self.confidence,
            "urgency":      self.urgency,
            "evidence":     self.evidence,
            "actions":      json.dumps(self.actions),
            "cycle":        self.cycle,
            "timestamp_ms": self.timestamp_ms,
            "source":       "oracle",
        }


# ── Oracle ────────────────────────────────────────────────────────────────────

class Oracle:
    """
    The Oracle — continuously reads the Reality Graph + agent outputs and
    formulates high-value revelations.
    """

    def __init__(self) -> None:
        self._http         = httpx.Client(timeout=20.0)
        self._stop_event   = threading.Event()
        self._cycle        = 0
        self._last_daily   = 0.0
        self._revelation_history: list[Revelation] = []
        self._event_flag   = threading.Event()   # for early wakeup on triggers
        self._team         = OracleTeam()         # Data-Synthesist, Market-Strategist, Risk-Critic

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info(
            "Oracle awakening | interval=%ds | model=%s | graph=%s",
            ORACLE_INTERVAL_S, ORACLE_MODEL, FORAGE_GRAPH_URL,
        )
        t = threading.Thread(target=self._loop, daemon=True, name="Oracle")
        t.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._event_flag.set()   # unblock any wait
        self._team.close()
        self._http.close()
        logger.info("Oracle stopped after %d cycles | revelations=%d",
                    self._cycle, len(self._revelation_history))

    def trigger(self, reason: str = "") -> None:
        """External trigger — forces an early Oracle cycle."""
        logger.info("Oracle triggered early: %s", reason)
        self._event_flag.set()

    def status(self) -> dict:
        return {
            "cycles": self._cycle,
            "revelations_emitted": len(self._revelation_history),
            "last_urgencies": [r.urgency for r in self._revelation_history[-5:]],
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._run_cycle()
            # Wait for interval OR early trigger
            self._event_flag.wait(timeout=ORACLE_INTERVAL_S)
            self._event_flag.clear()

    def _run_cycle(self) -> None:
        self._cycle += 1
        t0 = time.monotonic()
        logger.info("Oracle cycle #%d starting", self._cycle)

        try:
            # 1. INGEST
            graph_snapshot = self._ingest_graph()

            if not graph_snapshot:
                logger.warning("Oracle cycle #%d: empty graph snapshot — skipping synthesis", self._cycle)
                return

            # 2. DETECT (pre-filter before calling LLM)
            detections = self._detect(graph_snapshot)

            # 3. SYNTHESIZE via Claude
            revelations = self._synthesize(graph_snapshot, detections)

            # 4. PUBLISH
            for rev in revelations:
                self._publish(rev)

            # 5. MONITOR — check prior revelations for follow-through
            self._monitor_prior()

            elapsed = time.monotonic() - t0
            logger.info(
                "Oracle cycle #%d complete | revelations=%d | elapsed=%.1fs",
                self._cycle, len(revelations), elapsed,
            )

        except Exception as exc:
            logger.error("Oracle cycle #%d error: %s", self._cycle, exc)

    # ── 1. INGEST ─────────────────────────────────────────────────────────────

    def _ingest_graph(self) -> dict:
        """
        Pull a cross-type snapshot from the Forage Graph.
        Returns dict keyed by node type → list of nodes.
        """
        snapshot: dict[str, list[dict]] = {}

        node_types = [
            "Signal", "Revelation", "Narrative", "Regime",
            "Trade", "PredictionMarket", "OddsSnapshot", "PriceSnapshot",
            "Source", "Instrument",
        ]

        for ntype in node_types:
            nodes = self._query_graph(ntype, limit=max(20, ORACLE_MAX_NODES // len(node_types)))
            if nodes:
                snapshot[ntype] = nodes

        total = sum(len(v) for v in snapshot.values())
        logger.info("Oracle ingested %d nodes across %d types", total, len(snapshot))
        return snapshot

    def _query_graph(self, node_type: str, limit: int = 40) -> list[dict]:
        """Query graph for recent nodes of a given type."""
        if not GRAPH_API_SECRET:
            return []
        try:
            resp = self._http.post(
                f"{FORAGE_GRAPH_URL}/query",
                headers={
                    "Authorization": f"Bearer {GRAPH_API_SECRET}",
                    "Content-Type":  "application/json",
                },
                json={"type": node_type, "limit": limit, "order": "desc"},
                timeout=12.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else (data.get("nodes") or data.get("results") or [])
            logger.debug("Oracle graph query %s: %d", node_type, resp.status_code)
        except Exception as exc:
            logger.debug("Oracle graph query %s error: %s", node_type, exc)
        return []

    # ── 2. DETECT ─────────────────────────────────────────────────────────────

    def _detect(self, snapshot: dict) -> list[str]:
        """
        Rule-based pre-screening before LLM synthesis.
        Returns a list of textual detection strings fed to the LLM.
        """
        detections: list[str] = []

        # Mispricing detection: OddsSnapshots with large PM/Kalshi spread
        odds = snapshot.get("OddsSnapshot", [])
        poly_snaps  = [o for o in odds if o.get("venue") == "polymarket"]
        kalshi_snaps = {o.get("market_id", ""): o for o in odds if o.get("venue") == "kalshi"}
        for ps in poly_snaps:
            mid_pm = ps.get("yes_price")
            mid_k  = kalshi_snaps.get(ps.get("market_id", ""), {}).get("yes_price")
            if mid_pm is not None and mid_k is not None:
                gap = abs(float(mid_pm) - float(mid_k))
                if gap > 0.08:
                    detections.append(
                        f"LARGE MISPRICING: PM={mid_pm:.2f} Kalshi={mid_k:.2f} gap={gap:.2f} "
                        f"on market '{ps.get('market_id', '')[:60]}'"
                    )

        # Narrative clusters with high article count → regime signal
        for n in snapshot.get("Narrative", []):
            count = int(n.get("article_count") or 0)
            kw    = n.get("keyword") or ""
            if count >= 5:
                detections.append(
                    f"NARRATIVE CLUSTER: '{kw}' | {count} articles — potential regime shift"
                )

        # Recent losses (failed trades)
        trades = snapshot.get("Trade", [])
        failed = [t for t in trades if t.get("success") is False or float(t.get("pnl") or 0) < -10]
        if len(failed) >= 3:
            venues = list({t.get("venue", "?") for t in failed})
            detections.append(
                f"TRADE FAILURES: {len(failed)} failed trades | venues={venues} — check execution chain"
            )

        # High-urgency prior signals not acted on
        for sig in snapshot.get("Signal", [])[:20]:
            if float(sig.get("confidence") or 0) > 0.80:
                detections.append(
                    f"HIGH-CONFIDENCE SIGNAL pending: {sig.get('name', '')[:80]} "
                    f"| confidence={sig.get('confidence'):.2f} | source={sig.get('source', '?')}"
                )

        # Prior revelations with no follow-through
        prior_revs = [n for n in snapshot.get("Revelation", []) if n.get("source") == "oracle"]
        if len(prior_revs) > 0:
            detections.append(
                f"REVELATION HISTORY: {len(prior_revs)} prior revelations in graph — "
                "check for unacted or contradicted insights"
            )

        logger.info("Oracle detected %d pre-filter signals", len(detections))
        return detections

    # ── 3. SYNTHESIZE (via Oracle Team) ──────────────────────────────────────

    def _synthesize(self, snapshot: dict, detections: list[str]) -> list[Revelation]:
        """
        Delegate to the Oracle Team (Data-Synthesist → Market-Strategist → Risk-Critic)
        then adjudicate their output into final scored Revelations.
        """
        context = self._build_context(snapshot, detections)

        if not ANTHROPIC_API_KEY:
            logger.warning("Oracle: ANTHROPIC_API_KEY not set — rule-based fallback")
            return self._rule_based_revelations(detections)

        # Run the team pipeline
        team_output: TeamOutput = self._team.run_cycle(context)

        if not team_output.proposals:
            logger.info("Oracle: no proposals from team — rule-based fallback")
            return self._rule_based_revelations(detections)

        # Adjudicate: score each proposal/critique pair into a final revelation
        return self._adjudicate(team_output, context)

    def _adjudicate(self, team_output: TeamOutput, context: str) -> list[Revelation]:
        """
        Oracle's adjudication: score each proposal against its critique,
        merge with hypothesis evidence, and produce final Revelation objects.

        Score formula (0–1):
          impact      = expected_edge × strategist_confidence × (1 - risk_score/2)
          final_conf  = mean(synthesist_conf_for_matching_hyp, strategist_conf) × (1 - risk_score/3)
          urgency     = horizon + risk_score thresholds
        """
        revelations: list[Revelation] = []
        ts = int(time.time() * 1000)

        # Build hypothesis confidence map (for evidence enrichment)
        hyp_evidence_pool = []
        for h in team_output.hypotheses:
            hyp_evidence_pool.extend(h.evidence)

        for i, (proposal, critique) in enumerate(team_output.proposal_critique_pairs()):
            risk_score  = critique.risk_score  if critique else 0.4
            risk_miti   = critique.mitigations if critique else []
            safe_var    = critique.safe_variant if critique else ""
            failure_m   = critique.failure_modes if critique else []

            # Compute final confidence
            hyp_conf_avg = (
                sum(h.confidence for h in team_output.hypotheses) / len(team_output.hypotheses)
                if team_output.hypotheses else 0.5
            )
            final_conf = min(0.95, (
                (hyp_conf_avg + proposal.confidence) / 2
            ) * (1 - risk_score / 3))

            # Urgency mapping
            if proposal.horizon == "immediate" or risk_score >= 0.75:
                urgency = URGENCY_CRITICAL if risk_score >= 0.75 else URGENCY_HIGH
            elif proposal.horizon == "days":
                urgency = URGENCY_HIGH
            else:
                urgency = URGENCY_MEDIUM

            # Assemble evidence
            evidence = list(set(proposal.evidence[:2] + hyp_evidence_pool[:2]))[:4]

            # Actions: merge proposal actions with mitigations
            actions: list[dict] = []
            for act in proposal.actions:
                actions.append({"target": "CTO", "instruction": act})
            if risk_miti:
                actions.append({"target": "Risk_Manager", "instruction": "; ".join(risk_miti[:2])})
            if safe_var and safe_var.lower() != "proceed as stated":
                actions.append({"target": "Arena_Manager", "instruction": f"Use safer variant: {safe_var[:120]}"})

            description = proposal.explanation
            if critique and critique.risks:
                description += f" Risks: {'; '.join(critique.risks[:2])}."
            if failure_m:
                description += f" Watch for: {failure_m[0][:80]}."

            rev_id = f"revelation_{self._cycle}_{i}_{ts // 1000}"
            rev = Revelation(
                revelation_id=rev_id,
                title=proposal.title,
                description=description[:600],
                domains=["trading"] + (["data"] if "collector" in proposal.explanation.lower() else []),
                confidence=final_conf,
                urgency=urgency,
                evidence=evidence,
                actions=actions,
                cycle=self._cycle,
                timestamp_ms=ts,
            )
            revelations.append(rev)
            logger.info(
                "Oracle adjudicated: '%s' | conf=%.0f%% | urgency=%s | risk=%.2f",
                rev.title[:60], final_conf * 100, urgency, risk_score,
            )

        # Sort by confidence desc, cap at 5
        return sorted(revelations, key=lambda r: -r.confidence)[:5]

    def _parse_revelation(self, raw: dict) -> Revelation:
        ts = int(time.time() * 1000)
        rev_id = f"revelation_{self._cycle}_{ts // 1000}"
        return Revelation(
            revelation_id=rev_id,
            title=str(raw.get("title", "Unnamed revelation"))[:80],
            description=str(raw.get("description", "")),
            domains=raw.get("domains") or ["trading"],
            confidence=float(raw.get("confidence") or 0.5),
            urgency=str(raw.get("urgency") or URGENCY_MEDIUM),
            evidence=raw.get("evidence") or [],
            actions=raw.get("actions") or [],
            cycle=self._cycle,
            timestamp_ms=ts,
        )

    def _rule_based_revelations(self, detections: list[str]) -> list[Revelation]:
        """Fallback when LLM unavailable — emit one revelation per detection cluster."""
        if not detections:
            return []
        ts = int(time.time() * 1000)
        rev = Revelation(
            revelation_id=f"revelation_{self._cycle}_{ts // 1000}",
            title=f"Cycle #{self._cycle}: {len(detections)} signals detected",
            description=" | ".join(detections[:5]),
            domains=["trading", "data"],
            confidence=0.6,
            urgency=URGENCY_MEDIUM if len(detections) < 3 else URGENCY_HIGH,
            evidence=detections[:3],
            actions=[{"target": "CTO", "instruction": "Review detection list and act on highest-confidence items."}],
            cycle=self._cycle,
            timestamp_ms=ts,
        )
        return [rev]

    # ── 4. PUBLISH ────────────────────────────────────────────────────────────

    def _publish(self, rev: Revelation) -> None:
        self._revelation_history.append(rev)
        node = rev.to_graph_node()

        # Push to graph
        if GRAPH_API_SECRET:
            try:
                resp = self._http.post(
                    f"{FORAGE_GRAPH_URL}/ingest/bulk",
                    headers={
                        "Authorization": f"Bearer {GRAPH_API_SECRET}",
                        "Content-Type":  "application/json",
                    },
                    json={"nodes": [node], "source": "oracle"},
                    timeout=12.0,
                )
                logger.info(
                    "REVELATION [%s] '%s' | confidence=%.0f%% | urgency=%s | graph=%d",
                    rev.revelation_id, rev.title[:60], rev.confidence * 100,
                    rev.urgency, resp.status_code,
                )
            except Exception as exc:
                logger.error("Oracle publish error: %s", exc)
        else:
            logger.info("REVELATION [no graph] '%s' | urgency=%s", rev.title[:60], rev.urgency)

        # Log structured tasks for each action
        for action in rev.actions:
            logger.info(
                "  -> ORACLE TASK [%s]: %s",
                action.get("target", "?"), action.get("instruction", "")[:100],
            )

        # Critical urgency — log prominently
        if rev.urgency == URGENCY_CRITICAL:
            logger.warning(
                "ORACLE CRITICAL REVELATION: %s — %s", rev.title, rev.description[:200]
            )

    # ── 5. MONITOR ────────────────────────────────────────────────────────────

    def _monitor_prior(self) -> None:
        """
        Check if prior critical/high revelations had any follow-through.
        For now: log them. Future: query graph for REVELATION_ACTED edges.
        """
        high_priority = [
            r for r in self._revelation_history[-20:]
            if r.urgency in (URGENCY_CRITICAL, URGENCY_HIGH)
            and r.cycle < self._cycle - 1
        ]
        if high_priority:
            logger.info(
                "Oracle monitoring: %d unconfirmed high-priority revelations from prior cycles",
                len(high_priority),
            )

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, snapshot: dict, detections: list[str]) -> str:
        sections: list[str] = []

        # Pre-filter detections
        if detections:
            sections.append("== PRE-FILTER DETECTIONS ==")
            for d in detections[:10]:
                sections.append(f"  - {d}")

        # Recent signals
        sigs = snapshot.get("Signal", [])[:10]
        if sigs:
            sections.append("\n== RECENT SIGNALS ==")
            for s in sigs:
                sections.append(
                    f"  [{s.get('signal_type','?')}] {s.get('name','')[:80]} "
                    f"| conf={s.get('confidence','?')} | src={s.get('source','?')}"
                )

        # Active prediction markets (top 10 by volume)
        markets = sorted(snapshot.get("PredictionMarket", []),
                         key=lambda m: float(m.get("volume") or 0), reverse=True)[:10]
        if markets:
            sections.append("\n== TOP MARKETS BY VOLUME ==")
            for m in markets:
                sections.append(
                    f"  [{m.get('venue','?')}] {m.get('name','')[:80]} "
                    f"| yes={m.get('yes_price','?')} | vol={m.get('volume','?')}"
                )

        # Recent trades
        trades = snapshot.get("Trade", [])[:8]
        if trades:
            sections.append("\n== RECENT TRADES ==")
            for t in trades:
                sections.append(
                    f"  [{t.get('venue','?')}] {t.get('direction','?')} "
                    f"| success={t.get('success','?')} | pnl={t.get('pnl','?')} "
                    f"| q={str(t.get('question',''))[:60]}"
                )

        # Narratives
        narrs = snapshot.get("Narrative", [])[:5]
        if narrs:
            sections.append("\n== ACTIVE NARRATIVES ==")
            for n in narrs:
                sections.append(
                    f"  '{n.get('keyword','?')}' | {n.get('article_count','?')} articles"
                )

        # Prior revelations (for context + avoid repeating)
        prior = [r for r in snapshot.get("Revelation", []) if r.get("source") == "oracle"][:5]
        if prior:
            sections.append("\n== PRIOR ORACLE REVELATIONS ==")
            for p in prior:
                sections.append(
                    f"  [{p.get('urgency','?')}] {p.get('name','')[:80]}"
                )

        # Price snapshots
        prices = snapshot.get("PriceSnapshot", [])[:5]
        if prices:
            sections.append("\n== IG PRICE SNAPSHOTS ==")
            for p in prices:
                sections.append(
                    f"  {p.get('instrument_id','?')} | mid={p.get('mid','?')} "
                    f"| pct_chg={p.get('pct_change','?')}"
                )

        return "\n".join(sections)


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    oracle = Oracle()
    oracle.start()
    try:
        while True:
            time.sleep(60)
            logger.info("Oracle status: %s", oracle.status())
    except KeyboardInterrupt:
        oracle.stop()


if __name__ == "__main__":
    main()
