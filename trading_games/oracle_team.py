"""
Oracle Team — three specialist analysts that feed the Oracle.

Each analyst receives a graph snapshot and returns candidate ideas.
The Oracle adjudicates, scores, and merges them into final Revelations.

Employees:
  DataSynthesist    — "What is changing and why?"
  MarketStrategist  — "What should we do about it?"
  RiskCritic        — "What could kill us?"

All three run in parallel. Each calls Claude with a focused, role-specific
prompt and returns a list of typed candidate objects.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("oracle_team")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ORACLE_MODEL      = os.environ.get("ORACLE_MODEL", "claude-sonnet-4-6")


# ── Candidate types ───────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """Data-Synthesist output: a coherent explanation of what's changing."""
    title:      str
    explanation: str
    evidence:   list[str]
    confidence: float   # analyst's own confidence 0–1


@dataclass
class Proposal:
    """Market-Strategist output: an actionable opportunity."""
    title:          str
    actions:        list[str]   # e.g. ["Long SP500 via IG", "Increase Smarkets allocation"]
    expected_edge:  float       # % edge estimate
    horizon:        str         # "immediate" / "days" / "weeks"
    explanation:    str
    evidence:       list[str]
    confidence:     float


@dataclass
class Critique:
    """Risk-Critic output: red-team of a proposal."""
    proposal_title:     str
    risks:              list[str]
    failure_modes:      list[str]
    mitigations:        list[str]
    risk_score:         float   # 0 = safe, 1 = very risky
    safe_variant:       str     # description of scaled-down / hedged version


# ── Base analyst ──────────────────────────────────────────────────────────────

class BaseAnalyst:
    role_name: str = "Analyst"
    role_description: str = ""

    def __init__(self) -> None:
        self._http = httpx.Client(timeout=45.0)

    def _call_llm(self, prompt: str) -> str | None:
        if not ANTHROPIC_API_KEY:
            return None
        try:
            resp = self._http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      ORACLE_MODEL,
                    "max_tokens": 1500,
                    "messages":   [{"role": "user", "content": prompt}],
                },
                timeout=40.0,
            )
            if resp.status_code == 200:
                text = resp.json()["content"][0]["text"].strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return text
            logger.warning("[%s] LLM %d: %s", self.role_name, resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.error("[%s] LLM error: %s", self.role_name, exc)
        return None

    def close(self) -> None:
        self._http.close()


# ── Employee 1: Data-Synthesist ───────────────────────────────────────────────

class DataSynthesist(BaseAnalyst):
    role_name = "Data-Synthesist"

    def analyze(self, context: str) -> list[Hypothesis]:
        """Turn raw graph data into coherent hypotheses about what is changing."""
        prompt = f"""You are the Data-Synthesist on the Oracle team at Forage/Ernesta Labs — an AI trading organization.

Your mandate: turn raw multi-source data into coherent candidate hypotheses that explain what is changing in the world and why. You are the "what and why" analyst.

Here is the current Reality Graph snapshot:

{context}

Produce 2–4 hypothesis objects. Each must explain a pattern, anomaly, or shift you can see in the data. Look for:
- Narrative accelerations or reversals
- Persistent cross-venue price gaps
- New entity clusters (new orgs, people, events appearing together)
- On-chain flow signals contradicting market prices
- Simulation outputs diverging from actual market odds

Output ONLY a JSON array where each object has:
- "title": str (≤70 chars)
- "explanation": str (2-3 sentences: what pattern, which data sources support it)
- "evidence": list[str] (up to 3 specific data points from the snapshot)
- "confidence": float (0.0–1.0, your own honest assessment)"""

        raw = self._call_llm(prompt)
        if not raw:
            return []
        try:
            items = json.loads(raw)
            return [
                Hypothesis(
                    title=i.get("title", "")[:70],
                    explanation=i.get("explanation", ""),
                    evidence=i.get("evidence") or [],
                    confidence=float(i.get("confidence") or 0.5),
                )
                for i in items if isinstance(i, dict)
            ]
        except Exception as exc:
            logger.error("[DataSynthesist] parse error: %s | raw=%s", exc, raw[:200])
            return []


# ── Employee 2: Market-Strategist ─────────────────────────────────────────────

class MarketStrategist(BaseAnalyst):
    role_name = "Market-Strategist"

    def analyze(self, context: str, hypotheses: list[Hypothesis]) -> list[Proposal]:
        """Turn hypotheses into actionable trading/org opportunities."""
        hyp_text = "\n".join(
            f"  [{i+1}] {h.title} (conf={h.confidence:.2f}): {h.explanation}"
            for i, h in enumerate(hypotheses)
        ) or "  (none — work from raw context)"

        prompt = f"""You are the Market-Strategist on the Oracle team at Forage/Ernesta Labs.

Your mandate: translate hypotheses and data into actionable, opportunity-focused proposals across trading, allocation, agent hiring, and new verticals. You are the "what should we do" analyst.

The Data-Synthesist produced these hypotheses:
{hyp_text}

Reality Graph context:
{context}

Produce 2–4 proposal objects. Each must be concrete and actionable. Proposals may include:
- Increase/decrease capital on a market vertical (IG / Matchbook / Smarkets / Polymarket)
- Spin up a new specialist agent or retire a failing one
- Shift which markets the Arena traders (Dario, Quinn, Maya) focus on
- New data source or collector to integrate
- Allocate more simulation budget to a specific scenario

Output ONLY a JSON array where each object has:
- "title": str (≤70 chars)
- "actions": list[str] (1-3 concrete steps)
- "expected_edge": float (% edge, e.g. 0.08 = 8%)
- "horizon": str ("immediate"|"days"|"weeks")
- "explanation": str (why this opportunity exists)
- "evidence": list[str] (up to 3 evidence points)
- "confidence": float (0.0–1.0)"""

        raw = self._call_llm(prompt)
        if not raw:
            return []
        try:
            items = json.loads(raw)
            return [
                Proposal(
                    title=i.get("title", "")[:70],
                    actions=i.get("actions") or [],
                    expected_edge=float(i.get("expected_edge") or 0.0),
                    horizon=i.get("horizon") or "days",
                    explanation=i.get("explanation", ""),
                    evidence=i.get("evidence") or [],
                    confidence=float(i.get("confidence") or 0.5),
                )
                for i in items if isinstance(i, dict)
            ]
        except Exception as exc:
            logger.error("[MarketStrategist] parse error: %s | raw=%s", exc, raw[:200])
            return []


# ── Employee 3: Risk-Critic ───────────────────────────────────────────────────

class RiskCritic(BaseAnalyst):
    role_name = "Risk-Critic"

    def analyze(self, context: str, proposals: list[Proposal]) -> list[Critique]:
        """Red-team proposals and produce risk-adjusted critiques."""
        prop_text = "\n".join(
            f"  [{i+1}] {p.title} | edge={p.expected_edge:.1%} | horizon={p.horizon}\n"
            f"    actions={p.actions}\n    why: {p.explanation}"
            for i, p in enumerate(proposals)
        ) or "  (none)"

        prompt = f"""You are the Risk-Critic on the Oracle team at Forage/Ernesta Labs.

Your mandate: stress-test and red-team the Strategist's proposals. Surface hidden risks, correlated exposures, regulatory or operational failure modes, and suggest safer variants. You are the "what could kill us" analyst.

Proposals to critique:
{prop_text}

Reality Graph context (for regime labels, drawdown history, worst-case simulation data):
{context}

Produce one critique object per proposal (match by title). Be intellectually honest — sometimes the answer is "this is fine, proceed."

Output ONLY a JSON array where each object has:
- "proposal_title": str (must match a proposal title above)
- "risks": list[str] (top 2-3 risks)
- "failure_modes": list[str] (top 2 ways this blows up)
- "mitigations": list[str] (concrete guardrails or hedges)
- "risk_score": float (0.0=safe, 1.0=very risky)
- "safe_variant": str (safer version of the proposal, or "proceed as stated" if low risk)"""

        raw = self._call_llm(prompt)
        if not raw:
            return []
        try:
            items = json.loads(raw)
            return [
                Critique(
                    proposal_title=i.get("proposal_title", "")[:70],
                    risks=i.get("risks") or [],
                    failure_modes=i.get("failure_modes") or [],
                    mitigations=i.get("mitigations") or [],
                    risk_score=float(i.get("risk_score") or 0.5),
                    safe_variant=i.get("safe_variant") or "",
                )
                for i in items if isinstance(i, dict)
            ]
        except Exception as exc:
            logger.error("[RiskCritic] parse error: %s | raw=%s", exc, raw[:200])
            return []


# ── Oracle Team coordinator ───────────────────────────────────────────────────

class OracleTeam:
    """
    Runs all three analysts in parallel and assembles their output
    into a structured TeamOutput for the Oracle to adjudicate.
    """

    def __init__(self) -> None:
        self.synthesist  = DataSynthesist()
        self.strategist  = MarketStrategist()
        self.critic      = RiskCritic()

    def run_cycle(self, context: str) -> "TeamOutput":
        """
        Parallel Phase 1: Synthesist + (seed Strategist/Critic with context).
        Sequential Phase 2: Strategist reads hypotheses, Critic reads proposals.
        Returns fully assembled TeamOutput.
        """
        t0 = time.monotonic()

        # Phase 1 — Synthesist works independently
        with ThreadPoolExecutor(max_workers=1) as ex:
            future_hyp = ex.submit(self.synthesist.analyze, context)
            hypotheses = future_hyp.result()

        logger.info(
            "[OracleTeam] Synthesist produced %d hypotheses", len(hypotheses)
        )

        # Phase 2 — Strategist uses hypotheses
        with ThreadPoolExecutor(max_workers=1) as ex:
            future_prop = ex.submit(self.strategist.analyze, context, hypotheses)
            proposals = future_prop.result()

        logger.info(
            "[OracleTeam] Strategist produced %d proposals", len(proposals)
        )

        # Phase 3 — Critic red-teams proposals
        with ThreadPoolExecutor(max_workers=1) as ex:
            future_crit = ex.submit(self.critic.analyze, context, proposals)
            critiques = future_crit.result()

        logger.info(
            "[OracleTeam] Critic produced %d critiques | elapsed=%.1fs",
            len(critiques), time.monotonic() - t0,
        )

        return TeamOutput(
            hypotheses=hypotheses,
            proposals=proposals,
            critiques=critiques,
        )

    def close(self) -> None:
        for analyst in (self.synthesist, self.strategist, self.critic):
            analyst.close()


@dataclass
class TeamOutput:
    hypotheses: list[Hypothesis]
    proposals:  list[Proposal]
    critiques:  list[Critique]

    def proposal_critique_pairs(self) -> list[tuple[Proposal, Critique | None]]:
        """Zip proposals with their critiques by title."""
        crit_map = {c.proposal_title.lower(): c for c in self.critiques}
        return [
            (p, crit_map.get(p.title.lower()))
            for p in self.proposals
        ]
