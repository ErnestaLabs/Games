"""
LLM Router — routes inference calls to the cheapest model that can handle the task.

Priority levels:
  CRITICAL  → claude-sonnet-4-6         (trade execution, risk decisions)
  HIGH      → claude-haiku-4-5-20251001 (market analysis, signal evaluation)
  MEDIUM    → moonshot-v1-8k            (Kimi K2 — background research)
  LOW       → deepseek-chat             (cheap summarisation, formatting)

Usage:
  from trading_games.llm_router import llm, Priority
  reply = llm(Priority.HIGH, system_prompt, user_prompt)
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

import httpx

import os as _os

from trading_games.config import (
    ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, KIMI_API_KEY,
    LLM_CRITICAL, LLM_HIGH, LLM_MEDIUM, LLM_LOW,
)

OPENROUTER_API_KEY = _os.environ.get("OPENROUTER_API_KEY", "")

logger = logging.getLogger(__name__)


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


# ── Anthropic (Claude) ────────────────────────────────────────────────────────

def _claude(model: str, system: str, prompt: str, max_tokens: int = 600) -> str:
    if not ANTHROPIC_API_KEY:
        return ""
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception as exc:
        logger.debug("Claude %s error: %s", model, exc)
        return ""


# ── OpenAI-compatible (DeepSeek, Kimi) ───────────────────────────────────────

def _openai_compat(
    base_url: str, api_key: str, model: str,
    system: str, prompt: str, max_tokens: int = 600,
) -> str:
    if not api_key:
        return ""
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.debug("%s error: %s", model, exc)
        return ""


def _deepseek(system: str, prompt: str, max_tokens: int = 600) -> str:
    return _openai_compat(
        "https://api.deepseek.com/v1", DEEPSEEK_API_KEY,
        LLM_LOW, system, prompt, max_tokens,
    )


def _kimi(system: str, prompt: str, max_tokens: int = 600) -> str:
    return _openai_compat(
        "https://api.moonshot.cn/v1", KIMI_API_KEY,
        LLM_MEDIUM, system, prompt, max_tokens,
    )


def _openrouter(
    model: str, system: str, prompt: str, max_tokens: int = 600
) -> str:
    return _openai_compat(
        "https://openrouter.ai/api/v1", OPENROUTER_API_KEY,
        model, system, prompt, max_tokens,
    )


def _fusion(
    models: list[str], system: str, prompt: str, max_tokens: int = 600
) -> str:
    """
    OpenRouter Model Fusion (beta) — runs models in parallel, fuses into best result.
    Used for CRITICAL tier decisions (trade signals, risk calls).
    Passes `models` array; OpenRouter handles parallel execution and synthesis.
    """
    if not OPENROUTER_API_KEY:
        return ""
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "ForageTrading/Fusion",
            },
            json={
                "models": models,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "route": "fallback",          # fallback if fusion unavailable
                "provider": {"allow_fallbacks": True},
            },
            timeout=45.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # Fusion response may include a fused field or standard choices
        fused = data.get("fused_response") or data.get("fusion_result")
        if fused:
            return fused.strip()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.debug("Model Fusion error: %s", exc)
        return ""


# ── Public API ────────────────────────────────────────────────────────────────

def llm(
    priority: Priority,
    system: str,
    prompt: str,
    max_tokens: int = 600,
    fallback: bool = True,
) -> str:
    """
    Route to the cheapest capable model. If the primary fails and fallback=True,
    cascade to the next tier until one responds.

    Returns empty string only if all models in the cascade fail.
    """
    cascade: list[tuple[str, callable]] = []

    # OpenRouter model IDs per tier
    OR_CRITICAL = "anthropic/claude-sonnet-4-6"
    OR_HIGH     = "anthropic/claude-haiku-4-5"
    OR_MEDIUM   = "google/gemini-2.0-flash-exp:free"
    OR_LOW      = "deepseek/deepseek-chat"

    # Model Fusion ensemble for critical decisions
    FUSION_MODELS = [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4.1",
        "google/gemini-2.5-pro-preview",
    ]

    if priority == Priority.CRITICAL:
        cascade = [
            ("fusion/claude+gpt+gemini",   lambda: _fusion(FUSION_MODELS, system, prompt, max_tokens)),
            ("or/claude-sonnet-4-6",       lambda: _openrouter(OR_CRITICAL, system, prompt, max_tokens)),
            ("claude-sonnet-4-6",          lambda: _claude(LLM_CRITICAL, system, prompt, max_tokens)),
            ("or/claude-haiku-4-5",        lambda: _openrouter(OR_HIGH, system, prompt, max_tokens)),
        ]
    elif priority == Priority.HIGH:
        cascade = [
            ("or/claude-haiku-4-5",        lambda: _openrouter(OR_HIGH, system, prompt, max_tokens)),
            ("claude-haiku-4-5-20251001",  lambda: _claude(LLM_HIGH, system, prompt, max_tokens)),
            ("deepseek-chat",              lambda: _deepseek(system, prompt, max_tokens)),
        ]
    elif priority == Priority.MEDIUM:
        cascade = [
            ("or/gemini-flash",            lambda: _openrouter(OR_MEDIUM, system, prompt, max_tokens)),
            ("moonshot-v1-8k",             lambda: _kimi(system, prompt, max_tokens)),
            ("deepseek-chat",              lambda: _deepseek(system, prompt, max_tokens)),
        ]
    else:  # LOW
        cascade = [
            ("or/deepseek-chat",           lambda: _openrouter(OR_LOW, system, prompt, max_tokens)),
            ("deepseek-chat",              lambda: _deepseek(system, prompt, max_tokens)),
            ("moonshot-v1-8k",             lambda: _kimi(system, prompt, max_tokens)),
        ]

    for model_name, call in cascade:
        result = call()
        if result:
            logger.debug("LLM[%s] %s → %d chars", priority.value, model_name, len(result))
            return result
        if not fallback:
            break
        logger.debug("LLM[%s] %s returned empty — trying next tier", priority.value, model_name)

    logger.warning("LLM[%s] all tiers failed", priority.value)
    return ""
