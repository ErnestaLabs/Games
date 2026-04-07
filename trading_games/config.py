"""
The Trading Games — central configuration.

All values read from environment variables. No defaults contain secrets.
"""
from __future__ import annotations

import os
from datetime import date

# ── Game parameters ───────────────────────────────────────────────────────────

GAME_START_DATE = date(2026, 4, 7)   # Day 1 — live trading begins midnight tonight
GAME_DAYS = 30

# $100 USDC starting bankroll per agent (simulated in DRY_RUN)
STARTING_BANKROLL_USDC = float(os.environ.get("STARTING_BANKROLL", "100.0"))

# ── LLM API keys ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")     # Moonshot / Kimi K2

# LLM model IDs
LLM_CRITICAL = "claude-sonnet-4-6"          # trade decisions, risk calls
LLM_HIGH     = "claude-haiku-4-5-20251001"  # market analysis
LLM_MEDIUM   = "moonshot-v1-8k"             # background research (Kimi K2)
LLM_LOW      = "deepseek-chat"              # cheap summarisation

# ── Wallet ────────────────────────────────────────────────────────────────────

# BIP44 derivation: m/44'/137'/0'/0/{agent_index}   (137 = Polygon)
MASTER_MNEMONIC = os.environ.get("MASTER_MNEMONIC", "")

AGENT_WALLET_INDICES: dict[str, int] = {
    "arbitor":       0,
    "causal_prophet": 1,
    "yield_siphon":  2,
    "news_bolt":     3,
    "smart_watcher": 4,
}

# ── Polymarket ────────────────────────────────────────────────────────────────

CLOB_HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com")
CLOB_WS   = os.environ.get("CLOB_WS",  "wss://ws-subscriptions-clob.polymarket.com/ws/")
POLYGON_PRIVATE_KEY = os.environ.get("POLYGON_PRIVATE_KEY", "")
DRY_RUN   = os.environ.get("DRY_RUN", "true").lower() not in ("false", "0", "no")

# ── Forage Graph ──────────────────────────────────────────────────────────────

FORAGE_GRAPH_URL  = os.environ.get("FORAGE_GRAPH_URL",
                                   "https://forage-graph-production.up.railway.app")
GRAPH_API_SECRET  = os.environ.get("GRAPH_API_SECRET", "")
APIFY_TOKEN       = os.environ.get("APIFY_TOKEN", "")
FORAGE_ENDPOINT   = os.environ.get("FORAGE_ENDPOINT",
                                   "https://ernesta-labs--forage.apify.actor")

# ── Kalshi (UK-legal execution) ───────────────────────────────────────────────

KALSHI_API_KEY  = os.environ.get("KALSHI_API_KEY", "")
KALSHI_EMAIL    = os.environ.get("KALSHI_EMAIL", "")
KALSHI_PASSWORD = os.environ.get("KALSHI_PASSWORD", "")
KALSHI_DEMO     = os.environ.get("KALSHI_DEMO", "false").lower() not in ("false", "0", "no")

# ── IG Group (UK spread betting) ─────────────────────────────────────────────

IG_API_KEY    = os.environ.get("IG_API_KEY", "")
IG_USERNAME   = os.environ.get("IG_USERNAME", "")
IG_PASSWORD   = os.environ.get("IG_PASSWORD", "")
IG_ACCOUNT_ID = os.environ.get("IG_ACCOUNT_ID", "")
IG_DEMO       = os.environ.get("IG_DEMO", "true").lower() not in ("false", "0", "no")
IG_MAX_SIZE_PER_POINT = float(os.environ.get("IG_MAX_SIZE_PER_POINT", "1.0"))
IG_STOP_DISTANCE = float(os.environ.get("IG_STOP_DISTANCE", "0"))   # 0 = no auto-stop (set >0 in prod)
DIVERGENCE_THRESHOLD = float(os.environ.get("DIVERGENCE_THRESHOLD", "0.04"))   # 4¢ PM/Kalshi gap
BOND_THRESHOLD = float(os.environ.get("BOND_THRESHOLD", "0.97"))               # near-certain bond threshold

# ── Matchbook (UK betting exchange) ──────────────────────────────────────────

MATCHBOOK_USERNAME = os.environ.get("MATCHBOOK_USERNAME", "")
MATCHBOOK_PASSWORD = os.environ.get("MATCHBOOK_PASSWORD", "")

# ── Smarkets (UK prediction market exchange) ──────────────────────────────────

SMARKETS_API_KEY = os.environ.get("SMARKETS_API_KEY", "")

# ── Hyperliquid (HIP-4 arb) ───────────────────────────────────────────────────

HYPERLIQUID_API_URL = os.environ.get(
    "HYPERLIQUID_API_URL", "https://api.hyperliquid.xyz/info"
)
HIP4_MIN_SPREAD = float(os.environ.get("HIP4_MIN_SPREAD", "0.03"))  # 3%

# ── Scoring ───────────────────────────────────────────────────────────────────

SCORING_INTERVAL_SECS = int(os.environ.get("SCORING_INTERVAL", "3600"))  # 1 hour
LEADERBOARD_SUBMOLT   = os.environ.get("LEADERBOARD_SUBMOLT", "reality-games")
MOLTBOOK_FORAGEINTEL_KEY = os.environ.get(
    "MOLTBOOK_FORAGEINTEL_KEY", "moltbook_sk_h_dyxIOCFnZuGCO5ZvAxo9RpvelbpKDr"
)

# ── MoltLaunch agent identities ───────────────────────────────────────────────

AGENT_MOLTLAUNCH_IDS: dict[str, str] = {
    "arbitor":       "38633",   # $ARB — probability compressor
    "causal_prophet": "38634",  # $PROPHET — causal chain forecaster
    "yield_siphon":  "38635",   # $YIELD — fee-free market maker
    "news_bolt":     "38636",   # $BOLT — news-driven rapid entry
    "smart_watcher": "38628",   # $WATCH — smart money copier (same as TheWatcherSees)
}
