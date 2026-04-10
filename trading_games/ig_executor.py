"""
IGClient — IG Group spread-bet execution for The Trading Games.

FCA-regulated, tax-free spread-betting via IG Group's REST API.
CRITICAL: IG imposes a daily API allowance (~100 calls). This module
uses ONLY hardcoded epics — no dynamic market search — to avoid burning
the allowance on discovery calls.

Daily budget allocation (worst case):
  1 authenticate call (cached 6h, so typically 0-1 per run)
  7 get_prices calls  (one per hardcoded epic)
  N place_order calls (≤ matched signals, typically 0-3)
  Total: ≤ 11 calls per scan cycle, well within the 100 daily limit.

Environment variables:
  IG_API_KEY      — issued by IG on API application approval
  IG_USERNAME     — IG account identifier (email or account number)
  IG_PASSWORD     — IG account password
  IG_ACCOUNT_ID   — e.g. "ZAE23" (shown in My IG > Account > Account details)
  IG_DEMO         — "true" to use demo environment (default: "false")
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

from trading_games.config import DRY_RUN

logger = logging.getLogger(__name__)

# ── API base URLs ──────────────────────────────────────────────────────────────

_DEMO_BASE = "https://demo-api.ig.com/gateway/deal"
_LIVE_BASE = "https://api.ig.com/gateway/deal"

# ── Session TTL ────────────────────────────────────────────────────────────────

_SESSION_TTL_SECS = 6 * 3600  # IG sessions expire after 6 hours

# ── Minimum edge required to place an order ────────────────────────────────────

_MIN_EDGE = 0.03  # 3% — spread must be at least this wide to justify entry

# ── Hardcoded epic map ─────────────────────────────────────────────────────────
# Format: (epic, human_label, category, keyword_list)
# NEVER add dynamic search. These epics are the ONLY instruments traded.

_EPIC_MAP: list[tuple[str, str, str, list[str]]] = [
    (
        "UB.D.DWACUS.DAILY.IP",
        "Trump Media (DWAC) Daily",
        "trump_politics",
        ["trump", "republican", "maga", "election"],
    ),
    (
        "KA.D.FEDFLN.DAILY.IP",
        "Fed Funds (30-day) Daily",
        "fed_rates",
        ["fed", "rate", "fomc", "interest", "cut"],
    ),
    (
        "KA.D.FEDGLN.DAILY.IP",
        "Fed Funds (Guaranteed) Daily",
        "fed_rates",
        ["fed", "rate", "fomc", "interest", "cut"],
    ),
    (
        "IR.D.FF.Month7.IP",
        "US Fed Funds Month7",
        "fed_rates",
        ["fed", "rate", "fomc", "interest", "cut"],
    ),
    (
        "EN.D.LCO.Month5.IP",
        "Brent Crude Month5",
        "oil_energy",
        ["oil", "opec", "energy", "barrel", "crude"],
    ),
    (
        "KA.D.BRNTLN.DAILY.IP",
        "Brent Crude Daily",
        "oil_energy",
        ["oil", "opec", "energy", "barrel", "crude"],
    ),
    (
        "KA.D.PBRTLN.DAILY.IP",
        "Brent Crude (Physical) Daily",
        "oil_energy",
        ["oil", "opec", "energy", "barrel", "crude"],
    ),
    # ── Crypto ────────────────────────────────────────────────────────────────
    (
        "CS.D.BITCOIN.TODAY.IP",
        "Bitcoin Daily",
        "crypto",
        ["bitcoin", "btc", "crypto", "cryptocurrency"],
    ),
    (
        "CS.D.ETHUSD.TODAY.IP",
        "Ethereum Daily",
        "crypto",
        ["ethereum", "eth", "ether", "crypto", "defi"],
    ),
    (
        "CS.D.XRPUSD.TODAY.IP",
        "XRP Daily",
        "crypto",
        ["xrp", "ripple", "crypto"],
    ),
]


class IGClient:
    """
    IG Group spread-bet execution client.

    API allowance is strictly protected:
    - Session tokens cached for 6 hours (only 1 auth call per 6h window).
    - Prices fetched only for the 7 hardcoded epics above.
    - If _allowance_exceeded is ever set to True the client goes silent for
      the remainder of the process lifetime — no further API calls are made.
    """

    def __init__(self) -> None:
        _demo = os.environ.get("IG_DEMO", "false").lower() not in ("false", "0", "no")
        self._base_url: str = _DEMO_BASE if _demo else _LIVE_BASE
        self._api_key: str = os.environ.get("IG_API_KEY", "")
        self._username: str = os.environ.get("IG_USERNAME", "")
        self._password: str = os.environ.get("IG_PASSWORD", "")
        self._account_id: str = os.environ.get("IG_ACCOUNT_ID", "")

        # Session tokens — populated by authenticate()
        self._cst: str = ""
        self._security_token: str = ""
        self._auth_ts: float = 0.0  # unix timestamp of last successful auth

        self._authenticated: bool = False
        self._allowance_exceeded: bool = False

        self._http = httpx.Client(
            timeout=20.0,
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "Accept": "application/json; charset=UTF-8",
                "VERSION": "2",
                "X-IG-API-KEY": self._api_key,
            },
        )

        if self._api_key and self._username and self._password:
            self.authenticate()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _session_valid(self) -> bool:
        """True if current CST/token are within the 6-hour TTL."""
        if not self._cst or not self._security_token:
            return False
        return (time.time() - self._auth_ts) < _SESSION_TTL_SECS

    def _auth_headers(self) -> dict[str, str]:
        """Return per-request headers that carry the live session tokens."""
        return {
            "X-IG-API-KEY": self._api_key,
            "CST": self._cst,
            "X-SECURITY-TOKEN": self._security_token,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
        }

    # ── Public interface ───────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """
        Obtain a CST and X-SECURITY-TOKEN from IG.

        Skips the network call and returns True immediately if the current
        session is still within the 6-hour TTL — preserving the daily
        API allowance.

        Returns True on success, False on any error.
        """
        if self._session_valid():
            logger.debug("IG session still valid — skipping re-auth.")
            return True

        if not self._api_key or not self._username or not self._password:
            logger.error(
                "IG credentials incomplete (IG_API_KEY / IG_USERNAME / IG_PASSWORD)."
            )
            return False

        payload = {
            "identifier": self._username,
            "password": self._password,
        }
        if self._account_id:
            payload["encryptedPassword"] = False  # plain-text password flow

        try:
            resp = self._http.post(
                f"{self._base_url}/session",
                json=payload,
                headers={
                    "X-IG-API-KEY": self._api_key,
                    "VERSION": "2",
                    "Content-Type": "application/json; charset=UTF-8",
                    "Accept": "application/json; charset=UTF-8",
                },
            )
        except httpx.RequestError as exc:
            logger.error("IG auth network error: %s", exc)
            return False

        if resp.status_code == 200:
            self._cst = resp.headers.get("CST", "")
            self._security_token = resp.headers.get("X-SECURITY-TOKEN", "")
            self._auth_ts = time.time()
            self._authenticated = bool(self._cst and self._security_token)
            if self._authenticated:
                logger.info("IG authenticated successfully (demo=%s).", self._base_url == _DEMO_BASE)
            else:
                logger.error("IG auth response 200 but no CST/X-SECURITY-TOKEN in headers.")
            return self._authenticated

        # Handle rate-limit / allowance exhaustion at auth stage
        body_text = resp.text or ""
        if resp.status_code == 403 and "exceeded" in body_text.lower():
            logger.error("IG API allowance exceeded during authentication.")
            self._allowance_exceeded = True
            return False

        logger.error(
            "IG auth failed: HTTP %s — %s", resp.status_code, body_text[:200]
        )
        return False

    def get_prices(self, epic: str) -> Optional[dict]:
        """
        Fetch current bid/offer/mid for a single epic.

        Returns {"bid": float, "offer": float, "mid": float} or None.

        Allowance policy:
        - Sets _allowance_exceeded=True ONLY when the response body contains
          the word "exceeded" (IG's standard allowance error message).
        - Instrument-access 403s (e.g. market not available in this account
          tier) are logged as warnings and return None — they do NOT consume
          the allowance flag, since the session itself is still healthy.
        """
        if self._allowance_exceeded:
            logger.debug("IG allowance exceeded — skipping get_prices for %s.", epic)
            return None

        if not self._session_valid():
            logger.warning("IG session expired before get_prices — re-authenticating.")
            if not self.authenticate():
                return None

        try:
            resp = self._http.get(
                f"{self._base_url}/markets/{epic}",
                headers=self._auth_headers(),
            )
        except httpx.RequestError as exc:
            logger.warning("IG get_prices network error for %s: %s", epic, exc)
            return None

        body_text = resp.text or ""

        # Allowance exhaustion — must check body content, not just status code
        if "exceeded" in body_text.lower():
            logger.error(
                "IG API daily allowance exceeded (detected in response body for %s).", epic
            )
            self._allowance_exceeded = True
            return None

        # Instrument-access 403 — the market is inaccessible but the session is fine
        if resp.status_code == 403:
            logger.warning(
                "IG 403 for epic %s (instrument access denied, not allowance). "
                "Skipping — session remains valid.",
                epic,
            )
            return None

        if resp.status_code != 200:
            logger.warning(
                "IG get_prices unexpected HTTP %s for %s: %s",
                resp.status_code, epic, body_text[:200],
            )
            return None

        try:
            data = resp.json()
            snapshot = data.get("snapshot") or {}
            bid = snapshot.get("bid")
            offer = snapshot.get("offer")
            if bid is None or offer is None:
                logger.warning("IG snapshot missing bid/offer for %s: %s", epic, snapshot)
                return None
            bid = float(bid)
            offer = float(offer)
            mid = (bid + offer) / 2.0
            logger.debug("IG prices %s: bid=%.4f offer=%.4f mid=%.4f", epic, bid, offer, mid)
            return {"bid": bid, "offer": offer, "mid": mid}
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("IG price parse error for %s: %s", epic, exc)
            return None

    def place_order(
        self, epic: str, direction: str, size: float
    ) -> Optional[dict]:
        """
        Place a market spread-bet order on IG.

        Args:
            epic:      IG epic identifier (must be in _EPIC_MAP).
            direction: "BUY" or "SELL".
            size:      Contract size in GBP per point.

        Returns deal reference dict from IG on success, None on failure.
        In DRY_RUN mode logs the intended order and returns a simulated dict.
        """
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            logger.error("IG place_order: invalid direction '%s'. Must be BUY or SELL.", direction)
            return None

        if self._allowance_exceeded:
            logger.warning("IG allowance exceeded — cannot place order for %s.", epic)
            return None

        if DRY_RUN:
            logger.info(
                "[DRY_RUN] IG would place: epic=%s direction=%s size=%.2f",
                epic, direction, size,
            )
            return {
                "dry_run": True,
                "epic": epic,
                "direction": direction,
                "size": size,
                "dealReference": f"DRYRUN_{epic}_{direction}_{int(time.time())}",
            }

        if not self._session_valid():
            logger.warning("IG session expired before place_order — re-authenticating.")
            if not self.authenticate():
                return None

        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "orderType": "MARKET",
            "currencyCode": "GBP",
            "expiry": "-",
            "forceOpen": False,
            "guaranteedStop": False,
            "limitLevel": None,
            "stopLevel": None,
        }
        if self._account_id:
            payload["accountId"] = self._account_id

        try:
            resp = self._http.post(
                f"{self._base_url}/positions/otc",
                json=payload,
                headers={**self._auth_headers(), "VERSION": "2"},
            )
        except httpx.RequestError as exc:
            logger.error("IG place_order network error for %s: %s", epic, exc)
            return None

        body_text = resp.text or ""

        if "exceeded" in body_text.lower():
            logger.error("IG API allowance exceeded during place_order for %s.", epic)
            self._allowance_exceeded = True
            return None

        if resp.status_code in (200, 201):
            try:
                result = resp.json()
                deal_ref = result.get("dealReference", "")
                logger.info(
                    "IG order placed: epic=%s direction=%s size=%.2f dealRef=%s",
                    epic, direction, size, deal_ref,
                )
                return result
            except ValueError as exc:
                logger.error("IG place_order parse error: %s — body: %s", exc, body_text[:200])
                return None

        logger.error(
            "IG place_order failed: HTTP %s for %s — %s",
            resp.status_code, epic, body_text[:200],
        )
        return None

    def scan_and_execute(self, signals: list[dict]) -> list[dict]:
        """
        Match incoming agent signals against the hardcoded epic map, fetch
        prices for matched epics, and place orders where edge >= 3%.

        Allowance protection:
        - If _allowance_exceeded is True on entry, returns [] immediately.
        - After each get_prices() call, breaks out of the price loop if
          _allowance_exceeded has become True.
        - At most 7 get_prices() calls per invocation (one per epic).

        Args:
            signals: List of signal dicts. Each dict should contain at least:
                     {"keywords": list[str], "direction": str, "size": float,
                      "confidence": float}  where confidence is 0.0–1.0.

        Returns:
            List of executed signal dicts (with added "ig_result" key).
        """
        if self._allowance_exceeded:
            logger.warning("IG allowance exceeded — scan_and_execute skipped entirely.")
            return []

        if not signals:
            return []

        if not self.authenticate():
            logger.error("IG authentication failed — aborting scan_and_execute.")
            return []

        # Build a normalised keyword set for each signal
        # Support both TradeSignal dataclass and plain dict
        normalised_signals: list[dict] = []
        for sig in signals:
            if hasattr(sig, "__dataclass_fields__"):
                import dataclasses
                sig = dataclasses.asdict(sig)
            kws = {str(k).lower() for k in sig.get("keywords", [])}
            # Also pull keywords from question text for matching
            question = sig.get("question", "")
            kws.update(w.lower() for w in question.split() if len(w) > 3)
            normalised_signals.append({**sig, "_kws": kws})

        # Determine which epics match any incoming signal
        # Preserve order of _EPIC_MAP so we always iterate the same 7 epics.
        matched_epics: list[tuple[str, str, str, list[str]]] = []
        seen_epics: set[str] = set()
        for epic_tuple in _EPIC_MAP:
            epic, label, category, epic_keywords = epic_tuple
            if epic in seen_epics:
                continue
            epic_kw_set = set(epic_keywords)
            for sig in normalised_signals:
                if sig["_kws"] & epic_kw_set:
                    matched_epics.append(epic_tuple)
                    seen_epics.add(epic)
                    break  # One match is enough to include this epic

        if not matched_epics:
            logger.debug("IG scan: no epics matched signal keywords.")
            return []

        logger.info(
            "IG scan: %d signal(s), %d epic(s) matched — fetching prices.",
            len(signals), len(matched_epics),
        )

        # Fetch prices for matched epics (capped at 7 by _EPIC_MAP size)
        prices: dict[str, dict] = {}
        for epic_tuple in matched_epics:
            if self._allowance_exceeded:
                logger.warning(
                    "IG allowance exceeded mid-scan — stopping price fetch early."
                )
                break
            epic = epic_tuple[0]
            price_data = self.get_prices(epic)
            if price_data is not None:
                prices[epic] = price_data

        if not prices:
            logger.info("IG scan: no valid prices retrieved.")
            return []

        # Evaluate each signal against each matching epic
        executed: list[dict] = []
        for sig in normalised_signals:
            sig_kws = sig["_kws"]
            direction: str = str(sig.get("direction", "BUY")).upper()
            size: float = float(sig.get("size", 1.0))
            confidence: float = float(sig.get("confidence", 0.0))

            for epic_tuple in matched_epics:
                if self._allowance_exceeded:
                    break

                epic, label, category, epic_keywords = epic_tuple
                if not (sig_kws & set(epic_keywords)):
                    continue  # This epic doesn't match this signal

                price_data = prices.get(epic)
                if price_data is None:
                    continue

                bid = price_data["bid"]
                offer = price_data["offer"]
                spread = offer - bid
                mid = price_data["mid"]

                # Edge calculation: spread as a fraction of mid price
                edge = spread / mid if mid != 0 else 0.0

                if edge < _MIN_EDGE:
                    logger.debug(
                        "IG skip %s: edge=%.4f < min_edge=%.4f (bid=%.4f offer=%.4f)",
                        epic, edge, _MIN_EDGE, bid, offer,
                    )
                    continue

                logger.info(
                    "IG edge found: %s (%s) edge=%.4f conf=%.2f — placing %s x %.2f",
                    epic, label, edge, confidence, direction, size,
                )

                result = self.place_order(epic, direction, size)
                executed_sig = {
                    **sig,
                    "ig_epic": epic,
                    "ig_label": label,
                    "ig_category": category,
                    "ig_bid": bid,
                    "ig_offer": offer,
                    "ig_mid": mid,
                    "ig_edge": edge,
                    "ig_direction": direction,
                    "ig_size": size,
                    "ig_result": result,
                }
                # Remove the internal keyword set before returning
                executed_sig.pop("_kws", None)
                executed.append(executed_sig)

        logger.info(
            "IG scan complete: %d order(s) executed (dry_run=%s).",
            len(executed), DRY_RUN,
        )
        return executed

    def close(self) -> None:
        """Close the underlying HTTP session and invalidate IG credentials."""
        try:
            if self._session_valid():
                # Best-effort DELETE to release the server-side session token
                self._http.delete(
                    f"{self._base_url}/session",
                    headers=self._auth_headers(),
                )
        except Exception as exc:
            logger.debug("IG session logout error (non-fatal): %s", exc)
        finally:
            self._cst = ""
            self._security_token = ""
            self._authenticated = False
            self._auth_ts = 0.0
            self._http.close()
            logger.debug("IG HTTP client closed.")
