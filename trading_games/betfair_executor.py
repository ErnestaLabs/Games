"""
BetfairExecutor — Betfair Exchange execution venue for The Trading Games.

Authenticates to Betfair Exchange using cert-based login (primary) with
interactive login as fallback. Scans politics and specials markets for
edges against Polymarket implied probabilities, then places minimum-stake
BACK/LAY orders when edge >= 4% and DRY_RUN is False.

Betfair terminology:
    BACK = buy the outcome (equivalent to YES on Polymarket)
    LAY  = sell the outcome (equivalent to NO on Polymarket)

UK legal note: Betfair Exchange is a spread-bet licensed venue and is
fully legal for UK residents, replacing Smarkets which was geoblocked.

Event type IDs scanned:
    2378961 — Politics
    10      — Specials / novelty markets
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

from trading_games.config import DRY_RUN

logger = logging.getLogger(__name__)

# ── Betfair API constants ─────────────────────────────────────────────────────

_CERT_LOGIN_URL  = "https://identitysso-cert.betfair.com/api/certlogin"
_INTERACTIVE_URL = "https://identitysso.betfair.com/api/login"
_API_ENDPOINT    = "https://api.betfair.com/exchange/betting/json-rpc/v1"

# Minimum permitted stake on Betfair Exchange (GBP)
_MIN_STAKE_GBP = 2.0

# Edge threshold — only act if edge >= this value
_MIN_EDGE = 0.04  # 4%

# Market catalogue TTL in seconds (10 minutes)
_CATALOGUE_TTL = 600

# Event type IDs to scan
_POLITICS_EVENT_TYPE = "2378961"
_SPECIALS_EVENT_TYPE = "10"
_GOLF_EVENT_TYPE     = "3"    # Golf — Masters, majors, tour events

# Maximum markets to retrieve per catalogue call
_MAX_RESULTS = 100

# Max decimal odds for LAY orders — caps liability to MIN_STAKE * (MAX_LAY_ODDS-1)
# With £20 balance: 10.0 → max liability £18 per order
_MAX_LAY_ODDS = float(os.environ.get("MAX_LAY_ODDS", "10.0"))


def _resolve_cert_paths() -> tuple[str, str]:
    """
    Return (cert_path, key_path) for Betfair TLS client auth.

    Resolution order:
    1. BETFAIR_CERT_B64 + BETFAIR_KEY_B64 env vars — base64-encoded cert/key,
       decoded to temp files. Used on Railway/Docker where no filesystem certs exist.
    2. BETFAIR_CERT_PATH env var — explicit file path.
    3. Canonical local path: ../../../Forage_Landing/betfair_client.{crt,key}
    """
    import base64, tempfile

    # Strip all whitespace — Railway multi-line env vars embed spaces/newlines
    cert_b64 = "".join(os.environ.get("BETFAIR_CERT_B64", "").split())
    key_b64  = "".join(os.environ.get("BETFAIR_KEY_B64",  "").split())
    logger.info("Betfair cert resolution: BETFAIR_CERT_B64=%d chars, BETFAIR_KEY_B64=%d chars",
                len(cert_b64), len(key_b64))
    if cert_b64 and key_b64:
        tmp = Path(tempfile.gettempdir())
        cert_path = tmp / "betfair_client.crt"
        key_path  = tmp / "betfair_client.key"
        try:
            cert_path.write_bytes(base64.b64decode(cert_b64))
            key_path.write_bytes(base64.b64decode(key_b64))
            logger.info("Betfair cert decoded from env vars → %s", tmp)
            return str(cert_path), str(key_path)
        except Exception as exc:
            logger.warning(
                "Betfair base64 decode failed (BETFAIR_CERT_B64=%d chars, BETFAIR_KEY_B64=%d chars): %s — "
                "falling through to file path resolution",
                len(cert_b64), len(key_b64), exc,
            )

    env_cert = os.environ.get("BETFAIR_CERT_PATH", "")
    if env_cert:
        cert_path = Path(env_cert)
        key_path  = cert_path.with_suffix(".key")
        return str(cert_path), str(key_path)

    # Canonical path: same repo parent → Forage_Landing
    base = Path(__file__).parent.parent.parent / "Forage_Landing"
    return (
        str(base / "betfair_client.crt"),
        str(base / "betfair_client.key"),
    )


class BetfairExecutor:
    """
    Betfair Exchange executor for The Trading Games.

    Public interface consumed by agent_runner.py:
        executor._authenticated  — bool, checked before delegating signals
        executor.scan_and_execute(signals) → list[dict]
        executor.close()
    """

    def __init__(self) -> None:
        self._session_token: Optional[str]  = None
        self._authenticated: bool           = False
        self._app_key: str                  = os.environ.get("BETFAIR_APP_KEY", "")
        self._username: str                 = os.environ.get("BETFAIR_USERNAME", "")
        self._password: str                 = os.environ.get("BETFAIR_PASSWORD", "")

        # Market catalogue cache: list[dict] + timestamp
        self._catalogue_cache: list[dict]   = []
        self._catalogue_ts: float           = 0.0

        cert_path, key_path = _resolve_cert_paths()
        self._cert_path = cert_path
        self._key_path  = key_path

        # httpx client reused across calls (no cert attached here — cert only
        # needed for the initial certlogin call, which uses its own short-lived
        # client).
        self._http = httpx.Client(timeout=20.0)

        self._authenticate()

    # ── Authentication ────────────────────────────────────────────────────────

    def _authenticate(self) -> None:
        """Try cert-based login; fall back to interactive login on failure."""
        if not self._username or not self._password:
            logger.error(
                "BETFAIR_USERNAME / BETFAIR_PASSWORD not set — "
                "BetfairExecutor will not authenticate"
            )
            return

        if not self._app_key:
            logger.error("BETFAIR_APP_KEY not set — BetfairExecutor will not authenticate")
            return

        # 1. Cert-based login (preferred — no 2FA prompt)
        if self._cert_login():
            logger.info("Betfair: authenticated via cert-based login")
            self._authenticated = True
            return

        # 2. Interactive login fallback
        logger.warning("Betfair: cert login failed — attempting interactive login")
        if self._interactive_login():
            logger.info("Betfair: authenticated via interactive login")
            self._authenticated = True
            return

        logger.error("Betfair: all authentication methods failed")

    def _cert_login(self) -> bool:
        """
        POST to certlogin endpoint with TLS client certificate.
        Returns True on success and stores session token.
        """
        cert_file = self._cert_path
        key_file  = self._key_path

        if not Path(cert_file).exists() or not Path(key_file).exists():
            logger.warning(
                "Betfair cert files not found at %s / %s — skipping cert login",
                cert_file, key_file,
            )
            return False

        try:
            with httpx.Client(cert=(cert_file, key_file), timeout=20.0) as cert_client:
                resp = cert_client.post(
                    _CERT_LOGIN_URL,
                    data={
                        "username": self._username,
                        "password": self._password,
                    },
                    headers={
                        "X-Application": self._app_key,
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )

            if resp.status_code != 200:
                logger.warning(
                    "Betfair cert login HTTP %d: %s", resp.status_code, resp.text[:200]
                )
                return False

            body = resp.json()
            status = body.get("loginStatus") or body.get("status", "")
            token  = body.get("sessionToken") or body.get("token", "")

            if status == "SUCCESS" and token:
                self._session_token = token
                return True

            logger.warning("Betfair cert login status=%s (expected SUCCESS)", status)
            return False

        except Exception as exc:
            logger.warning("Betfair cert login exception: %s", exc)
            return False

    def _interactive_login(self) -> bool:
        """
        POST to interactive login endpoint (no client cert required).
        Returns True on success and stores session token.
        """
        try:
            resp = self._http.post(
                _INTERACTIVE_URL,
                data={
                    "username": self._username,
                    "password": self._password,
                },
                headers={
                    "X-Application": self._app_key,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )

            if resp.status_code != 200:
                logger.warning(
                    "Betfair interactive login HTTP %d: %s",
                    resp.status_code, resp.text[:200],
                )
                return False

            body   = resp.json()
            status = body.get("loginStatus") or body.get("status", "")
            token  = body.get("sessionToken") or body.get("token", "")

            if status == "SUCCESS" and token:
                self._session_token = token
                return True

            logger.warning(
                "Betfair interactive login status=%s (expected SUCCESS)", status
            )
            return False

        except Exception as exc:
            logger.warning("Betfair interactive login exception: %s", exc)
            return False

    # ── JSON-RPC helper ───────────────────────────────────────────────────────

    def _rpc(self, method: str, params: dict) -> Optional[dict]:
        """
        Execute a single Betfair Exchange JSON-RPC call.
        Returns the 'result' portion of the response, or None on error.
        """
        if not self._session_token:
            logger.error("Betfair: cannot make RPC call — no session token")
            return None

        payload = {
            "jsonrpc": "2.0",
            "method": f"SportsAPING/v1.0/{method}",
            "params": params,
            "id": 1,
        }

        try:
            resp = self._http.post(
                _API_ENDPOINT,
                json=payload,
                headers={
                    "X-Authentication": self._session_token,
                    "X-Application": self._app_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            if resp.status_code != 200:
                logger.warning(
                    "Betfair RPC %s HTTP %d: %s",
                    method, resp.status_code, resp.text[:300],
                )
                return None

            body = resp.json()

            # Session may have expired
            if "error" in body:
                rpc_error = body["error"]
                error_code = (
                    rpc_error.get("data", {}).get("APINGException", {}).get("errorCode", "")
                    if isinstance(rpc_error.get("data"), dict)
                    else ""
                )
                if error_code in ("NO_SESSION", "INVALID_SESSION_INFORMATION"):
                    logger.warning("Betfair session expired — re-authenticating")
                    self._session_token = None
                    self._authenticated = False
                    self._authenticate()
                    if self._authenticated:
                        return self._rpc(method, params)  # retry once
                logger.warning("Betfair RPC error: %s", json.dumps(rpc_error)[:300])
                return None

            return body.get("result")

        except Exception as exc:
            logger.error("Betfair RPC exception [%s]: %s", method, exc)
            return None

    # ── Market catalogue ──────────────────────────────────────────────────────

    def _get_market_catalogue(self) -> list[dict]:
        """
        Retrieve Betfair market catalogue for politics and specials events.
        Results are cached for _CATALOGUE_TTL seconds to avoid hammering the API.
        """
        now = time.monotonic()
        if self._catalogue_cache and (now - self._catalogue_ts) < _CATALOGUE_TTL:
            return self._catalogue_cache

        result = self._rpc(
            "listMarketCatalogue",
            {
                "filter": {
                    "eventTypeIds": [_POLITICS_EVENT_TYPE, _SPECIALS_EVENT_TYPE, _GOLF_EVENT_TYPE],
                    "marketTypeCodes": ["MATCH_ODDS", "OUTRIGHT_WINNER", "NEXT_PRESIDENT",
                                        "TOURNAMENT_WINNER", "WINNER", "EACH_WAY",
                                        "PLACE", "TOP_FINISHER"],
                    "inPlayOnly": False,
                },
                "marketProjection": [
                    "EVENT",
                    "MARKET_START_TIME",
                    "RUNNER_DESCRIPTION",
                    "RUNNER_METADATA",
                ],
                "sort": "LAST_TO_START",
                "maxResults": str(_MAX_RESULTS),
                "locale": "en",
            },
        )

        if result and isinstance(result, list):
            self._catalogue_cache = result
            self._catalogue_ts    = now
            logger.debug("Betfair: catalogue refreshed — %d markets", len(result))
        else:
            logger.warning("Betfair: listMarketCatalogue returned no results")

        return self._catalogue_cache

    # ── Best available odds ───────────────────────────────────────────────────

    def _get_best_odds(self, market_id: str) -> list[dict]:
        """
        Fetch best available back/lay odds for all runners in a market.
        Returns list of dicts:
            {runner_id, runner_name, best_back_price, best_lay_price}
        """
        result = self._rpc(
            "listMarketBook",
            {
                "marketIds": [market_id],
                "priceProjection": {
                    "priceData": ["EX_BEST_OFFERS"],
                    "exBestOffersOverrides": {
                        "bestPricesDepth": 1,
                        "rollupModel": "STAKE",
                        "rollupLimit": _MIN_STAKE_GBP,
                    },
                },
                "orderProjection": "EXECUTABLE",
                "matchProjection": "NO_ROLLUP",
            },
        )

        runners: list[dict] = []
        if not result or not isinstance(result, list) or not result:
            return runners

        for book in result:
            for runner in book.get("runners", []):
                ex = runner.get("ex", {})
                back_prices = ex.get("availableToBack", [])
                lay_prices  = ex.get("availableToLay",  [])

                best_back = float(back_prices[0]["price"]) if back_prices else None
                best_lay  = float(lay_prices[0]["price"])  if lay_prices  else None

                runners.append({
                    "runner_id":        runner.get("selectionId"),
                    "runner_name":      runner.get("runnerName", ""),
                    "best_back_price":  best_back,
                    "best_lay_price":   best_lay,
                    "status":           runner.get("status", ""),
                })

        return runners

    # ── Signal matching ───────────────────────────────────────────────────────

    def _match_signal_to_runner(
        self, signal: dict, catalogue_entry: dict
    ) -> Optional[dict]:
        """
        Attempt to match a Polymarket signal to a Betfair runner by keyword
        overlap between the signal question and the market/event name.

        Returns matched runner dict (with market_id injected) or None.
        """
        question    = signal.get("question", "").lower()
        event_name  = (
            catalogue_entry.get("event", {}).get("name", "")
            + " "
            + catalogue_entry.get("marketName", "")
        ).lower()

        # Require at least 1 meaningful word to overlap (player surnames, event names, etc.)
        q_words = {w for w in question.split() if len(w) > 4}
        e_words = {w for w in event_name.split() if len(w) > 4}
        overlap = q_words & e_words
        if len(overlap) < 1:
            return None

        market_id = catalogue_entry.get("marketId", "")

        # Match runner by name using catalogue's runner list (has runnerName)
        # listMarketBook does NOT return runnerName, so we must use catalogue data
        catalogue_runners = catalogue_entry.get("runners", [])
        best_runner_id: Optional[int] = None
        best_runner_name = ""
        best_name_overlap = 0

        for cr in catalogue_runners:
            r_name = (cr.get("runnerName") or "").lower()
            r_words = {w for w in r_name.split() if len(w) > 3}
            r_overlap = len(q_words & r_words)
            if r_overlap > best_name_overlap:
                best_name_overlap = r_overlap
                best_runner_id = cr.get("selectionId")
                best_runner_name = cr.get("runnerName", "")

        # Fetch live odds
        runners = self._get_best_odds(market_id)
        if not runners:
            return None

        # Find the matched runner in the odds data
        best_runner: Optional[dict] = None
        if best_runner_id:
            for runner in runners:
                if runner.get("runner_id") == best_runner_id:
                    best_runner = runner
                    break

        if not best_runner:
            # Fallback: first ACTIVE runner (two-runner / binary markets)
            active = [r for r in runners if r.get("status", "ACTIVE") == "ACTIVE"]
            best_runner = active[0] if active else (runners[0] if runners else None)

        if best_runner:
            best_runner = dict(best_runner)  # copy to avoid mutating cache
            best_runner["runner_name"] = best_runner_name or best_runner.get("runner_name", "")
            best_runner["market_id"] = market_id
        return best_runner

    # ── Edge calculation ──────────────────────────────────────────────────────

    @staticmethod
    def _decimal_to_prob(decimal_odds: float) -> float:
        """Convert decimal odds (e.g. 2.5) to implied probability (e.g. 0.40)."""
        if decimal_odds <= 1.0:
            return 1.0
        return 1.0 / decimal_odds

    @staticmethod
    def _edge(poly_price: float, betfair_prob: float, side: str) -> float:
        """
        Calculate edge between Polymarket price and Betfair implied probability.

        For a BACK (YES) signal: edge = betfair_prob_back - poly_price
            Positive means Betfair thinks it more likely than Polymarket.
        For a LAY (NO) signal:  edge = poly_price - betfair_prob_lay
            Positive means Polymarket overpays; lay on Betfair is profitable.
        """
        if side == "YES":
            return betfair_prob - poly_price
        # NO → LAY
        return poly_price - betfair_prob

    # ── Order placement ───────────────────────────────────────────────────────

    def _place_order(
        self,
        market_id:   str,
        selection_id: int,
        side:        str,  # "BACK" or "LAY"
        price:       float,
        size_gbp:    float = _MIN_STAKE_GBP,
    ) -> Optional[dict]:
        """
        Place a single limit order on Betfair Exchange.
        Returns the instruction report dict or None on failure.
        """
        instruction = {
            "selectionId": selection_id,
            "handicap":    "0",
            "side":        side,
            "orderType":   "LIMIT",
            "limitOrder": {
                "size":            f"{size_gbp:.2f}",
                "price":           f"{price:.2f}",
                "persistenceType": "LAPSE",  # cancel if unmatched at event start
            },
        }

        result = self._rpc(
            "placeOrders",
            {
                "marketId":             market_id,
                "instructions":         [instruction],
                "customerRef":          f"ttg_{market_id.replace('.','')[-6:]}_{selection_id % 100000}",
                "marketVersion":        None,
                "customerStrategyRef":  "trading_games",
                "async":                False,
            },
        )

        if not result:
            return None

        status = result.get("status", "")
        if status != "SUCCESS":
            logger.warning(
                "Betfair placeOrders non-success: status=%s errorCode=%s",
                status, result.get("errorCode", ""),
            )
            return None

        reports = result.get("instructionReports", [])
        if reports:
            report = reports[0]
            if report.get("status") == "SUCCESS":
                logger.info(
                    "Betfair order placed: market=%s sel=%s side=%s price=%.2f size=£%.2f | "
                    "betId=%s",
                    market_id, selection_id, side, price, size_gbp,
                    report.get("betId", "N/A"),
                )
                return report
            else:
                logger.warning(
                    "Betfair instruction failed: %s — %s",
                    report.get("status"),
                    report.get("errorCode", ""),
                )
        return None

    # ── Public interface ──────────────────────────────────────────────────────

    def scan_and_execute(self, signals: list[dict]) -> list[dict]:
        """
        Match each signal against Betfair markets, calculate edge, and
        optionally place orders.

        Parameters
        ----------
        signals : list[dict]
            Each dict must contain at least:
                market_id    : str  — Polymarket market identifier
                question     : str  — human-readable question text
                side         : str  — "YES" or "NO"
                market_price : float — Polymarket implied probability (0–1)
                edge         : float — pre-calculated Polymarket edge

        Returns
        -------
        list[dict]
            Subset of signals that were acted on, each enriched with:
                betfair_market_id, betfair_runner_id, betfair_side,
                betfair_price, betfair_implied_prob, betfair_edge,
                executed (bool), dry_run (bool)
        """
        if not self._authenticated:
            logger.warning("Betfair: not authenticated — skipping scan_and_execute")
            return []

        catalogue = self._get_market_catalogue()
        if not catalogue:
            logger.warning("Betfair: empty market catalogue — nothing to match")
            return []

        logger.info("Betfair: %d catalogue entries | %d signals to match",
                    len(catalogue), len(signals))
        for e in catalogue[:5]:
            logger.info("  Betfair market: '%s' / '%s'",
                        e.get("event", {}).get("name", "?"),
                        e.get("marketName", "?"))

        acted: list[dict] = []

        for _sig in signals:
            # Support both TradeSignal dataclass and plain dict
            if hasattr(_sig, "__dataclass_fields__"):
                import dataclasses
                signal = dataclasses.asdict(_sig)
            else:
                signal = _sig

            poly_price = float(signal.get("market_price", 0.5))
            poly_side  = signal.get("side", "YES")  # "YES" or "NO"
            question   = signal.get("question", "")

            matched = False
            # Try each catalogue entry for a match
            for entry in catalogue:
                runner = self._match_signal_to_runner(signal, entry)
                if not runner:
                    continue
                matched = True

                # Choose price direction
                if poly_side == "YES":
                    # BACK: we want good BACK price (high decimal = low implied prob
                    # = Betfair undervalues outcome = we have edge if betfair_prob <
                    # poly_price, but we want to back when betfair overvalues it).
                    # Edge: betfair_back_prob > poly_price → market underestimates
                    # the event; back at the offered price.
                    raw_price = runner.get("best_back_price")
                    bf_side   = "BACK"
                else:
                    # LAY: we want good LAY price (low decimal = high implied prob
                    # = Betfair overvalues the outcome; lay it and profit if it loses)
                    raw_price = runner.get("best_lay_price")
                    bf_side   = "LAY"

                if not raw_price or raw_price <= 1.0:
                    logger.debug(
                        "Betfair: no valid %s price for runner %s — skipping",
                        bf_side, runner.get("runner_name"),
                    )
                    continue

                # Cap LAY liability: skip if odds exceed max (protects small balance)
                if bf_side == "LAY" and raw_price > _MAX_LAY_ODDS:
                    logger.info(
                        "Betfair: LAY odds %.1f > MAX_LAY_ODDS %.1f — skipping %s",
                        raw_price, _MAX_LAY_ODDS, runner.get("runner_name", ""),
                    )
                    break

                bf_prob = self._decimal_to_prob(raw_price)
                edge    = self._edge(poly_price, bf_prob, poly_side)

                logger.info(
                    "Betfair signal match: question='%s...' | runner=%s | "
                    "side=%s | bf_price=%.2f | bf_prob=%.3f | poly_price=%.3f | edge=%.3f",
                    signal.get("question", "")[:60],
                    runner.get("runner_name", ""),
                    bf_side, raw_price, bf_prob, poly_price, edge,
                )

                if edge < _MIN_EDGE:
                    logger.debug(
                        "Betfair: edge %.3f below threshold %.3f — skipping",
                        edge, _MIN_EDGE,
                    )
                    # Still break — we found the right market, just no edge
                    break

                # Sufficient edge found
                enriched = dict(signal)
                enriched.update({
                    "betfair_market_id":    runner["market_id"],
                    "betfair_runner_id":    runner.get("runner_id"),
                    "betfair_runner_name":  runner.get("runner_name", ""),
                    "betfair_side":         bf_side,
                    "betfair_price":        raw_price,
                    "betfair_implied_prob": bf_prob,
                    "betfair_edge":         edge,
                    "dry_run":              DRY_RUN,
                    "executed":             False,
                })

                if DRY_RUN:
                    logger.info(
                        "[DRY_RUN] Would place Betfair %s on '%s' @ %.2f "
                        "(edge=%.3f, min_stake=£%.2f)",
                        bf_side,
                        runner.get("runner_name", ""),
                        raw_price,
                        edge,
                        _MIN_STAKE_GBP,
                    )
                    enriched["executed"] = False
                else:
                    report = self._place_order(
                        market_id    = runner["market_id"],
                        selection_id = runner["runner_id"],
                        side         = bf_side,
                        price        = raw_price,
                        size_gbp     = _MIN_STAKE_GBP,
                    )
                    enriched["executed"]        = report is not None
                    enriched["betfair_bet_id"]  = (
                        report.get("betId") if report else None
                    )

                acted.append(enriched)
                break  # One match per signal — move on to next signal

            if not matched:
                logger.info(
                    "Betfair: no catalogue match for '%s...' (checked %d markets)",
                    question[:80], len(catalogue),
                )

        logger.info(
            "Betfair scan_and_execute complete: %d/%d signals matched/acted on",
            len(acted), len(signals),
        )
        return acted

    def close(self) -> None:
        """Log out of the current Betfair session and close the HTTP client."""
        if self._session_token:
            try:
                self._http.get(
                    "https://identitysso.betfair.com/api/logout",
                    headers={
                        "X-Authentication": self._session_token,
                        "X-Application":    self._app_key,
                        "Accept":           "application/json",
                    },
                )
                logger.info("Betfair: session logged out")
            except Exception as exc:
                logger.debug("Betfair logout exception (non-critical): %s", exc)
            finally:
                self._session_token = None
                self._authenticated = False

        try:
            self._http.close()
        except Exception:
            pass
