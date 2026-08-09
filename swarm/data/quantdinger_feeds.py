"""
quantdinger_feeds.py — KRATOS v2
==================================
Market data + sentiment feeds ported from QuantDinger v3.0.1.
Provides multi-source forex quotes, VIX, Fear&Greed, and a
CircuitBreaker for each provider.

Sources (priority order per pair):
  1. TwelveData   (API key: TWELVE_DATA_API_KEY)
  2. yfinance     (no key needed)
  3. Tiingo       (API key: TIINGO_API_KEY)

Sentiment:
  - Fear & Greed Index (alternative.me)
  - VIX (yfinance → akshare fallback)
  - DXY proxy (UUP ETF via yfinance)
"""
from __future__ import annotations

import os, time, logging, requests
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Canonical pair map (QuantDinger v3.0.1) ────────────────────────────────
FOREX_PAIRS: List[Dict[str, str]] = [
    {"td": "EUR/USD", "yf": "EURUSD=X", "tiingo": "eurusd",  "name": "EUR/USD", "base": "EUR", "quote": "USD"},
    {"td": "GBP/USD", "yf": "GBPUSD=X", "tiingo": "gbpusd",  "name": "GBP/USD", "base": "GBP", "quote": "USD"},
    {"td": "USD/JPY", "yf": "USDJPY=X", "tiingo": "usdjpy",  "name": "USD/JPY", "base": "USD", "quote": "JPY"},
    {"td": "AUD/USD", "yf": "AUDUSD=X", "tiingo": "audusd",  "name": "AUD/USD", "base": "AUD", "quote": "USD"},
    {"td": "USD/CAD", "yf": "USDCAD=X", "tiingo": "usdcad",  "name": "USD/CAD", "base": "USD", "quote": "CAD"},
    {"td": "USD/CHF", "yf": "USDCHF=X", "tiingo": "usdchf",  "name": "USD/CHF", "base": "USD", "quote": "CHF"},
    {"td": "NZD/USD", "yf": "NZDUSD=X", "tiingo": "nzdusd",  "name": "NZD/USD", "base": "NZD", "quote": "USD"},
    {"td": "USD/CNH", "yf": "USDCNH=X", "tiingo": "usdcnh",  "name": "USD/CNH", "base": "USD", "quote": "CNH"},
]

PAIR_LOOKUP: Dict[str, Dict] = {p["name"]: p for p in FOREX_PAIRS}


# ── Circuit Breaker (QuantDinger pattern) ──────────────────────────────────
class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    CLOSED → fail N times → OPEN → cooldown → HALF_OPEN
    HALF_OPEN: success → CLOSED | fail → OPEN
    """
    def __init__(
        self,
        failure_threshold: int   = 3,
        cooldown_seconds:  float = 300.0,
        half_open_max:     int   = 1,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds  = cooldown_seconds
        self.half_open_max     = half_open_max
        self._state            = CircuitState.CLOSED
        self._failures         = 0
        self._opened_at:  Optional[float] = None
        self._half_calls       = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - (self._opened_at or 0) >= self.cooldown_seconds:
                self._state      = CircuitState.HALF_OPEN
                self._half_calls = 0
        return self._state

    def allow(self) -> bool:
        s = self.state
        if s == CircuitState.CLOSED:    return True
        if s == CircuitState.OPEN:      return False
        # HALF_OPEN: allow one probe
        self._half_calls += 1
        return self._half_calls <= self.half_open_max

    def record_success(self):
        self._failures = 0
        self._state    = CircuitState.CLOSED

    def record_failure(self):
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state     = CircuitState.OPEN
            self._opened_at = time.time()
            logger.warning("[CB] Circuit OPEN for provider (failures=%d)", self._failures)


# ── Per-provider circuit breakers ─────────────────────────────────────────
_cb: Dict[str, CircuitBreaker] = {
    "twelvedata": CircuitBreaker(),
    "yfinance":   CircuitBreaker(),
    "tiingo":     CircuitBreaker(),
}


# ── Provider helpers ───────────────────────────────────────────────────────
def _fetch_twelvedata(pair: Dict) -> Optional[Dict]:
    cb  = _cb["twelvedata"]
    if not cb.allow(): return None
    key = os.getenv("TWELVE_DATA_API_KEY","").strip()
    if not key: return None
    try:
        r = requests.get(
            "https://api.twelvedata.com/quote",
            params={"symbol": pair["td"], "apikey": key},
            timeout=10,
        )
        d = r.json()
        if d.get("status") == "error" or not d.get("close"):
            cb.record_failure(); return None
        cur  = float(d["close"])
        prev = float(d.get("previous_close") or cur)
        cb.record_success()
        return {
            "symbol":   pair["name"],
            "price":    round(cur, 5),
            "change":   round((cur - prev) / prev * 100 if prev else 0, 4),
            "source":   "twelvedata",
            "base":     pair["base"],
            "quote":    pair["quote"],
        }
    except Exception as e:
        cb.record_failure()
        logger.debug("TwelveData %s failed: %s", pair["name"], e)
        return None


def _fetch_yfinance(pair: Dict) -> Optional[Dict]:
    cb = _cb["yfinance"]
    if not cb.allow(): return None
    try:
        import yfinance as yf
        ticker = yf.Ticker(pair["yf"])
        hist   = ticker.history(period="2d")
        if hist.empty or len(hist) < 1:
            cb.record_failure(); return None
        cur  = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else cur
        cb.record_success()
        return {
            "symbol":   pair["name"],
            "price":    round(cur, 5),
            "change":   round((cur - prev) / prev * 100 if prev else 0, 4),
            "source":   "yfinance",
            "base":     pair["base"],
            "quote":    pair["quote"],
        }
    except Exception as e:
        cb.record_failure()
        logger.debug("yfinance %s failed: %s", pair["name"], e)
        return None


def _fetch_tiingo(pair: Dict) -> Optional[Dict]:
    cb  = _cb["tiingo"]
    if not cb.allow(): return None
    key = os.getenv("TIINGO_API_KEY","").strip()
    if not key: return None
    try:
        r = requests.get(
            f"https://api.tiingo.com/tiingo/fx/{pair['tiingo']}/top",
            headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
            timeout=10,
        )
        d = r.json()
        if not d: cb.record_failure(); return None
        item = d[0] if isinstance(d, list) else d
        cur  = float(item.get("midPrice") or item.get("askPrice") or 0)
        if not cur: cb.record_failure(); return None
        cb.record_success()
        return {
            "symbol":   pair["name"],
            "price":    round(cur, 5),
            "change":   0.0,
            "source":   "tiingo",
            "base":     pair["base"],
            "quote":    pair["quote"],
        }
    except Exception as e:
        cb.record_failure()
        logger.debug("Tiingo %s failed: %s", pair["name"], e)
        return None


def fetch_forex_quotes(pairs: Optional[List[str]] = None) -> List[Dict]:
    """
    Multi-source forex quote fetch with CB-protected fallback chain.
    pairs: list of 'EUR/USD' style names (default: all 8)
    """
    targets = [PAIR_LOOKUP[p] for p in (pairs or list(PAIR_LOOKUP)) if p in PAIR_LOOKUP]
    results = []
    for p in targets:
        for fetcher in (_fetch_twelvedata, _fetch_yfinance, _fetch_tiingo):
            q = fetcher(p)
            if q:
                results.append(q)
                break
        else:
            logger.warning("All providers failed for %s", p["name"])
    return results


# ── Sentiment feeds ───────────────────────────────────────────────────────
@dataclass
class SentimentSnapshot:
    fear_greed:   int   = 50
    fear_label:   str   = "Neutral"
    vix:          float = 18.0
    vix_level:    str   = "low"
    dxy:          float = 104.0
    timestamp:    float = field(default_factory=time.time)


def fetch_fear_greed() -> Dict[str, Any]:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        item = r.json()["data"][0]
        return {
            "value":          int(item["value"]),
            "classification": item["value_classification"],
            "timestamp":      int(item["timestamp"]),
            "source":         "alternative.me",
        }
    except Exception as e:
        logger.warning("Fear&Greed fetch failed: %s", e)
        return {"value": 50, "classification": "Neutral", "timestamp": 0, "source": "N/A"}


def fetch_vix() -> Dict[str, Any]:
    try:
        import yfinance as yf
        hist = yf.Ticker("^VIX").history(period="5d")
        if not hist.empty and len(hist) >= 2:
            cur  = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            chg  = round((cur - prev) / prev * 100, 2) if prev else 0
            level = "extreme" if cur > 30 else "high" if cur > 20 else "low"
            return {"value": round(cur, 2), "change": chg,
                    "level": level, "source": "yfinance"}
    except Exception as e:
        logger.warning("VIX yfinance failed: %s", e)
    return {"value": 18.0, "change": 0, "level": "low", "source": "default"}


def fetch_dxy() -> Dict[str, Any]:
    """DXY proxy via UUP ETF (PowerShares DB US Dollar Index Bullish)."""
    try:
        import yfinance as yf
        hist = yf.Ticker("UUP").history(period="5d")
        if not hist.empty:
            cur  = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else cur
            return {"value": round(cur, 4), "change": round((cur-prev)/prev*100,2),
                    "source": "yfinance_UUP"}
    except Exception as e:
        logger.warning("DXY proxy failed: %s", e)
    return {"value": 104.0, "change": 0, "source": "default"}


def fetch_sentiment_snapshot() -> SentimentSnapshot:
    fg  = fetch_fear_greed()
    vix = fetch_vix()
    dxy = fetch_dxy()
    snap = SentimentSnapshot(
        fear_greed  = fg["value"],
        fear_label  = fg["classification"],
        vix         = vix["value"],
        vix_level   = vix["level"],
        dxy         = dxy["value"],
    )
    logger.info("[SENTIMENT] FG=%d VIX=%.1f DXY=%.4f", snap.fear_greed, snap.vix, snap.dxy)
    return snap
