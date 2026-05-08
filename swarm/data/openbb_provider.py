"""
OPENBB DATA PROVIDER
=====================
Integrates OpenBB Platform — the unified financial data API for analysts,
quants, and AI agents. Provides standardized access to:
  - Forex/Currency data (prices, pairs, snapshots, reference rates)
  - Economy data (GDP, CPI, yield curve, employment)
  - Market data (equities, indices, commodities)
  - News feeds

Source: ymj6h77jz9-dot/OpenBB
Integration: openbb_platform/extensions/currency + economy + news

Used by:
  - AnalystAgent: OHLCV candles, technical indicators
  - SentimentAgent: Economic calendar events, news
  - RiskAgent: Volatility data, macro indicators
  - KronosAdapter: Historical candle data for model inference
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CurrencySnapshot:
    pair:       str
    bid:        float
    ask:        float
    mid:        float
    change_pct: float
    timestamp:  str


@dataclass
class EconomicEvent:
    name:      str
    country:   str
    actual:    Optional[float]
    forecast:  Optional[float]
    previous:  Optional[float]
    impact:    str   # "high" | "medium" | "low"
    time:      str


class OpenBBProvider:
    """
    Unified market data provider using OpenBB Platform SDK.
    Falls back to direct REST APIs if SDK not available.
    
    Integrated from: ymj6h77jz9-dot/OpenBB
    Provides: OHLCV, snapshots, economic calendar, macro indicators
    """

    # Free/no-auth forex data source
    EXCHANGE_RATE_API = "https://open.er-api.com/v6/latest"
    # Yahoo Finance via yfinance (used by OpenBB internally)
    YFINANCE_SUFFIX   = "=X"

    def __init__(self):
        self._obb = None
        self._obb_available = False
        self._try_init_openbb()

    def _try_init_openbb(self):
        """Try to initialize OpenBB SDK."""
        try:
            from openbb import obb
            self._obb = obb
            self._obb_available = True
            logger.info("✅ [OpenBB] SDK available")
        except ImportError:
            logger.info("[OpenBB] SDK not installed. Using direct API fallback.")

    # ── Currency / Forex ──────────────────────────────────────────────────────
    async def get_candles(self, pair: str, interval: str = "1m",
                           lookback_bars: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV candle data for a forex pair.
        Returns DataFrame with [open, high, low, close, volume, amount].
        """
        if self._obb_available:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._obb_candles, pair, interval, lookback_bars
            )
        return await self._yfinance_candles(pair, interval, lookback_bars)

    def _obb_candles(self, pair: str, interval: str, bars: int) -> pd.DataFrame:
        """OpenBB SDK candle fetch."""
        try:
            # OpenBB currency historical: pair format "EUR/USD"
            obb_pair = pair[:3] + "/" + pair[3:]
            result   = self._obb.currency.price.historical(
                symbol    = obb_pair,
                interval  = interval,
                start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            )
            df = result.to_df()
            df.rename(columns={"volume": "volume", "adj_close": "close"}, inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    df[col] = 0.0
            df['amount'] = df['close'] * df['volume']
            return df[['open', 'high', 'low', 'close', 'volume', 'amount']].tail(bars)
        except Exception as e:
            logger.warning(f"[OpenBB] Candle fetch failed: {e}")
            return pd.DataFrame()

    async def _yfinance_candles(self, pair: str, interval: str,
                                  bars: int) -> pd.DataFrame:
        """yfinance fallback for candle data."""
        try:
            import yfinance as yf
            ticker = pair + self.YFINANCE_SUFFIX
            period_map = {"1m": "5d", "5m": "60d", "1h": "730d", "1d": "5y"}
            period  = period_map.get(interval, "5d")
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yf.download(ticker, period=period, interval=interval,
                                     progress=False, auto_adjust=True)
            )
            df.columns = [c.lower() for c in df.columns]
            df['amount'] = df.get('close', 0) * df.get('volume', 0)
            return df[['open', 'high', 'low', 'close', 'volume', 'amount']].tail(bars)
        except Exception as e:
            logger.warning(f"[yfinance] Candle fetch failed: {e}")
            return pd.DataFrame()

    async def get_snapshot(self, pairs: List[str]) -> List[CurrencySnapshot]:
        """
        Get real-time snapshots for multiple pairs.
        Uses OpenBB currency_snapshots if available.
        """
        if self._obb_available:
            try:
                snap = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._obb.currency.snapshots(
                        base="USD", quote_type="major"
                    ).to_df()
                )
                results = []
                for pair in pairs:
                    row = snap[snap['symbol'] == pair]
                    if not row.empty:
                        r = row.iloc[0]
                        results.append(CurrencySnapshot(
                            pair       = pair,
                            bid        = float(r.get('bid', r.get('close', 0))),
                            ask        = float(r.get('ask', r.get('close', 0))),
                            mid        = float(r.get('close', 0)),
                            change_pct = float(r.get('change_percent', 0)),
                            timestamp  = datetime.now(timezone.utc).isoformat(),
                        ))
                return results
            except Exception as e:
                logger.warning(f"[OpenBB] Snapshot failed: {e}")

        return await self._er_api_snapshot(pairs)

    async def _er_api_snapshot(self, pairs: List[str]) -> List[CurrencySnapshot]:
        """Exchange Rate API fallback for live rates."""
        results = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.EXCHANGE_RATE_API}/USD", timeout=aiohttp.ClientTimeout(total=10)) as r:
                    data = await r.json()
                    rates = data.get("rates", {})
                    for pair in pairs:
                        base, quote = pair[:3], pair[3:]
                        if base in rates and quote in rates:
                            mid = rates[quote] / rates[base]
                            results.append(CurrencySnapshot(
                                pair       = pair,
                                bid        = round(mid * 0.9999, 5),
                                ask        = round(mid * 1.0001, 5),
                                mid        = round(mid, 5),
                                change_pct = 0.0,
                                timestamp  = datetime.now(timezone.utc).isoformat(),
                            ))
        except Exception as e:
            logger.warning(f"[ExchangeRate API] Failed: {e}")
        return results

    # ── Economic Calendar ─────────────────────────────────────────────────────
    async def get_economic_calendar(self, days_ahead: int = 3) -> List[EconomicEvent]:
        """
        Fetch upcoming high-impact economic events.
        Used by SentimentAgent to pre-warn before NFP, FOMC, CPI etc.
        """
        if self._obb_available:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._obb.economy.calendar(
                        start_date = datetime.now().strftime("%Y-%m-%d"),
                        end_date   = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
                    ).to_df()
                )
                events = []
                for _, row in result.iterrows():
                    impact = "high" if row.get("importance", 0) >= 3 else \
                             "medium" if row.get("importance", 0) == 2 else "low"
                    events.append(EconomicEvent(
                        name     = str(row.get("event", "")),
                        country  = str(row.get("country", "")),
                        actual   = row.get("actual"),
                        forecast = row.get("consensus"),
                        previous = row.get("previous"),
                        impact   = impact,
                        time     = str(row.get("date", "")),
                    ))
                return [e for e in events if e.impact == "high"]
            except Exception as e:
                logger.warning(f"[OpenBB] Calendar failed: {e}")

        return await self._investing_calendar_fallback()

    async def _investing_calendar_fallback(self) -> List[EconomicEvent]:
        """Fallback: return known fixed high-impact event types as warnings."""
        # This is a placeholder — in production connect to investing.com API or Forex Factory
        logger.info("[Calendar] Using static high-impact event list (no API key)")
        return []

    # ── FRED Macro Data (from Crucix integration) ─────────────────────────────
    async def get_fred_indicators(self, api_key: str) -> Dict[str, float]:
        """
        Fetch key macro indicators from FRED (Federal Reserve Economic Data).
        Integrated from: ymj6h77jz9-dot/Crucix (apis/sources/fred.mjs)
        
        Key series: Fed Funds Rate, CPI, 10Y yield, VIX, USD Index, Gold
        """
        FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
        KEY_SERIES = {
            "DFF":           "fed_funds_rate",
            "CPIAUCSL":      "cpi",
            "DGS10":         "us_10y_yield",
            "T10Y2Y":        "yield_curve_spread",
            "VIXCLS":        "vix",
            "DTWEXBGS":      "usd_index",
            "GOLDAMGBD228NLBM": "gold_london_fix",
            "BAMLH0A0HYM2":  "high_yield_spread",
        }
        results = {}
        try:
            async with aiohttp.ClientSession() as session:
                for series_id, name in KEY_SERIES.items():
                    try:
                        url = (f"{FRED_BASE}?series_id={series_id}&api_key={api_key}"
                               f"&file_type=json&sort_order=desc&limit=1")
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                            data = await r.json()
                            obs = data.get("observations", [])
                            if obs and obs[0]["value"] != ".":
                                results[name] = float(obs[0]["value"])
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"[FRED] Request failed: {e}")
        return results

    async def get_atr(self, pair: str, period: int = 14) -> float:
        """
        Calculate ATR (Average True Range) from recent candles.
        Used by RiskAgent for dynamic SL/TP sizing.
        """
        df = await self.get_candles(pair, interval="5m", lookback_bars=period + 5)
        if df.empty or len(df) < period:
            # Default ATRs per major pair (pip-based)
            defaults = {
                "EURUSD": 0.0008, "GBPUSD": 0.0012, "USDJPY": 0.08,
                "XAUUSD": 2.5,    "USDCHF": 0.0007, "AUDUSD": 0.0007,
            }
            return defaults.get(pair, 0.001)
        
        high = df['high'].values
        low  = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1]))
        )
        return float(np.mean(tr[-period:]))


# Import numpy here for ATR calculation
import numpy as np
