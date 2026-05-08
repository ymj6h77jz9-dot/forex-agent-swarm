"""
CRUCIX INTELLIGENCE LAYER
===========================
Integrates Crucix — "Your personal intelligence agent. Watches the world 
from multiple data sources and pings you when something changes."

Crucix runs a full multi-source intelligence sweep in parallel:
  Tier 1: OSINT & Geopolitical (GDELT, ACLED, ReliefWeb, OFAC, OpenSanctions)
  Tier 2: Economic & Financial (FRED, Treasury, BLS, EIA, GSCPI, Comtrade)
  Tier 3: Environment, Technology, Social

This adapter taps the Tier 2 (Economic/Financial) sources and maps them
to structured ForexIntelligence objects that feed the SentimentAgent and
the Orchestrator's macro context.

Source: ymj6h77jz9-dot/Crucix (apis/sources/fred.mjs, bls.mjs, eia.mjs)
Pattern: BRIEFING_PROMPT.md — "leverage-first intelligence note"
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)

FRED_API_KEY    = os.environ.get("FRED_API_KEY", "")
BLS_API_KEY     = os.environ.get("BLS_API_KEY", "")
NEWS_API_KEY    = os.environ.get("NEWS_API_KEY", "")


@dataclass
class MacroSignal:
    """A macro economic signal from Crucix intelligence sweep."""
    source:         str           # "FRED" | "BLS" | "EIA" | "NEWS"
    indicator:      str           # e.g. "fed_funds_rate"
    value:          Optional[float]
    previous:       Optional[float]
    change:         Optional[float]
    direction:      str           # "UP" | "DOWN" | "FLAT"
    impact_on_usd:  str           # "BULLISH_USD" | "BEARISH_USD" | "NEUTRAL"
    description:    str
    timestamp:      str           = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass 
class ForexIntelBriefing:
    """Structured intelligence briefing for the Sentiment Agent."""
    timestamp:       str
    macro_signals:   List[MacroSignal]
    news_headlines:  List[str]
    risk_regime:     str          # "RISK_ON" | "RISK_OFF" | "NEUTRAL"
    usd_bias:        str          # "BULLISH" | "BEARISH" | "NEUTRAL"
    key_levels_text: str          # Natural language summary of key levels
    briefing_text:   str          # Full Crucix-style leverage-first briefing


class CrucixIntelligence:
    """
    Crucix-inspired multi-source intelligence aggregator.
    
    Runs all data sources in parallel (asyncio.gather) and synthesizes
    into a ForexIntelBriefing that the SentimentAgent uses as context.
    
    Integrated from: ymj6h77jz9-dot/Crucix
    Pattern: apis/briefing.mjs — parallel orchestrator
    """

    FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
    NEWS_BASE = "https://newsapi.org/v2/everything"
    
    # USD impact mapping — based on Crucix BRIEFING_PROMPT.md "detect regime shifts"
    USD_IMPACT_MAP = {
        "fed_funds_rate":    {"UP": "BULLISH_USD",  "DOWN": "BEARISH_USD"},
        "cpi":               {"UP": "BULLISH_USD",  "DOWN": "BEARISH_USD"},   # Inflation → rate hike expectations
        "us_10y_yield":      {"UP": "BULLISH_USD",  "DOWN": "BEARISH_USD"},
        "vix":               {"UP": "BEARISH_USD",  "DOWN": "BULLISH_USD"},   # Risk-off → USD safe haven
        "usd_index":         {"UP": "BULLISH_USD",  "DOWN": "BEARISH_USD"},
        "gold_london_fix":   {"UP": "BEARISH_USD",  "DOWN": "BULLISH_USD"},   # Gold vs USD inverse
        "high_yield_spread": {"UP": "BEARISH_USD",  "DOWN": "BULLISH_USD"},   # Credit stress → risk-off
        "yield_curve_spread": {"UP": "BULLISH_USD", "DOWN": "BEARISH_USD"},
        "nonfarm_payrolls":  {"UP": "BULLISH_USD",  "DOWN": "BEARISH_USD"},
        "unemployment_rate": {"UP": "BEARISH_USD",  "DOWN": "BULLISH_USD"},
    }

    # Key FRED series to monitor
    FRED_SERIES = {
        "DFF":              "fed_funds_rate",
        "CPIAUCSL":         "cpi",
        "DGS10":            "us_10y_yield",
        "T10Y2Y":           "yield_curve_spread",
        "VIXCLS":           "vix",
        "DTWEXBGS":         "usd_index",
        "GOLDAMGBD228NLBM": "gold_london_fix",
        "BAMLH0A0HYM2":     "high_yield_spread",
        "PAYEMS":           "nonfarm_payrolls",
        "UNRATE":           "unemployment_rate",
        "M2SL":             "m2_money_supply",
    }

    async def get_briefing(self) -> ForexIntelBriefing:
        """
        Run full Crucix intelligence sweep in parallel.
        Returns structured briefing for SentimentAgent.
        """
        logger.info("[Crucix] Running intelligence sweep...")
        
        # Run all sources concurrently — Crucix pattern
        fred_task  = self._fetch_fred()
        news_task  = self._fetch_forex_news()
        
        fred_signals, news_headlines = await asyncio.gather(
            fred_task, news_task, return_exceptions=True
        )
        
        if isinstance(fred_signals, Exception):
            fred_signals  = []
        if isinstance(news_headlines, Exception):
            news_headlines = []

        all_signals: List[MacroSignal] = fred_signals

        # Determine regime
        risk_regime = self._assess_risk_regime(all_signals)
        usd_bias    = self._assess_usd_bias(all_signals)

        briefing_text = self._generate_briefing_text(
            all_signals, news_headlines, risk_regime, usd_bias
        )

        return ForexIntelBriefing(
            timestamp      = datetime.now(timezone.utc).isoformat(),
            macro_signals  = all_signals,
            news_headlines = news_headlines,
            risk_regime    = risk_regime,
            usd_bias       = usd_bias,
            key_levels_text = self._key_levels_text(all_signals),
            briefing_text  = briefing_text,
        )

    async def _fetch_fred(self) -> List[MacroSignal]:
        """Fetch FRED macro indicators — from Crucix apis/sources/fred.mjs."""
        if not FRED_API_KEY:
            return []
        
        signals = []
        try:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for series_id, name in self.FRED_SERIES.items():
                    url = (f"{self.FRED_BASE}?series_id={series_id}"
                           f"&api_key={FRED_API_KEY}&file_type=json"
                           f"&sort_order=desc&limit=2")
                    tasks.append(self._fetch_fred_series(session, series_id, name, url))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, MacroSignal):
                        signals.append(r)
        except Exception as e:
            logger.warning(f"[FRED] Fetch error: {e}")
        return signals

    async def _fetch_fred_series(self, session: aiohttp.ClientSession,
                                   series_id: str, name: str, url: str) -> Optional[MacroSignal]:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
                data = await r.json()
                obs  = [o for o in data.get("observations", []) if o["value"] != "."]
                if len(obs) < 2:
                    return None
                current  = float(obs[0]["value"])
                previous = float(obs[1]["value"])
                change   = current - previous
                direction = "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
                impact_map = self.USD_IMPACT_MAP.get(name, {})
                impact = impact_map.get(direction, "NEUTRAL")
                
                return MacroSignal(
                    source        = "FRED",
                    indicator     = name,
                    value         = round(current, 4),
                    previous      = round(previous, 4),
                    change        = round(change, 4),
                    direction     = direction,
                    impact_on_usd = impact,
                    description   = f"{name}: {current} (prev: {previous}, Δ{change:+.4f})",
                )
        except Exception:
            return None

    async def _fetch_forex_news(self) -> List[str]:
        """Fetch forex-relevant news headlines."""
        if not NEWS_API_KEY:
            return []
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "q":        "forex OR 'federal reserve' OR 'ECB' OR 'currency' OR 'interest rate'",
                    "language": "en",
                    "sortBy":   "publishedAt",
                    "pageSize": "10",
                    "apiKey":   NEWS_API_KEY,
                }
                async with session.get(
                    self.NEWS_BASE,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    data = await r.json()
                    return [a.get("title", "") for a in data.get("articles", [])[:10]]
        except Exception as e:
            logger.warning(f"[News] Fetch error: {e}")
            return []

    def _assess_risk_regime(self, signals: List[MacroSignal]) -> str:
        """Crucix-style regime detection: cross-domain signal correlation."""
        bearish_usd = sum(1 for s in signals if s.impact_on_usd == "BEARISH_USD")
        bullish_usd = sum(1 for s in signals if s.impact_on_usd == "BULLISH_USD")
        
        # VIX rising = risk-off
        vix_sig = next((s for s in signals if s.indicator == "vix"), None)
        hy_sig  = next((s for s in signals if s.indicator == "high_yield_spread"), None)
        
        risk_off_score = 0
        if vix_sig and vix_sig.direction == "UP":  risk_off_score += 2
        if hy_sig  and hy_sig.direction  == "UP":  risk_off_score += 2
        if bearish_usd > bullish_usd:              risk_off_score += 1
        
        if risk_off_score >= 3:   return "RISK_OFF"
        elif risk_off_score == 0: return "RISK_ON"
        return "NEUTRAL"

    def _assess_usd_bias(self, signals: List[MacroSignal]) -> str:
        """Assess overall USD directional bias from macro signals."""
        bullish = sum(1 for s in signals if s.impact_on_usd == "BULLISH_USD")
        bearish = sum(1 for s in signals if s.impact_on_usd == "BEARISH_USD")
        if bullish > bearish + 1: return "BULLISH"
        if bearish > bullish + 1: return "BEARISH"
        return "NEUTRAL"

    def _key_levels_text(self, signals: List[MacroSignal]) -> str:
        """Generate key levels natural language summary."""
        parts = []
        for s in signals:
            if s.indicator in ["fed_funds_rate", "vix", "usd_index", "gold_london_fix"]:
                parts.append(f"{s.indicator}: {s.value} ({s.direction})")
        return " | ".join(parts) if parts else "No macro data available"

    def _generate_briefing_text(self, signals: List[MacroSignal],
                                  headlines: List[str],
                                  regime: str, usd_bias: str) -> str:
        """
        Crucix BRIEFING_PROMPT.md pattern:
        "leverage-first intelligence note" — what can the trader DO with this?
        """
        lines = [
            f"REGIME: {regime} | USD BIAS: {usd_bias}",
            "",
            "KEY MACRO SIGNALS:",
        ]
        for s in signals[:8]:
            lines.append(f"  [{s.source}] {s.description} → {s.impact_on_usd}")
        
        if headlines:
            lines.append("\nBREAKING HEADLINES:")
            for h in headlines[:5]:
                lines.append(f"  • {h}")
        
        lines.append("\nTRADING IMPLICATIONS:")
        if usd_bias == "BULLISH":
            lines.append("  → Favour USD-long positions (buy USDCHF, USDJPY, sell EURUSD)")
        elif usd_bias == "BEARISH":
            lines.append("  → Favour USD-short positions (buy EURUSD, GBPUSD, sell USDJPY)")
        else:
            lines.append("  → Mixed signals — reduce size, wait for confirmation")
        
        return "\n".join(lines)
