"""
ANALYST AGENT — KRATOS v2
---------------------------
Technical analysis specialist. Reads price action, indicators (EMA, RSI,
MACD, ATR, Bollinger Bands), S/R levels, and candlestick patterns.

v2 CRITICAL FIX: Now receives OHLCV candle data from OpenBB via
market_state._candle_summary — no longer blind to price history.

Indicators computed in-process (no external TA lib required):
  EMA(8), EMA(21), EMA(50) — trend bias
  RSI(14) — momentum + divergence
  ATR(14) — volatility context
  Bollinger Bands(20,2) — range/breakout
  MACD(12,26,9) — momentum confirmation
  Session alignment — Asia/London/NY bias

LLM call: OpenRouter → openai/gpt-oss-120b:free (primary)
"""

import json
import logging
import numpy as np
from typing import Optional

from orchestrator_agent import AgentVote
from llm_client import llm_json

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """
You are an expert forex technical analyst inside an autonomous trading swarm.

You receive:
  1. Current market state (pair, bid, ask, ATR, session, spread)
  2. OHLCV technical summary (EMA crossovers, RSI, MACD, Bollinger Bands)

Your job:
  - Identify high-probability setups using indicator confluence
  - Consider session bias (Asia = range, London = breakout, NY = momentum)
  - Only assign confidence > 0.80 when 3+ indicators align
  - When in doubt → FLAT

Return ONLY valid JSON:
{
  "signal": "BUY" | "SELL" | "FLAT",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<1-2 sentences citing specific indicators>",
  "key_confluences": ["<ema_cross>", "<rsi_level>", ...],
  "session_alignment": true | false
}
"""


def _calc_ema(closes: np.ndarray, period: int) -> float:
    """Exponential moving average — last value."""
    if len(closes) < period:
        return float(closes[-1]) if len(closes) else 0.0
    k = 2.0 / (period + 1)
    ema = float(closes[0])
    for c in closes[1:]:
        ema = float(c) * k + ema * (1 - k)
    return round(ema, 6)


def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    """RSI(14) — last value."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    rs = avg_gain / max(avg_loss, 1e-10)
    return round(100 - (100 / (1 + rs)), 2)


def _calc_macd(closes: np.ndarray) -> dict:
    """MACD(12,26,9) — last histogram value."""
    if len(closes) < 26:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
    ema12 = _calc_ema(closes, 12)
    ema26 = _calc_ema(closes, 26)
    macd_line = ema12 - ema26
    # Approximate signal line from last 9 MACD values
    macd_vals = []
    for i in range(9, 0, -1):
        e12 = _calc_ema(closes[:-i] if i > 1 else closes, 12)
        e26 = _calc_ema(closes[:-i] if i > 1 else closes, 26)
        macd_vals.append(e12 - e26)
    macd_vals.append(macd_line)
    signal_line = _calc_ema(np.array(macd_vals), 9)
    return {
        "macd":      round(macd_line, 6),
        "signal":    round(signal_line, 6),
        "histogram": round(macd_line - signal_line, 6),
    }


def _calc_bb(closes: np.ndarray, period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands(20,2) — upper/mid/lower and %B."""
    if len(closes) < period:
        c = float(closes[-1])
        return {"upper": c, "mid": c, "lower": c, "pct_b": 0.5}
    window = closes[-period:]
    mid    = float(np.mean(window))
    std    = float(np.std(window))
    upper  = mid + std_dev * std
    lower  = mid - std_dev * std
    pct_b  = (float(closes[-1]) - lower) / max(upper - lower, 1e-10)
    return {
        "upper":  round(upper, 5),
        "mid":    round(mid,   5),
        "lower":  round(lower, 5),
        "pct_b":  round(pct_b, 4),
    }


def _calc_atr(highs, lows, closes, period: int = 14) -> float:
    """ATR(14) from OHLCV arrays."""
    if len(closes) < 2:
        return 0.001
    tr = [max(highs[i] - lows[i],
              abs(highs[i] - closes[i-1]),
              abs(lows[i]  - closes[i-1]))
          for i in range(1, len(closes))]
    return round(float(np.mean(tr[-period:])), 6)


def build_technical_summary(candle_df, current_price: float) -> str:
    """
    Compute all indicators from the OHLCV DataFrame and return
    a human-readable summary for the LLM prompt.
    """
    try:
        closes = candle_df["close"].values.astype(float)
        highs  = candle_df["high"].values.astype(float)  if "high"  in candle_df.columns else closes
        lows   = candle_df["low"].values.astype(float)   if "low"   in candle_df.columns else closes

        ema8  = _calc_ema(closes, 8)
        ema21 = _calc_ema(closes, 21)
        ema50 = _calc_ema(closes, 50)
        rsi   = _calc_rsi(closes, 14)
        macd  = _calc_macd(closes)
        bb    = _calc_bb(closes, 20)
        atr   = _calc_atr(highs, lows, closes, 14)

        # Trend bias
        trend = "BULLISH" if ema8 > ema21 > ema50 else \
                "BEARISH" if ema8 < ema21 < ema50 else "MIXED"

        # RSI state
        rsi_state = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"

        # MACD momentum
        macd_state = "BULLISH" if macd["histogram"] > 0 else "BEARISH"

        # BB squeeze
        bb_state = "UPPER_BAND" if bb["pct_b"] > 0.85 else \
                   "LOWER_BAND" if bb["pct_b"] < 0.15 else "MID_RANGE"

        # Price vs S/R (simple: recent high/low)
        recent_high = float(np.max(highs[-20:])) if len(highs) >= 20 else float(np.max(highs))
        recent_low  = float(np.min(lows[-20:]))  if len(lows)  >= 20 else float(np.min(lows))
        pct_from_high = round((current_price - recent_high) / max(recent_high, 1e-10) * 100, 3)
        pct_from_low  = round((current_price - recent_low)  / max(recent_low,  1e-10) * 100, 3)

        lines = [
            f"Bars analysed: {len(closes)}",
            f"EMA(8)={ema8:.5f}  EMA(21)={ema21:.5f}  EMA(50)={ema50:.5f}  → Trend: {trend}",
            f"RSI(14)={rsi:.1f}  → {rsi_state}",
            f"MACD histogram={macd['histogram']:+.6f}  signal={macd['signal']:.6f}  → {macd_state}",
            f"Bollinger %B={bb['pct_b']:.3f}  upper={bb['upper']:.5f}  lower={bb['lower']:.5f}  → {bb_state}",
            f"ATR(14)={atr:.5f}",
            f"20-bar high={recent_high:.5f} ({pct_from_high:+.3f}%)  low={recent_low:.5f} ({pct_from_low:+.3f}%)",
        ]
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"[AnalystAgent] Technical summary failed: {e}")
        return f"Technical summary unavailable: {e}"


class AnalystAgent:
    def __init__(self):
        self.name = "analyst"

    async def analyze(self, market_state) -> AgentVote:
        # Build technical summary from candle data (CRITICAL FIX: was missing)
        candle_df = getattr(market_state, "_candle_df", None)
        if candle_df is not None and not candle_df.empty:
            tech_summary = build_technical_summary(candle_df, market_state.ask)
        else:
            # Fallback: basic context without OHLCV
            tech_summary = (
                f"No OHLCV data available. Using spot context only.\n"
                f"Spread: {market_state.spread} pips | ATR: {market_state.atr}"
            )

        # Session bias
        session_bias = {
            "asia":    "Range-bound. Favour mean-reversion. Avoid breakout trades.",
            "london":  "Breakout session. Trend trades favoured. Watch for fake-outs.",
            "ny":      "Momentum session. Follow London trend or look for reversals.",
            "overlap": "High volatility overlap. Both breakout and reversal possible.",
        }.get(market_state.session, "Unknown session.")

        prompt = f"""
Currency Pair:  {market_state.pair}
Bid/Ask:        {market_state.bid} / {market_state.ask}
Spread:         {market_state.spread} pips
ATR(14):        {market_state.atr}
Session:        {market_state.session}  — {session_bias}
Timestamp:      {market_state.timestamp}

TECHNICAL ANALYSIS (computed in-process):
{tech_summary}

Additional context: {getattr(market_state, '_candle_summary', '')}

Based on all indicators above, provide your structured vote JSON.
"""
        try:
            data = await llm_json(
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return AgentVote(
                agent_name  = self.name,
                signal      = data.get("signal", "FLAT"),
                confidence  = float(data.get("confidence", 0.0)),
                reasoning   = data.get("reasoning", ""),
                pair        = market_state.pair,
            )
        except Exception as e:
            logger.error(f"[AnalystAgent] analyze failed: {e}")
            return AgentVote(self.name, "FLAT", 0.0, f"Error: {e}", market_state.pair)
