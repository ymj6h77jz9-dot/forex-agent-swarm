"""
ANALYST AGENT — Forex Agentic Swarm
-------------------------------------
Technical analysis specialist. Reads price action, indicators (EMA, RSI, MACD,
ATR), candlestick patterns, and S/R levels. Returns a structured vote with
confidence score and reasoning.
"""

import asyncio
from dataclasses import dataclass
from openai import AsyncOpenAI
import os

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

ANALYST_SYSTEM_PROMPT = """
You are an expert forex technical analyst agent operating inside an autonomous trading swarm.

Your job:
- Analyze the incoming market state (price, ATR, session, recent OHLCV data if provided)
- Identify high-probability setups based on: EMA crossovers, RSI divergence, MACD histogram, 
  S/R levels, candlestick patterns, and session timing
- Return ONLY a JSON object in this exact format:
  {
    "signal": "BUY" | "SELL" | "FLAT",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "<concise 1-2 sentence explanation>"
  }

Be disciplined. Only assign confidence > 0.8 when multiple confluences align.
When in doubt, return FLAT. Never hallucinate price data.
"""

class AnalystAgent:
    def __init__(self):
        self.name = "analyst"

    async def analyze(self, market_state) -> "AgentVote":
        from orchestrator_agent import AgentVote

        prompt = f"""
Market State:
- Pair:      {market_state.pair}
- Bid/Ask:   {market_state.bid} / {market_state.ask}
- Spread:    {market_state.spread} pips
- ATR(14):   {market_state.atr}
- Session:   {market_state.session}
- Timestamp: {market_state.timestamp}

Analyze this and return your structured vote.
"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            result = response.choices[0].message.content
            import json
            data = json.loads(result)

            return AgentVote(
                agent_name = self.name,
                signal     = data.get("signal", "FLAT"),
                confidence = float(data.get("confidence", 0.0)),
                reasoning  = data.get("reasoning", ""),
                pair       = market_state.pair,
            )

        except Exception as e:
            from orchestrator_agent import AgentVote
            return AgentVote(self.name, "FLAT", 0.0, f"Error: {e}", market_state.pair)
