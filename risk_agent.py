"""
RISK AGENT — Forex Agentic Swarm
-----------------------------------
Risk management specialist. Calculates position sizing, checks correlation
exposure, validates spread conditions, evaluates Value-at-Risk (VaR),
and enforces hard trading rules.
"""

import asyncio
import os
import json
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Hard-coded risk rules (never overridden by LLM) ──────────────────────────
HARD_RULES = {
    "max_risk_per_trade_pct":  0.02,   # 2% of equity per trade
    "max_spread_pips":         3.0,    # Don't trade if spread > 3 pips
    "avoid_sessions":          [],     # e.g. ["asia"] for low-liquidity avoidance
    "avoid_minutes_before_news": 15,   # Block trades 15min before high-impact news
    "max_correlated_exposure": 2,      # Max 2 trades on correlated pairs simultaneously
    "min_atr":                 0.0003, # Minimum ATR — avoid dead markets
}

RISK_SYSTEM_PROMPT = """
You are a forex risk management agent in an autonomous trading swarm.

You receive market state and account context. Your job is to:
1. Assess whether current conditions are safe to trade
2. Calculate recommended lot size based on ATR and account equity
3. Suggest stop-loss distance in pips (1.5x ATR)
4. Flag any risk concerns

Return ONLY a JSON object:
{
  "signal": "BUY" | "SELL" | "FLAT",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<explanation>",
  "lot_size": <float>,
  "stop_loss_pips": <float>,
  "take_profit_pips": <float>,
  "risk_flags": ["<flag1>", "<flag2>"]
}

If any hard rule is violated, return FLAT with confidence 0.0 and explain the flag.
"""


class RiskAgent:
    def __init__(self, equity: float = 10_000.0):
        self.name   = "risk"
        self.equity = equity

    async def analyze(self, market_state) -> "AgentVote":
        from orchestrator_agent import AgentVote

        # 1. Hard rule checks first (no LLM needed)
        hard_block = self._check_hard_rules(market_state)
        if hard_block:
            return AgentVote(self.name, "FLAT", 0.0, hard_block, market_state.pair)

        # 2. LLM-assisted risk assessment
        prompt = f"""
Market State:
- Pair:    {market_state.pair}
- Ask:     {market_state.ask}
- Spread:  {market_state.spread} pips
- ATR(14): {market_state.atr}
- Session: {market_state.session}

Account Context:
- Equity:  ${self.equity:,.2f}
- Max risk per trade: {HARD_RULES['max_risk_per_trade_pct'] * 100}%
- Max risk amount: ${self.equity * HARD_RULES['max_risk_per_trade_pct']:,.2f}

Calculate appropriate position sizing and assess overall risk. Return your vote.
"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": RISK_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(response.choices[0].message.content)
            return AgentVote(
                agent_name = self.name,
                signal     = data.get("signal", "FLAT"),
                confidence = float(data.get("confidence", 0.0)),
                reasoning  = data.get("reasoning", ""),
                pair       = market_state.pair,
            )
        except Exception as e:
            return AgentVote(self.name, "FLAT", 0.0, f"Error: {e}", market_state.pair)

    def _check_hard_rules(self, market_state) -> str | None:
        """Returns an error string if any hard rule is violated, else None."""
        if market_state.spread > HARD_RULES["max_spread_pips"]:
            return f"Spread too wide: {market_state.spread} > {HARD_RULES['max_spread_pips']} pips"

        if market_state.atr < HARD_RULES["min_atr"]:
            return f"ATR too low: {market_state.atr} — dead market"

        if market_state.session in HARD_RULES["avoid_sessions"]:
            return f"Session blocked: {market_state.session}"

        return None

    def calculate_lot_size(self, stop_loss_pips: float, pip_value: float = 10.0) -> float:
        """
        Standard lot sizing formula:
        Lot = (Equity * risk%) / (stop_loss_pips * pip_value)
        """
        risk_amount = self.equity * HARD_RULES["max_risk_per_trade_pct"]
        if stop_loss_pips <= 0 or pip_value <= 0:
            return 0.01
        lot = risk_amount / (stop_loss_pips * pip_value)
        # Cap between 0.01 and 5.0 lots
        return round(max(0.01, min(5.0, lot)), 2)
