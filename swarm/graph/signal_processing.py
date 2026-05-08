"""
SIGNAL PROCESSING — Integrated from ymj6h77jz9-dot/TradingAgents
-----------------------------------------------------------------
Processes raw agent votes and LLM outputs into clean, actionable
trading signals. Also handles the Bull/Bear debate resolution and
the Risk (Aggressive/Conservative) debate resolution.

Source: TradingAgents/graph/signal_processing.py
Enhanced with: multi-agent weighted voting, MiroFish integration
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SIGNAL_EXTRACT_PROMPT = """
You are a trading signal extractor. Given an analyst report or agent output,
extract the core directional decision.

Return EXACTLY one of these words (nothing else):
BUY | SELL | HOLD | FLAT

If the text is ambiguous or mixed, return FLAT.
"""

DEBATE_JUDGE_PROMPT = """
You are a neutral judge resolving a debate between trading analysts.
Given the bull and bear arguments, make a final ruling.

Return ONLY a JSON object:
{
  "decision": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining your ruling>"
}
"""

RISK_JUDGE_PROMPT = """
You are a risk arbitrator resolving a debate between aggressive and conservative risk analysts.
Given their arguments about position sizing, SL/TP, and market conditions, make a final ruling.

Return ONLY a JSON object:
{
  "approved": true | false,
  "lot_size_adjustment": <float, e.g. 0.8 means reduce lot size by 20%>,
  "sl_adjustment_pips": <float, additional pips to add to SL distance>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}
"""


class SignalProcessor:
    """
    Processes trading signals and resolves multi-agent debates.
    Integrated from TradingAgents — enhanced for forex swarm.
    """

    def __init__(self):
        pass

    async def extract_signal(self, text: str) -> str:
        """
        Extract BUY/SELL/HOLD/FLAT from any agent text output.
        Uses LLM as fallback after regex attempt.
        """
        # Fast regex path first
        text_upper = text.upper()
        for signal in ["BUY", "SELL", "HOLD", "FLAT"]:
            if re.search(rf"\b{signal}\b", text_upper):
                return signal

        # LLM fallback
        try:
            resp = await client.chat.completions.create(
                model    = "gpt-4o-mini",
                messages = [
                    {"role": "system", "content": SIGNAL_EXTRACT_PROMPT},
                    {"role": "user",   "content": text[:2000]},
                ],
                temperature = 0.0,
            )
            signal = resp.choices[0].message.content.strip().upper()
            if signal in ("BUY", "SELL", "HOLD", "FLAT"):
                return signal
        except Exception as e:
            logger.warning(f"Signal extraction LLM failed: {e}")

        return "FLAT"

    async def resolve_investment_debate(self, bull_argument: str,
                                         bear_argument: str,
                                         pair: str) -> Dict[str, Any]:
        """
        Bull vs Bear debate resolution.
        Mirrors TradingAgents InvestDebateState judge logic.
        """
        import json
        prompt = f"""
Pair: {pair}

BULL ARGUMENT:
{bull_argument[:1500]}

BEAR ARGUMENT:
{bear_argument[:1500]}

Rule on this debate.
"""
        try:
            resp = await client.chat.completions.create(
                model           = "gpt-4o",
                messages        = [
                    {"role": "system", "content": DEBATE_JUDGE_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format = {"type": "json_object"},
                temperature     = 0.2,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"Debate resolution failed: {e}")
            return {"decision": "HOLD", "confidence": 0.0, "reasoning": "Error"}

    async def resolve_risk_debate(self, aggressive_arg: str,
                                   conservative_arg: str) -> Dict[str, Any]:
        """
        Aggressive vs Conservative risk analyst debate resolution.
        Mirrors TradingAgents RiskDebateState judge logic.
        """
        import json
        prompt = f"""
AGGRESSIVE RISK ANALYST:
{aggressive_arg[:1500]}

CONSERVATIVE RISK ANALYST:
{conservative_arg[:1500]}

Rule on position sizing and risk parameters.
"""
        try:
            resp = await client.chat.completions.create(
                model           = "gpt-4o",
                messages        = [
                    {"role": "system", "content": RISK_JUDGE_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format = {"type": "json_object"},
                temperature     = 0.1,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"Risk debate resolution failed: {e}")
            return {"approved": False, "confidence": 0.0, "reasoning": "Error"}

    def compute_weighted_consensus(self, votes: List[Dict[str, Any]],
                                    weights: Dict[str, float],
                                    mirofish_prediction: Optional[Dict] = None,
                                    threshold: float = 0.72) -> Tuple[str, float]:
        """
        Compute weighted consensus from agent votes + MiroFish prediction.
        
        Returns: (decision: str, score: float)
        """
        buy_score  = 0.0
        sell_score = 0.0
        total_weight = sum(weights.get(v["agent"], 0.0) for v in votes)

        for vote in votes:
            w = weights.get(vote["agent"], 0.0) / (total_weight or 1.0)
            conf = vote.get("confidence", 0.0)
            if vote["signal"] == "BUY":
                buy_score  += w * conf
            elif vote["signal"] == "SELL":
                sell_score += w * conf

        # Blend in MiroFish probabilistic backbone
        if mirofish_prediction:
            mf_weight  = 0.20   # 20% weight to MiroFish
            mf_bull    = mirofish_prediction.get("bullish", 0.5)
            mf_bear    = mirofish_prediction.get("bearish", 0.5)
            mf_conf    = mirofish_prediction.get("confidence", 0.5)

            buy_score  = buy_score  * (1 - mf_weight) + mf_bull * mf_conf * mf_weight
            sell_score = sell_score * (1 - mf_weight) + mf_bear * mf_conf * mf_weight

        if buy_score >= threshold:
            return "BUY",  round(buy_score,  4)
        elif sell_score >= threshold:
            return "SELL", round(sell_score, 4)
        else:
            return "HOLD", round(max(buy_score, sell_score), 4)
