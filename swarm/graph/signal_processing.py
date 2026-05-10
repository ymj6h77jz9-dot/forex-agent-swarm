"""
SIGNAL PROCESSING — Forex Agentic Swarm
-----------------------------------------------------------------
Processes raw agent votes and LLM outputs into clean, actionable
trading signals. Handles Bull/Bear debate resolution and
Risk (Aggressive/Conservative) debate resolution.

Source: TradingAgents/graph/signal_processing.py
v2: Fully migrated to llm_client (OpenRouter free). No bare openai imports.
    Enhanced with Kronos weighting + robust JSON fallback.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from llm_client import llm, llm_json

logger = logging.getLogger(__name__)

# ─── Prompts ──────────────────────────────────────────────────────────────────

SIGNAL_EXTRACT_PROMPT = """
You are a trading signal extractor. Given an analyst report, extract the core direction.
Return EXACTLY one word only — no punctuation, no explanation:
BUY | SELL | HOLD | FLAT
"""

DEBATE_JUDGE_PROMPT = """
You are a neutral judge resolving a debate between a bull and a bear forex analyst.
Return ONLY valid JSON:
{
  "decision": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}
"""

RISK_JUDGE_PROMPT = """
You are a risk arbitrator. Resolve a debate between aggressive and conservative risk analysts.
Return ONLY valid JSON:
{
  "approved": true | false,
  "lot_size_adjustment": <float, e.g. 0.8 = reduce 20%>,
  "sl_adjustment_pips": <float>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}
"""


class SignalProcessor:
    """
    Processes trading signals and resolves multi-agent debates.
    All LLM calls route through llm_client → OpenRouter free models.
    """

    def __init__(self):
        pass

    # ── Signal extraction ─────────────────────────────────────────────────────

    async def extract_signal(self, text: str) -> str:
        """
        Extract BUY/SELL/HOLD/FLAT from any agent text output.
        Regex-first, LLM fallback.
        """
        text_upper = text.upper()
        for signal in ("BUY", "SELL", "HOLD", "FLAT"):
            if re.search(rf"\b{signal}\b", text_upper):
                return signal
        try:
            result = await llm(
                messages=[
                    {"role": "system", "content": SIGNAL_EXTRACT_PROMPT},
                    {"role": "user",   "content": text[:2000]},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            word = result.strip().upper()
            if word in ("BUY", "SELL", "HOLD", "FLAT"):
                return word
        except Exception as e:
            logger.warning(f"[SignalProcessor] extract_signal LLM failed: {e}")
        return "FLAT"

    # ── Debate resolution ─────────────────────────────────────────────────────

    async def resolve_investment_debate(
        self,
        bull_argument: str,
        bear_argument: str,
        pair: str,
    ) -> Dict[str, Any]:
        """
        Bull vs Bear debate. Returns judge ruling dict.
        """
        prompt = (
            f"Pair: {pair}\n\n"
            f"BULL ARGUMENT:\n{bull_argument[:1500]}\n\n"
            f"BEAR ARGUMENT:\n{bear_argument[:1500]}\n\n"
            "Rule on this debate. Return JSON."
        )
        try:
            result = await llm_json(
                messages=[
                    {"role": "system", "content": DEBATE_JUDGE_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            return result if result else {"decision": "HOLD", "confidence": 0.0, "reasoning": "Empty response"}
        except Exception as e:
            logger.error(f"[SignalProcessor] resolve_investment_debate failed: {e}")
            return {"decision": "HOLD", "confidence": 0.0, "reasoning": f"Error: {e}"}

    async def resolve_risk_debate(
        self,
        aggressive_arg: str,
        conservative_arg: str,
    ) -> Dict[str, Any]:
        """
        Aggressive vs Conservative risk debate. Returns arbitration dict.
        """
        prompt = (
            f"AGGRESSIVE ANALYST:\n{aggressive_arg[:1500]}\n\n"
            f"CONSERVATIVE ANALYST:\n{conservative_arg[:1500]}\n\n"
            "Rule on position sizing and risk. Return JSON."
        )
        try:
            result = await llm_json(
                messages=[
                    {"role": "system", "content": RISK_JUDGE_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            return result if result else {
                "approved": False, "lot_size_adjustment": 1.0,
                "sl_adjustment_pips": 0.0, "confidence": 0.0, "reasoning": "Empty response"
            }
        except Exception as e:
            logger.error(f"[SignalProcessor] resolve_risk_debate failed: {e}")
            return {
                "approved": False, "lot_size_adjustment": 1.0,
                "sl_adjustment_pips": 0.0, "confidence": 0.0, "reasoning": f"Error: {e}"
            }

    # ── Weighted consensus ────────────────────────────────────────────────────

    def compute_weighted_consensus(
        self,
        votes: List[Dict[str, Any]],
        weights: Dict[str, float],
        mirofish_prediction: Optional[Dict] = None,
        kronos_prediction:   Optional[Dict] = None,
        threshold: float = 0.70,
    ) -> Tuple[str, float]:
        """
        Compute weighted consensus from agent votes + MiroFish + Kronos.

        Weight allocation (defaults):
          Analyst   28% | Sentiment 22% | Risk 20%
          MiroFish  15% | Kronos    15%

        Args:
            votes:               List of {agent, signal, confidence}
            weights:             Dict of agent_name → weight (0.0–1.0)
            mirofish_prediction: {bullish, bearish, confidence}
            kronos_prediction:   {direction, magnitude_pct, confidence}
            threshold:           Min score to trigger BUY or SELL

        Returns:
            (decision, score)
        """
        buy_score  = 0.0
        sell_score = 0.0

        # Normalise agent weights
        total_agent_w = sum(weights.get(v.get("agent", ""), 0.0) for v in votes)
        if total_agent_w <= 0:
            total_agent_w = 1.0

        # Agent votes contribute 70% of the total (MiroFish 15% + Kronos 15% = 30%)
        agent_budget = 0.70

        for vote in votes:
            raw_w = weights.get(vote.get("agent", ""), 0.0)
            w     = (raw_w / total_agent_w) * agent_budget
            conf  = float(vote.get("confidence", 0.0))
            sig   = vote.get("signal", "FLAT")
            if sig == "BUY":
                buy_score  += w * conf
            elif sig == "SELL":
                sell_score += w * conf

        # MiroFish PSO probabilistic backbone (15%)
        if mirofish_prediction:
            mf_w    = 0.15
            mf_bull = float(mirofish_prediction.get("bullish", 0.5))
            mf_bear = float(mirofish_prediction.get("bearish", 0.5))
            mf_conf = float(mirofish_prediction.get("confidence", 0.5))
            buy_score  += mf_bull * mf_conf * mf_w
            sell_score += mf_bear * mf_conf * mf_w

        # Kronos foundation model (15%)
        if kronos_prediction:
            k_w    = 0.15
            k_dir  = kronos_prediction.get("direction", "neutral").upper()
            k_conf = float(kronos_prediction.get("confidence", 0.5))
            k_mag  = min(float(kronos_prediction.get("magnitude_pct", 0.0)) / 2.0, 1.0)
            k_score = k_conf * (0.5 + k_mag)
            if k_dir == "UP":
                buy_score  += k_score * k_w
            elif k_dir == "DOWN":
                sell_score += k_score * k_w

        # Clamp scores
        buy_score  = min(buy_score,  1.0)
        sell_score = min(sell_score, 1.0)

        if buy_score >= threshold and buy_score > sell_score:
            return "BUY",  round(buy_score,  4)
        elif sell_score >= threshold and sell_score > buy_score:
            return "SELL", round(sell_score, 4)
        else:
            return "HOLD", round(max(buy_score, sell_score), 4)
