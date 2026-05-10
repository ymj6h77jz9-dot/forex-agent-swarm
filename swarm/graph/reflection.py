"""
REFLECTION ENGINE — Forex Agentic Swarm
-----------------------------------------------------------------
After each trade closes, the Reflector reviews the full decision chain,
identifies what was correct/incorrect, and generates improvement lessons
stored in the MemPalace memory system for future cycles.

Source: TradingAgents/graph/reflection.py
v2: Fully migrated to llm_client (OpenRouter free). No bare openai imports.
    Enhanced with structured logging and cycle-level analytics.
"""

import json
import logging
from typing import Dict, Any, Optional

from llm_client import llm_json

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """
You are an expert forex trading analyst reviewing a completed trade decision cycle.
Extract actionable lessons to improve future decisions.

You will receive:
1. Full agent decision chain (votes, reasoning, confidence scores)
2. Actual trade outcome (PnL, price movement, execution details)

Your analysis MUST cover:
1. REASONING: Was each agent correct? (Analyst, Sentiment, Risk, MiroFish, Kronos)
2. IMPROVEMENT: What should change? (indicator weights, news reliability, session timing)
3. SUMMARY: 2-3 sentence lesson for memory storage.
4. MEMORY_QUERY: Single sentence (≤100 tokens) — the core lesson for BM25 retrieval.

Return ONLY valid JSON:
{
  "reasoning": "<detailed analysis>",
  "improvement": "<specific changes>",
  "summary": "<2-3 sentence lesson>",
  "memory_query": "<single sentence for BM25 storage>",
  "agent_accuracy": {
    "analyst":   <float 0-1>,
    "sentiment": <float 0-1>,
    "risk":      <float 0-1>,
    "mirofish":  <float 0-1>,
    "kronos":    <float 0-1>
  }
}
"""


class Reflector:
    """
    Post-trade reflection engine.
    All LLM calls route through llm_client → OpenRouter free models.
    """

    def __init__(self):
        self.reflection_history: list = []

    async def reflect(
        self,
        agent_state:      Dict[str, Any],
        trade_result:     Dict[str, Any],
        actual_direction: str,
    ) -> Dict[str, Any]:
        """
        Perform post-trade reflection and return lessons.

        Args:
            agent_state:      Serialised AgentState from the trade cycle
            trade_result:     Execution result (status, pnl, price, etc.)
            actual_direction: Actual price movement — "UP" | "DOWN" | "FLAT"

        Returns:
            dict with keys: reasoning, improvement, summary, memory_query, agent_accuracy
        """
        # Safely serialise agent_state (may contain non-JSON-serialisable objects)
        try:
            state_str = json.dumps(agent_state, indent=2, default=str)
        except Exception:
            state_str = str(agent_state)

        try:
            result_str = json.dumps(trade_result, indent=2, default=str)
        except Exception:
            result_str = str(trade_result)

        prompt = (
            f"=== AGENT DECISION STATE ===\n{state_str}\n\n"
            f"=== TRADE RESULT ===\n{result_str}\n\n"
            f"=== ACTUAL MARKET OUTCOME ===\n"
            f"Price moved: {actual_direction}\n"
            f"PnL: {trade_result.get('pnl', 'N/A')}\n\n"
            "Analyse this cycle. Return your structured reflection JSON."
        )

        _default = {
            "reasoning": "Reflection unavailable",
            "improvement": "",
            "summary": "No reflection generated",
            "memory_query": "",
            "agent_accuracy": {
                "analyst": 0.5, "sentiment": 0.5,
                "risk": 0.5, "mirofish": 0.5, "kronos": 0.5
            },
        }

        try:
            reflection = await llm_json(
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=700,
            )
            if not reflection:
                logger.warning("[Reflector] Empty response — using defaults")
                return _default

            self.reflection_history.append({
                "cycle_id":  agent_state.get("cycle_id"),
                "timestamp": agent_state.get("timestamp"),
                "pair":      agent_state.get("pair"),
                "direction": actual_direction,
                "pnl":       trade_result.get("pnl"),
                **reflection,
            })
            logger.info(
                f"[Reflector] cycle={agent_state.get('cycle_id')} "
                f"pair={agent_state.get('pair')} direction={actual_direction} "
                f"summary={reflection.get('summary', '')[:80]}"
            )
            return reflection

        except Exception as e:
            logger.error(f"[Reflector] reflect() failed: {e}")
            return _default

    def extract_situation_for_memory(self, agent_state: Dict[str, Any]) -> str:
        """
        Extract a searchable situation description for MemPalace BM25 indexing.
        Mirrors TradingAgents FinancialSituationMemory pattern.
        """
        votes = agent_state.get("agent_votes", [])
        vote_signals = [v.get("signal", "?") for v in votes]
        mf = agent_state.get("mirofish_prediction") or {}
        kronos = agent_state.get("kronos_prediction") or {}

        return (
            f"Pair={agent_state.get('pair', '?')} "
            f"Session={agent_state.get('session', '?')} "
            f"Decision={agent_state.get('final_decision', '?')} "
            f"Confidence={agent_state.get('confidence', 0.0):.2f} "
            f"MiroFish={mf.get('bullish_probability', 'N/A')} "
            f"Kronos={kronos.get('direction', 'N/A')} "
            f"Votes={vote_signals}"
        )

    def get_recent_accuracy(self, n: int = 10) -> Dict[str, float]:
        """
        Compute average agent accuracy over last N reflections.
        """
        recent = self.reflection_history[-n:]
        if not recent:
            return {}
        keys = ["analyst", "sentiment", "risk", "mirofish", "kronos"]
        totals = {k: 0.0 for k in keys}
        counts = {k: 0 for k in keys}
        for r in recent:
            acc = r.get("agent_accuracy", {})
            for k in keys:
                if k in acc:
                    totals[k] += float(acc[k])
                    counts[k] += 1
        return {k: round(totals[k] / counts[k], 3) if counts[k] else 0.5 for k in keys}
