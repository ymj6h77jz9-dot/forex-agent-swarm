"""
REFLECTION ENGINE — Integrated from ymj6h77jz9-dot/TradingAgents
-----------------------------------------------------------------
After each trade closes, the Reflector reviews the full decision chain,
identifies what was correct/incorrect, and generates improvement lessons
that are stored in the MemPalace memory system for future cycles.

Source: TradingAgents/graph/reflection.py
Enhanced with: MemPalace memory storage, MiroFish accuracy tracking
"""

import json
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

REFLECTION_SYSTEM_PROMPT = """
You are an expert forex trading analyst reviewing a completed trade decision cycle.
Your goal is to extract actionable lessons that will improve future trading decisions.

You will receive:
1. The full agent decision chain (votes, reasoning, state)
2. The actual trade outcome (profit/loss, price movement)

Your analysis must cover:
1. REASONING: Was each agent's vote correct? Why or why not?
   - Analyst (technical): Did the indicators align with actual movement?
   - Sentiment (news/email): Was the news signal accurate?
   - Risk (position sizing): Was the SL/TP placement optimal?
   - MiroFish (simulation): How accurate was the probabilistic prediction?

2. IMPROVEMENT: For incorrect decisions, what should have been different?
   - Specific indicator weightings
   - News sources that proved reliable/unreliable
   - Session timing lessons
   - Spread/volatility conditions

3. SUMMARY: 2-3 sentence lesson for the memory system.

4. MEMORY_QUERY: A single sentence (max 100 tokens) capturing the core lesson.
   This will be stored in MemPalace for BM25 retrieval in future similar situations.

Return as JSON:
{
  "reasoning": "<detailed analysis>",
  "improvement": "<specific changes>",
  "summary": "<2-3 sentence lesson>",
  "memory_query": "<single sentence for BM25 storage>",
  "agent_accuracy": {
    "analyst": <float 0-1>,
    "sentiment": <float 0-1>,
    "risk": <float 0-1>,
    "mirofish": <float 0-1>
  }
}
"""


class Reflector:
    """
    Post-trade reflection engine.
    Integrated from TradingAgents Reflector — enhanced with MemPalace storage.
    """

    def __init__(self):
        self.reflection_history = []

    async def reflect(self, agent_state: Dict[str, Any],
                       trade_result: Dict[str, Any],
                       actual_direction: str) -> Dict[str, Any]:
        """
        Perform post-trade reflection and return lessons.
        
        Args:
            agent_state:      Serialized AgentState from the trade cycle
            trade_result:     Execution result (status, pnl, price, etc.)
            actual_direction: Actual price movement: "UP" | "DOWN" | "FLAT"
        """
        prompt = f"""
=== AGENT DECISION STATE ===
{json.dumps(agent_state, indent=2)}

=== TRADE RESULT ===
{json.dumps(trade_result, indent=2)}

=== ACTUAL MARKET OUTCOME ===
Price moved: {actual_direction}
PnL: {trade_result.get('pnl', 'N/A')}

Analyze this trade cycle and return your structured reflection.
"""
        try:
            resp = await client.chat.completions.create(
                model    = "gpt-4o",
                messages = [
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format = {"type": "json_object"},
                temperature     = 0.3,
            )
            reflection = json.loads(resp.choices[0].message.content)
            self.reflection_history.append({
                "cycle_id":  agent_state.get("cycle_id"),
                "timestamp": agent_state.get("timestamp"),
                **reflection,
            })
            logger.info(f"[Reflector] Reflection complete: {reflection.get('summary', '')[:100]}")
            return reflection

        except Exception as e:
            logger.error(f"[Reflector] Reflection failed: {e}")
            return {"summary": "Reflection error", "memory_query": "", "agent_accuracy": {}}

    def extract_situation_for_memory(self, agent_state: Dict[str, Any]) -> str:
        """
        Extract a searchable situation description for MemPalace BM25 indexing.
        Mirrors TradingAgents FinancialSituationMemory pattern.
        """
        return (
            f"Pair: {agent_state.get('pair')} "
            f"Session: {agent_state.get('session')} "
            f"Decision: {agent_state.get('final_decision')} "
            f"Confidence: {agent_state.get('confidence', 0):.2f} "
            f"MiroFish: {agent_state.get('mirofish', {}).get('consensus', 'N/A')} "
            f"Votes: {[v.get('signal') for v in agent_state.get('agent_votes', [])]}"
        )
