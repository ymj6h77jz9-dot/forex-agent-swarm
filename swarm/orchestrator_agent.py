import logging

logger = logging.getLogger(__name__)

"""
ORCHESTRATOR AGENT — Forex Agentic Swarm
-----------------------------------------
Master controller. Receives market state, broadcasts to sub-agents,
aggregates votes, applies weighted consensus, approves/vetoes trades,
monitors global drawdown limits.
"""

import asyncio
import json
from typing import Dict, List
from dataclasses import dataclass, field

# ── Agent weights (must sum to 1.0) ──────────────────────────────────────────
AGENT_WEIGHTS = {
    "analyst":   0.40,
    "sentiment": 0.30,
    "risk":      0.30,
}

CONSENSUS_THRESHOLD = 0.72       # Minimum weighted score to fire a trade
MAX_DAILY_DRAWDOWN  = 0.05       # 5% max daily drawdown on account equity
MAX_OPEN_TRADES     = 3          # Hard cap on concurrent open positions


@dataclass
class MarketState:
    pair: str
    bid: float
    ask: float
    spread: float
    atr: float                    # Average True Range (14)
    session: str                  # "london" | "ny" | "asia" | "overlap"
    timestamp: str

@dataclass
class AgentVote:
    agent_name: str
    signal: str                   # "BUY" | "SELL" | "FLAT"
    confidence: float             # 0.0 – 1.0
    reasoning: str
    pair: str

@dataclass
class OrchestratorState:
    open_trades: int = 0
    daily_drawdown: float = 0.0
    equity: float = 10_000.0
    daily_pnl: float = 0.0
    trade_log: List[dict] = field(default_factory=list)


class OrchestratorAgent:
    def __init__(self):
        self.state = OrchestratorState()

    # ── Main entry point ──────────────────────────────────────────────────────
    async def process_market_event(self, market_state: MarketState) -> dict:
        logger.info(f"\n[ORCHESTRATOR] Market event: {market_state.pair} @ {market_state.ask}")

        # 1. Hard guards — check before broadcasting
        if not self._pre_flight_checks():
            return {"action": "BLOCKED", "reason": "Pre-flight checks failed"}

        # 2. Broadcast to sub-agents concurrently
        votes = await self._broadcast_and_collect(market_state)

        # 3. Aggregate votes
        decision = self._aggregate_votes(votes)
        logger.info(f"[ORCHESTRATOR] Consensus: {decision}")

        # 4. If approved, propagate to execution
        if decision["action"] == "EXECUTE":
            result = await self._propagate_to_execution(decision, market_state)
            self._update_state(result)
            return result

        return decision

    # ── Pre-flight risk checks ─────────────────────────────────────────────────
    def _pre_flight_checks(self) -> bool:
        if self.state.open_trades >= MAX_OPEN_TRADES:
            logger.info("[ORCHESTRATOR] BLOCKED: Max open trades reached")
            return False
        if self.state.daily_drawdown >= MAX_DAILY_DRAWDOWN:
            logger.info("[ORCHESTRATOR] BLOCKED: Daily drawdown limit hit")
            return False
        return True

    # ── Broadcast market state to all agents ──────────────────────────────────
    async def _broadcast_and_collect(self, market_state: MarketState) -> List[AgentVote]:
        from analyst_agent import AnalystAgent
        from sentiment_agent import SentimentAgent
        from risk_agent import RiskAgent

        agents = {
            "analyst":   AnalystAgent(),
            "sentiment": SentimentAgent(),
            "risk":      RiskAgent(),
        }

        # Fire all agents concurrently (true parallel analysis)
        tasks = {
            name: asyncio.create_task(agent.analyze(market_state))
            for name, agent in agents.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        votes = []
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.info(f"[ORCHESTRATOR] Agent '{name}' errored: {result}")
                # Inject a FLAT vote so it doesn't block consensus
                votes.append(AgentVote(name, "FLAT", 0.0, "Agent error", market_state.pair))
            else:
                votes.append(result)
                logger.info(f"[ORCHESTRATOR] {name} voted: {result.signal} ({result.confidence:.2f})")

        return votes

    # ── Weighted consensus engine ─────────────────────────────────────────────
    def _aggregate_votes(self, votes: List[AgentVote]) -> dict:
        buy_score  = 0.0
        sell_score = 0.0

        for vote in votes:
            w = AGENT_WEIGHTS.get(vote.agent_name, 0.0)
            if vote.signal == "BUY":
                buy_score  += w * vote.confidence
            elif vote.signal == "SELL":
                sell_score += w * vote.confidence

        direction = None
        score     = 0.0

        if buy_score >= CONSENSUS_THRESHOLD:
            direction, score = "BUY", buy_score
        elif sell_score >= CONSENSUS_THRESHOLD:
            direction, score = "SELL", sell_score

        if direction:
            return {
                "action":    "EXECUTE",
                "direction": direction,
                "score":     round(score, 4),
                "votes":     [v.__dict__ for v in votes],
            }

        return {
            "action": "HOLD",
            "buy_score":  round(buy_score, 4),
            "sell_score": round(sell_score, 4),
            "votes":      [v.__dict__ for v in votes],
        }

    # ── Propagate approved trade to Execution Agent ───────────────────────────
    async def _propagate_to_execution(self, decision: dict, market_state: MarketState) -> dict:
        from execution_agent import ExecutionAgent
        exec_agent = ExecutionAgent()
        result = await exec_agent.execute_trade(decision, market_state)
        logger.info(f"[ORCHESTRATOR] Execution result: {result}")
        return result

    # ── Update internal state after trade ─────────────────────────────────────
    def _update_state(self, trade_result: dict):
        if trade_result.get("status") == "FILLED":
            self.state.open_trades += 1
            self.state.trade_log.append(trade_result)
