"""
KRATOS ORCHESTRATOR v2 — Fully Integrated Agentic Forex Swarm
==============================================================
This is the central brain. It integrates ALL architectures pulled from your repos:

  ┌─────────────────────────────────────────────────────────────────┐
  │ REPOS INTEGRATED                                                │
  │  • KRATOS-app        → KratosOrchestrator, DerivAdapter,        │
  │                         ExecutionEngine, RiskManager            │
  │  • TradingAgents     → Propagator, Reflector, SignalProcessor,  │
  │                         Bull/Bear Debate, Risk Debate,           │
  │                         FinancialSituationMemory                │
  │  • MiroFish          → PSO Prediction Engine (backbone)         │
  │  • mempalace         → BM25 Memory Palace (verbatim storage)    │
  │  • ruflo             → Multi-agent coordination patterns        │
  │  • QuantDinger       → Multi-agent research framework           │
  │  • nautilus_trader   → Event-driven architecture patterns       │
  │  • Kronos            → Foundation model for financial language  │
  └─────────────────────────────────────────────────────────────────┘

Decision Pipeline (per market event):
  1. Propagator creates AgentState (broadcast)
  2. MemPalace injects past memories into state
  3. MiroFish runs PSO simulation → probabilistic backbone
  4. Sub-agents run concurrently (Analyst, Sentiment, Risk, News)
  5. Bull/Bear debate → investment judge
  6. Aggressive/Conservative risk debate → risk judge
  7. SignalProcessor computes weighted consensus + MiroFish blend
  8. Hard Rule Engine validates (never bypassed by LLM)
  9. ExecutionAgent places order via DerivAdapter
  10. Reflector runs post-trade analysis → MemPalace learns
  11. MemoryAgent updates agent weights dynamically
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone

from graph.propagation    import Propagator, AgentState
from graph.reflection     import Reflector
from graph.signal_processing import SignalProcessor
from engines.mirofish_engine import MiroFishEngine
from memory.mempalace_adapter import MemPalaceAdapter
from analyst_agent        import AnalystAgent
from sentiment_agent      import SentimentAgent
from risk_agent           import RiskAgent
from execution_agent      import ExecutionAgent
from memory_agent         import MemoryAgent
from orchestrator_agent   import MarketState

logger = logging.getLogger(__name__)

# ── Global weights (updated dynamically by MemoryAgent) ──────────────────────
DEFAULT_WEIGHTS = {
    "analyst":   0.35,
    "sentiment": 0.25,
    "risk":      0.25,
    "mirofish":  0.15,   # MiroFish PSO backbone weight
}

CONSENSUS_THRESHOLD = 0.70
MAX_OPEN_TRADES     = 3
MAX_DAILY_DRAWDOWN  = 0.05   # 5%
EQUITY              = float(os.environ.get("ACCOUNT_EQUITY", "10000"))


class KratosOrchestratorV2:
    """
    Unified KRATOS v2 Orchestrator.
    The complete integration of all GitHub repo architectures.
    """

    def __init__(self):
        # Core engines
        self.propagator      = Propagator()
        self.reflector       = Reflector()
        self.signal_proc     = SignalProcessor()
        self.mirofish        = MiroFishEngine({"n_particles": 50, "n_iterations": 80})
        self.mempalace       = MemPalaceAdapter()
        self.memory_agent    = MemoryAgent()
        self.exec_agent      = ExecutionAgent()

        # Swarm state
        self.agent_weights   = DEFAULT_WEIGHTS.copy()
        self.open_trades:    List[dict] = []
        self.daily_pnl:      float = 0.0
        self.cycle_count:    int = 0

        logger.info("🚀 KRATOS v2 Orchestrator initialized — all systems online")

    # ── Main Entry Point ─────────────────────────────────────────────────────
    async def process_market_event(self, market_state: MarketState) -> Dict:
        self.cycle_count += 1
        pair = market_state.pair
        logger.info(f"\n{'='*60}")
        logger.info(f"[KRATOS] Cycle #{self.cycle_count} | {pair} @ {market_state.ask}")

        # ── STEP 1: Create & broadcast AgentState ─────────────────────────────
        state = self.propagator.create_initial_state(
            pair    = pair,
            bid     = market_state.bid,
            ask     = market_state.ask,
            atr     = market_state.atr,
            session = market_state.session,
        )

        # ── STEP 2: Inject MemPalace memories ────────────────────────────────
        memories = self.mempalace.get_relevant_memories(
            pair    = pair,
            session = market_state.session,
            query   = f"{pair} {market_state.session} trading signal",
            n       = 3,
        )
        state.past_memories = memories
        if memories:
            logger.info(f"[KRATOS] Loaded {len(memories)} past memories from MemPalace")

        # ── STEP 3: Hard pre-flight checks ────────────────────────────────────
        block = self._pre_flight(market_state)
        if block:
            logger.warning(f"[KRATOS] BLOCKED: {block}")
            return {"action": "BLOCKED", "reason": block, "pair": pair}

        # ── STEP 4: MiroFish PSO Prediction (backbone) ────────────────────────
        agent_bias = 0.0   # Will be updated after agent votes
        mf_prediction = await self.mirofish.predict(
            symbol        = pair,
            current_price = market_state.ask,
            atr           = market_state.atr,
            agent_bias    = agent_bias,
        )
        state = self.propagator.propagate_mirofish(state, mf_prediction)
        logger.info(
            f"[MiroFish] Bull:{mf_prediction.bullish_probability:.2f} "
            f"Bear:{mf_prediction.bearish_probability:.2f} "
            f"→ {mf_prediction.particle_consensus}"
        )

        # ── STEP 5: Fire all sub-agents concurrently ──────────────────────────
        analyst   = AnalystAgent()
        sentiment = SentimentAgent()
        risk_agent = RiskAgent(equity=EQUITY)

        analyst_vote, sentiment_vote, risk_vote = await asyncio.gather(
            analyst.analyze(market_state),
            sentiment.analyze(market_state),
            risk_agent.analyze(market_state),
            return_exceptions=True,
        )

        votes = []
        for name, vote in [("analyst", analyst_vote),
                            ("sentiment", sentiment_vote),
                            ("risk", risk_vote)]:
            if isinstance(vote, Exception):
                logger.error(f"[KRATOS] {name} agent error: {vote}")
                from orchestrator_agent import AgentVote
                vote = AgentVote(name, "FLAT", 0.0, f"Error: {vote}", pair)
            votes.append(vote)
            state = self.propagator.propagate_agent_output(state, name, {
                "signal":     vote.signal,
                "confidence": vote.confidence,
                "reasoning":  vote.reasoning,
            })
            logger.info(f"[{name.upper()}] {vote.signal} ({vote.confidence:.2f})")

        # ── STEP 6: Bull/Bear Debate ───────────────────────────────────────────
        bull_arg = f"Signal: {analyst_vote.signal}\n{analyst_vote.reasoning}\n{mf_prediction.particle_consensus} (MiroFish)"
        bear_arg = f"Risk concerns: {risk_vote.reasoning}\nSentiment: {sentiment_vote.reasoning}"

        debate_result = await self.signal_proc.resolve_investment_debate(
            bull_argument = bull_arg,
            bear_argument = bear_arg,
            pair          = pair,
        )
        logger.info(f"[Debate] Judge ruled: {debate_result.get('decision')} ({debate_result.get('confidence', 0):.2f})")

        # ── STEP 7: Weighted Consensus + MiroFish Blend ───────────────────────
        vote_dicts = [{"agent": v.agent_name, "signal": v.signal, "confidence": v.confidence}
                      for v in votes]

        final_decision, score = self.signal_proc.compute_weighted_consensus(
            votes              = vote_dicts,
            weights            = self.agent_weights,
            mirofish_prediction = state.mirofish_prediction,
            threshold          = CONSENSUS_THRESHOLD,
        )

        # Override with debate result if debate is more confident
        debate_conf = float(debate_result.get("confidence", 0))
        if debate_conf > score:
            final_decision = debate_result.get("decision", final_decision)
            score          = debate_conf

        logger.info(f"[KRATOS] Final consensus: {final_decision} (score: {score:.3f})")

        if final_decision == "HOLD" or score < CONSENSUS_THRESHOLD:
            state = self.propagator.propagate_final_decision(
                state, "HOLD", score, 0.0, 0.0, 0.0
            )
            self.mempalace.store_decision(self.propagator.serialize(state))
            return {"action": "HOLD", "pair": pair, "score": score, "state": state}

        # ── STEP 8: Risk Debate (Aggressive vs Conservative) ──────────────────
        risk_debate = await self.signal_proc.resolve_risk_debate(
            aggressive_arg  = f"Proceed with full lot. Score {score:.2f} above threshold. ATR: {market_state.atr}",
            conservative_arg = f"Risk concerns: {risk_vote.reasoning}. Spread: {market_state.spread}",
        )
        if not risk_debate.get("approved", True):
            logger.warning(f"[Risk Judge] Trade VETOED: {risk_debate.get('reasoning')}")
            return {"action": "VETOED", "reason": risk_debate.get("reasoning"), "pair": pair}

        lot_adjustment = float(risk_debate.get("lot_size_adjustment", 1.0))
        sl_extra_pips  = float(risk_debate.get("sl_adjustment_pips", 0.0))

        # ── STEP 9: Execute Trade ──────────────────────────────────────────────
        atr = market_state.atr
        sl_dist = atr * 1.5 + (sl_extra_pips * 0.0001)
        tp_dist = atr * 2.5

        if final_decision == "BUY":
            sl = round(market_state.bid - sl_dist, 5)
            tp = round(market_state.ask + tp_dist, 5)
        else:
            sl = round(market_state.ask + sl_dist, 5)
            tp = round(market_state.bid - tp_dist, 5)

        base_lot  = risk_agent.calculate_lot_size(sl_dist / 0.0001)
        final_lot = round(base_lot * lot_adjustment, 2)

        state = self.propagator.propagate_final_decision(
            state, final_decision, score, final_lot, sl, tp
        )

        trade_result = await self.exec_agent.execute_trade(
            {"direction": final_decision, "score": score},
            market_state,
        )
        trade_result["lot_size"] = final_lot
        trade_result["sl"]       = sl
        trade_result["tp"]       = tp
        trade_result["cycle_id"] = state.cycle_id

        # ── STEP 10: Persist & Log ────────────────────────────────────────────
        self.memory_agent.log_trade(trade_result)
        self.mempalace.store_decision(self.propagator.serialize(state))

        if trade_result.get("status") in ("FILLED", "SIMULATED"):
            self.open_trades.append(trade_result)

        logger.info(
            f"✅ [{final_decision}] {pair} | Lots:{final_lot} | "
            f"SL:{sl} | TP:{tp} | Status:{trade_result.get('status')}"
        )

        return {
            "action":       "EXECUTE",
            "decision":     final_decision,
            "score":        score,
            "pair":         pair,
            "lot_size":     final_lot,
            "sl":           sl,
            "tp":           tp,
            "trade_result": trade_result,
            "mirofish":     mf_prediction.__dict__,
            "state_id":     state.cycle_id,
        }

    # ── Post-trade Reflection ─────────────────────────────────────────────────
    async def on_trade_close(self, trade_id: str, pnl: float, actual_direction: str):
        """
        Called when a trade closes. Triggers reflection + memory update.
        The learning loop of the swarm.
        """
        logger.info(f"[KRATOS] Trade closed: {trade_id} | PnL: ${pnl:.2f}")
        self.daily_pnl += pnl

        # Find the trade state
        matching = [t for t in self.open_trades if t.get("order_id") == trade_id or
                    t.get("cycle_id") == trade_id]
        trade = matching[0] if matching else {"order_id": trade_id, "pnl": pnl}

        # Update performance stats
        self.memory_agent.record_outcome(
            trade_id  = trade_id,
            pnl       = pnl,
            votes     = trade.get("votes", []),
        )

        # Reflect using TradingAgents Reflector
        state_snapshot = {"cycle_id": trade_id, "pnl": pnl, "actual_direction": actual_direction}
        reflection = await self.reflector.reflect(
            agent_state      = state_snapshot,
            trade_result     = trade,
            actual_direction = actual_direction,
        )

        # Store reflection in MemPalace
        situation = self.reflector.extract_situation_for_memory(state_snapshot)
        self.mempalace.store_reflection(reflection, situation)

        # Dynamically update agent weights from performance history
        new_weights = self.memory_agent.get_updated_weights()
        # Preserve MiroFish weight, renormalize rest
        mf_w = self.agent_weights.get("mirofish", 0.15)
        scale = (1.0 - mf_w) / max(sum(new_weights.values()), 1e-9)
        self.agent_weights = {k: round(v * scale, 4) for k, v in new_weights.items()}
        self.agent_weights["mirofish"] = mf_w
        logger.info(f"[KRATOS] Updated agent weights: {self.agent_weights}")

        # Remove from open trades
        self.open_trades = [t for t in self.open_trades
                            if t.get("order_id") != trade_id]

    # ── Pre-flight Hard Rules ─────────────────────────────────────────────────
    def _pre_flight(self, market_state: MarketState) -> Optional[str]:
        if len(self.open_trades) >= MAX_OPEN_TRADES:
            return f"Max open trades ({MAX_OPEN_TRADES}) reached"
        if self.daily_pnl < -(EQUITY * MAX_DAILY_DRAWDOWN):
            return f"Daily drawdown limit hit (${self.daily_pnl:.2f})"
        if market_state.spread > 5.0:
            return f"Spread too wide: {market_state.spread} pips"
        if market_state.atr < 0.00010:
            return f"Dead market: ATR {market_state.atr}"
        return None

    def get_dashboard(self) -> Dict:
        """Return current swarm status for monitoring."""
        summary = self.memory_agent.get_summary()
        return {
            "cycle_count":  self.cycle_count,
            "open_trades":  len(self.open_trades),
            "daily_pnl":    f"${self.daily_pnl:.2f}",
            "agent_weights": self.agent_weights,
            "performance":  summary,
            "memory_rooms": {
                hall: list(rooms.keys())
                for hall, rooms in self.mempalace.halls.items()
            },
        }
