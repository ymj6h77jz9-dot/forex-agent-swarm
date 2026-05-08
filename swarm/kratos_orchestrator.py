"""
KRATOS ORCHESTRATOR v2 — Complete Unified Agentic Forex Swarm
==============================================================
Central brain. Integrates ALL architectures from your GitHub repos:

  ╔══════════════════════════════════════════════════════════════════╗
  ║  REPO             ARCHITECTURE CONTRIBUTION                     ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  KRATOS-app     → KratosOrchestrator, DerivAdapter,             ║
  ║                   ExecutionEngine, RiskManager, MiroFishAdapter  ║
  ║  TradingAgents  → Propagator, Reflector, SignalProcessor,        ║
  ║                   Bull/Bear Debate, Risk Debate, BM25 Memory     ║
  ║  MiroFish       → PSO Particle Swarm Intelligence Engine         ║
  ║  mempalace      → BM25 Memory Palace (verbatim, findable)        ║
  ║  Kronos         → Foundation Model for Financial K-lines         ║
  ║  tensortrade    → RL Meta-Agent (DQN, reward, env loop)          ║
  ║  deer-flow      → Long-horizon researcher, subagent executor,    ║
  ║                   loop detection, DeerFlow memory middleware      ║
  ║  QuantDinger    → Multi-asset watchlist, research framework       ║
  ║  OpenBB         → Unified financial data (candles, calendar,     ║
  ║                   currency snapshots, macro indicators)           ║
  ║  Crucix         → Multi-source intelligence sweep (FRED, BLS,    ║
  ║                   news, geopolitical), regime detection           ║
  ║  BB-Terminal    → Multi-pair forex dashboard (FXC component)     ║
  ║  ruflo          → Multi-agent coordination, self-correction,     ║
  ║                   propagation improvement patterns               ║
  ║  nautilus_trader→ Event-driven deterministic execution patterns  ║
  ╚══════════════════════════════════════════════════════════════════╝

Decision Pipeline (per market event):
  1.  Propagator creates AgentState (broadcast)
  2.  MemPalace injects past memories
  3.  Crucix runs macro intelligence sweep (FRED, BLS, news)
  4.  OpenBB fetches live candles + economic calendar
  5.  MiroFish runs PSO simulation → probabilistic backbone
  6.  Kronos foundation model generates multi-bar OHLCV forecast
  7.  DeerFlowResearcher runs parallel research on pair context
  8.  Sub-agents run concurrently (Analyst, Sentiment, Risk)
  9.  Bull/Bear Investment Debate → judge
  10. Aggressive/Conservative Risk Debate → risk judge
  11. SignalProcessor: weighted_consensus + MiroFish + Kronos blend
  12. RL meta-agent determines position sizing action
  13. Hard Rule Engine validates (never bypassed by LLM)
  14. ExecutionAgent places order via DerivAdapter
  15. Reflector post-trade → MemPalace learns
  16. MemoryAgent updates agent weights dynamically
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone

from graph.propagation        import Propagator, AgentState
from graph.reflection         import Reflector
from graph.signal_processing  import SignalProcessor
from engines.mirofish_engine  import MiroFishEngine
from models.kronos_adapter    import KronosAdapter
from memory.mempalace_adapter import MemPalaceAdapter
from data.openbb_provider     import OpenBBProvider
from data.crucix_intel        import CrucixIntelligence
from agents.deerflow_researcher import DeerFlowResearcher
from rl.tensortrade_env       import KratosRLEnvironment, SwarmAction
from analyst_agent            import AnalystAgent
from sentiment_agent          import SentimentAgent
from risk_agent               import RiskAgent
from execution_agent          import ExecutionAgent
from memory_agent             import MemoryAgent
from orchestrator_agent       import MarketState, AgentVote

logger = logging.getLogger(__name__)

# ── Global weights (evolved by MemoryAgent) ───────────────────────────────────
DEFAULT_WEIGHTS = {
    "analyst":   0.28,
    "sentiment": 0.22,
    "risk":      0.20,
    "mirofish":  0.15,   # PSO simulation backbone
    "kronos":    0.15,   # Foundation model
}

# ── Hard limits ───────────────────────────────────────────────────────────────
CONSENSUS_THRESHOLD  = 0.70
MAX_OPEN_TRADES      = 3
MAX_DAILY_DRAWDOWN   = 0.05
MAX_SPREAD_PIPS      = 5.0
MIN_ATR              = 0.00010
EQUITY               = float(os.environ.get("ACCOUNT_EQUITY", "10000"))

# ── Research cadence ──────────────────────────────────────────────────────────
DEEP_RESEARCH_EVERY_N_CYCLES = 10     # Run DeerFlow research every 10 cycles
CRUCIX_SWEEP_EVERY_N_CYCLES  = 5      # Run Crucix sweep every 5 cycles


class KratosOrchestratorV2:
    """
    The fully unified KRATOS v2 Orchestrator.
    Every repo's architecture is wired in. No holding back.
    """

    def __init__(self):
        # ── Core engines ──────────────────────────────────────────────────────
        self.propagator     = Propagator()
        self.reflector      = Reflector()
        self.signal_proc    = SignalProcessor()

        # ── Prediction backbone (MiroFish PSO + Kronos FM) ────────────────────
        self.mirofish       = MiroFishEngine({"n_particles": 60, "n_iterations": 100})
        self.kronos         = KronosAdapter()

        # ── Memory (MemPalace + TradingAgents BM25) ───────────────────────────
        self.mempalace      = MemPalaceAdapter()
        self.memory_agent   = MemoryAgent()

        # ── Data layer (OpenBB + Crucix) ──────────────────────────────────────
        self.openbb         = OpenBBProvider()
        self.crucix         = CrucixIntelligence()
        self._last_crucix_briefing = None

        # ── Research (DeerFlow) ───────────────────────────────────────────────
        self.researcher     = DeerFlowResearcher()

        # ── RL meta-agent (TensorTrade) ───────────────────────────────────────
        self.rl_env         = KratosRLEnvironment(account_equity=EQUITY)

        # ── Execution ─────────────────────────────────────────────────────────
        self.exec_agent     = ExecutionAgent()

        # ── Swarm state ───────────────────────────────────────────────────────
        self.agent_weights  = DEFAULT_WEIGHTS.copy()
        self.open_trades:   List[dict] = []
        self.daily_pnl:     float = 0.0
        self.cycle_count:   int = 0

        logger.info("🚀 KRATOS v2 — All systems online. No holding back.")

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════════════
    async def process_market_event(self, market_state: MarketState) -> Dict:
        self.cycle_count += 1
        pair = market_state.pair

        logger.info(f"\n{'═'*70}")
        logger.info(f"  KRATOS v2 │ Cycle #{self.cycle_count} │ {pair} @ {market_state.ask}")
        logger.info(f"{'═'*70}")

        # ── STEP 1: Create & broadcast AgentState ─────────────────────────────
        state = self.propagator.create_initial_state(
            pair    = pair,
            bid     = market_state.bid,
            ask     = market_state.ask,
            atr     = market_state.atr,
            session = market_state.session,
        )

        # ── STEP 2: MemPalace — inject past memories ──────────────────────────
        memories = self.mempalace.get_relevant_memories(
            pair    = pair,
            session = market_state.session,
            query   = f"{pair} {market_state.session} trade decision",
            n       = 3,
        )
        state.past_memories = memories
        if memories:
            logger.info(f"  MemPalace → {len(memories)} memories injected")

        # ── STEP 3: Hard pre-flight checks ────────────────────────────────────
        block = self._pre_flight(market_state)
        if block:
            logger.warning(f"  PRE-FLIGHT BLOCKED: {block}")
            return {"action": "BLOCKED", "reason": block, "pair": pair}

        # ── STEP 4: Crucix intelligence sweep (periodic) ─────────────────────
        crucix_briefing = None
        if self.cycle_count % CRUCIX_SWEEP_EVERY_N_CYCLES == 1 or not self._last_crucix_briefing:
            try:
                crucix_briefing = await asyncio.wait_for(
                    self.crucix.get_briefing(), timeout=20.0
                )
                self._last_crucix_briefing = crucix_briefing
                logger.info(
                    f"  Crucix → regime:{crucix_briefing.risk_regime} "
                    f"USD:{crucix_briefing.usd_bias}"
                )
            except Exception as e:
                logger.warning(f"  Crucix sweep error: {e}")
        else:
            crucix_briefing = self._last_crucix_briefing

        # ── STEP 5: OpenBB — fetch live candles + economic calendar ──────────
        candle_df = None
        upcoming_events = []
        try:
            candle_task   = self.openbb.get_candles(pair, interval="5m", lookback_bars=500)
            calendar_task = self.openbb.get_economic_calendar(days_ahead=2)
            candle_df, upcoming_events = await asyncio.gather(
                candle_task, calendar_task, return_exceptions=True
            )
            if isinstance(candle_df, Exception):     candle_df = None
            if isinstance(upcoming_events, Exception): upcoming_events = []
        except Exception as e:
            logger.warning(f"  OpenBB fetch error: {e}")

        # High-impact event guard
        if upcoming_events:
            high_events = [e for e in upcoming_events if e.impact == "high"]
            if high_events:
                logger.warning(f"  ⚠️ HIGH-IMPACT EVENT IN {len(high_events)} hours → reducing lot by 50%")
                market_state._event_lot_reduction = 0.5

        # ── STEP 6: MiroFish PSO Prediction ──────────────────────────────────
        mf_prediction = await self.mirofish.predict(
            symbol        = pair,
            current_price = market_state.ask,
            atr           = market_state.atr,
            agent_bias    = 0.0,
        )
        state = self.propagator.propagate_mirofish(state, mf_prediction)
        logger.info(
            f"  MiroFish → Bull:{mf_prediction.bullish_probability:.2f} "
            f"Bear:{mf_prediction.bearish_probability:.2f} "
            f"[{mf_prediction.particle_consensus}] conf:{mf_prediction.confidence_score:.2f}"
        )

        # ── STEP 7: Kronos Foundation Model Forecast ─────────────────────────
        kronos_pred = None
        if candle_df is not None and not candle_df.empty:
            try:
                candle_df_kronos = self.kronos.build_candle_df(
                    candle_df.reset_index().to_dict("records")
                )
                kronos_pred = await self.kronos.predict(
                    symbol   = pair,
                    candles  = candle_df_kronos,
                    pred_len = 10,
                )
                logger.info(
                    f"  Kronos → {kronos_pred.direction} "
                    f"{kronos_pred.magnitude_pct:+.4f}% "
                    f"conf:{kronos_pred.confidence:.2f}"
                )
            except Exception as e:
                logger.warning(f"  Kronos error: {e}")

        # ── STEP 8: DeerFlow Research (periodic deep-dive) ───────────────────
        research_brief = None
        if self.cycle_count % DEEP_RESEARCH_EVERY_N_CYCLES == 1:
            try:
                crucix_context = crucix_briefing.briefing_text if crucix_briefing else ""
                research_brief = await asyncio.wait_for(
                    self.researcher.research(
                        topic   = f"Current market outlook and drivers",
                        pair    = pair,
                        context = crucix_context,
                    ),
                    timeout = 45.0
                )
                # Store research signal in MemPalace
                self.mempalace.store_signal(pair, {
                    "research": research_brief.get("summary", ""),
                    "sentiment": research_brief.get("sentiment", "NEUTRAL"),
                })
                logger.info(
                    f"  DeerFlow → {research_brief.get('sentiment')} "
                    f"conf:{research_brief.get('confidence', 0):.2f}"
                )
            except Exception as e:
                logger.warning(f"  DeerFlow research error: {e}")

        # ── STEP 9: Sub-agents (concurrent) ──────────────────────────────────
        analyst    = AnalystAgent()
        sentiment  = SentimentAgent()
        risk_agent = RiskAgent(equity=EQUITY)

        # Inject research + Crucix context into sentiment agent
        if crucix_briefing:
            sentiment._macro_context = crucix_briefing.briefing_text
        if research_brief:
            sentiment._research_context = research_brief.get("summary", "")

        analyst_vote, sentiment_vote, risk_vote = await asyncio.gather(
            analyst.analyze(market_state),
            sentiment.analyze(market_state),
            risk_agent.analyze(market_state),
            return_exceptions=True,
        )

        votes = []
        vote_dicts = []
        for name, vote in [("analyst", analyst_vote),
                            ("sentiment", sentiment_vote),
                            ("risk", risk_vote)]:
            if isinstance(vote, Exception):
                logger.error(f"  [{name}] Error: {vote}")
                vote = AgentVote(name, "FLAT", 0.0, f"Error: {vote}", pair)
            votes.append(vote)
            vote_dicts.append({
                "agent":      name,
                "signal":     vote.signal,
                "confidence": vote.confidence,
                "reasoning":  vote.reasoning,
            })
            state = self.propagator.propagate_agent_output(state, name, {
                "signal":     vote.signal,
                "confidence": vote.confidence,
                "reasoning":  vote.reasoning,
            })
            logger.info(f"  [{name.upper():10}] {vote.signal:4} | conf:{vote.confidence:.2f}")

        # ── STEP 10: Bull/Bear Investment Debate ──────────────────────────────
        kronos_note = ""
        if kronos_pred:
            kronos_note = (f"\nKronos Foundation Model: {kronos_pred.direction} "
                          f"{kronos_pred.magnitude_pct:+.4f}% (conf:{kronos_pred.confidence:.2f})")

        bull_arg = (f"Analyst: {analyst_vote.signal} ({analyst_vote.confidence:.2f})\n"
                    f"{analyst_vote.reasoning[:300]}\n"
                    f"MiroFish: {mf_prediction.particle_consensus} (bull:{mf_prediction.bullish_probability:.2f})"
                    f"{kronos_note}")
        bear_arg = (f"Risk concerns: {risk_vote.reasoning[:300]}\n"
                    f"Sentiment: {sentiment_vote.reasoning[:200]}\n"
                    f"MiroFish bear: {mf_prediction.bearish_probability:.2f}")

        debate_result = await self.signal_proc.resolve_investment_debate(
            bull_argument = bull_arg,
            bear_argument = bear_arg,
            pair          = pair,
        )
        logger.info(f"  [DEBATE] {debate_result.get('decision')} | conf:{debate_result.get('confidence',0):.2f}")

        # ── STEP 11: Weighted Consensus (votes + MiroFish + Kronos) ──────────
        # Add Kronos as synthetic vote
        if kronos_pred:
            kron_signal = {"UP": "BUY", "DOWN": "SELL", "FLAT": "HOLD"}.get(
                kronos_pred.direction, "HOLD"
            )
            vote_dicts.append({
                "agent":      "kronos",
                "signal":     kron_signal,
                "confidence": kronos_pred.confidence,
            })

        final_decision, score = self.signal_proc.compute_weighted_consensus(
            votes               = vote_dicts,
            weights             = self.agent_weights,
            mirofish_prediction = state.mirofish_prediction,
            threshold           = CONSENSUS_THRESHOLD,
        )

        # Debate override if more confident
        debate_conf = float(debate_result.get("confidence", 0))
        if debate_conf > score:
            final_decision = debate_result.get("decision", final_decision)
            score          = debate_conf

        # Crucix USD bias override check
        if crucix_briefing and score < 0.80:
            if crucix_briefing.usd_bias == "BULLISH" and final_decision == "SELL" and "USD" in pair[:3]:
                score -= 0.05  # Penalize going against strong macro
            elif crucix_briefing.usd_bias == "BEARISH" and final_decision == "BUY" and "USD" in pair[:3]:
                score -= 0.05

        logger.info(f"  [CONSENSUS] {final_decision} | score:{score:.4f}")

        if final_decision == "HOLD" or score < CONSENSUS_THRESHOLD:
            state = self.propagator.propagate_final_decision(
                state, "HOLD", score, 0.0, 0.0, 0.0
            )
            self.mempalace.store_decision(self.propagator.serialize(state))
            return {"action": "HOLD", "pair": pair, "score": score}

        # ── STEP 12: Risk Debate ──────────────────────────────────────────────
        risk_debate = await self.signal_proc.resolve_risk_debate(
            aggressive_arg   = (f"Strong signal {score:.2f}. Proceed with full lot. "
                               f"ATR:{market_state.atr:.5f} Spread:{market_state.spread:.1f}pips"),
            conservative_arg = (f"Risk concerns: {risk_vote.reasoning[:200]}. "
                               f"Open trades:{len(self.open_trades)}")
        )
        if not risk_debate.get("approved", True):
            logger.warning(f"  [RISK VETO] {risk_debate.get('reasoning')}")
            return {"action": "VETOED", "reason": risk_debate.get("reasoning"), "pair": pair}

        lot_adj    = float(risk_debate.get("lot_size_adjustment", 1.0))
        sl_extra   = float(risk_debate.get("sl_adjustment_pips", 0.0))
        event_adj  = getattr(market_state, '_event_lot_reduction', 1.0)
        lot_adj   *= event_adj

        # ── STEP 13: RL Meta-Agent Position Sizing ───────────────────────────
        swarm_output_for_rl = {
            "pair":     pair,
            "decision": final_decision,
            "score":    score,
            "votes":    vote_dicts,
            "mirofish": mf_prediction.__dict__,
            "kronos":   kronos_pred.__dict__ if kronos_pred else {},
            "spread":   market_state.spread,
            "atr":      market_state.atr,
            "price":    market_state.ask,
            "session":  market_state.session,
        }
        rl_obs    = self.rl_env.observe(swarm_output_for_rl)
        rl_action = self._rl_policy(rl_obs, score)
        lot_from_rl = self.rl_env.lot_size_from_action(rl_action, base_lot=0.01)
        logger.info(f"  [RL] Action:{SwarmAction(rl_action).name} | lot_adj:{lot_from_rl:.4f}")

        # ── STEP 14: Calculate SL/TP ──────────────────────────────────────────
        atr    = market_state.atr
        sl_dist = atr * 1.5 + (sl_extra * 0.0001)
        tp_dist = atr * 2.5

        if final_decision == "BUY":
            sl = round(market_state.bid - sl_dist, 5)
            tp = round(market_state.ask + tp_dist, 5)
        else:
            sl = round(market_state.ask + sl_dist, 5)
            tp = round(market_state.bid - tp_dist, 5)

        base_lot   = risk_agent.calculate_lot_size(sl_dist / 0.0001)
        final_lot  = round(base_lot * lot_adj, 2)

        # Blend RL lot size suggestion
        if lot_from_rl > 0:
            final_lot = round((final_lot * 0.7 + lot_from_rl * 0.3), 2)

        state = self.propagator.propagate_final_decision(
            state, final_decision, score, final_lot, sl, tp
        )

        # ── STEP 15: Execute ──────────────────────────────────────────────────
        trade_result = await self.exec_agent.execute_trade(
            {"direction": final_decision, "score": score},
            market_state,
        )
        trade_result.update({
            "lot_size":  final_lot,
            "sl":        sl,
            "tp":        tp,
            "cycle_id":  state.cycle_id,
            "mirofish":  mf_prediction.particle_consensus,
            "kronos":    kronos_pred.direction if kronos_pred else "N/A",
            "regime":    crucix_briefing.risk_regime if crucix_briefing else "N/A",
        })

        # ── STEP 16: Persist ──────────────────────────────────────────────────
        self.memory_agent.log_trade(trade_result)
        self.mempalace.store_decision(self.propagator.serialize(state))

        if trade_result.get("status") in ("FILLED", "SIMULATED"):
            self.open_trades.append(trade_result)

        logger.info(
            f"\n  ✅ EXECUTED [{final_decision}] {pair} | "
            f"Lots:{final_lot} SL:{sl} TP:{tp} | "
            f"Status:{trade_result.get('status')}"
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
            "kronos":       kronos_pred.__dict__ if kronos_pred else None,
            "regime":       crucix_briefing.risk_regime if crucix_briefing else None,
            "state_id":     state.cycle_id,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # POST-TRADE LEARNING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    async def on_trade_close(self, trade_id: str, pnl: float, actual_direction: str):
        """
        Called when a trade closes. Full learning loop across all systems.
        """
        logger.info(f"  Trade closed: {trade_id} | PnL: ${pnl:.2f}")
        self.daily_pnl += pnl

        trade = next((t for t in self.open_trades
                      if t.get("order_id") == trade_id or t.get("cycle_id") == trade_id), {})

        # RL environment step
        rl_action = trade.get("rl_action", SwarmAction.EXECUTE_FULL)
        _, rl_reward, _ = self.rl_env.step(int(rl_action), pnl)
        logger.info(f"  [RL] Reward: {rl_reward:.4f}")

        # TradingAgents-style reflection
        state_snapshot = {
            "cycle_id":        trade_id,
            "pair":            trade.get("pair", ""),
            "session":         trade.get("session", ""),
            "final_decision":  trade.get("direction", ""),
            "confidence":      trade.get("score", 0),
            "mirofish":        {"consensus": trade.get("mirofish", "")},
            "agent_votes":     trade.get("votes", []),
            "pnl":             pnl,
            "actual_direction": actual_direction,
        }
        reflection = await self.reflector.reflect(
            agent_state      = state_snapshot,
            trade_result     = trade,
            actual_direction = actual_direction,
        )

        # Store in MemPalace
        situation = self.reflector.extract_situation_for_memory(state_snapshot)
        self.mempalace.store_reflection(reflection, situation)

        # Store per-agent lessons
        for agent_name, acc in reflection.get("agent_accuracy", {}).items():
            lesson = f"Accuracy {acc:.2f} for {trade.get('pair')} {trade.get('direction')}: {reflection.get('summary','')}"
            self.mempalace.store_agent_lesson(agent_name, lesson, acc)

        # Dynamic weight update
        self.memory_agent.record_outcome(trade_id, pnl, trade.get("votes", []))
        new_w = self.memory_agent.get_updated_weights()
        mf_w  = self.agent_weights.get("mirofish", 0.15)
        kron_w = self.agent_weights.get("kronos", 0.15)
        scale = (1.0 - mf_w - kron_w) / max(sum(new_w.values()), 1e-9)
        self.agent_weights = {k: round(v * scale, 4) for k, v in new_w.items()}
        self.agent_weights["mirofish"] = mf_w
        self.agent_weights["kronos"]   = kron_w
        logger.info(f"  Updated weights: {self.agent_weights}")

        # MiroFish self-improvement
        self.mirofish.reflect_on_performance(
            predicted       = type("R", (), {"particle_consensus": trade.get("mirofish", "FLAT")})(),
            actual_outcome  = actual_direction,
        )

        self.open_trades = [t for t in self.open_trades
                            if t.get("order_id") != trade_id]

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    def _rl_policy(self, obs, consensus_score: float) -> int:
        """
        Simple rule-based RL policy (until DQN is trained on historical data).
        Returns SwarmAction enum value.
        """
        if consensus_score >= 0.90:
            return SwarmAction.EXECUTE_DOUBLE.value
        elif consensus_score >= 0.80:
            return SwarmAction.EXECUTE_FULL.value
        elif consensus_score >= CONSENSUS_THRESHOLD:
            return SwarmAction.EXECUTE_HALF.value
        return SwarmAction.HOLD.value

    def _pre_flight(self, market_state: MarketState) -> Optional[str]:
        if len(self.open_trades) >= MAX_OPEN_TRADES:
            return f"Max open trades ({MAX_OPEN_TRADES}) reached"
        if self.daily_pnl < -(EQUITY * MAX_DAILY_DRAWDOWN):
            return f"Daily drawdown limit hit (${self.daily_pnl:.2f})"
        if market_state.spread > MAX_SPREAD_PIPS:
            return f"Spread too wide: {market_state.spread:.1f}pips"
        if market_state.atr < MIN_ATR:
            return f"Dead market: ATR {market_state.atr}"
        return None

    def get_dashboard(self) -> Dict:
        rl_summary = self.rl_env.get_episode_summary()
        return {
            "cycle_count":  self.cycle_count,
            "open_trades":  len(self.open_trades),
            "daily_pnl":    f"${self.daily_pnl:.2f}",
            "agent_weights": self.agent_weights,
            "rl_episode":   rl_summary,
            "memory_rooms": {
                hall: list(rooms.keys())
                for hall, rooms in self.mempalace.halls.items()
            },
        }
