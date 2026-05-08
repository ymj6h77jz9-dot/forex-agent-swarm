"""
PROPAGATION ENGINE — Integrated from ymj6h77jz9-dot/TradingAgents
------------------------------------------------------------------
Handles state initialization and update propagation through the agent graph.
Every agent reads from and writes back to a shared AgentState dict,
enabling true information flow across the swarm.

Source: TradingAgents/graph/propagation.py
Enhanced with: forex-specific state, MiroFish integration, Deriv market state
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class InvestDebateState:
    """Bull vs Bear debate state for investment decision."""
    bull_history:     str = ""
    bear_history:     str = ""
    history:          str = ""
    current_response: str = ""
    judge_decision:   str = ""
    count:            int = 0


@dataclass
class RiskDebateState:
    """Aggressive vs Conservative risk debate state."""
    aggressive_history:          str = ""
    conservative_history:        str = ""
    neutral_history:             str = ""
    history:                     str = ""
    latest_speaker:              str = ""
    current_aggressive_response: str = ""
    current_conservative_response: str = ""
    current_neutral_response:    str = ""
    judge_decision:              str = ""
    count:                       int = 0


@dataclass
class AgentState:
    """
    Master shared state propagated through the entire swarm.
    Every agent reads this and appends their output to the relevant field.
    """
    # Market context
    pair:          str = ""
    bid:           float = 0.0
    ask:           float = 0.0
    spread:        float = 0.0
    atr:           float = 0.0
    session:       str = ""
    timestamp:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Agent reports (populated as agents run)
    market_report:      str = ""
    news_report:        str = ""
    sentiment_report:   str = ""
    fundamentals_report: str = ""
    risk_report:        str = ""
    mirofish_prediction: Dict[str, Any] = field(default_factory=dict)

    # Debate states
    investment_debate_state: InvestDebateState = field(default_factory=InvestDebateState)
    risk_debate_state:       RiskDebateState   = field(default_factory=RiskDebateState)

    # Final decision
    investment_plan: str = ""
    final_decision:  str = ""   # "BUY" | "SELL" | "HOLD"
    confidence:      float = 0.0
    lot_size:        float = 0.01
    stop_loss:       float = 0.0
    take_profit:     float = 0.0

    # Agent votes (raw)
    agent_votes: List[Dict[str, Any]] = field(default_factory=list)

    # Memory context
    past_memories: List[str] = field(default_factory=list)

    # Audit
    cycle_id:  str = ""
    messages:  List[Any] = field(default_factory=list)


class Propagator:
    """
    Manages state propagation through the agent graph.
    
    Integrated from TradingAgents Propagator — extended for forex swarm
    with MiroFish prediction state and Deriv market data.
    """

    def __init__(self, max_recur_limit: int = 100):
        self.max_recur_limit = max_recur_limit

    def create_initial_state(self, pair: str, bid: float, ask: float,
                              atr: float, session: str) -> AgentState:
        """
        Create a fresh AgentState for a new market event cycle.
        This is broadcast to all sub-agents at the start of each cycle.
        """
        import uuid
        return AgentState(
            pair      = pair,
            bid       = bid,
            ask       = ask,
            spread    = round((ask - bid) * 10000, 2),
            atr       = atr,
            session   = session,
            timestamp = datetime.now(timezone.utc).isoformat(),
            cycle_id  = str(uuid.uuid4())[:8],
            messages  = [("system", f"New market event: {pair} @ {ask}")]
        )

    def propagate_agent_output(self, state: AgentState, agent_name: str,
                                output: Dict[str, Any]) -> AgentState:
        """
        Apply an agent's output to the shared state.
        This is how information flows from sub-agents back to the orchestrator.
        """
        field_map = {
            "analyst":   "market_report",
            "sentiment": "sentiment_report",
            "risk":      "risk_report",
            "news":      "news_report",
        }

        if agent_name in field_map:
            setattr(state, field_map[agent_name], output.get("reasoning", ""))

        # Always append the raw vote
        state.agent_votes.append({
            "agent":      agent_name,
            "signal":     output.get("signal", "FLAT"),
            "confidence": output.get("confidence", 0.0),
            "reasoning":  output.get("reasoning", ""),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        })

        return state

    def propagate_mirofish(self, state: AgentState, prediction) -> AgentState:
        """Inject MiroFish prediction results into state."""
        state.mirofish_prediction = {
            "bullish":    prediction.bullish_probability,
            "bearish":    prediction.bearish_probability,
            "neutral":    prediction.neutral_probability,
            "consensus":  prediction.particle_consensus,
            "confidence": prediction.confidence_score,
            "high":       prediction.expected_high,
            "low":        prediction.expected_low,
        }
        return state

    def propagate_final_decision(self, state: AgentState, decision: str,
                                  confidence: float, lot_size: float,
                                  sl: float, tp: float) -> AgentState:
        """Propagate the orchestrator's final decision to all downstream agents."""
        state.final_decision = decision
        state.confidence     = confidence
        state.lot_size       = lot_size
        state.stop_loss      = sl
        state.take_profit    = tp
        return state

    def get_graph_config(self) -> Dict[str, Any]:
        """Config dict for LangGraph invocation."""
        return {
            "recursion_limit": self.max_recur_limit,
            "stream_mode":     "values",
        }

    def serialize(self, state: AgentState) -> Dict[str, Any]:
        """Convert state to JSON-serializable dict for logging/persistence."""
        return {
            "cycle_id":      state.cycle_id,
            "pair":          state.pair,
            "timestamp":     state.timestamp,
            "final_decision": state.final_decision,
            "confidence":    state.confidence,
            "agent_votes":   state.agent_votes,
            "mirofish":      state.mirofish_prediction,
            "sl":            state.stop_loss,
            "tp":            state.take_profit,
            "lot_size":      state.lot_size,
        }
