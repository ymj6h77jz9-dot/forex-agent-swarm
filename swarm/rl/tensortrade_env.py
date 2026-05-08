"""
TENSORTRADE REINFORCEMENT LEARNING ENVIRONMENT
===============================================
Integrates TensorTrade — "An open source reinforcement learning framework 
for training, evaluating, and deploying robust trading agents."

This module wraps the KRATOS swarm decision pipeline as a TensorTrade-style
trading environment, enabling:
  1. Backtesting the swarm on historical data with RL reward signals
  2. Training a DQN agent on top of the swarm's signal outputs
  3. Strategy optimization via simulated episodes

The RL agent acts as a META-AGENT that learns WHEN to trust/override 
the main swarm consensus based on historical outcomes.

Source: ymj6h77jz9-dot/tensortrade
Architecture:
  - TradingEnvironment: state/action/reward loop
  - SimpleProfitReward: PnL-based reward signal
  - SwarmObservation: state vector from agent votes + MiroFish + Kronos
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SwarmAction(int, Enum):
    """Actions the RL meta-agent can take."""
    HOLD          = 0    # Don't trade this signal
    EXECUTE_FULL  = 1    # Execute with full lot size
    EXECUTE_HALF  = 2    # Execute with 50% lot size
    EXECUTE_DOUBLE = 3   # Execute with 2x lot size (when very confident)
    OVERRIDE_VETO = 4    # Override the swarm veto (risky)


@dataclass
class SwarmObservation:
    """
    State vector fed to the RL agent.
    Combines swarm votes, model predictions, and market context.
    """
    # Agent votes
    analyst_signal:    float    # -1 SELL, 0 HOLD, +1 BUY
    analyst_conf:      float
    sentiment_signal:  float
    sentiment_conf:    float
    risk_signal:       float
    risk_conf:         float
    
    # MiroFish
    mf_bullish:        float
    mf_bearish:        float
    mf_confidence:     float
    
    # Kronos
    kronos_direction:  float    # -1/0/+1
    kronos_magnitude:  float    # expected % move
    kronos_confidence: float
    
    # Market context
    spread_pips:       float
    atr_normalized:    float    # ATR / price
    session_encoded:   float    # 0=asia, 0.5=london, 1.0=newyork
    hour_of_day:       float    # 0-23 normalized
    
    # Swarm consensus
    consensus_score:   float    # 0-1
    final_signal:      float    # -1/0/+1
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.analyst_signal, self.analyst_conf,
            self.sentiment_signal, self.sentiment_conf,
            self.risk_signal, self.risk_conf,
            self.mf_bullish, self.mf_bearish, self.mf_confidence,
            self.kronos_direction, self.kronos_magnitude, self.kronos_confidence,
            self.spread_pips, self.atr_normalized, self.session_encoded,
            self.hour_of_day, self.consensus_score, self.final_signal,
        ], dtype=np.float32)

    @property
    def shape(self) -> Tuple[int]:
        return (18,)


class SimpleProfitReward:
    """
    TensorTrade-style reward function.
    Source: tensortrade/env/default/rewards/simple_profit_strategy.py pattern
    
    Rewards: realized PnL + risk-adjusted component
    Penalties: drawdown, overtrading, ignoring high-confidence signals
    """

    def __init__(self, window: int = 10):
        self.window = window
        self.pnl_history: List[float] = []

    def calculate(self, pnl: float, action: SwarmAction,
                   obs: SwarmObservation, was_optimal: bool = False) -> float:
        """
        Calculate step reward.
        
        Args:
            pnl:         Realized PnL for this trade ($)
            action:      Action the RL agent took
            obs:         Observation at decision time
            was_optimal: Whether the oracle (hindsight) says this was the right call
        """
        self.pnl_history.append(pnl)
        
        # Base reward: normalized PnL
        base = pnl / 100.0  # normalize by ~$100 scale

        # Penalize holding when signal was very confident
        if action == SwarmAction.HOLD and obs.consensus_score > 0.85:
            base -= 0.1  # missed opportunity penalty

        # Reward executing when confidence is high
        if action in (SwarmAction.EXECUTE_FULL, SwarmAction.EXECUTE_DOUBLE):
            if obs.consensus_score > 0.80:
                base += 0.05  # confidence alignment bonus

        # Penalize overriding veto — very risky action
        if action == SwarmAction.OVERRIDE_VETO:
            base -= 0.2

        # Sharpe-like component — reward consistency
        if len(self.pnl_history) >= self.window:
            recent = self.pnl_history[-self.window:]
            sharpe = np.mean(recent) / (np.std(recent) + 1e-8)
            base  += sharpe * 0.01

        return float(np.clip(base, -1.0, 1.0))


class KratosRLEnvironment:
    """
    TensorTrade-inspired RL environment for KRATOS swarm.
    
    The RL agent observes the swarm's output and decides HOW to act on it,
    learning optimal position sizing and signal filtering over time.
    
    Integrated from: ymj6h77jz9-dot/tensortrade (env architecture)
    Extended with: KRATOS swarm observation space
    """

    def __init__(self, account_equity: float = 10000.0):
        self.equity          = account_equity
        self.initial_equity  = account_equity
        self.reward_fn       = SimpleProfitReward(window=20)
        self.current_obs: Optional[SwarmObservation] = None
        self.episode_trades: List[Dict[str, Any]] = []
        self.episode_pnl     = 0.0
        self.step_count      = 0
        self.action_space_n  = len(SwarmAction)
        self.obs_shape       = (18,)  # SwarmObservation.shape
        logger.info(f"[RL Env] Initialized | equity=${account_equity} | obs={self.obs_shape} | actions={self.action_space_n}")

    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.equity       = self.initial_equity
        self.episode_pnl  = 0.0
        self.step_count   = 0
        self.episode_trades = []
        self.reward_fn    = SimpleProfitReward(window=20)
        logger.debug("[RL Env] Episode reset")
        return np.zeros(self.obs_shape, dtype=np.float32)

    def observe(self, swarm_output: Dict[str, Any]) -> SwarmObservation:
        """
        Convert KRATOS swarm output dict to RL observation.
        Called by the orchestrator after each swarm cycle.
        """
        votes     = swarm_output.get("votes", [])
        mf        = swarm_output.get("mirofish", {})
        kronos    = swarm_output.get("kronos", {})
        pair      = swarm_output.get("pair", "EURUSD")
        
        signal_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0, "FLAT": 0.0}
        session_map = {"asian": 0.0, "london": 0.5, "newyork": 1.0, "overlap": 0.75}
        
        def get_vote(agent_name: str) -> Tuple[float, float]:
            v = next((v for v in votes if v.get("agent") == agent_name), {})
            return signal_map.get(v.get("signal", "HOLD"), 0.0), float(v.get("confidence", 0.0))

        analyst_sig, analyst_conf     = get_vote("analyst")
        sentiment_sig, sentiment_conf = get_vote("sentiment")
        risk_sig, risk_conf           = get_vote("risk")

        import datetime
        hour = float(datetime.datetime.now().hour) / 23.0

        obs = SwarmObservation(
            analyst_signal    = analyst_sig,
            analyst_conf      = analyst_conf,
            sentiment_signal  = sentiment_sig,
            sentiment_conf    = sentiment_conf,
            risk_signal       = risk_sig,
            risk_conf         = risk_conf,
            mf_bullish        = float(mf.get("bullish_probability", 0.5)),
            mf_bearish        = float(mf.get("bearish_probability", 0.5)),
            mf_confidence     = float(mf.get("confidence_score", 0.0)),
            kronos_direction  = {"UP": 1.0, "DOWN": -1.0}.get(kronos.get("direction", "FLAT"), 0.0),
            kronos_magnitude  = float(kronos.get("magnitude_pct", 0.0)),
            kronos_confidence = float(kronos.get("confidence", 0.0)),
            spread_pips       = float(swarm_output.get("spread", 0.0)),
            atr_normalized    = float(swarm_output.get("atr", 0.001)) / max(float(swarm_output.get("price", 1.0)), 0.001),
            session_encoded   = session_map.get(swarm_output.get("session", "london"), 0.5),
            hour_of_day       = hour,
            consensus_score   = float(swarm_output.get("score", 0.0)),
            final_signal      = signal_map.get(swarm_output.get("decision", "HOLD"), 0.0),
        )
        self.current_obs = obs
        return obs

    def step(self, action: int, pnl: float) -> Tuple[np.ndarray, float, bool]:
        """
        Execute RL step.
        
        Args:
            action: SwarmAction enum value
            pnl:    Realized PnL from this trade ($)
        
        Returns:
            (next_obs, reward, done)
        """
        self.step_count += 1
        self.equity     += pnl
        self.episode_pnl += pnl

        reward = self.reward_fn.calculate(
            pnl    = pnl,
            action = SwarmAction(action),
            obs    = self.current_obs,
        )

        self.episode_trades.append({
            "step":   self.step_count,
            "action": SwarmAction(action).name,
            "pnl":    pnl,
            "reward": reward,
            "equity": self.equity,
        })

        # Episode ends on drawdown or max steps
        drawdown = (self.initial_equity - self.equity) / self.initial_equity
        done = drawdown > 0.20 or self.step_count >= 1000

        next_obs = self.current_obs.to_array() if self.current_obs else np.zeros(self.obs_shape)
        return next_obs, reward, done

    def lot_size_from_action(self, action: int, base_lot: float) -> float:
        """Map RL action to actual lot size multiplier."""
        multipliers = {
            SwarmAction.HOLD:           0.0,
            SwarmAction.EXECUTE_FULL:   1.0,
            SwarmAction.EXECUTE_HALF:   0.5,
            SwarmAction.EXECUTE_DOUBLE: 2.0,
            SwarmAction.OVERRIDE_VETO:  0.25,  # Still restricted even when overriding
        }
        return base_lot * multipliers.get(SwarmAction(action), 0.0)

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return summary metrics for this episode."""
        if not self.episode_trades:
            return {}
        pnls   = [t["pnl"] for t in self.episode_trades]
        wins   = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        return {
            "total_trades": len(pnls),
            "win_rate":     wins / len(pnls) if pnls else 0,
            "total_pnl":    sum(pnls),
            "max_drawdown": min(pnls) if pnls else 0,
            "sharpe":       np.mean(pnls) / (np.std(pnls) + 1e-8),
            "final_equity": self.equity,
        }
