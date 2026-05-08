"""
MIROFISH PREDICTION ENGINE — Integrated from ymj6h77jz9-dot/KRATOS-app
------------------------------------------------------------------------
Swarm intelligence backbone. Runs probabilistic simulation of future
market scenarios using parallel digital agents. The Orchestrator uses
this to "rehearse the future" before committing to a trade.

Source: MiroFish — "A Simple and Universal Swarm Intelligence Engine, Predicting Anything"
Adapted for: Forex Agentic Swarm
"""

import asyncio
import logging
import random
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SimulationParticle:
    """A single agent in the MiroFish swarm simulation."""
    id: int
    position: float          # Price position
    velocity: float          # Rate of change
    best_position: float     # Personal best
    bias: str                # "bull" | "bear" | "neutral"
    confidence: float


@dataclass
class PredictionResult:
    symbol: str
    timestamp: str
    bullish_probability: float
    bearish_probability: float
    neutral_probability: float
    expected_high: float
    expected_low: float
    expected_mid: float
    confidence_score: float
    simulation_id: str
    iterations: int
    particle_consensus: str  # "BUY" | "SELL" | "FLAT"


class MiroFishEngine:
    """
    Particle Swarm Optimization (PSO) based market prediction engine.
    Runs N simulated "particles" (agents) that explore price space and
    converge toward the most probable future price range.
    
    Integrated from: KRATOS-app/server/_core/engines/mirofish_adapter.py
    Enhanced with: MiroFish swarm intelligence principles
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_particles   = self.config.get("n_particles", 50)
        self.n_iterations  = self.config.get("n_iterations", 100)
        self.inertia       = self.config.get("inertia", 0.729)       # PSO inertia weight
        self.cognitive     = self.config.get("cognitive", 1.494)     # Personal best weight
        self.social        = self.config.get("social", 1.494)        # Global best weight
        self.volatility_scale = self.config.get("volatility_scale", 0.002)
        self.simulation_history: List[PredictionResult] = []
        logger.info(f"MiroFish Engine initialized: {self.n_particles} particles, {self.n_iterations} iters")

    async def predict(self, symbol: str, current_price: float, atr: float,
                      agent_bias: float = 0.0) -> PredictionResult:
        """
        Run PSO simulation to predict probable price outcomes.
        
        Args:
            symbol:       Currency pair (e.g. "EURUSD")
            current_price: Current mid price
            atr:          Average True Range (volatility measure)
            agent_bias:   [-1.0 bearish ... +1.0 bullish] from LLM agent ensemble
        """
        logger.info(f"[MiroFish] Running prediction for {symbol} @ {current_price}")

        # Run simulation in thread pool to not block event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._run_pso, symbol, current_price, atr, agent_bias
        )

        self.simulation_history.append(result)
        return result

    def _run_pso(self, symbol: str, price: float, atr: float, bias: float) -> PredictionResult:
        """Core PSO simulation — runs synchronously in executor."""
        # Initialize particles with random positions around current price
        particles = self._init_particles(price, atr)
        global_best_pos   = price
        global_best_score = float("-inf")

        for iteration in range(self.n_iterations):
            for p in particles:
                # Evaluate particle fitness
                score = self._fitness(p.position, price, atr, bias)

                if score > global_best_score:
                    global_best_score = score
                    global_best_pos   = p.position

                if score > self._fitness(p.best_position, price, atr, bias):
                    p.best_position = p.position

            # Update velocities and positions
            for p in particles:
                r1, r2 = random.random(), random.random()
                p.velocity = (
                    self.inertia   * p.velocity
                    + self.cognitive * r1 * (p.best_position - p.position)
                    + self.social    * r2 * (global_best_pos  - p.position)
                )
                p.position += p.velocity

        # Analyse final particle distribution
        positions  = [p.position for p in particles]
        above      = sum(1 for pos in positions if pos > price)
        below      = sum(1 for pos in positions if pos < price)
        total      = len(positions)

        bullish_prob = above / total
        bearish_prob = below / total
        neutral_prob = 1.0 - bullish_prob - bearish_prob

        # Apply agent bias adjustment
        bullish_prob = min(1.0, max(0.0, bullish_prob + bias * 0.1))
        bearish_prob = min(1.0, max(0.0, bearish_prob - bias * 0.1))

        expected_high = price + atr * 2.0
        expected_low  = price - atr * 2.0
        expected_mid  = sum(positions) / len(positions)

        # Consensus
        if bullish_prob > 0.60:
            consensus = "BUY"
        elif bearish_prob > 0.60:
            consensus = "SELL"
        else:
            consensus = "FLAT"

        confidence = abs(bullish_prob - bearish_prob)

        return PredictionResult(
            symbol               = symbol,
            timestamp            = datetime.now(timezone.utc).isoformat(),
            bullish_probability  = round(bullish_prob, 4),
            bearish_probability  = round(bearish_prob, 4),
            neutral_probability  = round(neutral_prob, 4),
            expected_high        = round(expected_high, 5),
            expected_low         = round(expected_low,  5),
            expected_mid         = round(expected_mid,  5),
            confidence_score     = round(confidence, 4),
            simulation_id        = f"mf_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            iterations           = self.n_iterations,
            particle_consensus   = consensus,
        )

    def _init_particles(self, price: float, atr: float) -> List[SimulationParticle]:
        """Initialize PSO particles around current price."""
        particles = []
        bias_options = ["bull", "bear", "neutral"]

        for i in range(self.n_particles):
            noise    = random.gauss(0, atr * self.volatility_scale * 100)
            position = price + noise
            velocity = random.gauss(0, atr * 0.01)
            particles.append(SimulationParticle(
                id            = i,
                position      = position,
                velocity      = velocity,
                best_position = position,
                bias          = random.choice(bias_options),
                confidence    = random.random(),
            ))
        return particles

    def _fitness(self, position: float, current_price: float,
                 atr: float, bias: float) -> float:
        """
        Fitness function: reward positions that align with trend and ATR range.
        Higher score = more probable price target.
        """
        distance   = abs(position - current_price)
        atr_score  = math.exp(-distance / (atr * 2))    # Gaussian decay by ATR
        bias_score = bias * (position - current_price) / (atr + 1e-10)
        return atr_score + bias_score * 0.3

    def reflect_on_performance(self, predicted: PredictionResult, actual_outcome: str):
        """
        Self-improvement: log prediction accuracy to refine future simulations.
        Called by MemoryAgent after trade closes.
        """
        correct = predicted.particle_consensus == actual_outcome
        logger.info(
            f"[MiroFish] Reflection: predicted={predicted.particle_consensus} "
            f"actual={actual_outcome} correct={correct}"
        )
        # TODO: Adjust inertia/cognitive/social weights based on accuracy history
