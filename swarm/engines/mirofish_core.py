"""
MIROFISH CORE — KRATOS v2
===========================
Implements R1–R5: Decision Intelligence

  R1  Dynamic Agent Weighting via RL
  R2  Regime-Aware Routing
  R3  Adversarial Bull/Bear Debate
  R4  Probabilistic Confidence Fusion
  R5  Hierarchical Sub-Agent Spawning

Integrated from: ymj6h77jz9-dot/MiroFish (PSO backbone)
Enhanced with:   TradingAgents debate mechanism, ruflo coordination patterns,
                 KRATOS regime profiles, QuantDinger multi-asset watchlist

Usage:
    core = MiroFishCore()
    regime = await core.detect_regime(market_state, candle_df)
    result = await core.run_full_decision(market_state, signals, candle_df)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llm_client import llm_json

logger = logging.getLogger(__name__)


# ── Market Regimes ────────────────────────────────────────────────────────────

class MarketRegime(str, Enum):
    TRENDING_UP    = "trending_up"
    TRENDING_DOWN  = "trending_down"
    RANGING        = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    NEWS_DRIVEN    = "news_driven"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AgentSignal:
    agent_name:  str
    signal:      float        # -1.0 (SELL) to +1.0 (BUY), 0.0 = FLAT/HOLD
    confidence:  float        # 0.0–1.0
    reasoning:   str
    metadata:    Dict = field(default_factory=dict)


@dataclass
class DecisionResult:
    direction:         str     # "BUY" | "SELL" | "HOLD"
    confidence:        float
    recommended_size:  float   # Lot size multiplier (0.0–2.0)
    agent_signals:     List[AgentSignal]
    regime:            MarketRegime
    regime_weights:    Dict[str, float]
    debate_synthesis:  str
    risk_approved:     bool
    audit_id:          str
    raw_score:         float
    adaptive_threshold: float


@dataclass
class SubAgentResult:
    subagent_id: str
    task:        str
    success:     bool
    result:      Any
    error:       Optional[str] = None
    latency_ms:  float = 0.0


# ── Regime Profiles (R2) ──────────────────────────────────────────────────────

REGIME_PROFILES: Dict[MarketRegime, Dict[str, float]] = {
    MarketRegime.TRENDING_UP: {
        "analyst":   1.50,  # Technical signals amplified in trends
        "sentiment": 0.90,
        "risk":      0.80,
        "mirofish":  1.20,
        "kronos":    1.30,  # FM excels at predicting trend continuation
    },
    MarketRegime.TRENDING_DOWN: {
        "analyst":   1.50,
        "sentiment": 0.90,
        "risk":      1.00,
        "mirofish":  1.20,
        "kronos":    1.30,
    },
    MarketRegime.RANGING: {
        "analyst":   1.20,  # S/R more important than trend
        "sentiment": 0.80,
        "risk":      1.30,  # Risk more cautious in chop
        "mirofish":  1.50,  # PSO excels at range oscillation
        "kronos":    0.90,
    },
    MarketRegime.HIGH_VOLATILITY: {
        "analyst":   0.70,  # TA less reliable in high vol
        "sentiment": 1.10,
        "risk":      2.00,  # Risk agent dominates — protect capital
        "mirofish":  0.90,
        "kronos":    0.80,
    },
    MarketRegime.NEWS_DRIVEN: {
        "analyst":   0.60,  # TA irrelevant during news spikes
        "sentiment": 1.80,  # Sentiment most important
        "risk":      1.50,
        "mirofish":  0.70,
        "kronos":    0.90,
    },
}

# Weight bounds per R1 — enforced in RL update
WEIGHT_LO, WEIGHT_HI = 0.05, 0.40


# ── MiroFish Core ─────────────────────────────────────────────────────────────

class MiroFishCore:
    """
    R1–R5 Decision Intelligence Engine.

    This class is the upgraded replacement for the raw signal_processing calls
    in the orchestrator. It adds:
      - RL-based weight update (R1)
      - Regime detection & weight modulation (R2)
      - Adversarial debate (R3)
      - Probabilistic fusion with diversity bonus (R4)
      - Parallel sub-agent spawning with timeout (R5)
    """

    def __init__(self):
        self._weight_bounds = (WEIGHT_LO, WEIGHT_HI)
        # Performance accumulators per agent for RL updates (R1)
        self._perf: Dict[str, List[float]] = {
            "analyst":   [],
            "sentiment": [],
            "risk":      [],
            "mirofish":  [],
            "kronos":    [],
        }
        logger.info("[MiroFishCore] Initialised — R1-R5 active")

    # ─────────────────────────────────────────────────────────────────────────
    # R1: Dynamic Agent Weighting via RL
    # ─────────────────────────────────────────────────────────────────────────

    def update_agent_weight_rl(
        self,
        weights:      Dict[str, float],
        agent_name:   str,
        pnl:          float,
        confidence:   float,
    ) -> Dict[str, float]:
        """
        Apply RL reward signal to update a single agent's weight.

        Reward formula:
          base_reward  = clamp(risk_adj_return × 0.1, -0.03, +0.05)
          calibration  = -abs(confidence - oracle) × 0.01
          new_weight   = old × (1 + reward), clamped to WEIGHT_BOUNDS

        All weights re-normalised to sum = 1.0 after update.
        Kronos (15%) and MiroFish (15%) are protected — not RL-updated.
        """
        if agent_name not in ("analyst", "sentiment", "risk"):
            return weights  # Only RL-update the LLM agents

        lo, hi  = self._weight_bounds
        old_w   = weights.get(agent_name, 0.25)
        oracle  = 1.0 if pnl > 0 else 0.0
        cal_err = abs(confidence - oracle)

        if pnl > 0:
            reward = min(0.05,  pnl / 100.0 * 0.1)
        else:
            reward = max(-0.03, pnl / 100.0 * 0.1)
        reward -= cal_err * 0.01

        new_w = float(np.clip(old_w * (1.0 + reward), lo, hi))

        # Track performance
        self._perf.setdefault(agent_name, []).append(pnl)

        updated = dict(weights)
        updated[agent_name] = new_w

        # Re-normalise LLM agent weights (keep mirofish + kronos fixed)
        llm_agents = ["analyst", "sentiment", "risk"]
        protected  = {"mirofish": weights.get("mirofish", 0.15),
                      "kronos":   weights.get("kronos",   0.15)}
        llm_total  = sum(updated[a] for a in llm_agents)
        budget     = 1.0 - sum(protected.values())
        if llm_total > 0:
            for a in llm_agents:
                updated[a] = round(updated[a] / llm_total * budget, 4)
        updated.update(protected)

        logger.info(f"[R1] RL weight update {agent_name}: {old_w:.4f} → {updated[agent_name]:.4f}  reward={reward:+.4f}")
        return updated

    # ─────────────────────────────────────────────────────────────────────────
    # R2: Regime Detection & Routing
    # ─────────────────────────────────────────────────────────────────────────

    def detect_regime(
        self,
        atr:          float,
        price:        float,
        candle_df,
        news_impact:  float = 0.0,
    ) -> MarketRegime:
        """
        Classify market regime from price data and news impact.

        Returns: MarketRegime enum value
        """
        try:
            closes = candle_df["close"].values.astype(float) if candle_df is not None and not candle_df.empty else np.array([price])

            volatility = float(np.std(np.diff(closes) / closes[:-1]) * np.sqrt(252)) if len(closes) > 2 else 0.0
            trend_20   = abs(closes[-1] / closes[-20] - 1.0) if len(closes) >= 20 else 0.0
            bullish    = closes[-1] > closes[-20] if len(closes) >= 20 else True
            atr_norm   = atr / max(price, 1e-10)

        except Exception as e:
            logger.warning(f"[R2] Regime detection error: {e}")
            return MarketRegime.RANGING

        if news_impact > 0.70:
            regime = MarketRegime.NEWS_DRIVEN
        elif volatility > 0.20 or atr_norm > 0.015:
            regime = MarketRegime.HIGH_VOLATILITY
        elif trend_20 > 0.012:
            regime = MarketRegime.TRENDING_UP if bullish else MarketRegime.TRENDING_DOWN
        else:
            regime = MarketRegime.RANGING

        logger.info(f"[R2] Regime: {regime.value}  vol={volatility:.4f}  trend={trend_20:.4f}  news={news_impact:.2f}")
        return regime

    def get_regime_weights(
        self,
        base_weights: Dict[str, float],
        regime:       MarketRegime,
    ) -> Dict[str, float]:
        """
        Apply regime multipliers to base weights, then re-normalise.
        Protected agents (mirofish, kronos) use regime multipliers but
        their sum is kept stable.
        """
        profile   = REGIME_PROFILES.get(regime, {})
        modulated = {}
        for agent, base_w in base_weights.items():
            mult = profile.get(agent, 1.0)
            modulated[agent] = base_w * mult

        total = sum(modulated.values())
        if total > 0:
            modulated = {k: round(v / total, 4) for k, v in modulated.items()}

        return modulated

    # ─────────────────────────────────────────────────────────────────────────
    # R3: Adversarial Bull/Bear Debate
    # ─────────────────────────────────────────────────────────────────────────

    async def run_adversarial_debate(
        self,
        bull_signal:   AgentSignal,
        bear_signal:   AgentSignal,
        context:       Dict,
        max_rounds:    int = 2,
    ) -> Dict[str, Any]:
        """
        Moderate a structured bull vs bear debate.
        Each round the losing side gets to counter. Final judge rules.
        """
        pair = context.get("pair", "UNKNOWN")

        JUDGE_PROMPT = """You are a neutral forex trading judge.
You receive a bull argument and a bear argument for a trade on {pair}.
Rule on which side has the stronger case. Consider: technical confluence, macro context, risk, timing.

Return ONLY valid JSON:
{{
  "winner": "BULL" | "BEAR" | "DRAW",
  "direction": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "synthesis": "<1-2 sentences on the winning argument>",
  "weaknesses": ["<weakness1>", "<weakness2>"],
  "confidence_delta": <float -0.15 to +0.15>
}}""".format(pair=pair)

        bull_case = (
            f"BULL CASE ({bull_signal.agent_name}, conf={bull_signal.confidence:.2f}):\n"
            f"{bull_signal.reasoning[:600]}"
        )
        bear_case = (
            f"BEAR CASE ({bear_signal.agent_name}, conf={bear_signal.confidence:.2f}):\n"
            f"{bear_signal.reasoning[:600]}"
        )

        rounds_log = []
        result = {}

        for round_num in range(max_rounds):
            prompt = (
                f"Round {round_num+1}/{max_rounds}\n\n"
                f"{bull_case}\n\n{bear_case}"
            )
            if rounds_log:
                prompt += f"\n\nPrevious round: {rounds_log[-1].get('synthesis','')}"

            try:
                result = await llm_json(
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.15,
                    max_tokens=300,
                )
                rounds_log.append(result)
            except Exception as e:
                logger.warning(f"[R3] Debate round {round_num+1} failed: {e}")
                result = {"winner": "DRAW", "direction": "HOLD", "confidence": 0.0,
                          "synthesis": f"Debate error: {e}", "weaknesses": [], "confidence_delta": 0.0}

        logger.info(f"[R3] Debate: winner={result.get('winner')} dir={result.get('direction')} conf={result.get('confidence',0):.2f}")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # R4: Probabilistic Confidence Fusion
    # ─────────────────────────────────────────────────────────────────────────

    def compute_confidence(
        self,
        signals:        List[AgentSignal],
        weights:        Dict[str, float],
        regime_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Probabilistic fusion with:
          - Effective weight = base × regime_multiplier
          - Diversity bonus (more diverse agents → slight boost)
          - Uncertainty penalty (low avg confidence → penalty)

        Returns dict with confidence, raw_score, direction, effective_signals.
        """
        if not signals:
            return {"confidence": 0.0, "raw_score": 0.0, "direction": "HOLD",
                    "effective_signals": 0, "total_weight": 0.0}

        total_w   = 0.0
        buy_score = 0.0
        sel_score = 0.0
        conf_sum  = 0.0

        for sig in signals:
            base_w   = weights.get(sig.agent_name, 0.20)
            regime_m = regime_weights.get(sig.agent_name, 1.0)
            eff_w    = base_w * regime_m
            conf     = float(sig.confidence)

            if sig.signal > 0.3:          # BUY zone
                buy_score += sig.signal * eff_w * conf
            elif sig.signal < -0.3:       # SELL zone
                sel_score += abs(sig.signal) * eff_w * conf

            conf_sum  += conf * eff_w
            total_w   += eff_w

        if total_w == 0:
            return {"confidence": 0.0, "raw_score": 0.0, "direction": "HOLD",
                    "effective_signals": 0, "total_weight": 0.0}

        raw_buy  = buy_score / total_w
        raw_sell = sel_score / total_w
        avg_conf = conf_sum  / total_w

        # Diversity bonus — reward for having 4+ agents contributing
        n_active = sum(1 for s in signals if abs(s.signal) > 0.1)
        diversity = min(0.08, n_active * 0.02)

        # Uncertainty penalty — if agents are split, reduce confidence
        split_penalty = 0.0
        if buy_score > 0 and sel_score > 0:
            split_ratio   = min(buy_score, sel_score) / max(buy_score, sel_score)
            split_penalty = split_ratio * 0.12

        raw_score  = max(raw_buy, raw_sell)
        confidence = float(np.clip(raw_score * 0.6 + avg_conf * 0.4 + diversity - split_penalty, 0.0, 1.0))
        direction  = "BUY" if raw_buy > raw_sell else "SELL" if raw_sell > raw_buy else "HOLD"

        return {
            "confidence":        confidence,
            "raw_score":         round(raw_score,  4),
            "avg_confidence":    round(avg_conf,   4),
            "direction":         direction,
            "effective_signals": n_active,
            "total_weight":      round(total_w,    4),
            "diversity_bonus":   round(diversity,  4),
            "split_penalty":     round(split_penalty, 4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # R5: Hierarchical Sub-Agent Spawning
    # ─────────────────────────────────────────────────────────────────────────

    async def spawn_subagent(
        self,
        parent_agent: str,
        task:         str,
        context:      Dict,
        timeout:      float = 30.0,
    ) -> SubAgentResult:
        """
        Spawn an isolated sub-agent coroutine with strict timeout.
        All sub-agents run in isolation — parent state is never mutated.
        """
        import time, uuid
        subagent_id = f"{parent_agent}_sub_{task[:12]}_{str(uuid.uuid4())[:6]}"
        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._run_subagent_task(task, context),
                timeout=timeout,
            )
            latency = (time.perf_counter() - start) * 1000
            logger.info(f"[R5] Sub-agent {subagent_id} OK  {latency:.0f}ms")
            return SubAgentResult(subagent_id, task, True,  result, latency_ms=latency)
        except asyncio.TimeoutError:
            latency = timeout * 1000
            logger.warning(f"[R5] Sub-agent {subagent_id} TIMEOUT after {timeout}s")
            return SubAgentResult(subagent_id, task, False, None, "timeout", latency)
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.error(f"[R5] Sub-agent {subagent_id} ERROR: {e}")
            return SubAgentResult(subagent_id, task, False, None, str(e), latency)

    async def spawn_parallel_subagents(
        self,
        tasks:   List[Dict],
        timeout: float = 30.0,
    ) -> List[SubAgentResult]:
        """
        Spawn multiple sub-agents in parallel. All are isolated.
        Uses return_exceptions=True — one failure never kills the rest.
        """
        coros = [
            self.spawn_subagent(
                parent_agent = t.get("parent", "orchestrator"),
                task         = t["task"],
                context      = t.get("context", {}),
                timeout      = timeout,
            )
            for t in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        # Normalise exceptions into SubAgentResult
        out = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                out.append(SubAgentResult(
                    subagent_id = f"err_{i}",
                    task        = tasks[i].get("task", "unknown"),
                    success     = False,
                    result      = None,
                    error       = str(r),
                ))
            else:
                out.append(r)
        return out

    async def _run_subagent_task(self, task: str, context: Dict) -> Dict:
        """
        Dispatch sub-agent task to the correct specialised handler.
        Extends here as new task types are registered.
        """
        pair    = context.get("pair", "EURUSD")
        session = context.get("session", "london")

        if task == "deep_research":
            return await self._subagent_deep_research(pair, context)
        elif task == "sentiment_check":
            return await self._subagent_sentiment_check(pair, context)
        elif task == "regime_recheck":
            return {"regime": self.detect_regime(
                context.get("atr", 0.001), context.get("price", 1.0),
                context.get("candle_df", None), context.get("news_impact", 0.0),
            ).value}
        else:
            return {"task": task, "status": "unsupported", "context": context}

    async def _subagent_deep_research(self, pair: str, context: str) -> Dict:
        prompt = f"Research current macro drivers for {pair}. Context: {str(context)[:400]}"
        try:
            result = await llm_json(
                messages=[
                    {"role": "system", "content": "You are a forex macro researcher. Return JSON: {\"summary\": str, \"sentiment\": BUY|SELL|NEUTRAL, \"confidence\": float, \"key_factors\": [str]}"},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3, max_tokens=400,
            )
            return result
        except Exception as e:
            return {"summary": f"Error: {e}", "sentiment": "NEUTRAL", "confidence": 0.0}

    async def _subagent_sentiment_check(self, pair: str, context: Dict) -> Dict:
        base = pair[:3]
        prompt = f"Quick sentiment check for {pair}. Recent context: {str(context)[:200]}"
        try:
            result = await llm_json(
                messages=[
                    {"role": "system", "content": "Return JSON: {\"sentiment\": BUY|SELL|NEUTRAL, \"confidence\": float, \"reason\": str}"},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2, max_tokens=150,
            )
            return result
        except Exception as e:
            return {"sentiment": "NEUTRAL", "confidence": 0.0, "reason": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # FULL DECISION PIPELINE (R1–R5 integrated)
    # ─────────────────────────────────────────────────────────────────────────

    async def run_full_decision(
        self,
        market_state,
        agent_votes:   List[Dict],
        weights:       Dict[str, float],
        candle_df      = None,
        news_impact:   float = 0.0,
        mf_prediction: Optional[Dict] = None,
        kronos_pred:   Optional[Dict] = None,
    ) -> DecisionResult:
        """
        Full R1–R5 decision pipeline.

        1. R2: Detect regime, get regime weights
        2. Build AgentSignal objects from votes
        3. R3: Adversarial debate (top bull vs top bear)
        4. R4: Probabilistic confidence fusion
        5. Add MiroFish + Kronos as synthetic signals
        6. Return DecisionResult
        """
        import uuid

        # R2: Regime
        regime        = self.detect_regime(market_state.atr, market_state.ask, candle_df, news_impact)
        regime_w      = self.get_regime_weights(weights, regime)

        # Build AgentSignal list
        sig_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0, "FLAT": 0.0}
        signals: List[AgentSignal] = []
        for v in agent_votes:
            signals.append(AgentSignal(
                agent_name = v.get("agent", "unknown"),
                signal     = sig_map.get(v.get("signal", "FLAT"), 0.0),
                confidence = float(v.get("confidence", 0.0)),
                reasoning  = v.get("reasoning", ""),
            ))

        # Add MiroFish synthetic signal
        if mf_prediction:
            mf_bull = float(mf_prediction.get("bullish", 0.5))
            mf_bear = float(mf_prediction.get("bearish", 0.5))
            mf_sig  = mf_bull - mf_bear  # -1.0 to +1.0
            signals.append(AgentSignal(
                agent_name = "mirofish",
                signal     = mf_sig,
                confidence = float(mf_prediction.get("confidence", 0.5)),
                reasoning  = f"PSO: bull={mf_bull:.2f} bear={mf_bear:.2f}",
            ))

        # Add Kronos synthetic signal
        if kronos_pred:
            k_dir_map = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}
            k_sig = k_dir_map.get(kronos_pred.get("direction", "FLAT"), 0.0)
            signals.append(AgentSignal(
                agent_name = "kronos",
                signal     = k_sig,
                confidence = float(kronos_pred.get("confidence", 0.5)),
                reasoning  = f"Kronos FM: {kronos_pred.get('direction')} {kronos_pred.get('magnitude_pct',0):+.4f}%",
            ))

        # R3: Adversarial debate
        bull_sig = max(signals, key=lambda s: s.signal * s.confidence, default=signals[0])
        bear_sig = min(signals, key=lambda s: s.signal * s.confidence, default=signals[-1])
        debate   = {}
        if bull_sig.signal > 0 and bear_sig.signal < 0:
            debate = await self.run_adversarial_debate(bull_sig, bear_sig,
                                                       {"pair": market_state.pair})

        # R4: Confidence fusion
        fusion = self.compute_confidence(signals, weights, regime_w)

        # Debate override
        debate_conf = float(debate.get("confidence", 0.0))
        debate_dir  = debate.get("direction", "HOLD")
        if debate_conf > fusion["confidence"]:
            fusion["confidence"] = debate_conf
            fusion["direction"]  = debate_dir

        direction  = fusion.get("direction", "HOLD")
        confidence = fusion.get("confidence", 0.0)

        # Recommended size: 1.0 base, scaled by confidence tier
        if confidence >= 0.90:
            rec_size = 1.50
        elif confidence >= 0.80:
            rec_size = 1.00
        elif confidence >= 0.70:
            rec_size = 0.75
        else:
            rec_size = 0.0   # Below threshold — don't size

        return DecisionResult(
            direction         = direction,
            confidence        = confidence,
            recommended_size  = rec_size,
            agent_signals     = signals,
            regime            = regime,
            regime_weights    = regime_w,
            debate_synthesis  = debate.get("synthesis", ""),
            risk_approved     = True,   # Final approval from RiskEngine
            audit_id          = str(uuid.uuid4())[:8],
            raw_score         = fusion.get("raw_score", 0.0),
            adaptive_threshold = 0.70,
        )


class SubAgentManager:
    """
    Standalone wrapper around MiroFishCore sub-agent spawning (R5).
    Provides the SubAgentManager interface expected by external callers
    and the CI/CD spec.
    """

    def __init__(self, core: "MiroFishCore"):
        self._core = core

    async def spawn_subagent(
        self,
        parent_agent: str,
        task:         str,
        context:      dict,
        timeout:      float = 30.0,
    ) -> SubAgentResult:
        return await self._core.spawn_subagent(parent_agent, task, context, timeout)

    async def spawn_parallel_subagents(
        self,
        tasks:   list,
        timeout: float = 30.0,
    ) -> list:
        return await self._core.spawn_parallel_subagents(tasks, timeout)
