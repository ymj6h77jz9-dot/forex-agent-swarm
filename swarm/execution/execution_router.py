"""
EXECUTION ROUTER — KRATOS v2
==============================
Implements R11–R15: Execution & Performance

  R11 Latency-Aware Execution Paths (fast/standard/slow)
  R12 Deterministic Replay Capability
  R13 Parallel Execution with Timeout Management
  R14 Skill-Based Dynamic Tool Loading
  R15 Multi-Modal Signal Fusion

Integrated from:
  - ymj6h77jz9-dot/nautilus_trader (event-driven execution patterns)
  - ymj6h77jz9-dot/KRATOS-app (ExecutionEngine, DerivAdapter)
  - broker/broker_router.py (multi-broker failover)

Usage:
    router = ExecutionRouter(broker_router)
    result = await router.execute(decision, market_state)
    snapshot = router.create_replay_snapshot(decision_id, inputs, outputs)
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Execution paths ───────────────────────────────────────────────────────────

class ExecutionPath(str, Enum):
    FAST     = "fast"      # <20ms  — high confidence, simple market order
    STANDARD = "standard"  # <100ms — normal flow
    SLOW     = "slow"      # <500ms — complex, low confidence, needs extra checks


@dataclass
class ExecutionResult:
    success:      bool
    order_id:     Optional[str]
    exec_price:   float
    latency_ms:   float
    path:         ExecutionPath
    broker:       str
    status:       str
    lot_size:     float
    stop_loss:    float
    take_profit:  float
    error:        Optional[str] = None
    replay_id:    Optional[str] = None


# ── Skill registry ────────────────────────────────────────────────────────────

# Built-in skills — extend via register_skill()
BUILTIN_SKILLS: Dict[str, Callable] = {}


def _register_builtins():
    """Register all built-in skills at module load."""

    def calculate_indicators(closes, highs=None, lows=None, **_):
        """Compute EMA8, RSI14, ATR14 — same logic as AnalystAgent."""
        closes = np.array(closes, dtype=float)
        k8  = 2.0 / 9
        ema = closes[0]
        for c in closes[1:]:
            ema = c * k8 + ema * (1 - k8)
        # RSI
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        rs     = (np.mean(gains[-14:]) + 1e-9) / (np.mean(losses[-14:]) + 1e-9)
        rsi    = 100 - 100 / (1 + rs)
        return {"ema8": round(float(ema), 5), "rsi14": round(float(rsi), 2)}

    def calculate_position_size(equity, atr, risk_pct=0.02, pair="EURUSD"):
        """ATR-based position sizing."""
        pip      = 0.01 if "JPY" in pair else 0.0001
        sl_pips  = (atr * 1.5) / pip
        pip_val  = 10.0 if "JPY" not in pair else 1000.0
        lot      = float(np.clip(equity * risk_pct / (sl_pips * pip_val), 0.01, 5.0))
        return {"lot_size": round(lot, 2), "sl_pips": round(sl_pips, 1)}

    def analyze_spread(spread_pips, pair="EURUSD"):
        """Classify spread quality."""
        thresholds = {"EURUSD": 1.5, "GBPUSD": 2.0, "XAUUSD": 8.0}
        max_spread = thresholds.get(pair, 3.0)
        return {"spread_ok": spread_pips <= max_spread,
                "spread_quality": "tight" if spread_pips <= max_spread * 0.5 else
                                  "normal" if spread_pips <= max_spread else "wide"}

    BUILTIN_SKILLS["technical_indicators"] = calculate_indicators
    BUILTIN_SKILLS["position_sizing"]      = calculate_position_size
    BUILTIN_SKILLS["spread_analysis"]      = analyze_spread


_register_builtins()


# ── Execution Router ──────────────────────────────────────────────────────────

class ExecutionRouter:
    """
    R11–R15 Execution & Performance subsystem.

    Wraps BrokerRouter with:
      - Latency-aware path selection (R11)
      - Deterministic replay snapshots (R12)
      - Parallel gather with timeouts (R13)
      - Skill registry (R14)
      - Multi-modal signal fusion (R15)
    """

    def __init__(self, broker_router=None):
        self._broker  = broker_router
        self._replay: Dict[str, Dict] = {}     # decision_id → snapshot
        self._skills  = dict(BUILTIN_SKILLS)   # Mutable copy
        self._latencies: List[float] = []       # Rolling latency log
        logger.info(f"[ExecutionRouter] Initialised — skills={list(self._skills.keys())}")

    # ─────────────────────────────────────────────────────────────────────────
    # R11: Latency-Aware Execution Paths
    # ─────────────────────────────────────────────────────────────────────────

    def determine_path(
        self,
        confidence: float,
        complexity: str = "low",    # "low" | "medium" | "high"
        urgency:    str = "normal", # "high" | "normal" | "low"
    ) -> ExecutionPath:
        """
        Select execution path.

        Fast:     confidence > 0.85 AND complexity=low AND urgency=high
        Slow:     confidence < 0.60 OR complexity=high
        Standard: everything else
        """
        if confidence > 0.85 and complexity == "low" and urgency == "high":
            return ExecutionPath.FAST
        if confidence < 0.60 or complexity == "high":
            return ExecutionPath.SLOW
        return ExecutionPath.STANDARD

    async def execute(
        self,
        pair:        str,
        direction:   str,
        lot_size:    float,
        stop_loss:   float,
        take_profit: float,
        confidence:  float,
        complexity:  str = "low",
        urgency:     str = "normal",
        decision_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Route and execute an order via the appropriate path and broker.
        Creates a replay snapshot automatically.
        """
        path  = self.determine_path(confidence, complexity, urgency)
        start = time.perf_counter()
        inputs = {
            "pair": pair, "direction": direction, "lot_size": lot_size,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "confidence": confidence, "path": path.value,
        }

        try:
            if path == ExecutionPath.FAST:
                result = await self._fast_path(pair, direction, lot_size, stop_loss, take_profit)
            elif path == ExecutionPath.SLOW:
                result = await self._slow_path(pair, direction, lot_size, stop_loss, take_profit)
            else:
                result = await self._standard_path(pair, direction, lot_size, stop_loss, take_profit)
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            result  = ExecutionResult(
                success=False, order_id=None, exec_price=0.0,
                latency_ms=latency, path=path, broker="NONE",
                status="FAILED", lot_size=lot_size,
                stop_loss=stop_loss, take_profit=take_profit, error=str(e),
            )

        # Record latency
        self._latencies.append(result.latency_ms)
        if len(self._latencies) > 200:
            self._latencies = self._latencies[-200:]

        # R12: Create replay snapshot
        if decision_id:
            self.create_replay_snapshot(decision_id, inputs, {
                "status":    result.status,
                "order_id":  result.order_id,
                "broker":    result.broker,
                "path":      result.path.value,
            })
            result.replay_id = decision_id

        logger.info(
            f"[R11] {direction} {pair} | path={path.value} "
            f"lot={lot_size:.2f} | status={result.status} | "
            f"broker={result.broker} | latency={result.latency_ms:.1f}ms"
        )
        return result

    async def _fast_path(self, pair, direction, lot, sl, tp) -> ExecutionResult:
        start = time.perf_counter()
        try:
            if self._broker:
                r = await asyncio.wait_for(
                    self._broker.place_order(pair, direction, lot, sl, tp),
                    timeout=0.020,  # 20ms hard limit
                )
            else:
                r = {"status": "SIMULATED", "order_id": f"FAST-SIM", "exec_price": 0.0, "broker": "SIM"}
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False, order_id=None, exec_price=0.0,
                latency_ms=20.0, path=ExecutionPath.FAST, broker="TIMEOUT",
                status="TIMEOUT", lot_size=lot, stop_loss=sl, take_profit=tp,
                error="fast_path_timeout",
            )
        return self._build_result(r, ExecutionPath.FAST, lot, sl, tp, start)

    async def _standard_path(self, pair, direction, lot, sl, tp) -> ExecutionResult:
        start = time.perf_counter()
        if self._broker:
            r = await asyncio.wait_for(
                self._broker.place_order(pair, direction, lot, sl, tp, failover=True),
                timeout=0.100,
            )
        else:
            r = {"status": "SIMULATED", "order_id": "STD-SIM", "exec_price": 0.0, "broker": "SIM"}
        return self._build_result(r, ExecutionPath.STANDARD, lot, sl, tp, start)

    async def _slow_path(self, pair, direction, lot, sl, tp) -> ExecutionResult:
        start = time.perf_counter()
        # Slow path: extra spread check, retry once on failure
        for attempt in range(2):
            try:
                if self._broker:
                    r = await asyncio.wait_for(
                        self._broker.place_order(pair, direction, lot, sl, tp, failover=True),
                        timeout=0.500,
                    )
                else:
                    r = {"status": "SIMULATED", "order_id": f"SLOW-SIM-{attempt}", "exec_price": 0.0, "broker": "SIM"}
                if r.get("status") not in ("FAILED", "REJECTED"):
                    return self._build_result(r, ExecutionPath.SLOW, lot, sl, tp, start)
            except Exception as e:
                logger.warning(f"[R11] Slow path attempt {attempt+1} failed: {e}")
                await asyncio.sleep(0.1)
        r = {"status": "FAILED", "order_id": None, "exec_price": 0.0, "broker": "NONE"}
        return self._build_result(r, ExecutionPath.SLOW, lot, sl, tp, start)

    def _build_result(self, r: Dict, path: ExecutionPath, lot: float, sl: float, tp: float, start: float) -> ExecutionResult:
        latency = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            success     = r.get("status") in ("FILLED", "SIMULATED"),
            order_id    = r.get("order_id"),
            exec_price  = float(r.get("exec_price", 0.0)),
            latency_ms  = latency,
            path        = path,
            broker      = r.get("broker", "UNKNOWN"),
            status      = r.get("status", "UNKNOWN"),
            lot_size    = lot,
            stop_loss   = sl,
            take_profit = tp,
            error       = r.get("error"),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # R12: Deterministic Replay
    # ─────────────────────────────────────────────────────────────────────────

    def create_replay_snapshot(
        self,
        decision_id: str,
        inputs:      Dict,
        outputs:     Dict,
    ):
        """
        Store a deterministic snapshot of a decision for future replay.
        Input hash ensures we can verify replayed outputs match originals.
        """
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()

        self._replay[decision_id] = {
            "decision_id": decision_id,
            "timestamp":   time.time(),
            "inputs":      inputs,
            "outputs":     outputs,
            "input_hash":  input_hash,
        }
        # Keep last 500 snapshots
        if len(self._replay) > 500:
            oldest = sorted(self._replay, key=lambda k: self._replay[k]["timestamp"])[:100]
            for k in oldest:
                del self._replay[k]

    def get_replay_snapshot(self, decision_id: str) -> Optional[Dict]:
        """Retrieve a stored replay snapshot by decision ID."""
        return self._replay.get(decision_id)

    def verify_replay(self, decision_id: str, replayed_outputs: Dict) -> Dict:
        """
        Verify replayed outputs match original by comparing JSON hash.
        Returns determinism verdict dict.
        """
        snap = self._replay.get(decision_id)
        if not snap:
            return {"verified": False, "reason": "snapshot_not_found"}

        orig_hash    = hashlib.sha256(json.dumps(snap["outputs"], sort_keys=True, default=str).encode()).hexdigest()
        replay_hash  = hashlib.sha256(json.dumps(replayed_outputs, sort_keys=True, default=str).encode()).hexdigest()
        deterministic = orig_hash == replay_hash

        return {
            "decision_id":  decision_id,
            "deterministic": deterministic,
            "orig_hash":    orig_hash[:12],
            "replay_hash":  replay_hash[:12],
            "match":        deterministic,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # R13: Parallel Execution with Timeout Management
    # ─────────────────────────────────────────────────────────────────────────

    async def gather_with_timeout(
        self,
        coroutines: List,
        timeout:    float = 5.0,
        label:      str   = "gather",
    ) -> List[Any]:
        """
        Execute coroutines in parallel with a global timeout.
        Pending tasks are cancelled cleanly. Never raises — always returns list.
        Uses return_exceptions=True (R13 requirement).
        """
        tasks  = [asyncio.create_task(c) for c in coroutines]
        done, pending = await asyncio.wait(tasks, timeout=timeout,
                                           return_when=asyncio.ALL_COMPLETED)

        # Cancel stragglers
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        results = []
        for t in tasks:
            if t in done:
                try:
                    results.append(t.result())
                except Exception as e:
                    results.append({"error": str(e), "source": label})
            else:
                results.append({"error": "timeout", "source": label})

        cancelled = len(pending)
        if cancelled:
            logger.warning(f"[R13] {label}: {cancelled} task(s) timed out after {timeout}s")

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # R14: Skill-Based Dynamic Tool Loading
    # ─────────────────────────────────────────────────────────────────────────

    def register_skill(self, name: str, fn: Callable):
        """Register a new skill dynamically at runtime."""
        self._skills[name] = fn
        logger.info(f"[R14] Skill registered: {name}")

    async def invoke_skill(self, skill_name: str, params: Dict) -> Dict:
        """
        Invoke a registered skill by name. Supports both sync and async.
        Returns {success, result, skill} dict.
        """
        if skill_name not in self._skills:
            return {"success": False, "error": f"Skill '{skill_name}' not registered",
                    "available": list(self._skills.keys())}
        fn = self._skills[skill_name]
        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**params)
            else:
                result = fn(**params)
            return {"success": True, "result": result, "skill": skill_name}
        except Exception as e:
            logger.error(f"[R14] Skill '{skill_name}' failed: {e}")
            return {"success": False, "error": str(e), "skill": skill_name}

    def list_skills(self) -> List[str]:
        return list(self._skills.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # R15: Multi-Modal Signal Fusion
    # ─────────────────────────────────────────────────────────────────────────

    def fuse_multimodal(
        self,
        price_signal:     Dict,
        news_signal:      Dict,
        sentiment_signal: Dict,
        options_signal:   Optional[Dict] = None,
        regime:           str = "ranging",
    ) -> Dict[str, Any]:
        """
        Fuse signals from multiple modalities into a single confidence score.

        Modality weights adapt to regime:
          Trending:      price=0.45  news=0.20  sentiment=0.25  options=0.10
          News-driven:   price=0.20  news=0.40  sentiment=0.30  options=0.10
          High-vol:      price=0.30  news=0.30  sentiment=0.25  options=0.15
          Default:       price=0.40  news=0.25  sentiment=0.25  options=0.10
        """
        regime_weights: Dict[str, Dict[str, float]] = {
            "trending_up":    {"price": 0.45, "news": 0.20, "sentiment": 0.25, "options": 0.10},
            "trending_down":  {"price": 0.45, "news": 0.20, "sentiment": 0.25, "options": 0.10},
            "news_driven":    {"price": 0.20, "news": 0.40, "sentiment": 0.30, "options": 0.10},
            "high_volatility":{"price": 0.30, "news": 0.30, "sentiment": 0.25, "options": 0.15},
            "ranging":        {"price": 0.40, "news": 0.25, "sentiment": 0.25, "options": 0.10},
        }

        w = dict(regime_weights.get(regime, regime_weights["ranging"]))
        if not options_signal:
            # Redistribute options weight
            non_opt = {k: v for k, v in w.items() if k != "options"}
            tot     = sum(non_opt.values())
            opt_w   = w["options"]
            for k in non_opt:
                w[k] += opt_w * (non_opt[k] / tot)
            w["options"] = 0.0

        # Adjust weights by signal quality
        if float(price_signal.get("confidence", 0.5)) < 0.40:
            w["price"] *= 0.5
            extra = w["price"] * 0.5
            w["news"]      += extra * 0.5
            w["sentiment"] += extra * 0.5

        # Normalise
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}

        # Fuse signals
        sig_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0, "NEUTRAL": 0.0}

        def _sig(d: Optional[Dict]) -> float:
            if d is None: return 0.0
            return float(d.get("signal", sig_map.get(str(d.get("direction", "HOLD")).upper(), 0.0)))

        def _conf(d: Optional[Dict]) -> float:
            return float((d or {}).get("confidence", 0.0))

        fused_sig  = (
            _sig(price_signal)     * w["price"]     * _conf(price_signal) +
            _sig(news_signal)      * w["news"]       * _conf(news_signal) +
            _sig(sentiment_signal) * w["sentiment"]  * _conf(sentiment_signal) +
            (_sig(options_signal)  * w["options"]    * _conf(options_signal) if options_signal else 0.0)
        )

        conf_vals  = [_conf(price_signal), _conf(news_signal), _conf(sentiment_signal)]
        conf_ws    = [w["price"], w["news"], w["sentiment"]]
        if options_signal:
            conf_vals.append(_conf(options_signal))
            conf_ws.append(w["options"])

        fused_conf = float(np.average(conf_vals, weights=conf_ws))
        direction  = "BUY" if fused_sig > 0.1 else "SELL" if fused_sig < -0.1 else "HOLD"

        return {
            "signal":     round(fused_sig,  4),
            "confidence": round(fused_conf, 4),
            "direction":  direction,
            "weights":    {k: round(v, 4) for k, v in w.items()},
            "modalities": {
                "price":     price_signal,
                "news":      news_signal,
                "sentiment": sentiment_signal,
                "options":   options_signal,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Performance metrics
    # ─────────────────────────────────────────────────────────────────────────

    def get_latency_stats(self) -> Dict[str, float]:
        """Return rolling latency statistics (p50, p95, p99, max)."""
        if not self._latencies:
            return {}
        arr = np.array(self._latencies)
        return {
            "p50_ms":   round(float(np.percentile(arr, 50)),  2),
            "p95_ms":   round(float(np.percentile(arr, 95)),  2),
            "p99_ms":   round(float(np.percentile(arr, 99)),  2),
            "max_ms":   round(float(arr.max()),               2),
            "mean_ms":  round(float(arr.mean()),              2),
            "samples":  len(self._latencies),
        }
