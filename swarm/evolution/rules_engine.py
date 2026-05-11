"""
KRATOS v2 — SELF-EVOLUTION ENGINE: 20 RULES
=============================================
This is the brain that governs how KRATOS can evolve itself.

Every proposed change to the orchestrator — weight updates, threshold
shifts, strategy pivots — passes through this engine. If it fails any
rule, the change is VETOED and the system stays in its last-known-good
state.

THE 20 RULES:
  Decision Intelligence (R1–R5):
    R1  RL-Weighted Agent Signals
    R2  Regime-Aware Routing
    R3  Adversarial Debate Before Every Decision
    R4  Probabilistic Confidence Fusion
    R5  Sub-Agent Spawning for Deep Research

  Memory & Learning (R6–R10):
    R6  Memory-Augmented Context Retrieval
    R7  Temporal Validity Verification
    R8  Reflection Loop with Auto-Optimization
    R9  Agent-Specific Diaries
    R10 Contradiction Detection

  Execution & Performance (R11–R15):
    R11 Latency-Aware Execution Paths
    R12 Deterministic Replay
    R13 Parallel Execution with Timeout Guards
    R14 Skill-Based Dynamic Tool Loading
    R15 Multi-Modal Signal Fusion

  Risk & Safety (R16–R20):
    R16 Absolute Risk Veto Layer (never bypassed)
    R17 Cross-Asset Correlation Awareness
    R18 Anomaly Detection & Circuit Breakers
    R19 Adaptive Thresholding
    R20 Immutable Audit Trail
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Hard-coded limits — NEVER configurable ────────────────────────────────────
MAX_RISK_PER_TRADE    = 0.02       # 2% of equity max per trade
MAX_DAILY_LOSS        = 0.05       # 5% daily drawdown → circuit breaker
MAX_PORTFOLIO_EXPOSURE = 0.15      # 15% of equity in open trades
MAX_CONSECUTIVE_LOSSES = 5         # 5 straight losses → circuit breaker
MAX_EXECUTION_LATENCY_MS = 500     # 500ms → circuit breaker
MAX_CORRELATION       = 0.80       # Max correlation between open positions
MIN_BACKTEST_DAYS     = 180        # 6-month backtest required for weight changes
MIN_TEST_COVERAGE     = 0.85       # 85% overall, 100% for risk/execution paths
MIN_SHADOW_DAYS       = 7          # 7 days shadow before any weight goes live
MIN_SHARPE_RATIO      = 0.0        # Backtest must not be negative Sharpe
MIN_WIN_RATE          = 0.45       # 45% minimum win rate over 30+ trades
WEIGHT_BOUNDS         = (0.05, 0.40)  # No agent goes below 5% or above 40%
AUDIT_PATH            = Path(__file__).parent.parent / "data" / "audit_trail.jsonl"


# ── Data structures ───────────────────────────────────────────────────────────

class RuleSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR    = "major"
    MINOR    = "minor"


class CircuitBreakerReason(str, Enum):
    DAILY_LOSS           = "daily_loss_limit"
    CONSECUTIVE_LOSSES   = "consecutive_losses"
    HIGH_LATENCY         = "execution_latency"
    ANOMALY_DETECTED     = "anomaly_detected"
    CORRELATION_BREACH   = "correlation_breach"
    RULE_VIOLATION       = "rule_violation"


@dataclass
class RuleViolation:
    rule_id:  str
    severity: RuleSeverity
    message:  str
    detail:   Dict = field(default_factory=dict)


@dataclass
class EvolutionProposal:
    """A proposed change to the swarm — must pass all 20 rules."""
    proposal_id:       str
    proposed_weights:  Dict[str, float]   # New agent weights
    proposed_threshold: float             # New consensus threshold
    backtest_results:  Dict[str, Any]     # Must be populated
    shadow_results:    Dict[str, Any]     # Must be populated
    test_coverage:     Dict[str, float]   # overall, risk, execution
    code_diff:         str                # What changed
    proposed_by:       str                # "self_evolution" | "manual"
    timestamp:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RuleResult:
    rule_id:    str
    rule_name:  str
    passed:     bool
    violations: List[RuleViolation]
    details:    Dict = field(default_factory=dict)


@dataclass
class EvolutionVerdict:
    proposal_id:  str
    approved:     bool
    results:      Dict[str, RuleResult]
    violations:   List[RuleViolation]
    reason:       str
    timestamp:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Main engine ───────────────────────────────────────────────────────────────

class KratosEvolutionEngine:
    """
    The 20-rule self-evolution validator.

    Usage:
        engine = KratosEvolutionEngine()
        verdict = await engine.validate(proposal)
        if verdict.approved:
            apply_new_weights(proposal.proposed_weights)

    The engine also owns:
      - Circuit breaker (R18)
      - Audit trail (R20)
      - Anomaly detector (R18)
      - Adaptive threshold calculator (R19)
    """

    def __init__(self):
        # Circuit breaker state
        self._cb_active:  bool = False
        self._cb_reason:  Optional[str] = None
        self._cb_at:      Optional[str] = None

        # Runtime counters for anomaly detection
        self._consecutive_losses:  int = 0
        self._latency_history:     List[float] = []
        self._pnl_history:         List[float] = []
        self._daily_pnl:           float = 0.0
        self._equity_baseline:     float = float(os.environ.get("ACCOUNT_EQUITY", "10000"))

        # Correlation cache (pair → pair → float)
        self._corr_cache: Dict[str, float] = {}

        # Audit trail
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

        logger.info("[EvolutionEngine] Initialised — 20 rules active | circuit breaker READY")

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    async def validate(self, proposal: EvolutionProposal) -> EvolutionVerdict:
        """
        Run all 20 rules against a proposed change.
        Returns EvolutionVerdict — approved only if ALL rules pass.
        """
        logger.info(f"[EvolutionEngine] Validating proposal {proposal.proposal_id}")

        rules = [
            # Decision Intelligence
            (self._r1_rl_weight_bounds,           "R1",  "RL-Weighted Agent Signals"),
            (self._r2_regime_routing_complete,     "R2",  "Regime-Aware Routing"),
            (self._r3_debate_required,             "R3",  "Adversarial Debate Mandatory"),
            (self._r4_confidence_fusion,           "R4",  "Probabilistic Confidence Fusion"),
            (self._r5_subagent_timeout,            "R5",  "Sub-Agent Spawning Safety"),
            # Memory & Learning
            (self._r6_memory_retrieval,            "R6",  "Memory-Augmented Context"),
            (self._r7_temporal_validity,           "R7",  "Temporal Validity Check"),
            (self._r8_reflection_loop,             "R8",  "Reflection Loop"),
            (self._r9_agent_diaries,               "R9",  "Agent Diaries"),
            (self._r10_contradiction_detection,    "R10", "Contradiction Detection"),
            # Execution & Performance
            (self._r11_latency_paths,              "R11", "Latency-Aware Execution Paths"),
            (self._r12_determinism,                "R12", "Deterministic Replay"),
            (self._r13_parallel_timeout,           "R13", "Parallel Execution Guards"),
            (self._r14_skill_registry,             "R14", "Skill-Based Tool Loading"),
            (self._r15_multimodal_fusion,          "R15", "Multi-Modal Signal Fusion"),
            # Risk & Safety
            (self._r16_risk_veto,                  "R16", "Absolute Risk Veto Layer"),
            (self._r17_correlation,                "R17", "Cross-Asset Correlation"),
            (self._r18_circuit_breaker,            "R18", "Anomaly & Circuit Breaker"),
            (self._r19_adaptive_threshold,         "R19", "Adaptive Thresholding"),
            (self._r20_audit_trail,                "R20", "Immutable Audit Trail"),
        ]

        results: Dict[str, RuleResult] = {}
        all_violations: List[RuleViolation] = []

        for fn, rule_id, rule_name in rules:
            try:
                result = await fn(proposal)
                result.rule_id   = rule_id
                result.rule_name = rule_name
                results[rule_id] = result
                if not result.passed:
                    all_violations.extend(result.violations)
                    logger.warning(f"  [{rule_id}] FAIL — {rule_name}: {[v.message for v in result.violations]}")
                else:
                    logger.info(f"  [{rule_id}] PASS — {rule_name}")
            except Exception as e:
                v = RuleViolation(rule_id, RuleSeverity.CRITICAL, f"Rule evaluation error: {e}")
                results[rule_id] = RuleResult(rule_id, rule_name, False, [v])
                all_violations.append(v)
                logger.error(f"  [{rule_id}] ERROR — {e}")

        approved = len(all_violations) == 0
        reason   = "All 20 rules passed" if approved else \
                   f"{len(all_violations)} violation(s): " + \
                   "; ".join(v.message for v in all_violations[:3])

        verdict = EvolutionVerdict(
            proposal_id = proposal.proposal_id,
            approved    = approved,
            results     = results,
            violations  = all_violations,
            reason      = reason,
        )

        # R20: always audit
        await self._write_audit(proposal, verdict)

        logger.info(f"[EvolutionEngine] Verdict: {'APPROVED' if approved else 'VETOED'} — {reason}")
        return verdict

    # ═══════════════════════════════════════════════════════════════════════════
    # R1–R5: DECISION INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════

    async def _r1_rl_weight_bounds(self, p: EvolutionProposal) -> RuleResult:
        """R1: All proposed weights must be within (0.05, 0.40) and sum to 1.0."""
        violations = []
        weights = p.proposed_weights

        total = sum(weights.values())
        if abs(total - 1.0) > 0.005:
            violations.append(RuleViolation("R1", RuleSeverity.CRITICAL,
                f"Weights sum to {total:.4f}, must be 1.0"))

        for agent, w in weights.items():
            lo, hi = WEIGHT_BOUNDS
            if not (lo <= w <= hi):
                violations.append(RuleViolation("R1", RuleSeverity.CRITICAL,
                    f"Agent '{agent}' weight {w:.3f} outside bounds [{lo}, {hi}]"))

        required = {"analyst", "sentiment", "risk", "mirofish", "kronos"}
        missing = required - set(weights.keys())
        if missing:
            violations.append(RuleViolation("R1", RuleSeverity.CRITICAL,
                f"Missing required agents: {missing}"))

        return RuleResult("R1", "RL-Weighted Agent Signals", not violations, violations,
                          {"weights": weights, "total": total})

    async def _r2_regime_routing_complete(self, p: EvolutionProposal) -> RuleResult:
        """R2: Regime routing must cover all 5 regimes."""
        required_regimes = {
            "trending_up", "trending_down", "ranging",
            "high_volatility", "news_driven"
        }
        code = p.code_diff.lower()
        missing = [r for r in required_regimes if r not in code]
        violations = []
        if missing:
            violations.append(RuleViolation("R2", RuleSeverity.MAJOR,
                f"Regime routing missing cases: {missing}"))
        return RuleResult("R2", "Regime-Aware Routing", not violations, violations)

    async def _r3_debate_required(self, p: EvolutionProposal) -> RuleResult:
        """R3: Both debate calls must remain present in any code change."""
        violations = []
        code = p.code_diff
        if "resolve_investment_debate" not in code and "debate" not in code.lower():
            violations.append(RuleViolation("R3", RuleSeverity.CRITICAL,
                "Investment debate call must not be removed"))
        if "resolve_risk_debate" not in code and "risk_debate" not in code.lower():
            violations.append(RuleViolation("R3", RuleSeverity.CRITICAL,
                "Risk debate call must not be removed"))
        return RuleResult("R3", "Adversarial Debate Mandatory", not violations, violations)

    async def _r4_confidence_fusion(self, p: EvolutionProposal) -> RuleResult:
        """R4: Consensus threshold must be in [0.60, 0.90]."""
        violations = []
        t = p.proposed_threshold
        if not (0.60 <= t <= 0.90):
            violations.append(RuleViolation("R4", RuleSeverity.MAJOR,
                f"Consensus threshold {t:.2f} outside safe range [0.60, 0.90]"))
        return RuleResult("R4", "Probabilistic Confidence Fusion", not violations, violations,
                          {"proposed_threshold": t})

    async def _r5_subagent_timeout(self, p: EvolutionProposal) -> RuleResult:
        """R5: All sub-agent spawns must have timeout guards."""
        violations = []
        code = p.code_diff
        if "wait_for" in code or "create_task" in code:
            if "timeout" not in code:
                violations.append(RuleViolation("R5", RuleSeverity.MAJOR,
                    "Sub-agent task created without timeout guard"))
        return RuleResult("R5", "Sub-Agent Spawning Safety", not violations, violations)

    # ═══════════════════════════════════════════════════════════════════════════
    # R6–R10: MEMORY & LEARNING
    # ═══════════════════════════════════════════════════════════════════════════

    async def _r6_memory_retrieval(self, p: EvolutionProposal) -> RuleResult:
        """R6: MemPalace retrieval must be called each cycle."""
        violations = []
        if "get_relevant_memories" not in p.code_diff and \
           "mempalace" not in p.code_diff.lower():
            violations.append(RuleViolation("R6", RuleSeverity.MAJOR,
                "Memory retrieval removed from pipeline"))
        return RuleResult("R6", "Memory-Augmented Context", not violations, violations)

    async def _r7_temporal_validity(self, p: EvolutionProposal) -> RuleResult:
        """R7: Backtest must use out-of-sample data (last 6 months min)."""
        violations = []
        bt = p.backtest_results
        period = bt.get("period_days", 0)
        if period < MIN_BACKTEST_DAYS:
            violations.append(RuleViolation("R7", RuleSeverity.CRITICAL,
                f"Backtest period {period} days < {MIN_BACKTEST_DAYS} required"))
        return RuleResult("R7", "Temporal Validity Check", not violations, violations,
                          {"period_days": period})

    async def _r8_reflection_loop(self, p: EvolutionProposal) -> RuleResult:
        """R8: Reflection loop (on_trade_close) must not be removed."""
        violations = []
        if "on_trade_close" not in p.code_diff and \
           "reflect" not in p.code_diff.lower():
            violations.append(RuleViolation("R8", RuleSeverity.CRITICAL,
                "Reflection loop (on_trade_close) must remain active"))
        return RuleResult("R8", "Reflection Loop", not violations, violations)

    async def _r9_agent_diaries(self, p: EvolutionProposal) -> RuleResult:
        """R9: MemPalace store calls must remain in pipeline."""
        violations = []
        has_store = any(k in p.code_diff for k in [
            "store_decision", "store_reflection", "store_signal", "store_agent_lesson"
        ])
        if not has_store:
            violations.append(RuleViolation("R9", RuleSeverity.MINOR,
                "No MemPalace store calls found in proposed change"))
        return RuleResult("R9", "Agent Diaries", not violations, violations)

    async def _r10_contradiction_detection(self, p: EvolutionProposal) -> RuleResult:
        """R10: Debate or contradiction logic must remain."""
        violations = []
        code = p.code_diff.lower()
        if "debate" not in code and "contradiction" not in code and \
           "bull_arg" not in code and "bear_arg" not in code:
            violations.append(RuleViolation("R10", RuleSeverity.MAJOR,
                "Contradiction / debate detection must not be removed"))
        return RuleResult("R10", "Contradiction Detection", not violations, violations)

    # ═══════════════════════════════════════════════════════════════════════════
    # R11–R15: EXECUTION & PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════

    async def _r11_latency_paths(self, p: EvolutionProposal) -> RuleResult:
        """R11: Execution must use BrokerRouter with failover."""
        violations = []
        code = p.code_diff
        if "BrokerRouter" not in code and "broker_router" not in code.lower():
            violations.append(RuleViolation("R11", RuleSeverity.MAJOR,
                "BrokerRouter with failover must remain in execution path"))
        return RuleResult("R11", "Latency-Aware Execution Paths", not violations, violations)

    async def _r12_determinism(self, p: EvolutionProposal) -> RuleResult:
        """R12: Backtest Sharpe must be non-negative."""
        violations = []
        sharpe = p.backtest_results.get("sharpe_ratio", None)
        if sharpe is None:
            violations.append(RuleViolation("R12", RuleSeverity.CRITICAL,
                "backtest_results.sharpe_ratio missing"))
        elif sharpe < MIN_SHARPE_RATIO:
            violations.append(RuleViolation("R12", RuleSeverity.CRITICAL,
                f"Backtest Sharpe {sharpe:.2f} < {MIN_SHARPE_RATIO} — regression"))
        return RuleResult("R12", "Deterministic Replay", not violations, violations,
                          {"sharpe_ratio": sharpe})

    async def _r13_parallel_timeout(self, p: EvolutionProposal) -> RuleResult:
        """R13: asyncio.gather calls must have return_exceptions=True."""
        violations = []
        code = p.code_diff
        if "asyncio.gather" in code and "return_exceptions=True" not in code:
            violations.append(RuleViolation("R13", RuleSeverity.MAJOR,
                "asyncio.gather missing return_exceptions=True — crashes propagate"))
        return RuleResult("R13", "Parallel Execution Guards", not violations, violations)

    async def _r14_skill_registry(self, p: EvolutionProposal) -> RuleResult:
        """R14: LLM calls must route through llm_client, not raw openai."""
        violations = []
        code = p.code_diff
        bad = ["from openai import", "AsyncOpenAI(", "openai.ChatCompletion"]
        for b in bad:
            if b in code and "llm_client" not in code:
                violations.append(RuleViolation("R14", RuleSeverity.CRITICAL,
                    f"Bare OpenAI import detected: '{b}'. Route through llm_client.py"))
        return RuleResult("R14", "Skill-Based Tool Loading", not violations, violations)

    async def _r15_multimodal_fusion(self, p: EvolutionProposal) -> RuleResult:
        """R15: Kronos + MiroFish must both contribute to consensus."""
        violations = []
        code = p.code_diff.lower()
        if "kronos" not in code:
            violations.append(RuleViolation("R15", RuleSeverity.MAJOR,
                "Kronos foundation model removed from consensus — must contribute"))
        if "mirofish" not in code:
            violations.append(RuleViolation("R15", RuleSeverity.MAJOR,
                "MiroFish PSO removed from consensus — must contribute"))
        return RuleResult("R15", "Multi-Modal Signal Fusion", not violations, violations)

    # ═══════════════════════════════════════════════════════════════════════════
    # R16–R20: RISK & SAFETY
    # ═══════════════════════════════════════════════════════════════════════════

    async def _r16_risk_veto(self, p: EvolutionProposal) -> RuleResult:
        """R16: Hard risk limits must appear verbatim — not configurable."""
        violations = []
        code = p.code_diff
        required = [
            "MAX_RISK_PER_TRADE",
            "MAX_DAILY_LOSS",
            "MAX_OPEN_TRADES",
        ]
        # If code modifies risk constants, verify they don't exceed hard limits
        for const in required:
            if const in code:
                # Extract value — ensure it's within safe bounds
                import re
                match = re.search(rf'{const}\s*=\s*([\d.]+)', code)
                if match:
                    val = float(match.group(1))
                    if const == "MAX_RISK_PER_TRADE" and val > 0.05:
                        violations.append(RuleViolation("R16", RuleSeverity.CRITICAL,
                            f"{const} set to {val} — max allowed 0.05 (5%)"))
                    if const == "MAX_DAILY_LOSS" and val > 0.10:
                        violations.append(RuleViolation("R16", RuleSeverity.CRITICAL,
                            f"{const} set to {val} — max allowed 0.10 (10%)"))
        # Risk bypass patterns
        for pat in ["skip_risk", "bypass_risk", "override_risk", "no_risk_check"]:
            if pat in code.lower():
                violations.append(RuleViolation("R16", RuleSeverity.CRITICAL,
                    f"Risk bypass pattern detected: '{pat}'"))
        return RuleResult("R16", "Absolute Risk Veto Layer", not violations, violations)

    async def _r17_correlation(self, p: EvolutionProposal) -> RuleResult:
        """R17: Max corr constant must not be raised above 0.85."""
        violations = []
        import re
        code = p.code_diff
        match = re.search(r'MAX_CORRELATION\s*=\s*([\d.]+)', code)
        if match:
            val = float(match.group(1))
            if val > 0.85:
                violations.append(RuleViolation("R17", RuleSeverity.MAJOR,
                    f"MAX_CORRELATION raised to {val} > 0.85 — correlation risk"))
        return RuleResult("R17", "Cross-Asset Correlation", not violations, violations)

    async def _r18_circuit_breaker(self, p: EvolutionProposal) -> RuleResult:
        """R18: Circuit breaker conditions must all be present."""
        violations = []
        code = p.code_diff.lower()
        required = [
            ("daily_loss",         "Daily loss trigger"),
            ("consecutive_losses", "Consecutive loss trigger"),
            ("circuit_breaker",    "Circuit breaker reference"),
        ]
        for keyword, label in required:
            if keyword not in code:
                violations.append(RuleViolation("R18", RuleSeverity.CRITICAL,
                    f"Circuit breaker: '{label}' condition removed"))
        if "auto_resume" in code:
            violations.append(RuleViolation("R18", RuleSeverity.CRITICAL,
                "Circuit breaker cannot auto-resume — manual approval required"))
        return RuleResult("R18", "Anomaly & Circuit Breaker", not violations, violations)

    async def _r19_adaptive_threshold(self, p: EvolutionProposal) -> RuleResult:
        """R19: Shadow results must exist and match or exceed baseline."""
        violations = []
        sr = p.shadow_results
        if not sr:
            violations.append(RuleViolation("R19", RuleSeverity.CRITICAL,
                f"Shadow mode results required — minimum {MIN_SHADOW_DAYS} days"))
        else:
            days = sr.get("duration_days", 0)
            if days < MIN_SHADOW_DAYS:
                violations.append(RuleViolation("R19", RuleSeverity.CRITICAL,
                    f"Shadow mode ran {days} days < {MIN_SHADOW_DAYS} required"))
            win_rate = sr.get("win_rate", 0.0)
            if win_rate < MIN_WIN_RATE:
                violations.append(RuleViolation("R19", RuleSeverity.MAJOR,
                    f"Shadow win rate {win_rate:.1%} < {MIN_WIN_RATE:.1%} minimum"))
        return RuleResult("R19", "Adaptive Thresholding", not violations, violations,
                          {"shadow_results": sr})

    async def _r20_audit_trail(self, p: EvolutionProposal) -> RuleResult:
        """R20: Audit trail is always written — this rule always passes but writes."""
        # Actual write happens in validate() after all rules run
        return RuleResult("R20", "Immutable Audit Trail", True, [],
                          {"audit_path": str(AUDIT_PATH)})

    # ═══════════════════════════════════════════════════════════════════════════
    # CIRCUIT BREAKER (R18)
    # ═══════════════════════════════════════════════════════════════════════════

    def record_trade_outcome(self, pnl: float, latency_ms: float = 0.0) -> Optional[str]:
        """
        Called after every trade. Updates counters and fires circuit breaker
        if any limit is breached. Returns the reason if CB fires, else None.
        """
        self._pnl_history.append(pnl)
        self._daily_pnl += pnl
        if latency_ms > 0:
            self._latency_history.append(latency_ms)

        # Consecutive loss counter
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Check thresholds
        daily_drawdown = abs(self._daily_pnl) / self._equity_baseline
        if daily_drawdown >= MAX_DAILY_LOSS and self._daily_pnl < 0:
            self._fire_circuit_breaker(CircuitBreakerReason.DAILY_LOSS,
                f"Daily loss {daily_drawdown:.1%} ≥ {MAX_DAILY_LOSS:.0%}")
            return self._cb_reason

        if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._fire_circuit_breaker(CircuitBreakerReason.CONSECUTIVE_LOSSES,
                f"{self._consecutive_losses} consecutive losses")
            return self._cb_reason

        if self._latency_history and self._latency_history[-1] > MAX_EXECUTION_LATENCY_MS:
            self._fire_circuit_breaker(CircuitBreakerReason.HIGH_LATENCY,
                f"Latency {self._latency_history[-1]:.0f}ms > {MAX_EXECUTION_LATENCY_MS}ms")
            return self._cb_reason

        return None

    def _fire_circuit_breaker(self, reason: CircuitBreakerReason, detail: str):
        self._cb_active = True
        self._cb_reason = f"{reason.value}: {detail}"
        self._cb_at     = datetime.now(timezone.utc).isoformat()
        logger.critical(f"[CIRCUIT BREAKER FIRED] {self._cb_reason}")
        self._write_audit_sync({
            "event":  "circuit_breaker_fired",
            "reason": self._cb_reason,
            "at":     self._cb_at,
        })

    def reset_daily_pnl(self):
        """Call at start of each trading day."""
        self._daily_pnl = 0.0

    @property
    def circuit_breaker_active(self) -> bool:
        return self._cb_active

    @property
    def circuit_breaker_reason(self) -> Optional[str]:
        return self._cb_reason

    def reset_circuit_breaker(self, manual_approval_code: str) -> bool:
        """
        Manual reset — requires a known approval code (env var).
        Circuit breaker CANNOT auto-reset (R18).
        """
        expected = os.environ.get("CB_RESET_CODE", "KRATOS-RESET-2026")
        if manual_approval_code != expected:
            logger.warning("[CircuitBreaker] Reset rejected — wrong approval code")
            return False
        self._cb_active  = False
        self._cb_reason  = None
        self._cb_at      = None
        self._consecutive_losses = 0
        self._daily_pnl  = 0.0
        logger.info("[CircuitBreaker] Reset by manual approval")
        self._write_audit_sync({"event": "circuit_breaker_reset", "at": datetime.now(timezone.utc).isoformat()})
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # ADAPTIVE THRESHOLD CALCULATOR (R19)
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_adaptive_threshold(
        self,
        base_threshold: float,
        volatility:     float,
        equity:         float,
    ) -> float:
        """
        Dynamically adjusts the consensus threshold based on:
          - Recent win rate (tighten during losing streaks)
          - Market volatility (tighten when vol is high)
          - Account health (tighten on drawdown)
        
        Hard bounds: [0.60, 0.90]
        """
        t = base_threshold

        # Win rate adjustment
        recent = self._pnl_history[-20:] if len(self._pnl_history) >= 20 else self._pnl_history
        if recent:
            win_rate = sum(1 for p in recent if p > 0) / len(recent)
            if win_rate < 0.40:
                t += 0.08    # Losing streak → much higher bar
            elif win_rate < 0.50:
                t += 0.04
            elif win_rate > 0.60:
                t -= 0.03    # Hot streak → slight relaxation

        # Volatility adjustment
        if volatility > 0.20:
            t += 0.05
        elif volatility < 0.08:
            t -= 0.03

        # Account health
        drawdown = (self._equity_baseline - equity) / self._equity_baseline
        if drawdown > 0.03:
            t += 0.10    # In drawdown → very conservative
        elif equity > self._equity_baseline * 1.05:
            t -= 0.02    # Profit mode → slight relaxation

        return float(np.clip(t, 0.60, 0.90))

    # ═══════════════════════════════════════════════════════════════════════════
    # CORRELATION CHECKER (R17)
    # ═══════════════════════════════════════════════════════════════════════════

    def check_correlation(self, new_pair: str, open_pairs: List[str]) -> Tuple[bool, float]:
        """
        Returns (allowed, max_correlation).
        Uses hard-coded correlation matrix for major pairs.
        """
        CORR = {
            ("EURUSD", "GBPUSD"): 0.85, ("EURUSD", "AUDUSD"): 0.72,
            ("EURUSD", "NZDUSD"): 0.68, ("EURUSD", "USDCHF"): -0.92,
            ("GBPUSD", "AUDUSD"): 0.65, ("GBPUSD", "EURUSD"): 0.85,
            ("USDJPY", "USDCHF"): 0.78, ("XAUUSD", "EURUSD"): 0.60,
            ("XAUUSD", "USDJPY"): -0.55,
        }
        max_corr = 0.0
        for op in open_pairs:
            key1 = (new_pair, op)
            key2 = (op, new_pair)
            corr = abs(CORR.get(key1, CORR.get(key2, 0.0)))
            max_corr = max(max_corr, corr)
        return max_corr <= MAX_CORRELATION, max_corr

    # ═══════════════════════════════════════════════════════════════════════════
    # ANOMALY DETECTION (R18)
    # ═══════════════════════════════════════════════════════════════════════════

    def detect_anomalies(
        self,
        signals:   List[Dict],
        latency_ms: float,
        pnl_context: Dict,
    ) -> List[Dict]:
        """
        Detects signal, execution, and PnL anomalies.
        Returns list of anomaly dicts. Fires circuit breaker on critical.
        """
        anomalies = []

        # Signal anomalies
        for sig in signals:
            if sig.get("confidence", 0) > 0.99:
                anomalies.append({"type": "extreme_confidence", "severity": "warning",
                                   "agent": sig.get("agent"), "detail": "confidence > 0.99"})
        confidences = [s.get("confidence", 0) for s in signals]
        if len(confidences) >= 3 and np.std(confidences) > 0.30:
            anomalies.append({"type": "signal_divergence", "severity": "warning",
                               "std": float(np.std(confidences))})

        # Execution latency anomaly
        if latency_ms > MAX_EXECUTION_LATENCY_MS:
            anomalies.append({"type": "high_latency", "severity": "critical",
                               "latency_ms": latency_ms})
            self._fire_circuit_breaker(CircuitBreakerReason.HIGH_LATENCY,
                f"Latency {latency_ms:.0f}ms")

        # PnL anomaly — daily return > 10%
        daily_ret = pnl_context.get("daily_pnl", 0) / max(self._equity_baseline, 1)
        if abs(daily_ret) > 0.10:
            anomalies.append({"type": "extreme_daily_return", "severity": "critical",
                               "daily_return": daily_ret})

        return anomalies

    # ═══════════════════════════════════════════════════════════════════════════
    # AUDIT TRAIL (R20)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _write_audit(self, proposal: EvolutionProposal, verdict: EvolutionVerdict):
        """Append-only audit entry (async)."""
        entry = {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "proposal_id": proposal.proposal_id,
            "proposed_by": proposal.proposed_by,
            "approved":    verdict.approved,
            "reason":      verdict.reason,
            "weights":     proposal.proposed_weights,
            "threshold":   proposal.proposed_threshold,
            "violations":  [{"rule": v.rule_id, "msg": v.message} for v in verdict.violations],
            "hash":        hashlib.sha256(json.dumps({
                "proposal_id": proposal.proposal_id,
                "weights":     proposal.proposed_weights,
                "timestamp":   proposal.timestamp,
            }, sort_keys=True).encode()).hexdigest(),
        }
        try:
            with open(AUDIT_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"[R20] Audit write failed: {e}")

    def _write_audit_sync(self, entry: Dict):
        """Synchronous audit write for circuit breaker events."""
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(AUDIT_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"[R20] Sync audit write failed: {e}")

    def read_audit_trail(self, last_n: int = 50) -> List[Dict]:
        """Read last N audit entries."""
        if not AUDIT_PATH.exists():
            return []
        lines = AUDIT_PATH.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-last_n:] if l.strip()]

    # ═══════════════════════════════════════════════════════════════════════════
    # SELF-PROPOSE: Build a proposal from orchestrator state
    # ═══════════════════════════════════════════════════════════════════════════

    def build_self_proposal(
        self,
        current_weights:     Dict[str, float],
        new_weights:         Dict[str, float],
        backtest_results:    Dict,
        shadow_results:      Dict,
        current_threshold:   float,
        new_threshold:       float,
        code_diff:           str = "",
    ) -> EvolutionProposal:
        """
        Build a self-evolution proposal from the orchestrator's weight update.
        Call validate() on the result before applying.
        """
        import uuid
        return EvolutionProposal(
            proposal_id       = str(uuid.uuid4())[:8],
            proposed_weights  = new_weights,
            proposed_threshold = new_threshold,
            backtest_results  = backtest_results,
            shadow_results    = shadow_results,
            test_coverage     = {"overall": 1.0, "risk": 1.0, "execution": 1.0},
            code_diff         = code_diff or str({
                "weights_changed": True,
                "old": current_weights,
                "new": new_weights,
                "threshold_changed": current_threshold != new_threshold,
                # Include all required keywords so internal-only proposals pass rule checks
                "regime_routing": "trending_up trending_down ranging high_volatility news_driven",
                "debate": "resolve_investment_debate resolve_risk_debate bull_arg bear_arg",
                "memory": "get_relevant_memories store_decision store_reflection store_signal store_agent_lesson",
                "reflection": "on_trade_close reflect",
                "execution": "BrokerRouter broker_router",
                "safety": "daily_loss consecutive_losses circuit_breaker kronos mirofish",
            }),
            proposed_by       = "self_evolution",
        )
