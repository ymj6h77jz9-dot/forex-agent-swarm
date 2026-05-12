"""
SHADOW GATE — KRATOS v2 CI/CD
================================
R12: Shadow Mode Before Production.
Any weight change must run in shadow mode (paper trading alongside live)
for a minimum of 7 days before going live.

Shadow mode: KRATOS runs both old and new weights simultaneously.
Old weights execute real trades. New weights execute paper trades.
After MIN_SHADOW_DAYS, if paper ≥ live within tolerance → approved.
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_SHADOW_DAYS  = 7
WIN_RATE_TOL     = 0.90   # Shadow win rate must be ≥ 90% of live
LATENCY_TOL      = 1.20   # Shadow latency must be ≤ 120% of live
MIN_SHADOW_TRADES = 20    # Minimum trades to be statistically meaningful


@dataclass
class ShadowViolation:
    metric:   str
    shadow:   float
    live:     float
    message:  str


@dataclass
class ShadowGateResult:
    passed:     bool
    violations: List[ShadowViolation]
    shadow:     Dict
    live:       Dict
    report:     str


class ShadowGate:
    """R12: Shadow Mode Validation Gate."""

    def validate(
        self,
        shadow_results: Dict[str, Any],
        live_results:   Dict[str, Any],
    ) -> ShadowGateResult:
        violations: List[ShadowViolation] = []

        # C1: Minimum duration
        days = int(shadow_results.get("duration_days", 0))
        if days < MIN_SHADOW_DAYS:
            violations.append(ShadowViolation(
                "duration_days", days, MIN_SHADOW_DAYS,
                f"Shadow ran {days} days < {MIN_SHADOW_DAYS} required"
            ))

        # C2: Minimum trades
        trades = int(shadow_results.get("total_trades", 0))
        if trades < MIN_SHADOW_TRADES:
            violations.append(ShadowViolation(
                "total_trades", trades, MIN_SHADOW_TRADES,
                f"Shadow only {trades} trades < {MIN_SHADOW_TRADES} minimum"
            ))

        # C3: Win rate parity
        shadow_wr = float(shadow_results.get("win_rate", 0.0))
        live_wr   = float(live_results.get("win_rate", 0.50))
        if shadow_wr < live_wr * WIN_RATE_TOL:
            violations.append(ShadowViolation(
                "win_rate", shadow_wr, live_wr,
                f"Shadow win rate {shadow_wr:.1%} < {WIN_RATE_TOL:.0%} of live {live_wr:.1%}"
            ))

        # C4: Latency parity
        shadow_lat = float(shadow_results.get("avg_latency_ms", 0.0))
        live_lat   = float(live_results.get("avg_latency_ms", 100.0))
        if shadow_lat > live_lat * LATENCY_TOL:
            violations.append(ShadowViolation(
                "latency_ms", shadow_lat, live_lat,
                f"Shadow latency {shadow_lat:.1f}ms > {LATENCY_TOL:.0%} of live {live_lat:.1f}ms"
            ))

        # C5: No circuit breaker triggers during shadow
        cb_triggers = int(shadow_results.get("circuit_breaker_triggers", 0))
        if cb_triggers > 0:
            violations.append(ShadowViolation(
                "circuit_breaker_triggers", cb_triggers, 0,
                f"Shadow triggered circuit breaker {cb_triggers}x — new weights unsafe"
            ))

        passed = len(violations) == 0
        report = self._build_report(shadow_results, live_results, violations, passed)

        return ShadowGateResult(passed, violations, shadow_results, live_results, report)

    def _build_report(self, shadow, live, violations, passed) -> str:
        lines = [
            f"ShadowGate {'PASS' if passed else 'FAIL'}",
            f"  Duration:      {shadow.get('duration_days','?')} days",
            f"  Shadow trades: {shadow.get('total_trades','?')}",
            f"  Shadow win %:  {shadow.get('win_rate',0):.1%}  (live: {live.get('win_rate',0):.1%})",
            f"  Shadow lat:    {shadow.get('avg_latency_ms',0):.1f}ms  (live: {live.get('avg_latency_ms',0):.1f}ms)",
            f"  CB triggers:   {shadow.get('circuit_breaker_triggers',0)}",
        ]
        if violations:
            lines.append(f"\n  Violations ({len(violations)}):")
            for v in violations:
                lines.append(f"    ✗ {v.message}")
        return "\n".join(lines)


if __name__ == "__main__":
    shadow = {"duration_days": 10, "total_trades": 35, "win_rate": 0.54,
              "avg_latency_ms": 45.0, "circuit_breaker_triggers": 0}
    live   = {"win_rate": 0.52, "avg_latency_ms": 42.0}
    gate   = ShadowGate()
    result = gate.validate(shadow, live)
    print(result.report)
    sys.exit(0 if result.passed else 1)
