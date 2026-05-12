"""
BACKTEST GATE — KRATOS v2 CI/CD
================================
R11: Backtest Validation Required before any weight change goes live.

Validates:
  - Minimum 180-day out-of-sample period
  - Sharpe ratio ≥ 0.0 (non-negative risk-adjusted returns)
  - Max drawdown ≤ 15%
  - Win rate ≥ 45%
  - No performance regression vs baseline (within 10% tolerance)
  - Baseline comparison stored in ci/baselines/latest.json

Can run standalone or be called from ci/validators.py.
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASELINE_PATH = Path(__file__).parent.parent / "baselines" / "latest.json"

# ── Thresholds (matching R12 in evolution_rules.py) ──────────────────────────
MIN_PERIOD_DAYS  = 180
MIN_SHARPE       = 0.0
MAX_DRAWDOWN     = 0.15
MIN_WIN_RATE     = 0.45
REGRESSION_TOL   = 0.90   # New Sharpe must be ≥ 90% of baseline


@dataclass
class BacktestViolation:
    metric:   str
    value:    float
    limit:    float
    severity: str  # "critical" | "major"
    message:  str


@dataclass
class BacktestGateResult:
    passed:     bool
    violations: List[BacktestViolation]
    metrics:    Dict[str, Any]
    baseline:   Optional[Dict]
    report:     str


class BacktestGate:
    """
    R11: Backtest Validation Gate.

    In CI/CD: called before any orchestrator weight change merges.
    In self-evolution: called by KratosEvolutionEngine._r7_temporal_validity.
    """

    def validate(self, results: Dict[str, Any]) -> BacktestGateResult:
        """
        Validate backtest results against all thresholds.

        results dict must include:
          period_days, sharpe_ratio, max_drawdown, win_rate
        Optional:
          total_trades, profit_factor, avg_trade_pnl
        """
        violations: List[BacktestViolation] = []
        baseline = self._load_baseline()

        # C1: Period
        period = int(results.get("period_days", 0))
        if period < MIN_PERIOD_DAYS:
            violations.append(BacktestViolation(
                "period_days", period, MIN_PERIOD_DAYS, "critical",
                f"Backtest period {period} days < {MIN_PERIOD_DAYS} required (6 months)"
            ))

        # C2: Sharpe
        sharpe = float(results.get("sharpe_ratio", -99))
        if sharpe < MIN_SHARPE:
            violations.append(BacktestViolation(
                "sharpe_ratio", sharpe, MIN_SHARPE, "critical",
                f"Sharpe {sharpe:.3f} < {MIN_SHARPE} — negative risk-adjusted returns"
            ))

        # C3: Max drawdown
        mdd = float(results.get("max_drawdown", 1.0))
        if mdd > MAX_DRAWDOWN:
            violations.append(BacktestViolation(
                "max_drawdown", mdd, MAX_DRAWDOWN, "critical",
                f"Max drawdown {mdd:.1%} > {MAX_DRAWDOWN:.0%} limit"
            ))

        # C4: Win rate
        win_rate = float(results.get("win_rate", 0.0))
        if win_rate < MIN_WIN_RATE:
            violations.append(BacktestViolation(
                "win_rate", win_rate, MIN_WIN_RATE, "major",
                f"Win rate {win_rate:.1%} < {MIN_WIN_RATE:.0%} minimum"
            ))

        # C5: Regression vs baseline
        if baseline:
            baseline_sharpe = float(baseline.get("sharpe_ratio", 0.0))
            if baseline_sharpe > 0 and sharpe < baseline_sharpe * REGRESSION_TOL:
                violations.append(BacktestViolation(
                    "regression", sharpe, baseline_sharpe * REGRESSION_TOL, "major",
                    f"Performance regression: Sharpe {sharpe:.3f} < {REGRESSION_TOL:.0%} of baseline {baseline_sharpe:.3f}"
                ))

        passed = len(violations) == 0
        report = self._build_report(results, violations, baseline, passed)

        return BacktestGateResult(
            passed     = passed,
            violations = violations,
            metrics    = results,
            baseline   = baseline,
            report     = report,
        )

    def save_as_baseline(self, results: Dict[str, Any]):
        """Save passing results as the new performance baseline."""
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        baseline = {
            "sharpe_ratio":  results.get("sharpe_ratio"),
            "max_drawdown":  results.get("max_drawdown"),
            "win_rate":      results.get("win_rate"),
            "period_days":   results.get("period_days"),
            "saved_at":      datetime.utcnow().isoformat(),
        }
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info(f"[BacktestGate] New baseline saved: Sharpe={baseline['sharpe_ratio']}")

    def _load_baseline(self) -> Optional[Dict]:
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH) as f:
                return json.load(f)
        return None

    def _build_report(self, results, violations, baseline, passed) -> str:
        lines = [
            f"BacktestGate {'PASS' if passed else 'FAIL'}",
            f"  Period:     {results.get('period_days', '?')} days",
            f"  Sharpe:     {results.get('sharpe_ratio', '?'):.3f}",
            f"  MaxDD:      {results.get('max_drawdown', '?'):.1%}",
            f"  Win Rate:   {results.get('win_rate', '?'):.1%}",
        ]
        if baseline:
            lines.append(f"  Baseline Sharpe: {baseline.get('sharpe_ratio', '?'):.3f}")
        if violations:
            lines.append(f"\n  Violations ({len(violations)}):")
            for v in violations:
                lines.append(f"    [{v.severity.upper()}] {v.message}")
        return "\n".join(lines)


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "period_days":  180,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.08,
        "win_rate":     0.55,
        "total_trades": 240,
    }
    gate   = BacktestGate()
    result = gate.validate(sample)
    print(result.report)
    sys.exit(0 if result.passed else 1)
