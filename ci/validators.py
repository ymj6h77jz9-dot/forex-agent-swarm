"""
CI/CD VALIDATORS — KRATOS v2
==============================
Pre-merge validation runner for all 20 evolution rules.

Usage:
    python ci/validators.py '{"code_diff": "...", "backtest_results": {...}}'

    Or programmatically:
    result = asyncio.run(run_pre_merge_checks(pr_data))

Exit codes:
    0 = all rules pass (merge approved)
    1 = one or more rules fail (merge blocked)
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "swarm"))

from evolution.rules_engine import (
    KratosEvolutionEngine,
    EvolutionProposal,
)


async def run_pre_merge_checks(pr_data: Dict) -> bool:
    """
    Run all 20 evolution rules against a proposed change.
    Prints human-readable report. Returns True if approved.
    """
    print("\n" + "═" * 65)
    print("  KRATOS v2 — Evolution Rule Validation  (R1–R20)")
    print("═" * 65)

    engine = KratosEvolutionEngine()

    # Build proposal from PR data
    proposal = EvolutionProposal(
        proposal_id        = pr_data.get("proposal_id", f"pr-{int(time.time())}"),
        proposed_weights   = pr_data.get("proposed_weights", {
            "analyst": 0.28, "sentiment": 0.22, "risk": 0.20,
            "mirofish": 0.15, "kronos": 0.15,
        }),
        proposed_threshold = float(pr_data.get("proposed_threshold", 0.70)),
        backtest_results   = pr_data.get("backtest_results", {}),
        shadow_results     = pr_data.get("shadow_results", {}),
        test_coverage      = pr_data.get("coverage", {
            "overall": 1.0, "risk": 1.0, "execution": 1.0
        }),
        code_diff          = pr_data.get("code_diff", pr_data.get("code", "")),
        proposed_by        = pr_data.get("proposed_by", "ci_pipeline"),
    )

    verdict = await engine.validate(proposal)

    # Print results
    print(f"\n  Proposal:  {proposal.proposal_id}")
    print(f"  Proposed by: {proposal.proposed_by}")
    print(f"  Timestamp: {proposal.timestamp}\n")

    passed_rules = 0
    failed_rules = 0

    for rule_id, result in verdict.results.items():
        status     = "✅ PASS" if result.passed else "❌ FAIL"
        rule_name  = getattr(result, "rule_name", rule_id)
        print(f"  {rule_id:4s} {status}  {rule_name}")
        if not result.passed:
            failed_rules += 1
            for v in result.violations:
                print(f"         ⚠  [{v.severity.upper()}] {v.message}")
        else:
            passed_rules += 1

    print(f"\n{'═' * 65}")
    print(f"  Results: {passed_rules}/20 passed | {failed_rules} violation(s)")
    print(f"  Verdict: {'✅  APPROVED — merge cleared' if verdict.approved else '🔴  VETOED — merge blocked'}")
    print(f"  Reason:  {verdict.reason}")
    print("═" * 65 + "\n")

    return verdict.approved


async def run_shadow_comparison(shadow_results: Dict, prod_results: Dict) -> bool:
    """
    R19: Compare shadow mode metrics against production baseline.
    Shadow must match or exceed production within tolerance.
    """
    print("\n── Shadow Mode Comparison (R19) ──")
    checks = []

    shadow_wr = float(shadow_results.get("win_rate", 0))
    prod_wr   = float(prod_results.get("win_rate", 0.50))
    wr_ok     = shadow_wr >= prod_wr * 0.90
    checks.append(("Win Rate", shadow_wr, prod_wr, wr_ok))

    shadow_lat = float(shadow_results.get("avg_latency_ms", 999))
    prod_lat   = float(prod_results.get("avg_latency_ms", 100))
    lat_ok     = shadow_lat <= prod_lat * 1.20
    checks.append(("Avg Latency (ms)", shadow_lat, prod_lat, lat_ok))

    shadow_days = int(shadow_results.get("duration_days", 0))
    days_ok     = shadow_days >= 7
    checks.append(("Shadow Duration (days)", shadow_days, 7, days_ok))

    all_ok = all(c[3] for c in checks)
    for label, shadow_val, prod_val, ok in checks:
        sym = "✅" if ok else "❌"
        print(f"  {sym} {label}: shadow={shadow_val} vs prod={prod_val}")

    print(f"\n  Result: {'✅ Shadow approved for production' if all_ok else '🔴 Shadow comparison failed'}")
    return all_ok


async def run_performance_gate(benchmarks: Dict) -> bool:
    """
    R4: Verify performance budgets.
    """
    print("\n── Performance Gate (R4) ──")
    BUDGETS = {
        "api_latency_p95_ms":   100.0,
        "execution_latency_ms":  20.0,
        "signal_generation_ms": 200.0,
        "memory_retrieval_ms":   50.0,
    }
    all_ok = True
    for metric, budget in BUDGETS.items():
        val = float(benchmarks.get(metric, 0.0))
        ok  = val <= budget
        if not ok:
            all_ok = False
        sym = "✅" if ok else "❌"
        print(f"  {sym} {metric}: {val:.1f}ms (budget={budget:.0f}ms)")

    print(f"\n  Result: {'✅ Performance budget met' if all_ok else '🔴 Performance budget exceeded'}")
    return all_ok


if __name__ == "__main__":
    raw    = sys.argv[1] if len(sys.argv) > 1 else "{}"
    data   = json.loads(raw)
    result = asyncio.run(run_pre_merge_checks(data))
    sys.exit(0 if result else 1)
