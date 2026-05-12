"""
RISK ENGINE — KRATOS v2
=========================
Implements R16–R20: Risk & Safety

  R16 Absolute Risk Veto Layer (never bypassed by LLM)
  R17 Cross-Asset Correlation Awareness
  R18 Anomaly Detection & Circuit Breakers
  R19 Adaptive Thresholding
  R20 Immutable Audit Trail (delegates to KratosEvolutionEngine)

This module is the last gate before any order reaches the broker.
No LLM, no agent vote, no orchestrator override can bypass R16.
Hard limits are defined as module-level constants — not config.

Integrated from:
  - ymj6h77jz9-dot/KRATOS-app (RiskManager)
  - evolution/rules_engine.py  (CB state shared)

Usage:
    engine = RiskEngine(equity=10000)
    validation = engine.validate_trade(trade, portfolio)
    if validation.approved:
        await broker.place_order(...)
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── HARD LIMITS — NEVER configurable ─────────────────────────────────────────
MAX_RISK_PER_TRADE     = 0.02    # 2% max per trade
MAX_DAILY_LOSS         = 0.05    # 5% daily drawdown → circuit breaker
MAX_PORTFOLIO_EXPOSURE = 0.15    # 15% of equity in open positions
MAX_CONSECUTIVE_LOSSES = 5       # 5 straight losses → halt
MAX_LEVERAGE           = 30.0    # Max broker leverage used
MAX_SPREAD_PIPS        = 5.0     # Never trade above 5 pips spread (3 for majors)
MAX_CORRELATION        = 0.80    # Max correlation between open pairs
MIN_ATR                = 0.00010 # Don't trade in dead markets
MAX_LATENCY_MS         = 500.0   # Execution latency circuit breaker


# ── Data structures ───────────────────────────────────────────────────────────

class RiskCheckStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    WARNING  = "warning"


@dataclass
class RiskValidation:
    approved:    bool
    checks:      Dict[str, bool]
    rejections:  List[str]
    warnings:    List[str]
    risk_score:  float
    lot_adj:     float = 1.0   # Lot size multiplier from risk engine
    sl_adj_pips: float = 0.0   # Extra SL distance in pips


@dataclass
class AnomalyEvent:
    type:      str
    severity:  str     # "critical" | "warning"
    detail:    str
    value:     float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Correlation matrix — major forex pairs ────────────────────────────────────
# Source: 2020–2025 historical pair correlations (approximate)
PAIR_CORRELATION: Dict[Tuple[str, str], float] = {
    ("EURUSD", "GBPUSD"): 0.85, ("EURUSD", "AUDUSD"): 0.72,
    ("EURUSD", "NZDUSD"): 0.68, ("EURUSD", "USDCHF"): -0.92,
    ("EURUSD", "USDJPY"): -0.45,("EURUSD", "XAUUSD"):  0.60,
    ("GBPUSD", "AUDUSD"): 0.65, ("GBPUSD", "NZDUSD"):  0.62,
    ("GBPUSD", "USDCHF"):-0.80, ("GBPUSD", "USDJPY"): -0.40,
    ("AUDUSD", "NZDUSD"): 0.90, ("AUDUSD", "USDCAD"): -0.70,
    ("USDJPY", "USDCHF"): 0.78, ("USDJPY", "USDCAD"):  0.55,
    ("XAUUSD", "USDJPY"):-0.55, ("XAUUSD", "USDCHF"):  0.45,
    ("USDCAD", "XAUUSD"):-0.60,
}


def _get_correlation(pair_a: str, pair_b: str) -> float:
    """Look up pair correlation (bidirectional)."""
    key1 = (pair_a, pair_b)
    key2 = (pair_b, pair_a)
    return abs(PAIR_CORRELATION.get(key1, PAIR_CORRELATION.get(key2, 0.0)))


# ── Risk Engine ───────────────────────────────────────────────────────────────

class RiskEngine:
    """
    R16–R20 Risk & Safety gate.

    This is the FINAL check before execution. Its veto CANNOT be bypassed.
    All hard limits are baked into constants above — not environment variables.
    """

    def __init__(self, equity: float = 10_000.0):
        self.equity              = equity
        self.initial_equity      = equity
        self._daily_pnl:  float  = 0.0
        self._consecutive: int   = 0
        self._pnl_history: List[float] = []
        self._anomalies:   List[AnomalyEvent] = []
        self._cb_active:   bool  = False
        self._cb_reason:   Optional[str] = None
        logger.info(f"[RiskEngine] Initialised | equity=${equity:,.0f} | limits hardcoded")

    # ─────────────────────────────────────────────────────────────────────────
    # R16: Absolute Risk Veto Layer
    # ─────────────────────────────────────────────────────────────────────────

    def validate_trade(
        self,
        trade:       Dict[str, Any],
        portfolio:   Dict[str, Any],
        open_pairs:  Optional[List[str]] = None,
    ) -> RiskValidation:
        """
        Run all risk checks. Returns RiskValidation.
        This is the ABSOLUTE veto — approved=False means NO trade.

        trade:     {pair, direction, lot_size, stop_loss, take_profit, price, spread, atr}
        portfolio: {balance, equity, open_trades, daily_pnl}
        open_pairs: list of currently open pair strings
        """
        checks     = {}
        rejections = []
        warnings   = []
        lot_adj    = 1.0
        sl_adj     = 0.0

        balance    = float(portfolio.get("equity", self.equity))
        pair       = trade.get("pair", "")
        price      = float(trade.get("price", 1.0))
        atr        = float(trade.get("atr", 0.001))
        spread     = float(trade.get("spread", 0.0))
        lot        = float(trade.get("lot_size", 0.01))
        open_count = int(portfolio.get("open_trades", 0))

        # ── C1: Circuit breaker ──────────────────────────────────────────────
        checks["circuit_breaker"] = not self._cb_active
        if self._cb_active:
            rejections.append(f"Circuit breaker ACTIVE: {self._cb_reason}")

        # ── C2: Daily loss limit ─────────────────────────────────────────────
        daily_loss_pct = abs(self._daily_pnl) / max(balance, 1.0)
        daily_ok       = not (self._daily_pnl < 0 and daily_loss_pct >= MAX_DAILY_LOSS)
        checks["daily_loss"] = daily_ok
        if not daily_ok:
            rejections.append(f"Daily loss {daily_loss_pct:.1%} ≥ {MAX_DAILY_LOSS:.0%} limit")

        # ── C3: Consecutive losses ───────────────────────────────────────────
        consec_ok = self._consecutive < MAX_CONSECUTIVE_LOSSES
        checks["consecutive_losses"] = consec_ok
        if not consec_ok:
            rejections.append(f"{self._consecutive} consecutive losses ≥ {MAX_CONSECUTIVE_LOSSES} limit")

        # ── C4: Position size (2% max risk) ─────────────────────────────────
        pip        = 0.01 if "JPY" in pair else 0.0001
        sl_dist    = abs(price - float(trade.get("stop_loss", price - atr * 1.5)))
        sl_pips    = sl_dist / pip
        pip_val    = 1000.0 if "JPY" in pair else 10.0
        trade_risk = lot * sl_pips * pip_val
        risk_pct   = trade_risk / max(balance, 1.0)
        size_ok    = risk_pct <= MAX_RISK_PER_TRADE
        checks["position_size"] = size_ok
        if not size_ok:
            rejections.append(f"Position risk {risk_pct:.2%} > {MAX_RISK_PER_TRADE:.0%} max")
            # Suggest reduced lot
            max_lot  = (balance * MAX_RISK_PER_TRADE) / max(sl_pips * pip_val, 1.0)
            lot_adj  = min(lot_adj, max_lot / max(lot, 0.001))

        # ── C5: Portfolio exposure (15% max) ─────────────────────────────────
        exposure    = float(portfolio.get("total_exposure_pct", 0.0))
        exposure_ok = exposure + risk_pct <= MAX_PORTFOLIO_EXPOSURE
        checks["portfolio_exposure"] = exposure_ok
        if not exposure_ok:
            warnings.append(f"Portfolio exposure {exposure + risk_pct:.1%} > {MAX_PORTFOLIO_EXPOSURE:.0%}")
            lot_adj = min(lot_adj, 0.50)  # Reduce but don't reject

        # ── C6: Spread gate ──────────────────────────────────────────────────
        spread_limit = 3.0 if pair[:6] in ("EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD") else MAX_SPREAD_PIPS
        spread_ok    = spread <= spread_limit or "XAU" in pair or "XAG" in pair
        checks["spread"] = spread_ok
        if not spread_ok:
            rejections.append(f"Spread {spread:.1f} pips > {spread_limit:.0f} pip limit")

        # ── C7: ATR floor ────────────────────────────────────────────────────
        atr_ok = atr >= MIN_ATR
        checks["atr_floor"] = atr_ok
        if not atr_ok:
            rejections.append(f"ATR {atr:.6f} < {MIN_ATR} — dead market, no trade")

        # ── C8: Open trade count ─────────────────────────────────────────────
        max_trades = int(os.environ.get("MAX_OPEN_TRADES", "3"))
        count_ok   = open_count < max_trades
        checks["open_count"] = count_ok
        if not count_ok:
            rejections.append(f"Open trades {open_count} ≥ max {max_trades}")

        # ── C9: Correlation check ─────────────────────────────────────────────
        corr_ok  = True
        max_corr = 0.0
        if open_pairs:
            for op in open_pairs:
                corr = _get_correlation(pair, op)
                max_corr = max(max_corr, corr)
            corr_ok = max_corr <= MAX_CORRELATION
        checks["correlation"] = corr_ok
        if not corr_ok:
            rejections.append(f"Correlation {max_corr:.2f} with open pair > {MAX_CORRELATION}")

        # ── Composite risk score ──────────────────────────────────────────────
        score_inputs = [
            min(1.0, risk_pct / MAX_RISK_PER_TRADE),
            min(1.0, spread / MAX_SPREAD_PIPS),
            min(1.0, max_corr / MAX_CORRELATION),
        ]
        risk_score = float(np.mean(score_inputs))

        # Warnings for high risk score
        if risk_score > 0.70 and not rejections:
            warnings.append(f"High composite risk score {risk_score:.2f} — trade with caution")
            sl_adj = 2.0  # Widen SL slightly

        approved = len(rejections) == 0 and self.initial_equity > 0

        logger.info(
            f"[R16] {pair} {trade.get('direction','?')}: "
            f"{'APPROVED' if approved else 'REJECTED'} | "
            f"risk={risk_pct:.2%} spread={spread:.1f} corr={max_corr:.2f} "
            f"lot_adj={lot_adj:.2f}"
        )
        if rejections:
            for r in rejections:
                logger.warning(f"  → VETO: {r}")

        return RiskValidation(
            approved    = approved,
            checks      = checks,
            rejections  = rejections,
            warnings    = warnings,
            risk_score  = round(risk_score, 4),
            lot_adj     = round(max(0.01, min(1.0, lot_adj)), 4),
            sl_adj_pips = sl_adj,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # R17: Cross-Asset Correlation Awareness
    # ─────────────────────────────────────────────────────────────────────────

    def get_correlation_report(self, pair: str, open_pairs: List[str]) -> Dict:
        """
        Return correlation analysis between proposed pair and all open positions.
        """
        correlations = {}
        for op in open_pairs:
            correlations[op] = _get_correlation(pair, op)
        max_corr    = max(correlations.values(), default=0.0)
        high_corr   = {p: c for p, c in correlations.items() if c > 0.60}
        return {
            "pair":         pair,
            "open_pairs":   open_pairs,
            "correlations": {k: round(v, 3) for k, v in correlations.items()},
            "max_corr":     round(max_corr, 3),
            "high_corr":    high_corr,
            "risk":         "high" if max_corr > MAX_CORRELATION else
                            "medium" if max_corr > 0.60 else "low",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # R18: Anomaly Detection & Circuit Breakers
    # ─────────────────────────────────────────────────────────────────────────

    def detect_anomalies(
        self,
        signals:    List[Dict],
        latency_ms: float = 0.0,
        daily_pnl:  float = 0.0,
    ) -> List[AnomalyEvent]:
        """
        Detect anomalies in signals, latency, and PnL.
        Fires circuit breaker on critical events.
        """
        anomalies: List[AnomalyEvent] = []

        # Signal anomalies
        for sig in signals:
            conf = float(sig.get("confidence", 0.5))
            if conf > 0.99:
                anomalies.append(AnomalyEvent(
                    "extreme_confidence", "warning",
                    f"Agent {sig.get('agent')} confidence={conf}", conf,
                ))
        confs = [s.get("confidence", 0.5) for s in signals]
        if len(confs) >= 3 and float(np.std(confs)) > 0.30:
            anomalies.append(AnomalyEvent(
                "confidence_divergence", "warning",
                f"Signal std={np.std(confs):.3f} — agents disagree",
                float(np.std(confs)),
            ))

        # Execution latency
        if latency_ms > MAX_LATENCY_MS:
            ev = AnomalyEvent("high_latency", "critical",
                              f"Latency {latency_ms:.0f}ms > {MAX_LATENCY_MS:.0f}ms limit",
                              latency_ms)
            anomalies.append(ev)
            self._fire_circuit_breaker(f"high_latency: {latency_ms:.0f}ms")

        # PnL spike
        pnl_pct = abs(daily_pnl) / max(self.equity, 1.0)
        if pnl_pct > 0.10:
            ev = AnomalyEvent("extreme_daily_pnl", "critical",
                              f"Daily PnL {daily_pnl:+.2f} ({pnl_pct:.1%}) exceeds 10%",
                              pnl_pct)
            anomalies.append(ev)
            if daily_pnl < 0:
                self._fire_circuit_breaker(f"extreme_daily_loss: {pnl_pct:.1%}")

        self._anomalies.extend(anomalies)
        return anomalies

    def _fire_circuit_breaker(self, reason: str):
        """Fire circuit breaker — trading halted until manual reset."""
        if self._cb_active:
            return  # Already active
        self._cb_active = True
        self._cb_reason = reason
        logger.critical(f"[R18] 🔴 CIRCUIT BREAKER FIRED: {reason}")

    def reset_circuit_breaker(self, code: str) -> bool:
        """Manual reset — requires CB_RESET_CODE env var match."""
        expected = os.environ.get("CB_RESET_CODE", "KRATOS-RESET-2026")
        if code != expected:
            logger.warning("[R18] Circuit breaker reset rejected — wrong code")
            return False
        self._cb_active    = False
        self._cb_reason    = None
        self._consecutive  = 0
        logger.info("[R18] Circuit breaker reset by manual approval")
        return True

    @property
    def circuit_breaker_active(self) -> bool:
        return self._cb_active

    # ─────────────────────────────────────────────────────────────────────────
    # R19: Adaptive Thresholding
    # ─────────────────────────────────────────────────────────────────────────

    def compute_adaptive_threshold(
        self,
        base:       float,
        volatility: float,
        equity:     float,
    ) -> float:
        """
        Dynamically adjust consensus threshold.

        Tightens when:
          - Win rate < 40% (losing streak)
          - Volatility > 20% annualised
          - Equity in drawdown > 3%
        Relaxes when:
          - Win rate > 60%
          - Equity at new highs

        Hard bounds: [0.60, 0.90]
        """
        t = base

        # Win rate from recent history
        recent = self._pnl_history[-20:]
        if recent:
            wr = sum(1 for p in recent if p > 0) / len(recent)
            if wr < 0.40:
                t += 0.08
            elif wr < 0.50:
                t += 0.04
            elif wr > 0.60:
                t -= 0.03

        # Volatility adjustment
        if volatility > 0.20:
            t += 0.05
        elif volatility < 0.08:
            t -= 0.02

        # Drawdown adjustment
        drawdown = (self.initial_equity - equity) / max(self.initial_equity, 1.0)
        if drawdown > 0.03:
            t += 0.10
        elif equity > self.initial_equity * 1.05:
            t -= 0.02

        return float(np.clip(t, 0.60, 0.90))

    # ─────────────────────────────────────────────────────────────────────────
    # Trade outcome recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_outcome(self, pnl: float, latency_ms: float = 0.0) -> Optional[str]:
        """
        Called after every trade closes. Updates internal state.
        Returns circuit breaker reason if fired, else None.
        """
        self._pnl_history.append(pnl)
        self._daily_pnl  += pnl

        if pnl < 0:
            self._consecutive += 1
        else:
            self._consecutive  = 0

        # Check limits
        daily_dd = abs(self._daily_pnl) / max(self.equity, 1.0)
        if self._daily_pnl < 0 and daily_dd >= MAX_DAILY_LOSS:
            self._fire_circuit_breaker(f"daily_loss {daily_dd:.1%}")
            return self._cb_reason
        if self._consecutive >= MAX_CONSECUTIVE_LOSSES:
            self._fire_circuit_breaker(f"{self._consecutive}_consecutive_losses")
            return self._cb_reason
        if latency_ms > MAX_LATENCY_MS:
            self._fire_circuit_breaker(f"latency_{latency_ms:.0f}ms")
            return self._cb_reason

        return None

    def reset_daily(self):
        """Call at start of each trading day."""
        self._daily_pnl   = 0.0
        logger.info("[RiskEngine] Daily PnL reset")

    def get_status(self) -> Dict:
        """Current risk engine state."""
        recent = self._pnl_history[-20:]
        return {
            "circuit_breaker":  self._cb_active,
            "cb_reason":        self._cb_reason,
            "daily_pnl":        round(self._daily_pnl, 2),
            "consecutive_loss": self._consecutive,
            "equity":           self.equity,
            "win_rate_20":      round(sum(1 for p in recent if p > 0) / max(len(recent), 1), 3),
            "anomalies_count":  len(self._anomalies),
        }


class AnomalyDetector:
    """
    Standalone anomaly detector (R18).
    Wraps RiskEngine.detect_anomalies() for callers expecting a
    separate AnomalyDetector instance (matches CI/CD spec interface).
    """

    def __init__(self, risk_engine: "RiskEngine"):
        self._engine = risk_engine

    def check_signals(self, signals: list) -> list:
        events = self._engine.detect_anomalies(signals, 0.0, 0.0)
        return [{"type": e.type, "severity": e.severity, "detail": e.detail}
                for e in events]

    def check_execution(self, execution: dict) -> list:
        latency = float(execution.get("latency_ms", 0.0))
        events  = self._engine.detect_anomalies([], latency, 0.0)
        return [{"type": e.type, "severity": e.severity, "detail": e.detail}
                for e in events]

    def check_pnl(self, pnl_ctx: dict) -> list:
        daily_pnl = float(pnl_ctx.get("daily_pnl", 0.0))
        events    = self._engine.detect_anomalies([], 0.0, daily_pnl)
        return [{"type": e.type, "severity": e.severity, "detail": e.detail}
                for e in events]
