"""
EXECUTION AGENT — KRATOS v2
------------------------------
Order execution specialist. Receives approved trade decisions from the
Orchestrator and executes via the BrokerRouter (OANDA → Deriv → ICMarkets →
Alpaca → SIM, in priority order).

Handles: lot sizing, SL/TP calculation, ATR-based risk, failover,
         and execution logging back to MemPalace.

v2: Multi-broker via BrokerRouter. All print() → logger.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from broker.broker_router import BrokerRouter

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_ATR_SL_MULTIPLIER = 1.5    # SL = 1.5x ATR
DEFAULT_ATR_TP_MULTIPLIER = 2.5    # TP = 2.5x ATR
MAX_LOT_SIZE              = 5.0    # Hard cap on lot size
MIN_LOT_SIZE              = 0.01   # Minimum micro lot
DEFAULT_EQUITY            = float(os.environ.get("ACCOUNT_EQUITY", "10000"))
DEFAULT_RISK_PCT          = 0.02   # 2% risk per trade


class ExecutionAgent:
    """
    Multi-broker execution agent.
    Uses BrokerRouter to auto-select and failover between brokers.
    """

    def __init__(self, equity: float = DEFAULT_EQUITY):
        self.name    = "execution"
        self.equity  = equity
        self.router  = BrokerRouter()
        self._order_log: list = []

        logger.info(
            f"[Execution] Initialised | broker={self.router.active_broker_name} "
            f"| equity=${equity:,.0f} | live={self.router.is_live}"
        )

    # ── Main execution entry point ────────────────────────────────────────────

    async def execute_trade(
        self,
        decision:     Dict[str, Any],
        market_state,
    ) -> Dict[str, Any]:
        """
        Execute a trade decision.

        Args:
            decision:     {'direction': 'BUY'|'SELL', 'lot_size': float (optional),
                           'stop_loss': float (optional), 'take_profit': float (optional)}
            market_state: MarketState object (pair, bid, ask, atr, spread, session)

        Returns:
            Execution result dict with status, order_id, broker, pnl fields.
        """
        direction = decision.get("direction", "FLAT")
        pair      = market_state.pair
        atr       = getattr(market_state, "atr", 0.001)

        if direction not in ("BUY", "SELL"):
            return {"status": "SKIPPED", "reason": f"Direction={direction}", "pair": pair}

        # ── Spread gate ───────────────────────────────────────────────────────
        spread = getattr(market_state, "spread", 0.0)
        if spread > 3.0 and "XAU" not in pair and "XAG" not in pair:
            logger.warning(f"[Execution] Spread gate: {pair} spread={spread} pips — SKIPPED")
            return {"status": "SKIPPED", "reason": f"Spread too wide ({spread} pips)", "pair": pair}

        # ── Lot size ──────────────────────────────────────────────────────────
        lot_size = decision.get("lot_size") or self._calculate_lot_size(atr, pair)

        # ── SL / TP ───────────────────────────────────────────────────────────
        sl_dist = atr * DEFAULT_ATR_SL_MULTIPLIER
        tp_dist = atr * DEFAULT_ATR_TP_MULTIPLIER

        # Risk agent may override SL distance
        if "sl_adjustment_pips" in decision:
            pip = 0.01 if "JPY" in pair else 0.0001
            sl_dist += decision["sl_adjustment_pips"] * pip

        if direction == "BUY":
            stop_loss   = round(market_state.bid - sl_dist, 5)
            take_profit = round(market_state.ask + tp_dist, 5)
            entry_price = market_state.ask
        else:
            stop_loss   = round(market_state.ask + sl_dist, 5)
            take_profit = round(market_state.bid - tp_dist, 5)
            entry_price = market_state.bid

        logger.info(
            f"[Execution] {direction} {pair} | lots={lot_size:.2f} "
            f"| entry={entry_price} SL={stop_loss} TP={take_profit} "
            f"| broker={self.router.active_broker_name}"
        )

        # ── Execute via BrokerRouter (with failover) ──────────────────────────
        try:
            result = await self.router.place_order(
                pair        = pair,
                direction   = direction,
                lot_size    = lot_size,
                stop_loss   = stop_loss,
                take_profit = take_profit,
                failover    = True,
            )
        except Exception as e:
            logger.error(f"[Execution] Unhandled execution error: {e}")
            result = {
                "status":    "FAILED",
                "error":     str(e),
                "pair":      pair,
                "broker":    "NONE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # ── Enrich result ─────────────────────────────────────────────────────
        result.update({
            "pair":        pair,
            "direction":   direction,
            "lot_size":    lot_size,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "entry_price": entry_price,
            "session":     getattr(market_state, "session", "unknown"),
            "atr":         atr,
            "spread":      spread,
            "pnl":         0.0,    # Updated later by on_trade_close()
            "decision":    decision,
        })

        # Log it
        self._order_log.append(result)
        status = result.get("status", "?")
        logger.info(
            f"[Execution] Result: {status} | id={result.get('order_id', 'N/A')[:12]} "
            f"| broker={result.get('broker', '?')}"
        )
        return result

    # ── Close trade ───────────────────────────────────────────────────────────

    async def close_trade(
        self,
        order_id: str,
        pair:     str    = "",
        units:    str    = "ALL",
        pnl:      float  = 0.0,
    ) -> Dict[str, Any]:
        """Close an open trade and update PnL."""
        result = await self.router.close_trade(order_id, pair=pair, units=units)

        # Update order log with actual PnL
        for order in self._order_log:
            if order.get("order_id") == order_id:
                order["pnl"] = pnl
                order["closed_at"] = datetime.now(timezone.utc).isoformat()
                break

        logger.info(f"[Execution] Closed trade {order_id[:12]} | PnL={pnl:+.2f}")
        return result

    # ── Lot size calculation ──────────────────────────────────────────────────

    def _calculate_lot_size(self, atr: float, pair: str) -> float:
        """
        Dynamic lot sizing based on ATR + fixed risk %.
        Formula: Risk$ / (ATR * pip_value * 100000)
        """
        risk_amount = self.equity * DEFAULT_RISK_PCT

        # Pip value per lot (approximate)
        if "JPY" in pair:
            pip_value = 1000.0   # ~$1000/lot for JPY pairs
        elif "XAU" in pair:
            pip_value = 100.0    # ~$100/pip for gold
        elif "XAG" in pair:
            pip_value = 50.0
        else:
            pip_value = 10.0     # ~$10/pip for standard forex

        sl_pips = (atr * DEFAULT_ATR_SL_MULTIPLIER) / (
            0.01 if "JPY" in pair else 0.0001
        )
        sl_pips = max(sl_pips, 5.0)   # Minimum 5-pip SL

        lot = risk_amount / (sl_pips * pip_value)
        lot = round(max(MIN_LOT_SIZE, min(MAX_LOT_SIZE, lot)), 2)
        return lot

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return execution performance stats."""
        total    = len(self._order_log)
        filled   = sum(1 for o in self._order_log if o.get("status") in ("FILLED", "SIMULATED"))
        failed   = sum(1 for o in self._order_log if o.get("status") == "FAILED")
        total_pnl = sum(o.get("pnl", 0.0) for o in self._order_log)
        return {
            "broker":          self.router.active_broker_name,
            "is_live":         self.router.is_live,
            "total_orders":    total,
            "filled":          filled,
            "failed":          failed,
            "total_pnl":       round(total_pnl, 2),
            "fill_rate":       round(filled / total, 2) if total else 0.0,
        }

    def broker_status(self) -> Dict[str, Any]:
        return self.router.status()
