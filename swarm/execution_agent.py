import logging

logger = logging.getLogger(__name__)

"""
EXECUTION AGENT — Forex Agentic Swarm
----------------------------------------
Order execution specialist. Receives approved trade decisions from Orchestrator,
places orders via broker API (OANDA by default), manages SL/TP,
handles partial closes, and reports results back.
"""

import asyncio
import os
import json
import httpx
from datetime import datetime, timezone

# ── Broker config (OANDA REST API v20) ───────────────────────────────────────
OANDA_API_KEY    = os.environ.get("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_BASE_URL   = os.environ.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")  # practice by default

# Pair format mapping: "EURUSD" → "EUR_USD"
def format_pair(pair: str) -> str:
    if "_" in pair:
        return pair
    if len(pair) == 6:
        return f"{pair[:3]}_{pair[3:]}"
    return pair

class ExecutionAgent:
    def __init__(self):
        self.name = "execution"

    async def execute_trade(self, decision: dict, market_state) -> dict:
        direction   = decision["direction"]    # "BUY" or "SELL"
        pair        = market_state.pair
        atr         = market_state.atr

        # Calculate SL/TP from ATR
        sl_distance = round(atr * 1.5, 5)
        tp_distance = round(atr * 2.5, 5)

        if direction == "BUY":
            sl = round(market_state.bid - sl_distance, 5)
            tp = round(market_state.ask + tp_distance, 5)
        else:
            sl = round(market_state.ask + sl_distance, 5)
            tp = round(market_state.bid - tp_distance, 5)

        # Default lot size (Risk Agent refines this in production)
        lot_size = 0.10

        order_payload = {
            "order": {
                "type":         "MARKET",
                "instrument":   format_pair(pair),
                "units":        str(lot_size * 100_000) if direction == "BUY"
                                else str(-lot_size * 100_000),
                "stopLossOnFill":   {"price": str(sl)},
                "takeProfitOnFill": {"price": str(tp)},
                "timeInForce":  "FOK",  # Fill or Kill
            }
        }

        logger.info(f"[EXECUTION] Placing {direction} on {pair} | SL:{sl} TP:{tp} Lots:{lot_size}")

        if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
            # Simulation mode — log but don't call broker
            logger.info("[EXECUTION] ⚠️  Simulated — no OANDA credentials set")
            return {
                "status":    "SIMULATED",
                "direction": direction,
                "pair":      pair,
                "lot_size":  lot_size,
                "sl":        sl,
                "tp":        tp,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision":  decision,
            }

        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/orders",
                    headers={
                        "Authorization": f"Bearer {OANDA_API_KEY}",
                        "Content-Type":  "application/json",
                    },
                    json=order_payload,
                    timeout=10.0,
                )
                r.raise_for_status()
                response_data = r.json()

                order_id = (
                    response_data.get("orderFillTransaction", {}).get("id") or
                    response_data.get("orderCreateTransaction", {}).get("id")
                )

                return {
                    "status":    "FILLED",
                    "direction": direction,
                    "pair":      pair,
                    "lot_size":  lot_size,
                    "sl":        sl,
                    "tp":        tp,
                    "order_id":  order_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw":       response_data,
                }

        except Exception as e:
            logger.info(f"[EXECUTION] Order failed: {e}")
            return {
                "status":    "FAILED",
                "direction": direction,
                "pair":      pair,
                "error":     str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def close_trade(self, order_id: str, pair: str, units: str = "ALL") -> dict:
        """Partially or fully close an open trade."""
        if not OANDA_API_KEY:
            return {"status": "SIMULATED_CLOSE", "order_id": order_id}

        try:
            async with httpx.AsyncClient() as http:
                r = await http.put(
                    f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/trades/{order_id}/close",
                    headers={"Authorization": f"Bearer {OANDA_API_KEY}"},
                    json={"units": units},
                    timeout=10.0,
                )
                r.raise_for_status()
                return {"status": "CLOSED", "order_id": order_id, "raw": r.json()}
        except Exception as e:
            return {"status": "CLOSE_FAILED", "error": str(e)}
