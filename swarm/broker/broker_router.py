"""
BROKER ROUTER — KRATOS v2
---------------------------
Unified multi-broker interface. Automatically selects the active broker
based on environment configuration. Supports automatic failover.

Priority order:
  1. OANDA     — if OANDA_API_KEY + OANDA_ACCOUNT_ID set
  2. DERIV     — if DERIV_API_KEY set
  3. ICMARKETS — if ICMARKETS_CLIENT_ID + ICMARKETS_ACCOUNT_ID set
  4. ALPACA    — if ALPACA_API_KEY + ALPACA_SECRET_KEY set
  5. SIM       — always available as fallback

Usage:
    router = BrokerRouter()
    result = await router.place_order(pair, direction, lot_size, sl, tp)
"""

import logging
import os
from typing import Dict, Any, Optional, List, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Lazy imports — only load adapters whose keys are set
_OANDA_KEY      = os.environ.get("OANDA_API_KEY", "")
_OANDA_ACCOUNT  = os.environ.get("OANDA_ACCOUNT_ID", "")
_DERIV_KEY      = os.environ.get("DERIV_API_KEY", "")
_ALPACA_KEY     = os.environ.get("ALPACA_API_KEY", "")
_ALPACA_SECRET  = os.environ.get("ALPACA_SECRET_KEY", "")
_ICM_CLIENT     = os.environ.get("ICMARKETS_CLIENT_ID", "")
_ICM_ACCOUNT    = os.environ.get("ICMARKETS_ACCOUNT_ID", "")

# Lot size conversion per broker
def _units_for_oanda(lots: float, pair: str) -> float:
    """OANDA uses raw units. 1 lot = 100,000 units for 6-digit pairs."""
    multiplier = 100 if "JPY" in pair else 100_000
    return round(lots * multiplier, 0)

def _qty_for_alpaca(lots: float) -> float:
    """Alpaca uses qty in shares/units."""
    return round(lots, 2)


class SimBroker:
    """No-op simulation broker — always available."""
    BROKER_NAME = "SIM"

    def is_configured(self) -> bool:
        return True

    async def place_market_order(
        self, pair, direction, units=None, qty=None, volume=None,
        stop_loss=None, take_profit=None, **kwargs
    ) -> Dict[str, Any]:
        from datetime import datetime, timezone
        logger.info(f"[SIM] {direction} {pair} | SL={stop_loss} TP={take_profit}")
        return {
            "status":     "SIMULATED",
            "order_id":   f"SIM-{pair}-{direction}",
            "exec_price": 0.0,
            "broker":     "SIM",
            "timestamp":  datetime.utcnow().isoformat(),
        }

    async def close_trade(self, trade_id, **kwargs) -> Dict[str, Any]:
        return {"status": "SIMULATED_CLOSE", "trade_id": trade_id}

    async def get_open_positions(self) -> List[Dict]:
        return []

    async def get_balance(self) -> float:
        return float(os.environ.get("ACCOUNT_EQUITY", "10000"))

    async def get_candles(self, pair, **kwargs) -> List[Dict]:
        return []


class BrokerRouter:
    """
    Auto-selects and wraps the active broker.
    Provides a unified interface for order execution regardless of broker.
    Supports failover: if primary broker fails, tries next in priority chain.
    """

    def __init__(self):
        self._brokers: list  = []
        self._active         = None
        self._active_name    = "SIM"
        self._sim            = SimBroker()
        self._init_brokers()

    def _init_brokers(self):
        """Initialise available brokers in priority order."""
        priority = []

        if _OANDA_KEY and _OANDA_ACCOUNT:
            from broker.oanda_adapter import OANDAAdapter
            b = OANDAAdapter()
            priority.append(b)
            logger.info("[BrokerRouter] OANDA configured")

        if _DERIV_KEY:
            from broker.deriv_adapter import DerivAdapter
            b = DerivAdapter()
            priority.append(b)
            logger.info("[BrokerRouter] Deriv configured")

        if _ICM_CLIENT and _ICM_ACCOUNT:
            from broker.icmarkets_adapter import ICMarketsAdapter
            b = ICMarketsAdapter()
            priority.append(b)
            logger.info("[BrokerRouter] IC Markets configured")

        if _ALPACA_KEY and _ALPACA_SECRET:
            from broker.alpaca_adapter import AlpacaAdapter
            b = AlpacaAdapter()
            priority.append(b)
            logger.info("[BrokerRouter] Alpaca configured")

        self._brokers = priority
        if priority:
            self._active      = priority[0]
            self._active_name = priority[0].BROKER_NAME
            logger.info(f"[BrokerRouter] Active broker: {self._active_name}")
        else:
            self._active      = self._sim
            self._active_name = "SIM"
            logger.warning("[BrokerRouter] No broker keys found — running in SIM mode")

    @property
    def active_broker_name(self) -> str:
        return self._active_name

    @property
    def is_live(self) -> bool:
        return self._active_name != "SIM"

    # ── Unified Order Interface ───────────────────────────────────────────────

    async def place_order(
        self,
        pair:        str,
        direction:   str,       # "BUY" | "SELL"
        lot_size:    float,     # In standard lots
        stop_loss:   Optional[float] = None,
        take_profit: Optional[float] = None,
        failover:    bool = True,
    ) -> Dict[str, Any]:
        """
        Place a market order on the active broker.
        Automatically translates lot_size to broker-native units.
        """
        for broker in ([self._active] + (self._brokers[1:] if failover else [])):
            try:
                result = await self._dispatch_order(
                    broker, pair, direction, lot_size, stop_loss, take_profit
                )
                if result.get("status") not in ("FAILED", "REJECTED"):
                    return result
                logger.warning(
                    f"[BrokerRouter] {broker.BROKER_NAME} rejected — "
                    f"{result.get('error', '')} — trying next"
                )
            except Exception as e:
                logger.error(f"[BrokerRouter] {broker.BROKER_NAME} error: {e}")

        # All brokers failed — fall back to sim
        logger.error("[BrokerRouter] All brokers failed — falling back to SIM")
        return await self._sim.place_market_order(
            pair, direction, stop_loss=stop_loss, take_profit=take_profit
        )

    async def _dispatch_order(
        self,
        broker,
        pair:       str,
        direction:  str,
        lot_size:   float,
        stop_loss:  Optional[float],
        take_profit: Optional[float],
    ) -> Dict[str, Any]:
        """Translate lot_size to broker-native units and call the right method."""
        name = broker.BROKER_NAME

        if name == "OANDA":
            units = _units_for_oanda(lot_size, pair)
            return await broker.place_market_order(
                pair=pair, direction=direction, units=units,
                stop_loss=stop_loss, take_profit=take_profit,
            )
        elif name == "DERIV":
            # Deriv uses stake amount as volume
            return await broker.place_order(OrderRequest(  # type: ignore[name-defined]
                symbol      = pair,
                side        = direction,
                order_type  = "MARKET",
                volume      = lot_size * 10,   # Stake approximation
                stop_loss   = stop_loss,
                take_profit = take_profit,
            ))
        elif name == "ICMARKETS":
            return await broker.place_market_order(
                pair=pair, direction=direction, volume=lot_size,
                stop_loss=stop_loss, take_profit=take_profit,
            )
        elif name == "ALPACA":
            qty = _qty_for_alpaca(lot_size)
            return await broker.place_market_order(
                symbol=pair, direction=direction, qty=qty,
                stop_loss=stop_loss, take_profit=take_profit,
            )
        else:
            return await broker.place_market_order(
                pair, direction, stop_loss=stop_loss, take_profit=take_profit
            )

    # ── Unified Close ─────────────────────────────────────────────────────────

    async def close_trade(self, trade_id: str, pair: str = "", units: str = "ALL") -> Dict[str, Any]:
        name = self._active_name
        try:
            if name == "OANDA":
                return await self._active.close_trade(trade_id, units)
            elif name == "ICMARKETS":
                return await self._active.close_position(trade_id)
            elif name == "ALPACA":
                return await self._active.close_position(pair)
            elif name == "DERIV":
                return await self._active.close_position(trade_id)
            else:
                return await self._sim.close_trade(trade_id)
        except Exception as e:
            logger.error(f"[BrokerRouter] close_trade failed: {e}")
            return {"status": "CLOSE_FAILED", "error": str(e)}

    # ── Unified Account ───────────────────────────────────────────────────────

    async def get_balance(self) -> float:
        try:
            return await self._active.get_balance()
        except Exception:
            return float(os.environ.get("ACCOUNT_EQUITY", "10000"))

    async def get_open_positions(self) -> List[Dict]:
        try:
            return await self._active.get_open_positions()
        except Exception:
            return []

    # ── Candle data ───────────────────────────────────────────────────────────

    async def get_candles(
        self,
        pair:      str,
        timeframe: str = "M5",
        count:     int = 200,
    ) -> List[Dict]:
        """Fetch candles from the active broker."""
        if not self.is_live:
            return []
        try:
            if self._active_name == "OANDA":
                return await self._active.get_candles(pair, granularity=timeframe, count=count)
            elif self._active_name == "ALPACA":
                tf_map = {"M1": "1Min", "M5": "5Min", "M15": "15Min", "H1": "1Hour"}
                return await self._active.get_candles(pair, timeframe=tf_map.get(timeframe, "5Min"), limit=count)
            elif self._active_name == "ICMARKETS":
                return await self._active.get_candles(pair, period=timeframe, count=count)
            elif self._active_name == "DERIV":
                gran_map = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600}
                return await self._active.get_candles(pair, granularity=gran_map.get(timeframe, 300), count=count)
        except Exception as e:
            logger.warning(f"[BrokerRouter] get_candles failed: {e}")
        return []

    def status(self) -> Dict[str, Any]:
        return {
            "active_broker":     self._active_name,
            "is_live":           self.is_live,
            "brokers_available": [b.BROKER_NAME for b in self._brokers] or ["SIM"],
        }
