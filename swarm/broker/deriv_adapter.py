"""
DERIV BROKER ADAPTER — Integrated from ymj6h77jz9-dot/KRATOS-app
-----------------------------------------------------------------
WebSocket-based real-time integration with Deriv (formerly Binary.com).
Handles: market data streaming, account management, order execution.

Source: KRATOS-app/server/_core/deriv_adapter.py
Enhanced with: full reconnection logic, streaming candle support
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


@dataclass
class AccountInfo:
    account_id:  str
    balance:     float
    currency:    str
    equity:      float
    margin_used: float = 0.0
    margin_free: float = 0.0


@dataclass
class MarketData:
    symbol:    str
    bid:       float
    ask:       float
    spread:    float
    timestamp: str


@dataclass
class OrderRequest:
    symbol:      str
    side:        OrderSide
    order_type:  OrderType
    volume:      float
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None
    price:       Optional[float] = None   # For limit orders


@dataclass
class OrderResponse:
    order_id:        str
    status:          str          # "FILLED" | "PENDING" | "REJECTED"
    execution_price: Optional[float] = None
    error:           Optional[str]   = None
    timestamp:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DerivAdapter:
    """
    Deriv WebSocket broker adapter.
    Integrated from KRATOS-app with reconnection, streaming, and full order management.
    """

    def __init__(self):
        self.websocket            = None
        self.api_key: Optional[str] = None
        self.account_id: Optional[str] = None
        self.request_id           = 0
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.market_callbacks: Dict[str, Callable] = {}
        self.market_data: Dict[str, MarketData] = {}
        self.subscribed_symbols: set = set()
        self.account_info: Optional[AccountInfo] = None
        self.connected            = False
        self._reconnect_attempts  = 0
        self._max_reconnects      = 5

    # ── Connection ────────────────────────────────────────────────────────────
    async def connect(self, api_key: str, account_id: str = "") -> bool:
        try:
            import websockets
            self.api_key    = api_key
            self.account_id = account_id
            self.websocket  = await websockets.connect(DERIV_WS_URL)
            self.connected  = True
            self._reconnect_attempts = 0

            # Start background message handler
            asyncio.create_task(self._message_loop())

            # Authorize
            auth_resp = await self._send({"authorize": api_key})
            if auth_resp.get("error"):
                logger.error(f"Deriv auth failed: {auth_resp['error']['message']}")
                self.connected = False
                return False

            self.account_id = auth_resp.get("authorize", {}).get("loginid", account_id)
            await self._refresh_account_info(auth_resp.get("authorize", {}))
            logger.info(f"✅ Deriv connected — account: {self.account_id}")
            return True

        except Exception as e:
            logger.error(f"Deriv connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
        self.connected = False
        logger.info("Deriv disconnected.")

    async def _message_loop(self):
        """Background loop that routes incoming WebSocket messages."""
        try:
            async for raw in self.websocket:
                msg = json.loads(raw)
                req_id = msg.get("req_id")

                # Route to pending request future
                if req_id and req_id in self.pending_requests:
                    fut = self.pending_requests.pop(req_id)
                    if not fut.done():
                        fut.set_result(msg)

                # Route tick subscriptions
                if "tick" in msg:
                    tick   = msg["tick"]
                    symbol = tick.get("symbol", "")
                    md = MarketData(
                        symbol    = symbol,
                        bid       = float(tick.get("bid", tick.get("quote", 0))),
                        ask       = float(tick.get("ask", tick.get("quote", 0))),
                        spread    = 0.0,
                        timestamp = datetime.now(timezone.utc).isoformat(),
                    )
                    md.spread = round(md.ask - md.bid, 5)
                    self.market_data[symbol] = md

                    if symbol in self.market_callbacks:
                        asyncio.create_task(self.market_callbacks[symbol](md))

        except Exception as e:
            logger.warning(f"Deriv message loop ended: {e}")
            if self.connected:
                await self._attempt_reconnect()

    async def _attempt_reconnect(self):
        """Auto-reconnect with exponential backoff."""
        while self._reconnect_attempts < self._max_reconnects:
            wait = 2 ** self._reconnect_attempts
            logger.info(f"Reconnecting in {wait}s (attempt {self._reconnect_attempts+1})...")
            await asyncio.sleep(wait)
            self._reconnect_attempts += 1
            if await self.connect(self.api_key, self.account_id):
                return
        logger.error("Max reconnect attempts reached. Deriv offline.")

    # ── Requests ──────────────────────────────────────────────────────────────
    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    async def _send(self, payload: dict, timeout: float = 15.0) -> dict:
        """Send a request and await its response by req_id."""
        req_id = self._next_id()
        payload["req_id"] = req_id
        fut = asyncio.get_event_loop().create_future()
        self.pending_requests[req_id] = fut
        await self.websocket.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending_requests.pop(req_id, None)
            return {"error": {"message": "Request timed out"}}

    # ── Market Data ───────────────────────────────────────────────────────────
    async def subscribe_ticks(self, symbol: str, callback: Callable):
        """Subscribe to real-time tick data for a symbol."""
        self.market_callbacks[symbol] = callback
        if symbol not in self.subscribed_symbols:
            await self._send({"ticks": symbol, "subscribe": 1})
            self.subscribed_symbols.add(symbol)
            logger.info(f"Subscribed to ticks: {symbol}")

    async def get_candles(self, symbol: str, granularity: int = 60,
                          count: int = 100) -> List[dict]:
        """Fetch OHLCV candles. granularity in seconds (60=1m, 300=5m, 3600=1h)."""
        resp = await self._send({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   granularity,
            "count":         count,
            "end":           "latest",
        })
        return resp.get("candles", [])

    # ── Account ───────────────────────────────────────────────────────────────
    async def get_account_info(self) -> Optional[AccountInfo]:
        resp = await self._send({"balance": 1, "subscribe": 0})
        if "balance" in resp:
            b = resp["balance"]
            self.account_info = AccountInfo(
                account_id  = b.get("loginid", self.account_id),
                balance     = float(b.get("balance", 0)),
                currency    = b.get("currency", "USD"),
                equity      = float(b.get("balance", 0)),
            )
        return self.account_info

    async def _refresh_account_info(self, auth_data: dict):
        self.account_info = AccountInfo(
            account_id = auth_data.get("loginid", ""),
            balance    = float(auth_data.get("balance", 0)),
            currency   = auth_data.get("currency", "USD"),
            equity     = float(auth_data.get("balance", 0)),
        )

    # ── Order Execution ───────────────────────────────────────────────────────
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a market or limit order via Deriv API."""
        if not self.connected:
            return OrderResponse("", "REJECTED", error="Not connected to Deriv")

        # Deriv contract types: "CALL" = BUY, "PUT" = SELL for vanilla forex
        contract_type = "CALL" if order.side == OrderSide.BUY else "PUT"

        payload = {
            "buy": 1,
            "subscribe": 0,
            "price": order.volume,           # Stake amount for Deriv
            "parameters": {
                "amount":        order.volume,
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      "USD",
                "duration":      5,
                "duration_unit": "m",
                "symbol":        order.symbol,
            }
        }

        resp = await self._send(payload, timeout=20.0)

        if resp.get("error"):
            return OrderResponse("", "REJECTED", error=resp["error"].get("message", "Unknown"))

        buy_data = resp.get("buy", {})
        return OrderResponse(
            order_id        = str(buy_data.get("contract_id", "")),
            status          = "FILLED",
            execution_price = float(buy_data.get("start_time", 0)),
        )

    async def close_position(self, contract_id: str) -> dict:
        """Close / sell back an open contract."""
        resp = await self._send({
            "sell":      contract_id,
            "price":     0,            # Market price
        })
        return resp.get("sell", {})
