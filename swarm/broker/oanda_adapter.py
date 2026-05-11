"""
OANDA BROKER ADAPTER — KRATOS v2
----------------------------------
Full REST v20 + streaming integration with OANDA.
Supports: market/limit orders, SL/TP, position management,
          candle fetch, account info, live tick streaming.

Docs: https://developer.oanda.com/rest-live-v20/introduction/
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

OANDA_API_KEY    = os.environ.get("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_ENV        = os.environ.get("OANDA_ENV", "practice")   # "practice" | "live"

OANDA_REST_URLS = {
    "practice": "https://api-fxtrade.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}
OANDA_STREAM_URLS = {
    "practice": "https://stream-fxpractice.oanda.com",
    "live":     "https://stream-fxtrade.oanda.com",
}


def _instrument(pair: str) -> str:
    """EURUSD → EUR_USD"""
    if "_" in pair: return pair
    if "XAU" in pair: return "XAU_USD"
    if "XAG" in pair: return "XAG_USD"
    return f"{pair[:3]}_{pair[3:]}"


@dataclass
class OANDAPosition:
    trade_id:   str
    instrument: str
    units:      float
    open_price: float
    unrealized_pnl: float


class OANDAAdapter:
    """
    OANDA broker adapter.
    Uses REST v20 for orders/account. Streaming for live ticks.
    """

    BROKER_NAME = "OANDA"

    def __init__(self):
        self._key     = OANDA_API_KEY
        self._account = OANDA_ACCOUNT_ID
        self._base    = OANDA_REST_URLS.get(OANDA_ENV, OANDA_REST_URLS["practice"])
        self._stream  = OANDA_STREAM_URLS.get(OANDA_ENV, OANDA_STREAM_URLS["practice"])
        self._headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type":  "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    def is_configured(self) -> bool:
        return bool(self._key and self._account)

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_account(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(
                f"{self._base}/v3/accounts/{self._account}/summary",
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json().get("account", {})

    async def get_balance(self) -> float:
        acc = await self.get_account()
        return float(acc.get("balance", 0.0))

    # ── Market Data ───────────────────────────────────────────────────────────

    async def get_prices(self, pairs: List[str]) -> Dict[str, Dict]:
        instruments = ",".join(_instrument(p) for p in pairs)
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(
                f"{self._base}/v3/accounts/{self._account}/pricing",
                headers=self._headers,
                params={"instruments": instruments},
            )
            r.raise_for_status()
            prices = {}
            for p in r.json().get("prices", []):
                inst = p["instrument"].replace("_", "")
                bid  = float(p["bids"][0]["price"]) if p.get("bids") else 0.0
                ask  = float(p["asks"][0]["price"]) if p.get("asks") else 0.0
                prices[inst] = {"bid": bid, "ask": ask, "spread": round(ask - bid, 5)}
            return prices

    async def get_candles(
        self,
        pair:        str,
        granularity: str = "M5",
        count:       int = 200,
    ) -> List[Dict]:
        """Fetch OHLCV candles. granularity: S5 M1 M5 M15 M30 H1 H4 D"""
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.get(
                f"{self._base}/v3/instruments/{_instrument(pair)}/candles",
                headers=self._headers,
                params={"granularity": granularity, "count": count, "price": "MBA"},
            )
            r.raise_for_status()
            candles = []
            for c in r.json().get("candles", []):
                m = c.get("mid", {})
                candles.append({
                    "time":   c["time"],
                    "open":   float(m.get("o", 0)),
                    "high":   float(m.get("h", 0)),
                    "low":    float(m.get("l", 0)),
                    "close":  float(m.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })
            return candles

    # ── Order Execution ───────────────────────────────────────────────────────

    async def place_market_order(
        self,
        pair:       str,
        direction:  str,            # "BUY" | "SELL"
        units:      float,          # In base currency units (0.1 lot = 10000 units for 6-digit pairs)
        stop_loss:  Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        sign  = 1 if direction == "BUY" else -1
        order: Dict[str, Any] = {
            "order": {
                "type":        "MARKET",
                "instrument":  _instrument(pair),
                "units":       str(int(units * sign)),
                "timeInForce": "FOK",
            }
        }
        if stop_loss:
            order["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}", "timeInForce": "GTC"
            }
        if take_profit:
            order["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}", "timeInForce": "GTC"
            }

        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.post(
                f"{self._base}/v3/accounts/{self._account}/orders",
                headers=self._headers,
                content=json.dumps(order),
            )
            r.raise_for_status()
            data = r.json()

        fill = data.get("orderFillTransaction", {})
        create = data.get("orderCreateTransaction", {})
        order_id = fill.get("id") or create.get("id") or ""
        exec_price = float(fill.get("price", 0.0)) if fill else None

        logger.info(
            f"[OANDA] {direction} {pair} | units={units} | "
            f"id={order_id} | price={exec_price}"
        )
        return {
            "status":      "FILLED" if fill else "PENDING",
            "order_id":    order_id,
            "exec_price":  exec_price,
            "broker":      "OANDA",
            "raw":         data,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }

    async def close_trade(self, trade_id: str, units: str = "ALL") -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.put(
                f"{self._base}/v3/accounts/{self._account}/trades/{trade_id}/close",
                headers=self._headers,
                content=json.dumps({"units": units}),
            )
            r.raise_for_status()
            return {"status": "CLOSED", "trade_id": trade_id, "raw": r.json()}

    async def get_open_trades(self) -> List[OANDAPosition]:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(
                f"{self._base}/v3/accounts/{self._account}/openTrades",
                headers=self._headers,
            )
            r.raise_for_status()
            positions = []
            for t in r.json().get("trades", []):
                positions.append(OANDAPosition(
                    trade_id       = t["id"],
                    instrument     = t["instrument"].replace("_", ""),
                    units          = float(t["currentUnits"]),
                    open_price     = float(t["price"]),
                    unrealized_pnl = float(t.get("unrealizedPL", 0)),
                ))
            return positions

    async def set_sl_tp(
        self,
        trade_id:   str,
        stop_loss:  Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if stop_loss:
            body["stopLoss"] = {"price": f"{stop_loss:.5f}", "timeInForce": "GTC"}
        if take_profit:
            body["takeProfit"] = {"price": f"{take_profit:.5f}", "timeInForce": "GTC"}
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.put(
                f"{self._base}/v3/accounts/{self._account}/trades/{trade_id}/orders",
                headers=self._headers,
                content=json.dumps(body),
            )
            r.raise_for_status()
            return r.json()

    # ── Live Streaming ────────────────────────────────────────────────────────

    async def stream_prices(
        self,
        pairs:    List[str],
        callback: Callable,
    ) -> None:
        """
        Stream live prices. callback(pair, bid, ask) called on each tick.
        Runs indefinitely — wrap in a task.
        """
        instruments = ",".join(_instrument(p) for p in pairs)
        url = (
            f"{self._stream}/v3/accounts/{self._account}"
            f"/pricing/stream?instruments={instruments}"
        )
        while True:
            try:
                async with httpx.AsyncClient(timeout=None) as http:
                    async with http.stream("GET", url, headers=self._headers) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            msg = json.loads(line)
                            if msg.get("type") != "PRICE":
                                continue
                            inst  = msg["instrument"].replace("_", "")
                            bid   = float(msg["bids"][0]["price"]) if msg.get("bids") else 0.0
                            ask   = float(msg["asks"][0]["price"]) if msg.get("asks") else 0.0
                            await callback(inst, bid, ask)
            except Exception as e:
                logger.warning(f"[OANDA] Stream interrupted: {e} — retrying in 5s")
                await asyncio.sleep(5)
