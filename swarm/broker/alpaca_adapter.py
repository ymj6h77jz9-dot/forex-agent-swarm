"""
ALPACA BROKER ADAPTER — KRATOS v2
------------------------------------
Alpaca Markets REST API v2 integration.
Supports: forex (via Alpaca's FX data), crypto, stocks.
Free paper trading account — great for testing without real money.

API Docs: https://docs.alpaca.markets/reference/
Paper: https://paper-api.alpaca.markets
Live:  https://api.alpaca.markets
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_ENV        = os.environ.get("ALPACA_ENV", "paper")   # "paper" | "live"

ALPACA_BASE_URLS = {
    "paper": "https://paper-api.alpaca.markets",
    "live":  "https://api.alpaca.markets",
}
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Alpaca forex/crypto symbol mapping
PAIR_MAP = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD", "EURGBP": "EUR/GBP", "EURJPY": "EUR/JPY",
    "XAUUSD": "XAUUSD",  # Gold via crypto endpoint
}


class AlpacaAdapter:
    """
    Alpaca Markets broker adapter.
    Best for: paper trading, US stocks, crypto. FX via Alpaca Broker API.
    """

    BROKER_NAME = "ALPACA"

    def __init__(self):
        self._key    = ALPACA_API_KEY
        self._secret = ALPACA_SECRET_KEY
        self._base   = ALPACA_BASE_URLS.get(ALPACA_ENV, ALPACA_BASE_URLS["paper"])
        self._headers = {
            "APCA-API-KEY-ID":     self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Content-Type":        "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self._key and self._secret)

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_account(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(f"{self._base}/v2/account", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def get_balance(self) -> float:
        acc = await self.get_account()
        return float(acc.get("equity", acc.get("cash", 0.0)))

    # ── Market Data ───────────────────────────────────────────────────────────

    async def get_latest_bar(self, symbol: str) -> Dict[str, Any]:
        """Get latest bar (OHLCV) for a symbol."""
        alpaca_sym = PAIR_MAP.get(symbol, symbol)
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                r = await http.get(
                    f"{ALPACA_DATA_URL}/v2/stocks/{alpaca_sym}/bars/latest",
                    headers=self._headers,
                )
                r.raise_for_status()
                return r.json().get("bar", {})
        except Exception as e:
            logger.warning(f"[Alpaca] Latest bar failed for {symbol}: {e}")
            return {}

    async def get_candles(
        self,
        symbol:      str,
        timeframe:   str = "5Min",
        limit:       int = 200,
    ) -> List[Dict]:
        """Fetch OHLCV bars. timeframe: 1Min 5Min 15Min 1Hour 1Day"""
        alpaca_sym = PAIR_MAP.get(symbol, symbol)
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                r = await http.get(
                    f"{ALPACA_DATA_URL}/v2/stocks/{alpaca_sym}/bars",
                    headers=self._headers,
                    params={"timeframe": timeframe, "limit": limit, "adjustment": "raw"},
                )
                r.raise_for_status()
                return [
                    {
                        "time":   b["t"],
                        "open":   float(b["o"]),
                        "high":   float(b["h"]),
                        "low":    float(b["l"]),
                        "close":  float(b["c"]),
                        "volume": int(b.get("v", 0)),
                    }
                    for b in r.json().get("bars", [])
                ]
        except Exception as e:
            logger.warning(f"[Alpaca] Candles failed for {symbol}: {e}")
            return []

    # ── Order Execution ───────────────────────────────────────────────────────

    async def place_market_order(
        self,
        symbol:     str,
        direction:  str,            # "BUY" | "SELL"
        qty:        float,
        stop_loss:  Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        alpaca_sym = PAIR_MAP.get(symbol, symbol)
        order_body: Dict[str, Any] = {
            "symbol":        alpaca_sym,
            "qty":           str(qty),
            "side":          direction.lower(),
            "type":          "market",
            "time_in_force": "gtc",
        }
        # Alpaca bracket order for SL/TP
        if stop_loss and take_profit:
            order_body["order_class"]  = "bracket"
            order_body["stop_loss"]    = {"stop_price": str(round(stop_loss, 5))}
            order_body["take_profit"]  = {"limit_price": str(round(take_profit, 5))}

        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.post(
                f"{self._base}/v2/orders",
                headers=self._headers,
                content=json.dumps(order_body),
            )
            r.raise_for_status()
            data = r.json()

        logger.info(
            f"[Alpaca] {direction} {symbol} | qty={qty} | "
            f"id={data.get('id', '')[:8]} | status={data.get('status')}"
        )
        return {
            "status":     data.get("status", "PENDING").upper(),
            "order_id":   data.get("id", ""),
            "exec_price": float(data.get("filled_avg_price") or 0),
            "broker":     "ALPACA",
            "raw":        data,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }

    async def close_position(self, symbol: str) -> Dict[str, Any]:
        alpaca_sym = PAIR_MAP.get(symbol, symbol)
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.delete(
                f"{self._base}/v2/positions/{alpaca_sym}",
                headers=self._headers,
            )
            if r.status_code == 404:
                return {"status": "NO_POSITION", "symbol": symbol}
            r.raise_for_status()
            return {"status": "CLOSED", "symbol": symbol, "raw": r.json()}

    async def get_open_positions(self) -> List[Dict]:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(f"{self._base}/v2/positions", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def cancel_all_orders(self) -> None:
        async with httpx.AsyncClient(timeout=10) as http:
            await http.delete(f"{self._base}/v2/orders", headers=self._headers)
        logger.info("[Alpaca] All open orders cancelled")
