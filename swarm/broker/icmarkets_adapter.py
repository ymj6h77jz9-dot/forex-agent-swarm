"""
IC MARKETS (cTrader) ADAPTER — KRATOS v2
------------------------------------------
IC Markets integration via the cTrader Open API (Spotware).
IC Markets is one of the tightest-spread ECN brokers globally.
Uses WebSocket for order streaming and REST for account data.

cTrader Open API: https://help.ctrader.com/open-api/
Requires: cTrader account + developer app credentials.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

ICMARKETS_CLIENT_ID     = os.environ.get("ICMARKETS_CLIENT_ID", "")
ICMARKETS_CLIENT_SECRET = os.environ.get("ICMARKETS_CLIENT_SECRET", "")
ICMARKETS_ACCOUNT_ID    = os.environ.get("ICMARKETS_ACCOUNT_ID", "")
ICMARKETS_ENV           = os.environ.get("ICMARKETS_ENV", "demo")  # "demo" | "live"

CTRADER_AUTH_URL  = "https://connect.spotware.com/apps/token"
CTRADER_API_URL   = "https://api.ctrader.com"

# cTrader symbol IDs for major pairs (from spotware symbol list)
# These are fetched dynamically in production — using known IDs here
SYMBOL_IDS = {
    "EURUSD": 1,  "GBPUSD": 2,  "USDJPY": 3,  "USDCHF": 4,
    "AUDUSD": 5,  "USDCAD": 6,  "NZDUSD": 7,  "EURGBP": 8,
    "EURJPY": 9,  "GBPJPY": 10, "XAUUSD": 41, "XAGUSD": 42,
}


class ICMarketsAdapter:
    """
    IC Markets cTrader Open API adapter.
    Best for: ECN tight spreads, institutional execution, high-frequency.
    """

    BROKER_NAME = "ICMARKETS"

    def __init__(self):
        self._client_id     = ICMARKETS_CLIENT_ID
        self._client_secret = ICMARKETS_CLIENT_SECRET
        self._account_id    = ICMARKETS_ACCOUNT_ID
        self._access_token: Optional[str] = None
        self._token_expires: float = 0.0

    def is_configured(self) -> bool:
        return bool(self._client_id and self._client_secret and self._account_id)

    # ── Authentication ────────────────────────────────────────────────────────

    async def _get_token(self) -> str:
        """OAuth2 client credentials flow for cTrader."""
        import time
        if self._access_token and time.time() < self._token_expires - 60:
            return self._access_token

        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.post(
                CTRADER_AUTH_URL,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self._client_id,
                    "client_secret": self._client_secret,
                },
            )
            r.raise_for_status()
            data = r.json()
            self._access_token  = data["access_token"]
            self._token_expires = time.time() + int(data.get("expires_in", 3600))
            return self._access_token

    def _headers(self, token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_account(self) -> Dict[str, Any]:
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(
                f"{CTRADER_API_URL}/v2/tradingaccounts/{self._account_id}",
                headers=self._headers(token),
            )
            r.raise_for_status()
            return r.json()

    async def get_balance(self) -> float:
        try:
            acc = await self.get_account()
            return float(acc.get("balance", 0.0)) / 100.0  # cTrader returns cents
        except Exception as e:
            logger.warning(f"[ICMarkets] get_balance failed: {e}")
            return 0.0

    # ── Market Data ───────────────────────────────────────────────────────────

    async def get_candles(
        self,
        pair:        str,
        period:      str = "M5",   # M1 M5 M15 M30 H1 H4 D1 W1
        count:       int = 200,
    ) -> List[Dict]:
        """Fetch OHLCV from cTrader."""
        symbol_id = SYMBOL_IDS.get(pair)
        if not symbol_id:
            logger.warning(f"[ICMarkets] Unknown pair: {pair}")
            return []
        try:
            token = await self._get_token()
            async with httpx.AsyncClient(timeout=15) as http:
                r = await http.get(
                    f"{CTRADER_API_URL}/v2/symbols/{symbol_id}/trendbars",
                    headers=self._headers(token),
                    params={"period": period, "count": count},
                )
                r.raise_for_status()
                bars = r.json().get("trendBar", [])
                return [
                    {
                        "time":   b.get("utcTimestampInMinutes", 0) * 60,
                        "open":   b.get("open", 0) / 100000.0,
                        "high":   (b.get("open", 0) + b.get("high", 0)) / 100000.0,
                        "low":    (b.get("open", 0) - b.get("low", 0)) / 100000.0,
                        "close":  b.get("close", 0) / 100000.0,
                        "volume": b.get("volume", 0),
                    }
                    for b in bars
                ]
        except Exception as e:
            logger.warning(f"[ICMarkets] get_candles failed: {e}")
            return []

    # ── Order Execution ───────────────────────────────────────────────────────

    async def place_market_order(
        self,
        pair:        str,
        direction:   str,            # "BUY" | "SELL"
        volume:      float,          # In lots (0.01 = micro lot)
        stop_loss:   Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place a market order via cTrader Open API."""
        symbol_id = SYMBOL_IDS.get(pair)
        if not symbol_id:
            return {"status": "REJECTED", "error": f"Unknown pair: {pair}", "broker": "ICMARKETS"}

        # cTrader volume in units (1 lot = 100,000 units)
        volume_units = int(volume * 100_000)

        body = {
            "tradingAccountId": int(self._account_id),
            "symbolId":         symbol_id,
            "tradeSide":        "BUY" if direction == "BUY" else "SELL",
            "volume":           volume_units,
            "orderType":        "MARKET",
        }
        if stop_loss:
            body["stopLoss"] = int(stop_loss * 100000)
        if take_profit:
            body["takeProfit"] = int(take_profit * 100000)

        try:
            token = await self._get_token()
            async with httpx.AsyncClient(timeout=15) as http:
                r = await http.post(
                    f"{CTRADER_API_URL}/v2/tradingaccounts/{self._account_id}/orders",
                    headers=self._headers(token),
                    content=json.dumps(body),
                )
                r.raise_for_status()
                data = r.json()

            order_id = str(data.get("orderId", data.get("id", "")))
            logger.info(f"[ICMarkets] {direction} {pair} | lots={volume} | id={order_id}")
            return {
                "status":     "FILLED",
                "order_id":   order_id,
                "exec_price": float(data.get("executionPrice", 0)) / 100000.0,
                "broker":     "ICMARKETS",
                "raw":        data,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"[ICMarkets] Order failed: {e}")
            return {"status": "FAILED", "error": str(e), "broker": "ICMARKETS",
                    "timestamp": datetime.now(timezone.utc).isoformat()}

    async def close_position(self, position_id: str, volume: Optional[float] = None) -> Dict[str, Any]:
        try:
            token = await self._get_token()
            body  = {"positionId": int(position_id)}
            if volume:
                body["volume"] = int(volume * 100_000)
            async with httpx.AsyncClient(timeout=15) as http:
                r = await http.post(
                    f"{CTRADER_API_URL}/v2/tradingaccounts/{self._account_id}/positions/close",
                    headers=self._headers(token),
                    content=json.dumps(body),
                )
                r.raise_for_status()
                return {"status": "CLOSED", "position_id": position_id, "raw": r.json()}
        except Exception as e:
            logger.error(f"[ICMarkets] Close failed: {e}")
            return {"status": "CLOSE_FAILED", "error": str(e)}

    async def get_open_positions(self) -> List[Dict]:
        try:
            token = await self._get_token()
            async with httpx.AsyncClient(timeout=10) as http:
                r = await http.get(
                    f"{CTRADER_API_URL}/v2/tradingaccounts/{self._account_id}/positions",
                    headers=self._headers(token),
                )
                r.raise_for_status()
                return r.json().get("position", [])
        except Exception as e:
            logger.warning(f"[ICMarkets] get_open_positions failed: {e}")
            return []
