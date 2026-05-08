"""
KRONOS FOUNDATION MODEL ADAPTER
================================
Integrates Kronos — the first open-source foundation model for financial
candlestick data (K-lines), trained on 45+ global exchanges.

This adapter wraps the KronosPredictor to generate multi-step price 
forecasts (OHLCV) directly from candlestick history. The Orchestrator 
uses Kronos as a THIRD prediction layer alongside MiroFish PSO and GPT-4o.

Source: ymj6h77jz9-dot/Kronos (NeoQuasar/Kronos-small via HuggingFace)
Architecture:
  - KronosTokenizer: Binary Spherical Quantization (BSQ) encoder/decoder
  - Kronos: Autoregressive Transformer on tokenized candlestick sequences
  - KronosPredictor: Wraps model with temperature-controlled sampling

Models available (HuggingFace):
  NeoQuasar/Kronos-mini   4.1M params  context:2048  (fast)
  NeoQuasar/Kronos-small  24.7M params context:512   (balanced) ← default
  NeoQuasar/Kronos-base   102.3M params context:512  (best quality)
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

KRONOS_MODEL_ID     = os.environ.get("KRONOS_MODEL_ID", "NeoQuasar/Kronos-small")
KRONOS_TOKENIZER_ID = os.environ.get("KRONOS_TOKENIZER_ID", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_CONTEXT      = int(os.environ.get("KRONOS_CONTEXT", "400"))
KRONOS_PRED_LEN     = int(os.environ.get("KRONOS_PRED_LEN", "10"))   # bars ahead


@dataclass
class KronosPrediction:
    symbol:          str
    pred_len:        int
    forecast_close:  List[float]    # Predicted close prices
    forecast_high:   List[float]
    forecast_low:    List[float]
    forecast_volume: List[float]
    direction:       str            # "UP" | "DOWN" | "FLAT"
    magnitude_pct:   float          # Expected % move
    confidence:      float          # 0.0 - 1.0
    timestamp:       str


class KronosAdapter:
    """
    Adapter for the Kronos Foundation Model.
    
    Handles lazy loading (model only loads if GPU/CPU available and 
    huggingface hub packages installed). Falls back to statistical 
    baseline if Kronos unavailable.
    """

    def __init__(self):
        self._tokenizer  = None
        self._model      = None
        self._predictor  = None
        self._available  = False
        self._load_attempted = False

    def _try_load(self):
        """Lazy-load Kronos from HuggingFace Hub."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor
            logger.info(f"[Kronos] Loading tokenizer: {KRONOS_TOKENIZER_ID}")
            self._tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_ID)
            logger.info(f"[Kronos] Loading model: {KRONOS_MODEL_ID}")
            self._model     = Kronos.from_pretrained(KRONOS_MODEL_ID)
            self._predictor = KronosPredictor(
                self._model, self._tokenizer, max_context=KRONOS_CONTEXT
            )
            self._available = True
            logger.info("✅ [Kronos] Foundation model loaded successfully")
        except ImportError:
            logger.warning("[Kronos] model package not found. Install: pip install -e ./Kronos")
        except Exception as e:
            logger.warning(f"[Kronos] Model load failed: {e}. Using statistical fallback.")

    async def predict(self, symbol: str, candles: pd.DataFrame,
                      pred_len: int = KRONOS_PRED_LEN) -> KronosPrediction:
        """
        Generate a multi-bar OHLCV forecast using Kronos.
        
        Args:
            symbol:   Currency pair e.g. "EURUSD"
            candles:  DataFrame with columns [open, high, low, close, volume, amount]
                      Sorted ascending by time, at least KRONOS_CONTEXT bars
            pred_len: Number of future bars to predict
        
        Returns:
            KronosPrediction with forecast vectors and directional signal
        """
        self._try_load()

        if self._available and self._predictor is not None:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._run_kronos_sync, symbol, candles, pred_len
            )
        else:
            return self._statistical_fallback(symbol, candles, pred_len)

    def _run_kronos_sync(self, symbol: str, candles: pd.DataFrame,
                          pred_len: int) -> KronosPrediction:
        """Synchronous Kronos inference (runs in executor)."""
        try:
            lookback = min(KRONOS_CONTEXT, len(candles))
            x_df = candles.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']]
            
            # Build timestamp series
            x_ts = pd.date_range(
                end   = datetime.now(timezone.utc),
                periods = lookback,
                freq    = '1min'
            )
            y_ts = pd.date_range(
                start  = x_ts[-1],
                periods = pred_len + 1,
                freq    = '1min'
            )[1:]

            pred_df = self._predictor.predict(
                df            = x_df,
                x_timestamp   = x_ts,
                y_timestamp   = y_ts,
                pred_len      = pred_len,
                T             = 0.8,    # Temperature
                top_p         = 0.9,
                sample_count  = 1,
                verbose       = False,
            )

            current_close = float(candles['close'].iloc[-1])
            forecast_close = pred_df['close'].tolist()
            end_close     = forecast_close[-1]
            magnitude     = (end_close - current_close) / current_close * 100

            if magnitude > 0.05:
                direction = "UP"
            elif magnitude < -0.05:
                direction = "DOWN"
            else:
                direction = "FLAT"

            # Confidence based on trend consistency
            diffs = [forecast_close[i+1] - forecast_close[i] for i in range(len(forecast_close)-1)]
            positives = sum(1 for d in diffs if d > 0)
            consistency = positives / len(diffs) if diffs else 0.5
            confidence = abs(consistency - 0.5) * 2  # 0.0 at 50/50, 1.0 at all up/down

            return KronosPrediction(
                symbol         = symbol,
                pred_len       = pred_len,
                forecast_close = [round(c, 5) for c in forecast_close],
                forecast_high  = pred_df['high'].tolist() if 'high' in pred_df else [],
                forecast_low   = pred_df['low'].tolist()  if 'low'  in pred_df else [],
                forecast_volume = pred_df['volume'].tolist() if 'volume' in pred_df else [],
                direction      = direction,
                magnitude_pct  = round(magnitude, 4),
                confidence     = round(confidence, 4),
                timestamp      = datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.error(f"[Kronos] Inference error: {e}")
            return self._statistical_fallback(symbol, candles, pred_len)

    def _statistical_fallback(self, symbol: str, candles: pd.DataFrame,
                               pred_len: int) -> KronosPrediction:
        """
        Statistical fallback when Kronos model is unavailable.
        Uses simple linear regression + volatility expansion.
        """
        closes = candles['close'].values[-20:]
        if len(closes) < 2:
            return KronosPrediction(symbol, pred_len, [], [], [], [], "FLAT", 0.0, 0.0,
                                    datetime.now(timezone.utc).isoformat())

        # Linear trend
        x      = np.arange(len(closes))
        slope  = np.polyfit(x, closes, 1)[0]
        last   = closes[-1]
        std    = np.std(np.diff(closes))

        forecast = [round(last + slope * (i + 1) + np.random.normal(0, std * 0.3), 5)
                    for i in range(pred_len)]

        magnitude = (forecast[-1] - last) / last * 100
        direction = "UP" if magnitude > 0.03 else "DOWN" if magnitude < -0.03 else "FLAT"

        return KronosPrediction(
            symbol        = symbol,
            pred_len      = pred_len,
            forecast_close = forecast,
            forecast_high  = [f + std for f in forecast],
            forecast_low   = [f - std for f in forecast],
            forecast_volume = [],
            direction      = direction,
            magnitude_pct  = round(magnitude, 4),
            confidence     = 0.45,   # Low confidence for fallback
            timestamp      = datetime.now(timezone.utc).isoformat(),
        )

    def build_candle_df(self, ohlcv_list: List[Dict]) -> pd.DataFrame:
        """
        Convert raw OHLCV dict list to a Kronos-compatible DataFrame.
        Input format: [{'open':..,'high':..,'low':..,'close':..,'volume':..,'time':..}]
        """
        df = pd.DataFrame(ohlcv_list)
        df.rename(columns={'vol': 'volume', 'amt': 'amount'}, inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col not in df.columns:
                df[col] = 0.0
        return df[['open', 'high', 'low', 'close', 'volume', 'amount']].astype(float)
