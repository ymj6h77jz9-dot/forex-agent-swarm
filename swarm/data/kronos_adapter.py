"""
kronos_adapter.py — KRATOS v2
================================
Wraps the Kronos foundation model (KronosPredictor) so the swarm can call
predict_batch() with correct normalised inputs.

Key facts from Kronos source (model/kronos.py):
  - Input: pandas DataFrame with columns [open, high, low, close, vol, amt_vol]
  - Requires NO NaNs in price/volume columns
  - Internally normalises: x = (x - mean) / (std + 1e-5) then clips to ±clip
  - Returns a DataFrame indexed by y_timestamp with same columns
  - predict_batch() runs multiple series in parallel

KRATOS integration:
  - Called once per cycle with a candle_df per pair
  - Returns direction + confidence at 15% weight in consensus
  - Falls back to 50/50 HOLD if model unavailable
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy  as np
import pandas as pd

logger = logging.getLogger(__name__)

_model_instance = None   # lazy singleton


@dataclass
class KronosPrediction:
    direction:  str    # "BUY" | "SELL" | "HOLD"
    confidence: float  # 0.0 – 1.0
    pred_close: float  # predicted close price
    horizon:    int    # bars ahead


def _load_model():
    """Lazy-load KronosPredictor (heavy import)."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    try:
        import sys, importlib
        kronos_path = os.getenv("KRONOS_MODEL_PATH", "")
        if kronos_path and kronos_path not in sys.path:
            sys.path.insert(0, kronos_path)
        from model.kronos import KronosPredictor
        model_dir  = os.getenv("KRONOS_WEIGHTS_DIR", "")
        pred_len   = int(os.getenv("KRONOS_PRED_LEN", "10"))
        price_cols = ["open", "high", "low", "close"]
        vol_col    = "volume"
        amt_col    = "amount"
        _model_instance = KronosPredictor(
            model_dir  = model_dir,
            price_cols = price_cols,
            vol_col    = vol_col,
            amt_vol    = amt_col,
            clip       = float(os.getenv("KRONOS_CLIP", "5.0")),
        )
        logger.info("[KRONOS] KronosPredictor loaded from %s", model_dir)
        return _model_instance
    except Exception as e:
        logger.warning("[KRONOS] Model unavailable (will use fallback): %s", e)
        return None


def _prepare_df(candle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure candle_df has all required Kronos columns.
    Renames common column variants to Kronos canonical names.
    Fills missing volume/amount with zeros (as Kronos does internally).
    """
    df = candle_df.copy()

    # Normalise column names
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("open", "o"):    col_map[c] = "open"
        elif lc in ("high", "h"):  col_map[c] = "high"
        elif lc in ("low", "l"):   col_map[c] = "low"
        elif lc in ("close", "c"): col_map[c] = "close"
        elif lc in ("volume", "vol", "v"): col_map[c] = "volume"
        elif lc in ("amount", "amt", "amt_vol", "turnover"): col_map[c] = "amount"
    df.rename(columns=col_map, inplace=True)

    # Fill missing volume / amount
    if "volume" not in df.columns: df["volume"] = 0.0
    if "amount" not in df.columns: df["amount"] = df["volume"] * df.get("close", pd.Series([1.0]*len(df))).values

    # Drop NaNs
    df.dropna(subset=["open","high","low","close"], inplace=True)
    return df


def predict_single(
    pair:       str,
    candle_df:  pd.DataFrame,
    pred_len:   int  = 10,
    temperature: float = 1.0,
    top_p:      float = 0.9,
) -> KronosPrediction:
    """
    Run Kronos prediction for a single pair.
    Returns KronosPrediction with direction/confidence.
    Falls back to HOLD(0.5) if model unavailable.
    """
    model = _load_model()
    if model is None or candle_df.empty or len(candle_df) < 20:
        return KronosPrediction("HOLD", 0.5, 0.0, pred_len)

    try:
        df = _prepare_df(candle_df)
        if len(df) < 20:
            return KronosPrediction("HOLD", 0.5, 0.0, pred_len)

        x_timestamp = df.index
        last_ts     = df.index[-1]

        # Build future timestamps (assuming hourly candles by default)
        freq = _infer_freq(df)
        y_timestamp = pd.date_range(last_ts + freq, periods=pred_len, freq=freq, tz=timezone.utc)

        pred_df = model.predict(
            df           = df,
            x_timestamp  = x_timestamp,
            y_timestamp  = y_timestamp,
            pred_len     = pred_len,
            T            = temperature,
            top_p        = top_p,
            sample_count = 1,
            verbose      = False,
        )

        current_close = float(df["close"].iloc[-1])
        pred_close    = float(pred_df["close"].iloc[-1])
        change_pct    = (pred_close - current_close) / (current_close + 1e-9)

        # Direction + confidence from predicted price movement
        threshold  = 0.0005  # 0.05% minimum move to signal
        if change_pct > threshold:
            direction  = "BUY"
            confidence = min(0.95, 0.5 + abs(change_pct) * 50)
        elif change_pct < -threshold:
            direction  = "SELL"
            confidence = min(0.95, 0.5 + abs(change_pct) * 50)
        else:
            direction  = "HOLD"
            confidence = 0.5

        logger.info(
            "[KRONOS] %s pred_close=%.5f current=%.5f change=%.4f%% → %s (conf=%.3f)",
            pair, pred_close, current_close, change_pct * 100, direction, confidence
        )
        return KronosPrediction(direction, confidence, pred_close, pred_len)

    except Exception as e:
        logger.error("[KRONOS] predict_single failed for %s: %s", pair, e)
        return KronosPrediction("HOLD", 0.5, 0.0, pred_len)


def predict_batch(
    pairs:       List[str],
    candle_dfs:  List[pd.DataFrame],
    pred_len:    int   = 10,
    temperature: float = 1.0,
) -> List[KronosPrediction]:
    """
    Batch prediction across multiple pairs using KronosPredictor.predict_batch().
    Falls back to individual predict_single() if batch unavailable.
    """
    model = _load_model()
    if model is None or not hasattr(model, "predict_batch"):
        return [predict_single(p, df, pred_len, temperature) for p, df in zip(pairs, candle_dfs)]

    try:
        prepared      = [_prepare_df(df) for df in candle_dfs]
        valid_indices = [i for i, df in enumerate(prepared) if len(df) >= 20]
        if not valid_indices:
            return [KronosPrediction("HOLD", 0.5, 0.0, pred_len)] * len(pairs)

        df_list      = [prepared[i] for i in valid_indices]
        x_ts_list    = [df.index for df in df_list]
        freq_list    = [_infer_freq(df) for df in df_list]
        y_ts_list    = [
            pd.date_range(df.index[-1] + freq, periods=pred_len, freq=freq, tz=timezone.utc)
            for df, freq in zip(df_list, freq_list)
        ]

        pred_dfs = model.predict_batch(
            df_list         = df_list,
            x_timestamp_list = x_ts_list,
            y_timestamp_list = y_ts_list,
            pred_len        = pred_len,
            T               = temperature,
            top_p           = 0.9,
            sample_count    = 1,
            verbose         = False,
        )

        results = [KronosPrediction("HOLD", 0.5, 0.0, pred_len)] * len(pairs)
        for idx, (i, pred_df) in enumerate(zip(valid_indices, pred_dfs)):
            df = prepared[i]
            cur   = float(df["close"].iloc[-1])
            pred  = float(pred_df["close"].iloc[-1])
            chg   = (pred - cur) / (cur + 1e-9)
            t     = 0.0005
            if chg > t:
                d, c = "BUY",  min(0.95, 0.5 + abs(chg) * 50)
            elif chg < -t:
                d, c = "SELL", min(0.95, 0.5 + abs(chg) * 50)
            else:
                d, c = "HOLD", 0.5
            results[i] = KronosPrediction(d, c, pred, pred_len)
            logger.info("[KRONOS BATCH] %s → %s (conf=%.3f)", pairs[i], d, c)

        return results

    except Exception as e:
        logger.error("[KRONOS] predict_batch failed: %s — falling back to single", e)
        return [predict_single(p, df, pred_len, temperature) for p, df in zip(pairs, candle_dfs)]


def _infer_freq(df: pd.DataFrame) -> timedelta:
    """Infer bar frequency from DatetimeIndex differences."""
    if len(df) < 2:
        return timedelta(hours=1)
    diffs = df.index[1:] - df.index[:-1]
    median_secs = float(np.median([d.total_seconds() for d in diffs]))
    return timedelta(seconds=max(median_secs, 60))
