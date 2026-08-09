"""
market_data_utils.py — KRATOS v2
==================================
DST-aware timezone utilities ported from AI-Trader (price_fetcher.py).
All market timestamps normalised to UTC before use in the swarm.

Pattern from AI-Trader:
  - zoneinfo-based America/New_York (DST-aware)
  - Falls back to fixed UTC-5 if zoneinfo unavailable
  - All datetimes converted to UTC immediately on ingress
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── DST-aware ET timezone (AI-Trader pattern) ─────────────────────────────
try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
    logger.debug("[UTILS] ET_TZ = America/New_York (zoneinfo DST-aware)")
except ImportError:
    from datetime import timezone
    ET_TZ = timezone(timedelta(hours=-5))
    logger.warning("[UTILS] zoneinfo unavailable — using fixed UTC-5 (no DST)")

UTC = timezone.utc


def to_utc(dt: datetime) -> datetime:
    """
    Normalise any datetime to UTC.
    - Naive datetimes are assumed to be in ET (AI-Trader convention).
    - Already-UTC datetimes are returned unchanged.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET_TZ)
    return dt.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)


def session_from_utc(dt: Optional[datetime] = None) -> str:
    """
    Classify a UTC datetime into a forex trading session.
    Returns: "sydney" | "tokyo" | "london" | "newyork" | "overlap_lon_ny"
    """
    dt = dt or utc_now()
    h  = dt.hour

    if 22 <= h or h < 7:   return "sydney"
    if 0 <= h < 9:          return "tokyo"
    if 7 <= h < 12:         return "london"
    if 12 <= h < 16:        return "overlap_lon_ny"
    if 16 <= h < 22:        return "newyork"
    return "off_hours"


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Forex is 24/5 — closed only on weekends (UTC)."""
    dt = dt or utc_now()
    return dt.weekday() < 5   # 0=Mon, 4=Fri, 5=Sat, 6=Sun


def normalise_candle_index(df) -> "pd.DataFrame":
    """
    Ensure a candle DataFrame has a UTC-aware DatetimeIndex.
    Safe to call multiple times (idempotent).
    """
    import pandas as pd
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(ET_TZ).tz_convert(UTC)
    else:
        idx = idx.tz_convert(UTC)
    df = df.copy()
    df.index = idx
    return df
