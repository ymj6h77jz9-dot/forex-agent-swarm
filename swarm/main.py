"""
MAIN ENTRY POINT — KRATOS v2 Forex Agentic Swarm
--------------------------------------------------
Bootstraps the full 16-step orchestrator pipeline.

Usage:
    python main.py --pairs EURUSD XAUUSD GBPUSD --mode sim
    python main.py --pairs EURUSD --mode live

Environment variables (set in .env before running):
    OPENROUTER_API_KEY   — OpenRouter free LLM (primary)
    OANDA_API_KEY        — OANDA broker
    OANDA_ACCOUNT_ID     — OANDA account ID
    DERIV_API_KEY        — Deriv (alternative broker)
    FRED_API_KEY         — FRED macro data (free)
    NEWS_API_KEY         — NewsAPI headlines (free)
    REPORT_EMAIL         — Daily report recipient
    ACCOUNT_EQUITY       — Account balance in USD (default: 10000)
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Load .env if present ────────────────────────────────────────────────────
_env_file = Path(__file__).parent.parent / ".agents" / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
logger = logging.getLogger("kratos.main")

# ── Imports ──────────────────────────────────────────────────────────────────
from kratos_orchestrator import KratosOrchestratorV2
from orchestrator_agent  import MarketState
from llm_client          import get_active_provider


# ── Session helper ────────────────────────────────────────────────────────────

def get_session(hour_utc: int) -> str:
    if 22 <= hour_utc or hour_utc < 7:  return "asia"
    if 7  <= hour_utc < 9:              return "overlap"   # Asia / London
    if 9  <= hour_utc < 13:             return "london"
    if 13 <= hour_utc < 17:             return "overlap"   # London / NY
    if 17 <= hour_utc < 22:             return "ny"
    return "off"


# ── Simulation mode ───────────────────────────────────────────────────────────

async def simulate_market_events(
    pairs:        list,
    orchestrator: KratosOrchestratorV2,
    n_cycles:     int = 10,
    cycle_delay:  float = 3.0,
) -> None:
    """
    Simulation mode — synthetic price events for testing the full pipeline.
    Replace with real-time price feed for production.
    """
    import random

    # Realistic mid-prices for major pairs
    base_prices = {
        "EURUSD": 1.0845, "GBPUSD": 1.2634, "XAUUSD": 2345.50,
        "USDJPY": 155.20, "AUDUSD": 0.6512, "USDCHF": 0.9105,
        "USDCAD": 1.3640, "NZDUSD": 0.5980, "EURGBP": 0.8595,
        "EURJPY": 168.42, "GBPJPY": 196.10, "XAGUSD": 29.45,
    }
    # Pair-specific spread and ATR profiles
    pair_profiles = {
        "EURUSD": (0.00015, 0.0008), "GBPUSD": (0.00020, 0.0012),
        "XAUUSD": (0.30,    2.5),    "USDJPY": (0.015,   0.08),
        "AUDUSD": (0.00018, 0.0007), "USDCHF": (0.00018, 0.0007),
    }

    logger.info(f"SIM MODE | pairs={pairs} | cycles={n_cycles}")

    wins = 0
    for cycle in range(1, n_cycles + 1):
        logger.info(f"\n{'─'*50}")
        logger.info(f"CYCLE {cycle}/{n_cycles}")

        for pair in pairs:
            base = base_prices.get(pair, 1.0)
            # Random walk with drift
            noise   = random.gauss(0, base * 0.0004)
            bid     = round(base + noise, 5)
            spread_, atr_ = pair_profiles.get(pair, (0.0002, 0.001))
            ask     = round(bid + spread_ * random.uniform(0.8, 1.4), 5)
            atr     = round(max(atr_ * 0.5, abs(random.gauss(atr_, atr_ * 0.3))), 5)
            now     = datetime.now(timezone.utc)

            state = MarketState(
                pair      = pair,
                bid       = bid,
                ask       = ask,
                spread    = round((ask - bid) * (100 if "JPY" in pair or "XAU" in pair else 10000), 2),
                atr       = atr,
                session   = get_session(now.hour),
                timestamp = now.isoformat(),
            )

            try:
                result = await orchestrator.process_market_event(state)
                action = result.get("action", "HOLD")
                score  = result.get("score", 0.0)
                reason = result.get("reason", "")[:60]
                logger.info(f"  {pair}: {action} | score={score:.3f} | {reason}")
                if action in ("BUY", "SELL"):
                    wins += 1
            except Exception as e:
                logger.error(f"  {pair}: Pipeline error — {e}")

            await asyncio.sleep(0.5)   # brief pause between pairs

        logger.info(f"  [Trade signals this cycle: {wins}]")
        await asyncio.sleep(cycle_delay)

    # Print dashboard
    try:
        dash = orchestrator.get_dashboard()
        logger.info(f"\n{'='*55}")
        logger.info("SESSION DASHBOARD:")
        logger.info(f"  Cycles run:     {dash.get('cycles_run', 0)}")
        logger.info(f"  Trades opened:  {dash.get('open_trades', 0)}")
        logger.info(f"  Daily P&L:      {dash.get('daily_pnl', '$0.00')}")
        logger.info(f"  Agent weights:  {dash.get('agent_weights', {})}")
        logger.info(f"{'='*55}")
    except Exception as e:
        logger.warning(f"Dashboard error: {e}")


# ── Live mode (OANDA streaming) ───────────────────────────────────────────────

async def live_market_feed(
    pairs:        list,
    orchestrator: KratosOrchestratorV2,
) -> None:
    """
    Live mode — connects to OANDA streaming API.
    Requires: OANDA_API_KEY, OANDA_ACCOUNT_ID in .env
    """
    import httpx

    oanda_key     = os.environ.get("OANDA_API_KEY", "")
    oanda_account = os.environ.get("OANDA_ACCOUNT_ID", "")

    if not oanda_key or not oanda_account:
        logger.error("LIVE mode requires OANDA_API_KEY and OANDA_ACCOUNT_ID in .env")
        return

    # Convert pair symbols to OANDA instrument format
    def to_oanda(pair: str) -> str:
        if "XAU" in pair:  return pair[:3] + "_" + pair[3:]
        return pair[:3] + "_" + pair[3:]

    instruments = ",".join(to_oanda(p) for p in pairs)
    stream_url   = (
        f"https://stream-fxtrade.oanda.com/v3/accounts/{oanda_account}"
        f"/pricing/stream?instruments={instruments}"
    )
    headers = {
        "Authorization":  f"Bearer {oanda_key}",
        "Accept-Encoding": "gzip",
    }

    logger.info(f"LIVE MODE | Connecting to OANDA stream | pairs={pairs}")
    price_cache: dict = {}   # last price per pair

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", stream_url, headers=headers) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    import json
                    msg = json.loads(line)
                    if msg.get("type") != "PRICE":
                        continue

                    pair_raw = msg.get("instrument", "")
                    pair     = pair_raw.replace("_", "")
                    bids     = msg.get("bids", [{}])
                    asks     = msg.get("asks", [{}])
                    bid      = float(bids[0].get("price", 0))
                    ask      = float(asks[0].get("price", 0))
                    now      = datetime.now(timezone.utc)

                    # Debounce — only process if price moved meaningfully
                    prev = price_cache.get(pair, bid)
                    if abs(bid - prev) < bid * 0.0001:
                        continue
                    price_cache[pair] = bid

                    # Estimate ATR from recent tick range (simplified)
                    atr = abs(ask - bid) * 10 if "XAU" in pair else abs(ask - bid) * 3

                    state = MarketState(
                        pair      = pair,
                        bid       = bid,
                        ask       = ask,
                        spread    = round((ask - bid) * (100 if "JPY" in pair else 10000), 2),
                        atr       = max(atr, 0.0003),
                        session   = get_session(now.hour),
                        timestamp = now.isoformat(),
                    )

                    result = await orchestrator.process_market_event(state)
                    action = result.get("action", "HOLD")
                    if action in ("BUY", "SELL"):
                        logger.info(
                            f"[LIVE] {pair}: {action} | "
                            f"score={result.get('score', 0):.3f} | "
                            f"{result.get('reason', '')[:60]}"
                        )
                except Exception as e:
                    logger.warning(f"[LIVE] Tick parse error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="KRATOS v2 Forex Agentic Swarm")
    parser.add_argument(
        "--pairs", nargs="+",
        default=["EURUSD", "XAUUSD"],
        help="Pairs to trade (e.g. EURUSD XAUUSD GBPUSD)",
    )
    parser.add_argument(
        "--mode", choices=["live", "sim"],
        default="sim",
        help="live = OANDA streaming | sim = synthetic events",
    )
    parser.add_argument(
        "--cycles", type=int, default=10,
        help="(sim only) Number of cycles to run",
    )
    parser.add_argument(
        "--equity", type=float,
        default=float(os.environ.get("ACCOUNT_EQUITY", "10000")),
        help="Account equity in USD for lot sizing",
    )
    args = parser.parse_args()

    # Banner
    provider = get_active_provider()
    print(f"""
╔══════════════════════════════════════════════════════╗
║           KRATOS v2 — Forex Agentic Swarm            ║
╠══════════════════════════════════════════════════════╣
║  Mode   : {args.mode:<43} ║
║  Pairs  : {', '.join(args.pairs):<43} ║
║  Equity : ${args.equity:<42,.0f} ║
║  LLM    : {provider[:43]:<43} ║
║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M UTC'):<43} ║
╚══════════════════════════════════════════════════════╝
""")

    orchestrator = KratosOrchestratorV2(account_equity=args.equity)

    if args.mode == "sim":
        await simulate_market_events(
            pairs        = args.pairs,
            orchestrator = orchestrator,
            n_cycles     = args.cycles,
        )
    else:
        await live_market_feed(
            pairs        = args.pairs,
            orchestrator = orchestrator,
        )


if __name__ == "__main__":
    asyncio.run(main())
