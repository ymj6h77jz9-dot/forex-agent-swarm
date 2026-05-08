"""
MAIN ENTRY POINT — Forex Agentic Swarm
-----------------------------------------
Bootstraps the swarm. Connects to market data feed, starts the event loop,
and wires up all agents. Run this to start the live swarm.

Usage:
    python main.py --pairs EURUSD XAUUSD GBPUSD --mode live
    python main.py --mode sim   # simulation mode, no real orders

Environment variables required:
    OPENAI_API_KEY      — LLM for all agents
    OANDA_API_KEY       — Broker API key
    OANDA_ACCOUNT_ID    — Broker account ID
    NEWS_API_KEY        — NewsAPI.org key (optional but recommended)
    REPORT_EMAIL        — Email to receive daily reports
"""

import asyncio
import argparse
import os
from datetime import datetime, timezone

from orchestrator_agent import OrchestratorAgent, MarketState
from memory_agent import MemoryAgent
from gmail_monitor import GmailMonitor


def get_session(hour_utc: int) -> str:
    if 22 <= hour_utc or hour_utc < 7:   return "asia"
    if 7  <= hour_utc < 9:               return "overlap"  # Asia/London
    if 9  <= hour_utc < 13:              return "london"
    if 13 <= hour_utc < 17:              return "overlap"  # London/NY
    if 17 <= hour_utc < 22:              return "ny"
    return "off"


async def simulate_market_events(pairs: list, orchestrator: OrchestratorAgent):
    """
    Simulation mode: generates synthetic market events for testing.
    Replace this with a real price feed (OANDA streaming, TwelveData websocket, etc.)
    """
    import random

    base_prices = {
        "EURUSD": 1.0845,
        "GBPUSD": 1.2634,
        "XAUUSD": 2345.50,
        "USDJPY": 155.20,
        "AUDUSD": 0.6512,
    }

    print(f"\n🤖 SWARM SIMULATION STARTED — pairs: {pairs}")
    print("=" * 55)

    for cycle in range(10):
        for pair in pairs:
            base  = base_prices.get(pair, 1.0)
            noise = random.gauss(0, base * 0.0005)
            bid   = round(base + noise, 5)
            ask   = round(bid + 0.00015, 5)
            atr   = round(abs(random.gauss(0.0012, 0.0003)), 5)
            now   = datetime.now(timezone.utc)

            state = MarketState(
                pair      = pair,
                bid       = bid,
                ask       = ask,
                spread    = round((ask - bid) * 10000, 2),
                atr       = atr,
                session   = get_session(now.hour),
                timestamp = now.isoformat(),
            )

            result = await orchestrator.process_market_event(state)
            print(f"\n[{pair}] Cycle {cycle+1}: {result.get('action')} "
                  f"(score: {result.get('score', result.get('buy_score', 0)):.3f})")

            await asyncio.sleep(2)   # 2s between events in sim

        await asyncio.sleep(5)       # 5s between cycles


async def main():
    parser = argparse.ArgumentParser(description="Forex Agentic Swarm")
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "XAUUSD"], help="Pairs to trade")
    parser.add_argument("--mode",  choices=["live", "sim"], default="sim", help="Run mode")
    args = parser.parse_args()

    orchestrator = OrchestratorAgent()
    memory       = MemoryAgent()

    print(f"""
╔══════════════════════════════════════════╗
║      FOREX AGENTIC SWARM — v1.0          ║
║  Mode:    {args.mode:<30} ║
║  Pairs:   {', '.join(args.pairs):<30} ║
║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M'):<30} ║
╚══════════════════════════════════════════╝
""")

    if args.mode == "sim":
        await simulate_market_events(args.pairs, orchestrator)
        summary = memory.get_summary()
        print(f"\n📊 SESSION SUMMARY:\n{summary}")
    else:
        print("Live mode: connect your real-time price feed here.")
        print("See README.md for OANDA streaming setup.")


if __name__ == "__main__":
    asyncio.run(main())
