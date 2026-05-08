# 🤖 Forex Agentic Swarm

A multi-agent AI trading system for forex markets. Built with GPT-4o-powered agents,
weighted consensus decision-making, real-time update propagation, and Gmail-integrated
sentiment monitoring.

## Architecture

```
Orchestrator Agent
├── Analyst Agent        (technical analysis, 40% weight)
├── Sentiment Agent      (news + Gmail signals, 30% weight)
└── Risk Agent           (position sizing, hard rules, 30% weight)
         ↓ (consensus > 0.72)
Execution Agent          (OANDA REST API)
         ↓
Memory & Feedback Agent  (learning loop, dynamic weight updates)
```

## Update Propagation Flow

1. Market event fires → Orchestrator receives price state
2. Orchestrator **broadcasts** to Analyst, Sentiment, Risk agents **concurrently**
3. Each agent independently analyzes and returns a structured vote
4. Orchestrator runs **weighted consensus** — score must exceed 0.72 to execute
5. Execution Agent places order via OANDA API (or simulates)
6. Memory Agent persists outcome, recalculates agent weights dynamically

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in your keys
python main.py --pairs EURUSD XAUUSD --mode sim
```

## Gmail Integration

The `GmailMonitor` class is triggered by the Base44 Gmail connector automation.
It:
- Monitors your inbox for broker alerts, news digests, economic calendar emails
- Classifies emails using GPT-4o-mini (cheap, fast)
- Injects structured signals into the Sentiment Agent
- Sends you a daily HTML performance report

## Live Trading

Set `OANDA_BASE_URL=https://api-fxtrade.oanda.com` for live trading.
**Always test on paper/practice first.**

## Hard Risk Rules (never bypassed by LLM)

| Rule | Value |
|---|---|
| Max risk per trade | 2% of equity |
| Max spread allowed | 3.0 pips |
| Max open trades | 3 |
| Max daily drawdown | 5% |
| Min ATR (market activity) | 0.0003 |

## Agent Performance & Learning

The Memory Agent tracks each agent's win/loss attribution.
After enough trades, `get_updated_weights()` returns dynamic weights
based on actual performance — the swarm literally learns which agents
to trust more over time.
