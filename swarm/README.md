# 🤖 KRATOS v2 — Unified Agentic Forex Swarm

A fully integrated multi-agent forex trading system that merges the architectures 
from every relevant GitHub repository into a single, production-grade AI trading engine.

## 📦 Repos Integrated (No Holding Back)

| Repo | Architecture Contribution |
|---|---|
| **KRATOS-app** | `KratosOrchestrator`, `DerivAdapter` (WebSocket), `ExecutionEngine`, `RiskManager` |
| **TradingAgents** | `Propagator`, `Reflector`, `SignalProcessor`, Bull/Bear Debate, Risk Debate, BM25 Memory |
| **MiroFish** | PSO Particle Swarm Intelligence Engine (60 particles, 100 iterations) |
| **mempalace** | BM25 Memory Palace — verbatim storage, wing/hall/room structure |
| **Kronos** | Foundation Model for financial K-lines (NeoQuasar HuggingFace, 45+ exchanges) |
| **tensortrade** | RL meta-agent environment — DQN reward loop, position sizing actions |
| **deer-flow** | Long-horizon researcher, parallel subagent executor, loop detection, DeerFlow memory |
| **QuantDinger** | Multi-agent research framework, multi-asset watchlist |
| **OpenBB** | Unified financial data — candles, snapshots, economic calendar, macro indicators |
| **Crucix** | Multi-source macro intelligence — FRED, BLS, news, regime detection |
| **BB-Terminal** | Multi-pair forex dashboard (FXC component — all major pairs) |
| **ruflo** | Multi-agent coordination, self-correction, consensus propagation patterns |
| **nautilus_trader** | Event-driven deterministic execution architecture |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/ymj6h77jz9-dot/forex-agent-swarm
cd forex-agent-swarm/swarm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Fill in your API keys (see .env.example for all variables)

# 4. Run in simulation mode (no real orders)
python main.py --mode sim --pairs EURUSD XAUUSD GBPUSD

# 5. Run in live mode (real orders via OANDA)
python main.py --mode live --pairs EURUSD XAUUSD
```

---

## 🏗️ 16-Step Decision Pipeline

```
Market Event
     │
     ▼
 1. Propagator.create_initial_state()   ← AgentState broadcast
     │
     ▼
 2. MemPalace.get_relevant_memories()   ← BM25 retrieval (mempalace)
     │
     ▼
 3. CrucixIntelligence.get_briefing()   ← FRED + BLS + News sweep (every 5 cycles)
     │
     ▼
 4. OpenBBProvider.get_candles()        ← Live OHLCV + Economic Calendar
     │  + get_economic_calendar()        → High-impact event guard (50% lot reduction)
     ▼
 5. MiroFishEngine.predict()            ← PSO simulation (60p × 100i)
     │
     ▼
 6. KronosAdapter.predict()             ← Foundation model 10-bar forecast
     │
     ▼
 7. DeerFlowResearcher.research()       ← Long-horizon pair research (every 10 cycles)
     │   └─ parallel subagents (max 3)
     │   └─ loop detection middleware
     │   └─ DeerFlow memory updater
     ▼
 8. asyncio.gather(                     ← All sub-agents run concurrently
     AnalystAgent.analyze(),
     SentimentAgent.analyze(),          ← + Crucix/DeerFlow context injected
     RiskAgent.analyze(),
    )
     │
     ▼
 9. SignalProcessor.resolve_investment_debate()   ← Bull vs Bear debate → judge
     │
     ▼
10. SignalProcessor.compute_weighted_consensus()  ← votes + MiroFish + Kronos blend
     │   Weights: Analyst 28% | Sentiment 22% | Risk 20% | MiroFish 15% | Kronos 15%
     │   Threshold: 0.70
     ▼
11. Crucix USD bias override check
     │
     ▼
12. SignalProcessor.resolve_risk_debate()   ← Aggressive vs Conservative → risk judge
     │
     ▼
13. KratosRLEnvironment.observe()           ← 18-dim observation vector
     └─ _rl_policy()                         → HOLD / EXECUTE_FULL / EXECUTE_HALF / EXECUTE_DOUBLE
     │
     ▼
14. Hard Rule Engine (never bypassed)
     │   spread < 5 pips | ATR > 0.0001 | drawdown < 5% | open trades < 3
     ▼
15. ExecutionAgent.execute_trade()       ← DerivAdapter WebSocket | OANDA REST
     │
     ▼
16. Reflector.reflect()                  ← Post-trade analysis (TradingAgents pattern)
     └─ MemPalace.store_reflection()     ← Lesson stored in wing/hall/room
     └─ MemoryAgent.get_updated_weights() ← Dynamic weight evolution
     └─ MiroFish.reflect_on_performance() ← PSO self-improvement
```

---

## 📁 File Structure

```
swarm/
├── kratos_orchestrator.py       ← THE BRAIN — 16-step pipeline
├── orchestrator_agent.py        ← Base + AgentVote + MarketState
├── analyst_agent.py             ← Technical analysis (GPT-4o)
├── sentiment_agent.py           ← News + Gmail + Crucix context
├── risk_agent.py                ← VaR + lot sizing + hard rules
├── execution_agent.py           ← Order placement
├── memory_agent.py              ← Performance tracking + weight evolution
├── gmail_monitor.py             ← Gmail broker alert classifier
├── main.py                      ← Entry point
│
├── models/
│   └── kronos_adapter.py        ← Kronos foundation model (HuggingFace)
│
├── engines/
│   └── mirofish_engine.py       ← PSO prediction engine
│
├── rl/
│   └── tensortrade_env.py       ← RL meta-agent (TensorTrade-style)
│
├── agents/
│   └── deerflow_researcher.py   ← DeerFlow long-horizon researcher
│
├── broker/
│   └── deriv_adapter.py         ← Deriv WebSocket adapter
│
├── graph/
│   ├── propagation.py           ← AgentState propagation (TradingAgents)
│   ├── reflection.py            ← Post-trade reflection
│   └── signal_processing.py    ← Debate resolution + consensus
│
├── memory/
│   └── mempalace_adapter.py     ← BM25 Memory Palace
│
├── data/
│   ├── openbb_provider.py       ← OpenBB unified data layer
│   └── crucix_intel.py          ← Crucix multi-source macro intel
│
├── requirements.txt             ← All dependencies
└── .env.example                 ← All environment variables
```

---

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | GPT-4o for all agent reasoning |
| `OANDA_API_KEY` | OANDA broker (primary execution) |
| `OANDA_ACCOUNT_ID` | OANDA account ID |
| `DERIV_API_KEY` | Deriv WebSocket key (alternative) |
| `KRONOS_MODEL_ID` | HuggingFace model ID (default: `NeoQuasar/Kronos-small`) |
| `FRED_API_KEY` | FRED macro data (Fed Funds, CPI, VIX, yield curve) |
| `NEWS_API_KEY` | NewsAPI.org for real-time news sentiment |
| `ACCOUNT_EQUITY` | Starting equity for lot sizing |
| `RUN_MODE` | `sim` (no real orders) or `live` |
| `WATCH_PAIRS` | Comma-separated pairs to monitor |

---

## 🧠 Agent Weight Evolution

Weights start at defaults and evolve after every closed trade:

```
Analyst:   28%  (Technical analysis accuracy)
Sentiment: 22%  (News + Gmail + Crucix intel accuracy)
Risk:      20%  (Risk assessment accuracy)
MiroFish:  15%  (PSO prediction accuracy — fixed)
Kronos:    15%  (Foundation model accuracy — fixed)
```

MemoryAgent tracks per-agent win rates and adjusts the non-fixed weights every trade close.

---

## 📊 Gmail Automation

Live on Base44. Fires on every new email:
- Classifies broker alerts → `SentimentAgent` signal feed
- Ignores promotional/spam emails
- Logs relevant signals to `ForexEmailSignal` entity
- End-of-day reports sent to `REPORT_EMAIL`
