# 🤖 KRATOS v2 — Unified Agentic Forex Swarm

A fully integrated multi-agent forex trading system that merges the architectures from all your GitHub repositories into a single, production-grade AI trading engine.

## 📦 Repos Integrated

| Repo | Architecture Contribution |
|---|---|
| **KRATOS-app** | KratosOrchestrator, DerivAdapter, ExecutionEngine, RiskManager |
| **TradingAgents** | Propagator, Reflector, SignalProcessor, Bull/Bear Debate, Risk Debate |
| **MiroFish** | PSO Swarm Intelligence Prediction Engine (probabilistic backbone) |
| **mempalace** | BM25 Memory Palace — verbatim storage, findable retrieval |
| **ruflo** | Multi-agent coordination, self-correction patterns |
| **QuantDinger** | Multi-agent research framework, data provider abstraction |
| **nautilus_trader** | Event-driven architecture, deterministic execution patterns |
| **Kronos** | Financial language foundation model concepts |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   KRATOS v2 ORCHESTRATOR                         │
│  (kratos_orchestrator.py)                                        │
│                                                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │  Propagator  │  │  MemPalace    │  │  MiroFish PSO Engine  │  │
│  │ (AgentState  │  │  (BM25 memory │  │  (Particle Swarm     │  │
│  │  broadcast)  │  │   palace)     │  │   prediction)        │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────────────── AGENT ENSEMBLE ──────────────────┐   │
│  │  AnalystAgent   SentimentAgent   RiskAgent   (parallel)  │   │
│  │  (Technical)    (News+Gmail)     (VaR+Rules)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────── DEBATE LAYER ────────────────────────┐   │
│  │  Bull/Bear Investment Debate  →  Judge                   │   │
│  │  Aggressive/Conservative Risk Debate  →  Risk Judge      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────── SIGNAL LAYER ────────────────────────┐   │
│  │  SignalProcessor: weighted_consensus + MiroFish blend    │   │
│  │  Threshold: 0.70 | Hard Rules: spread, ATR, drawdown     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────── EXECUTION ───────────────────────────┐   │
│  │  ExecutionAgent → DerivAdapter (WebSocket) / OANDA REST  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────── LEARNING LOOP ───────────────────────┐   │
│  │  Reflector (post-trade) → MemPalace (store lessons)      │   │
│  │  MemoryAgent (update weights) → Dynamic weight evolution  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your env vars
cp .env.example .env

# 3. Run in simulation mode (no real orders)
python main.py --mode sim --pairs EURUSD XAUUSD

# 4. Run in live mode (real Deriv orders)
python main.py --mode live --pairs EURUSD XAUUSD GBPUSD
```

---

## 📁 File Structure

```
swarm/
├── kratos_orchestrator.py      ← MAIN BRAIN (integrates everything)
├── orchestrator_agent.py       ← Base orchestrator + AgentVote
├── analyst_agent.py            ← Technical analysis (GPT-4o)
├── sentiment_agent.py          ← News + Gmail signals (GPT-4o)
├── risk_agent.py               ← Risk management + hard rules
├── execution_agent.py          ← Order placement (OANDA)
├── memory_agent.py             ← Performance tracking + weight evolution
├── gmail_monitor.py            ← Gmail broker alert classifier
├── main.py                     ← Entry point
│
├── engines/
│   └── mirofish_engine.py      ← PSO simulation backbone (from MiroFish)
│
├── broker/
│   └── deriv_adapter.py        ← Deriv WebSocket adapter (from KRATOS-app)
│
├── graph/
│   ├── propagation.py          ← AgentState propagation (from TradingAgents)
│   ├── reflection.py           ← Post-trade reflection (from TradingAgents)
│   └── signal_processing.py   ← Signal extraction + debate resolution
│
├── memory/
│   └── mempalace_adapter.py   ← BM25 Memory Palace (from mempalace)
│
└── data/
    ├── trade_memory.json       ← Trade log + performance stats
    └── mempalace.json          ← Persistent memory palace
```

---

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | GPT-4o for all agent reasoning |
| `OANDA_API_KEY` | OANDA broker key (primary execution) |
| `OANDA_ACCOUNT_ID` | OANDA account ID |
| `OANDA_BASE_URL` | `https://api-fxpractice.oanda.com` (practice) or `https://api-fxtrade.oanda.com` (live) |
| `DERIV_API_KEY` | Deriv WebSocket API key (alternative broker) |
| `NEWS_API_KEY` | NewsAPI.org for real-time news sentiment |
| `ACCOUNT_EQUITY` | Starting equity for lot sizing (default: 10000) |
| `REPORT_EMAIL` | Email for daily swarm reports |

---

## 🧠 How Update Propagation Works

1. `MarketState` event fires → `Propagator.create_initial_state()` broadcasts to all agents
2. `MemPalace` injects relevant past memories into the shared `AgentState`
3. `MiroFishEngine` runs 50-particle PSO simulation → probabilistic price forecast
4. `AnalystAgent`, `SentimentAgent`, `RiskAgent` run **concurrently** via `asyncio.gather()`
5. Each agent writes its vote back into `AgentState` via `Propagator.propagate_agent_output()`
6. Bull/Bear debate → judge decision propagates to `AgentState.investment_plan`
7. `SignalProcessor.compute_weighted_consensus()` blends votes + MiroFish at 20% weight
8. If score > 0.70 threshold → `ExecutionAgent` places order via `DerivAdapter`
9. `Reflector` runs post-trade → stores lesson in `MemPalace` room
10. `MemoryAgent.get_updated_weights()` evolves agent weights based on accuracy history

---

## 📊 Gmail Integration

The `GmailMonitor` listens to your inbox and classifies broker alerts as trading signals:
- Broker alerts → `SentimentAgent.ingest_email_signal()`
- Daily performance reports → sent to `REPORT_EMAIL`
- Triggered automatically via Base44 Gmail connector automation
