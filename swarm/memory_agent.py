import logging

logger = logging.getLogger(__name__)

"""
MEMORY & FEEDBACK AGENT — Forex Agentic Swarm
------------------------------------------------
Persists trade outcomes, tracks agent performance, updates agent weights
dynamically based on win/loss attribution, and feeds learnings back to
the Orchestrator. This is the learning loop that improves the swarm over time.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

MEMORY_FILE = Path(__file__).parent / "data" / "trade_memory.json"


class MemoryAgent:
    def __init__(self):
        self.name   = "memory"
        self.memory = self._load()

    def _load(self) -> dict:
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        return {
            "trades":          [],
            "agent_stats":     {
                "analyst":   {"wins": 0, "losses": 0, "total": 0},
                "sentiment": {"wins": 0, "losses": 0, "total": 0},
                "risk":      {"wins": 0, "losses": 0, "total": 0},
            },
            "performance": {
                "total_trades": 0,
                "wins":         0,
                "losses":       0,
                "win_rate":     0.0,
                "total_pnl":    0.0,
                "best_trade":   None,
                "worst_trade":  None,
            }
        }

    def _save(self):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=2)

    def log_trade(self, trade: dict):
        """Persist a new trade to memory."""
        trade["logged_at"] = datetime.now(timezone.utc).isoformat()
        self.memory["trades"].append(trade)
        self.memory["performance"]["total_trades"] += 1
        self._save()
        logger.info(f"[MEMORY] Trade logged: {trade.get('pair')} {trade.get('direction')}")

    def record_outcome(self, trade_id: str, pnl: float, votes: list):
        """
        Called when a trade closes. Updates win/loss stats and
        agent performance attribution based on their votes.
        """
        outcome = "WIN" if pnl > 0 else "LOSS"
        self.memory["performance"]["total_pnl"] += pnl

        if pnl > 0:
            self.memory["performance"]["wins"] += 1
        else:
            self.memory["performance"]["losses"] += 1

        total = self.memory["performance"]["total_trades"]
        wins  = self.memory["performance"]["wins"]
        self.memory["performance"]["win_rate"] = round(wins / total, 4) if total else 0.0

        # Attribute win/loss to each agent based on their vote alignment
        for vote in votes:
            agent_name = vote.get("agent_name")
            if agent_name in self.memory["agent_stats"]:
                stats = self.memory["agent_stats"][agent_name]
                stats["total"] += 1
                if outcome == "WIN":
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

        # Update best/worst trade tracking
        if (
            self.memory["performance"]["best_trade"] is None or
            pnl > self.memory["performance"]["best_trade"].get("pnl", float("-inf"))
        ):
            self.memory["performance"]["best_trade"] = {"trade_id": trade_id, "pnl": pnl}

        if (
            self.memory["performance"]["worst_trade"] is None or
            pnl < self.memory["performance"]["worst_trade"].get("pnl", float("inf"))
        ):
            self.memory["performance"]["worst_trade"] = {"trade_id": trade_id, "pnl": pnl}

        self._save()
        logger.info(f"[MEMORY] Outcome recorded: {outcome} | PnL: ${pnl:.2f}")

    def get_updated_weights(self) -> dict:
        """
        Dynamically recalculate agent weights based on historical accuracy.
        Returns normalized weights that the Orchestrator can apply.
        """
        stats    = self.memory["agent_stats"]
        accuracy = {}

        for agent, s in stats.items():
            if s["total"] > 0:
                accuracy[agent] = s["wins"] / s["total"]
            else:
                accuracy[agent] = 0.33  # Equal weight when no history

        total_acc = sum(accuracy.values()) or 1.0
        weights   = {a: round(v / total_acc, 4) for a, v in accuracy.items()}

        logger.info(f"[MEMORY] Updated weights: {weights}")
        return weights

    def get_summary(self) -> dict:
        """Return a human-readable performance summary."""
        p = self.memory["performance"]
        return {
            "total_trades":  p["total_trades"],
            "win_rate":      f"{p['win_rate'] * 100:.1f}%",
            "total_pnl":     f"${p['total_pnl']:,.2f}",
            "best_trade":    p["best_trade"],
            "worst_trade":   p["worst_trade"],
            "agent_stats":   self.memory["agent_stats"],
        }
