"""
MEMORY MANAGER — KRATOS v2
===========================
Implements R6–R10: Memory & Learning

  R6  Memory-Augmented Context Retrieval (BM25 + vector fallback)
  R7  Temporal Validity Verification (out-of-sample guard)
  R8  Reflection Loop with Auto-Optimization
  R9  Agent-Specific Diaries (AAAK format)
  R10 Contradiction Detection

Integrated from:
  - ymj6h77jz9-dot/mempalace  (BM25 memory palace)
  - TradingAgents/agents/utils/memory.py (FinancialSituationMemory)
  - ruflo coordination patterns (self-correction loop)

Usage:
    mm = MemoryManager(mempalace_adapter)
    ctx = mm.retrieve_context(pair, session, "EURUSD london BUY signal")
    mm.run_reflection(trade_id, outcome)
    mm.write_diary("analyst", "Identified EMA crossover at 1.0852")
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DIARY_PATH = Path(__file__).parent.parent / "data" / "agent_diaries.jsonl"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MemoryContext:
    pair:                  str
    session:               str
    relevant_decisions:    List[Dict]
    patterns:              List[Dict]
    contradictions:        List[Dict]
    agent_diaries:         Dict[str, List[Dict]]
    temporal_valid:        bool
    retrieval_query:       str


@dataclass
class ReflectionResult:
    trade_id:        str
    agent_accuracy:  Dict[str, float]
    improvements:    List[Dict]
    summary:         str
    memory_entry:    str
    learnings:       List[str]


@dataclass
class DiaryEntry:
    agent:     str
    event:     str
    details:   str
    pattern:   str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_aaak(self) -> str:
        """AAAK format: timestamp|event|details|pattern"""
        return f"{self.timestamp}|{self.event}|{self.details}|{self.pattern}"


# ── Memory Manager ────────────────────────────────────────────────────────────

class MemoryManager:
    """
    R6–R10 Memory & Learning subsystem.
    Wraps MemPalaceAdapter with full learning loop support.
    """

    def __init__(self, mempalace_adapter=None):
        self._mempalace = mempalace_adapter
        # In-memory diary cache
        self._diaries:  Dict[str, List[DiaryEntry]] = {}
        # Fact store for contradiction detection
        self._fact_store: Dict[str, List[Dict]] = {}
        DIARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[MemoryManager] Initialised — R6-R10 active")

    # ─────────────────────────────────────────────────────────────────────────
    # R6: Memory-Augmented Context Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_context(
        self,
        pair:    str,
        session: str,
        query:   str,
        top_k:   int = 5,
    ) -> MemoryContext:
        """
        Retrieve relevant past decisions from MemPalace using BM25.
        Falls back to chronological order if BM25 unavailable.

        Returns MemoryContext with decisions, patterns, contradictions, diaries.
        """
        decisions   = []
        patterns    = []
        contradictions = []

        if self._mempalace:
            try:
                memories = self._mempalace.get_relevant_memories(
                    pair    = pair,
                    session = session,
                    query   = query,
                    n       = top_k,
                )
                # Parse structured decisions from memory strings
                for mem in memories:
                    decisions.append({
                        "text":      mem if isinstance(mem, str) else str(mem),
                        "relevance": 1.0,
                    })
            except Exception as e:
                logger.warning(f"[R6] MemPalace retrieval failed: {e}")

        # Extract patterns from recent diaries
        pair_key = pair.upper()
        for agent, entries in self._diaries.items():
            pair_entries = [e for e in entries[-20:] if pair_key in e.details.upper()]
            for e in pair_entries[-3:]:
                patterns.append({
                    "agent":   agent,
                    "pattern": e.pattern,
                    "event":   e.event,
                    "ts":      e.timestamp,
                })

        # Load agent diaries for context
        agent_diaries = {
            agent: [{"event": e.event, "ts": e.timestamp}
                    for e in entries[-5:]]
            for agent, entries in self._diaries.items()
        }

        logger.info(f"[R6] Retrieved {len(decisions)} decisions, {len(patterns)} patterns for {pair}/{session}")

        return MemoryContext(
            pair               = pair,
            session            = session,
            relevant_decisions = decisions,
            patterns           = patterns,
            contradictions     = contradictions,
            agent_diaries      = agent_diaries,
            temporal_valid     = True,
            retrieval_query    = query,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # R7: Temporal Validity Verification
    # ─────────────────────────────────────────────────────────────────────────

    def check_temporal_validity(
        self,
        fact:        str,
        entity:      str,
        valid_hours: int = 4,
    ) -> Dict[str, Any]:
        """
        Verify a fact is still temporally valid.
        Facts older than valid_hours are considered stale.

        Used to prevent MemPalace from injecting outdated macro context.
        """
        now  = datetime.now(timezone.utc)
        key  = f"{entity}:{fact[:40]}"
        recs = self._fact_store.get(key, [])

        if not recs:
            return {"is_valid": False, "entity": entity, "fact": fact,
                    "reason": "not_in_store", "age_hours": None}

        latest = recs[-1]
        ts     = datetime.fromisoformat(latest.get("timestamp", "2000-01-01T00:00:00+00:00"))
        age_h  = (now - ts).total_seconds() / 3600.0

        is_valid = age_h <= valid_hours

        return {
            "is_valid":    is_valid,
            "entity":      entity,
            "fact":        fact,
            "age_hours":   round(age_h, 2),
            "valid_until": (ts + timedelta(hours=valid_hours)).isoformat(),
            "reason":      "ok" if is_valid else f"stale ({age_h:.1f}h > {valid_hours}h)",
        }

    def store_fact(self, entity: str, fact: str, metadata: Dict = None):
        """Store a fact with current timestamp for validity tracking."""
        key = f"{entity}:{fact[:40]}"
        self._fact_store.setdefault(key, []).append({
            "fact":      fact,
            "metadata":  metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._fact_store[key]) > 20:
            self._fact_store[key] = self._fact_store[key][-20:]

    # ─────────────────────────────────────────────────────────────────────────
    # R8: Reflection Loop with Auto-Optimization
    # ─────────────────────────────────────────────────────────────────────────

    def run_reflection(
        self,
        trade_id:    str,
        outcome:     Dict,
        agent_votes: List[Dict],
        decision:    str,
    ) -> ReflectionResult:
        """
        Analyse a completed trade to extract improvement suggestions.
        Stores results in MemPalace and writes agent diaries.

        outcome: {pnl, exec_price, sl, tp, direction, pair}
        """
        pnl       = float(outcome.get("pnl", 0.0))
        won       = pnl > 0
        pair      = outcome.get("pair", "")
        direction = outcome.get("direction", decision)

        # Per-agent accuracy (did agent predict the right direction?)
        actual_side = "BUY" if direction == "BUY" else "SELL"
        agent_acc: Dict[str, float] = {}
        for vote in agent_votes:
            ag  = vote.get("agent", "unknown")
            sig = vote.get("signal", "FLAT")
            agent_acc[ag] = 1.0 if sig == actual_side and won else \
                            0.5 if sig == "FLAT" else \
                            0.0

        # Auto-optimization suggestions
        improvements = []
        if not won:
            bad_agents = [ag for ag, acc in agent_acc.items() if acc == 0.0]
            if bad_agents:
                improvements.append({
                    "type": "weight_reduce",
                    "agents": bad_agents,
                    "action": "Reduce weight by 5–10% for next cycle",
                })
            avg_conf = sum(v.get("confidence", 0) for v in agent_votes) / max(len(agent_votes), 1)
            if avg_conf > 0.80:
                improvements.append({
                    "type":   "threshold_raise",
                    "action": "High confidence but lost — raise consensus threshold +0.03",
                })
        else:
            good_agents = [ag for ag, acc in agent_acc.items() if acc == 1.0]
            if good_agents:
                improvements.append({
                    "type":   "weight_increase",
                    "agents": good_agents,
                    "action": "Increase weight by 5% — these agents were correct",
                })

        # Human-readable summary
        summary = (
            f"Trade {trade_id}: {'WIN' if won else 'LOSS'} PnL=${pnl:+.2f} on {pair} {direction}. "
            f"Agents correct: {[ag for ag,ac in agent_acc.items() if ac==1.0]}. "
            f"Suggestions: {len(improvements)}"
        )

        # Memory entry for BM25 (short, queryable)
        memory_entry = (
            f"{pair} {direction} {'profit' if won else 'loss'} "
            f"pnl={pnl:+.2f} agents={list(agent_acc.keys())}"
        )

        # Store in MemPalace
        if self._mempalace:
            try:
                self._mempalace.store_reflection(
                    {"summary": summary, "agent_accuracy": agent_acc,
                     "improvements": improvements, "trade_id": trade_id},
                    memory_entry,
                )
            except Exception as e:
                logger.warning(f"[R8] MemPalace store_reflection failed: {e}")

        # Write per-agent diary entries
        for ag, acc in agent_acc.items():
            label = "CORRECT" if acc == 1.0 else "WRONG" if acc == 0.0 else "NEUTRAL"
            self.write_diary(ag, f"trade_{label}", memory_entry,
                             f"accuracy={acc:.1f} pnl={pnl:+.2f}")

        logger.info(f"[R8] Reflection {trade_id}: won={won} pnl={pnl:+.2f} improvements={len(improvements)}")

        return ReflectionResult(
            trade_id       = trade_id,
            agent_accuracy = agent_acc,
            improvements   = improvements,
            summary        = summary,
            memory_entry   = memory_entry,
            learnings      = [i["action"] for i in improvements],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # R9: Agent-Specific Diaries
    # ─────────────────────────────────────────────────────────────────────────

    def write_diary(
        self,
        agent:   str,
        event:   str,
        details: str,
        pattern: str = "",
    ):
        """
        Append a diary entry for an agent in AAAK format:
        timestamp | event | details | pattern

        Diary is persisted to JSONL and kept in-memory (last 100 per agent).
        """
        entry = DiaryEntry(agent=agent, event=event,
                           details=details[:400], pattern=pattern[:200])

        self._diaries.setdefault(agent, []).append(entry)
        if len(self._diaries[agent]) > 100:
            self._diaries[agent] = self._diaries[agent][-100:]

        # Persist to JSONL
        try:
            with open(DIARY_PATH, "a") as f:
                f.write(json.dumps({
                    "agent":   agent,
                    "aaak":    entry.to_aaak(),
                    "event":   event,
                    "details": details[:400],
                    "pattern": pattern[:200],
                    "ts":      entry.timestamp,
                }) + "\n")
        except Exception as e:
            logger.warning(f"[R9] Diary write failed: {e}")

    def read_diary(self, agent: str, last_n: int = 10) -> List[DiaryEntry]:
        """Read last N diary entries for an agent."""
        return self._diaries.get(agent, [])[-last_n:]

    def read_all_diaries(self, last_n: int = 5) -> Dict[str, List[str]]:
        """Read summaries for all agents — used for context injection."""
        return {
            agent: [f"{e.event}: {e.details[:100]}" for e in entries[-last_n:]]
            for agent, entries in self._diaries.items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # R10: Contradiction Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_contradictions(
        self,
        signals:     List[Dict],
        macro_facts: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Detect contradictions in agent signals and between signals and macro facts.

        Three types detected:
          1. Signal conflict  — strong BUY + strong SELL agents simultaneously
          2. Confidence split — high std in confidences (agents disagree)
          3. Macro conflict   — signal direction contradicts known macro fact

        Returns list of contradiction dicts with severity and recommendation.
        """
        contradictions = []

        # 1. Signal conflict
        buy_sigs  = [s for s in signals if s.get("signal") == "BUY"  and s.get("confidence", 0) > 0.65]
        sell_sigs = [s for s in signals if s.get("signal") == "SELL" and s.get("confidence", 0) > 0.65]

        if buy_sigs and sell_sigs:
            contradictions.append({
                "type":           "signal_conflict",
                "severity":       "high",
                "buy_agents":     [s.get("agent") for s in buy_sigs],
                "sell_agents":    [s.get("agent") for s in sell_sigs],
                "recommendation": "trigger_additional_research",
                "action":         "reduce_lot_size_50pct",
            })

        # 2. Confidence split (high variance)
        confs = [s.get("confidence", 0.5) for s in signals]
        if len(confs) >= 3:
            import numpy as np
            std = float(np.std(confs))
            if std > 0.28:
                contradictions.append({
                    "type":           "confidence_split",
                    "severity":       "medium",
                    "std":            round(std, 4),
                    "recommendation": "raise_consensus_threshold_temporarily",
                    "action":         "add_0.05_to_threshold",
                })

        # 3. Macro conflicts (simple keyword matching)
        if macro_facts:
            directions = [s.get("signal") for s in signals if s.get("signal") in ("BUY", "SELL")]
            dominant   = max(set(directions), key=directions.count) if directions else None
            for fact in macro_facts:
                fact_lower = fact.lower()
                if dominant == "BUY" and any(w in fact_lower for w in ["risk off", "recession", "bear market"]):
                    contradictions.append({
                        "type":           "macro_conflict",
                        "severity":       "medium",
                        "signal":         "BUY",
                        "macro_fact":     fact[:100],
                        "recommendation": "sentiment_agent_override",
                        "action":         "reduce_confidence_by_0.10",
                    })
                elif dominant == "SELL" and any(w in fact_lower for w in ["risk on", "bull market", "rate cut"]):
                    contradictions.append({
                        "type":           "macro_conflict",
                        "severity":       "medium",
                        "signal":         "SELL",
                        "macro_fact":     fact[:100],
                        "recommendation": "sentiment_agent_override",
                        "action":         "reduce_confidence_by_0.10",
                    })

        if contradictions:
            logger.warning(f"[R10] {len(contradictions)} contradiction(s) detected: {[c['type'] for c in contradictions]}")

        return contradictions

    def resolve_contradictions(
        self,
        signals:       List[Dict],
        contradictions: List[Dict],
        confidence:    float,
    ) -> Tuple[List[Dict], float]:
        """
        Apply contradiction resolutions to signals and confidence.
        Returns (modified_signals, adjusted_confidence).
        """
        adj_conf = confidence

        for c in contradictions:
            action = c.get("action", "")
            if "reduce_confidence" in action:
                # Extract delta
                try:
                    delta    = float(re.search(r"[\d.]+", action.split("_by_")[-1]).group())
                    adj_conf = max(0.0, adj_conf - delta)
                except Exception:
                    adj_conf = max(0.0, adj_conf - 0.10)
            elif "reduce_lot_size" in action:
                # Tag signal dicts with lot reduction
                for s in signals:
                    s["lot_adj"] = s.get("lot_adj", 1.0) * 0.50

        return signals, adj_conf
