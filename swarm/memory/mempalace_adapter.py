"""
MEMPALACE MEMORY ADAPTER — Integrated from ymj6h77jz9-dot/mempalace
--------------------------------------------------------------------
The highest-scoring AI memory system ever benchmarked (96.6% LongMemEval).
Stores raw verbatim trade cycles without summarization, organized into:
  - Wing:  "forex_swarm" (project)
  - Hall:  "decisions" | "reflections" | "signals" (memory type)
  - Room:  pair + session + date (specific context)

Uses BM25 (Best Matching 25) for retrieval — same algorithm as TradingAgents
FinancialSituationMemory but extended with room/hall organization.

Source: mempalace — "store everything, then make it findable"
Source: TradingAgents/agents/utils/memory.py (BM25 implementation)
"""

import re
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed. Install with: pip install rank-bm25")


class MemoryRoom:
    """
    A single 'room' in the Memory Palace — stores related memories together.
    e.g., Room("EURUSD_london_2026-05") contains all EU london session memories for May 2026.
    """

    def __init__(self, name: str):
        self.name      = name
        self.documents: List[str] = []
        self.metadata:  List[Dict[str, Any]] = []
        self.bm25:      Optional[Any] = None

    def add(self, text: str, meta: Dict[str, Any] = None):
        self.documents.append(text)
        self.metadata.append(meta or {})
        self._rebuild()

    def _rebuild(self):
        if BM25_AVAILABLE and self.documents:
            tokenized = [self._tokenize(d) for d in self.documents]
            self.bm25  = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        if not self.documents:
            return []
        if BM25_AVAILABLE and self.bm25:
            scores = self.bm25.get_scores(self._tokenize(query))
            ranked = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )[:n]
            return [
                {
                    "text":  self.documents[i],
                    "score": float(s),
                    "meta":  self.metadata[i],
                }
                for i, s in ranked if s > 0
            ]
        # Fallback: return most recent
        return [{"text": d, "score": 0.0, "meta": m}
                for d, m in zip(self.documents[-n:], self.metadata[-n:])]


class MemPalaceAdapter:
    """
    Memory Palace for the Forex Agentic Swarm.
    
    Structure:
        Wing:  "forex_swarm"
        Halls: "decisions", "reflections", "signals", "agent_lessons"
        Rooms: Named by pair + session + month (e.g., "EURUSD_london_2026-05")

    Integrated from:
        - mempalace (raw verbatim storage, BM25 retrieval)
        - TradingAgents FinancialSituationMemory (BM25 pattern)
    """

    PALACE_FILE = Path(__file__).parent.parent / "data" / "mempalace.json"

    def __init__(self):
        self.halls: Dict[str, Dict[str, MemoryRoom]] = {
            "decisions":    {},
            "reflections":  {},
            "signals":      {},
            "agent_lessons": {},
        }
        self._load()
        logger.info("[MemPalace] Memory Palace initialized.")

    # ── Storage ───────────────────────────────────────────────────────────────
    def store_decision(self, agent_state: Dict[str, Any]):
        """Store a full trade decision cycle verbatim."""
        room_key = self._room_key(agent_state.get("pair", "UNKNOWN"),
                                   agent_state.get("session", "unknown"))
        text = json.dumps(agent_state)
        self._get_room("decisions", room_key).add(text, {
            "pair":      agent_state.get("pair"),
            "session":   agent_state.get("session"),
            "decision":  agent_state.get("final_decision"),
            "timestamp": agent_state.get("timestamp"),
        })
        self._save()

    def store_reflection(self, reflection: Dict[str, Any], context: str):
        """Store post-trade reflection and lesson."""
        room_key = "lessons_" + datetime.now(timezone.utc).strftime("%Y-%m")
        self._get_room("reflections", room_key).add(
            context + "\n\nLESSON: " + reflection.get("summary", ""),
            {"memory_query": reflection.get("memory_query", ""), **reflection}
        )
        self._save()

    def store_signal(self, pair: str, signal: Dict[str, Any]):
        """Store an email/news signal."""
        room_key = f"signals_{pair}_{datetime.now(timezone.utc).strftime('%Y-%m')}"
        self._get_room("signals", room_key).add(
            json.dumps(signal), signal
        )
        self._save()

    def store_agent_lesson(self, agent_name: str, lesson: str, accuracy: float):
        """Store a per-agent performance lesson."""
        self._get_room("agent_lessons", agent_name).add(
            lesson,
            {"agent": agent_name, "accuracy": accuracy,
             "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        self._save()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def get_relevant_memories(self, pair: str, session: str,
                               query: str, n: int = 3) -> List[str]:
        """
        Retrieve the most relevant past decisions/lessons for the current context.
        Called by Orchestrator before each trade cycle.
        """
        results = []

        # 1. Search decision room for this pair/session
        room_key  = self._room_key(pair, session)
        dec_room  = self._get_room("decisions", room_key)
        results  += dec_room.search(query, n=2)

        # 2. Search reflections broadly
        for room in self.halls["reflections"].values():
            results += room.search(query, n=1)

        # Sort by score and deduplicate
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            key = r["text"][:50]
            if key not in seen:
                seen.add(key)
                unique.append(r["text"][:500])   # cap per memory

        return unique[:n]

    def get_agent_lessons(self, agent_name: str, query: str, n: int = 2) -> List[str]:
        """Get past lessons specific to one agent."""
        room = self._get_room("agent_lessons", agent_name)
        return [r["text"] for r in room.search(query, n=n)]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _room_key(self, pair: str, session: str) -> str:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return f"{pair}_{session}_{month}"

    def _get_room(self, hall: str, room_key: str) -> MemoryRoom:
        if room_key not in self.halls[hall]:
            self.halls[hall][room_key] = MemoryRoom(room_key)
        return self.halls[hall][room_key]

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save(self):
        self.PALACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for hall_name, rooms in self.halls.items():
            data[hall_name] = {}
            for room_key, room in rooms.items():
                data[hall_name][room_key] = {
                    "documents": room.documents,
                    "metadata":  room.metadata,
                }
        with open(self.PALACE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not self.PALACE_FILE.exists():
            return
        try:
            with open(self.PALACE_FILE) as f:
                data = json.load(f)
            for hall_name, rooms in data.items():
                if hall_name not in self.halls:
                    self.halls[hall_name] = {}
                for room_key, room_data in rooms.items():
                    room = MemoryRoom(room_key)
                    for doc, meta in zip(room_data["documents"], room_data["metadata"]):
                        room.add(doc, meta)
                    self.halls[hall_name][room_key] = room
            logger.info("[MemPalace] Loaded from disk.")
        except Exception as e:
            logger.warning(f"[MemPalace] Load error: {e}")
