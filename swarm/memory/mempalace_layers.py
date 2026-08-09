"""
mempalace_layers.py — KRATOS v2
==================================
4-Layer memory stack ported from mempalace v3.0.12 (layers.py).

Layer 0: Identity       (~100 tokens)  — always loaded. Who is the swarm?
Layer 1: Essential Story (~500 tokens) — always loaded. Top trade moments.
Layer 2: On-Demand      (~200 tokens)  — loaded per pair/session topic.
Layer 3: Deep Search    (unlimited)    — full ChromaDB semantic search.

Wake-up cost: L0+L1 only (~600 tokens). Leaves 95%+ context window free.
L2 and L3 only load when explicitly requested by a pair or query term.

Integration into KRATOS:
  - MemoryManager.retrieve_context() calls L1+L2 for cycle-start injection
  - L3 is called only during DeerFlow deep-dive (every 10 cycles)
  - AuditLogger can query L3 for anomaly cross-reference
"""
from __future__ import annotations

import json, logging, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

MEMPALACE_DIR = Path(os.getenv("MEMPALACE_DIR", os.path.expanduser("~/.mempalace")))
IDENTITY_FILE = MEMPALACE_DIR / "identity.txt"
STORY_FILE    = MEMPALACE_DIR / "story.jsonl"
DIARIES_DIR   = MEMPALACE_DIR / "diaries"


@dataclass
class MemoryContext:
    identity:    str        = ""
    story_items: List[str]  = field(default_factory=list)
    topic_items: List[str]  = field(default_factory=list)
    deep_hits:   List[str]  = field(default_factory=list)
    total_tokens_est: int   = 0


# ── Layer 0: Identity ─────────────────────────────────────────────────────
class Layer0:
    """~100 tokens. Always loaded. Swarm identity and standing rules."""
    _cache: Optional[str] = None

    @classmethod
    def load(cls) -> str:
        if cls._cache:
            return cls._cache
        if IDENTITY_FILE.exists():
            cls._cache = IDENTITY_FILE.read_text(encoding="utf-8").strip()
        else:
            cls._cache = (
                "I am KRATOS v2 — a 16-step agentic forex trading swarm. "
                "I use a weighted consensus of Sentiment, Analyst, Risk, MiroFish, "
                "Kronos, and RL agents to trade major forex pairs. "
                "Zero tolerance for syntax errors. All LLM calls route through llm_client.py."
            )
        logger.debug("[L0] Identity loaded (%d chars)", len(cls._cache))
        return cls._cache


# ── Layer 1: Essential Story ──────────────────────────────────────────────
class Layer1:
    """
    ~500 tokens. Always loaded.
    Top N trade moments from story.jsonl (sorted by significance score).
    """
    @staticmethod
    def load(top_n: int = 5) -> List[str]:
        if not STORY_FILE.exists():
            return []
        items = []
        try:
            with open(STORY_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning("[L1] Story read failed: %s", e)
            return []
        # Sort by significance (higher = more important)
        items.sort(key=lambda x: float(x.get("significance", 0)), reverse=True)
        result = []
        for it in items[:top_n]:
            text = it.get("text") or it.get("summary") or str(it)
            result.append(text)
        logger.debug("[L1] Loaded %d story items", len(result))
        return result


# ── Layer 2: On-Demand ────────────────────────────────────────────────────
class Layer2:
    """
    ~200 tokens per topic. Loaded only when pair/session/topic matches.
    Reads from ~/.mempalace/diaries/<topic>.jsonl
    """
    @staticmethod
    def load(topic: str, last_n: int = 5) -> List[str]:
        if not topic:
            return []
        # Normalise topic to filename
        safe_topic = topic.replace("/", "_").replace(" ", "_").lower()
        diary_file = DIARIES_DIR / f"{safe_topic}.jsonl"
        if not diary_file.exists():
            # Try pair-only prefix
            prefix = safe_topic.split("_")[0]
            matches = list(DIARIES_DIR.glob(f"{prefix}*.jsonl")) if DIARIES_DIR.exists() else []
            if matches:
                diary_file = matches[0]
            else:
                return []
        items = []
        try:
            with open(diary_file) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        items.append({"text": line})
        except Exception as e:
            logger.warning("[L2] Diary read failed for %s: %s", topic, e)
            return []
        # Return last N (most recent)
        result = [it.get("text") or str(it) for it in items[-last_n:]]
        logger.debug("[L2] Loaded %d diary items for '%s'", len(result), topic)
        return result


# ── Layer 3: Deep Search ──────────────────────────────────────────────────
class Layer3:
    """
    Unlimited. Full ChromaDB semantic search.
    Only called during DeerFlow deep-dives or explicit queries.
    """
    _client = None
    _collection = None

    @classmethod
    def _init(cls):
        if cls._client is not None:
            return True
        try:
            import chromadb
            cls._client     = chromadb.PersistentClient(path=str(MEMPALACE_DIR / "chromadb"))
            cls._collection = cls._client.get_or_create_collection("kratos_memories")
            logger.info("[L3] ChromaDB initialised at %s", MEMPALACE_DIR / "chromadb")
            return True
        except Exception as e:
            logger.warning("[L3] ChromaDB unavailable: %s", e)
            return False

    @classmethod
    def search(cls, query: str, n_results: int = 10) -> List[str]:
        if not cls._init():
            return []
        try:
            results = cls._collection.query(
                query_texts = [query],
                n_results   = min(n_results, cls._collection.count()),
            )
            docs = results.get("documents", [[]])[0]
            logger.debug("[L3] Deep search '%s' → %d results", query[:40], len(docs))
            return docs
        except Exception as e:
            logger.warning("[L3] Search failed: %s", e)
            return []

    @classmethod
    def store(cls, text: str, metadata: Optional[Dict] = None):
        if not cls._init():
            return
        try:
            doc_id = f"kratos_{int(time.time()*1000)}"
            cls._collection.add(
                documents = [text],
                ids       = [doc_id],
                metadatas = [metadata or {}],
            )
        except Exception as e:
            logger.warning("[L3] Store failed: %s", e)


# ── Unified loader ────────────────────────────────────────────────────────
def load_memory_context(
    topic:       Optional[str] = None,
    deep_query:  Optional[str] = None,
    story_top_n: int           = 5,
    diary_last_n: int          = 5,
) -> MemoryContext:
    """
    Load all relevant memory layers for a cycle.
    topic:      pair or session string for L2 (e.g. "EUR/USD_london")
    deep_query: if set, runs L3 ChromaDB search
    """
    ctx = MemoryContext()

    # L0
    ctx.identity = Layer0.load()
    # L1
    ctx.story_items = Layer1.load(top_n=story_top_n)
    # L2
    if topic:
        ctx.topic_items = Layer2.load(topic, last_n=diary_last_n)
    # L3 (only if explicitly requested)
    if deep_query:
        ctx.deep_hits = Layer3.search(deep_query, n_results=10)

    # Rough token estimate (4 chars ≈ 1 token)
    total_chars = (
        len(ctx.identity) +
        sum(len(s) for s in ctx.story_items) +
        sum(len(s) for s in ctx.topic_items) +
        sum(len(s) for s in ctx.deep_hits)
    )
    ctx.total_tokens_est = total_chars // 4

    logger.info(
        "[MEMORY] L0=%d L1=%d L2=%d L3=%d (~%d tokens)",
        len(ctx.identity), len(ctx.story_items),
        len(ctx.topic_items), len(ctx.deep_hits),
        ctx.total_tokens_est,
    )
    return ctx


def store_to_layer3(text: str, metadata: Optional[Dict] = None):
    """Convenience wrapper to persist a trade reflection to ChromaDB (L3)."""
    Layer3.store(text, metadata)
