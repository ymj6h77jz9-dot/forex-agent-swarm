"""
DEERFLOW RESEARCH AGENT
========================
Long-horizon research agent — parallel subagents, loop detection,
memory-backed context, and structured synthesis.

Source: ymj6h77jz9-dot/deer-flow
v2: Fully migrated to llm_client (OpenRouter free). All LLM calls
    go through llm() / llm_json() — no bare openai imports.
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from llm_client import llm, llm_json

logger = logging.getLogger(__name__)


class SubagentStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class ResearchTask:
    task_id:      str
    query:        str
    agent_type:   str
    status:       SubagentStatus = SubagentStatus.PENDING
    result:       str = ""
    error:        str = ""
    started_at:   Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ResearchMemory:
    """DeerFlow-style structured memory — deerflow/agents/memory/storage.py pattern."""
    version:          str = "1.0"
    work_context:     str = ""
    personal_context: str = ""
    top_of_mind:      str = ""
    recent_months:    str = ""
    facts:            List[str] = field(default_factory=list)


RESEARCHER_SYSTEM_PROMPT = """
You are a deep-dive forex research agent. Provide comprehensive research briefs
the trading swarm uses to make informed decisions.

Break research into parallel sub-tasks, synthesize findings, extract key facts,
and flag conflicting signals.

Return ONLY valid JSON:
{
  "summary":       "<2-3 sentence executive summary>",
  "key_findings":  ["finding1", "finding2"],
  "sentiment":     "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence":    <float 0-1>,
  "facts":         ["fact1", "fact2"],
  "conflicts":     ["conflict1"],
  "recommendation": "<actionable trading implication>"
}
"""

SUBAGENT_SYSTEM_PROMPT = """
You are a forex research subagent. Provide concise, fact-based analysis.
Focus on: market-moving data, central bank signals, key events, technical levels.
Be specific. No filler. Max 200 words.
"""


class DeerFlowResearcher:
    """
    Long-horizon research agent using DeerFlow's architecture.
    All LLM calls route through llm_client → OpenRouter free.
    """

    SUBAGENT_TIMEOUT = 30.0
    MAX_CONCURRENT   = 3
    _loop_fingerprints: set = set()

    def __init__(self):
        self.memory         = ResearchMemory()
        self.research_cache: Dict[str, str] = {}
        self.token_count    = 0
        self.max_tokens     = 50_000

    async def research(self, topic: str, pair: str, context: str = "") -> Dict[str, Any]:
        """
        Run a comprehensive research pass on a forex topic.

        Args:
            topic:   e.g. "ECB rate decision impact on EURUSD"
            pair:    Currency pair
            context: Recent price action / current signals

        Returns:
            Structured research brief dict
        """
        fingerprint = f"{topic[:50]}:{pair}"

        # Loop detection — DeerFlow middleware pattern
        if fingerprint in self._loop_fingerprints:
            logger.info(f"[DeerFlow] Loop detected for: {fingerprint}")
            cached = self.research_cache.get(fingerprint)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass

        self._loop_fingerprints.add(fingerprint)

        sub_tasks = self._decompose_research(topic, pair)
        results   = await self._execute_subagents(sub_tasks)
        brief     = await self._synthesize(topic, pair, results, context)
        self._update_memory(topic, brief)

        try:
            self.research_cache[fingerprint] = json.dumps(brief)
        except Exception:
            pass

        return brief

    # ── Task decomposition ────────────────────────────────────────────────────

    def _decompose_research(self, topic: str, pair: str) -> List[ResearchTask]:
        import uuid
        base  = pair[:3]
        quote = pair[3:]
        return [
            ResearchTask(
                task_id    = str(uuid.uuid4())[:8],
                query      = f"{base} central bank policy interest rate outlook {topic}",
                agent_type = "fundamental",
            ),
            ResearchTask(
                task_id    = str(uuid.uuid4())[:8],
                query      = f"Breaking news {pair} {topic} forex signal",
                agent_type = "news",
            ),
            ResearchTask(
                task_id    = str(uuid.uuid4())[:8],
                query      = f"{quote} economic outlook inflation employment {topic}",
                agent_type = "fundamental",
            ),
        ][:self.MAX_CONCURRENT]

    # ── Subagent execution ────────────────────────────────────────────────────

    async def _execute_subagents(self, tasks: List[ResearchTask]) -> List[ResearchTask]:
        """Parallel subagent execution with timeout — executor.py pattern."""
        async def run_task(task: ResearchTask) -> ResearchTask:
            task.status     = SubagentStatus.RUNNING
            task.started_at = datetime.now(timezone.utc).isoformat()
            try:
                if self.token_count >= self.max_tokens:
                    task.status = SubagentStatus.FAILED
                    task.error  = "Token budget exhausted"
                    return task
                result = await asyncio.wait_for(
                    self._run_research_subagent(task.query, task.agent_type),
                    timeout=self.SUBAGENT_TIMEOUT,
                )
                task.result       = result
                task.status       = SubagentStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc).isoformat()
            except asyncio.TimeoutError:
                task.status = SubagentStatus.TIMED_OUT
                task.error  = f"Timeout after {self.SUBAGENT_TIMEOUT}s"
                logger.warning(f"[DeerFlow] Subagent {task.task_id} timed out")
            except Exception as e:
                task.status = SubagentStatus.FAILED
                task.error  = str(e)
                logger.error(f"[DeerFlow] Subagent {task.task_id} failed: {e}")
            return task

        return list(await asyncio.gather(*[run_task(t) for t in tasks]))

    async def _run_research_subagent(self, query: str, agent_type: str) -> str:
        """Individual subagent execution via llm_client."""
        prompt = (
            f"Research task ({agent_type}): {query}\n\n"
            "Focus on: market-moving data, central bank signals, key events, "
            "technical levels. Be specific. Max 200 words."
        )
        result = await llm(
            messages=[
                {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        # Rough token estimate (4 chars ≈ 1 token)
        self.token_count += len(result) // 4
        return result

    # ── Synthesis ─────────────────────────────────────────────────────────────

    async def _synthesize(
        self,
        topic:   str,
        pair:    str,
        tasks:   List[ResearchTask],
        context: str,
    ) -> Dict[str, Any]:
        """Synthesize subagent results into a structured research brief."""
        completed = [t for t in tasks if t.status == SubagentStatus.COMPLETED]

        if not completed:
            logger.warning(f"[DeerFlow] No subagents completed for: {topic}")
            return {
                "summary":        "Research failed — all subagents timed out or errored",
                "key_findings":   [],
                "sentiment":      "NEUTRAL",
                "confidence":     0.0,
                "facts":          [],
                "conflicts":      [],
                "recommendation": "No research available",
            }

        research_text = "\n\n".join(
            f"[{t.agent_type.upper()}]\n{t.result}" for t in completed
        )
        past_ctx = (
            f"\nPAST RESEARCH CONTEXT:\n{self.memory.work_context[:500]}\n"
            if self.memory.work_context else ""
        )

        prompt = (
            f"Research Topic: {topic}\nPair: {pair}{past_ctx}\n\n"
            f"SUBAGENT RESULTS:\n{research_text}\n\n"
            f"CURRENT CONTEXT:\n{context[:500] if context else 'N/A'}\n\n"
            "Synthesize into a structured research brief. Return JSON only."
        )

        try:
            brief = await llm_json(
                messages=[
                    {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return brief if brief else {
                "summary": "Synthesis error", "sentiment": "NEUTRAL",
                "confidence": 0.0, "key_findings": [], "facts": [],
                "conflicts": [], "recommendation": "Research unavailable",
            }
        except Exception as e:
            logger.error(f"[DeerFlow] _synthesize failed: {e}")
            return {
                "summary": f"Synthesis error: {e}", "sentiment": "NEUTRAL",
                "confidence": 0.0, "key_findings": [], "facts": [],
                "conflicts": [], "recommendation": "Research unavailable",
            }

    # ── Memory management ─────────────────────────────────────────────────────

    def _update_memory(self, topic: str, brief: Dict[str, Any]):
        """Update DeerFlow-style structured memory — memory/updater.py pattern."""
        new_facts = brief.get("facts", [])
        self.memory.facts.extend(new_facts)
        self.memory.facts = list(set(self.memory.facts))[-50:]

        summary = brief.get("summary", "")
        if summary:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.memory.work_context = (
                f"[{ts}] {topic}: {summary}\n" + self.memory.work_context
            )[:2000]

        logger.debug(f"[DeerFlow] Memory updated | facts={len(self.memory.facts)}")
