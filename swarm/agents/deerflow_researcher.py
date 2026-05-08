"""
DEERFLOW RESEARCH AGENT
========================
Integrates DeerFlow — "An open-source long-horizon SuperAgent harness that
researches, codes, and creates. With sandboxes, memories, tools, skill,
subagents and message gateway."

The DeerFlowResearcher is a specialized sub-agent that runs long-horizon
research tasks that the main swarm agents can't do in a single pass:
  - Deep-dive research on currency pairs (central bank policy history)
  - Multi-source news aggregation with loop detection
  - Parallel subagent dispatch (mirrors DeerFlow executor.py)
  - Memory-backed context from past research sessions
  - Middleware stack: clarification, loop detection, token usage, memory

Source: ymj6h77jz9-dot/deer-flow
Architecture:
  - lead_agent/agent.py → middleware chain
  - subagents/executor.py → parallel execution with timeout
  - agents/memory/storage.py → structured memory (workContext, facts)
  - agents/middlewares/ → loop detection, clarification, token limits
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class SubagentStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class ResearchTask:
    """A research sub-task dispatched to a subagent."""
    task_id:    str
    query:      str
    agent_type: str         # "web_search" | "fundamental" | "news" | "technical"
    status:     SubagentStatus = SubagentStatus.PENDING
    result:     str = ""
    error:      str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ResearchMemory:
    """
    DeerFlow-style structured memory.
    Source: deerflow/agents/memory/storage.py create_empty_memory()
    """
    version:     str = "1.0"
    work_context: str = ""     # Current research context summary
    personal_context: str = "" # User preferences / trading style
    top_of_mind:  str = ""     # Most urgent items
    recent_months: str = ""    # Historical research summaries
    facts:        List[str] = field(default_factory=list)


RESEARCHER_SYSTEM_PROMPT = """
You are a deep-dive forex research agent with access to web search, news feeds, 
and economic data. Your role is to provide comprehensive research briefs that the 
trading swarm uses to make informed decisions.

When given a research task:
1. Break it into parallel sub-tasks (max 3 concurrent)
2. Synthesize findings into a structured brief
3. Extract key facts for memory storage
4. Flag any conflicting signals or uncertainties

Output format (JSON):
{
  "summary": "<2-3 sentence executive summary>",
  "key_findings": ["finding1", "finding2", ...],
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0-1>,
  "facts": ["fact1", "fact2"],  // for memory storage
  "conflicts": ["conflict1"],   // contradictory signals
  "recommendation": "<actionable trading implication>"
}
"""


class DeerFlowResearcher:
    """
    Long-horizon research agent using DeerFlow's architecture.
    
    Implements:
      - Parallel subagent dispatch with timeout (executor.py pattern)
      - Loop detection middleware (don't repeat identical searches)
      - Memory middleware (inject past research into context)
      - Token usage tracking
    """

    SUBAGENT_TIMEOUT  = 30.0   # seconds
    MAX_CONCURRENT    = 3      # DeerFlow subagent concurrency limit
    LOOP_DETECT_HASH  = set()  # Tracks query fingerprints to avoid loops

    def __init__(self):
        self.memory          = ResearchMemory()
        self.research_cache: Dict[str, str] = {}
        self.token_count     = 0
        self.max_tokens      = 50000   # Budget per session

    async def research(self, topic: str, pair: str,
                        context: str = "") -> Dict[str, Any]:
        """
        Run a comprehensive research pass on a forex topic.
        
        Args:
            topic:   Research topic e.g. "ECB rate decision impact on EURUSD"
            pair:    Currency pair context
            context: Additional context (recent price action, current signals)
        
        Returns:
            Structured research brief dict
        """
        # Loop detection — DeerFlow middleware pattern
        fingerprint = f"{topic[:50]}:{pair}"
        if fingerprint in self.LOOP_DETECT_HASH:
            logger.info(f"[DeerFlow] Loop detected for: {topic[:50]}")
            cached = self.research_cache.get(fingerprint, "")
            if cached:
                return json.loads(cached)

        self.LOOP_DETECT_HASH.add(fingerprint)

        # Build parallel sub-tasks
        sub_tasks = self._decompose_research(topic, pair)
        
        # Execute subagents with timeout (DeerFlow executor.py pattern)
        results = await self._execute_subagents(sub_tasks)
        
        # Synthesize with memory context
        brief = await self._synthesize(topic, pair, results, context)
        
        # Update memory (DeerFlow memory/updater.py pattern)
        self._update_memory(topic, brief)
        
        # Cache result
        self.research_cache[fingerprint] = json.dumps(brief)
        
        return brief

    def _decompose_research(self, topic: str, pair: str) -> List[ResearchTask]:
        """
        Decompose research into parallel sub-tasks.
        DeerFlow pattern: break into max MAX_CONCURRENT parallel tasks.
        """
        import uuid
        base_currency = pair[:3]
        quote_currency = pair[3:]
        
        tasks = [
            ResearchTask(
                task_id    = str(uuid.uuid4())[:8],
                query      = f"Latest {base_currency} central bank policy and interest rate outlook {topic}",
                agent_type = "fundamental",
            ),
            ResearchTask(
                task_id    = str(uuid.uuid4())[:8],
                query      = f"Breaking news {pair} {topic} forex trading signal",
                agent_type = "news",
            ),
            ResearchTask(
                task_id    = str(uuid.uuid4())[:8],
                query      = f"{quote_currency} economic outlook inflation employment {topic}",
                agent_type = "fundamental",
            ),
        ]
        return tasks[:self.MAX_CONCURRENT]

    async def _execute_subagents(self, tasks: List[ResearchTask]) -> List[ResearchTask]:
        """
        Execute subagents in parallel with timeout.
        Source: deerflow/subagents/executor.py pattern.
        """
        async def run_task(task: ResearchTask) -> ResearchTask:
            task.status    = SubagentStatus.RUNNING
            task.started_at = datetime.now(timezone.utc).isoformat()
            try:
                # Check token budget
                if self.token_count >= self.max_tokens:
                    task.status = SubagentStatus.FAILED
                    task.error  = "Token budget exhausted"
                    return task

                result = await asyncio.wait_for(
                    self._run_research_subagent(task.query, task.agent_type),
                    timeout = self.SUBAGENT_TIMEOUT
                )
                task.result       = result
                task.status       = SubagentStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc).isoformat()
            except asyncio.TimeoutError:
                task.status = SubagentStatus.TIMED_OUT
                task.error  = f"Timed out after {self.SUBAGENT_TIMEOUT}s"
            except Exception as e:
                task.status = SubagentStatus.FAILED
                task.error  = str(e)
            return task

        return await asyncio.gather(*[run_task(t) for t in tasks])

    async def _run_research_subagent(self, query: str, agent_type: str) -> str:
        """
        Individual subagent execution. Uses GPT-4o as the research LLM.
        In production, this would call web search tools.
        """
        prompt = f"""
Research task ({agent_type}): {query}

Provide a concise, fact-based analysis focused on:
- Current market-moving data points
- Central bank signals or policy changes  
- Key economic events coming up
- Technical level context if relevant

Be specific. No filler. Max 200 words.
"""
        resp = await client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 300,
        )
        tokens = resp.usage.total_tokens if resp.usage else 0
        self.token_count += tokens
        return resp.choices[0].message.content

    async def _synthesize(self, topic: str, pair: str,
                           tasks: List[ResearchTask],
                           context: str) -> Dict[str, Any]:
        """Synthesize subagent results into structured research brief."""
        completed = [t for t in tasks if t.status == SubagentStatus.COMPLETED]
        
        if not completed:
            return {
                "summary": "Research failed — all subagents timed out or errored",
                "key_findings": [], "sentiment": "NEUTRAL",
                "confidence": 0.0, "facts": [], "conflicts": [],
                "recommendation": "No research available"
            }

        research_text = "\n\n".join([
            f"[{t.agent_type.upper()}]\n{t.result}" for t in completed
        ])

        past_context = ""
        if self.memory.work_context:
            past_context = f"\nPAST RESEARCH CONTEXT:\n{self.memory.work_context[:500]}"

        prompt = f"""
Research Topic: {topic}
Pair: {pair}
{past_context}

SUBAGENT RESEARCH RESULTS:
{research_text}

CURRENT MARKET CONTEXT:
{context[:500] if context else 'Not provided'}

Synthesize into a structured research brief. Return JSON only.
"""
        try:
            resp = await client.chat.completions.create(
                model           = "gpt-4o",
                messages        = [
                    {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format = {"type": "json_object"},
                temperature     = 0.2,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"[DeerFlow] Synthesis failed: {e}")
            return {"summary": "Synthesis error", "sentiment": "NEUTRAL",
                    "confidence": 0.0, "key_findings": [], "facts": [],
                    "conflicts": [], "recommendation": "Research unavailable"}

    def _update_memory(self, topic: str, brief: Dict[str, Any]):
        """
        Update DeerFlow-style structured memory.
        Source: deerflow/agents/memory/updater.py pattern.
        """
        # Add new facts
        new_facts = brief.get("facts", [])
        self.memory.facts.extend(new_facts)
        self.memory.facts = list(set(self.memory.facts))[-50:]   # Keep last 50

        # Update work context
        summary = brief.get("summary", "")
        if summary:
            self.memory.work_context = (
                f"{topic}: {summary}\n"
                + self.memory.work_context
            )[:1000]   # Rolling context window
