"""
LLM CLIENT — KRATOS v2
=======================
Unified LLM gateway for all swarm agents.

Priority chain:
  1. OpenRouter (FREE) — openai/gpt-oss-120b:free  ← PRIMARY
  2. OpenRouter (FREE) — nvidia/nemotron-3-super-120b-a12b:free  ← FALLBACK
  3. OpenAI direct (sk-...) — gpt-4o-mini  ← if OPENAI_API_KEY set
  4. OpenAI direct — gpt-4o  ← high-stakes calls only

Every agent calls: await llm(messages, json_mode=True)
No agent imports openai directly — everything goes through here.

Free model tested live:
  openai/gpt-oss-120b:free  — 131K ctx, JSON-mode compliant, 120B params
  nvidia/nemotron-3-super-120b-a12b:free — 262K ctx, 120B params
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any

import httpx

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY   = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL  = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_PRIMARY   = os.environ.get("OPENROUTER_PRIMARY_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_FALLBACK  = os.environ.get("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENAI_API_KEY       = os.environ.get("OPENAI_API_KEY", "")

OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type":  "application/json",
    "HTTP-Referer":  "https://kratos-swarm.ai",
    "X-Title":       "KRATOS v2 Forex Swarm",
}

# ── Main Gateway ──────────────────────────────────────────────────────────────

async def llm(
    messages:    List[Dict[str, str]],
    json_mode:   bool = False,
    temperature: float = 0.3,
    max_tokens:  int = 800,
    model:       Optional[str] = None,       # override specific model
    high_stakes: bool = False,               # True → use GPT-4o if key available
) -> str:
    """
    Unified LLM call. Tries OpenRouter free models first, falls back to OpenAI.
    
    Returns raw string content. Caller is responsible for JSON parsing.
    
    Args:
        messages:    Chat messages list [{"role":..,"content":..}]
        json_mode:   Request JSON output (adds instruction to system prompt)
        temperature: Sampling temperature
        max_tokens:  Max output tokens
        model:       Force specific model ID
        high_stakes: Use paid GPT-4o if available (for critical decisions)
    
    Returns:
        str: Model response content
    """
    # Inject JSON instruction if needed (OpenRouter free models don't all support response_format)
    if json_mode:
        messages = _inject_json_instruction(messages)

    # 1. Try OpenRouter free models
    if OPENROUTER_API_KEY:
        models_to_try = [model] if model else [OPENROUTER_PRIMARY, OPENROUTER_FALLBACK]
        for m in models_to_try:
            try:
                result = await _openrouter_call(m, messages, temperature, max_tokens)
                if result:
                    logger.debug(f"[LLM] {m} → {len(result)} chars")
                    return result
            except Exception as e:
                logger.warning(f"[LLM] {m} failed: {e}")
                await asyncio.sleep(0.5)

    # 2. Fallback to OpenAI direct
    if OPENAI_API_KEY:
        openai_model = "gpt-4o" if high_stakes else "gpt-4o-mini"
        try:
            result = await _openai_call(openai_model, messages, temperature, max_tokens, json_mode)
            if result:
                logger.debug(f"[LLM] OpenAI/{openai_model} → {len(result)} chars")
                return result
        except Exception as e:
            logger.error(f"[LLM] OpenAI fallback failed: {e}")

    raise RuntimeError(
        "All LLM providers failed. Check OPENROUTER_API_KEY or OPENAI_API_KEY."
    )


async def llm_json(
    messages:    List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens:  int = 600,
    model:       Optional[str] = None,
    high_stakes: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper — always returns parsed dict.
    Retries up to 2 times if JSON parsing fails.
    """
    for attempt in range(3):
        raw = await llm(messages, json_mode=True, temperature=temperature,
                        max_tokens=max_tokens, model=model, high_stakes=high_stakes)
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            # Try to extract JSON object from response
            import re
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            if attempt < 2:
                logger.warning(f"[LLM] JSON parse failed (attempt {attempt+1}), retrying...")
                await asyncio.sleep(0.3)
            else:
                logger.error(f"[LLM] Could not parse JSON after 3 attempts. Raw: {raw[:200]}")
                return {}
    return {}


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _openrouter_call(model: str, messages: List[Dict],
                            temperature: float, max_tokens: int) -> str:
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=OPENROUTER_HEADERS,
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError(f"Model {model} returned null content")
        return content


async def _openai_call(model: str, messages: List[Dict],
                        temperature: float, max_tokens: int, json_mode: bool) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    kwargs: Dict[str, Any] = dict(
        model       = model,
        messages    = messages,
        temperature = temperature,
        max_tokens  = max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _inject_json_instruction(messages: List[Dict]) -> List[Dict]:
    """Add JSON output instruction to system prompt."""
    result = list(messages)
    json_note = "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no code fences, no extra text."
    if result and result[0]["role"] == "system":
        result[0] = {**result[0], "content": result[0]["content"] + json_note}
    else:
        result.insert(0, {"role": "system", "content": "Respond with valid JSON only." + json_note})
    return result


# ── Model info ────────────────────────────────────────────────────────────────

def get_active_provider() -> str:
    """Returns which provider is active."""
    if OPENROUTER_API_KEY:
        return f"OpenRouter FREE ({OPENROUTER_PRIMARY})"
    if OPENAI_API_KEY:
        return "OpenAI direct (gpt-4o-mini)"
    return "No LLM configured"


async def list_free_models() -> List[str]:
    """List all free models available on OpenRouter."""
    if not OPENROUTER_API_KEY:
        return []
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        )
        data = r.json()
        return [
            m["id"] for m in data.get("data", [])
            if float(m.get("pricing", {}).get("prompt", "1")) == 0
        ]
