"""
SENTIMENT AGENT — Forex Agentic Swarm
----------------------------------------
News & sentiment specialist. Scrapes financial news, economic calendar events,
and social signals. Uses LLM to classify market sentiment and bias direction.
Integrates Gmail monitoring for broker alerts and news digests.

v2: Uses llm_client (OpenRouter free). Accepts _macro_context (Crucix)
    and _research_context (DeerFlow) injected by the orchestrator.
"""

import os
import json
import httpx
from orchestrator_agent import AgentVote
from llm_client import llm_json

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

SENTIMENT_SYSTEM_PROMPT = """
You are a forex market sentiment analyst operating inside an autonomous trading swarm.

Your job:
- Analyze news headlines, economic events, macro signals, and broker email alerts
- Determine the net sentiment impact on the given currency pair
- Return ONLY a JSON object in this exact format:
  {
    "signal": "BUY" | "SELL" | "FLAT",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "<concise explanation referencing specific news/events>",
    "key_drivers": ["<driver1>", "<driver2>"]
  }

Rules:
- High-impact news (NFP, CPI, FOMC, ECB) → confidence 0.7+
- Crucix RISK_OFF macro regime → lower confidence for aggressive signals
- DeerFlow research aligned with signal → increase confidence
- Contradictory signals → return FLAT with low confidence
- No relevant news → return FLAT with confidence 0.0
"""

CURRENCY_TERMS = {
    "EUR": "Euro ECB eurozone",
    "USD": "dollar Federal Reserve Fed",
    "GBP": "pound sterling Bank of England",
    "JPY": "yen Bank of Japan BOJ",
    "AUD": "Australian dollar RBA",
    "NZD": "New Zealand dollar RBNZ",
    "CAD": "Canadian dollar Bank of Canada",
    "CHF": "Swiss franc SNB",
    "XAU": "gold commodity",
}


class SentimentAgent:
    def __init__(self):
        self.name = "sentiment"
        self.gmail_signals = []

        # Injected by KratosOrchestratorV2 before analyze() is called
        self._macro_context:    str = ""   # Crucix intelligence briefing
        self._research_context: str = ""   # DeerFlow research summary

    async def analyze(self, market_state) -> AgentVote:
        headlines     = await self._fetch_news(market_state.pair)
        email_signals = self._get_email_signals(market_state.pair)

        macro_section = (
            f"\nCRUCIX MACRO INTELLIGENCE:\n{self._macro_context[:800]}\n"
            if self._macro_context else ""
        )
        research_section = (
            f"\nDEERFLOW RESEARCH BRIEF:\n{self._research_context[:400]}\n"
            if self._research_context else ""
        )

        prompt = f"""
Currency Pair: {market_state.pair}
Timestamp:     {market_state.timestamp}
Session:       {market_state.session}

Recent News Headlines:
{json.dumps(headlines, indent=2)}

Email Broker Alerts:
{json.dumps(email_signals, indent=2)}
{macro_section}{research_section}
Analyze all signals. Return your structured vote JSON.
"""
        try:
            data = await llm_json(
                messages=[
                    {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return AgentVote(
                agent_name = self.name,
                signal     = data.get("signal", "FLAT"),
                confidence = float(data.get("confidence", 0.0)),
                reasoning  = data.get("reasoning", ""),
                pair       = market_state.pair,
            )
        except Exception as e:
            return AgentVote(self.name, "FLAT", 0.0, f"Error: {e}", market_state.pair)

    async def _fetch_news(self, pair: str) -> list:
        base_currency = pair[:3]
        query = CURRENCY_TERMS.get(base_currency, base_currency)
        if not NEWS_API_KEY:
            return [{"headline": "NewsAPI key not configured", "source": "system"}]
        try:
            async with httpx.AsyncClient(timeout=8.0) as http:
                r = await http.get(
                    "https://newsapi.org/v2/everything",
                    params={"q": query, "sortBy": "publishedAt",
                            "pageSize": 10, "language": "en", "apiKey": NEWS_API_KEY},
                )
                r.raise_for_status()
                articles = r.json().get("articles", [])
                return [{"headline": a["title"], "source": a["source"]["name"]}
                        for a in articles[:10]]
        except Exception as e:
            return [{"headline": f"News fetch error: {e}", "source": "system"}]

    def _get_email_signals(self, pair: str) -> list:
        return [s for s in self.gmail_signals if s.get("pair") == pair]

    def ingest_email_signal(self, signal: dict):
        self.gmail_signals.append(signal)
        if len(self.gmail_signals) > 50:
            self.gmail_signals = self.gmail_signals[-50:]
