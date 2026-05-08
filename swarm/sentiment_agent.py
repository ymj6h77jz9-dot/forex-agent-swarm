"""
SENTIMENT AGENT — Forex Agentic Swarm
----------------------------------------
News & sentiment specialist. Scrapes financial news, economic calendar events,
and social signals. Uses LLM to classify market sentiment and bias direction.
Integrates Gmail monitoring for broker alerts and news digests.
"""

import asyncio
import os
import json
import httpx
from openai import AsyncOpenAI
from datetime import datetime, timezone

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

SENTIMENT_SYSTEM_PROMPT = """
You are a forex market sentiment analyst operating inside an autonomous trading swarm.

Your job:
- Analyze news headlines, economic events, and any provided email alerts
- Determine the net sentiment impact on the given currency pair
- Return ONLY a JSON object in this exact format:
  {
    "signal": "BUY" | "SELL" | "FLAT",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "<concise explanation referencing specific news/events>",
    "key_drivers": ["<driver1>", "<driver2>"]
  }

Rules:
- High-impact news (NFP, CPI, FOMC, ECB) should yield confidence 0.7+
- Contradictory signals across sources → return FLAT with low confidence
- If no relevant news found, return FLAT with confidence 0.0
"""


class SentimentAgent:
    def __init__(self):
        self.name = "sentiment"
        self.gmail_signals = []  # Populated by Gmail monitor automation

    async def analyze(self, market_state) -> "AgentVote":
        from orchestrator_agent import AgentVote

        # 1. Fetch latest news for the pair's base currency
        headlines = await self._fetch_news(market_state.pair)

        # 2. Include any Gmail-sourced signals
        email_signals = self._get_email_signals(market_state.pair)

        prompt = f"""
Currency Pair: {market_state.pair}
Timestamp: {market_state.timestamp}
Session: {market_state.session}

Recent News Headlines:
{json.dumps(headlines, indent=2)}

Email Broker Alerts:
{json.dumps(email_signals, indent=2)}

Analyze sentiment and return your structured vote.
"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            data = json.loads(response.choices[0].message.content)
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
        """Fetch news headlines from NewsAPI for the pair's base currency."""
        base_currency = pair[:3]  # e.g. "EUR" from "EURUSD"
        currency_terms = {
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
        query = currency_terms.get(base_currency, base_currency)

        if not NEWS_API_KEY:
            return [{"headline": "NewsAPI key not configured", "source": "system"}]

        try:
            async with httpx.AsyncClient() as http:
                r = await http.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q":        query,
                        "sortBy":   "publishedAt",
                        "pageSize": 10,
                        "language": "en",
                        "apiKey":   NEWS_API_KEY,
                    },
                    timeout=8.0,
                )
                r.raise_for_status()
                articles = r.json().get("articles", [])
                return [
                    {"headline": a["title"], "source": a["source"]["name"]}
                    for a in articles[:10]
                ]
        except Exception as e:
            return [{"headline": f"News fetch error: {e}", "source": "system"}]

    def _get_email_signals(self, pair: str) -> list:
        """Return any Gmail-sourced signals relevant to this pair."""
        return [s for s in self.gmail_signals if s.get("pair") == pair]

    def ingest_email_signal(self, signal: dict):
        """Called by the Gmail automation when a relevant email arrives."""
        self.gmail_signals.append(signal)
        # Keep only last 50 signals in memory
        if len(self.gmail_signals) > 50:
            self.gmail_signals = self.gmail_signals[-50:]
