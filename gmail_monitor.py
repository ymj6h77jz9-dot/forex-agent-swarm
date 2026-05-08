"""
GMAIL MONITOR — Forex Agentic Swarm
--------------------------------------
Listens for incoming broker alerts, economic calendar digests, and market
news in Gmail. Parses and classifies emails using LLM, then injects
structured signals into the Sentiment Agent.

This module is triggered by the Base44 Gmail connector automation.
"""

import os
import json
import base64
import re
from openai import AsyncOpenAI
import httpx

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMAIL_CLASSIFIER_PROMPT = """
You are a forex market intelligence classifier. You receive the subject and body
of an email from a broker, news service, or analyst.

Your job: determine if this email contains actionable forex market signals.

Return ONLY a JSON object:
{
  "is_relevant": true | false,
  "pairs": ["EURUSD", "XAUUSD"],          // affected currency pairs (empty if none)
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0.0 to 1.0>,
  "summary": "<one-sentence summary>",
  "event_type": "news" | "economic_data" | "broker_alert" | "analyst_report" | "other"
}

Only mark is_relevant=true if the email contains specific, actionable market information.
Marketing emails, account updates, and confirmations should be is_relevant=false.
"""

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY",
    "XAUUSD", "XAGUSD",
]


class GmailMonitor:
    def __init__(self):
        self.processed_ids = set()

    async def process_new_emails(self, gmail_access_token: str, message_ids: list) -> list:
        """
        Called by the Base44 Gmail automation.
        Fetches, classifies, and returns actionable signals.
        """
        signals = []

        for msg_id in message_ids:
            if msg_id in self.processed_ids:
                continue

            email_data = await self._fetch_email(gmail_access_token, msg_id)
            if not email_data:
                continue

            signal = await self._classify_email(email_data)
            if signal and signal.get("is_relevant"):
                signals.append({
                    "email_id":   msg_id,
                    "pair":       signal["pairs"][0] if signal["pairs"] else None,
                    "all_pairs":  signal["pairs"],
                    "sentiment":  signal["sentiment"],
                    "confidence": signal["confidence"],
                    "summary":    signal["summary"],
                    "event_type": signal["event_type"],
                    "direction":  "BUY" if signal["sentiment"] == "BULLISH" else
                                  "SELL" if signal["sentiment"] == "BEARISH" else "FLAT",
                })
                self.processed_ids.add(msg_id)
                print(f"[GMAIL] Signal extracted: {signal['summary']}")

        return signals

    async def _fetch_email(self, access_token: str, message_id: str) -> dict | None:
        """Fetch a Gmail message and extract subject + body."""
        try:
            async with httpx.AsyncClient() as http:
                r = await http.get(
                    f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
                    params={"format": "full"},
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0,
                )
                r.raise_for_status()
                msg = r.json()

            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject", "")
            body    = self._extract_body(msg.get("payload", {}))

            return {"subject": subject, "body": body[:3000]}  # cap at 3k chars
        except Exception as e:
            print(f"[GMAIL] Failed to fetch {message_id}: {e}")
            return None

    def _extract_body(self, payload: dict) -> str:
        """Recursively extract plain text from Gmail message payload."""
        mime_type = payload.get("mimeType", "")

        if mime_type == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")

        for part in payload.get("parts", []):
            result = self._extract_body(part)
            if result:
                return result

        return ""

    async def _classify_email(self, email_data: dict) -> dict | None:
        """Use LLM to classify the email and extract trading signals."""
        try:
            prompt = f"""
Subject: {email_data['subject']}

Body:
{email_data['body']}
"""
            response = await client.chat.completions.create(
                model="gpt-4o-mini",        # cheaper model for classification
                messages=[
                    {"role": "system", "content": EMAIL_CLASSIFIER_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[GMAIL] Classification error: {e}")
            return None

    async def send_daily_report(self, gmail_access_token: str, to_email: str, report: dict):
        """Send a daily swarm performance report via Gmail."""
        import email.mime.multipart
        import email.mime.text

        subject = f"📊 Forex Swarm Daily Report — {report.get('date', 'Today')}"

        html_body = f"""
<html><body style="font-family: sans-serif; max-width: 600px; margin: auto;">
<h2>🤖 Agentic Swarm Report</h2>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%">
  <tr><td><strong>Total Trades</strong></td><td>{report.get('total_trades', 0)}</td></tr>
  <tr><td><strong>Win Rate</strong></td><td>{report.get('win_rate', 'N/A')}</td></tr>
  <tr><td><strong>Total P&L</strong></td><td>{report.get('total_pnl', '$0.00')}</td></tr>
  <tr><td><strong>Best Trade</strong></td><td>{report.get('best_trade', 'N/A')}</td></tr>
  <tr><td><strong>Worst Trade</strong></td><td>{report.get('worst_trade', 'N/A')}</td></tr>
</table>
<h3>Agent Performance</h3>
<pre>{json.dumps(report.get('agent_stats', {}), indent=2)}</pre>
<p style="color:#888;font-size:12px">Generated by your Forex Agentic Swarm</p>
</body></html>
"""

        msg = email.mime.multipart.MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = "me"
        msg["To"]      = to_email
        msg.attach(email.mime.text.MIMEText(html_body, "html"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                    headers={
                        "Authorization": f"Bearer {gmail_access_token}",
                        "Content-Type":  "application/json",
                    },
                    json={"raw": raw},
                    timeout=15.0,
                )
                r.raise_for_status()
                print(f"[GMAIL] Daily report sent to {to_email}")
        except Exception as e:
            print(f"[GMAIL] Failed to send report: {e}")
