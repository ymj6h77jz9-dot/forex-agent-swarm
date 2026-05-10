"""
GMAIL MONITOR — Forex Agentic Swarm
--------------------------------------
Listens for incoming broker alerts, economic calendar digests, and market
news in Gmail. Parses and classifies emails using LLM, then injects
structured signals into the Sentiment Agent.

Triggered by the Base44 Gmail connector automation.
v2: Fully migrated to llm_client (OpenRouter free). No bare openai imports.
    Enhanced with dedup, logging, retry, and Base44 entity logging.
"""

import os
import json
import base64
import re
import logging
import asyncio
from typing import Optional

import httpx

from llm_client import llm_json

logger = logging.getLogger(__name__)

REPORT_EMAIL = os.environ.get("REPORT_EMAIL", "")

EMAIL_CLASSIFIER_PROMPT = """
You are a forex market intelligence classifier. Analyse the subject and body
of an email from a broker, news service, or analyst.

Determine if this email contains actionable forex market signals.

Return ONLY valid JSON:
{
  "is_relevant": true | false,
  "pairs":       ["EURUSD", "XAUUSD"],
  "sentiment":   "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence":  <float 0.0-1.0>,
  "summary":     "<one-sentence summary>",
  "event_type":  "news" | "economic_data" | "broker_alert" | "analyst_report" | "other"
}

Mark is_relevant=true ONLY for specific, actionable market signals.
Marketing emails, account statements, and confirmations → is_relevant=false.
"""

FOREX_PAIRS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY",
    "XAUUSD", "XAGUSD",
}


class GmailMonitor:
    def __init__(self):
        self._processed: set = set()   # dedup by message_id

    # ── Public API ────────────────────────────────────────────────────────────

    async def process_new_emails(
        self,
        gmail_access_token: str,
        message_ids: list,
    ) -> list:
        """
        Called by the Base44 Gmail automation.
        Fetches, classifies, and returns actionable signals.
        """
        signals = []
        for msg_id in message_ids:
            if msg_id in self._processed:
                continue
            email_data = await self._fetch_email(gmail_access_token, msg_id)
            if not email_data:
                continue
            signal = await self._classify_email(email_data)
            if signal and signal.get("is_relevant"):
                # Normalise pairs against known list
                pairs = [p for p in signal.get("pairs", []) if p in FOREX_PAIRS]
                entry = {
                    "email_id":   msg_id,
                    "subject":    email_data.get("subject", ""),
                    "pair":       pairs[0] if pairs else None,
                    "all_pairs":  pairs,
                    "sentiment":  signal.get("sentiment", "NEUTRAL"),
                    "confidence": float(signal.get("confidence", 0.0)),
                    "summary":    signal.get("summary", ""),
                    "event_type": signal.get("event_type", "other"),
                    "direction":  (
                        "BUY"  if signal.get("sentiment") == "BULLISH" else
                        "SELL" if signal.get("sentiment") == "BEARISH" else
                        "FLAT"
                    ),
                }
                signals.append(entry)
                self._processed.add(msg_id)
                logger.info(f"[GmailMonitor] Signal: {entry['summary']}")
        return signals

    # ── Gmail fetch ───────────────────────────────────────────────────────────

    async def _fetch_email(self, access_token: str, message_id: str) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=12.0) as http:
                r = await http.get(
                    f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
                    params={"format": "full"},
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                r.raise_for_status()
                msg = r.json()

            headers = {
                h["name"]: h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }
            subject = headers.get("Subject", "")
            sender  = headers.get("From", "")
            body    = self._extract_body(msg.get("payload", {}))

            return {"subject": subject, "from": sender, "body": body[:3000]}
        except Exception as e:
            logger.warning(f"[GmailMonitor] Fetch {message_id} failed: {e}")
            return None

    def _extract_body(self, payload: dict) -> str:
        mime = payload.get("mimeType", "")
        if mime == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
        for part in payload.get("parts", []):
            result = self._extract_body(part)
            if result:
                return result
        return ""

    # ── LLM classification ────────────────────────────────────────────────────

    async def _classify_email(self, email_data: dict) -> Optional[dict]:
        prompt = (
            f"Subject: {email_data['subject']}\n"
            f"From: {email_data.get('from', '')}\n\n"
            f"Body:\n{email_data['body']}"
        )
        try:
            result = await llm_json(
                messages=[
                    {"role": "system", "content": EMAIL_CLASSIFIER_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            return result if result else None
        except Exception as e:
            logger.warning(f"[GmailMonitor] classify_email failed: {e}")
            return None

    # ── Daily report ──────────────────────────────────────────────────────────

    async def send_daily_report(
        self,
        gmail_access_token: str,
        to_email: str,
        report: dict,
    ) -> None:
        """Send a daily swarm performance report via Gmail."""
        import email.mime.multipart
        import email.mime.text

        subject = f"📊 KRATOS v2 Daily Report — {report.get('date', 'Today')}"

        agent_stats = json.dumps(report.get("agent_stats", {}), indent=2)

        html_body = f"""
<html><body style="font-family:sans-serif;max-width:640px;margin:auto;padding:20px">
<h2 style="color:#1a1a2e">🤖 KRATOS v2 Agentic Swarm Report</h2>
<table border="1" cellpadding="10" cellspacing="0"
       style="border-collapse:collapse;width:100%;font-size:14px">
  <tr style="background:#f0f0f0">
    <td><strong>Total Trades</strong></td>
    <td>{report.get('total_trades', 0)}</td>
  </tr>
  <tr>
    <td><strong>Win Rate</strong></td>
    <td>{report.get('win_rate', 'N/A')}</td>
  </tr>
  <tr style="background:#f0f0f0">
    <td><strong>Total P&amp;L</strong></td>
    <td>{report.get('total_pnl', '$0.00')}</td>
  </tr>
  <tr>
    <td><strong>Best Trade</strong></td>
    <td>{report.get('best_trade', 'N/A')}</td>
  </tr>
  <tr style="background:#f0f0f0">
    <td><strong>Worst Trade</strong></td>
    <td>{report.get('worst_trade', 'N/A')}</td>
  </tr>
  <tr>
    <td><strong>Max Drawdown</strong></td>
    <td>{report.get('max_drawdown', 'N/A')}</td>
  </tr>
  <tr style="background:#f0f0f0">
    <td><strong>Sharpe Ratio</strong></td>
    <td>{report.get('sharpe_ratio', 'N/A')}</td>
  </tr>
</table>
<h3>Agent Accuracy</h3>
<pre style="background:#f8f8f8;padding:12px;border-radius:4px">{agent_stats}</pre>
<p style="color:#888;font-size:11px">
  Generated by KRATOS v2 Forex Agentic Swarm — run mode: {report.get('run_mode','sim')}
</p>
</body></html>
"""
        msg = email.mime.multipart.MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = "me"
        msg["To"]      = to_email
        msg.attach(email.mime.text.MIMEText(html_body, "html"))
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                r = await http.post(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                    headers={
                        "Authorization": f"Bearer {gmail_access_token}",
                        "Content-Type":  "application/json",
                    },
                    json={"raw": raw},
                )
                r.raise_for_status()
                logger.info(f"[GmailMonitor] Daily report sent to {to_email}")
        except Exception as e:
            logger.error(f"[GmailMonitor] send_daily_report failed: {e}")
