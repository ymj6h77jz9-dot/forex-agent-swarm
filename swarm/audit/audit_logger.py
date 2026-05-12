"""
AUDIT LOGGER — KRATOS v2
==========================
Implements R20: Immutable Audit Trail

Every decision, trade, weight update, circuit breaker event, and
evolution proposal is written to an append-only JSONL audit log.
Entries are SHA-256 hashed for tamper detection.

The audit trail is the single source of truth for:
  - Compliance review
  - Post-trade analysis
  - Evolution validation (R20)
  - Debugging and replay (R12)

Design principles:
  - Append-only — no record is ever overwritten or deleted
  - Each entry is independently hash-verifiable
  - Rolling daily files — easy archival
  - Sync write — no buffering, no data loss

Usage:
    logger = AuditLogger()
    logger.log_decision(cycle_id, pair, direction, confidence, score)
    logger.log_trade(trade_dict)
    logger.log_evolution(proposal_id, approved, reason)
    report = logger.verify_integrity()
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

AUDIT_DIR  = Path(__file__).parent.parent / "data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


class AuditEventType(str, Enum):
    DECISION         = "decision"
    TRADE_OPEN       = "trade_open"
    TRADE_CLOSE      = "trade_close"
    WEIGHT_UPDATE    = "weight_update"
    CIRCUIT_BREAKER  = "circuit_breaker"
    EVOLUTION        = "evolution_proposal"
    RISK_VETO        = "risk_veto"
    ANOMALY          = "anomaly"
    SYSTEM_START     = "system_start"
    SYSTEM_STOP      = "system_stop"


class AuditLogger:
    """
    R20: Immutable, append-only audit trail for KRATOS v2.
    Thread-safe for single-process async use.
    """

    def __init__(self, audit_dir: Optional[Path] = None):
        self._dir      = audit_dir or AUDIT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._today    = self._today_str()
        _log.info(f"[AuditLogger] Initialised — writing to {self._dir}")
        self.log_event(AuditEventType.SYSTEM_START, {"message": "KRATOS v2 audit trail started"})

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience methods
    # ─────────────────────────────────────────────────────────────────────────

    def log_decision(
        self,
        cycle_id:   str,
        pair:       str,
        direction:  str,
        confidence: float,
        score:      float,
        regime:     str = "",
        weights:    Optional[Dict] = None,
        agent_votes: Optional[List] = None,
    ):
        self.log_event(AuditEventType.DECISION, {
            "cycle_id":   cycle_id,
            "pair":       pair,
            "direction":  direction,
            "confidence": round(confidence, 4),
            "score":      round(score, 4),
            "regime":     regime,
            "weights":    weights or {},
            "agent_votes": agent_votes or [],
        })

    def log_trade_open(self, trade: Dict):
        self.log_event(AuditEventType.TRADE_OPEN, trade)

    def log_trade_close(self, trade_id: str, pnl: float, pair: str, direction: str):
        self.log_event(AuditEventType.TRADE_CLOSE, {
            "trade_id":  trade_id,
            "pnl":       round(pnl, 4),
            "pair":      pair,
            "direction": direction,
        })

    def log_weight_update(self, old_weights: Dict, new_weights: Dict, reason: str = ""):
        self.log_event(AuditEventType.WEIGHT_UPDATE, {
            "old": old_weights,
            "new": new_weights,
            "reason": reason,
        })

    def log_circuit_breaker(self, reason: str, fired: bool):
        self.log_event(AuditEventType.CIRCUIT_BREAKER, {
            "fired":  fired,
            "reason": reason,
        })

    def log_evolution(self, proposal_id: str, approved: bool, reason: str, weights: Dict = None):
        self.log_event(AuditEventType.EVOLUTION, {
            "proposal_id": proposal_id,
            "approved":    approved,
            "reason":      reason,
            "weights":     weights or {},
        })

    def log_risk_veto(self, pair: str, direction: str, rejections: List[str]):
        self.log_event(AuditEventType.RISK_VETO, {
            "pair":       pair,
            "direction":  direction,
            "rejections": rejections,
        })

    def log_anomaly(self, anomaly_type: str, severity: str, detail: str, value: float = 0.0):
        self.log_event(AuditEventType.ANOMALY, {
            "anomaly_type": anomaly_type,
            "severity":     severity,
            "detail":       detail,
            "value":        value,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Core write
    # ─────────────────────────────────────────────────────────────────────────

    def log_event(self, event_type: AuditEventType, data: Dict[str, Any]):
        """
        Append a single event to the daily audit log.
        Each entry includes:
          - sequence number
          - ISO timestamp (UTC)
          - event type
          - data payload
          - SHA-256 hash of (seq + ts + type + data)
        """
        self._sequence += 1
        today = self._today_str()
        if today != self._today:
            self._today = today  # Roll to new daily file

        ts = datetime.now(timezone.utc).isoformat()
        entry: Dict[str, Any] = {
            "seq":   self._sequence,
            "ts":    ts,
            "type":  event_type.value,
            "data":  data,
        }

        # Hash for tamper detection
        hash_payload = json.dumps({
            "seq":  entry["seq"],
            "ts":   entry["ts"],
            "type": entry["type"],
            "data": entry["data"],
        }, sort_keys=True, default=str).encode()
        entry["hash"] = hashlib.sha256(hash_payload).hexdigest()

        path = self._daily_path(today)
        try:
            with open(path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            _log.error(f"[AuditLogger] Write failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Read & verification
    # ─────────────────────────────────────────────────────────────────────────

    def read_today(self, last_n: int = 100) -> List[Dict]:
        """Read the most recent N entries from today's log."""
        return self._read_file(self._daily_path(self._today_str()), last_n)

    def read_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Read all entries between start_date and end_date (YYYY-MM-DD inclusive).
        """
        from datetime import date, timedelta
        result = []
        d = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        while d <= end:
            result.extend(self._read_file(self._daily_path(str(d))))
            d += timedelta(days=1)
        return result

    def verify_integrity(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify hash integrity of audit log for a given date (default: today).
        Returns {ok, total, corrupted: list_of_seq}.
        """
        target  = date_str or self._today_str()
        entries = self._read_file(self._daily_path(target))
        corrupted = []
        for e in entries:
            stored_hash = e.get("hash", "")
            check_payload = json.dumps({
                "seq":  e["seq"],
                "ts":   e["ts"],
                "type": e["type"],
                "data": e["data"],
            }, sort_keys=True, default=str).encode()
            computed = hashlib.sha256(check_payload).hexdigest()
            if computed != stored_hash:
                corrupted.append(e["seq"])
        return {
            "date":      target,
            "ok":        len(corrupted) == 0,
            "total":     len(entries),
            "corrupted": corrupted,
        }

    def get_summary(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Return event-type summary for a given day."""
        entries = self._read_file(self._daily_path(date_str or self._today_str()))
        summary: Dict[str, int] = {}
        for e in entries:
            t = e.get("type", "unknown")
            summary[t] = summary.get(t, 0) + 1
        return {
            "date":   date_str or self._today_str(),
            "total":  len(entries),
            "by_type": summary,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _daily_path(self, date_str: str) -> Path:
        return self._dir / f"audit_{date_str}.jsonl"

    @staticmethod
    def _today_str() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _read_file(path: Path, last_n: Optional[int] = None) -> List[Dict]:
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            entries = []
            for ln in lines:
                if ln.strip():
                    try:
                        entries.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
            return entries[-last_n:] if last_n else entries
        except Exception as e:
            _log.warning(f"[AuditLogger] Read failed {path}: {e}")
            return []
