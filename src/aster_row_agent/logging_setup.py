"""Structured (JSON) logging for observability.

Plain structured logs are sufficient per the project brief — no dashboard,
no log aggregation service. Every log line is a single JSON object so a
reviewer (or a later observability module) can grep/parse it mechanically.

Callers should never pass secrets or forbidden customer fields as `extra`
context; this module does not attempt to redact — redaction is the
responsibility of the caller (e.g. the order sanitizer reuses one allowlist
for both customer responses and trace logs, added in a later phase).
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render each log record as a single line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit structured JSON to stderr.

    Idempotent: safe to call more than once (e.g. once from the CLI entry
    point and once from a test fixture) without duplicating handlers.
    """
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a structured log line with arbitrary key/value context fields."""
    logger.log(level, message, extra={"context": fields})
