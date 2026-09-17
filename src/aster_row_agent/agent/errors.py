"""Exceptions raised by the agent layer.

Follows the same pattern as `rag/errors.py` and `orders/errors.py`: every
failure is a specific, named subclass with enough context to diagnose it.
Orchestration catches only these specific types (never a bare
``except Exception``) and converts them into a safe, deterministic
customer-facing fallback — see `orchestration.py`.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent-layer failures."""


class LLMError(AgentError):
    """Base class for LLM-client failures."""


class LLMTimeoutError(LLMError):
    """The LLM provider did not respond within the configured timeout."""


class LLMProviderError(LLMError):
    """The LLM provider returned an error (rate limit, auth, 5xx, ...)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class LLMMalformedResponseError(LLMError):
    """The LLM's response could not be parsed into the expected structured output."""

    def __init__(self, raw_text: str) -> None:
        self.raw_text_length = len(raw_text)
        super().__init__(
            f"LLM response could not be parsed as structured output "
            f"({self.raw_text_length} chars received)"
        )
