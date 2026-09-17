"""A small, typed, provider-independent LLM client abstraction.

Deliberately minimal — this is not an agent framework. `LLMClient` is a
`Protocol` with one method; the real implementation
(`AnthropicLLMClient`) is a thin wrapper around one provider's SDK, and
`FakeLLMClient` is a fully deterministic test double that never touches
the network.

The LLM's structured output intentionally does NOT include a
handoff/disposition decision: only `answer` (the phrased text) and
`cited_filenames` (which sources it believes it used) are read from the
model. Whether the turn is actually a handoff, a refusal, or an error is
decided entirely by deterministic code (`handoff.py`,
`validation.py`) from `EvidenceBundle`/`OrderLookupResult`/`RoutingDecision`
— the LLM has no channel through which it could override that decision,
by construction, not by convention.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from aster_row_agent.agent.errors import (
    LLMMalformedResponseError,
    LLMProviderError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMRequest(BaseModel):
    """Everything one call to the LLM needs — already fully assembled by
    `prompts.py`. The client itself does no prompt construction."""

    model_config = ConfigDict(frozen=True)

    model: str
    system: str
    """The trusted application-instructions block, passed via the
    provider's dedicated system-prompt channel — structurally separate
    from `user_content` below, never concatenated with untrusted data."""
    user_content: str
    """The single untrusted-data block: user message, conversation
    history, retrieved evidence, and order result, each clearly
    delimited and labeled as data (see `prompts.py`)."""
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


class LLMResponse(BaseModel):
    """The raw text returned by the provider. Parsing into structured
    `answer`/`cited_filenames` happens in `orchestration.py`, not here —
    this keeps `LLMClient` provider-agnostic and trivially fakeable."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    model: str


class LLMClient(Protocol):
    """The only interface `orchestration.py` depends on."""

    def generate(self, request: LLMRequest) -> LLMResponse: ...


class StructuredLLMOutput(BaseModel):
    """The expected shape of the model's JSON response."""

    model_config = ConfigDict(frozen=True)

    answer: str
    cited_filenames: tuple[str, ...] = ()


def parse_structured_output(response: LLMResponse) -> StructuredLLMOutput:
    """Parse the model's raw text as the expected JSON contract.

    Raises `LLMMalformedResponseError` (never a bare `json.JSONDecodeError`
    or `pydantic.ValidationError`) so `orchestration.py` has one exception
    type to catch and fall back from.
    """
    text = response.raw_text.strip()
    # Models occasionally wrap JSON in a markdown code fence despite
    # instructions not to — tolerate that one harmless formatting quirk,
    # nothing else.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        payload = json.loads(text)
        return StructuredLLMOutput.model_validate(payload)
    except Exception as exc:
        raise LLMMalformedResponseError(response.raw_text) from exc


class AnthropicLLMClient:
    """The real provider client — a thin wrapper over the `anthropic` SDK.

    Imported lazily (inside `__init__`, mirroring
    `rag.embeddings.FastEmbedProvider`'s pattern for its optional heavy
    dependency) so that nothing in this module requires the `anthropic`
    package, an API key, or network access merely to be imported — only
    to be *constructed* for real use. Every test in this project uses
    `FakeLLMClient` instead.
    """

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic_module = anthropic
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        import anthropic

        try:
            message = self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user_content}],
                timeout=request.timeout_seconds,
            )
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except anthropic.APIError as exc:
            raise LLMProviderError(str(exc)) from exc

        text_blocks = [block.text for block in message.content if block.type == "text"]
        return LLMResponse(raw_text="".join(text_blocks), model=request.model)


class FakeLLMClient:
    """A fully deterministic, network-free test double.

    Configured with a queue of canned `LLMResponse`s (or exceptions to
    raise) so tests can script exact model behavior, including malicious
    or malformed output, without ever depending on a real provider.
    """

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeLLMClient has no more scripted responses")
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    @classmethod
    def with_answer(cls, answer: str, cited_filenames: tuple[str, ...] = ()) -> FakeLLMClient:
        """Convenience constructor for the common case of one canned,
        well-formed structured answer."""
        payload = json.dumps({"answer": answer, "cited_filenames": list(cited_filenames)})
        return cls([LLMResponse(raw_text=payload, model=DEFAULT_MODEL)])
