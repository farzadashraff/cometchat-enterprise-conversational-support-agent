from __future__ import annotations

import json

import pytest

from aster_row_agent.agent.errors import (
    LLMMalformedResponseError,
    LLMProviderError,
    LLMTimeoutError,
)
from aster_row_agent.agent.llm import (
    DEFAULT_MODEL,
    FakeLLMClient,
    LLMRequest,
    LLMResponse,
    parse_structured_output,
)


def _request(text: str = "hello") -> LLMRequest:
    return LLMRequest(model=DEFAULT_MODEL, system="system instructions", user_content=text)


# --- FakeLLMClient ---------------------------------------------------------------------


def test_fake_client_returns_scripted_response() -> None:
    client = FakeLLMClient.with_answer("Hello!")
    response = client.generate(_request())
    output = parse_structured_output(response)
    assert output.answer == "Hello!"


def test_fake_client_returns_responses_in_order() -> None:
    client = FakeLLMClient(
        [
            LLMResponse(raw_text=json.dumps({"answer": "first"}), model=DEFAULT_MODEL),
            LLMResponse(raw_text=json.dumps({"answer": "second"}), model=DEFAULT_MODEL),
        ]
    )
    assert parse_structured_output(client.generate(_request())).answer == "first"
    assert parse_structured_output(client.generate(_request())).answer == "second"


def test_fake_client_raises_scripted_timeout() -> None:
    client = FakeLLMClient([LLMTimeoutError("simulated timeout")])
    with pytest.raises(LLMTimeoutError):
        client.generate(_request())


def test_fake_client_raises_scripted_provider_error() -> None:
    client = FakeLLMClient([LLMProviderError("simulated 500")])
    with pytest.raises(LLMProviderError):
        client.generate(_request())


def test_fake_client_records_requests() -> None:
    client = FakeLLMClient.with_answer("ok")
    client.generate(_request("what is the return policy?"))
    assert len(client.requests) == 1
    assert client.requests[0].user_content == "what is the return policy?"


def test_fake_client_raises_when_exhausted() -> None:
    client = FakeLLMClient([])
    with pytest.raises(AssertionError):
        client.generate(_request())


# --- parse_structured_output -----------------------------------------------------------


def test_parse_structured_output_success() -> None:
    response = LLMResponse(
        raw_text=json.dumps({"answer": "The window is 30 days.", "cited_filenames": ["01.md"]}),
        model=DEFAULT_MODEL,
    )
    output = parse_structured_output(response)
    assert output.answer == "The window is 30 days."
    assert output.cited_filenames == ("01.md",)


def test_parse_structured_output_defaults_cited_filenames_to_empty() -> None:
    response = LLMResponse(raw_text=json.dumps({"answer": "ok"}), model=DEFAULT_MODEL)
    output = parse_structured_output(response)
    assert output.cited_filenames == ()


def test_parse_structured_output_tolerates_markdown_code_fence() -> None:
    payload = json.dumps({"answer": "ok", "cited_filenames": []})
    response = LLMResponse(raw_text=f"```json\n{payload}\n```", model=DEFAULT_MODEL)
    output = parse_structured_output(response)
    assert output.answer == "ok"


def test_parse_structured_output_raises_on_invalid_json() -> None:
    response = LLMResponse(raw_text="not json at all", model=DEFAULT_MODEL)
    with pytest.raises(LLMMalformedResponseError):
        parse_structured_output(response)


def test_parse_structured_output_raises_on_missing_required_field() -> None:
    response = LLMResponse(raw_text=json.dumps({"cited_filenames": []}), model=DEFAULT_MODEL)
    with pytest.raises(LLMMalformedResponseError):
        parse_structured_output(response)


def test_parse_structured_output_raises_on_non_object_json() -> None:
    response = LLMResponse(raw_text=json.dumps(["not", "an", "object"]), model=DEFAULT_MODEL)
    with pytest.raises(LLMMalformedResponseError):
        parse_structured_output(response)


def test_parse_structured_output_malformed_error_does_not_include_raw_text_length_only() -> None:
    """The malformed-response error must not embed the full raw text (which
    could be arbitrarily large or, in a compromised scenario, sensitive) —
    only its length, for diagnosis."""
    huge_text = "x" * 10_000
    response = LLMResponse(raw_text=huge_text, model=DEFAULT_MODEL)
    with pytest.raises(LLMMalformedResponseError) as exc_info:
        parse_structured_output(response)
    assert huge_text not in str(exc_info.value)
    assert "10000" in str(exc_info.value)


# --- LLMRequest / LLMResponse are plain, provider-independent models -------------------


def test_llm_request_is_frozen() -> None:
    from pydantic import ValidationError

    request = _request()
    with pytest.raises(ValidationError):
        request.model = "different-model"


def test_llm_request_carries_system_and_user_content_separately() -> None:
    request = LLMRequest(model=DEFAULT_MODEL, system="TRUSTED", user_content="untrusted stuff")
    assert request.system == "TRUSTED"
    assert request.user_content == "untrusted stuff"
    assert "TRUSTED" not in request.user_content
