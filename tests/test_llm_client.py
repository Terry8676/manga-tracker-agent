"""LLM integration module: error classification and backoff."""

from __future__ import annotations

import pytest

from agent.llm_client import (
    LLMError,
    MockLLM,
    build_client,
    call_with_backoff,
    classify_error,
)


def test_rate_limit_is_retryable():
    message, retryable = classify_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert retryable
    assert "rate limited" in message


def test_quota_message_is_retryable():
    _, retryable = classify_error(RuntimeError("Quota exceeded for this model"))
    assert retryable


def test_server_error_is_retryable():
    _, retryable = classify_error(RuntimeError("503 Service Unavailable"))
    assert retryable


def test_timeout_is_retryable():
    _, retryable = classify_error(TimeoutError("deadline exceeded"))
    assert retryable


def test_bad_key_is_not_retryable_and_names_the_secret():
    message, retryable = classify_error(RuntimeError("403 API key not valid"))
    assert not retryable
    assert "GEMINI_API_KEY" in message


def test_unknown_error_is_not_retryable():
    _, retryable = classify_error(ValueError("something odd"))
    assert not retryable


def test_backoff_retries_then_succeeds():
    attempts = {"n": 0}
    slept: list[float] = []

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("429 rate limit")
        return "finally"

    # base_delay is large relative to the 0-0.4s jitter so the doubling is
    # visible; sleep is mocked, so nothing actually waits.
    result, used = call_with_backoff(
        flaky, max_attempts=5, base_delay=1.0, sleep=slept.append
    )
    assert result == "finally"
    assert used == 3
    # Delays grow between attempts (exponential backoff).
    assert len(slept) == 2 and slept[1] > slept[0]


def test_backoff_gives_up_and_raises_llm_error():
    def always_limited():
        raise RuntimeError("429 rate limit")

    with pytest.raises(LLMError) as err:
        call_with_backoff(always_limited, max_attempts=3, base_delay=0.001, sleep=lambda s: None)
    assert err.value.retryable


def test_backoff_does_not_retry_auth_failure():
    calls = {"n": 0}

    def bad_key():
        calls["n"] += 1
        raise RuntimeError("401 api key invalid")

    with pytest.raises(LLMError):
        call_with_backoff(bad_key, max_attempts=4, base_delay=0.001, sleep=lambda s: None)
    assert calls["n"] == 1  # no pointless retries


def test_mock_llm_returns_responses_in_order():
    mock = MockLLM(responses=['{"actions": []}', '{"actions": [], "reply": "second"}'])
    assert "actions" in mock.complete("sys", "first").text
    assert "second" in mock.complete("sys", "second").text


def test_mock_llm_records_prompts():
    mock = MockLLM(responses=['{"actions": []}'])
    mock.complete("SYSTEM", "USER")
    assert mock.calls[0] == ("SYSTEM", "USER")


def test_mock_llm_can_simulate_rate_limiting():
    mock = MockLLM(responses=['{"actions": []}'], raise_times=1)
    with pytest.raises(RuntimeError):
        mock.complete("s", "u")
    assert mock.complete("s", "u").text  # succeeds on the next call


def test_mock_llm_default_demo_payload():
    mock = MockLLM()
    assert "demo mode" in mock.complete("s", "u").text.lower()


def test_build_client_falls_back_to_mock_without_key(settings):
    client = build_client(settings)
    assert isinstance(client, MockLLM)
    assert "demo" in client.label.lower()


def test_gemini_client_requires_a_key(settings):
    from agent.llm_client import GeminiClient

    with pytest.raises(LLMError) as err:
        GeminiClient(settings)
    assert "GEMINI_API_KEY" in err.value.message


def test_retired_model_gives_an_actionable_message():
    """Google shuts down model IDs on a schedule; a 404 should say so rather
    than surfacing a raw provider error."""
    message, retryable = classify_error(
        RuntimeError(
            "404 NOT_FOUND: models/gemini-2.0-flash is not found for API "
            "version v1beta"
        )
    )
    assert not retryable
    assert "LLM_MODEL" in message
    assert "retires model" in message


def test_model_error_is_not_mistaken_for_a_rate_limit():
    _, retryable = classify_error(RuntimeError("404 model not found"))
    assert not retryable


def test_default_model_is_not_a_retired_one():
    """Guard against the shipped default silently going stale."""
    from agent.config import Settings

    retired = {
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    }
    assert Settings().llm_model not in retired
