"""LLM integration module.

Wraps the provider SDK behind one small interface so the rest of the app never
touches a vendor API directly. Handles:
  * rate limiting (HTTP 429) and transient server errors with exponential
    backoff plus jitter,
  * timeouts,
  * a readable error message instead of a stack trace when the key is wrong or
    the quota is gone,
  * a MockLLM used by the test suite and by local runs with no API key.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings, get_settings


class LLMError(Exception):
    """A failure the user should be told about in plain language."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


@dataclass
class LLMResponse:
    text: str
    attempts: int = 1
    provider: str = "mock"
    model: str = "mock"


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...

    @property
    def label(self) -> str: ...


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "quota")
_TRANSIENT_MARKERS = ("500", "502", "503", "504", "unavailable", "deadline", "timeout")
_AUTH_MARKERS = ("401", "403", "api key", "permission_denied", "unauthenticated")
# Google retires model IDs on a published schedule; a retired or misspelled
# name comes back as 404 NOT_FOUND, which is worth naming explicitly.
_MODEL_MARKERS = ("404", "not_found", "is not found for api version", "not supported for")


def classify_error(exc: Exception) -> tuple[str, bool]:
    """Map a provider exception onto (message, retryable)."""
    text = f"{type(exc).__name__}: {exc}".lower()

    if any(m in text for m in _AUTH_MARKERS):
        return (
            "The LLM rejected the API key. Check GEMINI_API_KEY in your secrets.",
            False,
        )
    if any(m in text for m in _RATE_LIMIT_MARKERS):
        return (
            "The LLM is rate limited right now. Waiting and retrying.",
            True,
        )
    if any(m in text for m in _MODEL_MARKERS):
        return (
            "The LLM does not recognise that model name. Google retires model "
            "IDs over time, so set LLM_MODEL in your secrets to a current one "
            "(see https://aistudio.google.com/docs/models), or run "
            "'python selfcheck.py' to list the models your key can reach.",
            False,
        )
    if any(m in text for m in _TRANSIENT_MARKERS):
        return ("The LLM service is temporarily unavailable. Retrying.", True)
    return (f"The LLM call failed: {exc}", False)


def call_with_backoff(fn, *, max_attempts: int = 4, base_delay: float = 1.0,
                      sleep=time.sleep) -> tuple[str, int]:
    """Run ``fn`` with exponential backoff on retryable failures.

    Returns (result_text, attempts_used). Raises LLMError on give-up.
    """
    last_message = "The LLM call failed."
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), attempt
        except LLMError:
            raise
        except Exception as exc:  # provider SDKs raise many different types
            message, retryable = classify_error(exc)
            last_message = message
            if not retryable or attempt == max_attempts:
                raise LLMError(message, retryable=retryable) from exc
            # Exponential backoff with jitter so repeated 429s spread out.
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.4)
            sleep(min(delay, 16.0))
    raise LLMError(last_message, retryable=True)


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


class GeminiClient:
    """Google Gemini via the google-genai SDK."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.gemini_api_key:
            raise LLMError(
                "No Gemini API key found. Add GEMINI_API_KEY to your secrets "
                "or run in demo mode."
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise LLMError(
                "The google-genai package is not installed. Run "
                "'pip install -r requirements.txt'."
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=self.settings.gemini_api_key)

    @property
    def label(self) -> str:
        return f"Gemini ({self.settings.llm_model})"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        from google.genai import types as genai_types

        def _call() -> str:
            response = self._client.models.generate_content(
                model=self.settings.llm_model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    # Low temperature: we want a machine-readable plan, not prose.
                    temperature=0.1,
                    response_mime_type="application/json",
                    max_output_tokens=2048,
                    http_options=genai_types.HttpOptions(
                        timeout=int(self.settings.llm_timeout_seconds * 1000)
                    ),
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise LLMError(
                    "The model returned an empty response. Try rewording your request."
                )
            return text

        text, attempts = call_with_backoff(
            _call, max_attempts=4, base_delay=1.0
        )
        return LLMResponse(
            text=text,
            attempts=attempts,
            provider="gemini",
            model=self.settings.llm_model,
        )


# --------------------------------------------------------------------------- #
# Mock (tests + no-key demo mode)
# --------------------------------------------------------------------------- #


@dataclass
class MockLLM:
    """Deterministic stand-in for a real provider.

    ``responses`` is consumed in order; the last one repeats once exhausted.
    Set ``raise_times`` to simulate rate limiting before a success.
    """

    responses: list[str] = field(default_factory=list)
    raise_times: int = 0
    error_to_raise: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)
    _raised: int = 0

    @property
    def label(self) -> str:
        return "Mock LLM (demo mode)"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        if self._raised < self.raise_times:
            self._raised += 1
            raise self.error_to_raise or RuntimeError("429 RESOURCE_EXHAUSTED")
        if not self.responses:
            payload = {
                "reasoning": "Demo mode: no LLM key configured.",
                "reply": (
                    "I'm running in demo mode without an LLM key, so I can't "
                    "interpret free-text requests. Add a Gemini API key to enable "
                    "the agent."
                ),
                "actions": [],
            }
            return LLMResponse(text=json.dumps(payload))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return LLMResponse(text=self.responses[index], attempts=1)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_client(settings: Settings | None = None) -> LLMClient:
    """Return a live client when a key is configured, else the mock."""
    settings = settings or get_settings()
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        return GeminiClient(settings)
    return MockLLM()
