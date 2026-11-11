"""Configuration loaded from Streamlit secrets or environment variables.

Secrets are never hard-coded. Locally the app falls back to a SQLite file and
a mock LLM so it runs with zero setup; in deployment both values come from
Streamlit's secrets manager.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def secrets_status() -> dict:
    """Report what the app can actually see in Streamlit secrets.

    Returns ``{"available": bool, "error": str | None, "keys": [str]}``.

    Only key NAMES are returned, never values, so the result is safe to show
    in the UI. This exists because swallowing the exception from
    ``st.secrets`` made three very different problems - no secrets file, a
    secrets file that fails to parse, and a file missing one key - produce
    one indistinguishable "no key set" message, which is impossible to debug
    on a deployed app.
    """
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover - streamlit is a dependency
        return {"available": False, "error": f"streamlit unavailable: {exc}", "keys": []}

    try:
        keys = sorted(st.secrets.keys())
    except Exception as exc:
        name = type(exc).__name__
        # "No secrets file at all" is normal when running locally; anything
        # else means the file exists but could not be read.
        if "SecretNotFound" in name or "FileNotFound" in name:
            return {"available": False, "error": None, "keys": []}
        return {"available": False, "error": f"{name}: {exc}", "keys": []}

    return {"available": True, "error": None, "keys": list(keys)}


def _secret(key: str, default: str | None = None) -> str | None:
    """Read a config value from the environment, then Streamlit secrets."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        import streamlit as st

        # Accessing .secrets raises if no secrets file exists at all.
        if key in st.secrets:
            text = str(st.secrets[key]).strip()
            if text:
                return text
    except Exception:
        # Diagnosis lives in secrets_status(); this path only decides whether
        # a value was found.
        pass
    return default


def diagnose_missing_secret(key: str) -> str:
    """One sentence saying why ``key`` could not be read.

    Written for the person looking at a deployed app with no shell access.
    """
    status = secrets_status()
    if status["error"]:
        return (
            f"The secrets file was found but could not be read "
            f"({status['error']}). That usually means the TOML is malformed - "
            f"check for missing quotes around values."
        )
    if not status["available"]:
        return (
            "No secrets were found at all. On Streamlit Community Cloud, set "
            "them under Manage app -> Settings -> Secrets, then reboot the app."
        )
    if key not in status["keys"]:
        found = ", ".join(status["keys"]) or "(none)"
        return (
            f"Secrets loaded fine, but {key} is not one of them. "
            f"Keys found: {found}. Check the spelling, and that the line is "
            f'written as {key} = "value".'
        )
    return (
        f"{key} is present in secrets but its value is empty. "
        "Make sure the value is inside quotes and not blank."
    )


@dataclass
class Settings:
    """Runtime settings for the agent."""

    # --- Database -----------------------------------------------------------
    # Postgres (Neon) in deployment; SQLite file when running locally.
    database_url: str = field(
        default_factory=lambda: _secret("DATABASE_URL") or "sqlite:///manga_tracker.db"
    )

    # --- LLM ----------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _secret("LLM_PROVIDER") or "gemini")
    gemini_api_key: str | None = field(default_factory=lambda: _secret("GEMINI_API_KEY"))
    llm_model: str = field(default_factory=lambda: _secret("LLM_MODEL") or "gemini-3.6-flash")

    # --- Safety limits ------------------------------------------------------
    max_actions_per_request: int = 8
    max_llm_retries: int = 2
    llm_timeout_seconds: float = 45.0

    # --- External API politeness -------------------------------------------
    # MangaDex allows ~5 req/sec globally; AniList is degraded to 30 req/min.
    mangadex_min_interval: float = 0.25
    anilist_min_interval: float = 2.1
    api_cache_ttl_seconds: int = 900
    http_timeout_seconds: float = 20.0
    user_agent: str = "cs4680-manga-tracker/1.0 (educational project)"

    @property
    def llm_configured(self) -> bool:
        """True when a real LLM key is available; otherwise mock mode is used."""
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return False

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgres://", "postgresql://"))


def get_settings() -> Settings:
    """Build a fresh Settings object (cheap; re-reads secrets each call)."""
    return Settings()
