"""Configuration loaded from Streamlit secrets or environment variables.

Secrets are never hard-coded. Locally the app falls back to a SQLite file and
a mock LLM so it runs with zero setup; in deployment both values come from
Streamlit's secrets manager.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _secret(key: str, default: str | None = None) -> str | None:
    """Read a config value from the environment, then Streamlit secrets."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        import streamlit as st

        # Accessing .secrets raises if no secrets file exists at all.
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


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
