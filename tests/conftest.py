"""Shared test fixtures. Every test gets a throwaway SQLite database."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import db as db_module  # noqa: E402
from agent.auth import sign_up  # noqa: E402
from agent.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at a fresh SQLite file, with no LLM key."""
    s = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        llm_provider="gemini",
        gemini_api_key=None,
        mangadex_min_interval=0.0,
        anilist_min_interval=0.0,
        api_cache_ttl_seconds=60,
    )
    db_module.reset_engine()
    db_module.init_db(s, force=True)
    yield s
    db_module.reset_engine()


@pytest.fixture
def user(settings):
    result = sign_up("reader_one", "hunter2hunter2")
    assert result.ok, result.message
    return result


@pytest.fixture
def other_user(settings):
    result = sign_up("reader_two", "password1234")
    assert result.ok, result.message
    return result


@pytest.fixture
def session(settings):
    with db_module.session_scope() as s:
        yield s
