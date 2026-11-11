"""Config resolution and the Postgres/SQLite connection-string handling."""

from __future__ import annotations

from agent.config import Settings
from agent.db import _normalise_url


def test_postgres_scheme_is_upgraded_for_sqlalchemy():
    url = _normalise_url("postgres://u:p@host/db")
    assert url.startswith("postgresql+psycopg2://")


def test_postgresql_scheme_gets_the_driver():
    url = _normalise_url("postgresql://u:p@host/db")
    assert url.startswith("postgresql+psycopg2://")


def test_ssl_is_forced_for_hosted_postgres():
    assert "sslmode=require" in _normalise_url("postgres://u:p@host/db")


def test_existing_sslmode_is_not_duplicated():
    url = _normalise_url("postgres://u:p@host/db?sslmode=verify-full")
    assert url.count("sslmode=") == 1
    assert "verify-full" in url


def test_sslmode_is_appended_with_ampersand_when_params_exist():
    url = _normalise_url("postgres://u:p@host/db?connect_timeout=10")
    assert "connect_timeout=10&sslmode=require" in url


def test_neon_style_url_is_handled():
    neon = (
        "postgresql://neondb_owner:secret@ep-cool-name-123456.us-east-2."
        "aws.neon.tech/neondb?sslmode=require"
    )
    url = _normalise_url(neon)
    assert url.startswith("postgresql+psycopg2://")
    assert url.count("sslmode=") == 1


def test_sqlite_url_is_left_alone():
    assert _normalise_url("sqlite:///local.db") == "sqlite:///local.db"


def test_default_database_is_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert Settings().database_url.startswith("sqlite")


def test_env_var_overrides_default_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h/d")
    assert Settings().database_url == "postgres://u:p@h/d"


def test_is_postgres_detection():
    assert Settings(database_url="postgres://u:p@h/d").is_postgres
    assert Settings(database_url="postgresql://u:p@h/d").is_postgres
    assert not Settings(database_url="sqlite:///x.db").is_postgres


def test_llm_configured_requires_a_key():
    assert not Settings(gemini_api_key=None).llm_configured
    assert Settings(gemini_api_key="abc123").llm_configured


def test_api_key_comes_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert Settings().gemini_api_key == "from-env"


def test_safety_limits_have_sane_defaults():
    s = Settings()
    assert s.max_actions_per_request >= 1
    assert s.max_llm_retries >= 1
    assert s.mangadex_min_interval >= 0.2  # stays under ~5 req/sec
    assert s.anilist_min_interval >= 2.0  # stays under 30 req/min
