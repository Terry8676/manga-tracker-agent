"""Config resolution and the Postgres/SQLite connection-string handling."""

from __future__ import annotations

import pytest

from agent.config import Settings
from agent.db import _normalise_url


@pytest.fixture
def no_streamlit_secrets(monkeypatch):
    """Isolate Settings from any real .streamlit/secrets.toml.

    Without this, a test asserting a default behaves differently on a
    machine that happens to have secrets configured - which is exactly what
    happened: the SQLite-default test passed in CI and failed on a developer
    machine holding a real DATABASE_URL.
    """
    import streamlit as st

    class Missing(Exception):
        pass

    Missing.__name__ = "StreamlitSecretNotFoundError"

    class EmptySecrets:
        def keys(self):
            raise Missing()

        def __contains__(self, key):
            raise Missing()

        def __getitem__(self, key):
            raise Missing()

    monkeypatch.setattr(st, "secrets", EmptySecrets(), raising=False)


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


def test_default_database_is_sqlite(monkeypatch, no_streamlit_secrets):
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


def test_safety_limits_have_sane_defaults(no_streamlit_secrets):
    s = Settings()
    assert s.max_actions_per_request >= 1
    assert s.max_llm_retries >= 1
    assert s.mangadex_min_interval >= 0.2  # stays under ~5 req/sec
    assert s.anilist_min_interval >= 2.0  # stays under 30 req/min


# -- secret diagnostics ----------------------------------------------------- #


def test_secrets_status_reports_no_secrets_file(monkeypatch):
    """Running with no secrets file is normal, not an error."""
    import agent.config as cfg

    class FakeSecretNotFoundError(Exception):
        pass

    FakeSecretNotFoundError.__name__ = "StreamlitSecretNotFoundError"

    class FakeSecrets:
        def keys(self):
            raise FakeSecretNotFoundError("no secrets")

    import streamlit as st

    monkeypatch.setattr(st, "secrets", FakeSecrets(), raising=False)
    status = cfg.secrets_status()
    assert status["available"] is False
    assert status["error"] is None
    assert status["keys"] == []


def test_secrets_status_reports_a_parse_failure(monkeypatch):
    """A malformed TOML must be distinguishable from a missing file."""
    import streamlit as st

    import agent.config as cfg

    class FakeSecrets:
        def keys(self):
            raise ValueError("expected '=' after key")

    monkeypatch.setattr(st, "secrets", FakeSecrets(), raising=False)
    status = cfg.secrets_status()
    assert status["available"] is False
    assert "expected" in status["error"]


def test_secrets_status_lists_key_names_only(monkeypatch):
    import streamlit as st

    import agent.config as cfg

    class FakeSecrets:
        def keys(self):
            return ["DATABASE_URL", "GEMINI_API_KEY"]

    monkeypatch.setattr(st, "secrets", FakeSecrets(), raising=False)
    status = cfg.secrets_status()
    assert status["available"] is True
    assert status["keys"] == ["DATABASE_URL", "GEMINI_API_KEY"]


def test_diagnosis_names_the_missing_key_and_lists_what_was_found(monkeypatch):
    import streamlit as st

    import agent.config as cfg

    class FakeSecrets:
        def keys(self):
            return ["DATABASE_URL"]

    monkeypatch.setattr(st, "secrets", FakeSecrets(), raising=False)
    message = cfg.diagnose_missing_secret("GEMINI_API_KEY")
    assert "GEMINI_API_KEY is not one of them" in message
    assert "DATABASE_URL" in message


def test_diagnosis_explains_a_malformed_file(monkeypatch):
    import streamlit as st

    import agent.config as cfg

    class FakeSecrets:
        def keys(self):
            raise ValueError("bad toml")

    monkeypatch.setattr(st, "secrets", FakeSecrets(), raising=False)
    assert "malformed" in cfg.diagnose_missing_secret("GEMINI_API_KEY")


def test_diagnosis_points_at_the_cloud_dashboard_when_nothing_is_set(monkeypatch):
    import streamlit as st

    import agent.config as cfg

    class Missing(Exception):
        pass

    Missing.__name__ = "StreamlitSecretNotFoundError"

    class FakeSecrets:
        def keys(self):
            raise Missing()

    monkeypatch.setattr(st, "secrets", FakeSecrets(), raising=False)
    assert "Settings" in cfg.diagnose_missing_secret("GEMINI_API_KEY")


def test_blank_secret_value_is_treated_as_unset(monkeypatch):
    import streamlit as st

    import agent.config as cfg

    class FakeSecrets:
        _data = {"GEMINI_API_KEY": "   "}

        def keys(self):
            return list(self._data)

        def __contains__(self, key):
            return key in self._data

        def __getitem__(self, key):
            return self._data[key]

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(st, "secrets", FakeSecrets(), raising=False)
    assert cfg._secret("GEMINI_API_KEY") is None
