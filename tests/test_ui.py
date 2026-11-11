"""UI tests driven through Streamlit's own AppTest harness.

These run the real app.py: they click buttons, fill forms and read what the
page renders. The LLM and the manga APIs are patched to deterministic stubs,
and the database is a throwaway SQLite file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

from agent import db as db_module
from agent.config import Settings


def plan_json(actions, reply="ok"):
    return json.dumps({"reasoning": "r", "reply": reply, "actions": actions})


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A fresh app instance with a temp database and stubbed externals."""
    import streamlit as st

    from agent import agent as agent_module
    from agent import llm_client as llm_module
    from agent import manga_api as api_module
    from agent.llm_client import MockLLM

    # app.py caches the settings, API client and LLM client with
    # @st.cache_resource, and that cache is process-global. Without clearing
    # it, the second test in a run would keep the first test's database and
    # mock model.
    st.cache_resource.clear()

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'ui.db'}",
        gemini_api_key=None,
        mangadex_min_interval=0.0,
        anilist_min_interval=0.0,
    )
    monkeypatch.setattr("agent.config.get_settings", lambda: settings)

    db_module.reset_engine()
    db_module.init_db(settings, force=True)

    # The script calls build_client() through a cache_resource wrapper; patch
    # the underlying factory so every rerun gets our scripted model.
    holder = {"client": MockLLM(responses=[plan_json([])])}
    monkeypatch.setattr(llm_module, "build_client", lambda s=None: holder["client"])
    monkeypatch.setattr(agent_module, "build_client", lambda s=None: holder["client"])

    class StubAPI:
        def __init__(self, *a, **k):
            pass

        def search_manga(self, query, limit=5):
            return [{"title": "Dandadan", "year": 2021, "mangadex_id": "md1",
                     "publication_status": "ongoing", "description": "aliens",
                     "last_chapter": None}]

        def resolve_series(self, title):
            return {"mangadex_id": f"md-{title}", "title": title}

        def latest_chapter(self, mangadex_id, language="en"):
            return 1125.0

        def recommendations(self, **kw):
            return [{"title": "Kingdom", "genres": ["Action"], "chapters": 700,
                     "status": "releasing", "score": 90, "description": "d"}]

    monkeypatch.setattr(api_module, "MangaAPI", StubAPI)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at._llm_holder = holder  # so tests can script the next response
    at.run()
    yield at
    st.cache_resource.clear()
    db_module.reset_engine()


def script(at, *responses):
    """Script the next LLM replies.

    app.py wraps build_client() in @st.cache_resource, so the app keeps using
    the first client instance it was given. The holder's object is mutated in
    place rather than swapped, and the call counter is reset so each turn
    walks the new response list from the start (which is what the
    self-correction retry loop consumes).
    """
    client = at._llm_holder["client"]
    client.responses = list(responses)
    client.calls = []
    client._raised = 0
    client.raise_times = 0


def register(at, username="demo_user", password="password1234"):
    at.text_input(key="su_user").set_value(username)
    at.text_input(key="su_pass").set_value(password)
    at.button[1].click().run()  # 1 = "Create account"
    return at


def all_text(at) -> str:
    """Everything rendered on the page, flattened, for assertions."""
    chunks = []
    for element in at.markdown:
        chunks.append(element.value)
    for element in at.caption:
        chunks.append(element.value)
    for element in at.warning:
        chunks.append(element.value)
    for element in at.error:
        chunks.append(element.value)
    for element in at.info:
        chunks.append(element.value)
    for element in at.success:
        chunks.append(element.value)
    for element in at.title:
        chunks.append(element.value)
    for element in at.header:
        chunks.append(element.value)
    for element in at.subheader:
        chunks.append(element.value)
    return "\n".join(c for c in chunks if c)


# -- login screen ----------------------------------------------------------- #


def test_app_starts_without_exceptions(app):
    assert not app.exception
    assert "Manga Tracker Agent" in all_text(app)


def test_login_screen_shows_both_forms(app):
    text = all_text(app)
    assert "Sign in" in text
    assert "Create an account" in text


def test_chat_is_hidden_until_signed_in(app):
    assert len(app.chat_input) == 0


def test_signup_signs_the_user_in(app):
    register(app)
    assert not app.exception
    assert app.session_state["username"] == "demo_user"
    assert len(app.chat_input) == 1


def test_signup_rejects_short_password(app):
    app.text_input(key="su_user").set_value("shorty")
    app.text_input(key="su_pass").set_value("abc")
    app.button[1].click().run()
    assert any("at least 8" in e.value for e in app.error)
    assert app.session_state["user_id"] is None


def test_login_with_wrong_password_shows_error(app):
    register(app)
    app.session_state["user_id"] = None
    app.session_state["username"] = None
    app.run()
    app.text_input(key="login_user").set_value("demo_user")
    app.text_input(key="login_pass").set_value("wrongpassword")
    app.button[0].click().run()
    assert any("Incorrect username or password" in e.value for e in app.error)


def test_login_with_correct_password_works(app):
    register(app)
    app.session_state["user_id"] = None
    app.run()
    app.text_input(key="login_user").set_value("demo_user")
    app.text_input(key="login_pass").set_value("password1234")
    app.button[0].click().run()
    assert app.session_state["user_id"] is not None


# -- signed-in shell -------------------------------------------------------- #


def test_metrics_render_for_empty_library(app):
    register(app)
    labels = [m.label for m in app.metric]
    assert "Series tracked" in labels
    assert app.metric[0].value == "0"


def test_sidebar_warns_when_no_llm_key(app):
    register(app)
    assert any("No LLM key" in w.value for w in app.warning)


def test_sign_out_returns_to_login(app):
    register(app)
    signout = [b for b in app.button if b.label == "Sign out"][0]
    signout.click().run()
    assert app.session_state.get("user_id") is None
    assert "Sign in" in all_text(app)


# -- chat turns ------------------------------------------------------------- #


def test_adding_a_series_updates_library_and_metrics(app):
    register(app)
    script(
        app,
        plan_json(
            [{"action": "add_series", "title": "Dandadan", "status": "plan_to_read"}],
            reply="Added Dandadan.",
        ),
    )
    app.chat_input[0].set_value("add dandadan to plan to read").run()

    assert not app.exception
    text = all_text(app)
    assert "Added Dandadan." in text
    assert "add_series" in text
    assert app.metric[0].value == "1"


def test_plan_expander_lists_each_action_result(app):
    register(app)
    script(
        app,
        plan_json([
            {"action": "add_series", "title": "A"},
            {"action": "add_series", "title": "B"},
        ]),
    )
    app.chat_input[0].set_value("add a and b").run()
    text = all_text(app)
    # One status line per action inside the plan expander.
    assert text.count("`add_series`") == 2
    assert "Added 'A'" in text and "Added 'B'" in text
    assert "Reasoning:" in text


def test_failed_action_is_shown_without_crashing(app):
    register(app)
    script(
        app,
        plan_json([{"action": "update_progress", "title": "Ghost", "chapters_read": 5}]),
    )
    app.chat_input[0].set_value("update ghost").run()
    assert not app.exception
    assert "not in your library" in all_text(app)


def test_release_check_renders_a_table(app):
    register(app)
    script(app, plan_json([{"action": "add_series", "title": "One Piece",
                            "status": "reading", "chapters_read": 1120}]))
    app.chat_input[0].set_value("add one piece at 1120").run()

    script(app, plan_json([{"action": "check_new_chapters"}]))
    app.chat_input[0].set_value("anything new?").run()

    assert not app.exception
    assert len(app.dataframe) >= 1
    assert "Behind on" in all_text(app)


def test_recommendations_render(app):
    register(app)
    script(app, plan_json([{"action": "get_recommendations", "limit": 3}]))
    app.chat_input[0].set_value("recommend something").run()
    assert "Kingdom" in all_text(app)


def test_search_results_render(app):
    register(app)
    script(app, plan_json([{"action": "search_manga", "query": "dandadan"}]))
    app.chat_input[0].set_value("search for dandadan").run()
    assert "Dandadan" in all_text(app)


def test_export_offers_a_download_button(app):
    register(app)
    script(app, plan_json([{"action": "add_series", "title": "Frieren"}]))
    app.chat_input[0].set_value("add frieren").run()

    script(app, plan_json([{"action": "export_list", "format": "csv"}]))
    app.chat_input[0].set_value("export my list").run()

    labels = [b.label for b in app.download_button]
    assert any("manga-list.csv" in label for label in labels)


def test_llm_error_is_shown_as_a_warning_message(app):
    register(app)
    script(app, "this is not json", "still not json", "nope")
    app.chat_input[0].set_value("do something weird").run()
    assert not app.exception
    assert "couldn't turn that into a valid set of actions" in all_text(app)


def test_example_button_sends_a_request(app):
    register(app)
    script(app, plan_json([{"action": "query_library"}], reply="Here's your list."))
    example = [b for b in app.button if b.label.startswith("Export my list")]
    assert example, "example buttons should render for a new user"
    example[0].click().run()
    assert not app.exception


# -- confirmation flow ------------------------------------------------------ #


def _seed_and_request_delete(app):
    script(app, plan_json([{"action": "add_series", "title": "Doomed"}]))
    app.chat_input[0].set_value("add doomed").run()
    script(
        app,
        plan_json([{"action": "remove_series", "title": "Doomed"}],
                  reply="Want me to delete it?"),
    )
    app.chat_input[0].set_value("delete doomed").run()


def test_delete_asks_before_acting(app):
    register(app)
    _seed_and_request_delete(app)

    assert app.session_state["pending"] is not None
    assert any("needs your confirmation" in w.value for w in app.warning)
    assert [b.label for b in app.button if b.label == "Confirm"]
    # Still in the library.
    assert app.metric[0].value == "1"


def test_confirming_performs_the_delete(app):
    register(app)
    _seed_and_request_delete(app)
    [b for b in app.button if b.label == "Confirm"][0].click().run()

    assert not app.exception
    assert app.session_state["pending"] is None
    assert app.metric[0].value == "0"


def test_cancelling_keeps_the_series(app):
    register(app)
    _seed_and_request_delete(app)
    [b for b in app.button if b.label == "Cancel"][0].click().run()

    assert app.session_state["pending"] is None
    assert app.metric[0].value == "1"
    assert "Cancelled" in all_text(app)


def test_dropping_a_series_also_asks(app):
    register(app)
    script(app, plan_json([{"action": "add_series", "title": "Boruto",
                            "status": "reading", "chapters_read": 18}]))
    app.chat_input[0].set_value("add boruto").run()
    script(app, plan_json([{"action": "set_status", "title": "Boruto",
                            "status": "dropped"}]))
    app.chat_input[0].set_value("drop boruto").run()
    assert app.session_state["pending"] is not None


# -- audit log in the sidebar ---------------------------------------------- #


def test_action_log_appears_in_the_sidebar(app):
    register(app)
    script(app, plan_json([{"action": "add_series", "title": "Logged"}]))
    app.chat_input[0].set_value("add logged").run()

    assert any("add_series" in m.value for m in app.markdown)
    assert any("action-log.csv" in b.label or "log" in b.label.lower()
               for b in app.download_button)


def test_log_download_contains_the_action(app):
    register(app)
    script(app, plan_json([{"action": "add_series", "title": "Logged"}]))
    app.chat_input[0].set_value("add logged").run()

    log_buttons = [b for b in app.download_button if "log" in b.label.lower()]
    assert log_buttons


# -- data separation through the real UI ----------------------------------- #


def test_second_account_sees_an_empty_library(app):
    register(app, "first_user", "password1234")
    script(app, plan_json([{"action": "add_series", "title": "Private Series"}]))
    app.chat_input[0].set_value("add private series").run()
    assert app.metric[0].value == "1"

    [b for b in app.button if b.label == "Sign out"][0].click().run()
    register(app, "second_user", "password1234")

    assert app.metric[0].value == "0"
    assert "Private Series" not in all_text(app)
