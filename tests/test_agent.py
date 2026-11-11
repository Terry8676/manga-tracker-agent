"""End-to-end agent loop with a mocked LLM.

These are the tests that prove the whole pipeline works: prompt -> model ->
validator -> confirmation gate -> executor -> audit log.
"""

from __future__ import annotations

import json

from agent.agent import MangaAgent
from agent.db import session_scope
from agent.executor import Executor
from agent.llm_client import MockLLM
from agent.logger import recent_actions
from agent.prompts import build_user_prompt, render_library_context
from agent.schemas import AddSeries

from .test_executor import FakeAPI


def plan_json(actions, reply="ok", reasoning="because"):
    return json.dumps({"reasoning": reasoning, "reply": reply, "actions": actions})


def make_agent(settings, user, responses, api=None, upload=None):
    return MangaAgent(
        user.user_id,
        settings=settings,
        client=MockLLM(responses=responses),
        api=api or FakeAPI(),
        upload_provider=(lambda: upload),
    )


def seed_library(settings, user, *series):
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=FakeAPI())
        for kwargs in series:
            ex.execute(AddSeries(**kwargs))


# -- happy paths ------------------------------------------------------------ #


def test_single_action_turn(settings, user):
    seed_library(settings, user, {"title": "One Piece", "status": "reading",
                                 "chapters_read": 1120})
    agent = make_agent(
        settings,
        user,
        [plan_json(
            [{"action": "update_progress", "title": "One Piece", "chapters_read": 1125}],
            reply="You're current on One Piece.",
        )],
    )
    turn = agent.handle("caught up on OP")

    assert turn.ok
    assert turn.reply == "You're current on One Piece."
    assert len(turn.results) == 1 and turn.results[0].ok
    assert agent.library_snapshot()[0]["chapters_read"] == 1125


def test_multi_action_turn(settings, user):
    seed_library(
        settings,
        user,
        {"title": "Kagurabachi", "status": "reading", "chapters_read": 141},
        {"title": "Boruto: Two Blue Vortex", "status": "reading", "chapters_read": 18},
    )
    agent = make_agent(
        settings,
        user,
        [plan_json([
            {"action": "update_progress", "title": "Kagurabachi", "chapters_read": 142},
            {"action": "add_series", "title": "Dandadan", "status": "plan_to_read"},
        ])],
    )
    turn = agent.handle("on ch 142 of kagurabachi, add dandadan to plan to read")

    assert all(r.ok for r in turn.results)
    library = {s["title"]: s for s in agent.library_snapshot()}
    assert library["Kagurabachi"]["chapters_read"] == 142
    assert library["Dandadan"]["status"] == "plan_to_read"


def test_chat_with_no_actions(settings, user):
    agent = make_agent(settings, user, [plan_json([], reply="Ask me to track something.")])
    turn = agent.handle("hey what can you do")
    assert turn.ok
    assert turn.results == []
    assert "track" in turn.reply


def test_release_watch_flow(settings, user):
    seed_library(settings, user, {"title": "One Piece", "status": "reading",
                                  "chapters_read": 1120})
    api = FakeAPI(latest={"One Piece": 1125.0})
    agent = make_agent(
        settings,
        user,
        [plan_json([
            {"action": "check_new_chapters"},
            {"action": "query_library", "behind_only": True, "sort_by": "chapters_behind"},
        ])],
        api=api,
    )
    turn = agent.handle("anything new to read?")

    assert all(r.ok for r in turn.results)
    behind = turn.results[1].data
    assert behind[0]["title"] == "One Piece"
    assert behind[0]["chapters_behind"] == 5


def test_export_produces_a_download(settings, user):
    seed_library(settings, user, {"title": "Frieren", "chapters_read": 130})
    agent = make_agent(settings, user, [plan_json([{"action": "export_list", "format": "csv"}])])
    turn = agent.handle("export my list")
    assert turn.report.downloads
    assert "Frieren" in turn.report.downloads[0]["content"]


# -- confirmation gate ------------------------------------------------------ #


def test_destructive_action_pauses_for_confirmation(settings, user):
    seed_library(settings, user, {"title": "Black Clover", "chapters_read": 368})
    agent = make_agent(
        settings,
        user,
        [plan_json([{"action": "remove_series", "title": "Black Clover"}],
                   reply="Want me to go ahead?")],
    )
    turn = agent.handle("delete black clover")

    assert turn.pending_confirmation
    assert turn.report is None
    # Nothing was deleted yet.
    assert len(agent.library_snapshot()) == 1
    assert "go ahead" in turn.reply


def test_confirming_runs_the_held_plan_without_calling_the_llm_again(settings, user):
    seed_library(settings, user, {"title": "Black Clover", "chapters_read": 368})
    client = MockLLM(responses=[plan_json([{"action": "remove_series", "title": "Black Clover"}])])
    agent = MangaAgent(
        user.user_id, settings=settings, client=client, api=FakeAPI()
    )

    held = agent.handle("delete black clover")
    assert held.pending_confirmation
    calls_before = len(client.calls)

    final = agent.handle(
        "delete black clover", confirmed=True, preapproved_plan=held.plan
    )

    assert final.results[0].ok
    assert agent.library_snapshot() == []
    # The model is not consulted again, so the plan cannot change after
    # the user approved it.
    assert len(client.calls) == calls_before


def test_cancelling_leaves_data_untouched(settings, user):
    seed_library(settings, user, {"title": "Keep Me", "chapters_read": 10})
    agent = make_agent(
        settings, user, [plan_json([{"action": "remove_series", "title": "Keep Me"}])]
    )
    agent.handle("remove keep me")
    # User simply never confirms.
    assert [s["title"] for s in agent.library_snapshot()] == ["Keep Me"]


def test_mixed_plan_runs_safe_actions_and_holds_destructive_ones(settings, user):
    """A plan containing a delete is held in full, so the user sees the whole
    thing before any of it runs."""
    seed_library(settings, user, {"title": "Doomed", "chapters_read": 5})
    agent = make_agent(
        settings,
        user,
        [plan_json([
            {"action": "add_series", "title": "Safe Addition"},
            {"action": "remove_series", "title": "Doomed"},
        ])],
    )
    turn = agent.handle("add safe addition and delete doomed")

    assert turn.pending_confirmation
    titles = {s["title"] for s in agent.library_snapshot()}
    assert titles == {"Doomed"}  # nothing applied yet


# -- self-correction -------------------------------------------------------- #


def test_retries_after_malformed_json(settings, user):
    agent = make_agent(
        settings,
        user,
        [
            "Sure, I'll help! (no JSON here at all)",
            plan_json([{"action": "add_series", "title": "Dandadan"}]),
        ],
    )
    turn = agent.handle("add dandadan")

    assert turn.ok
    assert turn.validation_retries == 1
    assert [s["title"] for s in agent.library_snapshot()] == ["Dandadan"]


def test_retries_after_invalid_action_name(settings, user):
    agent = make_agent(
        settings,
        user,
        [
            plan_json([{"action": "delete_everything"}]),
            plan_json([{"action": "query_library"}]),
        ],
    )
    turn = agent.handle("show my list")
    assert turn.ok
    assert turn.validation_retries == 1


def test_retry_prompt_contains_the_validator_feedback(settings, user):
    client = MockLLM(responses=[
        plan_json([{"action": "update_progress", "title": "x", "chapters_read": -1}]),
        plan_json([{"action": "query_library"}]),
    ])
    agent = MangaAgent(user.user_id, settings=settings, client=client, api=FakeAPI())
    agent.handle("set x to -1")

    retry_prompt = client.calls[1][1]
    assert "Correction needed" in retry_prompt
    assert "chapters_read" in retry_prompt


def test_gives_up_after_max_retries_and_logs_it(settings, user):
    agent = make_agent(settings, user, ["not json", "still not json", "nope"])
    turn = agent.handle("do something")

    assert not turn.ok
    assert "couldn't turn that into a valid set of actions" in turn.reply
    with session_scope() as session:
        logs = recent_actions(session, user.user_id)
    assert logs[0].action_name == "validation_failed"
    assert logs[0].status == "error"


# -- LLM failures ----------------------------------------------------------- #


def test_llm_rate_limit_is_reported_not_crashed(settings, user):
    from agent.llm_client import LLMError

    class AlwaysLimited:
        label = "stub"

        def complete(self, system, user_prompt):
            raise LLMError("The LLM is rate limited right now.", retryable=True)

    agent = MangaAgent(
        user.user_id, settings=settings, client=AlwaysLimited(), api=FakeAPI()
    )
    turn = agent.handle("add something")

    assert not turn.ok
    assert "rate limited" in turn.reply
    with session_scope() as session:
        assert recent_actions(session, user.user_id)[0].action_name == "llm_error"


def test_empty_request_is_rejected_without_calling_the_llm(settings, user):
    client = MockLLM(responses=[plan_json([])])
    agent = MangaAgent(user.user_id, settings=settings, client=client, api=FakeAPI())
    turn = agent.handle("   ")
    assert not turn.ok
    assert client.calls == []


# -- action cap ------------------------------------------------------------- #


def test_action_cap_is_enforced_end_to_end(settings, user):
    settings.max_actions_per_request = 2
    agent = make_agent(
        settings,
        user,
        [plan_json([{"action": "add_series", "title": f"S{i}"} for i in range(5)])],
    )
    turn = agent.handle("add five things")

    assert len(turn.results) == 2
    assert turn.notes and "Capped at 2" in turn.notes[0]
    assert len(agent.library_snapshot()) == 2


# -- prompt construction ---------------------------------------------------- #


def test_library_context_is_injected_into_the_prompt(settings, user):
    seed_library(settings, user, {"title": "Kagurabachi", "status": "reading",
                                  "chapters_read": 141})
    client = MockLLM(responses=[plan_json([])])
    agent = MangaAgent(user.user_id, settings=settings, client=client, api=FakeAPI())
    agent.handle("what should I read")

    prompt = client.calls[0][1]
    assert "Kagurabachi" in prompt
    assert "ch 141" in prompt


def test_system_prompt_lists_every_action(settings, user):
    from agent.schemas import ACTION_NAMES

    client = MockLLM(responses=[plan_json([])])
    agent = MangaAgent(user.user_id, settings=settings, client=client, api=FakeAPI())
    agent.handle("hi")

    system = client.calls[0][0]
    for name in ACTION_NAMES:
        assert name in system


def test_prompt_states_whether_an_upload_exists(settings, user):
    client = MockLLM(responses=[plan_json([]), plan_json([])])
    with_upload = MangaAgent(
        user.user_id, settings=settings, client=client, api=FakeAPI(),
        upload_provider=lambda: "title\nFoo\n",
    )
    with_upload.handle("import my csv")
    assert "A CSV file has been uploaded" in client.calls[0][1]

    client2 = MockLLM(responses=[plan_json([])])
    without = MangaAgent(user.user_id, settings=settings, client=client2, api=FakeAPI())
    without.handle("import my csv")
    assert "No CSV has been uploaded" in client2.calls[0][1]


def test_conversation_history_is_included(settings, user):
    client = MockLLM(responses=[plan_json([])])
    agent = MangaAgent(user.user_id, settings=settings, client=client, api=FakeAPI())
    agent.handle("and the other one?", history=[("user", "add dandadan"),
                                                ("assistant", "Added Dandadan.")])
    prompt = client.calls[0][1]
    assert "add dandadan" in prompt
    assert "Added Dandadan." in prompt


def test_empty_library_context_is_explicit():
    assert "empty" in render_library_context([])


def test_library_context_truncates_long_lists():
    rows = [{"title": f"S{i}", "status": "reading", "chapters_read": i} for i in range(60)]
    rendered = render_library_context(rows, max_rows=10)
    assert "and 50 more" in rendered


def test_user_prompt_ends_with_the_request():
    prompt = build_user_prompt("add dandadan", [])
    assert prompt.rstrip().endswith("JSON object only.")
    assert "User: add dandadan" in prompt


# -- isolation through the full agent --------------------------------------- #


def test_agents_for_different_users_see_different_libraries(settings, user, other_user):
    seed_library(settings, user, {"title": "Mine", "chapters_read": 1})
    seed_library(settings, other_user, {"title": "Theirs", "chapters_read": 1})

    a = make_agent(settings, user, [plan_json([{"action": "query_library"}])])
    b = make_agent(settings, other_user, [plan_json([{"action": "query_library"}])])

    assert [r["title"] for r in a.handle("list").results[0].data] == ["Mine"]
    assert [r["title"] for r in b.handle("list").results[0].data] == ["Theirs"]


def test_audit_log_is_per_user(settings, user, other_user):
    make_agent(settings, user, [plan_json([{"action": "add_series", "title": "A"}])]).handle("add a")
    make_agent(settings, other_user, [plan_json([{"action": "add_series", "title": "B"}])]).handle("add b")

    with session_scope() as session:
        mine = recent_actions(session, user.user_id)
        theirs = recent_actions(session, other_user.user_id)

    assert len(mine) == 1 and len(theirs) == 1
    assert json.loads(mine[0].action_json)["title"] == "A"
    assert json.loads(theirs[0].action_json)["title"] == "B"
