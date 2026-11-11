"""Safety layer: nothing the LLM says is trusted."""

from __future__ import annotations

import json

import pytest

from agent.config import Settings
from agent.validator import ValidationProblem, extract_json, validate_plan


def _raw(actions, **extra):
    return json.dumps({"reasoning": "r", "reply": "ok", "actions": actions, **extra})


# -- JSON extraction -------------------------------------------------------- #


def test_extracts_bare_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extracts_from_markdown_fence():
    raw = 'Sure!\n```json\n{"actions": []}\n```\nHope that helps.'
    assert extract_json(raw) == {"actions": []}


def test_extracts_from_unlabelled_fence():
    assert extract_json('```\n{"actions": []}\n```') == {"actions": []}


def test_extracts_with_surrounding_prose():
    raw = 'Here is the plan: {"reply": "hi", "actions": []} Let me know!'
    assert extract_json(raw)["reply"] == "hi"


def test_bare_action_array_is_wrapped():
    raw = '[{"action": "query_library"}]'
    assert extract_json(raw)["actions"][0]["action"] == "query_library"


def test_empty_response_raises():
    with pytest.raises(ValidationProblem) as err:
        extract_json("   ")
    assert "empty" in err.value.feedback.lower()


def test_unparseable_response_raises_with_feedback():
    with pytest.raises(ValidationProblem) as err:
        extract_json("I'm afraid I can't do that, Dave.")
    assert "json" in err.value.feedback.lower()


# -- Schema validation ------------------------------------------------------ #


def test_valid_plan_passes():
    plan = validate_plan(
        _raw([{"action": "update_progress", "title": "One Piece", "chapters_read": 1125}])
    )
    assert len(plan.actions) == 1
    assert plan.actions[0].chapters_read == 1125
    assert plan.reply == "ok"


def test_unknown_action_is_rejected_by_name():
    with pytest.raises(ValidationProblem) as err:
        validate_plan(_raw([{"action": "drop_database", "title": "x"}]))
    assert "drop_database" in err.value.feedback
    assert "add_series" in err.value.feedback  # lists the legal set


def test_negative_chapter_is_rejected():
    with pytest.raises(ValidationProblem) as err:
        validate_plan(
            _raw([{"action": "update_progress", "title": "x", "chapters_read": -5}])
        )
    assert "chapters_read" in err.value.feedback


def test_absurd_chapter_is_rejected():
    with pytest.raises(ValidationProblem):
        validate_plan(
            _raw([{"action": "update_progress", "title": "x", "chapters_read": 999999}])
        )


def test_out_of_range_rating_is_rejected():
    with pytest.raises(ValidationProblem):
        validate_plan(_raw([{"action": "rate_series", "title": "x", "rating": 50}]))


def test_invalid_status_is_rejected():
    with pytest.raises(ValidationProblem) as err:
        validate_plan(_raw([{"action": "set_status", "title": "x", "status": "banana"}]))
    assert "status" in err.value.feedback


def test_blank_title_is_rejected():
    with pytest.raises(ValidationProblem):
        validate_plan(_raw([{"action": "add_series", "title": "   "}]))


def test_missing_required_field_is_rejected():
    with pytest.raises(ValidationProblem):
        validate_plan(_raw([{"action": "update_progress", "title": "x"}]))


def test_extra_unexpected_field_is_rejected():
    with pytest.raises(ValidationProblem):
        validate_plan(
            _raw([{"action": "add_series", "title": "x", "sql": "DROP TABLE users"}])
        )


def test_actions_must_be_a_list():
    with pytest.raises(ValidationProblem) as err:
        validate_plan(json.dumps({"actions": "query_library"}))
    assert "array" in err.value.feedback.lower()


def test_missing_actions_key_defaults_to_empty():
    plan = validate_plan(json.dumps({"reply": "just chatting"}))
    assert plan.actions == []
    assert plan.reply == "just chatting"


def test_status_is_normalised():
    plan = validate_plan(_raw([{"action": "set_status", "title": "x", "status": "Plan To Read"}]))
    assert plan.actions[0].status == "plan_to_read"


def test_decimal_chapters_allowed():
    plan = validate_plan(
        _raw([{"action": "update_progress", "title": "x", "chapters_read": 1045.5}])
    )
    assert plan.actions[0].chapters_read == 1045.5


# -- Safety rules ----------------------------------------------------------- #


def test_destructive_flag_is_rederived_not_trusted():
    """The LLM claims no confirmation is needed; we override it."""
    plan = validate_plan(
        _raw([{"action": "remove_series", "title": "x", "requires_confirmation": False}])
    )
    assert plan.actions[0].requires_confirmation is True
    assert plan.needs_confirmation


def test_dropping_a_series_needs_confirmation():
    plan = validate_plan(_raw([{"action": "set_status", "title": "x", "status": "dropped"}]))
    assert plan.actions[0].requires_confirmation is True


def test_completing_a_series_does_not_need_confirmation():
    plan = validate_plan(_raw([{"action": "set_status", "title": "x", "status": "completed"}]))
    assert plan.actions[0].requires_confirmation is False


def test_import_needs_confirmation():
    plan = validate_plan(_raw([{"action": "import_list", "overwrite": True}]))
    assert plan.needs_confirmation


def test_safe_action_claiming_confirmation_is_reset():
    plan = validate_plan(
        _raw([{"action": "query_library", "requires_confirmation": True}])
    )
    assert plan.actions[0].requires_confirmation is False


def test_action_count_is_capped():
    s = Settings(max_actions_per_request=3)
    actions = [{"action": "add_series", "title": f"Series {i}"} for i in range(10)]
    plan = validate_plan(_raw(actions), s)
    assert len(plan.actions) == 3
    assert len(plan.truncated) == 7
