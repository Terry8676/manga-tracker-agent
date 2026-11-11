"""Validation and safety checks between the LLM and the executor.

Nothing the LLM produces is trusted. This module:
  1. extracts JSON from the raw model output (models like to wrap it in prose
     or markdown fences),
  2. validates it against the schemas in ``schemas.py``,
  3. re-derives the ``requires_confirmation`` flag itself rather than believing
     what the model claimed,
  4. caps how many actions a single request can trigger,
  5. produces a human-readable error that can be fed back to the LLM so it can
     correct its own output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from .config import Settings, get_settings
from .schemas import ACTION_NAMES, Action, AgentPlan

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class ValidationProblem(Exception):
    """Raised when LLM output cannot be turned into a usable plan."""

    def __init__(self, message: str, *, feedback: str | None = None):
        super().__init__(message)
        self.message = message
        #: Text handed back to the LLM on a retry.
        self.feedback = feedback or message


@dataclass
class ValidatedPlan:
    reasoning: str
    reply: str
    actions: list[Action] = field(default_factory=list)
    #: Actions dropped because the plan exceeded the per-request cap.
    truncated: list[str] = field(default_factory=list)

    @property
    def needs_confirmation(self) -> bool:
        return any(a.requires_confirmation for a in self.actions)


def extract_json(raw: str) -> dict:
    """Pull the JSON object out of a model response.

    Handles bare JSON, fenced code blocks, and JSON with leading or trailing
    commentary.
    """
    if not raw or not raw.strip():
        raise ValidationProblem(
            "The model returned an empty response.",
            feedback="Your response was empty. Reply with a single JSON object.",
        )

    candidates: list[str] = []
    for match in _FENCE_RE.findall(raw):
        candidates.append(match.strip())
    candidates.append(raw.strip())

    # Last resort: the outermost {...} span.
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # Model returned just the action array.
            return {"actions": parsed}

    raise ValidationProblem(
        "The model's response was not valid JSON.",
        feedback=(
            "Your previous reply could not be parsed as JSON. Respond with "
            "exactly one JSON object and no other text, no markdown fences."
        ),
    )


def _format_pydantic_error(err: ValidationError) -> str:
    lines = []
    for e in err.errors()[:6]:
        loc = ".".join(str(p) for p in e["loc"] if p != "actions")
        lines.append(f"- {loc or 'root'}: {e['msg']}")
    return "\n".join(lines)


def validate_plan(raw: str, settings: Settings | None = None) -> ValidatedPlan:
    """Turn raw model output into a ValidatedPlan, or raise ValidationProblem."""
    settings = settings or get_settings()
    payload = extract_json(raw)

    if "actions" not in payload:
        payload["actions"] = []
    if not isinstance(payload["actions"], list):
        raise ValidationProblem(
            "'actions' must be a list.",
            feedback="The 'actions' key must be a JSON array of action objects.",
        )

    # Catch unknown action names before Pydantic does, so the feedback names
    # the offending action instead of dumping a discriminator error.
    unknown = []
    for item in payload["actions"]:
        if isinstance(item, dict):
            name = item.get("action")
            if name not in ACTION_NAMES:
                unknown.append(repr(name))
    if unknown:
        raise ValidationProblem(
            f"Unknown action(s): {', '.join(unknown)}",
            feedback=(
                f"You used action name(s) {', '.join(unknown)}, which do not exist. "
                f"Only these actions are allowed: {', '.join(ACTION_NAMES)}. "
                "Rewrite the plan using only those."
            ),
        )

    try:
        plan = AgentPlan.model_validate(payload)
    except ValidationError as err:
        detail = _format_pydantic_error(err)
        raise ValidationProblem(
            f"Action arguments failed validation:\n{detail}",
            feedback=(
                "Your plan failed schema validation:\n"
                f"{detail}\n"
                "Fix these fields and return the corrected JSON object."
            ),
        ) from err

    actions = list(plan.actions)
    truncated: list[str] = []
    cap = settings.max_actions_per_request
    if len(actions) > cap:
        truncated = [a.action for a in actions[cap:]]
        actions = actions[:cap]

    # Re-derive confirmation from our own rules, ignoring what the LLM claimed.
    for action in actions:
        action.requires_confirmation = action.is_destructive()

    return ValidatedPlan(
        reasoning=plan.reasoning.strip(),
        reply=plan.reply.strip(),
        actions=actions,
        truncated=truncated,
    )
