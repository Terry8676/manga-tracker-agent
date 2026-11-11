"""Agent orchestration: the loop that turns one user message into executed
actions.

    user text
      -> prompt (system + few-shot + library context)
      -> LLM
      -> validator  --(invalid)--> feedback -> LLM  (up to N retries)
      -> plan
      -> confirmation gate for destructive actions
      -> executor
      -> audit log
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, get_settings
from .db import session_scope
from .executor import ExecutionReport, Executor
from .llm_client import LLMClient, LLMError, build_client
from .logger import record_action
from .manga_api import MangaAPI
from .prompts import SYSTEM_PROMPT, build_retry_prompt, build_user_prompt
from .validator import ValidatedPlan, ValidationProblem, validate_plan


@dataclass
class AgentTurn:
    """Everything the UI needs to render one exchange."""

    ok: bool
    reply: str
    reasoning: str = ""
    plan: ValidatedPlan | None = None
    report: ExecutionReport | None = None
    pending_confirmation: bool = False
    error: str | None = None
    llm_attempts: int = 0
    validation_retries: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def results(self) -> list:
        return self.report.results if self.report else []


class MangaAgent:
    """One instance serves one logged-in user."""

    def __init__(
        self,
        user_id: int,
        *,
        settings: Settings | None = None,
        client: LLMClient | None = None,
        api: MangaAPI | None = None,
        upload_provider=None,
    ):
        self.user_id = user_id
        self.settings = settings or get_settings()
        self.client = client or build_client(self.settings)
        self.api = api or MangaAPI(self.settings)
        self.upload_provider = upload_provider or (lambda: None)

    # -- context ------------------------------------------------------------ #

    def library_snapshot(self) -> list[dict]:
        with session_scope() as session:
            executor = Executor(
                session, self.user_id, settings=self.settings, api=self.api
            )
            return [row.to_dict() for row in executor._all_series()]

    # -- planning ----------------------------------------------------------- #

    def plan(
        self, request: str, history: list[tuple[str, str]] | None = None
    ) -> tuple[ValidatedPlan, int, int]:
        """Ask the LLM for a plan, retrying on validation failure.

        Returns (plan, llm_attempts, validation_retries).
        Raises LLMError or ValidationProblem if it cannot get a usable plan.
        """
        series = self.library_snapshot()
        upload_available = self.upload_provider() is not None
        user_prompt = build_user_prompt(
            request, series, history=history, upload_available=upload_available
        )

        attempts = 0
        prompt = user_prompt
        last_problem: ValidationProblem | None = None

        for retry in range(self.settings.max_llm_retries + 1):
            response = self.client.complete(SYSTEM_PROMPT, prompt)
            attempts += response.attempts
            try:
                return validate_plan(response.text, self.settings), attempts, retry
            except ValidationProblem as problem:
                last_problem = problem
                prompt = build_retry_prompt(
                    user_prompt, response.text, problem.feedback
                )

        assert last_problem is not None
        raise last_problem

    # -- full turn ---------------------------------------------------------- #

    def handle(
        self,
        request: str,
        *,
        history: list[tuple[str, str]] | None = None,
        confirmed: bool = False,
        preapproved_plan: ValidatedPlan | None = None,
    ) -> AgentTurn:
        """Plan and execute one user message.

        When ``preapproved_plan`` is given, the LLM is skipped entirely - this
        is the path taken after the user clicks "Confirm" on a destructive
        action, so the model cannot change the plan between the prompt and the
        approval.
        """
        request = (request or "").strip()
        if not request and preapproved_plan is None:
            return AgentTurn(False, "Type a request first.", error="empty request")

        llm_attempts = 0
        retries = 0

        if preapproved_plan is not None:
            plan = preapproved_plan
        else:
            try:
                plan, llm_attempts, retries = self.plan(request, history)
            except LLMError as exc:
                self._log_failure(request, "llm_error", exc.message)
                return AgentTurn(
                    False,
                    exc.message,
                    error=exc.message,
                    llm_attempts=llm_attempts,
                )
            except ValidationProblem as exc:
                message = (
                    "I couldn't turn that into a valid set of actions, even after "
                    f"retrying. ({exc.message}) Try rewording it."
                )
                self._log_failure(request, "validation_failed", exc.message)
                return AgentTurn(
                    False,
                    message,
                    error=exc.message,
                    validation_retries=self.settings.max_llm_retries,
                )

        notes: list[str] = []
        if plan.truncated:
            notes.append(
                f"Capped at {self.settings.max_actions_per_request} actions; skipped: "
                + ", ".join(plan.truncated)
            )

        # Destructive actions stop here until the user clicks confirm.
        if plan.needs_confirmation and not confirmed:
            with session_scope() as session:
                for action in plan.actions:
                    if action.requires_confirmation:
                        record_action(
                            session,
                            user_id=self.user_id,
                            user_request=request,
                            action_name=action.action,
                            payload=action.model_dump(mode="json"),
                            status="cancelled",
                            result_summary="Awaiting user confirmation.",
                        )
            return AgentTurn(
                True,
                plan.reply or "This needs your confirmation before I run it.",
                reasoning=plan.reasoning,
                plan=plan,
                pending_confirmation=True,
                llm_attempts=llm_attempts,
                validation_retries=retries,
                notes=notes,
            )

        with session_scope() as session:
            executor = Executor(
                session,
                self.user_id,
                settings=self.settings,
                api=self.api,
                upload_provider=self.upload_provider,
            )
            report = executor.execute_plan(plan.actions, request, confirmed=confirmed)

        return AgentTurn(
            True,
            plan.reply or "Done.",
            reasoning=plan.reasoning,
            plan=plan,
            report=report,
            llm_attempts=llm_attempts,
            validation_retries=retries,
            notes=notes,
        )

    def _log_failure(self, request: str, kind: str, detail: str) -> None:
        with session_scope() as session:
            record_action(
                session,
                user_id=self.user_id,
                user_request=request,
                action_name=kind,
                payload=None,
                status="error",
                result_summary=detail,
            )
