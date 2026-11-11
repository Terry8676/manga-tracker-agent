"""Pydantic schemas for every action the agent is allowed to perform.

This module is the contract between the LLM and the executor. The LLM may only
emit actions named here, with arguments shaped exactly as declared. Anything
else is rejected before it can run.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .db import VALID_STATUSES

MAX_CHAPTER = 10_000.0
EXPORT_FORMATS = ("csv", "markdown", "json")


class _ActionBase(BaseModel):
    """Common behaviour for all actions."""

    model_config = ConfigDict(extra="forbid")

    #: Set by the LLM, re-derived by the validator. Never trusted as given.
    requires_confirmation: bool = False

    # A ClassVar, not a field: the LLM cannot send this key and talk its way
    # out of a confirmation prompt.
    DESTRUCTIVE: ClassVar[bool] = False

    def is_destructive(self) -> bool:
        return bool(self.DESTRUCTIVE)

    def describe(self) -> str:
        return self.action


class _TitleMixin(BaseModel):
    title: str = Field(min_length=1, max_length=255)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title cannot be blank")
        return v


# --------------------------------------------------------------------------- #
# Database actions
# --------------------------------------------------------------------------- #


class AddSeries(_ActionBase, _TitleMixin):
    action: Literal["add_series"] = "add_series"
    status: str = "plan_to_read"
    chapters_read: float = Field(default=0.0, ge=0, le=MAX_CHAPTER)
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        v = (v or "").strip().lower().replace(" ", "_").replace("-", "_")
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {', '.join(VALID_STATUSES)}")
        return v

    def describe(self) -> str:
        return f"Add '{self.title}' as {self.status.replace('_', ' ')}"


class UpdateProgress(_ActionBase, _TitleMixin):
    action: Literal["update_progress"] = "update_progress"
    chapters_read: float = Field(ge=0, le=MAX_CHAPTER)

    def describe(self) -> str:
        return f"Set '{self.title}' progress to chapter {self.chapters_read:g}"


class SetStatus(_ActionBase, _TitleMixin):
    action: Literal["set_status"] = "set_status"
    status: str

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        v = (v or "").strip().lower().replace(" ", "_").replace("-", "_")
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {', '.join(VALID_STATUSES)}")
        return v

    def is_destructive(self) -> bool:
        # Dropping loses the user's place in a series they were reading.
        return self.status == "dropped"

    def describe(self) -> str:
        return f"Mark '{self.title}' as {self.status.replace('_', ' ')}"


class RateSeries(_ActionBase, _TitleMixin):
    action: Literal["rate_series"] = "rate_series"
    rating: float = Field(ge=0, le=10)

    def describe(self) -> str:
        return f"Rate '{self.title}' {self.rating:g}/10"


class RemoveSeries(_ActionBase, _TitleMixin):
    action: Literal["remove_series"] = "remove_series"
    DESTRUCTIVE: ClassVar[bool] = True

    def describe(self) -> str:
        return f"Delete '{self.title}' from the library"


class QueryLibrary(_ActionBase):
    action: Literal["query_library"] = "query_library"
    status: str | None = None
    behind_only: bool = False
    min_rating: float | None = Field(default=None, ge=0, le=10)
    sort_by: Literal["title", "updated", "chapters_behind", "rating"] = "updated"
    limit: int = Field(default=50, ge=1, le=200)

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {', '.join(VALID_STATUSES)}")
        return v

    def describe(self) -> str:
        bits = []
        if self.status:
            bits.append(self.status.replace("_", " "))
        if self.behind_only:
            bits.append("with new chapters")
        if self.min_rating is not None:
            bits.append(f"rated {self.min_rating:g}+")
        return "Look up library" + (f" ({', '.join(bits)})" if bits else "")


# --------------------------------------------------------------------------- #
# External API actions
# --------------------------------------------------------------------------- #


class SearchManga(_ActionBase):
    action: Literal["search_manga"] = "search_manga"
    query: str = Field(min_length=1, max_length=120)
    limit: int = Field(default=5, ge=1, le=10)

    def describe(self) -> str:
        return f"Search MangaDex for '{self.query}'"


class CheckNewChapters(_ActionBase):
    action: Literal["check_new_chapters"] = "check_new_chapters"
    #: None means "every series the user is currently reading".
    title: str | None = Field(default=None, max_length=255)

    def describe(self) -> str:
        if self.title:
            return f"Check for new chapters of '{self.title}'"
        return "Check every series I'm reading for new chapters"


class GetRecommendations(_ActionBase):
    action: Literal["get_recommendations"] = "get_recommendations"
    genres: list[str] = Field(default_factory=list, max_length=5)
    similar_to: str | None = Field(default=None, max_length=255)
    finished_only: bool = False
    max_chapters: int | None = Field(default=None, ge=1, le=MAX_CHAPTER)
    limit: int = Field(default=5, ge=1, le=10)

    def describe(self) -> str:
        if self.similar_to:
            return f"Find manga similar to '{self.similar_to}'"
        if self.genres:
            return f"Recommend {', '.join(self.genres)} manga"
        return "Recommend something to read"


# --------------------------------------------------------------------------- #
# File actions
# --------------------------------------------------------------------------- #


class ExportList(_ActionBase):
    action: Literal["export_list"] = "export_list"
    format: Literal["csv", "markdown", "json"] = "csv"
    status: str | None = None

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {', '.join(VALID_STATUSES)}")
        return v

    def describe(self) -> str:
        scope = f" ({self.status.replace('_', ' ')})" if self.status else ""
        return f"Export library as {self.format}{scope}"


class ImportList(_ActionBase):
    action: Literal["import_list"] = "import_list"
    #: When true, existing rows with the same title are overwritten.
    overwrite: bool = False
    DESTRUCTIVE: ClassVar[bool] = True

    def describe(self) -> str:
        mode = "overwriting existing entries" if self.overwrite else "skipping duplicates"
        return f"Import the uploaded CSV, {mode}"


# --------------------------------------------------------------------------- #
# Discriminated union + plan wrapper
# --------------------------------------------------------------------------- #

Action = Annotated[
    Union[
        AddSeries,
        UpdateProgress,
        SetStatus,
        RateSeries,
        RemoveSeries,
        QueryLibrary,
        SearchManga,
        CheckNewChapters,
        GetRecommendations,
        ExportList,
        ImportList,
    ],
    Field(discriminator="action"),
]

ACTION_NAMES: tuple[str, ...] = (
    "add_series",
    "update_progress",
    "set_status",
    "rate_series",
    "remove_series",
    "query_library",
    "search_manga",
    "check_new_chapters",
    "get_recommendations",
    "export_list",
    "import_list",
)


class AgentPlan(BaseModel):
    """The whole response the LLM must return."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = Field(default="", max_length=2000)
    reply: str = Field(default="", max_length=2000)
    actions: list[Action] = Field(default_factory=list)
