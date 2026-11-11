"""Executor: the three action domains plus failure handling."""

from __future__ import annotations

import csv
import io
import json

import pytest

from agent.db import Series, session_scope
from agent.executor import Executor
from agent.logger import recent_actions
from agent.manga_api import MangaAPIError
from agent.schemas import (
    AddSeries,
    CheckNewChapters,
    ExportList,
    GetRecommendations,
    ImportList,
    QueryLibrary,
    RateSeries,
    RemoveSeries,
    SearchManga,
    SetStatus,
    UpdateProgress,
)


class FakeAPI:
    """Stand-in for MangaAPI with no network access."""

    def __init__(self, latest=None, search=None, recs=None, fail=False):
        self.latest_map = latest or {}
        self.search_results = search if search is not None else []
        self.recs = recs if recs is not None else []
        self.fail = fail
        self.calls: list[str] = []

    def search_manga(self, query, limit=5):
        self.calls.append(f"search:{query}")
        if self.fail:
            raise MangaAPIError("Could not reach api.mangadex.org: network error.")
        return self.search_results[:limit]

    def resolve_series(self, title):
        self.calls.append(f"resolve:{title}")
        if self.fail:
            raise MangaAPIError("Could not reach api.mangadex.org: network error.")
        if title in self.latest_map:
            return {"mangadex_id": f"md-{title}", "title": title}
        return None

    def latest_chapter(self, mangadex_id, language="en"):
        self.calls.append(f"latest:{mangadex_id}")
        if self.fail:
            raise MangaAPIError("Could not reach api.mangadex.org: network error.")
        title = mangadex_id.removeprefix("md-")
        return self.latest_map.get(title)

    def recommendations(self, **kwargs):
        self.calls.append("recommend")
        if self.fail:
            raise MangaAPIError("Could not reach graphql.anilist.co: network error.")
        exclude = {t.lower() for t in (kwargs.get("exclude_titles") or set())}
        return [r for r in self.recs if r["title"].lower() not in exclude][
            : kwargs.get("limit", 5)
        ]


@pytest.fixture
def ex(settings, user):
    """An executor bound to a live session, with a fake API."""
    api = FakeAPI()
    with session_scope() as session:
        yield Executor(session, user.user_id, settings=settings, api=api)


def seed(executor, title, **kw):
    return executor.execute(AddSeries(title=title, **kw))


# -- database domain -------------------------------------------------------- #


def test_add_series(ex):
    result = seed(ex, "Kagurabachi", status="reading", chapters_read=141)
    assert result.ok
    assert "Kagurabachi" in result.summary
    assert result.data["chapters_read"] == 141


def test_add_duplicate_fails(ex):
    seed(ex, "Kagurabachi")
    second = seed(ex, "Kagurabachi")
    assert second.status == "error"
    assert "already in your library" in second.summary


def test_update_progress(ex):
    seed(ex, "One Piece", status="reading", chapters_read=1120)
    result = ex.execute(UpdateProgress(title="One Piece", chapters_read=1125))
    assert result.ok
    assert result.data["chapters_read"] == 1125


def test_update_progress_promotes_plan_to_reading(ex):
    seed(ex, "Dandadan", status="plan_to_read")
    result = ex.execute(UpdateProgress(title="Dandadan", chapters_read=5))
    assert result.data["status"] == "reading"


def test_update_unknown_series_lists_library(ex):
    seed(ex, "One Piece")
    result = ex.execute(UpdateProgress(title="Berserk", chapters_read=50))
    assert result.status == "error"
    assert "not in your library" in result.summary
    assert "One Piece" in result.summary


def test_fuzzy_title_match_when_unambiguous(ex):
    seed(ex, "Boruto: Two Blue Vortex", status="reading", chapters_read=18)
    result = ex.execute(UpdateProgress(title="Boruto", chapters_read=20))
    assert result.ok
    assert result.data["chapters_read"] == 20


def test_sequel_with_overlapping_title_can_be_added(ex):
    """Regression: duplicate detection must be exact, not substring, or a
    spin-off whose title contains an existing title gets rejected."""
    assert seed(ex, "Attack on Titan").ok
    assert seed(ex, "Attack on Titan: Before the Fall").ok
    titles = {r["title"] for r in ex.execute(QueryLibrary()).data}
    assert titles == {"Attack on Titan", "Attack on Titan: Before the Fall"}


def test_very_short_title_does_not_fuzzy_match(ex):
    seed(ex, "Kagurabachi")
    result = ex.execute(UpdateProgress(title="Ka", chapters_read=5))
    assert result.status == "error"


def test_fuzzy_match_refuses_when_ambiguous(ex):
    seed(ex, "Attack on Titan")
    seed(ex, "Attack on Titan: Before the Fall")
    result = ex.execute(UpdateProgress(title="Attack on Titan:", chapters_read=5))
    assert result.status == "error"


def test_set_status(ex):
    seed(ex, "Frieren", status="reading")
    result = ex.execute(SetStatus(title="Frieren", status="completed"))
    assert result.ok
    assert result.data["status"] == "completed"


def test_rate_series(ex):
    seed(ex, "Vinland Saga")
    result = ex.execute(RateSeries(title="Vinland Saga", rating=9.5))
    assert result.ok
    assert result.data["rating"] == 9.5


def test_remove_series(ex):
    seed(ex, "Black Clover")
    result = ex.execute(RemoveSeries(title="Black Clover"))
    assert result.ok
    assert ex.execute(QueryLibrary()).data == []


def test_query_library_filters_by_status(ex):
    seed(ex, "A", status="reading")
    seed(ex, "B", status="completed")
    seed(ex, "C", status="reading")
    result = ex.execute(QueryLibrary(status="reading"))
    assert {r["title"] for r in result.data} == {"A", "C"}


def test_query_library_behind_only(ex):
    seed(ex, "Behind", status="reading", chapters_read=10)
    seed(ex, "Current", status="reading", chapters_read=10)
    with ex.session.no_autoflush:
        ex.session.query(Series).filter(Series.title == "Behind").one().latest_chapter = 15
        ex.session.query(Series).filter(Series.title == "Current").one().latest_chapter = 10
    ex.session.flush()
    result = ex.execute(QueryLibrary(behind_only=True))
    assert [r["title"] for r in result.data] == ["Behind"]
    assert result.data[0]["chapters_behind"] == 5


def test_query_library_sorts_and_limits(ex):
    for title in ("Zeta", "Alpha", "Mu"):
        seed(ex, title)
    result = ex.execute(QueryLibrary(sort_by="title", limit=2))
    assert [r["title"] for r in result.data] == ["Alpha", "Mu"]


def test_min_rating_filter(ex):
    seed(ex, "Great")
    seed(ex, "Meh")
    ex.execute(RateSeries(title="Great", rating=9))
    ex.execute(RateSeries(title="Meh", rating=4))
    result = ex.execute(QueryLibrary(min_rating=8))
    assert [r["title"] for r in result.data] == ["Great"]


# -- API domain ------------------------------------------------------------- #


def test_search_manga(settings, user):
    api = FakeAPI(search=[{"title": "Dandadan", "year": 2021, "mangadex_id": "abc"}])
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=api)
        result = ex.execute(SearchManga(query="Dandadan"))
    assert result.ok
    assert result.data[0]["title"] == "Dandadan"


def test_search_handles_api_failure(settings, user):
    api = FakeAPI(fail=True)
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=api)
        result = ex.execute(SearchManga(query="Dandadan"))
    assert result.status == "error"
    assert "Could not reach" in result.summary


def test_check_new_chapters_single(settings, user):
    api = FakeAPI(latest={"One Piece": 1125.0})
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=api)
        seed(ex, "One Piece", status="reading", chapters_read=1120)
        result = ex.execute(CheckNewChapters(title="One Piece"))
    assert result.ok
    assert result.data[0]["chapters_behind"] == 5
    assert "Behind on" in result.summary


def test_check_new_chapters_all_reading(settings, user):
    api = FakeAPI(latest={"A": 10.0, "B": 20.0})
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=api)
        seed(ex, "A", status="reading", chapters_read=10)
        seed(ex, "B", status="reading", chapters_read=15)
        seed(ex, "C", status="plan_to_read")  # must be skipped
        result = ex.execute(CheckNewChapters())
    titles = {d["title"] for d in result.data}
    assert titles == {"A", "B"}
    assert "C" not in titles


def test_check_new_chapters_when_nothing_is_being_read(ex):
    seed(ex, "Queued", status="plan_to_read")
    result = ex.execute(CheckNewChapters())
    assert result.ok
    assert "Nothing is marked as currently reading" in result.summary


def test_check_reports_series_missing_from_mangadex(settings, user):
    api = FakeAPI(latest={})
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=api)
        seed(ex, "Obscure Doujin", status="reading")
        result = ex.execute(CheckNewChapters())
    assert "Could not check" in result.summary
    assert "not found on MangaDex" in result.summary


def test_caught_up_clears_behind_count(settings, user):
    api = FakeAPI(latest={"One Piece": 1125.0})
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=api)
        seed(ex, "One Piece", status="reading", chapters_read=1120)
        ex.execute(CheckNewChapters(title="One Piece"))
        result = ex.execute(UpdateProgress(title="One Piece", chapters_read=1125))
    assert result.data["chapters_behind"] == 0


def test_recommendations_exclude_own_library(settings, user):
    api = FakeAPI(
        recs=[
            {"title": "Vinland Saga", "genres": [], "chapters": 211, "status": "finished", "score": 88, "description": ""},
            {"title": "Kingdom", "genres": [], "chapters": 700, "status": "releasing", "score": 90, "description": ""},
        ]
    )
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=api)
        seed(ex, "Vinland Saga", status="completed")
        result = ex.execute(GetRecommendations(limit=5))
    assert [r["title"] for r in result.data] == ["Kingdom"]


def test_recommendations_empty_is_not_an_error(settings, user):
    api = FakeAPI(recs=[])
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=api)
        result = ex.execute(GetRecommendations())
    assert result.ok
    assert "No recommendations" in result.summary


# -- file domain ------------------------------------------------------------ #


def test_export_csv(ex):
    seed(ex, "One Piece", status="reading", chapters_read=1120)
    result = ex.execute(ExportList(format="csv"))
    assert result.ok
    assert result.download["filename"] == "manga-list.csv"
    rows = list(csv.DictReader(io.StringIO(result.download["content"])))
    assert rows[0]["title"] == "One Piece"
    assert rows[0]["chapters_read"] == "1120.0"


def test_export_json(ex):
    seed(ex, "Frieren", chapters_read=130)
    result = ex.execute(ExportList(format="json"))
    data = json.loads(result.download["content"])
    assert data[0]["title"] == "Frieren"


def test_export_markdown(ex):
    seed(ex, "Dandadan")
    result = ex.execute(ExportList(format="markdown"))
    assert "| Title |" in result.download["content"]
    assert "Dandadan" in result.download["content"]


def test_export_filtered_by_status(ex):
    seed(ex, "Reading one", status="reading")
    seed(ex, "Done one", status="completed")
    result = ex.execute(ExportList(format="csv", status="completed"))
    assert "Done one" in result.download["content"]
    assert "Reading one" not in result.download["content"]
    assert result.download["filename"] == "manga-list-completed.csv"


def test_export_empty_library_is_an_error(ex):
    result = ex.execute(ExportList(format="csv"))
    assert result.status == "error"
    assert "empty" in result.summary


def test_import_without_upload_is_an_error(ex):
    result = ex.execute(ImportList())
    assert result.status == "error"
    assert "No CSV has been uploaded" in result.summary


def test_import_adds_rows(settings, user):
    csv_text = (
        "title,status,chapters_read,rating\n"
        "Chainsaw Man,reading,190,9\n"
        "Berserk,on hold,374,10\n"
    )
    with session_scope() as session:
        ex = Executor(
            session, user.user_id, settings=settings, api=FakeAPI(),
            upload_provider=lambda: csv_text,
        )
        result = ex.execute(ImportList())
        titles = {r["title"] for r in ex.execute(QueryLibrary()).data}
    assert result.ok
    assert result.data["added"] == 2
    assert titles == {"Chainsaw Man", "Berserk"}


def test_import_skips_duplicates_by_default(settings, user):
    csv_text = "title,chapters_read\nOne Piece,999\n"
    with session_scope() as session:
        ex = Executor(
            session, user.user_id, settings=settings, api=FakeAPI(),
            upload_provider=lambda: csv_text,
        )
        seed(ex, "One Piece", chapters_read=1120)
        result = ex.execute(ImportList(overwrite=False))
        row = ex.execute(QueryLibrary()).data[0]
    assert result.data["skipped"] == 1
    assert row["chapters_read"] == 1120


def test_import_overwrites_when_asked(settings, user):
    csv_text = "title,chapters_read\nOne Piece,1125\n"
    with session_scope() as session:
        ex = Executor(
            session, user.user_id, settings=settings, api=FakeAPI(),
            upload_provider=lambda: csv_text,
        )
        seed(ex, "One Piece", chapters_read=1120)
        result = ex.execute(ImportList(overwrite=True))
        row = ex.execute(QueryLibrary()).data[0]
    assert result.data["updated"] == 1
    assert row["chapters_read"] == 1125


def test_import_sanitises_bad_values(settings, user):
    csv_text = (
        "title,status,chapters_read,rating\n"
        "Good,reading,10,8\n"
        ",reading,5,5\n"
        "Bad Status,nonsense,abc,99\n"
    )
    with session_scope() as session:
        ex = Executor(
            session, user.user_id, settings=settings, api=FakeAPI(),
            upload_provider=lambda: csv_text,
        )
        result = ex.execute(ImportList())
        rows = {r["title"]: r for r in ex.execute(QueryLibrary()).data}
    assert "blank title" in result.summary
    assert rows["Bad Status"]["status"] == "plan_to_read"
    assert rows["Bad Status"]["chapters_read"] == 0.0
    assert rows["Bad Status"]["rating"] == 10.0  # clamped from 99


def test_import_rejects_csv_without_title_column(settings, user):
    with session_scope() as session:
        ex = Executor(
            session, user.user_id, settings=settings, api=FakeAPI(),
            upload_provider=lambda: "name,chapters\nfoo,1\n",
        )
        result = ex.execute(ImportList())
    assert result.status == "error"
    assert "'title' column" in result.summary


# -- plan execution + logging ----------------------------------------------- #


def test_execute_plan_logs_every_action(settings, user):
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=FakeAPI())
        report = ex.execute_plan(
            [AddSeries(title="A"), AddSeries(title="A")], "add A twice"
        )
        logs = recent_actions(session, user.user_id)

    assert report.results[0].ok
    assert report.results[1].status == "error"
    assert report.any_failed
    assert len(logs) == 2
    assert {log.status for log in logs} == {"ok", "error"}
    assert all(log.user_request == "add A twice" for log in logs)


def test_destructive_action_is_held_until_confirmed(settings, user):
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=FakeAPI())
        seed(ex, "Doomed")
        action = RemoveSeries(title="Doomed")
        action.requires_confirmation = True

        held = ex.execute_plan([action], "delete it", confirmed=False)
        still_there = ex.execute(QueryLibrary()).data

        gone = ex.execute_plan([action], "delete it", confirmed=True)
        after = ex.execute(QueryLibrary()).data

    assert held.results[0].status == "cancelled"
    assert len(still_there) == 1
    assert gone.results[0].ok
    assert after == []


def test_confirmed_flag_is_recorded_in_the_log(settings, user):
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=FakeAPI())
        seed(ex, "Doomed")
        action = RemoveSeries(title="Doomed")
        action.requires_confirmation = True
        ex.execute_plan([action], "delete it", confirmed=True)
        logs = recent_actions(session, user.user_id)

    removal = [log for log in logs if log.action_name == "remove_series"][0]
    assert removal.confirmed_by_user == 1


def test_one_failing_action_does_not_abort_the_rest(settings, user):
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=FakeAPI())
        report = ex.execute_plan(
            [
                AddSeries(title="First"),
                UpdateProgress(title="Nonexistent", chapters_read=5),
                AddSeries(title="Third"),
            ],
            "mixed plan",
        )
        titles = {r["title"] for r in ex.execute(QueryLibrary()).data}

    assert [r.status for r in report.results] == ["ok", "error", "ok"]
    assert titles == {"First", "Third"}


def test_unhandled_exception_is_caught(settings, user, monkeypatch):
    with session_scope() as session:
        ex = Executor(session, user.user_id, settings=settings, api=FakeAPI())

        def boom(action):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(ex, "_do_query_library", boom)
        result = ex.execute(QueryLibrary())

    assert result.status == "error"
    assert "kaboom" in result.summary
