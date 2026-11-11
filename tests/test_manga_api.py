"""MangaDex + AniList clients, driven by canned HTTP responses.

The sandbox has no outbound access to these hosts, so a fake requests.Session
replays response shapes taken from the published API schemas. This exercises
the parsing, throttling, caching and retry logic without the network.
"""

from __future__ import annotations

import pytest
import requests

from agent.config import Settings
from agent.manga_api import MangaAPI, MangaAPIError, RateLimiter, TTLCache


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self):
        if self._payload is _MALFORMED:
            raise ValueError("not json")
        return self._payload


_MALFORMED = object()


class FakeSession:
    """Replays a queue of responses (or raises queued exceptions)."""

    def __init__(self, queue):
        self.queue = list(queue)
        self.requests: list[dict] = []
        self.headers: dict = {}

    def request(self, method, url, params=None, json=None, timeout=None):
        self.requests.append(
            {"method": method, "url": url, "params": params, "json": json}
        )
        if not self.queue:
            raise AssertionError(f"unexpected extra request to {url}")
        nxt = self.queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.fixture
def fast_settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path/'x.db'}",
        mangadex_min_interval=0.0,
        anilist_min_interval=0.0,
        api_cache_ttl_seconds=60,
    )


def api_with(fast_settings, queue):
    return MangaAPI(fast_settings, session=FakeSession(queue))


# -- building blocks -------------------------------------------------------- #


def test_rate_limiter_waits_between_calls():
    slept: list[float] = []
    limiter = RateLimiter(0.5)
    limiter.wait(sleep=slept.append)  # first call is free
    limiter.wait(sleep=slept.append)  # second must wait out the interval
    assert len(slept) == 1
    assert 0 < slept[0] <= 0.5


def test_rate_limiter_does_not_wait_when_interval_passed(monkeypatch):
    slept: list[float] = []
    limiter = RateLimiter(0.5)
    limiter.wait(sleep=slept.append)
    # Pretend a second has gone by.
    limiter._last -= 1.0
    limiter.wait(sleep=slept.append)
    assert slept == []


def test_ttl_cache_expires():
    cache = TTLCache(ttl=-1)  # already expired
    cache.set("k", "v")
    assert cache.get("k") is None


def test_ttl_cache_hits_within_ttl():
    cache = TTLCache(ttl=60)
    cache.set("k", "v")
    assert cache.get("k") == "v"


# -- MangaDex search -------------------------------------------------------- #

SEARCH_PAYLOAD = {
    "data": [
        {
            "id": "68112dc1-2b80-4f20-beb8-2f2a8716a4a5",
            "attributes": {
                "title": {"en": "Dandadan"},
                "altTitles": [{"ja": "ダンダダン"}],
                "description": {"en": "A boy who believes in aliens." * 20},
                "year": 2021,
                "status": "ongoing",
                "lastChapter": None,
            },
        }
    ]
}


def test_search_parses_results(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, SEARCH_PAYLOAD)])
    results = api.search_manga("Dandadan")
    assert results[0]["title"] == "Dandadan"
    assert results[0]["year"] == 2021
    assert results[0]["mangadex_id"].startswith("68112dc1")
    assert len(results[0]["description"]) <= 300


def test_search_sends_correct_params(fast_settings):
    session = FakeSession([FakeResponse(200, SEARCH_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.search_manga("Dandadan", limit=3)
    params = session.requests[0]["params"]
    assert params["title"] == "Dandadan"
    assert params["limit"] == 3
    assert "safe" in params["contentRating[]"]


def test_search_sets_user_agent(fast_settings):
    session = FakeSession([FakeResponse(200, SEARCH_PAYLOAD)])
    MangaAPI(fast_settings, session=session)
    assert "manga-tracker" in session.headers["User-Agent"]


def test_search_falls_back_to_non_english_title(fast_settings):
    payload = {
        "data": [
            {
                "id": "abc",
                "attributes": {
                    "title": {"ja-ro": "Kagurabachi"},
                    "altTitles": [],
                    "description": {},
                    "year": 2023,
                    "status": "ongoing",
                },
            }
        ]
    }
    api = api_with(fast_settings, [FakeResponse(200, payload)])
    assert api.search_manga("kagura")[0]["title"] == "Kagurabachi"


def test_search_handles_empty_results(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, {"data": []})])
    assert api.search_manga("zzzznotreal") == []


def test_search_result_is_cached(fast_settings):
    # Only one response queued; a second call must come from cache.
    api = api_with(fast_settings, [FakeResponse(200, SEARCH_PAYLOAD)])
    first = api.search_manga("Dandadan")
    second = api.search_manga("Dandadan")
    assert first == second


# -- MangaDex chapter feed -------------------------------------------------- #

FEED_PAYLOAD = {
    "data": [
        {"id": "c1", "attributes": {"chapter": "145"}},
        {"id": "c2", "attributes": {"chapter": "144.5"}},
        {"id": "c3", "attributes": {"chapter": None}},
        {"id": "c4", "attributes": {"chapter": "oneshot"}},
    ]
}


def test_latest_chapter_picks_highest_number(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, FEED_PAYLOAD)])
    assert api.latest_chapter("md-id") == 145.0


def test_latest_chapter_handles_decimals_only(fast_settings):
    payload = {"data": [{"attributes": {"chapter": "1045.5"}}]}
    api = api_with(fast_settings, [FakeResponse(200, payload)])
    assert api.latest_chapter("md-id") == 1045.5


def test_latest_chapter_none_when_no_chapters_anywhere(fast_settings):
    api = api_with(
        fast_settings,
        [
            FakeResponse(200, {"data": []}),  # empty feed
            FakeResponse(200, {"data": {"attributes": {"lastChapter": None}}}),
        ],
    )
    assert api.latest_chapter("md-id") is None


def test_empty_feed_falls_back_to_last_chapter_attribute(fast_settings):
    """Licensed series list no chapters on MangaDex, but the series record
    still carries a lastChapter value worth using."""
    api = api_with(
        fast_settings,
        [
            FakeResponse(200, {"data": []}),
            FakeResponse(200, {"data": {"attributes": {"lastChapter": "215"}}}),
        ],
    )
    assert api.latest_chapter("md-id") == 215.0


def test_fallback_failure_returns_none_rather_than_raising(fast_settings, monkeypatch):
    monkeypatch.setattr("agent.manga_api.time.sleep", lambda s: None)
    api = api_with(
        fast_settings,
        [FakeResponse(200, {"data": []})] + [FakeResponse(503, {})] * 3,
    )
    assert api.latest_chapter("md-id") is None


def test_feed_does_not_exclude_externally_hosted_chapters(fast_settings):
    """Regression: includeExternalUrl=0 emptied the feed for licensed series
    whose chapters are external links (MangaPlus and similar)."""
    session = FakeSession([FakeResponse(200, FEED_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.latest_chapter("md-id")
    params = session.requests[0]["params"]
    assert "includeExternalUrl" not in params
    assert "contentRating[]" not in params


def test_feed_requests_a_large_page(fast_settings):
    """order[chapter] sorts as a string, so one page of results is scanned
    for the true maximum rather than trusting the first entry."""
    session = FakeSession([FakeResponse(200, FEED_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.latest_chapter("md-id")
    assert session.requests[0]["params"]["limit"] >= 100


def test_string_sorted_chapters_still_yield_the_true_maximum(fast_settings):
    payload = {
        "data": [
            {"attributes": {"chapter": "99"}},
            {"attributes": {"chapter": "100"}},
            {"attributes": {"chapter": "9"}},
        ]
    }
    api = api_with(fast_settings, [FakeResponse(200, payload)])
    assert api.latest_chapter("md-id") == 100.0


def test_feed_requests_english_descending(fast_settings):
    session = FakeSession([FakeResponse(200, FEED_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.latest_chapter("md-id")
    params = session.requests[0]["params"]
    assert params["translatedLanguage[]"] == ["en"]
    assert params["order[chapter]"] == "desc"


def test_resolve_series_prefers_exact_title(fast_settings):
    payload = {
        "data": [
            {"id": "wrong", "attributes": {"title": {"en": "Dandadan Spinoff"},
                                           "altTitles": [], "description": {}}},
            {"id": "right", "attributes": {"title": {"en": "Dandadan"},
                                           "altTitles": [], "description": {}}},
        ]
    }
    api = api_with(fast_settings, [FakeResponse(200, payload)])
    assert api.resolve_series("Dandadan")["mangadex_id"] == "right"


def test_resolve_series_none_when_no_match(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, {"data": []})])
    assert api.resolve_series("Nonexistent") is None


# -- error handling --------------------------------------------------------- #


def test_429_is_retried_honouring_retry_after(fast_settings, monkeypatch):
    monkeypatch.setattr("agent.manga_api.time.sleep", lambda s: None)
    api = api_with(
        fast_settings,
        [
            FakeResponse(429, {}, {"Retry-After": "1"}),
            FakeResponse(200, SEARCH_PAYLOAD),
        ],
    )
    assert api.search_manga("Dandadan")[0]["title"] == "Dandadan"


def test_repeated_429_raises_a_clear_error(fast_settings, monkeypatch):
    monkeypatch.setattr("agent.manga_api.time.sleep", lambda s: None)
    api = api_with(fast_settings, [FakeResponse(429, {}) for _ in range(3)])
    with pytest.raises(MangaAPIError) as err:
        api.search_manga("Dandadan")
    assert "rate limiting" in str(err.value)


def test_500_is_retried(fast_settings, monkeypatch):
    monkeypatch.setattr("agent.manga_api.time.sleep", lambda s: None)
    api = api_with(
        fast_settings, [FakeResponse(503, {}), FakeResponse(200, SEARCH_PAYLOAD)]
    )
    assert api.search_manga("Dandadan")


def test_404_fails_immediately(fast_settings):
    api = api_with(fast_settings, [FakeResponse(404, {})])
    with pytest.raises(MangaAPIError) as err:
        api.latest_chapter("missing")
    assert "404" in str(err.value)


def test_timeout_is_retried_then_reported(fast_settings, monkeypatch):
    monkeypatch.setattr("agent.manga_api.time.sleep", lambda s: None)
    api = api_with(fast_settings, [requests.Timeout() for _ in range(3)])
    with pytest.raises(MangaAPIError) as err:
        api.search_manga("x")
    assert "timed out" in str(err.value)


def test_network_error_is_reported_cleanly(fast_settings, monkeypatch):
    monkeypatch.setattr("agent.manga_api.time.sleep", lambda s: None)
    api = api_with(
        fast_settings, [requests.ConnectionError("proxy refused") for _ in range(3)]
    )
    with pytest.raises(MangaAPIError) as err:
        api.search_manga("x")
    assert "Could not reach api.mangadex.org" in str(err.value)


def test_malformed_json_is_reported(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, _MALFORMED)])
    with pytest.raises(MangaAPIError) as err:
        api.search_manga("x")
    assert "malformed" in str(err.value)


# -- AniList ---------------------------------------------------------------- #

GENRE_PAYLOAD = {
    "data": {
        "Page": {
            "media": [
                {
                    "title": {"english": "Vinland Saga", "romaji": "Vinland Saga"},
                    "genres": ["Action", "Drama"],
                    "chapters": 211,
                    "status": "FINISHED",
                    "averageScore": 88,
                    "description": "Viking epic.",
                },
                {
                    "title": {"english": None, "romaji": "Kingdom"},
                    "genres": ["Action"],
                    "chapters": None,
                    "status": "RELEASING",
                    "averageScore": 90,
                    "description": "Warring states.",
                },
            ]
        }
    }
}


def test_recommendations_by_genre(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, GENRE_PAYLOAD)])
    results = api.recommendations(genres=["action"], limit=5)
    assert {r["title"] for r in results} == {"Vinland Saga", "Kingdom"}
    assert results[0]["status"] == "finished"


def test_recommendations_fall_back_to_romaji(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, GENRE_PAYLOAD)])
    titles = [r["title"] for r in api.recommendations()]
    assert "Kingdom" in titles


def test_recommendations_finished_only_filter(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, GENRE_PAYLOAD)])
    results = api.recommendations(finished_only=True)
    assert [r["title"] for r in results] == ["Vinland Saga"]


def test_recommendations_max_chapters_filter(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, GENRE_PAYLOAD)])
    results = api.recommendations(max_chapters=150)
    # Vinland Saga is 211 chapters so it is excluded; Kingdom has unknown
    # length and is kept.
    assert [r["title"] for r in results] == ["Kingdom"]


def test_recommendations_exclude_titles(fast_settings):
    api = api_with(fast_settings, [FakeResponse(200, GENRE_PAYLOAD)])
    results = api.recommendations(exclude_titles={"vinland saga"})
    assert [r["title"] for r in results] == ["Kingdom"]


def test_similar_to_uses_recommendation_nodes(fast_settings):
    payload = {
        "data": {
            "Media": {
                "title": {"english": "Vinland Saga"},
                "recommendations": {
                    "nodes": [
                        {
                            "mediaRecommendation": {
                                "title": {"english": "Kingdom"},
                                "genres": ["Action"],
                                "chapters": 700,
                                "status": "RELEASING",
                                "averageScore": 90,
                                "description": "d",
                            }
                        },
                        {"mediaRecommendation": None},
                    ]
                },
            }
        }
    }
    api = api_with(fast_settings, [FakeResponse(200, payload)])
    results = api.recommendations(similar_to="Vinland Saga")
    assert [r["title"] for r in results] == ["Kingdom"]


def test_graphql_errors_are_surfaced(fast_settings):
    api = api_with(
        fast_settings,
        [FakeResponse(200, {"errors": [{"message": "Unknown field 'nope'"}]})],
    )
    with pytest.raises(MangaAPIError) as err:
        api.recommendations(genres=["action"])
    assert "AniList rejected" in str(err.value)


def test_anilist_uses_post(fast_settings):
    session = FakeSession([FakeResponse(200, GENRE_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.recommendations(genres=["action"])
    assert session.requests[0]["method"] == "POST"
    assert "graphql.anilist.co" in session.requests[0]["url"]


# -- AniList filter construction (regression) ------------------------------- #


def test_unused_filters_are_absent_from_the_query(fast_settings):
    """Regression: AniList answered 500 when chapters_lesser and status_in
    arrived as explicit nulls."""
    session = FakeSession([FakeResponse(200, GENRE_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.recommendations(genres=["action"])

    body = session.requests[0]["json"]
    assert "chapters_lesser" not in body["query"]
    assert "status_in" not in body["query"]
    assert "genre_in" in body["query"]
    assert None not in body["variables"].values()
    assert set(body["variables"]) == {"perPage", "genres"}


def test_no_filters_at_all_sends_only_perpage(fast_settings):
    session = FakeSession([FakeResponse(200, GENRE_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.recommendations()

    body = session.requests[0]["json"]
    assert set(body["variables"]) == {"perPage"}
    assert "genre_in" not in body["query"]


def test_all_filters_are_declared_when_used(fast_settings):
    session = FakeSession([FakeResponse(200, GENRE_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.recommendations(genres=["action"], max_chapters=150, finished_only=True)

    body = session.requests[0]["json"]
    for fragment in ("genre_in", "chapters_lesser", "status_in"):
        assert fragment in body["query"]
    assert body["variables"]["maxChapters"] == 150
    assert body["variables"]["statusIn"] == ["FINISHED"]
    assert None not in body["variables"].values()


def test_blank_genre_strings_are_dropped(fast_settings):
    session = FakeSession([FakeResponse(200, GENRE_PAYLOAD)])
    api = MangaAPI(fast_settings, session=session)
    api.recommendations(genres=["  ", ""])

    body = session.requests[0]["json"]
    assert "genre_in" not in body["query"]


def test_every_declared_variable_is_used_in_the_query(fast_settings):
    """A declared-but-unused GraphQL variable is itself a query error."""
    for kwargs in (
        {},
        {"genres": ["action"]},
        {"max_chapters": 100},
        {"finished_only": True},
        {"genres": ["action"], "max_chapters": 100, "finished_only": True},
    ):
        session = FakeSession([FakeResponse(200, GENRE_PAYLOAD)])
        MangaAPI(fast_settings, session=session).recommendations(**kwargs)
        body = session.requests[0]["json"]
        declared = body["query"].split(")")[0]
        for name in body["variables"]:
            assert f"${name}" in declared, f"{name} not declared for {kwargs}"


# -- error reporting -------------------------------------------------------- #


def test_500_error_includes_the_servers_own_message(fast_settings, monkeypatch):
    monkeypatch.setattr("agent.manga_api.time.sleep", lambda s: None)
    api = api_with(
        fast_settings,
        [FakeResponse(500, {"errors": [{"message": "Internal Server Error"}]})] * 3,
    )
    with pytest.raises(MangaAPIError) as err:
        api.recommendations(genres=["action"])
    assert "Internal Server Error" in str(err.value)


def test_500_without_a_body_still_reports_the_status(fast_settings, monkeypatch):
    monkeypatch.setattr("agent.manga_api.time.sleep", lambda s: None)
    api = api_with(fast_settings, [FakeResponse(500, {})] * 3)
    with pytest.raises(MangaAPIError) as err:
        api.recommendations(genres=["action"])
    assert "500" in str(err.value)
