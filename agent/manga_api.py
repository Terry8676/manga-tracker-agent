"""External manga APIs: MangaDex (search, chapter feed) and AniList
(recommendations).

Both are public and need no key. Both are also shared community services, so
this module is deliberately polite:
  * a token-bucket style minimum interval between calls per host
    (MangaDex allows ~5 req/sec; AniList is currently capped at 30 req/min),
  * an in-process TTL cache so repeated questions don't re-hit the API,
  * a descriptive User-Agent,
  * ``Retry-After`` honoured on 429 instead of hammering.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import Settings, get_settings

MANGADEX_BASE = "https://api.mangadex.org"
ANILIST_URL = "https://graphql.anilist.co"


class MangaAPIError(Exception):
    """Network or API failure, phrased for the end user."""


# --------------------------------------------------------------------------- #
# Throttling + caching
# --------------------------------------------------------------------------- #


class RateLimiter:
    """Enforces a minimum interval between calls, thread-safe."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self, sleep=time.sleep) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self.min_interval:
                sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


@dataclass
class TTLCache:
    ttl: float
    _store: dict[str, tuple[float, Any]] = field(default_factory=dict)

    def get(self, key: str) -> Any | None:
        hit = self._store.get(key)
        if hit is None:
            return None
        stored_at, value = hit
        if time.monotonic() - stored_at > self.ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def _error_detail(response: Any) -> str:
    """Extract a short server-side error message, if the body carries one.

    A bare status code makes a failing GraphQL filter indistinguishable from
    an outage, so the body's own message is worth surfacing.
    """
    try:
        body = response.json()
    except Exception:
        return ""
    if not isinstance(body, dict):
        return ""
    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        message = first.get("message") if isinstance(first, dict) else first
        if message:
            return f" ({str(message)[:200]})"
    for key in ("message", "detail", "title"):
        if body.get(key):
            return f" ({str(body[key])[:200]})"
    return ""


def _parse_chapter(value: Any) -> float | None:
    """MangaDex chapter numbers arrive as strings, sometimes '1045.5' or None."""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


class MangaAPI:
    """MangaDex + AniList access with shared throttling and caching."""

    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None):
        self.settings = settings or get_settings()
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": self.settings.user_agent, "Accept": "application/json"}
        )
        self._md_limiter = RateLimiter(self.settings.mangadex_min_interval)
        self._al_limiter = RateLimiter(self.settings.anilist_min_interval)
        self.cache = TTLCache(self.settings.api_cache_ttl_seconds)

    # -- low level ---------------------------------------------------------- #

    def _request(
        self,
        method: str,
        url: str,
        *,
        limiter: RateLimiter,
        params: dict | None = None,
        json_body: dict | None = None,
        attempts: int = 3,
    ) -> dict:
        last_error = "Unknown error"
        for attempt in range(1, attempts + 1):
            limiter.wait()
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    timeout=self.settings.http_timeout_seconds,
                )
            except requests.Timeout:
                last_error = "the request timed out"
            except requests.RequestException as exc:
                last_error = f"network error ({type(exc).__name__})"
            else:
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if (retry_after or "").isdigit() else 2.0 * attempt
                    last_error = "the service is rate limiting us"
                    if attempt < attempts:
                        time.sleep(min(delay, 10.0))
                        continue
                elif 500 <= response.status_code < 600:
                    last_error = (
                        f"the service returned {response.status_code}"
                        f"{_error_detail(response)}"
                    )
                    if attempt < attempts:
                        time.sleep(1.0 * attempt)
                        continue
                elif response.status_code >= 400:
                    raise MangaAPIError(
                        f"{url.split('/')[2]} returned {response.status_code}."
                    )
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise MangaAPIError("The API returned a malformed response.") from exc
            if attempt < attempts:
                time.sleep(0.5 * attempt)
        raise MangaAPIError(f"Could not reach {url.split('/')[2]}: {last_error}.")

    # -- MangaDex ----------------------------------------------------------- #

    def search_manga(self, query: str, limit: int = 5) -> list[dict]:
        """Search MangaDex by title. Returns id/title/year/status/description."""
        cache_key = f"md:search:{query.lower()}:{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        payload = self._request(
            "GET",
            f"{MANGADEX_BASE}/manga",
            limiter=self._md_limiter,
            params={
                "title": query,
                "limit": limit,
                "order[relevance]": "desc",
                "contentRating[]": ["safe", "suggestive"],
            },
        )

        results = []
        for item in payload.get("data", []):
            attrs = item.get("attributes", {}) or {}
            titles = attrs.get("title", {}) or {}
            alts = attrs.get("altTitles", []) or []
            title = titles.get("en") or next(iter(titles.values()), None)
            if not title:
                for alt in alts:
                    if isinstance(alt, dict) and alt.get("en"):
                        title = alt["en"]
                        break
            descriptions = attrs.get("description", {}) or {}
            description = (descriptions.get("en") or "").strip()
            results.append(
                {
                    "mangadex_id": item.get("id"),
                    "title": title or "(untitled)",
                    "year": attrs.get("year"),
                    "publication_status": attrs.get("status"),
                    "last_chapter": _parse_chapter(attrs.get("lastChapter")),
                    "description": description[:300],
                }
            )

        self.cache.set(cache_key, results)
        return results

    def latest_chapter(self, mangadex_id: str, language: str = "en") -> float | None:
        """Highest translated chapter number currently on MangaDex.

        Two deliberate choices here, both learned from the feed coming back
        empty for popular series:

        * ``includeExternalUrl`` is NOT set to 0. Licensed series (Shueisha
          titles serialised on MangaPlus, for example) list their chapters on
          MangaDex as external links rather than hosted pages. Excluding
          those emptied the feed for exactly the series people are most
          likely to be reading.
        * No ``contentRating`` filter. The series is already in the user's
          library, so filtering it out of its own chapter list is pointless
          and only creates another way to get no answer.

        A high ``limit`` is used because ``order[chapter]`` sorts chapter
        numbers as strings, so "99" can outrank "100"; the maximum is taken
        over the page rather than trusting its first element.
        """
        cache_key = f"md:latest:{mangadex_id}:{language}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        payload = self._request(
            "GET",
            f"{MANGADEX_BASE}/manga/{mangadex_id}/feed",
            limiter=self._md_limiter,
            params={
                "translatedLanguage[]": [language],
                "order[chapter]": "desc",
                "limit": 100,
            },
        )

        best: float | None = None
        for item in payload.get("data", []):
            number = _parse_chapter((item.get("attributes") or {}).get("chapter"))
            if number is not None and (best is None or number > best):
                best = number

        if best is None:
            best = self._last_chapter_attribute(mangadex_id)

        self.cache.set(cache_key, best)
        return best

    def _last_chapter_attribute(self, mangadex_id: str) -> float | None:
        """Fallback: the series' own ``lastChapter`` field.

        Used when the chapter feed yields nothing, which happens for series
        whose chapters are not distributed through MangaDex at all. Returns
        None rather than raising, so a failed fallback leaves the caller with
        "unknown" instead of an error.
        """
        try:
            payload = self._request(
                "GET",
                f"{MANGADEX_BASE}/manga/{mangadex_id}",
                limiter=self._md_limiter,
            )
        except MangaAPIError:
            return None
        attributes = ((payload.get("data") or {}).get("attributes")) or {}
        return _parse_chapter(attributes.get("lastChapter"))

    def resolve_series(self, title: str) -> dict | None:
        """Best-guess MangaDex match for a title already in the library."""
        matches = self.search_manga(title, limit=5)
        if not matches:
            return None
        wanted = title.strip().lower()
        for match in matches:
            if match["title"].strip().lower() == wanted:
                return match
        return matches[0]

    # -- AniList ------------------------------------------------------------ #

    _MEDIA_FIELDS = (
        "          title { romaji english }\n"
        "          genres\n"
        "          chapters\n"
        "          status\n"
        "          averageScore\n"
        "          description(asHtml: false)\n"
    )

    @classmethod
    def _build_rec_query(
        cls, *, genres: bool = False, max_chapters: bool = False, status: bool = False
    ) -> str:
        """Build the recommendation query using only the filters in play.

        AniList's resolver answers with a 500 when a filter argument arrives
        as an explicit null, so an unused filter is left out of the query
        text altogether instead of being declared and passed as null.
        """
        var_defs = ["$perPage: Int"]
        filters = ["type: MANGA", "sort: SCORE_DESC", "isAdult: false"]
        if genres:
            var_defs.append("$genres: [String]")
            filters.append("genre_in: $genres")
        if max_chapters:
            var_defs.append("$maxChapters: Int")
            filters.append("chapters_lesser: $maxChapters")
        if status:
            var_defs.append("$statusIn: [MediaStatus]")
            filters.append("status_in: $statusIn")
        return (
            "query (" + ", ".join(var_defs) + ") {\n"
            "  Page(page: 1, perPage: $perPage) {\n"
            "    media(" + ", ".join(filters) + ") {\n"
            + cls._MEDIA_FIELDS
            + "    }\n"
            "  }\n"
            "}\n"
        )

    _SIMILAR_QUERY = """
    query ($search: String, $perPage: Int) {
      Media(search: $search, type: MANGA, isAdult: false) {
        title { romaji english }
        recommendations(sort: RATING_DESC, perPage: $perPage) {
          nodes {
            mediaRecommendation {
              title { romaji english }
              genres
              chapters
              status
              averageScore
              description(asHtml: false)
            }
          }
        }
      }
    }
    """

    def _graphql(self, query: str, variables: dict) -> dict:
        payload = self._request(
            "POST",
            ANILIST_URL,
            limiter=self._al_limiter,
            json_body={"query": query, "variables": variables},
        )
        if payload.get("errors"):
            first = payload["errors"][0].get("message", "unknown error")
            raise MangaAPIError(f"AniList rejected the query: {first}")
        return payload.get("data") or {}

    @staticmethod
    def _shape_media(media: dict) -> dict:
        titles = media.get("title", {}) or {}
        description = (media.get("description") or "").strip()
        return {
            "title": titles.get("english") or titles.get("romaji") or "(untitled)",
            "genres": media.get("genres") or [],
            "chapters": media.get("chapters"),
            "status": (media.get("status") or "").lower().replace("_", " "),
            "score": media.get("averageScore"),
            "description": description[:300],
        }

    def recommendations(
        self,
        *,
        genres: list[str] | None = None,
        similar_to: str | None = None,
        finished_only: bool = False,
        max_chapters: int | None = None,
        limit: int = 5,
        exclude_titles: set[str] | None = None,
    ) -> list[dict]:
        """Recommend manga, either similar to one title or filtered by genre."""
        exclude = {t.strip().lower() for t in (exclude_titles or set())}
        cache_key = (
            f"al:rec:{similar_to or ''}:{','.join(sorted(genres or []))}:"
            f"{finished_only}:{max_chapters}:{limit}"
        )
        cached = self.cache.get(cache_key)
        if cached is None:
            if similar_to:
                data = self._graphql(
                    self._SIMILAR_QUERY,
                    {"search": similar_to, "perPage": max(limit * 3, 10)},
                )
                media = data.get("Media") or {}
                nodes = ((media.get("recommendations") or {}).get("nodes")) or []
                raw = [
                    self._shape_media(n["mediaRecommendation"])
                    for n in nodes
                    if n.get("mediaRecommendation")
                ]
            else:
                genre_list = [
                    g.strip().title() for g in (genres or []) if g and g.strip()
                ]
                variables: dict = {"perPage": max(limit * 3, 10)}
                if genre_list:
                    variables["genres"] = genre_list
                if max_chapters is not None:
                    variables["maxChapters"] = int(max_chapters)
                if finished_only:
                    variables["statusIn"] = ["FINISHED"]
                query = self._build_rec_query(
                    genres=bool(genre_list),
                    max_chapters=max_chapters is not None,
                    status=finished_only,
                )
                data = self._graphql(query, variables)
                raw = [
                    self._shape_media(m)
                    for m in ((data.get("Page") or {}).get("media") or [])
                ]
            self.cache.set(cache_key, raw)
            cached = raw

        # Filtering happens outside the cache so a user's own library doesn't
        # pollute a shared cache entry.
        filtered = [r for r in cached if r["title"].strip().lower() not in exclude]
        if finished_only:
            filtered = [r for r in filtered if r["status"] == "finished"]
        if max_chapters is not None:
            filtered = [
                r for r in filtered if r["chapters"] is None or r["chapters"] <= max_chapters
            ]
        return filtered[:limit]
