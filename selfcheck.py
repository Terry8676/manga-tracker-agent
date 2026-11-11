"""Pre-flight check: verifies the database, the LLM key and both manga APIs.

Run this once after setting your secrets, before launching the app:

    python selfcheck.py

Every check is independent, so a failure tells you exactly which credential or
service is the problem.
"""

from __future__ import annotations

import sys

from agent.config import get_settings

PASS = "  [ok]  "
FAIL = "  [FAIL]"
WARN = "  [warn]"


def check_database(settings) -> bool:
    from sqlalchemy import text

    from agent.db import init_db

    kind = "Postgres" if settings.is_postgres else "SQLite"
    try:
        engine = init_db(settings, force=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"{PASS} Database ({kind}) reachable, tables ready.")
        if not settings.is_postgres:
            print(
                f"{WARN} Using a local SQLite file. On Streamlit Community Cloud "
                "this is wiped on restart - set DATABASE_URL to your Neon "
                "connection string before deploying."
            )
        return True
    except Exception as exc:
        print(f"{FAIL} Database ({kind}): {type(exc).__name__}: {exc}")
        print("        Check DATABASE_URL in .streamlit/secrets.toml.")
        return False


def check_llm(settings) -> bool:
    import json

    from agent.llm_client import LLMError, build_client
    from agent.prompts import SYSTEM_PROMPT, build_user_prompt
    from agent.validator import ValidationProblem, validate_plan

    if not settings.llm_configured:
        print(f"{WARN} No GEMINI_API_KEY set. The app will run in demo mode and "
              "cannot interpret free text.")
        return True

    client = build_client(settings)
    prompt = build_user_prompt(
        "add Dandadan to my plan to read list",
        [{"title": "One Piece", "status": "reading", "chapters_read": 1120,
          "latest_chapter": 1125, "rating": None}],
    )
    try:
        response = client.complete(SYSTEM_PROMPT, prompt)
    except LLMError as exc:
        print(f"{FAIL} LLM ({client.label}): {exc.message}")
        list_available_models(settings)
        return False

    try:
        plan = validate_plan(response.text, settings)
    except ValidationProblem as exc:
        print(f"{FAIL} LLM returned output the validator rejected: {exc.message}")
        print(f"        Raw output: {response.text[:300]}")
        return False

    names = [a.action for a in plan.actions]
    print(f"{PASS} LLM ({client.label}) responded with a valid plan: {names or 'no actions'}")
    if "add_series" not in names:
        print(f"{WARN} Expected an add_series action for that test prompt; the "
              "model produced something else. The app still works, but check "
              "your model choice.")
    return True


def list_available_models(settings) -> None:
    """Print the text models this key can reach.

    Google retires model IDs on a schedule, so the most common first-run
    failure is a model name that was valid when the code was written and
    has since been shut down. Showing the live list turns that into a
    one-line fix.
    """
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        names = []
        for model in client.models.list():
            actions = getattr(model, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            name = (model.name or "").removeprefix("models/")
            if name and "embedding" not in name:
                names.append(name)
    except Exception as exc:
        print(f"        (could not list models: {type(exc).__name__}: {exc})")
        return

    if not names:
        print("        (your key returned no text-generation models)")
        return

    print("        Models your key can reach:")
    for name in sorted(names)[:25]:
        marker = " <- currently configured" if name == settings.llm_model else ""
        print(f"          {name}{marker}")
    if len(names) > 25:
        print(f"          ... and {len(names) - 25} more")
    print("        Set LLM_MODEL in .streamlit/secrets.toml to one of these.")


def check_mangadex(settings) -> bool:
    from agent.manga_api import MangaAPI, MangaAPIError

    api = MangaAPI(settings)
    try:
        results = api.search_manga("Dandadan", limit=3)
    except MangaAPIError as exc:
        print(f"{FAIL} MangaDex: {exc}")
        return False
    if not results:
        print(f"{WARN} MangaDex reachable but returned no results for a known title.")
        return True
    print(f"{PASS} MangaDex search returned: {results[0]['title']}")

    try:
        latest = api.latest_chapter(results[0]["mangadex_id"])
    except MangaAPIError as exc:
        print(f"{FAIL} MangaDex chapter feed: {exc}")
        return False
    if latest is None:
        print(
            f"{WARN} MangaDex chapter feed reachable, but no chapter number was "
            f"found for '{results[0]['title']}'. The release watcher will treat "
            "that series as unknown rather than behind."
        )
    else:
        print(f"{PASS} MangaDex chapter feed: latest English chapter is {latest}")
    return True


def check_anilist(settings) -> bool:
    from agent.manga_api import MangaAPI, MangaAPIError

    api = MangaAPI(settings)
    try:
        results = api.recommendations(genres=["Action"], limit=3)
    except MangaAPIError as exc:
        print(f"{FAIL} AniList (filtered query): {exc}")
        # Probe with the simplest possible query to tell an outage apart from
        # a filter this code is building wrongly.
        try:
            api.cache.clear()
            plain = api.recommendations(limit=3)
        except MangaAPIError as plain_exc:
            print(f"        A filter-free query also failed: {plain_exc}")
            print("        That points at AniList itself rather than this code. "
                  "It is a community-run service and does have outages - try "
                  "again in a few minutes.")
        else:
            got = plain[0]["title"] if plain else "no results"
            print(f"        A filter-free query DID work (returned: {got}).")
            print("        So the failure is in how the filters are built, not "
                  "in AniList. Report this output.")
        return False
    if not results:
        print(f"{WARN} AniList reachable but returned no recommendations.")
        return True
    print(f"{PASS} AniList recommendations returned: {results[0]['title']}")
    return True


def main() -> int:
    settings = get_settings()
    print("Manga Tracker Agent - self check\n")

    checks = [
        ("database", check_database),
        ("LLM", check_llm),
        ("MangaDex", check_mangadex),
        ("AniList", check_anilist),
    ]
    failures = [name for name, fn in checks if not fn(settings)]

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed. Start the app with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
