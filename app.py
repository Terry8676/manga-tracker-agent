"""Manga Reading Tracker + Release Watcher - Streamlit front end.

CS4680 AI Agent Project.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import json

import streamlit as st

from agent.agent import MangaAgent
from agent.auth import log_in, sign_up
from agent.config import diagnose_missing_secret, get_settings
from agent.db import init_db, session_scope
from agent.executor import Executor
from agent.llm_client import build_client
from agent.logger import log_to_csv, recent_actions
from agent.manga_api import MangaAPI
from agent.schemas import QueryLibrary

st.set_page_config(
    page_title="Manga Tracker Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_LABELS = {
    "reading": "Reading",
    "completed": "Completed",
    "dropped": "Dropped",
    "plan_to_read": "Plan to read",
    "on_hold": "On hold",
}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def bootstrap():
    """Initialise the database and the shared API client once per process."""
    settings = get_settings()
    init_db(settings)
    return settings, MangaAPI(settings)


@st.cache_resource(show_spinner=False)
def llm_client():
    return build_client(get_settings())


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def init_state() -> None:
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("upload_text", None)
    st.session_state.setdefault("upload_name", None)
    st.session_state.setdefault("downloads", [])
    st.session_state.setdefault("auth_error", None)


def current_agent() -> MangaAgent:
    settings, api = bootstrap()
    return MangaAgent(
        st.session_state.user_id,
        settings=settings,
        client=llm_client(),
        api=api,
        upload_provider=lambda: st.session_state.get("upload_text"),
    )


def sign_out() -> None:
    for key in (
        "user_id", "username", "messages", "pending",
        "upload_text", "upload_name", "downloads",
    ):
        st.session_state.pop(key, None)
    init_state()


# --------------------------------------------------------------------------- #
# Login screen
# --------------------------------------------------------------------------- #


def render_login() -> None:
    st.title("📚 Manga Tracker Agent")
    st.caption(
        "Tell it what you've read in plain English. It updates your library, "
        "checks MangaDex for new chapters, and recommends what to read next."
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Sign in")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.form_submit_button("Sign in", type="primary"):
                result = log_in(username, password)
                if result.ok:
                    st.session_state.user_id = result.user_id
                    st.session_state.username = result.username
                    st.session_state.auth_error = None
                    st.rerun()
                else:
                    st.session_state.auth_error = result.message

    with right:
        st.subheader("Create an account")
        st.caption("Your list is private to your account.")
        with st.form("signup_form"):
            new_user = st.text_input("Pick a username", key="su_user")
            new_pass = st.text_input(
                "Pick a password", type="password", key="su_pass",
                help="At least 8 characters.",
            )
            if st.form_submit_button("Create account"):
                result = sign_up(new_user, new_pass)
                if result.ok:
                    st.session_state.user_id = result.user_id
                    st.session_state.username = result.username
                    st.session_state.auth_error = None
                    st.rerun()
                else:
                    st.session_state.auth_error = result.message

    if st.session_state.get("auth_error"):
        st.error(st.session_state.auth_error)

    with st.expander("What can it do?"):
        st.markdown(
            """
            - **Track progress** - "I'm on chapter 142 of Kagurabachi"
            - **Change status** - "drop Boruto", "finished Vinland Saga"
            - **Watch releases** - "anything new to read?" checks MangaDex
              for every series you're reading
            - **Recommend** - "something like Vinland Saga but shorter and finished"
            - **Export / import** - CSV, Markdown or JSON
            """
        )


# --------------------------------------------------------------------------- #
# Sidebar: library, log, exports
# --------------------------------------------------------------------------- #


def diagnose_missing_secret_for_db() -> str:
    return diagnose_missing_secret("DATABASE_URL")


def load_library() -> list[dict]:
    settings, api = bootstrap()
    with session_scope() as session:
        executor = Executor(
            session, st.session_state.user_id, settings=settings, api=api
        )
        return executor.execute(QueryLibrary(limit=200, sort_by="updated")).data or []


def render_sidebar(library: list[dict]) -> None:
    settings, _ = bootstrap()

    st.sidebar.markdown(f"**{st.session_state.username}**")
    if st.sidebar.button("Sign out", use_container_width=True):
        sign_out()
        st.rerun()

    client = llm_client()
    if settings.llm_configured:
        st.sidebar.success(f"LLM: {client.label}", icon="✅")
    else:
        st.sidebar.warning(
            "No LLM key set, so the agent can't interpret free text.\n\n"
            + diagnose_missing_secret("GEMINI_API_KEY"),
            icon="⚠️",
        )

    # Which database is actually in use. Worth stating plainly: a missing
    # DATABASE_URL silently falls back to a local SQLite file, and on
    # Streamlit Community Cloud that file is wiped on every restart, so
    # every account and library would quietly disappear.
    if settings.is_postgres:
        st.sidebar.success("Storage: Postgres (persistent)", icon="✅")
    else:
        st.sidebar.error(
            "Storage: local SQLite. If this is the deployed app, accounts and "
            "libraries will be erased on every restart.\n\n"
            + diagnose_missing_secret_for_db(),
            icon="🚨",
        )

    st.sidebar.divider()

    # -- Library ----------------------------------------------------------- #
    st.sidebar.subheader(f"Your library ({len(library)})")
    if not library:
        st.sidebar.caption("Empty. Try: \"add Dandadan to plan to read\"")
    else:
        behind = [s for s in library if (s.get("chapters_behind") or 0) > 0]
        if behind:
            st.sidebar.markdown("**New chapters waiting**")
            for row in behind:
                st.sidebar.markdown(
                    f"- {row['title']} · **+{row['chapters_behind']:g}** "
                    f"(you: {row['chapters_read']:g})"
                )
            st.sidebar.markdown("---")

        for status in ("reading", "on_hold", "plan_to_read", "completed", "dropped"):
            rows = [s for s in library if s["status"] == status]
            if not rows:
                continue
            with st.sidebar.expander(
                f"{STATUS_LABELS[status]} ({len(rows)})", expanded=(status == "reading")
            ):
                for row in rows:
                    line = f"**{row['title']}** — ch {row['chapters_read']:g}"
                    if row.get("latest_chapter") is not None:
                        line += f" / {row['latest_chapter']:g}"
                    if row.get("rating") is not None:
                        line += f" · {row['rating']:g}/10"
                    st.markdown(line)

    st.sidebar.divider()

    # -- Import ------------------------------------------------------------ #
    st.sidebar.subheader("Import CSV")
    uploaded = st.sidebar.file_uploader(
        "A CSV with a 'title' column",
        type=["csv"],
        label_visibility="collapsed",
        key="csv_uploader",
    )
    if uploaded is not None:
        try:
            st.session_state.upload_text = uploaded.getvalue().decode("utf-8")
            st.session_state.upload_name = uploaded.name
            st.sidebar.caption(
                f"`{uploaded.name}` ready. Ask: \"import my csv\""
            )
        except UnicodeDecodeError:
            st.sidebar.error("That file isn't UTF-8 text.")
            st.session_state.upload_text = None

    st.sidebar.divider()

    # -- Audit log --------------------------------------------------------- #
    st.sidebar.subheader("Action log")
    with session_scope() as session:
        entries = recent_actions(session, st.session_state.user_id, limit=100)
        rows = [e.to_dict() for e in entries]
        csv_text = log_to_csv(entries)

    if not rows:
        st.sidebar.caption("Nothing logged yet.")
    else:
        icons = {"ok": "✅", "error": "❌", "cancelled": "⏸️", "rejected": "🚫"}
        for row in rows[:12]:
            stamp = (row["timestamp"] or "")[11:16]
            st.sidebar.markdown(
                f"<span style='font-family:monospace;font-size:0.78rem'>"
                f"{stamp} {icons.get(row['status'], '•')} {row['action']}</span>",
                unsafe_allow_html=True,
            )
        if len(rows) > 12:
            st.sidebar.caption(f"...and {len(rows) - 12} older entries")
        st.sidebar.download_button(
            "Download full log (CSV)",
            data=csv_text,
            file_name="action-log.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# Rendering one agent turn
# --------------------------------------------------------------------------- #


def render_results(turn) -> None:
    """Show the plan and what each action did."""
    if turn.plan and turn.plan.actions:
        icons = {"ok": "✅", "error": "❌", "cancelled": "⏸️", "rejected": "🚫"}
        lines = []
        results = {id(a): None for a in turn.plan.actions}
        for action, result in zip(turn.plan.actions, turn.results or []):
            results[id(action)] = result

        for action in turn.plan.actions:
            result = results.get(id(action))
            if result is None:
                icon = "⏸️" if action.requires_confirmation else "⏳"
                detail = "needs confirmation" if action.requires_confirmation else "queued"
            else:
                icon = icons.get(result.status, "•")
                detail = result.summary
            lines.append(f"{icon} `{action.action}` — {detail}")

        with st.expander(
            f"Plan: {len(turn.plan.actions)} action(s)", expanded=True
        ):
            for line in lines:
                st.markdown(line)
            if turn.reasoning:
                st.caption(f"Reasoning: {turn.reasoning}")

    for note in turn.notes:
        st.info(note, icon="ℹ️")

    # Tables for data-returning actions.
    for result in turn.results or []:
        if not result.ok or not result.data:
            continue
        if result.action in ("query_library", "check_new_chapters"):
            if isinstance(result.data, list) and result.data:
                st.dataframe(result.data, use_container_width=True, hide_index=True)
        elif result.action == "search_manga":
            for item in result.data:
                year = f" ({item['year']})" if item.get("year") else ""
                st.markdown(f"**{item['title']}**{year} — {item.get('publication_status', '')}")
                if item.get("description"):
                    st.caption(item["description"])
        elif result.action == "get_recommendations":
            for item in result.data:
                bits = []
                if item.get("chapters"):
                    bits.append(f"{item['chapters']} ch")
                if item.get("status"):
                    bits.append(item["status"])
                if item.get("score"):
                    bits.append(f"{item['score']}/100")
                st.markdown(f"**{item['title']}** — {' · '.join(bits)}")
                if item.get("genres"):
                    st.caption(", ".join(item["genres"]))
                if item.get("description"):
                    st.caption(item["description"])

    # Downloads produced by export_list.
    for download in (turn.report.downloads if turn.report else []):
        st.download_button(
            f"Download {download['filename']}",
            data=download["content"],
            file_name=download["filename"],
            mime=download["mime"],
            key=f"dl-{download['filename']}-{len(st.session_state.messages)}",
        )


def run_turn(request: str, *, confirmed=False, preapproved=None):
    agent = current_agent()
    history = [
        (m["role"], m["content"])
        for m in st.session_state.messages
        if m.get("content")
    ]
    with st.spinner("Thinking..."):
        return agent.handle(
            request,
            history=history,
            confirmed=confirmed,
            preapproved_plan=preapproved,
        )


# --------------------------------------------------------------------------- #
# Main chat screen
# --------------------------------------------------------------------------- #


def render_app() -> None:
    library = load_library()
    render_sidebar(library)

    st.title("📚 Manga Tracker Agent")

    behind_count = sum(1 for s in library if (s.get("chapters_behind") or 0) > 0)
    cols = st.columns(4)
    cols[0].metric("Series tracked", len(library))
    cols[1].metric("Currently reading", sum(1 for s in library if s["status"] == "reading"))
    cols[2].metric("New chapters", behind_count)
    cols[3].metric(
        "Chapters read", f"{sum(s['chapters_read'] for s in library):,.0f}"
    )

    if not st.session_state.messages:
        st.caption("Try one of these:")
        examples = [
            "Add Dandadan and Kagurabachi to plan to read",
            "Anything new to read?",
            "Something like Vinland Saga but shorter and finished",
            "Export my list as CSV",
        ]
        example_cols = st.columns(len(examples))
        for col, text in zip(example_cols, examples):
            if col.button(text, use_container_width=True, key=f"ex-{text[:14]}"):
                st.session_state.pending_input = text
                st.rerun()

    st.divider()

    # Replay the conversation.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("turn") is not None:
                render_results(message["turn"])

    # Confirmation card for a held destructive plan.
    held = st.session_state.get("pending")
    if held is not None:
        with st.chat_message("assistant"):
            st.warning("This needs your confirmation before I run it.", icon="⚠️")
            for action in held["plan"].actions:
                marker = "⚠️" if action.requires_confirmation else "•"
                st.markdown(f"{marker} {action.describe()}")
            confirm_col, cancel_col, _ = st.columns([1, 1, 3])
            if confirm_col.button("Confirm", type="primary", key="confirm_btn"):
                turn = run_turn(
                    held["request"], confirmed=True, preapproved=held["plan"]
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Done.", "turn": turn}
                )
                st.session_state.pending = None
                st.rerun()
            if cancel_col.button("Cancel", key="cancel_btn"):
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Cancelled. Nothing was changed.",
                        "turn": None,
                    }
                )
                st.session_state.pending = None
                st.rerun()

    # Input.
    typed = st.chat_input("What have you been reading?")
    queued = st.session_state.pop("pending_input", None)
    request = typed or queued

    if request:
        st.session_state.messages.append(
            {"role": "user", "content": request, "turn": None}
        )
        turn = run_turn(request)

        if not turn.ok:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {turn.reply}", "turn": None}
            )
        elif turn.pending_confirmation:
            st.session_state.pending = {"request": request, "plan": turn.plan}
            st.session_state.messages.append(
                {"role": "assistant", "content": turn.reply, "turn": None}
            )
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": turn.reply, "turn": turn}
            )
        st.rerun()


# --------------------------------------------------------------------------- #

def main() -> None:
    init_state()
    try:
        bootstrap()
    except Exception as exc:
        st.error(
            "Could not connect to the database. Check DATABASE_URL in your "
            f"secrets.\n\n`{type(exc).__name__}: {exc}`"
        )
        st.stop()

    if st.session_state.get("user_id") is None:
        render_login()
    else:
        render_app()


main()
