"""Audit logging.

Every action the agent attempts gets a row, whether it succeeded, failed,
was rejected by the validator, or was cancelled by the user. The log lives in
the database alongside the user's library, so it survives app restarts and
redeploys, and it is scoped per user.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from .db import ActionLog

# Console logging as well, so errors show up in the Streamlit Cloud logs.
_console = logging.getLogger("manga_agent")
if not _console.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _console.addHandler(handler)
    _console.setLevel(logging.INFO)


def record_action(
    session: Session,
    *,
    user_id: int,
    user_request: str,
    action_name: str,
    payload: dict | None,
    status: str,
    result_summary: str,
    confirmed: bool = False,
) -> ActionLog:
    """Write one audit row and return it."""
    entry = ActionLog(
        user_id=user_id,
        user_request=user_request[:2000],
        action_name=action_name[:64],
        action_json=json.dumps(payload or {}, default=str)[:4000],
        status=status,
        result_summary=result_summary[:2000],
        confirmed_by_user=1 if confirmed else 0,
    )
    session.add(entry)
    session.flush()

    level = logging.INFO if status == "ok" else logging.WARNING
    _console.log(level, "user=%s action=%s status=%s", user_id, action_name, status)
    return entry


def recent_actions(session: Session, user_id: int, limit: int = 50) -> list[ActionLog]:
    return (
        session.query(ActionLog)
        .filter(ActionLog.user_id == user_id)
        .order_by(ActionLog.timestamp.desc(), ActionLog.id.desc())
        .limit(limit)
        .all()
    )


def log_to_csv(entries: list[ActionLog]) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["timestamp", "request", "action", "payload", "status", "result", "confirmed"]
    )
    for e in entries:
        writer.writerow(
            [
                e.timestamp.isoformat() if e.timestamp else "",
                e.user_request,
                e.action_name,
                e.action_json,
                e.status,
                e.result_summary,
                bool(e.confirmed_by_user),
            ]
        )
    return buffer.getvalue()
