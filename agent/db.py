"""Database layer: SQLAlchemy models and session management.

The same models run on SQLite (local development) and Postgres (Neon, in
deployment) - only the connection string changes. Every row that belongs to a
user carries a ``user_id`` foreign key, and every query in the executor is
filtered by the logged-in user, so one account can never read or modify
another account's library.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterator

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from .config import Settings, get_settings

# Statuses a series can hold. Kept here so the validator, prompts and UI all
# agree on one vocabulary.
VALID_STATUSES = ("reading", "completed", "dropped", "plan_to_read", "on_hold")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    series: Mapped[list["Series"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    logs: Mapped[list["ActionLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Series(Base):
    """One manga in one user's library."""

    __tablename__ = "series"
    __table_args__ = (UniqueConstraint("user_id", "title", name="uq_series_user_title"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="reading")
    chapters_read: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Cached from MangaDex so the release watcher can diff without re-searching.
    mangadex_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latest_chapter: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_checked: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    user: Mapped[User] = relationship(back_populates="series")

    @property
    def chapters_behind(self) -> float | None:
        """How many chapters have been released past the user's progress."""
        if self.latest_chapter is None:
            return None
        return max(0.0, round(self.latest_chapter - self.chapters_read, 1))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "chapters_read": self.chapters_read,
            "rating": self.rating,
            "notes": self.notes,
            "latest_chapter": self.latest_chapter,
            "chapters_behind": self.chapters_behind,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
        }


class ActionLog(Base):
    """Audit trail. One row per action the agent attempted, successful or not."""

    __tablename__ = "action_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    user_request: Mapped[str] = mapped_column(Text, nullable=False)
    action_name: Mapped[str] = mapped_column(String(64), nullable=False)
    action_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # ok|error|rejected|cancelled
    result_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confirmed_by_user: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    user: Mapped[User] = relationship(back_populates="logs")

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "request": self.user_request,
            "action": self.action_name,
            "payload": self.action_json,
            "status": self.status,
            "result": self.result_summary,
            "confirmed": bool(self.confirmed_by_user),
        }


# --------------------------------------------------------------------------- #
# Engine / session plumbing
# --------------------------------------------------------------------------- #

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _normalise_url(url: str) -> str:
    """Neon and Heroku hand out ``postgres://``; SQLAlchemy 2 wants the
    ``postgresql+psycopg2://`` form. Also force TLS for hosted Postgres."""
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://") :]
    if url.startswith("postgresql+psycopg2://") and "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Turn on foreign keys for SQLite so ON DELETE CASCADE actually fires."""
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def init_db(settings: Settings | None = None, *, force: bool = False) -> Engine:
    """Create the engine (once) and ensure tables exist."""
    global _engine, _SessionFactory
    if _engine is not None and not force:
        return _engine

    settings = settings or get_settings()
    url = _normalise_url(settings.database_url)

    kwargs: dict = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Streamlit reruns scripts constantly; recycle connections so Neon
        # doesn't hand back a socket it has already closed.
        kwargs["pool_recycle"] = 300
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 5

    _engine = create_engine(url, **kwargs)
    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_session() -> Session:
    """Return a new session, initialising the engine on first use."""
    if _SessionFactory is None:
        init_db()
    assert _SessionFactory is not None
    return _SessionFactory()


class session_scope:
    """Context manager that commits on success and rolls back on failure."""

    def __enter__(self) -> Session:
        self.session = get_session()
        return self.session

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.close()
        return False


def reset_engine() -> None:
    """Drop the cached engine. Used by the test suite between cases."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


def iter_user_series(session: Session, user_id: int) -> Iterator[Series]:
    """All series for one user, newest activity first."""
    return iter(
        session.query(Series)
        .filter(Series.user_id == user_id)
        .order_by(Series.updated_at.desc())
        .all()
    )
