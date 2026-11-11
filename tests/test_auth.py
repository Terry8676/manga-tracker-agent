"""Accounts: hashing, login, and the guarantee that libraries are isolated."""

from __future__ import annotations

from agent.auth import hash_password, log_in, sign_up, verify_password
from agent.db import Series, User, session_scope
from agent.executor import Executor
from agent.schemas import AddSeries, QueryLibrary


def test_password_is_hashed_not_stored_plain(settings):
    sign_up("alice", "correct horse battery")
    with session_scope() as session:
        user = session.query(User).filter(User.username == "alice").one()
        assert user.password_hash != "correct horse battery"
        assert user.password_hash.startswith("$2")
        assert verify_password("correct horse battery", user.password_hash)


def test_hash_is_salted(settings):
    a = hash_password("samepassword")
    b = hash_password("samepassword")
    assert a != b
    assert verify_password("samepassword", a)
    assert verify_password("samepassword", b)


def test_long_password_is_truncated_consistently(settings):
    long_pw = "x" * 200
    h = hash_password(long_pw)
    assert verify_password(long_pw, h)
    assert verify_password("x" * 72, h)


def test_signup_rejects_short_password(settings):
    result = sign_up("bob", "short")
    assert not result.ok
    assert "at least" in result.message


def test_signup_rejects_bad_username(settings):
    assert not sign_up("a", "longenoughpassword").ok
    assert not sign_up("has space", "longenoughpassword").ok
    assert not sign_up("bad/char", "longenoughpassword").ok


def test_signup_rejects_duplicate(settings):
    assert sign_up("carol", "longenoughpassword").ok
    second = sign_up("carol", "differentpassword")
    assert not second.ok
    assert "taken" in second.message


def test_login_success_and_failure(settings):
    sign_up("dave", "longenoughpassword")
    good = log_in("dave", "longenoughpassword")
    assert good.ok and good.user_id is not None

    bad = log_in("dave", "wrongpassword")
    assert not bad.ok

    missing = log_in("nobody", "longenoughpassword")
    assert not missing.ok
    # Same wording either way, so the form can't enumerate usernames.
    assert bad.message == missing.message


def test_libraries_are_isolated_between_users(settings, user, other_user):
    with session_scope() as session:
        Executor(session, user.user_id).execute(
            AddSeries(title="Kagurabachi", status="reading", chapters_read=140)
        )
        Executor(session, other_user.user_id).execute(
            AddSeries(title="Dandadan", status="reading", chapters_read=10)
        )

    with session_scope() as session:
        mine = Executor(session, user.user_id).execute(QueryLibrary())
        theirs = Executor(session, other_user.user_id).execute(QueryLibrary())

    assert [r["title"] for r in mine.data] == ["Kagurabachi"]
    assert [r["title"] for r in theirs.data] == ["Dandadan"]


def test_one_user_cannot_modify_another_users_series(settings, user, other_user):
    from agent.schemas import RemoveSeries, UpdateProgress

    with session_scope() as session:
        Executor(session, user.user_id).execute(AddSeries(title="One Piece"))

    with session_scope() as session:
        attacker = Executor(session, other_user.user_id)
        update = attacker.execute(UpdateProgress(title="One Piece", chapters_read=999))
        delete = attacker.execute(RemoveSeries(title="One Piece"))

    assert update.status == "error" and "not in your library" in update.summary
    assert delete.status == "error"

    with session_scope() as session:
        row = session.query(Series).filter(Series.title == "One Piece").one()
        assert row.user_id == user.user_id
        assert row.chapters_read == 0.0


def test_deleting_a_user_removes_their_rows(settings, user):
    with session_scope() as session:
        Executor(session, user.user_id).execute(AddSeries(title="Frieren"))

    with session_scope() as session:
        session.delete(session.get(User, user.user_id))

    with session_scope() as session:
        assert session.query(Series).count() == 0
