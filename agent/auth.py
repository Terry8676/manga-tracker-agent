"""Username + password accounts.

Passwords are hashed with bcrypt and never stored or logged in plain text.
Each account owns its own library and its own audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import bcrypt

from .db import User, session_scope

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LENGTH = 8
# bcrypt silently ignores bytes past 72; truncate explicitly so a long
# password can never be mistaken for a different one.
_BCRYPT_MAX_BYTES = 72


@dataclass
class AuthResult:
    ok: bool
    message: str
    user_id: int | None = None
    username: str | None = None


def _prepare(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def validate_credentials(username: str, password: str) -> str | None:
    """Return an error message, or None when the credentials are acceptable."""
    if not USERNAME_RE.match(username or ""):
        return (
            "Username must be 3-32 characters, using letters, numbers, "
            "underscore, dot or hyphen."
        )
    if len(password or "") < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    return None


def sign_up(username: str, password: str) -> AuthResult:
    username = (username or "").strip()
    problem = validate_credentials(username, password)
    if problem:
        return AuthResult(False, problem)

    with session_scope() as session:
        existing = session.query(User).filter(User.username == username).one_or_none()
        if existing is not None:
            return AuthResult(False, "That username is taken. Pick another one.")
        user = User(username=username, password_hash=hash_password(password))
        session.add(user)
        session.flush()
        return AuthResult(True, f"Account created. Welcome, {username}.", user.id, username)


def log_in(username: str, password: str) -> AuthResult:
    username = (username or "").strip()
    with session_scope() as session:
        user = session.query(User).filter(User.username == username).one_or_none()
        # Same message either way, so the form can't be used to discover
        # which usernames exist.
        if user is None or not verify_password(password, user.password_hash):
            return AuthResult(False, "Incorrect username or password.")
        return AuthResult(True, f"Signed in as {username}.", user.id, user.username)
