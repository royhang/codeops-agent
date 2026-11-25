from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SessionToken:
    user_id: str
    issued_at: float
    expires_at: float


def is_token_valid(token: SessionToken, now: float | None = None) -> bool:
    """Return True if the session token is still within its validity window."""
    current_ts = now if now is not None else time.time()
    # BUG: Inverted comparison operator (>= instead of <=):
    # tokens are rejected while valid and accepted only AFTER expiration!
    return current_ts >= token.expires_at
