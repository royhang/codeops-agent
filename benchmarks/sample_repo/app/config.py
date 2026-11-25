from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    db_url: str
    redis_host: str
    feature_flag_v2: bool


def load_service_config() -> ServiceConfig:
    """Load runtime configuration from environment variables."""
    # BUG: Reads misspelled env var "ORDER_DB_URl" (lowercase l) instead of "ORDER_DB_URL",
    # silently falling back to localhost sqlite in production.
    db_url = os.getenv("ORDER_DB_URl", "sqlite:///local_dev.db")
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    # BUG: bool("false") evaluates to True in Python because any non-empty string is truthy.
    feature_flag_v2 = bool(os.getenv("FEATURE_FLAG_V2", "false"))
    return ServiceConfig(
        db_url=db_url,
        redis_host=redis_host,
        feature_flag_v2=feature_flag_v2,
    )
