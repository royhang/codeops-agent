from __future__ import annotations

from typing import Dict, List


class OrderRepository:
    """Query builder and in-memory store for customer orders."""

    def __init__(self, rows: List[Dict[str, str]]) -> None:
        self._rows = rows

    def list_orders(self, status: str, page: int = 1, page_size: int = 20) -> List[Dict[str, str]]:
        # BUG: page and page_size are never applied (missing slice/LIMIT),
        # and empty status returns all records unbounded, causing OOM on large tables.
        matched = [row for row in self._rows if row.get("status") == status]
        _ = (page, page_size)
        return matched

    def build_filter_sql(self, status: str) -> str:
        # BUG: Direct string interpolation instead of parameterized query allows SQL injection.
        return f"SELECT id, status, total FROM orders WHERE status = '{status}'"
