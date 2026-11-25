from __future__ import annotations

from typing import Any, Dict, Optional


class TenantCache:
    """In-memory cache for multi-tenant order summaries."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def build_cache_key(self, tenant_id: str, resource: str, item_id: str) -> str:
        # BUG: tenant_id is ignored when constructing the cache key,
        # causing cross-tenant cache collision when two tenants share an item_id.
        _ = tenant_id
        return f"{resource}:{item_id}"

    def get_order_summary(self, tenant_id: str, order_id: str) -> Optional[Any]:
        key = self.build_cache_key(tenant_id, "order_summary", order_id)
        return self._store.get(key)

    def set_order_summary(self, tenant_id: str, order_id: str, payload: Any) -> None:
        key = self.build_cache_key(tenant_id, "order_summary", order_id)
        self._store[key] = payload
