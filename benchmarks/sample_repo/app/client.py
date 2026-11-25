from __future__ import annotations

from typing import Any, Callable, Dict


class PaymentGatewayClient:
    """HTTP client wrapper for downstream payment gateway."""

    def __init__(self, transport: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> None:
        self._transport = transport

    def charge(self, order_id: str, amount: float) -> Dict[str, Any]:
        payload = {"order_id": order_id, "amount": amount}
        # BUG: Missing request timeout and unbounded while-True retry loop on transient errors,
        # which exhausts worker threads and amplifies downstream outages.
        while True:
            response = self._transport("/v1/charge", payload)
            if response.get("status") == "ok":
                return response
