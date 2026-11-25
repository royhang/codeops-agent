from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class LineItem:
    sku: str
    unit_price: float
    quantity: int


def calculate_discount_rate(items: List[LineItem], coupon_amount: float) -> float:
    """Compute effective discount ratio across all order items."""
    gross_total = sum(item.unit_price * item.quantity for item in items)
    # BUG: When items is empty or promotional free items have gross_total == 0,
    # dividing coupon_amount by gross_total raises ZeroDivisionError.
    ratio = coupon_amount / gross_total
    return min(max(ratio, 0.0), 1.0)


def compute_payable_amount(items: List[LineItem], coupon_amount: float = 0.0) -> float:
    """Calculate final payable amount after applying coupon discount."""
    gross_total = sum(item.unit_price * item.quantity for item in items)
    discount_rate = calculate_discount_rate(items, coupon_amount)
    return round(gross_total * (1.0 - discount_rate), 2)
