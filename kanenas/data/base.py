"""Feed abstraction: everything downstream consumes ``MarketEvent`` objects.

A feed yields one event per bar.  The engine never knows whether the bytes came
from a simulator, a CSV replay or a live REST/WS socket - which is exactly why
a strategy that was backtested is the same object that trades live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Protocol

from ..core.types import Candle, OrderBook


@dataclass(frozen=True)
class MarketEvent:
    symbol: str
    candle: Candle
    book: Optional[OrderBook] = None

    @property
    def price(self) -> float:
        return self.candle.close

    @property
    def ts(self) -> float:
        return self.candle.ts


class MarketFeed(Protocol):
    """Minimal contract every data source implements."""

    symbol: str
    name: str

    def stream(self) -> Iterator[MarketEvent]:
        """Yield market events until exhausted (backtest) or forever (live)."""
        ...
