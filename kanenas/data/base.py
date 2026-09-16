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
    #: True for a bar replayed to fill a gap after an outage. Such a bar is
    #: real history and must still be tested against stops - that is the whole
    #: point of replaying it - but it must never *open* a position: entering at
    #: a price from half an hour ago is a fill that could not have happened.
    backfill: bool = False

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

    def stream(self) -> Iterator[Optional[MarketEvent]]:
        """Yield market events until exhausted (backtest) or forever (live).

        A real-time feed may also yield ``None`` as a heartbeat, meaning "still
        alive, no new bar yet". Consumers must skip it. Without it a caller
        blocked in ``next()`` cannot tell a quiet market from a dead feed, and
        an open position's stop would go unevaluated for as long as the silence
        lasted. Historical feeds never yield ``None``.
        """
        ...
