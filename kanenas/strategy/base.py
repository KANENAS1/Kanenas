"""Strategy contract.

A strategy is a pure function of market state -> ``Signal``.  It owns no money,
places no orders and knows nothing about position sizing.  That separation is
what makes each one testable in isolation and swappable at runtime, and it keeps
risk decisions in exactly one place (``kanenas.risk``) instead of smeared across
five files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.indicators import IndicatorSet
from ..core.types import Candle, Direction, OrderBook, Position, Signal


@dataclass(frozen=True)
class StrategyContext:
    """Everything a strategy is allowed to see for the current bar."""

    symbol: str
    candle: Candle
    ind: IndicatorSet
    book: Optional[OrderBook]
    position: Position
    equity: float

    @property
    def price(self) -> float:
        return self.candle.close


class Strategy:
    name = "strategy"
    #: relative weight in the ensemble before adaptive adjustment
    weight = 1.0

    def evaluate(self, ctx: StrategyContext) -> Signal:  # pragma: no cover - abstract
        raise NotImplementedError

    # convenience for subclasses
    def _sig(self, direction: Direction, confidence: float, reason: str) -> Signal:
        return Signal(direction, max(0.0, min(1.0, confidence)), self.name, reason)

    def flat(self, reason: str = "no edge") -> Signal:
        return Signal.flat(self.name, reason)
