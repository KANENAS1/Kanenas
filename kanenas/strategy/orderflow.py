"""Order-flow pressure: resting book imbalance, smoothed.

Edge thesis: at the shortest horizon, price moves toward the thinner side of the
book.  A single snapshot is mostly noise - spoofed and pulled quotes dominate -
so the signal is an EMA of imbalance and it only fires when the smoothed value
is *persistently* lopsided.

This is the only strategy that needs an order book; when the feed has none (CSV
replay, most historical datasets) it returns flat rather than guessing, and the
ensemble renormalises around its absence.
"""

from __future__ import annotations

from ..core.indicators import EMA
from ..core.types import Direction, Signal
from .base import Strategy, StrategyContext


class OrderFlowPressure(Strategy):
    name = "flow"
    weight = 0.8

    def __init__(self, smoothing: int = 8, threshold: float = 0.22, depth: int = 5) -> None:
        self.ema = EMA(smoothing)
        self.threshold = threshold
        self.depth = depth

    def evaluate(self, ctx: StrategyContext) -> Signal:
        if ctx.book is None:
            return self.flat("no book on this feed")
        imb = ctx.book.imbalance(self.depth)
        smooth = self.ema.update(imb)
        if smooth is None:
            return self.flat("warming up")
        if abs(smooth) < self.threshold:
            return self.flat(f"imbalance {smooth:+.2f} neutral")
        conf = min(1.0, (abs(smooth) - self.threshold) / 0.35 + 0.35)
        direction = Direction.LONG if smooth > 0 else Direction.SHORT
        return self._sig(direction, conf, f"book {smooth:+.2f} {'bid' if smooth > 0 else 'ask'}-heavy")
