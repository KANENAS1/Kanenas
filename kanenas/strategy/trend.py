"""Trend following: EMA stack alignment confirmed by MACD.

Edge thesis: crypto trends persist longer than a random walk would allow, so
being long while fast > slow > trend (and short in the mirror case) harvests
that persistence.  Confidence scales with *separation* between the EMAs
normalised by ATR, so a 3-ATR-wide stack is trusted far more than a hairline
cross - which is what kills naive crossover bots in chop.
"""

from __future__ import annotations

from ..core.types import Direction, Signal
from .base import Strategy, StrategyContext


class TrendFollow(Strategy):
    name = "trend"
    weight = 1.2

    def __init__(self, min_separation_atr: float = 0.25, min_efficiency: float = 0.30) -> None:
        self.min_separation_atr = min_separation_atr
        #: below this Efficiency Ratio the market is thrashing, not trending.
        #: Set to 0.0 to disable the gate.
        self.min_efficiency = min_efficiency

    def evaluate(self, ctx: StrategyContext) -> Signal:
        ind = ctx.ind
        if not (ind.ema_fast.ready and ind.ema_slow.ready and ind.ema_trend.ready and ind.atr.ready):
            return self.flat("warming up")
        # A stacked EMA set says price moved; it does not say price *travelled*
        # there. In a whipsaw the stack aligns and re-aligns every few bars and
        # each flip costs a round trip, which is what broke this strategy in
        # the stress suite. The Efficiency Ratio is the missing question.
        if self.min_efficiency > 0.0:
            er = ind.efficiency.value
            if er is None:
                return self.flat("warming up")
            if er < self.min_efficiency:
                return self.flat(f"chop: efficiency {er:.2f} < {self.min_efficiency:.2f}")

        fast, slow, trend = ind.ema_fast.value, ind.ema_slow.value, ind.ema_trend.value
        atr = max(ind.atr.value, 1e-9)
        sep = (fast - slow) / atr
        if abs(sep) < self.min_separation_atr:
            return self.flat(f"EMAs within {abs(sep):.2f} ATR")

        stacked_up = fast > slow > trend
        stacked_dn = fast < slow < trend
        if not (stacked_up or stacked_dn):
            return self.flat("stack not aligned")

        conf = min(1.0, abs(sep) / 1.5)
        hist = ind.macd.hist
        if hist is not None:
            # MACD agreeing is a bonus; disagreeing halves conviction
            agrees = (hist > 0) == stacked_up
            conf *= 1.15 if agrees else 0.5
        direction = Direction.LONG if stacked_up else Direction.SHORT
        return self._sig(direction, conf, f"stack {sep:+.2f} ATR, macd {hist:+.1f}" if hist is not None else f"stack {sep:+.2f} ATR")
