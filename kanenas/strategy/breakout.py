"""Volatility breakout: Donchian channel break with a volatility-squeeze filter.

Edge thesis: volatility is serially correlated, so a *quiet* market that breaks
its N-bar range tends to keep going - the squeeze is the setup, the break is the
trigger.  Requiring the break to clear the channel by a fraction of ATR filters
the one-tick pokes that immediately reverse.

The channel is read from the bars *preceding* the current one.  Including the
current bar would put its own high at the top of the channel, so price could
never be above it and the strategy would silently never fire.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from ..core.types import Direction, Signal
from .base import Strategy, StrategyContext


class VolatilityBreakout(Strategy):
    name = "breakout"
    weight = 1.1

    def __init__(self, buffer_atr: float = 0.10, squeeze_lookback: int = 60) -> None:
        self.buffer_atr = buffer_atr
        self.widths: Deque[float] = deque(maxlen=squeeze_lookback)

    def evaluate(self, ctx: StrategyContext) -> Signal:
        ind = ctx.ind
        if not (ind.donchian.ready and ind.atr.ready and ind.bb.ready):
            return self.flat("warming up")

        width = ind.bb.width
        self.widths.append(width)
        price = ctx.price
        atr = max(ind.atr.value, 1e-9)
        buf = self.buffer_atr * atr
        # the channel as it stood *before* this bar - see Donchian's docstring
        upper, lower = ind.donchian.prev_upper, ind.donchian.prev_lower
        if upper is None or lower is None:
            return self.flat("channel not established")

        # squeeze percentile: where does current band width sit in its own history?
        if len(self.widths) >= 20:
            below = sum(1 for w in self.widths if w < width)
            pct = below / len(self.widths)
        else:
            pct = 0.5
        squeeze_bonus = 1.25 if pct < 0.35 else (0.75 if pct > 0.85 else 1.0)

        if price > upper - buf:
            extent = (price - (upper - buf)) / atr
            conf = min(1.0, (0.45 + extent * 0.9) * squeeze_bonus)
            return self._sig(Direction.LONG, conf, f"break high +{extent:.2f} ATR, squeeze p{pct*100:.0f}")
        if price < lower + buf:
            extent = ((lower + buf) - price) / atr
            conf = min(1.0, (0.45 + extent * 0.9) * squeeze_bonus)
            return self._sig(Direction.SHORT, conf, f"break low -{extent:.2f} ATR, squeeze p{pct*100:.0f}")
        return self.flat(f"inside channel p{pct*100:.0f}")
