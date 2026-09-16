"""Mean reversion: Bollinger z-score with an RSI trigger.

Edge thesis: inside a range, price overshoots and snaps back.  The z-score says
*how far* price has stretched from its own mean in units of its own volatility;
RSI confirms the stretch is exhaustion rather than the start of a trend.

Critically this strategy **stands down when a trend is in force** (price far
from the long EMA).  Fading a real trend is the single fastest way to lose
money, so the filter is a hard gate, not a confidence penalty.
"""

from __future__ import annotations

from ..core.types import Direction, Signal
from .base import Strategy, StrategyContext


class MeanReversion(Strategy):
    name = "revert"
    weight = 1.0

    def __init__(self, z_entry: float = 1.8, rsi_low: float = 32.0, rsi_high: float = 68.0,
                 trend_veto_atr: float = 2.5, max_efficiency: float = 0.45) -> None:
        #: the mirror of the trend gate - this strategy *wants* an inefficient,
        #: thrashing market. Set to 1.0 to disable.
        self.max_efficiency = max_efficiency
        self.z_entry = z_entry
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.trend_veto_atr = trend_veto_atr

    def evaluate(self, ctx: StrategyContext) -> Signal:
        ind = ctx.ind
        if not (ind.bb.ready and ind.rsi.ready and ind.atr.ready and ind.ema_trend.ready):
            return self.flat("warming up")

        price = ctx.price
        z = ind.bb.stats.zscore(price)
        rsi = ind.rsi.value
        atr = max(ind.atr.value, 1e-9)

        # hard veto: strong directional displacement means this is a trend, not a stretch
        trend_dist = (price - ind.ema_trend.value) / atr
        if abs(trend_dist) > self.trend_veto_atr:
            return self.flat(f"trend veto {trend_dist:+.1f} ATR")
        # second veto: an efficient market is going somewhere, so a stretch
        # away from the mean is the move starting, not an overshoot to fade
        if self.max_efficiency < 1.0:
            er = ind.efficiency.value
            if er is not None and er > self.max_efficiency:
                return self.flat(f"efficient market {er:.2f} - not a fade")

        if z <= -self.z_entry and rsi <= self.rsi_low:
            conf = min(1.0, (abs(z) - self.z_entry) / 1.2 + 0.45)
            return self._sig(Direction.LONG, conf, f"z {z:+.2f}, rsi {rsi:.0f} oversold")
        if z >= self.z_entry and rsi >= self.rsi_high:
            conf = min(1.0, (abs(z) - self.z_entry) / 1.2 + 0.45)
            return self._sig(Direction.SHORT, conf, f"z {z:+.2f}, rsi {rsi:.0f} overbought")
        return self.flat(f"z {z:+.2f} inside band")
