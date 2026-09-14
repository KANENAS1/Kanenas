"""Cross-sectional-style momentum on a single series: risk-adjusted drift.

Edge thesis: the Sharpe of recent returns is a better momentum estimate than the
raw return, because it discounts moves that were pure volatility.  Here it is
the mean log return over N bars divided by their stdev - a t-statistic on drift.
Requiring |t| above a threshold means the bot only calls something a trend when
it is statistically distinguishable from noise.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

from ..core.types import Direction, Signal
from .base import Strategy, StrategyContext


class RiskAdjustedMomentum(Strategy):
    name = "momo"
    weight = 1.0

    def __init__(self, lookback: int = 40, t_entry: float = 1.1) -> None:
        self.lookback = lookback
        self.t_entry = t_entry
        self.rets: Deque[float] = deque(maxlen=lookback)
        self._prev: float | None = None

    def evaluate(self, ctx: StrategyContext) -> Signal:
        price = ctx.price
        if self._prev is not None and self._prev > 0 and price > 0:
            self.rets.append(math.log(price / self._prev))
        self._prev = price
        if len(self.rets) < self.lookback:
            return self.flat("warming up")

        n = len(self.rets)
        mean = math.fsum(self.rets) / n
        var = math.fsum((r - mean) ** 2 for r in self.rets) / (n - 1)
        sd = math.sqrt(var)
        if sd <= 1e-12:
            return self.flat("zero variance")
        t = mean / (sd / math.sqrt(n))  # t-stat of the drift
        if abs(t) < self.t_entry:
            return self.flat(f"t {t:+.2f} insignificant")
        conf = min(1.0, (abs(t) - self.t_entry) / 1.8 + 0.4)
        direction = Direction.LONG if t > 0 else Direction.SHORT
        return self._sig(direction, conf, f"drift t={t:+.2f} over {n} bars")
