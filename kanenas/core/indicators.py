"""Streaming technical indicators.

Every indicator here is *incremental*: you push one value and it updates in
constant time.  That matters because the same code runs a 200k-bar backtest and
a live feed; recomputing a window from scratch on each tick would make the
backtest quadratic and the live loop jittery.

No numpy on purpose - the bot must run on a bare Python install.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Iterable, Optional

from .types import Candle


class Indicator:
    """Base: push a value, read ``.value``, check ``.ready``."""

    __slots__ = ("_value", "_count")

    def __init__(self) -> None:
        self._value: Optional[float] = None
        self._count = 0

    @property
    def value(self) -> Optional[float]:
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None

    def update(self, x: float) -> Optional[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def prime(self, values: Iterable[float]) -> "Indicator":
        for v in values:
            self.update(v)
        return self


class SMA(Indicator):
    __slots__ = ("period", "_win", "_sum")

    def __init__(self, period: int) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._win: Deque[float] = deque(maxlen=period)
        self._sum = 0.0

    def update(self, x: float) -> Optional[float]:
        if len(self._win) == self.period:
            self._sum -= self._win[0]
        self._win.append(x)
        self._sum += x
        self._count += 1
        if len(self._win) == self.period:
            self._value = self._sum / self.period
        return self._value


class EMA(Indicator):
    """Exponential MA seeded with an SMA so early bars are not biased by x0."""

    __slots__ = ("period", "alpha", "_seed", "_seed_sum")

    def __init__(self, period: int) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self._seed = 0
        self._seed_sum = 0.0

    def update(self, x: float) -> Optional[float]:
        self._count += 1
        if self._value is None:
            self._seed += 1
            self._seed_sum += x
            if self._seed >= self.period:
                self._value = self._seed_sum / self.period
            return self._value
        self._value += self.alpha * (x - self._value)
        return self._value


class WilderMA(Indicator):
    """Wilder's smoothing (alpha = 1/n) - what RSI/ATR are actually defined on."""

    __slots__ = ("period", "_seed", "_seed_sum")

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self._seed = 0
        self._seed_sum = 0.0

    def update(self, x: float) -> Optional[float]:
        self._count += 1
        if self._value is None:
            self._seed += 1
            self._seed_sum += x
            if self._seed >= self.period:
                self._value = self._seed_sum / self.period
            return self._value
        self._value = (self._value * (self.period - 1) + x) / self.period
        return self._value


class RollingStats(Indicator):
    """Rolling mean and *sample* stdev over a fixed window.

    Recomputed from the deque rather than kept as a running sum-of-squares:
    windows are short (tens of bars) and the naive form loses precision badly
    on price series where mean >> variance.
    """

    __slots__ = ("period", "_win", "_std")

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self._win: Deque[float] = deque(maxlen=period)
        self._std = 0.0

    def update(self, x: float) -> Optional[float]:
        self._win.append(x)
        self._count += 1
        if len(self._win) < max(2, self.period):
            return None
        n = len(self._win)
        mean = math.fsum(self._win) / n
        var = math.fsum((v - mean) ** 2 for v in self._win) / (n - 1)
        self._value = mean
        self._std = math.sqrt(var)
        return self._value

    @property
    def mean(self) -> Optional[float]:
        return self._value

    @property
    def stdev(self) -> float:
        return self._std

    def zscore(self, x: float) -> float:
        if self._value is None or self._std <= 1e-12:
            return 0.0
        return (x - self._value) / self._std


class RSI(Indicator):
    """Wilder RSI in [0, 100]."""

    __slots__ = ("period", "_prev", "_gain", "_loss")

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.period = period
        self._prev: Optional[float] = None
        self._gain = WilderMA(period)
        self._loss = WilderMA(period)

    def update(self, x: float) -> Optional[float]:
        self._count += 1
        if self._prev is None:
            self._prev = x
            return None
        delta = x - self._prev
        self._prev = x
        up = self._gain.update(max(delta, 0.0))
        down = self._loss.update(max(-delta, 0.0))
        if up is None or down is None:
            return None
        if down <= 1e-12:
            self._value = 100.0
        else:
            rs = up / down
            self._value = 100.0 - (100.0 / (1.0 + rs))
        return self._value


class ATR(Indicator):
    """Average True Range - the bot's unit of risk, used for every stop."""

    __slots__ = ("period", "_ma", "_prev_close")

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.period = period
        self._ma = WilderMA(period)
        self._prev_close: Optional[float] = None

    def update_candle(self, c: Candle) -> Optional[float]:
        if self._prev_close is None:
            tr = c.high - c.low
        else:
            tr = max(
                c.high - c.low,
                abs(c.high - self._prev_close),
                abs(c.low - self._prev_close),
            )
        self._prev_close = c.close
        self._count += 1
        self._value = self._ma.update(tr)
        return self._value

    def update(self, x: float) -> Optional[float]:  # close-only fallback
        return self.update_candle(Candle(0.0, x, x, x, x, 0.0))


class MACD:
    """Classic 12/26/9.  Exposes line, signal and histogram."""

    __slots__ = ("fast", "slow", "signal", "line", "hist")

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = EMA(fast)
        self.slow = EMA(slow)
        self.signal = EMA(signal)
        self.line: Optional[float] = None
        self.hist: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self.hist is not None

    def update(self, x: float) -> Optional[float]:
        f = self.fast.update(x)
        s = self.slow.update(x)
        if f is None or s is None:
            return None
        self.line = f - s
        sig = self.signal.update(self.line)
        if sig is not None:
            self.hist = self.line - sig
        return self.line


class Donchian:
    """Highest high / lowest low over N bars - the breakout reference.

    Exposes two views.  ``upper``/``lower`` include the bar just pushed;
    ``prev_upper``/``prev_lower`` are the channel *as it stood before* it.
    Breakout logic must use the ``prev_`` pair: a bar that makes a new high is
    itself the new channel top, so comparing the bar against a channel that
    already contains it can never register a break.
    """

    __slots__ = ("period", "_highs", "_lows", "_prev_upper", "_prev_lower")

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self._highs: Deque[float] = deque(maxlen=period)
        self._lows: Deque[float] = deque(maxlen=period)
        self._prev_upper: Optional[float] = None
        self._prev_lower: Optional[float] = None

    @property
    def ready(self) -> bool:
        return len(self._highs) == self.period

    @property
    def upper(self) -> Optional[float]:
        return max(self._highs) if self._highs else None

    @property
    def lower(self) -> Optional[float]:
        return min(self._lows) if self._lows else None

    @property
    def mid(self) -> Optional[float]:
        if not self.ready:
            return None
        return (self.upper + self.lower) / 2.0

    @property
    def prev_upper(self) -> Optional[float]:
        return self._prev_upper

    @property
    def prev_lower(self) -> Optional[float]:
        return self._prev_lower

    def update_candle(self, c: Candle) -> None:
        # snapshot the channel *before* this bar joins it
        self._prev_upper = max(self._highs) if self._highs else None
        self._prev_lower = min(self._lows) if self._lows else None
        self._highs.append(c.high)
        self._lows.append(c.low)


class Bollinger:
    __slots__ = ("stats", "mult")

    def __init__(self, period: int = 20, mult: float = 2.0) -> None:
        self.stats = RollingStats(period)
        self.mult = mult

    @property
    def ready(self) -> bool:
        return self.stats.ready

    def update(self, x: float) -> Optional[float]:
        return self.stats.update(x)

    @property
    def upper(self) -> Optional[float]:
        if not self.ready:
            return None
        return self.stats.mean + self.mult * self.stats.stdev

    @property
    def lower(self) -> Optional[float]:
        if not self.ready:
            return None
        return self.stats.mean - self.mult * self.stats.stdev

    @property
    def width(self) -> float:
        """Band width as a fraction of price - a cheap volatility regime gauge."""
        if not self.ready or not self.stats.mean:
            return 0.0
        return (self.upper - self.lower) / self.stats.mean


class RealizedVol(Indicator):
    """Annualised realised volatility from log returns of the last N bars."""

    __slots__ = ("period", "bars_per_year", "_prev", "_rets")

    def __init__(self, period: int = 30, bars_per_year: float = 525600.0) -> None:
        super().__init__()
        self.period = period
        self.bars_per_year = bars_per_year
        self._prev: Optional[float] = None
        self._rets: Deque[float] = deque(maxlen=period)

    def update(self, x: float) -> Optional[float]:
        if self._prev is not None and self._prev > 0 and x > 0:
            self._rets.append(math.log(x / self._prev))
        self._prev = x
        self._count += 1
        if len(self._rets) < max(2, self.period // 2):
            return None
        n = len(self._rets)
        mean = math.fsum(self._rets) / n
        var = math.fsum((r - mean) ** 2 for r in self._rets) / (n - 1)
        self._value = math.sqrt(var * self.bars_per_year)
        return self._value


class IndicatorSet:
    """Every indicator the strategies need, updated once per bar.

    Strategies read from this shared set instead of each keeping their own
    copies - one pass over the bar, no duplicated state to drift out of sync.
    """

    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        trend: int = 55,
        rsi_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        donchian: int = 20,
        vol_period: int = 30,
        bars_per_year: float = 525600.0,
    ) -> None:
        self.ema_fast = EMA(fast)
        self.ema_slow = EMA(slow)
        self.ema_trend = EMA(trend)
        self.rsi = RSI(rsi_period)
        self.atr = ATR(atr_period)
        self.macd = MACD()
        self.bb = Bollinger(bb_period)
        self.donchian = Donchian(donchian)
        self.vol = RealizedVol(vol_period, bars_per_year)
        self.closes: Deque[float] = deque(maxlen=512)
        self.candles: Deque[Candle] = deque(maxlen=512)
        self.bars = 0

    @property
    def warm(self) -> bool:
        """True once every indicator has enough history to be meaningful."""
        return (
            self.ema_trend.ready
            and self.rsi.ready
            and self.atr.ready
            and self.bb.ready
            and self.donchian.ready
        )

    def update(self, c: Candle) -> None:
        self.bars += 1
        self.closes.append(c.close)
        self.candles.append(c)
        self.ema_fast.update(c.close)
        self.ema_slow.update(c.close)
        self.ema_trend.update(c.close)
        self.rsi.update(c.close)
        self.atr.update_candle(c)
        self.macd.update(c.close)
        self.bb.update(c.close)
        self.donchian.update_candle(c)
        self.vol.update(c.close)

    def atr_pct(self, price: float) -> float:
        if not self.atr.ready or price <= 0:
            return 0.0
        return self.atr.value / price
