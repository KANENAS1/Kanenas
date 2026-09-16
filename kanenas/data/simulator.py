"""A market simulator good enough to develop and stress a strategy against.

Real exchange data is the goal, but you cannot unit-test against a live market:
it is not reproducible, not reachable from CI, and never hands you the tail
events you need.  So this module generates price paths that carry the features
that actually break trading bots:

* **regime switching** - a 3-state Markov chain (trend up / chop / trend down)
  so a trend-follower cannot just ride one drift forever;
* **stochastic volatility** - GARCH(1,1)-style clustering, so quiet stretches
  are followed by quiet stretches and risk sizing has to adapt;
* **jumps** - a Poisson jump component for the gaps that blow through stops;
* **microstructure** - each bar is built from sub-ticks, so highs/lows and the
  order book are consistent with the path rather than painted on afterwards.

Seeded, so every run is reproducible: same seed, same market, byte for byte.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from ..core.types import BookLevel, Candle, OrderBook
from .base import MarketEvent


@dataclass
class Regime:
    name: str
    drift: float  # per-bar log drift
    vol_mult: float
    persistence: float  # probability of staying in this regime next bar


DEFAULT_REGIMES: List[Regime] = [
    Regime("BULL", 0.00035, 0.85, 0.986),
    Regime("CHOP", 0.00000, 1.00, 0.975),
    Regime("BEAR", -0.00040, 1.25, 0.982),
]


#: Fixed epoch for generated bars: 2024-01-01T00:00:00Z.
#: Seeding the clock from ``time.time()`` instead made results depend on the
#: wall-clock hour you ran them - the risk manager rolls its daily-loss window
#: on UTC day boundaries, so where a run's bars fell relative to midnight
#: changed which trades were halted. Same seed, different afternoon, different
#: equity curve. A fixed origin makes a seed fully reproducible.
DEFAULT_START_TS = 1_704_067_200.0


@dataclass
class SimulatorConfig:
    symbol: str = "BTC-USD"
    start_price: float = 78_000.0
    bar_seconds: float = 60.0
    base_vol: float = 0.0016      # per-bar stdev of log returns (~0.16%)
    garch_alpha: float = 0.11     # weight on last shock  (clustering)
    garch_beta: float = 0.86      # weight on last variance (persistence)
    jump_prob: float = 0.004      # per-bar probability of a jump
    jump_scale: float = 0.012     # stdev of jump size in log space
    ticks_per_bar: int = 24       # sub-steps used to carve the OHLC
    spread_bps: float = 1.2       # half-spread in basis points of mid
    book_depth: int = 8
    base_liquidity: float = 6.0   # size at top of book, in base units
    seed: Optional[int] = 7
    #: epoch seconds of the first bar; pass ``time.time()`` for a live-looking
    #: clock, at the cost of reproducibility
    start_ts: float = DEFAULT_START_TS
    regimes: List[Regime] = field(default_factory=lambda: list(DEFAULT_REGIMES))


class MarketSimulator:
    """Generates bars, an order book and a running regime label."""

    name = "simulator"

    def __init__(self, config: Optional[SimulatorConfig] = None, bars: Optional[int] = None) -> None:
        self.cfg = config or SimulatorConfig()
        self.symbol = self.cfg.symbol
        self.bars = bars
        self.rng = random.Random(self.cfg.seed)
        self.price = self.cfg.start_price
        self.variance = self.cfg.base_vol ** 2
        self.regime_idx = 1  # start in CHOP
        self.ts = self.cfg.start_ts
        self.bar_count = 0

    # ---------------------------------------------------------------- regime

    @property
    def regime(self) -> Regime:
        return self.cfg.regimes[self.regime_idx]

    def _step_regime(self) -> None:
        r = self.regime
        if self.rng.random() < r.persistence:
            return
        choices = [i for i in range(len(self.cfg.regimes)) if i != self.regime_idx]
        self.regime_idx = self.rng.choice(choices)

    # ------------------------------------------------------------ volatility

    def _step_variance(self, last_shock: float) -> None:
        cfg = self.cfg
        omega = cfg.base_vol ** 2 * (1.0 - cfg.garch_alpha - cfg.garch_beta)
        self.variance = omega + cfg.garch_alpha * last_shock ** 2 + cfg.garch_beta * self.variance
        # keep vol in a sane band so a tail run cannot explode the series
        self.variance = max(self.variance, (cfg.base_vol * 0.25) ** 2)
        self.variance = min(self.variance, (cfg.base_vol * 8.0) ** 2)

    # ----------------------------------------------------------------- bars

    def next_bar(self) -> MarketEvent:
        cfg = self.cfg
        self._step_regime()
        regime = self.regime

        sigma = math.sqrt(self.variance) * regime.vol_mult
        tick_sigma = sigma / math.sqrt(cfg.ticks_per_bar)
        tick_drift = regime.drift / cfg.ticks_per_bar

        open_price = self.price
        high = low = open_price
        price = open_price
        volume = 0.0
        last_shock = 0.0

        for _ in range(cfg.ticks_per_bar):
            shock = self.rng.gauss(0.0, tick_sigma)
            move = tick_drift + shock
            if self.rng.random() < cfg.jump_prob / cfg.ticks_per_bar:
                move += self.rng.gauss(0.0, cfg.jump_scale) * (1 if self.rng.random() < 0.5 else -1)
            price *= math.exp(move)
            high = max(high, price)
            low = min(low, price)
            # volume rises with absolute move - the usual vol/volume coupling
            volume += cfg.base_liquidity * (0.35 + abs(move) / max(tick_sigma, 1e-9) * 0.22)
            last_shock = move

        self.price = price
        self._step_variance(last_shock)
        candle = Candle(self.ts, open_price, high, low, price, volume)
        self.ts += cfg.bar_seconds
        self.bar_count += 1
        return MarketEvent(self.symbol, candle, self.make_book(price, sigma))

    # ------------------------------------------------------------------ book

    def make_book(self, mid: float, sigma: float) -> OrderBook:
        cfg = self.cfg
        half = mid * cfg.spread_bps / 10_000.0
        # liquidity thins out when volatility spikes - the reason slippage
        # is worst exactly when you most want out
        liq = cfg.base_liquidity * max(0.25, min(2.0, cfg.base_vol / max(sigma, 1e-9)))
        skew = self.rng.gauss(0.0, 0.18) + (0.25 if self.regime.name == "BULL" else -0.25 if self.regime.name == "BEAR" else 0.0)
        bids, asks = [], []
        for i in range(cfg.book_depth):
            step = half * (1 + i * 1.7)
            decay = math.exp(-0.22 * i)
            bids.append(BookLevel(mid - step, max(0.01, liq * decay * (1 + skew) * self.rng.uniform(0.7, 1.3))))
            asks.append(BookLevel(mid + step, max(0.01, liq * decay * (1 - skew) * self.rng.uniform(0.7, 1.3))))
        return OrderBook(self.ts, tuple(bids), tuple(asks))

    # ---------------------------------------------------------------- stream

    def stream(self) -> Iterator[MarketEvent]:
        n = 0
        while self.bars is None or n < self.bars:
            yield self.next_bar()
            n += 1

    def history(self, n: int) -> List[Candle]:
        """Pre-roll ``n`` bars, e.g. to warm indicators before trading starts."""
        return [self.next_bar().candle for _ in range(n)]
