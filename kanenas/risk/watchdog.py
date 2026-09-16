"""Treat silence as a signal.

Exits are only evaluated when a bar arrives, so a feed that stops delivering
leaves an open position with an unenforced stop: price can travel straight
through it and nothing reacts.  The dashboard meanwhile keeps showing the last
known price, which looks exactly like a calm market.

Two things follow, and only one of them is obvious.

The obvious one is to notice and say so.  The less obvious one is what a bot
*cannot* do about it: with no data there is no price, so "flatten on the feed
dying" is not available - any fill would be invented.  The honest responses are
to stop taking on new risk, to shout, and - the part that actually protects the
position - to **replay the bars that were missed** once data returns, so the
stop is evaluated against the prices it would have triggered at rather than
against whatever the market happens to be doing on reconnect.

State escalates by how long the silence has run, measured in bar intervals
rather than seconds so the same thresholds hold on 1m and 1h candles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class FeedHealth(str, Enum):
    STARTING = "STARTING"   # nothing received yet
    HEALTHY = "HEALTHY"     # a bar arrived when expected
    LATE = "LATE"           # overdue, but within tolerance
    STALE = "STALE"         # no new risk taken on
    LOST = "LOST"           # long outage; trading stops until an operator looks

    @property
    def can_open_positions(self) -> bool:
        return self in (FeedHealth.HEALTHY, FeedHealth.LATE)


@dataclass
class WatchdogConfig:
    #: a bar is "late" once this many intervals have passed with nothing
    late_after: float = 1.5
    #: stop opening positions after this long without data
    stale_after: float = 3.0
    #: treat the feed as lost, and halt, after this long
    lost_after: float = 15.0
    #: never replay more than this many missed bars on reconnect; a longer gap
    #: means the market has moved on and the position needs a human, not a bot
    max_backfill_bars: int = 500


class FeedWatchdog:
    """Tracks the gap since the last bar and escalates."""

    def __init__(self, bar_seconds: float, config: Optional[WatchdogConfig] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.bar_seconds = max(1.0, bar_seconds)
        self.cfg = config or WatchdogConfig()
        self.clock = clock
        self.last_bar_at: Optional[float] = None
        self.health = FeedHealth.STARTING
        self.gaps = 0
        self.longest_gap_bars = 0.0
        self._announced = FeedHealth.STARTING

    def record_bar(self) -> None:
        self.last_bar_at = self.clock()
        self.health = FeedHealth.HEALTHY

    @property
    def silence_bars(self) -> float:
        """How many bar intervals since the last bar arrived."""
        if self.last_bar_at is None:
            return 0.0
        return (self.clock() - self.last_bar_at) / self.bar_seconds

    def check(self) -> FeedHealth:
        if self.last_bar_at is None:
            return self.health
        silence = self.silence_bars
        self.longest_gap_bars = max(self.longest_gap_bars, silence)
        cfg = self.cfg
        if silence >= cfg.lost_after:
            health = FeedHealth.LOST
        elif silence >= cfg.stale_after:
            health = FeedHealth.STALE
        elif silence >= cfg.late_after:
            health = FeedHealth.LATE
        else:
            health = FeedHealth.HEALTHY
        if health is not FeedHealth.HEALTHY and self.health is FeedHealth.HEALTHY:
            self.gaps += 1
        self.health = health
        return health

    def take_announcement(self) -> Optional[str]:
        """Return a message the first time each state is entered, else None.

        Keeps the log readable: a ten-minute outage should produce a handful of
        lines, not one per poll.
        """
        if self.health is self._announced:
            return None
        self._announced = self.health
        bars = self.silence_bars
        if self.health is FeedHealth.HEALTHY:
            return "feed recovered - data flowing again"
        if self.health is FeedHealth.LATE:
            return f"bar overdue ({bars:.1f} intervals) - watching"
        if self.health is FeedHealth.STALE:
            return (f"FEED STALE - no data for {bars:.1f} intervals. No new positions. "
                    f"An open position's stop CANNOT be enforced without prices.")
        if self.health is FeedHealth.LOST:
            return (f"FEED LOST - no data for {bars:.1f} intervals. Trading halted; "
                    f"check the connection and restart.")
        return None
