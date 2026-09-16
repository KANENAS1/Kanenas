"""Drives the engine in real time and keeps the dashboards fed.

Separate from ``TradingEngine`` because pacing is a *presentation* concern: the
engine processes whatever bar you hand it as fast as you hand it over.  This is
what makes a simulated session watchable (``--speed``) without any notion of
wall-clock leaking into the trading logic, where it would make backtests
non-deterministic.
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

from .data.base import MarketEvent
from .engine import TradingEngine
from .risk.watchdog import FeedHealth, FeedWatchdog, WatchdogConfig
from .ui.dashboard import Dashboard


@dataclass
class RunnerConfig:
    speed: float = 8.0          # bars per second for simulated feeds (0 = as fast as possible)
    max_bars: Optional[int] = None
    render: bool = True
    render_every: int = 1
    mode: str = "PAPER"
    venue: str = "simulator"
    #: bar interval in seconds; enables the feed watchdog when > 0
    bar_seconds: float = 0.0


class LiveRunner:
    def __init__(self, engine: TradingEngine, feed, config: Optional[RunnerConfig] = None,
                 web=None, feed_factory: Optional[Callable[[str, str], object]] = None) -> None:
        self.engine = engine
        self.feed = feed
        #: builds a feed for (symbol, interval); enables switching at runtime
        self.feed_factory = feed_factory
        self.switch_request: Optional[Tuple[str, str]] = None
        self.cfg = config or RunnerConfig()
        self.web = web
        self.dashboard: Optional[Dashboard] = None
        self.stop_requested = False
        self.bars = 0
        self._last_event: Optional[MarketEvent] = None
        # Only a real-time feed can go stale; a simulator or a CSV cannot.
        self.watchdog: Optional[FeedWatchdog] = (
            FeedWatchdog(self.cfg.bar_seconds, WatchdogConfig())
            if self.cfg.bar_seconds > 0 else None)

    def request_stop(self, *_args) -> None:
        self.stop_requested = True

    def request_switch(self, symbol: str, interval: str) -> None:
        """Ask to trade a different instrument at the next opportunity."""
        self.switch_request = (symbol, interval)

    def _apply_switch(self, symbol: str, interval: str) -> None:
        from .cli import BARS_PER_YEAR
        if self._last_event is not None:
            self.engine.close_all(self._last_event)      # never carry a position across
        self.feed = self.feed_factory(symbol, interval)
        self.engine.switch_instrument(symbol, BARS_PER_YEAR.get(interval, 525_600.0))
        self.cfg.bar_seconds = getattr(self.feed, "bar_seconds", 0.0)
        self.watchdog = (FeedWatchdog(self.cfg.bar_seconds, WatchdogConfig())
                         if self.cfg.bar_seconds > 0 else None)
        history = getattr(self.feed, "fetch_history", None)
        if history is not None:
            candles = history(300)
            closed = candles[:-1] if len(candles) > 1 else candles
            self.engine.prime(closed)
            self.feed.mark_seen(closed[-1].ts)
        self.cfg.venue = getattr(self.feed, "name", self.cfg.venue)
        if self.web is not None:
            self.web.venue = self.cfg.venue

    def _consume(self, cfg, delay) -> None:
        """Drain the current feed until it ends, a stop, or a switch."""
        for event in self.feed.stream():
            if event is None:
                # heartbeat: no bar, but the feed is still talking to us.
                # This is the only moment a stalled feed can be noticed,
                # since nothing else runs while we wait for the next bar.
                self._watch()
                if self.dashboard:
                    self.dashboard.draw()
                if self.stop_requested or self.switch_request:
                    break
                continue

            self._last_event = event
            if self.watchdog:
                self.watchdog.record_bar()
                self._watch()
            self.engine.process(event)
            self.bars += 1
            if self.dashboard and self.bars % cfg.render_every == 0:
                self.dashboard.draw()
            if cfg.max_bars and self.bars >= cfg.max_bars:
                break
            if self.stop_requested or self.switch_request:
                break
            if delay:
                time.sleep(delay)

    def _watch(self) -> None:
        """Escalate on silence.

        A dead feed cannot be traded out of - with no prices, any exit would be
        an invented fill - so the responses available are to stop taking on new
        risk, to say so loudly, and (in the feed itself) to replay the missed
        bars once data returns so the stop is finally tested against them.
        """
        if self.watchdog is None:
            return
        health = self.watchdog.check()
        note = self.watchdog.take_announcement()
        self.engine.set_feed_health(health.value, health.can_open_positions, note)
        if health is FeedHealth.LOST and not self.engine.risk.halted:
            self.engine.halt(f"feed lost - no data for "
                             f"{self.watchdog.silence_bars:.0f} bar intervals")

    def run(self) -> TradingEngine:
        cfg = self.cfg
        if cfg.render:
            self.dashboard = Dashboard(self.engine, mode=cfg.mode, venue=cfg.venue)
            self.dashboard.enter()

        # Ctrl-C must flatten the book and restore the terminal, not just die.
        prev_int = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.request_stop)
        delay = (1.0 / cfg.speed) if cfg.speed and cfg.speed > 0 else 0.0

        try:
            while True:
                self._consume(cfg, delay)
                if self.switch_request is None or self.feed_factory is None:
                    break
                symbol, interval = self.switch_request
                self.switch_request = None
                self._apply_switch(symbol, interval)
        finally:
            signal.signal(signal.SIGINT, prev_int)
            if self._last_event is not None:
                # Flatten on shutdown: leaving a position open would report
                # unrealised P&L as if it had been banked.
                self.engine.close_all(self._last_event)
                self.engine.portfolio.mark(self._last_event.ts, self._last_event.candle.close)
            if self.dashboard:
                self.dashboard.draw()
                self.dashboard.exit()
        return self.engine
