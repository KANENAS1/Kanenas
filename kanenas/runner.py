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
from typing import Iterator, Optional

from .data.base import MarketEvent
from .engine import TradingEngine
from .ui.dashboard import Dashboard


@dataclass
class RunnerConfig:
    speed: float = 8.0          # bars per second for simulated feeds (0 = as fast as possible)
    max_bars: Optional[int] = None
    render: bool = True
    render_every: int = 1
    mode: str = "PAPER"
    venue: str = "simulator"


class LiveRunner:
    def __init__(self, engine: TradingEngine, feed, config: Optional[RunnerConfig] = None,
                 web=None) -> None:
        self.engine = engine
        self.feed = feed
        self.cfg = config or RunnerConfig()
        self.web = web
        self.dashboard: Optional[Dashboard] = None
        self.stop_requested = False
        self.bars = 0
        self._last_event: Optional[MarketEvent] = None

    def request_stop(self, *_args) -> None:
        self.stop_requested = True

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
            for event in self.feed.stream():
                self._last_event = event
                self.engine.process(event)
                self.bars += 1
                if self.dashboard and self.bars % cfg.render_every == 0:
                    self.dashboard.draw()
                if cfg.max_bars and self.bars >= cfg.max_bars:
                    break
                if self.stop_requested:
                    break
                if delay:
                    time.sleep(delay)
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
