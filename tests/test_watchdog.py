"""Feed watchdog: treating silence as a signal.

Exits only run when a bar arrives, so a stalled feed leaves an open position
with an unenforced stop while the dashboard keeps showing a last-known price
that looks like a calm market.
"""

import unittest

from kanenas.core.types import Candle, ExitReason
from kanenas.data.base import MarketEvent
from kanenas.data.simulator import MarketSimulator, SimulatorConfig
from kanenas.engine import EngineConfig, TradingEngine
from kanenas.risk.watchdog import FeedHealth, FeedWatchdog, WatchdogConfig
from kanenas.runner import LiveRunner, RunnerConfig


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TestWatchdogStates(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.w = FeedWatchdog(60.0, WatchdogConfig(late_after=1.5, stale_after=3.0,
                                                   lost_after=15.0), clock=self.clock)

    def test_starts_before_any_bar(self):
        self.assertIs(self.w.health, FeedHealth.STARTING)
        self.assertIs(self.w.check(), FeedHealth.STARTING)

    def test_escalates_with_silence(self):
        self.w.record_bar()
        for seconds, expected in ((30, FeedHealth.HEALTHY), (60, FeedHealth.LATE),
                                  (120, FeedHealth.STALE), (900, FeedHealth.LOST)):
            self.clock.advance(seconds)
            self.assertIs(self.w.check(), expected, f"after {self.w.silence_bars:.1f} bars")

    def test_a_new_bar_clears_the_alarm(self):
        self.w.record_bar()
        self.clock.advance(60 * 20)          # 20 bar intervals, past lost_after=15
        self.assertIs(self.w.check(), FeedHealth.LOST)
        self.w.record_bar()
        self.assertIs(self.w.check(), FeedHealth.HEALTHY)

    def test_only_healthy_and_late_may_open_positions(self):
        self.assertTrue(FeedHealth.HEALTHY.can_open_positions)
        self.assertTrue(FeedHealth.LATE.can_open_positions)
        self.assertFalse(FeedHealth.STALE.can_open_positions)
        self.assertFalse(FeedHealth.LOST.can_open_positions)

    def test_thresholds_are_in_bar_intervals_not_seconds(self):
        """The same config must behave identically on 1m and 1h candles."""
        slow = FeedWatchdog(3600.0, WatchdogConfig(stale_after=3.0), clock=self.clock)
        slow.record_bar()
        self.clock.advance(3600 * 2)
        self.assertIs(slow.check(), FeedHealth.LATE)
        self.clock.advance(3600 * 2)
        self.assertIs(slow.check(), FeedHealth.STALE)

    def test_announces_each_state_once(self):
        self.w.record_bar()
        self.clock.advance(300)
        self.w.check()
        first = self.w.take_announcement()
        self.assertIsNotNone(first)
        self.assertIn("STALE", first)
        self.w.check()
        self.assertIsNone(self.w.take_announcement())   # no log spam while it persists

    def test_counts_gaps(self):
        self.w.record_bar()
        self.clock.advance(300); self.w.check()
        self.w.record_bar(); self.w.check()
        self.clock.advance(300); self.w.check()
        self.assertEqual(self.w.gaps, 2)


class TestStaleDataGuard(unittest.TestCase):
    def events(self, n=900, seed=2024):
        return list(MarketSimulator(SimulatorConfig(seed=seed), bars=n).stream())

    def test_stale_feed_blocks_new_entries(self):
        eng = TradingEngine(EngineConfig())
        events = self.events()
        eng.prime([e.candle for e in events[:400]])
        eng.set_feed_health("STALE", can_open=False, note="feed stale")
        for e in events[400:520]:
            eng.process(e)
        self.assertFalse(eng.portfolio.position.is_open)
        self.assertTrue(any("stale data" in l.message for l in eng.state.log))

    def test_recovery_allows_trading_again(self):
        eng = TradingEngine(EngineConfig())
        events = self.events()
        eng.prime([e.candle for e in events[:400]])
        eng.set_feed_health("STALE", can_open=False)
        for e in events[400:450]:
            eng.process(e)
        eng.set_feed_health("HEALTHY", can_open=True)
        for e in events[450:600]:
            eng.process(e)
        self.assertTrue(eng.portfolio.trades or eng.portfolio.position.is_open)


class TestBackfill(unittest.TestCase):
    """Bars replayed after an outage must close positions but never open them."""

    def open_position(self):
        eng = TradingEngine(EngineConfig())
        events = list(MarketSimulator(SimulatorConfig(seed=2024), bars=900).stream())
        for i, e in enumerate(events):
            eng.process(e)
            if eng.portfolio.position.is_open:
                return eng, events, i
        self.fail("no position opened")

    def test_backfilled_bar_can_trigger_a_stop(self):
        """The entire reason for replaying the gap."""
        eng, events, idx = self.open_position()
        pos = eng.portfolio.position
        crash = Candle(events[idx + 1].candle.ts, pos.entry_price, pos.entry_price,
                       pos.stop_price - 200, pos.stop_price - 150, 1.0)
        eng.process(MarketEvent("BTC-USD", crash, None, backfill=True))
        self.assertFalse(eng.portfolio.position.is_open)
        self.assertIn(eng.portfolio.trades[-1].reason,
                      (ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP))

    def test_backfilled_bars_never_open_a_position(self):
        """Entering at a price from half an hour ago is a fill that never was."""
        eng = TradingEngine(EngineConfig())
        events = list(MarketSimulator(SimulatorConfig(seed=2024), bars=900).stream())
        eng.prime([e.candle for e in events[:600]])
        for e in events[600:700]:
            eng.process(MarketEvent(e.symbol, e.candle, e.book, backfill=True))
        self.assertFalse(eng.portfolio.position.is_open)
        self.assertTrue(any("backfilled bar" in l.message for l in eng.state.log))

    def test_live_bars_still_open_positions(self):
        eng = TradingEngine(EngineConfig())
        events = list(MarketSimulator(SimulatorConfig(seed=2024), bars=900).stream())
        eng.prime([e.candle for e in events[:600]])
        for e in events[600:750]:
            eng.process(e)
        self.assertTrue(eng.portfolio.trades or eng.portfolio.position.is_open)


class TestRunnerWatchdog(unittest.TestCase):
    class FlakyFeed:
        symbol, name, bar_seconds = "BTC-USD", "flaky", 60.0

        def __init__(self, events, silence=40):
            self.events, self.silence = events, silence

        def stream(self):
            for e in self.events[:10]:
                yield e
            for _ in range(self.silence):
                yield None
            for e in self.events[10:20]:
                yield e

    def test_runner_halts_on_a_lost_feed(self):
        events = list(MarketSimulator(SimulatorConfig(seed=2024), bars=900).stream())
        eng = TradingEngine(EngineConfig())
        eng.prime([e.candle for e in events[:400]])
        runner = LiveRunner(eng, self.FlakyFeed(events[400:430]),
                            RunnerConfig(speed=0, render=False, bar_seconds=60.0))
        clock = Clock()
        runner.watchdog = FeedWatchdog(60.0, WatchdogConfig(), clock=clock)
        original = runner._watch
        def watch():
            clock.advance(60)      # each heartbeat is a bar interval of silence
            original()
        runner._watch = watch
        runner.run()
        self.assertGreater(runner.watchdog.longest_gap_bars, 15.0)
        self.assertTrue(any("FEED LOST" in l.message for l in eng.state.log))

    def test_heartbeats_are_not_traded(self):
        eng = TradingEngine(EngineConfig())
        events = list(MarketSimulator(SimulatorConfig(seed=5), bars=300).stream())
        runner = LiveRunner(eng, self.FlakyFeed(events, silence=5),
                            RunnerConfig(speed=0, render=False, bar_seconds=60.0))
        runner.run()
        self.assertEqual(runner.bars, 20)      # 20 real bars, 5 heartbeats ignored


if __name__ == "__main__":
    unittest.main()
