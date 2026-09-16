"""End-to-end engine behaviour, including the properties that make a backtest
believable: no lookahead, stops honoured, determinism, flat at shutdown.
"""

import unittest

from kanenas.backtest import run_backtest, summarise
from kanenas.core.types import Candle, Direction, ExitReason, Side
from kanenas.data.base import MarketEvent
from kanenas.data.simulator import MarketSimulator, SimulatorConfig
from kanenas.engine import EngineConfig, TradingEngine
from kanenas.execution.broker import ExecutionConfig, PaperBroker
from kanenas.risk.manager import RiskConfig, RiskManager
from kanenas.strategy.base import Strategy
from kanenas.strategy.ensemble import StrategyEnsemble
from kanenas.core.types import Signal


class AlwaysLong(Strategy):
    name = "always_long"

    def evaluate(self, ctx):
        return Signal(Direction.LONG, 1.0, self.name, "forced")


class AlwaysFlat(Strategy):
    name = "always_flat"

    def evaluate(self, ctx):
        return Signal.flat(self.name)


def forced_engine(strategy=None, risk=None, cash=10_000.0, warmup=5):
    return TradingEngine(
        EngineConfig(starting_cash=cash, warmup_bars=warmup),
        ensemble=StrategyEnsemble([strategy or AlwaysLong()], entry_threshold=0.1, adaptive=False),
        risk=RiskManager(risk or RiskConfig(round_trip_cost_bps=0.0)),
        broker=PaperBroker(ExecutionConfig(seed=1, latency_bps=0.0, impact_coefficient=0.0,
                                           taker_fee_bps=0.0, half_spread_bps=0.0)),
    )


def flat_then(prices, spread=1.0):
    """Turn a close series into events with a small symmetric range."""
    out = []
    for i, px in enumerate(prices):
        out.append(MarketEvent("BTC-USD", Candle(i * 60.0, px, px + spread / 2,
                                                 px - spread / 2, px, 1.0), None))
    return out


class TestEngineBasics(unittest.TestCase):
    def test_runs_and_produces_an_equity_curve(self):
        eng = TradingEngine(EngineConfig())
        eng.run(MarketSimulator(SimulatorConfig(seed=5), bars=600).stream())
        self.assertEqual(eng.state.bar, 600)
        self.assertEqual(len(eng.portfolio.equity_curve), 600)

    def test_no_trades_before_warmup_completes(self):
        eng = forced_engine(warmup=100)
        eng.run(MarketSimulator(SimulatorConfig(seed=5), bars=90).stream())
        self.assertEqual(len(eng.portfolio.trades), 0)
        self.assertFalse(eng.portfolio.position.is_open)

    def test_flat_strategy_never_trades(self):
        eng = forced_engine(strategy=AlwaysFlat())
        eng.run(MarketSimulator(SimulatorConfig(seed=5), bars=800).stream())
        self.assertEqual(len(eng.portfolio.trades), 0)

    def test_identical_seeds_give_identical_results(self):
        def run():
            eng = TradingEngine(EngineConfig(), broker=PaperBroker(ExecutionConfig(seed=3)))
            eng.run(MarketSimulator(SimulatorConfig(seed=77), bars=1_200).stream())
            return [t.net_pnl for t in eng.portfolio.trades], eng.portfolio.cash
        self.assertEqual(run(), run())

    def test_close_all_flattens_the_book(self):
        events = flat_then([100 + i * 0.5 for i in range(200)])
        eng = forced_engine()
        for e in events:
            eng.process(e)
        if eng.portfolio.position.is_open:
            eng.close_all(events[-1])
        self.assertFalse(eng.portfolio.position.is_open)


class TestLivePriming(unittest.TestCase):
    """A live feed only yields bars as they close, so a cold start means an
    empty chart that grows a candle a minute and an hour of dead time before
    the indicators are usable."""

    def history(self, n=300, seed=4):
        return [e.candle for e in MarketSimulator(SimulatorConfig(seed=seed), bars=n).stream()]

    def test_prime_warms_indicators_and_fills_the_chart(self):
        eng = TradingEngine(EngineConfig())
        self.assertFalse(eng.ind.warm)
        n = eng.prime(self.history())
        self.assertEqual(n, 300)
        self.assertTrue(eng.ind.warm)
        self.assertEqual(len(eng.ind.candles), 300)   # the chart opens full

    def test_prime_invents_no_trades_or_equity(self):
        """These bars already happened - trading them would fabricate a P&L."""
        eng = TradingEngine(EngineConfig())
        eng.prime(self.history())
        self.assertEqual(eng.portfolio.trades, [])
        self.assertEqual(eng.portfolio.equity_curve, [])
        self.assertEqual(eng.portfolio.cash, eng.portfolio.starting_cash)
        self.assertFalse(eng.portfolio.position.is_open)
        self.assertEqual(eng.portfolio.fees_paid, 0.0)

    def test_primed_engine_can_act_on_the_very_first_live_bar(self):
        primed = TradingEngine(EngineConfig())
        primed.prime(self.history())
        cold = TradingEngine(EngineConfig())
        nxt = next(iter(MarketSimulator(SimulatorConfig(seed=99), bars=1).stream()))
        primed.process(nxt)
        cold.process(nxt)
        self.assertIsNotNone(primed.state.decision)    # has an opinion immediately
        self.assertIsNone(cold.state.decision)         # still warming up

    def test_prime_reports_the_last_price(self):
        eng = TradingEngine(EngineConfig())
        hist = self.history()
        eng.prime(hist)
        self.assertAlmostEqual(eng.state.price, hist[-1].close)
        self.assertEqual(eng.state.bar, len(hist))

    def test_prime_of_nothing_is_a_noop(self):
        eng = TradingEngine(EngineConfig())
        self.assertEqual(eng.prime([]), 0)
        self.assertEqual(eng.state.bar, 0)

    def test_mark_seen_suppresses_already_primed_bars(self):
        from kanenas.data.rest import RestFeed
        feed = RestFeed("BTCUSDT", "binance", "1m")
        feed.mark_seen(1_700_000_000.0)
        self.assertEqual(feed._last_ts, 1_700_000_000.0)
        feed.mark_seen(1_600_000_000.0)                # never goes backwards
        self.assertEqual(feed._last_ts, 1_700_000_000.0)


class TestNoLookahead(unittest.TestCase):
    def test_decisions_do_not_depend_on_future_bars(self):
        """The defining property of an honest backtest.

        Run the engine over a prefix, then over the same prefix followed by
        wildly different future data.  Every decision taken inside the prefix
        must be byte-identical; if any of it changes, information from the
        future is reaching the strategies.
        """
        base = [e for e in MarketSimulator(SimulatorConfig(seed=31), bars=400).stream()]
        cut = 300

        def log_of(events):
            eng = TradingEngine(EngineConfig(), broker=PaperBroker(ExecutionConfig(seed=9)))
            for e in events:
                eng.process(e)
            return [(x.bar, x.kind, x.message) for x in eng.state.log if x.bar <= cut]

        prefix_only = log_of(base[:cut])
        # same prefix, then a violent crash the first run never saw
        crashed = list(base[:cut])
        px = base[cut - 1].candle.close
        for i in range(100):
            px *= 0.97
            crashed.append(MarketEvent("BTC-USD", Candle((cut + i) * 60.0, px, px * 1.001,
                                                         px * 0.999, px, 1.0), None))
        self.assertEqual(prefix_only, log_of(crashed))

    def test_indicators_only_see_closed_bars(self):
        eng = TradingEngine(EngineConfig())
        events = [e for e in MarketSimulator(SimulatorConfig(seed=8), bars=120).stream()]
        for e in events:
            eng.process(e)
        self.assertEqual(list(eng.ind.closes)[-1], events[-1].candle.close)
        self.assertEqual(eng.ind.bars, len(events))


class TestExitDiscipline(unittest.TestCase):
    def test_stop_is_triggered_by_the_bar_low_not_the_close(self):
        """A wick through the stop is a stop-out, even if the bar closes above."""
        eng = forced_engine()
        for e in flat_then([100.0] * 80):
            eng.process(e)
        self.assertTrue(eng.portfolio.position.is_open)
        stop = eng.portfolio.position.stop_price
        # a bar that dips below the stop but recovers to close unchanged
        eng.process(MarketEvent("BTC-USD", Candle(99 * 60.0, 100.0, 100.5,
                                                  stop - 5.0, 100.0, 1.0), None))
        self.assertTrue(eng.portfolio.trades)
        self.assertIn(eng.portfolio.trades[-1].reason,
                      (ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP))

    def test_when_stop_and_target_share_a_bar_the_stop_wins(self):
        """The pessimistic branch: we cannot know which came first."""
        eng = forced_engine()
        for e in flat_then([100.0] * 80):
            eng.process(e)
        pos = eng.portfolio.position
        self.assertTrue(pos.is_open)
        stop, target = pos.stop_price, pos.take_profit
        eng.process(MarketEvent("BTC-USD", Candle(99 * 60.0, 100.0, target + 5.0,
                                                  stop - 5.0, 100.0, 1.0), None))
        self.assertIn(eng.portfolio.trades[-1].reason,
                      (ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP))

    def test_time_stop_closes_a_stale_position(self):
        eng = forced_engine(risk=RiskConfig(max_bars_in_trade=10, round_trip_cost_bps=0.0,
                                            atr_stop_mult=50.0, atr_target_mult=50.0))
        for e in flat_then([100.0] * 120):
            eng.process(e)
        self.assertTrue(any(t.reason is ExitReason.TIME_STOP for t in eng.portfolio.trades))

    def test_exit_price_is_clamped_inside_the_bar(self):
        eng = forced_engine()
        for e in flat_then([100.0] * 80):
            eng.process(e)
        stop = eng.portfolio.position.stop_price
        bar = Candle(99 * 60.0, 100.0, 100.5, stop - 5.0, 100.0, 1.0)
        eng.process(MarketEvent("BTC-USD", bar, None))
        exit_px = eng.portfolio.trades[-1].exit_price
        self.assertGreaterEqual(exit_px, bar.low - 1e-9)
        self.assertLessEqual(exit_px, bar.high + 1e-9)

    def test_risk_halt_flattens_and_stops_trading(self):
        eng = forced_engine(risk=RiskConfig(max_drawdown=0.001, round_trip_cost_bps=0.0))
        for e in flat_then([100 - i * 0.5 for i in range(200)]):
            eng.process(e)
        self.assertTrue(eng.risk.halted)
        self.assertFalse(eng.portfolio.position.is_open)


class TestOperatorControls(unittest.TestCase):
    def engine_with_position(self, bars=820, seed=2024):
        eng = TradingEngine(EngineConfig())
        events = list(MarketSimulator(SimulatorConfig(seed=seed), bars=bars).stream())
        for e in events:
            eng.process(e)
        return eng, events

    def test_pause_blocks_new_entries(self):
        eng, events = self.engine_with_position(600)
        eng.pause()
        before = len(eng.portfolio.trades)
        opened = eng.portfolio.position.is_open
        for e in MarketSimulator(SimulatorConfig(seed=31), bars=150).stream():
            eng.process(e)
        if not opened:
            self.assertFalse(eng.portfolio.position.is_open)
        self.assertTrue(any("paused" in x.message for x in eng.state.log))

    def test_pause_keeps_managing_an_open_position(self):
        """Pause must not abandon risk management on a live position."""
        eng = TradingEngine(EngineConfig())
        events = list(MarketSimulator(SimulatorConfig(seed=2024), bars=900).stream())
        for e in events:
            eng.process(e)
            if eng.portfolio.position.is_open:
                break
        self.assertTrue(eng.portfolio.position.is_open)
        eng.pause()
        stop, target = eng.portfolio.position.stop_price, eng.portfolio.position.take_profit
        self.assertGreater(stop, 0)
        self.assertGreater(target, 0)
        for e in MarketSimulator(SimulatorConfig(seed=55), bars=300).stream():
            eng.process(e)
            if not eng.portfolio.position.is_open:
                break
        self.assertFalse(eng.portfolio.position.is_open)   # an exit still fired

    def test_resume_restores_trading(self):
        eng, _ = self.engine_with_position(600)
        eng.pause(); eng.resume()
        self.assertFalse(eng.paused)

    def test_flatten_closes_on_the_next_bar_and_does_not_re_enter(self):
        eng = TradingEngine(EngineConfig())
        events = list(MarketSimulator(SimulatorConfig(seed=2024), bars=900).stream())
        idx = 0
        for i, e in enumerate(events):
            eng.process(e)
            if eng.portfolio.position.is_open:
                idx = i
                break
        self.assertTrue(eng.request_flatten())
        eng.process(events[idx + 1])
        self.assertFalse(eng.portfolio.position.is_open)
        self.assertIs(eng.portfolio.trades[-1].reason, ExitReason.MANUAL)

    def test_flatten_when_flat_reports_nothing_to_do(self):
        eng = TradingEngine(EngineConfig())
        self.assertFalse(eng.request_flatten())

    def test_halt_stops_trading_and_flattens(self):
        eng, events = self.engine_with_position(820)
        eng.halt("test kill switch")
        self.assertTrue(eng.risk.halted)
        self.assertTrue(eng.paused)
        for e in MarketSimulator(SimulatorConfig(seed=7), bars=120).stream():
            eng.process(e)
        self.assertFalse(eng.portfolio.position.is_open)


class TestReporting(unittest.TestCase):
    def test_report_fields_are_consistent(self):
        eng, rep = run_backtest(MarketSimulator(SimulatorConfig(seed=17), bars=2_000).stream(),
                                EngineConfig())
        self.assertEqual(rep.trades, len(eng.portfolio.trades))
        self.assertAlmostEqual(rep.final_equity, eng.portfolio.equity_curve[-1].equity)
        self.assertGreaterEqual(rep.win_rate, 0.0)
        self.assertLessEqual(rep.win_rate, 1.0)
        self.assertGreaterEqual(rep.max_drawdown, 0.0)
        self.assertGreater(rep.sample_days, 0.0)

    def test_short_samples_are_flagged_not_annualised(self):
        _, rep = run_backtest(MarketSimulator(SimulatorConfig(seed=17), bars=1_000).stream(),
                              EngineConfig())
        self.assertLess(rep.sample_days, 30)
        self.assertTrue(any("short" in c.lower() or "Annualised" in c for c in rep.caveats))
        self.assertNotIn("Annualised        +", rep.render())

    def test_report_warns_when_the_cap_sets_the_size(self):
        """A user reading --risk deserves to know when it is not the binding rule."""
        _, rep = run_backtest(MarketSimulator(SimulatorConfig(seed=2024), bars=3_000).stream(),
                              EngineConfig())
        self.assertIn("position_cap", rep.sized_by)
        self.assertTrue(any("exposure cap" in c for c in rep.caveats))

    def test_backtest_leaves_no_open_position(self):
        eng, rep = run_backtest(MarketSimulator(SimulatorConfig(seed=21), bars=1_500).stream(),
                                EngineConfig())
        self.assertFalse(eng.portfolio.position.is_open)

    def test_render_does_not_crash_with_zero_trades(self):
        eng = forced_engine(strategy=AlwaysFlat())
        _, rep = run_backtest(MarketSimulator(SimulatorConfig(seed=4), bars=300).stream(),
                              eng.cfg, engine=eng)
        self.assertEqual(rep.trades, 0)
        self.assertIsInstance(rep.render(), str)


if __name__ == "__main__":
    unittest.main()
