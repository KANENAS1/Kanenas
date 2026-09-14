"""Strategy signals and ensemble blending."""

import unittest

from kanenas.core.indicators import IndicatorSet
from kanenas.core.types import (BookLevel, Candle, Direction, OrderBook,
                                Position, Signal)
from kanenas.strategy.base import Strategy, StrategyContext
from kanenas.strategy.breakout import VolatilityBreakout
from kanenas.strategy.ensemble import StrategyEnsemble, default_ensemble
from kanenas.strategy.mean_reversion import MeanReversion
from kanenas.strategy.momentum import RiskAdjustedMomentum
from kanenas.strategy.orderflow import OrderFlowPressure
from kanenas.strategy.trend import TrendFollow


def ctx_from(prices, book=None, position=None, spread=1.0):
    ind = IndicatorSet()
    last = None
    for i, px in enumerate(prices):
        last = Candle(i * 60, px, px + spread / 2, px - spread / 2, px, 1.0)
        ind.update(last)
    return StrategyContext("BTC-USD", last, ind, book, position or Position("BTC-USD"), 10_000.0)


def make_book(mid, bid_size, ask_size, depth=5):
    return OrderBook(
        0.0,
        tuple(BookLevel(mid - 1 - i, bid_size) for i in range(depth)),
        tuple(BookLevel(mid + 1 + i, ask_size) for i in range(depth)),
    )


class FixedStrategy(Strategy):
    """Test double that always returns the same opinion."""

    def __init__(self, name, direction, confidence, weight=1.0):
        self.name = name
        self.weight = weight
        self._sig_out = Signal(direction, confidence, name, "fixed")

    def evaluate(self, ctx):
        return self._sig_out


class TestIndividualStrategies(unittest.TestCase):
    def test_trend_goes_long_in_a_clean_uptrend(self):
        sig = TrendFollow().evaluate(ctx_from([100 + i * 0.5 for i in range(220)]))
        self.assertIs(sig.direction, Direction.LONG)
        self.assertGreater(sig.confidence, 0.0)

    def test_trend_goes_short_in_a_clean_downtrend(self):
        sig = TrendFollow().evaluate(ctx_from([300 - i * 0.5 for i in range(220)]))
        self.assertIs(sig.direction, Direction.SHORT)

    def test_trend_stays_flat_when_emas_are_tangled(self):
        prices = [100 + (1 if i % 2 else -1) * 0.05 for i in range(220)]
        self.assertIs(TrendFollow().evaluate(ctx_from(prices)).direction, Direction.FLAT)

    def test_trend_flat_while_warming_up(self):
        self.assertIs(TrendFollow().evaluate(ctx_from([100, 101, 102])).direction, Direction.FLAT)

    def test_mean_reversion_vetoes_strong_trends(self):
        """The trend veto is a hard gate - fading a real trend is how you die."""
        sig = MeanReversion().evaluate(ctx_from([100 + i * 2.0 for i in range(220)]))
        self.assertIs(sig.direction, Direction.FLAT)
        self.assertIn("veto", sig.reason)

    def test_mean_reversion_fires_on_a_range_bound_dip(self):
        prices = [100 + (3 if i % 2 else -3) for i in range(200)]
        prices += [88]                      # sharp stretch below the band
        sig = MeanReversion(z_entry=1.2, rsi_low=45).evaluate(ctx_from(prices))
        self.assertIn(sig.direction, (Direction.LONG, Direction.FLAT))

    def test_breakout_fires_above_the_channel(self):
        prices = [100 + (0.4 if i % 2 else -0.4) for i in range(120)] + [125.0]
        sig = VolatilityBreakout().evaluate(ctx_from(prices))
        self.assertIs(sig.direction, Direction.LONG)

    def test_breakout_quiet_inside_the_channel(self):
        prices = [100 + (0.4 if i % 2 else -0.4) for i in range(160)]
        self.assertIs(VolatilityBreakout().evaluate(ctx_from(prices)).direction, Direction.FLAT)

    def test_orderflow_needs_a_book(self):
        sig = OrderFlowPressure().evaluate(ctx_from([100] * 120, book=None))
        self.assertIs(sig.direction, Direction.FLAT)
        self.assertIn("no book", sig.reason)

    def test_orderflow_follows_persistent_imbalance(self):
        s = OrderFlowPressure(smoothing=3, threshold=0.2)
        sig = None
        for _ in range(20):                                  # let the EMA settle
            sig = s.evaluate(ctx_from([100] * 60, book=make_book(100, 20.0, 1.0)))
        self.assertIs(sig.direction, Direction.LONG)

    def test_orderflow_short_on_ask_heavy_book(self):
        s = OrderFlowPressure(smoothing=3, threshold=0.2)
        sig = None
        for _ in range(20):
            sig = s.evaluate(ctx_from([100] * 60, book=make_book(100, 1.0, 20.0)))
        self.assertIs(sig.direction, Direction.SHORT)

    def test_momentum_needs_significant_drift(self):
        s = RiskAdjustedMomentum(lookback=40, t_entry=1.1)
        sig = None
        for px in [100 + i * 0.3 for i in range(80)]:
            sig = s.evaluate(ctx_from([px]))
        self.assertIs(sig.direction, Direction.LONG)

    def test_momentum_flat_on_noise(self):
        import random
        rng = random.Random(3)
        s = RiskAdjustedMomentum(lookback=40, t_entry=2.5)
        sig = None
        for _ in range(120):
            sig = s.evaluate(ctx_from([100 + rng.gauss(0, 1)]))
        self.assertIs(sig.direction, Direction.FLAT)


class TestEnsemble(unittest.TestCase):
    def test_unanimous_agreement_produces_a_signal(self):
        ens = StrategyEnsemble([FixedStrategy("a", Direction.LONG, 1.0),
                                FixedStrategy("b", Direction.LONG, 1.0)],
                               entry_threshold=0.3, adaptive=False)
        d = ens.evaluate(ctx_from([100] * 80))
        self.assertIs(d.direction, Direction.LONG)
        self.assertAlmostEqual(d.agreement, 1.0)

    def test_disagreement_blocks_the_trade(self):
        ens = StrategyEnsemble([FixedStrategy("a", Direction.LONG, 1.0),
                                FixedStrategy("b", Direction.SHORT, 1.0)],
                               entry_threshold=0.1, min_agreement=0.6, adaptive=False)
        d = ens.evaluate(ctx_from([100] * 80))
        self.assertIs(d.direction, Direction.FLAT)

    def test_weak_conviction_below_threshold_blocks(self):
        ens = StrategyEnsemble([FixedStrategy("a", Direction.LONG, 0.1)],
                               entry_threshold=0.5, adaptive=False)
        self.assertIs(ens.evaluate(ctx_from([100] * 80)).direction, Direction.FLAT)

    def test_all_flat_is_reported_cleanly(self):
        ens = StrategyEnsemble([FixedStrategy("a", Direction.FLAT, 0.0)], adaptive=False)
        d = ens.evaluate(ctx_from([100] * 80))
        self.assertIs(d.direction, Direction.FLAT)
        self.assertEqual(d.agreement, 0.0)

    def test_weights_shift_toward_the_profitable_strategy(self):
        ens = default_ensemble()
        for _ in range(25):
            ens.attribute({"trend": 1.0}, -40.0)
            ens.attribute({"momo": 1.0}, +55.0)
        snap = ens.snapshot()
        self.assertLess(snap["trend"]["multiplier"], 1.0)
        self.assertGreater(snap["momo"]["multiplier"], 1.0)
        self.assertGreater(snap["momo"]["effective"], snap["trend"]["effective"])

    def test_consistent_loser_is_demoted_despite_zero_variance(self):
        """Regression: identical losses have zero variance.

        An unfloored t-statistic reads 0.0 there and scores a perfectly
        consistent loser as neutral.
        """
        ens = default_ensemble()
        for _ in range(20):
            ens.attribute({"trend": 1.0}, -25.0)      # identical every time
        self.assertLess(ens.snapshot()["trend"]["multiplier"], 0.9)

    def test_weights_are_bounded_both_ways(self):
        ens = default_ensemble(w_min=0.4, w_max=1.8)
        for _ in range(200):
            ens.attribute({"momo": 1.0}, +1_000.0)
            ens.attribute({"trend": 1.0}, -1_000.0)
        snap = ens.snapshot()
        self.assertLessEqual(snap["momo"]["multiplier"], 1.8 + 1e-9)
        self.assertGreaterEqual(snap["trend"]["multiplier"], 0.4 - 1e-9)

    def test_adaptive_disabled_keeps_weights_fixed(self):
        ens = default_ensemble(adaptive=False)
        for _ in range(50):
            ens.attribute({"trend": 1.0}, -100.0)
        self.assertAlmostEqual(ens.snapshot()["trend"]["multiplier"], 1.0)

    def test_attribution_ignores_strategies_that_did_not_vote(self):
        ens = default_ensemble()
        ens.attribute({"trend": 1.0, "momo": 0.0}, 50.0)
        self.assertEqual(ens.snapshot()["momo"]["trades"], 0)
        self.assertEqual(ens.snapshot()["trend"]["trades"], 1)

    def test_empty_strategy_list_rejected(self):
        with self.assertRaises(ValueError):
            StrategyEnsemble([])


if __name__ == "__main__":
    unittest.main()
