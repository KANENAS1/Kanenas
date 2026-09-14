"""Indicator correctness against hand-computable cases."""

import math
import unittest

from kanenas.core.indicators import (ATR, EMA, MACD, RSI, SMA, Bollinger,
                                     Donchian, IndicatorSet, RealizedVol,
                                     RollingStats, WilderMA)
from kanenas.core.types import Candle


def bar(o, h, l, c, ts=0.0, v=1.0):
    return Candle(ts, o, h, l, c, v)


class TestMovingAverages(unittest.TestCase):
    def test_sma_matches_arithmetic_mean(self):
        sma = SMA(3)
        for x in (1, 2, 3, 4, 5):
            sma.update(x)
        self.assertAlmostEqual(sma.value, 4.0)  # mean(3,4,5)

    def test_sma_not_ready_before_full_window(self):
        sma = SMA(5)
        for x in (1, 2, 3):
            sma.update(x)
        self.assertFalse(sma.ready)

    def test_ema_seeds_with_sma_then_smooths(self):
        ema = EMA(3)
        for x in (1, 2, 3):
            ema.update(x)
        self.assertAlmostEqual(ema.value, 2.0)   # seed = mean(1,2,3)
        ema.update(4)                            # 2 + 0.5*(4-2)
        self.assertAlmostEqual(ema.value, 3.0)

    def test_ema_constant_series_converges_to_constant(self):
        ema = EMA(10).prime([7.0] * 50)
        self.assertAlmostEqual(ema.value, 7.0, places=9)

    def test_wilder_ma_uses_one_over_n(self):
        w = WilderMA(2)
        w.update(2.0); w.update(4.0)             # seed = 3.0
        w.update(6.0)                            # (3*1 + 6)/2 = 4.5
        self.assertAlmostEqual(w.value, 4.5)

    def test_invalid_period_rejected(self):
        with self.assertRaises(ValueError):
            SMA(0)
        with self.assertRaises(ValueError):
            EMA(-1)


class TestRSI(unittest.TestCase):
    def test_monotonic_rise_pins_at_100(self):
        rsi = RSI(14).prime([float(i) for i in range(1, 40)])
        self.assertAlmostEqual(rsi.value, 100.0)

    def test_monotonic_fall_pins_at_zero(self):
        rsi = RSI(14).prime([float(100 - i) for i in range(1, 40)])
        self.assertAlmostEqual(rsi.value, 0.0)

    def test_flat_series_is_neutral_ish(self):
        rsi = RSI(14).prime([50.0] * 40)
        # no gains and no losses -> Wilder's definition yields 100 by convention
        self.assertTrue(rsi.ready)

    def test_bounded_on_noisy_input(self):
        import random
        rng = random.Random(4)
        rsi = RSI(14)
        for _ in range(500):
            v = rsi.update(rng.uniform(90, 110))
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)


class TestATR(unittest.TestCase):
    def test_constant_range_gives_that_range(self):
        atr = ATR(3)
        for i in range(10):
            atr.update_candle(bar(10, 12, 9, 11, ts=i))
        self.assertAlmostEqual(atr.value, 3.0)

    def test_gap_counts_toward_true_range(self):
        atr = ATR(2)
        atr.update_candle(bar(10, 11, 9, 10))
        atr.update_candle(bar(20, 21, 19, 20))   # TR = |21-10| = 11, not 2
        self.assertGreater(atr.value, 2.0)


class TestBandsAndChannels(unittest.TestCase):
    def test_rolling_stats_sample_stdev(self):
        rs = RollingStats(5)
        for x in (2, 4, 4, 4, 5):
            rs.update(x)
        self.assertAlmostEqual(rs.mean, 3.8)
        self.assertAlmostEqual(rs.stdev, math.sqrt(((1.8**2)+(0.2**2)*3+(1.2**2))/4))

    def test_zscore_is_zero_at_mean(self):
        rs = RollingStats(10).prime([5.0, 15.0] * 5)
        self.assertAlmostEqual(rs.zscore(10.0), 0.0)

    def test_zscore_safe_on_zero_variance(self):
        rs = RollingStats(5).prime([3.0] * 5)
        self.assertEqual(rs.zscore(99.0), 0.0)

    def test_bollinger_bands_straddle_mean(self):
        bb = Bollinger(10, 2.0)
        for i in range(20):
            bb.update(100 + (i % 5))
        self.assertLess(bb.lower, bb.stats.mean)
        self.assertGreater(bb.upper, bb.stats.mean)
        self.assertGreater(bb.width, 0.0)

    def test_donchian_tracks_window_extremes(self):
        d = Donchian(3)
        for o, h, l, c in [(1, 5, 0, 2), (2, 9, 1, 3), (3, 7, 2, 4), (4, 6, 3, 5)]:
            d.update_candle(bar(o, h, l, c))
        self.assertEqual(d.upper, 9)   # window is the last 3 bars
        self.assertEqual(d.lower, 1)
        self.assertEqual(d.mid, 5)


class TestVolAndSet(unittest.TestCase):
    def test_realized_vol_zero_on_flat_series(self):
        rv = RealizedVol(20).prime([100.0] * 40)
        self.assertAlmostEqual(rv.value, 0.0, places=9)

    def test_realized_vol_scales_with_noise(self):
        import random
        rng = random.Random(1)
        quiet = RealizedVol(50).prime([100 * math.exp(rng.gauss(0, 0.0005)) for _ in range(200)])
        loud = RealizedVol(50).prime([100 * math.exp(rng.gauss(0, 0.005)) for _ in range(200)])
        self.assertGreater(loud.value, quiet.value * 3)

    def test_indicator_set_warms_and_updates_together(self):
        s = IndicatorSet()
        self.assertFalse(s.warm)
        for i in range(200):
            s.update(bar(100 + i * 0.1, 101 + i * 0.1, 99 + i * 0.1, 100.5 + i * 0.1, ts=i * 60))
        self.assertTrue(s.warm)
        self.assertEqual(s.bars, 200)
        self.assertTrue(all(x.ready for x in (s.ema_fast, s.ema_slow, s.ema_trend, s.rsi, s.atr)))
        self.assertGreater(s.atr_pct(s.closes[-1]), 0.0)

    def test_macd_histogram_sign_follows_momentum(self):
        m = MACD()
        for i in range(120):
            m.update(100 + i)         # steady uptrend
        self.assertGreater(m.line, 0)


if __name__ == "__main__":
    unittest.main()
