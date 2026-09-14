"""Risk sizing, caps, kill switches and stop management."""

import unittest

from kanenas.core.indicators import IndicatorSet
from kanenas.core.types import Candle, Direction, Position
from kanenas.risk.manager import RiskConfig, RiskManager


def warm_indicators(atr_target=2.0, price=100.0, bars=80):
    """Build an IndicatorSet whose ATR is exactly ``atr_target``."""
    ind = IndicatorSet()
    half = atr_target / 2
    for i in range(bars):
        ind.update(Candle(i * 60, price, price + half, price - half, price, 1.0))
    return ind


class TestSizing(unittest.TestCase):
    def setUp(self):
        self.ind = warm_indicators()
        self.rm = RiskManager(RiskConfig(risk_per_trade=0.01, atr_stop_mult=2.0,
                                         round_trip_cost_bps=0.0))

    def test_atr_is_what_the_test_expects(self):
        self.assertAlmostEqual(self.ind.atr.value, 2.0)

    def test_risks_exactly_the_configured_fraction(self):
        d = self.rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, self.ind, 10)
        self.assertTrue(d.approved)
        self.assertAlmostEqual(d.risk_amount, 100.0)      # 1% of 10k
        self.assertAlmostEqual(d.stop, 96.0)              # 2 x ATR below

    def test_short_stop_is_above_entry(self):
        d = self.rm.evaluate_entry(Direction.SHORT, 1.0, 100.0, 10_000.0, self.ind, 10)
        self.assertGreater(d.stop, 100.0)
        self.assertLess(d.target, 100.0)

    def test_lower_confidence_risks_less(self):
        hi = self.rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, self.ind, 10)
        lo = self.rm.evaluate_entry(Direction.LONG, 0.4, 100.0, 10_000.0, self.ind, 10)
        self.assertLess(lo.risk_amount, hi.risk_amount)

    def test_notional_cap_binds(self):
        rm = RiskManager(RiskConfig(risk_per_trade=0.9, max_position_pct=0.35,
                                    round_trip_cost_bps=0.0))
        d = rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, self.ind, 10)
        self.assertAlmostEqual(d.qty * 100.0, 3_500.0)

    def test_no_leverage_beyond_equity(self):
        rm = RiskManager(RiskConfig(risk_per_trade=5.0, max_position_pct=10.0,
                                    leverage=1.0, round_trip_cost_bps=0.0))
        d = rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 1_000.0, self.ind, 10)
        self.assertLessEqual(d.qty * 100.0, 1_000.0 + 1e-6)

    def test_flat_direction_rejected(self):
        self.assertFalse(self.rm.evaluate_entry(Direction.FLAT, 1.0, 100.0, 1e4, self.ind, 1).approved)

    def test_low_confidence_rejected(self):
        rm = RiskManager(RiskConfig(min_confidence=0.5, round_trip_cost_bps=0.0))
        self.assertFalse(rm.evaluate_entry(Direction.LONG, 0.2, 100.0, 1e4, self.ind, 1).approved)

    def test_cold_indicators_rejected(self):
        self.assertFalse(self.rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 1e4,
                                                IndicatorSet(), 1).approved)

    def test_tiny_notional_rejected(self):
        d = self.rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 1.0, self.ind, 10)
        self.assertFalse(d.approved)
        self.assertIn("minimum", d.reason)


class TestCostGate(unittest.TestCase):
    def test_thin_edge_rejected_when_costs_are_high(self):
        """A target that barely clears the round trip is not a trade."""
        ind = warm_indicators(atr_target=0.2, price=100.0)   # target = 3.2 * 0.2 = 0.64 => 64bp
        rm = RiskManager(RiskConfig(round_trip_cost_bps=40.0, min_edge_over_cost=2.5))
        d = rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, ind, 10)
        self.assertFalse(d.approved)
        self.assertIn("edge too thin", d.reason)

    def test_fat_edge_passes_the_same_gate(self):
        ind = warm_indicators(atr_target=5.0, price=100.0)   # target = 16.0 => 1600bp
        rm = RiskManager(RiskConfig(round_trip_cost_bps=40.0, min_edge_over_cost=2.5))
        self.assertTrue(rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, ind, 10).approved)

    def test_zero_cost_disables_the_gate(self):
        ind = warm_indicators(atr_target=0.01, price=100.0)
        rm = RiskManager(RiskConfig(round_trip_cost_bps=0.0))
        self.assertTrue(rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, ind, 10).approved)


class TestKillSwitches(unittest.TestCase):
    def setUp(self):
        self.ind = warm_indicators()

    def test_drawdown_halt(self):
        rm = RiskManager(RiskConfig(max_drawdown=0.20))
        rm.on_bar(0.0, 7_900.0, 10_000.0, 1)
        self.assertTrue(rm.halted)
        self.assertFalse(rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 7_900.0, self.ind, 2).approved)

    def test_drawdown_halt_does_not_clear_next_day(self):
        rm = RiskManager(RiskConfig(max_drawdown=0.20))
        rm.on_bar(0.0, 7_900.0, 10_000.0, 1)
        rm.on_bar(86_400.0 * 3, 7_900.0, 10_000.0, 2)     # a later day
        self.assertTrue(rm.halted)

    def test_daily_loss_halt_clears_on_a_new_day(self):
        rm = RiskManager(RiskConfig(daily_loss_limit=0.05, max_drawdown=0.99))
        rm.on_bar(0.0, 10_000.0, 10_000.0, 1)
        rm.on_bar(3_600.0, 9_400.0, 10_000.0, 2)          # -6% same day
        self.assertTrue(rm.halted)
        rm.on_bar(86_400.0 * 2, 9_400.0, 10_000.0, 3)     # new day
        self.assertFalse(rm.halted)

    def test_loss_streak_triggers_cooldown(self):
        rm = RiskManager(RiskConfig(loss_streak_cooldown=3, cooldown_bars=5,
                                    round_trip_cost_bps=0.0))
        rm.note_trade_closed(is_win=False, loss_streak=3, bar_index=10)
        d = rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, self.ind, 12)
        self.assertFalse(d.approved)
        self.assertIn("cooldown", d.reason)
        self.assertTrue(rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 1e4, self.ind, 16).approved)

    def test_wins_do_not_trigger_cooldown(self):
        rm = RiskManager(RiskConfig(round_trip_cost_bps=0.0))
        rm.note_trade_closed(is_win=True, loss_streak=0, bar_index=10)
        self.assertTrue(rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 1e4, self.ind, 11).approved)

    def test_binding_constraint_is_recorded(self):
        """Which rule set the size is diagnostic information, not a detail.

        On tight ATR stops the exposure cap binds instead of the risk budget,
        so effective risk is far below the configured fraction.
        """
        rm = RiskManager(RiskConfig(risk_per_trade=0.9, max_position_pct=0.35,
                                    round_trip_cost_bps=0.0))
        rm.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, self.ind, 10)
        self.assertEqual(rm.sized_by.get("position_cap"), 1)

        loose = RiskManager(RiskConfig(risk_per_trade=0.0001, max_position_pct=0.9,
                                       round_trip_cost_bps=0.0))
        loose.evaluate_entry(Direction.LONG, 1.0, 100.0, 10_000.0, self.ind, 10)
        self.assertEqual(loose.sized_by.get("risk_budget"), 1)

    def test_rejections_are_tallied(self):
        rm = RiskManager(RiskConfig(min_confidence=0.9, round_trip_cost_bps=0.0))
        for _ in range(3):
            rm.evaluate_entry(Direction.LONG, 0.1, 100.0, 1e4, self.ind, 1)
        self.assertEqual(sum(rm.rejections.values()), 3)


class TestTrailingStop(unittest.TestCase):
    def setUp(self):
        self.ind = warm_indicators()
        self.rm = RiskManager(RiskConfig(atr_stop_mult=2.0, trailing_atr_mult=2.0,
                                         trail_activate_r=1.0))

    def test_trails_up_and_never_back_down(self):
        pos = Position("BTC-USD", qty=1.0, entry_price=100.0, stop_price=96.0, peak_price=100.0)
        self.rm.update_trailing_stop(pos, 104.0, self.ind)   # +1R -> trail activates
        first = pos.stop_price
        self.assertGreater(first, 96.0)
        self.rm.update_trailing_stop(pos, 110.0, self.ind)
        self.assertGreater(pos.stop_price, first)
        high_water = pos.stop_price
        self.rm.update_trailing_stop(pos, 101.0, self.ind)   # pullback
        self.assertAlmostEqual(pos.stop_price, high_water)   # stop does not loosen

    def test_no_trail_before_activation_threshold(self):
        pos = Position("BTC-USD", qty=1.0, entry_price=100.0, stop_price=96.0, peak_price=100.0)
        self.rm.update_trailing_stop(pos, 101.0, self.ind)   # only +0.25R
        self.assertAlmostEqual(pos.stop_price, 96.0)

    def test_short_trails_down(self):
        pos = Position("BTC-USD", qty=-1.0, entry_price=100.0, stop_price=104.0, trough_price=100.0)
        self.rm.update_trailing_stop(pos, 96.0, self.ind)
        self.assertLess(pos.stop_price, 104.0)
        tight = pos.stop_price
        self.rm.update_trailing_stop(pos, 99.0, self.ind)
        self.assertAlmostEqual(pos.stop_price, tight)

    def test_flat_position_is_a_noop(self):
        pos = Position("BTC-USD")
        self.assertEqual(self.rm.update_trailing_stop(pos, 100.0, self.ind), 0.0)


if __name__ == "__main__":
    unittest.main()
