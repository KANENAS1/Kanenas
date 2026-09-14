"""Portfolio accounting and broker fill modelling.

These are the tests that matter most: a bug here does not make the bot trade
badly, it makes the bot *lie about how it traded*.
"""

import unittest

from kanenas.core.types import (Direction, ExitReason, Fill, Order, OrderBook,
                                BookLevel, OrderType, Side)
from kanenas.execution.broker import ExecutionConfig, LiveBroker, PaperBroker
from kanenas.execution.portfolio import Portfolio


def fill(side, price, qty, fee=0.0, ts=0.0, tag=""):
    return Fill(Order("BTC-USD", side, qty, tag=tag), price, qty, fee, 0.0, ts)


def book(mid=100.0, spread=1.0, size=5.0, depth=5):
    half = spread / 2
    return OrderBook(
        0.0,
        tuple(BookLevel(mid - half - i, size) for i in range(depth)),
        tuple(BookLevel(mid + half + i, size) for i in range(depth)),
    )


class TestPortfolioAccounting(unittest.TestCase):
    def test_long_round_trip_nets_correctly(self):
        p = Portfolio("BTC-USD", 1_000.0)
        p.apply_fill(fill(Side.BUY, 100, 1, fee=1.0))
        trade = p.apply_fill(fill(Side.SELL, 110, 1, fee=1.0), ExitReason.TAKE_PROFIT)
        self.assertAlmostEqual(trade.gross_pnl, 10.0)
        self.assertAlmostEqual(trade.fees, 2.0)
        self.assertAlmostEqual(trade.net_pnl, 8.0)
        self.assertAlmostEqual(p.cash, 1_008.0)
        self.assertTrue(trade.is_win)

    def test_short_round_trip_nets_correctly(self):
        p = Portfolio("BTC-USD", 1_000.0)
        p.apply_fill(fill(Side.SELL, 100, 2))
        trade = p.apply_fill(fill(Side.BUY, 90, 2), ExitReason.TAKE_PROFIT)
        self.assertAlmostEqual(trade.net_pnl, 20.0)
        self.assertIs(trade.direction, Direction.SHORT)
        self.assertAlmostEqual(p.cash, 1_020.0)

    def test_equity_is_mark_to_market_not_cash_plus_pnl(self):
        """Regression: cash already carries the entry notional.

        Defining equity as ``cash + unrealised`` double-counts it and inflates
        the account while a position is open.
        """
        p = Portfolio("BTC-USD", 1_000.0)
        p.apply_fill(fill(Side.BUY, 100, 1))
        self.assertAlmostEqual(p.equity(100), 1_000.0)   # nothing gained yet
        self.assertAlmostEqual(p.equity(110), 1_010.0)
        self.assertAlmostEqual(p.equity(90), 990.0)

    def test_short_equity_marks_correctly(self):
        p = Portfolio("BTC-USD", 1_000.0)
        p.apply_fill(fill(Side.SELL, 100, 2))
        self.assertAlmostEqual(p.equity(100), 1_000.0)
        self.assertAlmostEqual(p.equity(90), 1_020.0)
        self.assertAlmostEqual(p.equity(110), 980.0)

    def test_flip_closes_then_reverses(self):
        p = Portfolio("BTC-USD", 1_000.0)
        p.apply_fill(fill(Side.BUY, 100, 1))
        trade = p.apply_fill(fill(Side.SELL, 120, 3))
        self.assertAlmostEqual(trade.net_pnl, 20.0)
        self.assertAlmostEqual(p.position.qty, -2.0)
        self.assertAlmostEqual(p.position.entry_price, 120.0)
        self.assertAlmostEqual(p.equity(120), 1_020.0)

    def test_scale_in_uses_volume_weighted_entry(self):
        p = Portfolio("BTC-USD", 10_000.0)
        p.apply_fill(fill(Side.BUY, 100, 1))
        p.apply_fill(fill(Side.BUY, 110, 1))
        self.assertAlmostEqual(p.position.entry_price, 105.0)
        self.assertAlmostEqual(p.position.qty, 2.0)

    def test_partial_close_leaves_remainder_open(self):
        p = Portfolio("BTC-USD", 10_000.0)
        p.apply_fill(fill(Side.BUY, 100, 2))
        trade = p.apply_fill(fill(Side.SELL, 110, 1))
        self.assertAlmostEqual(trade.qty, 1.0)
        self.assertAlmostEqual(p.position.qty, 1.0)
        self.assertAlmostEqual(p.position.entry_price, 100.0)

    def test_streaks_and_metrics(self):
        p = Portfolio("BTC-USD", 10_000.0)
        for exit_price in (110, 120, 130, 90):       # 3 wins then a loss
            p.apply_fill(fill(Side.BUY, 100, 1))
            p.apply_fill(fill(Side.SELL, exit_price, 1))
        self.assertEqual(p.best_win_streak, 3)
        self.assertEqual(p.loss_streak, 1)
        self.assertAlmostEqual(p.win_rate, 0.75)
        self.assertAlmostEqual(p.profit_factor, 60.0 / 10.0)
        self.assertAlmostEqual(p.expectancy, 50.0 / 4)

    def test_drawdown_tracks_peak(self):
        p = Portfolio("BTC-USD", 1_000.0)
        p.mark(0, 100); p.apply_fill(fill(Side.BUY, 100, 1))
        p.mark(1, 150)                                # equity 1050, new peak
        p.mark(2, 50)                                 # equity 950
        self.assertAlmostEqual(p.peak_equity, 1_050.0)
        self.assertAlmostEqual(p.max_drawdown, 100.0 / 1_050.0)

    def test_profit_factor_infinite_with_no_losses(self):
        p = Portfolio("BTC-USD", 1_000.0)
        p.apply_fill(fill(Side.BUY, 100, 1)); p.apply_fill(fill(Side.SELL, 110, 1))
        self.assertEqual(p.profit_factor, float("inf"))


class TestPaperBroker(unittest.TestCase):
    def test_buy_lifts_the_ask_sell_hits_the_bid(self):
        b = PaperBroker(ExecutionConfig(seed=1, latency_bps=0.0, impact_coefficient=0.0))
        ob = book(mid=100.0, spread=2.0)
        buy = b.execute(Order("BTC-USD", Side.BUY, 0.01), 100.0, ob)
        sell = b.execute(Order("BTC-USD", Side.SELL, 0.01), 100.0, ob)
        self.assertAlmostEqual(buy.price, 101.0)     # the ask
        self.assertAlmostEqual(sell.price, 99.0)     # the bid
        self.assertGreater(buy.price, sell.price)    # crossing always costs

    def test_slippage_increases_with_size(self):
        b = PaperBroker(ExecutionConfig(seed=1, latency_bps=0.0))
        ob = book(size=2.0)
        small = b.execute(Order("BTC-USD", Side.BUY, 0.1), 100.0, ob)
        large = b.execute(Order("BTC-USD", Side.BUY, 50.0), 100.0, ob)
        self.assertGreater(large.price, small.price)
        self.assertGreater(large.slippage, small.slippage)

    def test_fee_is_bps_of_notional(self):
        b = PaperBroker(ExecutionConfig(seed=1, taker_fee_bps=10.0))
        f = b.execute(Order("BTC-USD", Side.BUY, 2.0), 100.0, None)
        self.assertAlmostEqual(f.fee, f.price * 2.0 * 10.0 / 10_000.0)

    def test_limit_orders_pay_maker_fee(self):
        b = PaperBroker(ExecutionConfig(seed=1, taker_fee_bps=10.0, maker_fee_bps=1.0))
        taker = b.execute(Order("BTC-USD", Side.BUY, 1.0, OrderType.MARKET), 100.0, None)
        maker = b.execute(Order("BTC-USD", Side.BUY, 1.0, OrderType.LIMIT), 100.0, None)
        self.assertLess(maker.fee, taker.fee)

    def test_zero_and_negative_quantity_rejected(self):
        b = PaperBroker()
        self.assertIsNone(b.execute(Order("BTC-USD", Side.BUY, 0.0), 100.0, None))
        self.assertIsNone(b.execute(Order("BTC-USD", Side.BUY, -1.0), 100.0, None))

    def test_deterministic_for_a_given_seed(self):
        a = PaperBroker(ExecutionConfig(seed=42))
        c = PaperBroker(ExecutionConfig(seed=42))
        pa = [a.execute(Order("BTC-USD", Side.BUY, 1.0), 100.0, None).price for _ in range(20)]
        pc = [c.execute(Order("BTC-USD", Side.BUY, 1.0), 100.0, None).price for _ in range(20)]
        self.assertEqual(pa, pc)

    def test_rejections_are_counted(self):
        b = PaperBroker(ExecutionConfig(seed=5, reject_probability=1.0))
        self.assertIsNone(b.execute(Order("BTC-USD", Side.BUY, 1.0), 100.0, None))
        self.assertEqual(b.orders_rejected, 1)

    def test_fill_price_is_always_positive(self):
        b = PaperBroker(ExecutionConfig(seed=2, impact_coefficient=50.0))
        f = b.execute(Order("BTC-USD", Side.SELL, 1e6), 0.01, book(mid=0.01, spread=0.001, size=0.001))
        self.assertGreater(f.price, 0.0)


class TestLiveBrokerGuard(unittest.TestCase):
    def test_refuses_without_credentials(self):
        with self.assertRaises(PermissionError):
            LiveBroker("binance")

    def test_refuses_without_explicit_acknowledgement(self):
        with self.assertRaises(PermissionError):
            LiveBroker("binance", "key", "secret", i_understand_the_risk=False)

    def test_armed_broker_still_refuses_to_place_orders(self):
        lb = LiveBroker("binance", "key", "secret", i_understand_the_risk=True)
        with self.assertRaises(NotImplementedError):
            lb.execute(Order("BTC-USD", Side.BUY, 1.0), 100.0, None)


if __name__ == "__main__":
    unittest.main()
