"""Market data: simulator realism, CSV round trips, venue adapters."""

import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from kanenas.core.types import Candle
from kanenas.data.replay import ReplayFeed, load_csv, write_csv
from kanenas.data.rest import (VENUES, FeedError, RestFeed, _binance_parse,
                               _coinbase_parse, _kraken_parse)
from kanenas.data.simulator import MarketSimulator, SimulatorConfig


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.events = list(MarketSimulator(SimulatorConfig(seed=42), bars=2_000).stream())

    def test_ohlc_invariants_hold_on_every_bar(self):
        for e in self.events:
            c = e.candle
            self.assertLessEqual(c.low, c.open)
            self.assertLessEqual(c.low, c.close)
            self.assertGreaterEqual(c.high, c.open)
            self.assertGreaterEqual(c.high, c.close)
            self.assertGreater(c.low, 0.0)
            self.assertGreaterEqual(c.volume, 0.0)

    def test_timestamps_are_strictly_increasing(self):
        ts = [e.candle.ts for e in self.events]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(len(set(ts)), len(ts))

    def test_reproducible_for_a_given_seed(self):
        a = [e.candle.close for e in MarketSimulator(SimulatorConfig(seed=11), bars=200).stream()]
        b = [e.candle.close for e in MarketSimulator(SimulatorConfig(seed=11), bars=200).stream()]
        self.assertEqual(a, b)

    def test_different_seeds_diverge(self):
        a = [e.candle.close for e in MarketSimulator(SimulatorConfig(seed=1), bars=200).stream()]
        b = [e.candle.close for e in MarketSimulator(SimulatorConfig(seed=2), bars=200).stream()]
        self.assertNotEqual(a, b)

    def test_volatility_is_in_a_plausible_range_for_crypto(self):
        closes = [e.candle.close for e in self.events]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        annualised = statistics.pstdev(rets) * math.sqrt(525_600)
        self.assertGreater(annualised, 0.10)    # not a flatline
        self.assertLess(annualised, 4.0)        # not nonsense

    def test_volatility_clusters(self):
        """GARCH dynamics: |return| should autocorrelate at lag 1."""
        closes = [e.candle.close for e in self.events]
        r = [abs(math.log(closes[i] / closes[i - 1])) for i in range(1, len(closes))]
        mean = statistics.fmean(r)
        num = sum((r[i] - mean) * (r[i - 1] - mean) for i in range(1, len(r)))
        den = sum((x - mean) ** 2 for x in r)
        self.assertGreater(num / den, 0.0)

    def test_book_is_well_formed(self):
        for e in self.events[:200]:
            b = e.book
            self.assertGreater(b.best_ask, b.best_bid)
            self.assertGreater(b.spread, 0.0)
            self.assertTrue(all(l.size > 0 for l in b.bids + b.asks))
            self.assertGreaterEqual(b.imbalance(), -1.0)
            self.assertLessEqual(b.imbalance(), 1.0)
            # bids descend, asks ascend
            self.assertEqual([l.price for l in b.bids], sorted([l.price for l in b.bids], reverse=True))
            self.assertEqual([l.price for l in b.asks], sorted([l.price for l in b.asks]))

    def test_history_helper_returns_candles(self):
        sim = MarketSimulator(SimulatorConfig(seed=3))
        self.assertEqual(len(sim.history(50)), 50)

    def test_regimes_actually_switch(self):
        sim = MarketSimulator(SimulatorConfig(seed=99), bars=3_000)
        seen = set()
        for _ in sim.stream():
            seen.add(sim.regime.name)
        self.assertGreater(len(seen), 1)


class TestReplay(unittest.TestCase):
    def test_csv_round_trip_preserves_values(self):
        candles = [e.candle for e in MarketSimulator(SimulatorConfig(seed=1), bars=40).stream()]
        with tempfile.TemporaryDirectory() as d:
            path = write_csv(Path(d) / "c.csv", candles)
            back = load_csv(path)
        self.assertEqual(len(back), len(candles))
        for a, b in zip(candles, back):
            self.assertAlmostEqual(a.close, b.close, places=6)
            self.assertAlmostEqual(a.high, b.high, places=6)

    def test_accepts_alternative_column_names(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tv.csv"
            p.write_text("time,o,h,l,c,vol\n2024-01-01 00:00:00,10,12,9,11,5\n")
            candles = load_csv(p)
        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0].high, 12)

    def test_millisecond_timestamps_are_demoted_to_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ms.csv"
            p.write_text("timestamp,open,high,low,close,volume\n1700000000000,1,2,0.5,1.5,3\n")
            c = load_csv(p)[0]
        self.assertAlmostEqual(c.ts, 1_700_000_000.0)

    def test_rows_are_sorted_by_time(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "u.csv"
            p.write_text("ts,open,high,low,close,volume\n200,1,2,0.5,1.5,3\n100,1,2,0.5,1.5,3\n")
            c = load_csv(p)
        self.assertLess(c[0].ts, c[1].ts)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("/nonexistent/path/candles.csv")

    def test_malformed_row_raises_with_context(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.csv"
            p.write_text("ts,open,high,low,close,volume\n100,notanumber,2,0.5,1.5,3\n")
            with self.assertRaises(ValueError):
                load_csv(p)

    def test_replay_feed_streams_every_candle_once(self):
        candles = [Candle(i, 1, 2, 0.5, 1.5, 1) for i in range(10)]
        self.assertEqual(len(list(ReplayFeed("X", candles).stream())), 10)


class TestVenueAdapters(unittest.TestCase):
    """Parsers are tested against recorded response shapes - no network needed."""

    def test_binance_kline_shape(self):
        raw = [[1700000000000, "100.0", "110.0", "90.0", "105.0", "12.5",
                1700000059999, "1", 1, "1", "1", "0"]]
        c = _binance_parse(raw)[0]
        self.assertAlmostEqual(c.ts, 1_700_000_000.0)
        self.assertEqual((c.open, c.high, c.low, c.close, c.volume), (100.0, 110.0, 90.0, 105.0, 12.5))

    def test_coinbase_orders_oldest_first(self):
        raw = [[1700000060, 90, 110, 100, 105, 3], [1700000000, 91, 111, 101, 106, 4]]
        candles = _coinbase_parse(raw)
        self.assertLess(candles[0].ts, candles[1].ts)
        self.assertEqual(candles[0].open, 101)   # [time, low, high, open, close, volume]

    def test_kraken_shape_and_error_surface(self):
        raw = {"error": [], "result": {"XXBTZUSD": [[1700000000, "1", "2", "0.5", "1.5", "1", "9", 3]],
                                       "last": 1}}
        c = _kraken_parse(raw)[0]
        self.assertEqual((c.open, c.high, c.low, c.close), (1.0, 2.0, 0.5, 1.5))
        with self.assertRaises(FeedError):
            _kraken_parse({"error": ["EQuery:Unknown asset pair"], "result": {}})

    def test_url_construction_per_venue(self):
        self.assertIn("symbol=BTCUSDT", VENUES["binance"].klines_url("BTCUSDT", "1m", 5))
        self.assertIn("granularity=60", VENUES["coinbase"].klines_url("BTC-USD", "60", 5))
        self.assertIn("interval=1", VENUES["kraken"].klines_url("XBTUSD", "1", 5))

    def test_unknown_venue_rejected(self):
        with self.assertRaises(ValueError):
            RestFeed("BTCUSDT", "not_a_venue")

    def test_unsupported_interval_rejected(self):
        with self.assertRaises(ValueError):
            RestFeed("BTCUSDT", "binance", "7s")

    def test_feed_names_itself_usefully(self):
        self.assertEqual(RestFeed("BTCUSDT", "binance", "5m").name, "binance:BTCUSDT:5m")


if __name__ == "__main__":
    unittest.main()
