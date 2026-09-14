"""Dashboards and CLI. These guard the surfaces a user actually touches."""

import io
import json
import unittest
import urllib.request
from contextlib import redirect_stdout

from kanenas.cli import build_parser, main
from kanenas.data.simulator import MarketSimulator, SimulatorConfig
from kanenas.engine import EngineConfig, TradingEngine
from kanenas.ui import render as R
from kanenas.ui.dashboard import Dashboard
from kanenas.ui.web import DashboardServer, serialise


def warm_engine(bars=700, seed=2024):
    eng = TradingEngine(EngineConfig())
    eng.run(MarketSimulator(SimulatorConfig(seed=seed), bars=bars).stream())
    return eng


class TestRenderPrimitives(unittest.TestCase):
    def setUp(self):
        R.enable_color(False)

    def test_visible_len_ignores_escape_codes(self):
        R.enable_color(True)
        coloured = R.c("hello", R.GREEN)
        self.assertGreater(len(coloured), 5)
        self.assertEqual(R.visible_len(coloured), 5)
        R.enable_color(False)

    def test_pad_and_truncate_respect_visible_width(self):
        self.assertEqual(R.visible_len(R.pad("ab", 6)), 6)
        self.assertEqual(R.visible_len(R.truncate("abcdefgh", 4)), 4)

    def test_pad_alignments(self):
        self.assertTrue(R.pad("x", 5, "right").startswith(" "))
        self.assertTrue(R.pad("x", 5, "center").startswith(" "))

    def test_sparkline_length_matches_input(self):
        self.assertEqual(len(R.sparkline([1, 2, 3, 4, 5], 10)), 5)

    def test_sparkline_handles_flat_and_empty(self):
        self.assertEqual(R.sparkline([]), "")
        self.assertEqual(len(R.sparkline([7.0] * 5, 10)), 5)

    def test_panel_is_exactly_the_requested_width(self):
        for line in R.panel("T", ["a", "bb"], 24):
            self.assertEqual(R.visible_len(line), 24)

    def test_panel_truncates_overlong_rows(self):
        for line in R.panel("T", ["x" * 200], 20):
            self.assertEqual(R.visible_len(line), 20)

    def test_charts_return_requested_height(self):
        cs = [e.candle for e in MarketSimulator(SimulatorConfig(seed=3), bars=40).stream()]
        self.assertEqual(len(R.candle_chart(cs, 20, 8)), 8)
        self.assertEqual(len(R.area_chart([c.close for c in cs], 20, 5)), 5)
        self.assertEqual(len(R.price_axis(cs, 5)), 5)

    def test_charts_survive_empty_input(self):
        self.assertEqual(len(R.candle_chart([], 20, 4)), 4)
        self.assertEqual(len(R.area_chart([], 20, 4)), 4)

    def test_bar_gauge_clamps_out_of_range_values(self):
        self.assertEqual(R.visible_len(R.bar_gauge(5.0, 10)), 10)
        self.assertEqual(R.visible_len(R.bar_gauge(-3.0, 10)), 10)

    def test_hjoin_pads_to_tallest_block(self):
        self.assertEqual(len(R.hjoin([["a"], ["b", "c", "d"]])), 3)


class TestTerminalDashboard(unittest.TestCase):
    def setUp(self):
        R.enable_color(False)

    def test_frame_renders_all_panels(self):
        frame = Dashboard(warm_engine()).frame(width=120)
        for heading in ("WALLET", "SIGNAL MATRIX", "ORDER BOOK", "EQUITY",
                        "RECENT TRADES", "EXECUTION LOG"):
            self.assertIn(heading, frame)

    def test_frame_renders_before_any_bar_arrives(self):
        self.assertIsInstance(Dashboard(TradingEngine(EngineConfig())).frame(width=110), str)

    def test_frame_width_is_respected(self):
        for width in (96, 120, 160):
            frame = Dashboard(warm_engine(bars=300)).frame(width=width)
            longest = max(R.visible_len(l) for l in frame.splitlines())
            self.assertLessEqual(longest, width + 2)

    def test_draw_writes_to_a_non_tty_stream(self):
        buf = io.StringIO()
        Dashboard(warm_engine(bars=200), stream=buf).draw()
        self.assertIn("WALLET", buf.getvalue())


class TestWebDashboard(unittest.TestCase):
    def test_state_is_json_serialisable_and_complete(self):
        payload = serialise(warm_engine(), "PAPER", "simulator", 0.0)
        text = json.dumps(payload)           # must not raise
        self.assertGreater(len(text), 500)
        for key in ("price", "equity", "candles", "signals", "trades", "log",
                    "position", "book", "win_rate", "equity_curve", "attribution"):
            self.assertIn(key, payload)

    def test_infinite_profit_factor_is_json_safe(self):
        """json.dumps emits bare Infinity, which no JSON parser accepts."""
        eng = TradingEngine(EngineConfig())
        eng.portfolio.trades = []
        payload = serialise(eng, "PAPER", "sim", 0.0)
        self.assertNotIn("Infinity", json.dumps(payload))

    def test_serialise_does_not_mutate_the_engine(self):
        eng = warm_engine(bars=300)
        before = (eng.state.bar, eng.portfolio.cash, len(eng.portfolio.trades))
        serialise(eng, "PAPER", "sim", 0.0)
        self.assertEqual((eng.state.bar, eng.portfolio.cash, len(eng.portfolio.trades)), before)

    def test_server_serves_page_state_and_404(self):
        eng = warm_engine(bars=300)
        srv = DashboardServer(eng, port=0)
        srv.start()
        port = srv._httpd.server_address[1]
        try:
            html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
            self.assertIn("KANENAS", html)
            state = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/state", timeout=5).read().decode())
            self.assertEqual(state["symbol"], "BTC-USD")
            self.assertGreater(state["bar"], 0)
            health = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5)
            self.assertEqual(health.status, 200)
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
        finally:
            srv.stop()

    def test_server_binds_loopback_by_default(self):
        self.assertEqual(DashboardServer(TradingEngine(EngineConfig())).host, "127.0.0.1")


class TestCli(unittest.TestCase):
    def test_parser_exposes_every_command(self):
        p = build_parser()
        for cmd in ("run", "backtest", "robustness", "stress", "fetch", "doctor"):
            self.assertIsNotNone(p.parse_args([cmd] if cmd in ("doctor",) else [cmd]))

    def test_backtest_command_prints_a_report(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["backtest", "--bars", "600", "--seed", "5"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("BACKTEST", out)
        self.assertIn("Sharpe", out)
        self.assertIn("Buy & hold", out)

    def test_backtest_writes_a_json_report(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.json"
            with redirect_stdout(io.StringIO()):
                main(["backtest", "--bars", "400", "--report", str(path)])
            data = json.loads(path.read_text())
        self.assertIn("sharpe", data)
        self.assertIn("caveats", data)

    def test_run_command_completes_a_bounded_session(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["run", "--bars", "300", "--speed", "0", "--no-web", "--no-dashboard"])
        self.assertEqual(rc, 0)
        self.assertIn("BACKTEST", buf.getvalue())

    def test_unknown_command_exits_nonzero(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["frobnicate"])

    def test_risk_flags_are_threaded_into_the_engine(self):
        from kanenas.cli import build_engine
        args = build_parser().parse_args(["backtest", "--risk", "0.02", "--max-dd", "0.1",
                                          "--fee-bps", "12", "--threshold", "0.5"])
        eng = build_engine(args)
        self.assertAlmostEqual(eng.risk.cfg.risk_per_trade, 0.02)
        self.assertAlmostEqual(eng.risk.cfg.max_drawdown, 0.1)
        self.assertAlmostEqual(eng.broker.cfg.taker_fee_bps, 12.0)
        self.assertAlmostEqual(eng.ensemble.entry_threshold, 0.5)
        # the cost gate must reflect the fee the broker will really charge
        self.assertAlmostEqual(eng.risk.cfg.round_trip_cost_bps, 2 * (12.0 + 1.0))


if __name__ == "__main__":
    unittest.main()
