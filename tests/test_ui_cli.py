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

    def test_frame_renders_all_panels_when_there_is_room(self):
        frame = Dashboard(warm_engine()).frame(width=120, height=60)
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

    def test_frame_never_exceeds_the_window_height(self):
        """Regression: the frame ignored terminal height entirely.

        A frame taller than the window scrolls on every redraw, so the display
        marches down the screen instead of updating in place - it looks like
        the dashboard has gone haywire. It was a fixed 54 lines, which overflows
        a default PowerShell window by 24.
        """
        d = Dashboard(warm_engine(bars=400))
        for height in (60, 50, 40, 34, 30, 24, 20, 15, 12, 10):
            lines = d.frame(width=120, height=height).splitlines()
            self.assertLessEqual(len(lines), height - 1,
                                 f"frame overflows a {height}-row window")

    def test_dense_layouts_keep_position_state_and_the_log(self):
        """When panels must go, these two survive longest.

        Position state answers "what am I holding"; the log answers "what just
        happened". Equity curve and trade history are reconstructable after the
        fact, so they are dropped first.
        """
        d = Dashboard(warm_engine(bars=400))
        for height in (40, 30, 24):
            frame = d.frame(width=120, height=height)
            self.assertIn("WALLET", frame, f"wallet dropped at {height} rows")
            self.assertIn("EXECUTION LOG", frame, f"log dropped at {height} rows")

    def test_panels_drop_in_priority_order_as_the_window_shrinks(self):
        d = Dashboard(warm_engine(bars=400))
        roomy = d.frame(width=120, height=60)
        tight = d.frame(width=120, height=26)
        self.assertIn("RECENT TRADES", roomy)
        self.assertNotIn("RECENT TRADES", tight)   # dropped before the log
        self.assertIn("EXECUTION LOG", tight)

    def test_draw_emits_no_trailing_newline_on_a_tty(self):
        """A newline on the final row scrolls the window once per frame."""
        import io

        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        buf = FakeTTY()
        Dashboard(warm_engine(bars=200), stream=buf).draw()
        out = buf.getvalue()
        self.assertFalse(out.endswith("\n"))
        self.assertTrue(out.startswith("\x1b[H"))
        self.assertTrue(out.endswith("\x1b[0J"))   # clears any taller leftover

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

    def _post(self, port, body, origin=None):
        import urllib.error, urllib.request
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/control",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            r = urllib.request.urlopen(req, timeout=5)
            return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_control_requires_the_token(self):
        """Localhost with no auth stops being safe once a request can *do*
        something - any page you have open could POST and flatten your book."""
        srv = DashboardServer(warm_engine(bars=300), port=0)
        srv.start()
        port = srv._httpd.server_address[1]
        try:
            self.assertEqual(self._post(port, {"action": "pause"})[0], 403)
            self.assertEqual(self._post(port, {"action": "pause", "token": "wrong"})[0], 403)
            self.assertFalse(srv.engine.paused)
        finally:
            srv.stop()

    def test_control_refuses_a_cross_origin_post(self):
        srv = DashboardServer(warm_engine(bars=300), port=0)
        srv.start()
        port = srv._httpd.server_address[1]
        try:
            code, body = self._post(port, {"action": "pause", "token": srv.token},
                                    origin="http://evil.example")
            self.assertEqual(code, 403)
            self.assertIn("cross-origin", body["error"])
            self.assertFalse(srv.engine.paused)
        finally:
            srv.stop()

    def test_control_actions_work_with_a_valid_token(self):
        srv = DashboardServer(warm_engine(bars=300), port=0)
        srv.start()
        port = srv._httpd.server_address[1]
        try:
            t = srv.token
            self.assertEqual(self._post(port, {"action": "pause", "token": t})[0], 200)
            self.assertTrue(srv.engine.paused)
            self.assertEqual(self._post(port, {"action": "resume", "token": t})[0], 200)
            self.assertFalse(srv.engine.paused)
            self.assertEqual(self._post(port, {"action": "halt", "token": t})[0], 200)
            self.assertTrue(srv.engine.risk.halted)
            # once halted, only a restart resumes
            self.assertEqual(self._post(port, {"action": "resume", "token": t})[0], 409)
            self.assertEqual(self._post(port, {"action": "bogus", "token": t})[0], 400)
        finally:
            srv.stop()

    def test_token_is_published_only_in_the_page(self):
        import urllib.request
        srv = DashboardServer(warm_engine(bars=300), port=0)
        srv.start()
        port = srv._httpd.server_address[1]
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
            self.assertIn(srv.token, page)
            self.assertNotIn("__CONTROL_TOKEN__", page)
            state = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5).read().decode()
            self.assertNotIn(srv.token, state)      # never leaked through the read API
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
            rc = main(["backtest", "--sim", "--bars", "600", "--seed", "5"])
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
                main(["backtest", "--sim", "--bars", "400", "--report", str(path)])
            data = json.loads(path.read_text())
        self.assertIn("sharpe", data)
        self.assertIn("caveats", data)

    def test_run_command_completes_a_bounded_session(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["run", "--sim", "--bars", "300", "--speed", "0",
                       "--no-web", "--no-dashboard"])
        self.assertEqual(rc, 0)
        self.assertIn("BACKTEST", buf.getvalue())

    def test_live_data_is_the_default(self):
        """--sim must be opt-in: the bot defaults to real BTC prices."""
        args = build_parser().parse_args(["backtest"])
        self.assertFalse(args.sim)
        self.assertEqual(args.symbol, "BTC")
        self.assertIsNone(args.venue)          # auto-failover across venues

    def test_assets_resolve_to_each_venues_own_spelling(self):
        from kanenas.data.rest import ASSETS, SYMBOLS, VENUE_ORDER, resolve_symbol
        self.assertEqual(resolve_symbol("binance", "BTC"), "BTCUSDT")
        self.assertEqual(resolve_symbol("kraken", "BTC"), "XBTUSD")   # Kraken says XBT
        self.assertEqual(resolve_symbol("coinbase", "btc"), "BTC-USD")
        self.assertEqual(resolve_symbol("binance", "ETH"), "ETHUSDT")
        self.assertEqual(resolve_symbol("okx", "SOL"), "SOL-USDT")
        self.assertEqual(resolve_symbol("bitstamp", "XRP"), "xrpusd")
        self.assertEqual(resolve_symbol("binance", "ADAUSDT"), "ADAUSDT")   # passthrough
        for asset in ASSETS:
            for v in VENUE_ORDER:
                self.assertIn(v, SYMBOLS[asset], f"{asset} missing on {v}")

    def test_aliases_people_actually_type(self):
        from kanenas.data.rest import canonical_asset
        for typed, expected in (("bitcoin", "BTC"), ("ethereum", "ETH"), ("ripple", "XRP"),
                                ("solana", "SOL"), ("SOL-USD", "SOL"), ("xrpusdt", "XRP")):
            self.assertEqual(canonical_asset(typed), expected)
        self.assertIsNone(canonical_asset("ADAUSDT"))   # unknown: passed through

    def test_no_live_venue_refuses_rather_than_simulating(self):
        """Silently trading synthetic prices you believe are live is the worst
        failure this tool could have, so the failure is loud and explicit."""
        from kanenas.data import rest
        original = rest._get
        rest._get = lambda *a, **k: (_ for _ in ()).throw(rest.FeedError("blocked"))
        try:
            with self.assertRaises(rest.FeedError) as ctx:
                rest.open_live_feed("BTC", "1m")
            msg = str(ctx.exception)
            self.assertIn("No live venue reachable", msg)
            self.assertIn("--sim", msg)
            for v in rest.VENUE_ORDER:
                self.assertIn(v, msg)          # every attempt is reported
        finally:
            rest._get = original

    def test_failover_picks_the_first_venue_that_answers(self):
        from kanenas.data import rest
        original, seen = rest._get, []

        def fake(url, timeout=12.0):
            seen.append(url)
            if "binance" in url or "coinbase" in url:
                raise rest.FeedError("geo-blocked")
            if "kraken" in url:
                return {"error": [], "result": {"XXBTZUSD": [
                    [1700000000, "1", "2", "0.5", "1.5", "1", "9", 3]]}}
            raise rest.FeedError("unexpected venue")

        rest._get = fake
        try:
            feed = rest.open_live_feed("BTC", "1m")
            self.assertEqual(feed.venue, "kraken")
            self.assertEqual(feed.symbol, "XBTUSD")
            self.assertTrue(any("binance" in u for u in seen))   # tried first
        finally:
            rest._get = original

    def test_unknown_command_exits_nonzero(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["frobnicate"])

    def test_risk_flags_are_threaded_into_the_engine(self):
        from kanenas.cli import build_engine
        args = build_parser().parse_args(["backtest", "--sim", "--risk", "0.02", "--max-dd", "0.1",
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
