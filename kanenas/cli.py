"""Command line interface.

    kanenas run          - live/paper session with the terminal + web dashboards
    kanenas backtest     - single backtest with a full statistical report
    kanenas robustness   - the same strategy across many random markets
    kanenas fetch        - download candles from a venue to CSV
    kanenas doctor       - check Python, venue reachability and config sanity
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .backtest import monte_carlo, run_backtest
from .data.replay import ReplayFeed, load_csv, write_csv
from .data.simulator import MarketSimulator, SimulatorConfig
from .engine import EngineConfig, TradingEngine
from .execution.broker import ExecutionConfig, PaperBroker
from .risk.manager import RiskConfig, RiskManager
from .runner import LiveRunner, RunnerConfig
from .strategy.ensemble import default_ensemble
from .ui import render as R

BARS_PER_YEAR = {"1m": 525_600.0, "5m": 105_120.0, "15m": 35_040.0,
                 "1h": 8_760.0, "4h": 2_190.0, "6h": 1_460.0, "1d": 365.0}

BANNER = r"""
  _  __                                   _____              _ _
 | |/ /__ _ _ __   ___ _ __   __ _ ___   |_   _| __ __ _  __| (_)_ __   __ _
 | ' // _` | '_ \ / _ \ '_ \ / _` / __|    | || '__/ _` |/ _` | | '_ \ / _` |
 | . \ (_| | | | |  __/ | | | (_| \__ \    | || | | (_| | (_| | | | | | (_| |
 |_|\_\__,_|_| |_|\___|_| |_|\__,_|___/    |_||_|  \__,_|\__,_|_|_| |_|\__, |
                                                                       |___/
"""


# --------------------------------------------------------------- assembly

def build_engine(args) -> TradingEngine:
    bpy = BARS_PER_YEAR.get(args.interval, 525_600.0)
    eng_cfg = EngineConfig(
        symbol=args.symbol,
        starting_cash=args.cash,
        warmup_bars=args.warmup,
        allow_shorts=not args.no_shorts,
        bars_per_year=bpy,
    )
    risk_cfg = RiskConfig(
        risk_per_trade=args.risk,
        atr_stop_mult=args.stop_atr,
        atr_target_mult=args.target_atr,
        max_drawdown=args.max_dd,
        max_position_pct=args.max_position,
        min_confidence=args.min_confidence,
    )
    exec_cfg = ExecutionConfig(taker_fee_bps=args.fee_bps, seed=args.seed)
    # Risk must price the *same* costs execution will actually charge, so the
    # gate is derived from the broker config rather than guessed separately.
    risk_cfg = replace(risk_cfg, round_trip_cost_bps=2 * (exec_cfg.taker_fee_bps + exec_cfg.half_spread_bps),
                       min_edge_over_cost=args.min_edge)
    ensemble = default_ensemble(
        entry_threshold=args.threshold,
        min_agreement=args.agreement,
        adaptive=not args.no_adaptive,
    )
    return TradingEngine(
        eng_cfg,
        ensemble=ensemble,
        risk=RiskManager(risk_cfg),
        broker=PaperBroker(exec_cfg),
    )


def build_feed(args):
    """Pick the data source.

    Live BTC is the default. The simulator is opt-in behind ``--sim`` and always
    labelled as such - a bot quietly trading synthetic prices while you believe
    it is on the real market is the worst failure mode this tool has.
    """
    if args.csv:
        return ReplayFeed.from_csv(args.symbol, args.csv), f"csv:{Path(args.csv).name}"

    if args.sim:
        sim_cfg = SimulatorConfig(symbol=args.symbol, start_price=args.start_price, seed=args.seed)
        return MarketSimulator(sim_cfg, bars=getattr(args, "bars", None)), "simulator (NOT live)"

    from .data.rest import FeedError, open_live_feed
    venues = [args.venue] if args.venue else None
    try:
        feed = open_live_feed(args.symbol, args.interval, venues=venues)
    except FeedError as exc:
        raise SystemExit(f"\n{exc}\n")
    print(f"live data      {feed.name}")
    return feed, feed.name


# ----------------------------------------------------------------- commands

def cmd_run(args) -> int:
    engine = build_engine(args)
    feed, venue = build_feed(args)
    mode = "PAPER-SIM" if args.sim else "PAPER-LIVE"

    web = None
    if not args.no_web:
        from .ui.web import DashboardServer
        web = DashboardServer(engine, host=args.host, port=args.port,
                              mode=mode, venue=venue)
        url = web.start(open_browser=args.open)
        print(f"web dashboard  {url}")
        if not args.dashboard:
            print("(terminal dashboard disabled with --no-dashboard; the web view is live)")

    runner = LiveRunner(
        engine, feed,
        RunnerConfig(speed=args.speed, max_bars=args.bars, render=args.dashboard,
                     mode=mode, venue=venue),
        web=web,
    )
    try:
        runner.run()
    finally:
        if web:
            web.stop()

    from .backtest import summarise
    report = summarise(engine, engine.cfg.bars_per_year)
    print(report.render())
    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"\nreport written to {args.report}")
    return 0


def cmd_backtest(args) -> int:
    engine = build_engine(args)
    feed, venue = build_feed(args)
    from .data.rest import RestFeed
    if isinstance(feed, RestFeed):
        # a backtest needs a fixed history, not a live poll
        candles = feed.fetch_history(args.bars or 500)
        print(f"fetched {len(candles)} live {args.interval} bars from {venue} "
              f"({candles[0].close:,.2f} -> {candles[-1].close:,.2f})")
        feed = ReplayFeed(args.symbol, candles, name=venue)

    engine, report = run_backtest(feed.stream(), engine.cfg, engine=engine)
    print(report.render())
    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"\nreport written to {args.report}")
    return 0


def cmd_robustness(args) -> int:
    cfg = EngineConfig(symbol=args.symbol, starting_cash=args.cash,
                       bars_per_year=BARS_PER_YEAR.get(args.interval, 525_600.0))
    sim_cfg = SimulatorConfig(symbol=args.symbol, start_price=args.start_price)

    def progress(i, total, rep):
        bar = "█" * int(i / total * 28)
        sys.stdout.write(f"\r  running {i:>3}/{total}  |{bar:<28}|  last {rep.total_return:+7.2%}")
        sys.stdout.flush()

    print("Robustness testing runs on generated markets by necessity - you cannot")
    print("re-run history 24 different ways. Live data is used everywhere else.\n")
    print(f"Running {args.runs} independent markets x {args.bars} bars…")
    result = monte_carlo(runs=args.runs, bars=args.bars, config=cfg,
                         sim_config=sim_cfg, progress=progress)
    print("\n")
    print(result.render())
    if result.profitable_share < 0.6:
        print("\n  VERDICT: not robust. The edge does not survive a change of market.")
    elif min(result.returns) < -0.15:
        print("\n  VERDICT: profitable on average, but the tail is severe. Size down.")
    else:
        print("\n  VERDICT: holds up across markets. Still simulated - paper trade it next.")
    return 0


def cmd_stress(args) -> int:
    from .stress import render, run_all
    cfg = EngineConfig(symbol=args.symbol, starting_cash=args.cash,
                       bars_per_year=BARS_PER_YEAR.get(args.interval, 525_600.0))
    risk_cfg = RiskConfig(risk_per_trade=args.risk, atr_stop_mult=args.stop_atr,
                          atr_target_mult=args.target_atr, max_drawdown=args.max_dd)

    def progress(i, total, sc):
        sys.stdout.write(f"\r  scenario {i}/{total}: {sc.label:<16}")
        sys.stdout.flush()

    print("Stress scenarios are generated by necessity - hostile conditions have to be")
    print("constructed, not waited for. Live data is used everywhere else.\n")
    print(f"Stress testing across {args.runs} markets x {args.bars} bars per scenario…")
    results = run_all(runs=args.runs, bars=args.bars, engine_config=cfg,
                      risk_config=risk_cfg, progress=progress)
    print("\r" + " " * 50)
    print(render(results))
    return 0


def cmd_fetch(args) -> int:
    from .data.rest import FeedError, open_live_feed
    try:
        feed = open_live_feed(args.symbol, args.interval,
                              venues=[args.venue] if args.venue else None)
        candles = feed.fetch_history(args.bars or 500)
    except FeedError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    print(f"source: {feed.name}")
    out = write_csv(args.out, candles)
    print(f"wrote {len(candles)} bars to {out}")
    return 0


def cmd_doctor(args) -> int:
    from .data.rest import VENUES, FeedError, RestFeed
    ok = True
    print(f"python            {sys.version.split()[0]}")
    print(f"package           kanenas (no third-party dependencies)")
    print("")
    print("live BTC reachability:")
    from .data.rest import VENUE_ORDER, resolve_symbol
    reachable = []
    for name in VENUE_ORDER:
        sym = resolve_symbol(name, "BTC")
        try:
            candles = RestFeed(sym, name, "1m").fetch_history(2)
            reachable.append(name)
            print(f"  {name:<9} OK    {sym:<9} last close {candles[-1].close:,.2f}")
        except FeedError as exc:
            ok = False
            print(f"  {name:<9} FAIL  {sym:<9} {str(exc)[:70]}")
    if reachable:
        ok = True
        print(f"\n  {len(reachable)} venue(s) reachable - live BTC will be used automatically.")
    print("")
    print("self-test (simulator + engine):")
    sim = MarketSimulator(SimulatorConfig(seed=1), bars=800)
    eng, rep = run_backtest(sim.stream(), EngineConfig())
    print(f"  engine ran {rep.bars} bars, {rep.trades} trades, return {rep.total_return:+.2%}")
    if not ok:
        print("\n  No venue reachable from this network. Live commands will refuse to run")
        print("  rather than silently substitute simulated prices. Use --sim to work")
        print("  offline against the built-in simulator, which is labelled as such.")
    return 0


# -------------------------------------------------------------------- parse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kanenas",
        description="Kanenas crypto trading bot - multi-strategy, risk-managed, paper-first.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Live BTC data is the default; --sim opts into the simulator. Fills are "
               "always simulated - nothing here places a real order.",
    )
    p.add_argument("--version", action="version", version="kanenas 1.0.0")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--symbol", default="BTC",
                        help="BTC resolves to each venue's own spelling (XBTUSD on Kraken, etc.)")
        sp.add_argument("--cash", type=float, default=10_000.0, help="starting capital")
        sp.add_argument("--interval", default="1m", choices=sorted(BARS_PER_YEAR))
        sp.add_argument("--seed", type=int, default=7, help="simulator/broker seed (reproducibility)")
        sp.add_argument("--start-price", type=float, default=78_000.0)
        sp.add_argument("--csv", help="replay candles from a CSV file instead of live data")
        sp.add_argument("--sim", action="store_true",
                        help="use the built-in simulator instead of live data (clearly labelled)")
        sp.add_argument("--venue", default=None,
                        choices=["binance", "coinbase", "kraken", "bitstamp", "okx", "bybit"],
                        help="pin one venue; default tries each until one answers")
        # strategy / risk
        sp.add_argument("--risk", type=float, default=0.0075, help="fraction of equity risked per trade")
        sp.add_argument("--stop-atr", type=float, default=1.8)
        sp.add_argument("--target-atr", type=float, default=3.2)
        sp.add_argument("--max-dd", type=float, default=0.20, help="halt trading at this drawdown")
        sp.add_argument("--max-position", type=float, default=0.35)
        sp.add_argument("--min-confidence", type=float, default=0.35)
        sp.add_argument("--threshold", type=float, default=0.28, help="ensemble score needed to trade")
        sp.add_argument("--agreement", type=float, default=0.55, help="required strategy agreement")
        sp.add_argument("--no-adaptive", action="store_true", help="freeze strategy weights")
        sp.add_argument("--no-shorts", action="store_true")
        sp.add_argument("--fee-bps", type=float, default=5.0, help="taker fee in basis points")
        sp.add_argument("--min-edge", type=float, default=2.5,
                        help="a setup's target must be this multiple of its round-trip cost")
        sp.add_argument("--warmup", type=int, default=60)
        sp.add_argument("--report", help="write the JSON report to this path")

    r = sub.add_parser("run", help="live paper session with dashboards")
    common(r)
    r.add_argument("--bars", type=int, default=None, help="stop after N bars")
    r.add_argument("--speed", type=float, default=8.0, help="simulated bars per second (0 = unlimited)")
    r.add_argument("--port", type=int, default=8787)
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--no-web", action="store_true")
    r.add_argument("--open", action="store_true", help="open the web dashboard in a browser")
    r.add_argument("--no-dashboard", dest="dashboard", action="store_false",
                   help="disable the terminal dashboard")
    r.set_defaults(func=cmd_run, dashboard=True)

    b = sub.add_parser("backtest", help="run a backtest and print a full report")
    common(b)
    b.add_argument("--bars", type=int, default=5_000)
    b.set_defaults(func=cmd_backtest)

    m = sub.add_parser("robustness", help="re-run the strategy across many random markets")
    common(m)
    m.add_argument("--bars", type=int, default=4_000)
    m.add_argument("--runs", type=int, default=20)
    m.set_defaults(func=cmd_robustness)

    st = sub.add_parser("stress", help="try to break the strategy with hostile markets")
    common(st)
    st.add_argument("--bars", type=int, default=3_000)
    st.add_argument("--runs", type=int, default=8, help="markets per scenario")
    st.set_defaults(func=cmd_stress)

    f = sub.add_parser("fetch", help="download candles from a venue to CSV")
    f.add_argument("--symbol", default="BTC")
    f.add_argument("--venue", default=None,
                   choices=["binance", "coinbase", "kraken", "bitstamp", "okx", "bybit"])
    f.add_argument("--interval", default="1m", choices=sorted(BARS_PER_YEAR))
    f.add_argument("--bars", type=int, default=500)
    f.add_argument("--out", default="data/candles.csv")
    f.set_defaults(func=cmd_fetch)

    d = sub.add_parser("doctor", help="check the environment and venue reachability")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("run",):
        print(R.c(BANNER, R.MAGENTA))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
