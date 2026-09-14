"""Backtesting and performance measurement.

Two jobs here.

**Metrics that cannot flatter you.**  Return alone says nothing - it does not
tell you whether you were paid for risk, whether you survived the path, or
whether you simply held a rising market.  So every report carries Sharpe,
Sortino, Calmar, max drawdown, profit factor, expectancy, exposure and the fee
drag, plus a **buy-and-hold benchmark**: a bot that trades 400 times to
underperform holding is a worse bot, however green its curve is.

**Robustness over a single pretty run.**  ``monte_carlo`` re-runs the whole
strategy across many independently seeded markets.  One backtest is an anecdote;
the distribution across seeds is the closest thing to evidence you get before
risking money.  A strategy whose median is good but whose 5th percentile is
catastrophic is not a strategy, it is a bet.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

from .core.types import Trade
from .data.base import MarketEvent
from .engine import EngineConfig, TradingEngine
from .execution.portfolio import Portfolio


@dataclass
class BacktestReport:
    symbol: str
    bars: int
    starting_cash: float
    final_equity: float
    total_return: float
    buy_hold_return: float
    annualised_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    volatility: float
    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_bars_held: float
    exposure: float
    fees_paid: float
    slippage_paid: float
    fee_drag: float
    best_win_streak: int
    worst_loss_streak: int
    sample_days: float = 0.0
    caveats: List[str] = field(default_factory=list)
    exit_breakdown: dict = field(default_factory=dict)
    strategy_attribution: dict = field(default_factory=dict)
    risk_rejections: dict = field(default_factory=dict)
    sized_by: dict = field(default_factory=dict)
    avg_risk_per_trade: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ view

    def render(self) -> str:
        def pct(x: float) -> str:
            return f"{x * 100:+.2f}%"

        edge = self.total_return - self.buy_hold_return
        # Anything annualised from a handful of days is arithmetic, not
        # information: 17% over 3 days extrapolates to six-figure percentages.
        # Withhold those numbers rather than print something indefensible.
        short = self.sample_days < MIN_DAYS_TO_ANNUALISE
        ann = "n/a (short sample)" if short else pct(self.annualised_return)
        cal = "n/a" if short else f"{self.calmar:.2f}"
        lines = [
            f"╔══ BACKTEST · {self.symbol} · {self.bars:,} bars · {self.sample_days:.1f} days " + "═" * max(0, 22 - len(self.symbol)) + "╗",
            f"  Start capital     ${self.starting_cash:>14,.2f}",
            f"  Final equity      ${self.final_equity:>14,.2f}",
            f"  Total return      {pct(self.total_return):>15}   over {self.sample_days:.1f} days",
            f"  Buy & hold        {pct(self.buy_hold_return):>15}",
            f"  Edge vs hold      {pct(edge):>15}   {'BEAT' if edge > 0 else 'LOST TO'} the benchmark",
            f"  Annualised        {ann:>15}",
            "",
            f"  Sharpe            {self.sharpe:>15.2f}",
            f"  Sortino           {self.sortino:>15.2f}",
            f"  Calmar            {cal:>15}",
            f"  Max drawdown      {self.max_drawdown * 100:>14.2f}%",
            f"  Volatility (ann.) {self.volatility * 100:>14.2f}%",
            f"  Exposure          {self.exposure * 100:>14.2f}%   (share of bars in a position)",
            "",
            f"  Trades            {self.trades:>15,}",
            f"  Win rate          {self.win_rate * 100:>14.2f}%",
            f"  Profit factor     {self.profit_factor:>15.2f}",
            f"  Expectancy/trade  ${self.expectancy:>14,.2f}",
            f"  Avg win / loss    ${self.avg_win:>8,.2f} / ${self.avg_loss:,.2f}",
            f"  Best / worst      ${self.best_trade:>8,.2f} / ${self.worst_trade:,.2f}",
            f"  Avg bars held     {self.avg_bars_held:>15.1f}",
            f"  Win/loss streak   {self.best_win_streak:>7} / {self.worst_loss_streak}",
            "",
            f"  Fees paid         ${self.fees_paid:>14,.2f}",
            f"  Slippage paid     ${self.slippage_paid:>14,.2f}",
            f"  Fee drag          {self.fee_drag * 100:>14.2f}%   (of starting capital)",
        ]
        if self.exit_breakdown:
            lines += ["", "  Exits: " + "  ".join(f"{k}={v}" for k, v in sorted(self.exit_breakdown.items()))]
        if self.sized_by:
            lines += ["  Sized by: " + "  ".join(f"{k}={v}" for k, v in sorted(self.sized_by.items()))]
        if self.risk_rejections:
            lines += ["  Risk vetoes: " + "  ".join(f"{k}={v}" for k, v in sorted(self.risk_rejections.items()))]
        if self.caveats:
            lines += ["", "  ⚠ READ THIS BEFORE BELIEVING THE NUMBERS ABOVE:"]
            lines += [f"    - {c}" for c in self.caveats]
        if self.strategy_attribution:
            lines += ["", "  Strategy attribution (rolling window):"]
            for name, s in self.strategy_attribution.items():
                lines.append(f"    {name:<10} weight {s['effective']:.2f}  "
                             f"({s['trades']} trades, ${s['attributed_pnl']:+,.2f})")
        lines.append("╚" + "═" * 60 + "╝")
        return "\n".join(lines)


#: below this many days of data, annualised figures are noise amplified
MIN_DAYS_TO_ANNUALISE = 30.0
#: a per-bar Sharpe this high on a short sample means the sample, not the edge
IMPLAUSIBLE_SHARPE = 4.0


def _returns(equity: List[float]) -> List[float]:
    out = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        if prev > 0:
            out.append(equity[i] / prev - 1.0)
    return out


def summarise(engine: TradingEngine, bars_per_year: float) -> BacktestReport:
    p: Portfolio = engine.portfolio
    curve = [pt.equity for pt in p.equity_curve]
    prices = [pt.price for pt in p.equity_curve]
    if len(curve) < 2:
        raise ValueError("not enough bars to summarise")

    rets = _returns(curve)
    n = len(rets)
    mean = statistics.fmean(rets) if rets else 0.0
    sd = statistics.pstdev(rets) if n > 1 else 0.0
    downside = [r for r in rets if r < 0]
    dsd = statistics.pstdev(downside) if len(downside) > 1 else 0.0

    ann_factor = math.sqrt(bars_per_year)
    sharpe = (mean / sd * ann_factor) if sd > 1e-12 else 0.0
    sortino = (mean / dsd * ann_factor) if dsd > 1e-12 else 0.0
    vol = sd * ann_factor

    total_return = p.total_return
    years = max(n / bars_per_year, 1e-9)
    growth = curve[-1] / curve[0] if curve[0] > 0 else 1.0
    annualised = (growth ** (1.0 / years) - 1.0) if growth > 0 else -1.0
    # a tiny sample can imply absurd CAGR; clamp the display, not the maths
    annualised = max(min(annualised, 1e4), -1.0)
    calmar = (annualised / p.max_drawdown) if p.max_drawdown > 1e-9 else 0.0

    buy_hold = (prices[-1] / prices[0] - 1.0) if prices[0] > 0 else 0.0

    trades: List[Trade] = p.trades
    wins = [t.net_pnl for t in trades if t.is_win]
    losses = [t.net_pnl for t in trades if not t.is_win]
    in_market = sum(t.bars_held for t in trades)

    exit_breakdown: dict = {}
    for t in trades:
        exit_breakdown[t.reason.value] = exit_breakdown.get(t.reason.value, 0) + 1

    span_seconds = max(p.equity_curve[-1].ts - p.equity_curve[0].ts, 0.0)
    sample_days = span_seconds / 86_400.0 if span_seconds else n / bars_per_year * 365.0

    # What fraction of equity a stop-out actually costs, as sized.
    sized = engine.risk.sized_by
    capped = sized.get("position_cap", 0) + sized.get("no_leverage", 0)
    total_sized = sum(sized.values())
    configured_risk = engine.risk.cfg.risk_per_trade
    # Measure the real figure from the trades that actually hit their stop,
    # rather than recomputing it from config - what a stop-out *cost* is the
    # only number worth quoting here.
    avg_risk = 0.0
    stopped = [t for t in trades if t.reason.value == "SL"]
    if stopped and p.starting_cash > 0:
        avg_risk = abs(statistics.fmean(t.net_pnl for t in stopped)) / p.starting_cash

    caveats: List[str] = []
    if total_sized and capped / total_sized > 0.5 and configured_risk > 0:
        caveats.append(
            f"Position size was set by the exposure cap, not --risk, on "
            f"{capped / total_sized:.0%} of entries. A typical stop-out cost "
            f"{avg_risk:.2%} of starting capital, not the {configured_risk:.2%} configured - "
            f"ATR stops this tight would need leverage to risk the full amount.")
    if sample_days < MIN_DAYS_TO_ANNUALISE:
        caveats.append(
            f"Sample is {sample_days:.1f} days. Annualised return and Calmar are withheld - "
            f"extrapolating a short window produces absurd figures, not forecasts.")
    if sharpe > IMPLAUSIBLE_SHARPE:
        caveats.append(
            f"Sharpe {sharpe:.1f} is far above anything sustainable in live markets "
            f"(a great real fund runs 1-3). Treat it as a property of this dataset, not of the strategy.")
    if len(trades) < 30:
        caveats.append(f"Only {len(trades)} trades - too few for win rate or profit factor to mean anything.")
    if p.win_rate >= 0.95 and len(trades) >= 10:
        caveats.append("Win rate near 100% almost always indicates a bug or a lookahead leak, not an edge.")
    if total_return > 0 and p.fees_paid > abs(curve[-1] - curve[0]):
        caveats.append(
            f"Fees (${p.fees_paid:,.2f}) exceed net profit (${curve[-1] - curve[0]:,.2f}) - "
            f"the strategy is trading too often to keep what it earns.")
    if total_return <= buy_hold:
        caveats.append("Underperformed buy-and-hold: the trading added risk and cost without adding return.")

    return BacktestReport(
        symbol=engine.cfg.symbol,
        bars=len(curve),
        starting_cash=p.starting_cash,
        final_equity=curve[-1],
        total_return=total_return,
        buy_hold_return=buy_hold,
        annualised_return=annualised,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=p.max_drawdown,
        volatility=vol,
        trades=len(trades),
        win_rate=p.win_rate,
        profit_factor=p.profit_factor,
        expectancy=p.expectancy,
        avg_win=statistics.fmean(wins) if wins else 0.0,
        avg_loss=statistics.fmean(losses) if losses else 0.0,
        best_trade=max((t.net_pnl for t in trades), default=0.0),
        worst_trade=min((t.net_pnl for t in trades), default=0.0),
        avg_bars_held=statistics.fmean([t.bars_held for t in trades]) if trades else 0.0,
        exposure=in_market / len(curve) if curve else 0.0,
        fees_paid=p.fees_paid,
        slippage_paid=p.slippage_paid,
        fee_drag=p.fees_paid / p.starting_cash if p.starting_cash else 0.0,
        best_win_streak=p.best_win_streak,
        worst_loss_streak=p.worst_loss_streak,
        sample_days=sample_days,
        caveats=caveats,
        exit_breakdown=exit_breakdown,
        strategy_attribution=engine.ensemble.snapshot(),
        risk_rejections=dict(engine.risk.rejections),
        sized_by=dict(engine.risk.sized_by),
        avg_risk_per_trade=avg_risk,
    )


def run_backtest(
    events: Iterable[MarketEvent],
    config: Optional[EngineConfig] = None,
    engine: Optional[TradingEngine] = None,
    **engine_kwargs,
) -> tuple[TradingEngine, BacktestReport]:
    cfg = config or EngineConfig()
    eng = engine or TradingEngine(cfg, **engine_kwargs)
    last: Optional[MarketEvent] = None
    for event in events:
        eng.process(event)
        last = event
    if last is not None:
        eng.close_all(last)   # never report an open position as if it were cash
        eng.portfolio.mark(last.ts, last.candle.close)
    return eng, summarise(eng, cfg.bars_per_year)


@dataclass
class MonteCarloResult:
    runs: int
    returns: List[float]
    sharpes: List[float]
    drawdowns: List[float]
    win_rates: List[float]

    def _q(self, data: List[float], q: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[idx]

    @property
    def profitable_share(self) -> float:
        return sum(1 for r in self.returns if r > 0) / len(self.returns) if self.returns else 0.0

    def render(self) -> str:
        r = self.returns
        lines = [
            f"╔══ ROBUSTNESS · {self.runs} independently seeded markets " + "═" * 12 + "╗",
            f"  Profitable runs   {self.profitable_share * 100:>14.1f}%",
            f"  Return  p05/med/p95  {self._q(r,.05)*100:>+7.2f}% / {self._q(r,.5)*100:>+7.2f}% / {self._q(r,.95)*100:>+7.2f}%",
            f"  Sharpe  p05/med/p95  {self._q(self.sharpes,.05):>7.2f}  / {self._q(self.sharpes,.5):>7.2f}  / {self._q(self.sharpes,.95):>7.2f}",
            f"  MaxDD   med/p95      {self._q(self.drawdowns,.5)*100:>7.2f}% / {self._q(self.drawdowns,.95)*100:>7.2f}%",
            f"  WinRate med          {self._q(self.win_rates,.5)*100:>7.2f}%",
            f"  Worst run            {min(r)*100:>+7.2f}%      Best run  {max(r)*100:>+7.2f}%",
            "╚" + "═" * 60 + "╝",
        ]
        return "\n".join(lines)


def monte_carlo(runs: int = 20, bars: int = 5_000, config: Optional[EngineConfig] = None,
                base_seed: int = 1000, sim_config=None, progress=None) -> MonteCarloResult:
    """Re-run the identical strategy over ``runs`` different random markets."""
    from dataclasses import replace
    from .data.simulator import MarketSimulator, SimulatorConfig
    from .strategy.ensemble import default_ensemble

    cfg = config or EngineConfig()
    base_sim = sim_config or SimulatorConfig()
    returns, sharpes, dds, wrs = [], [], [], []
    for i in range(runs):
        sim_cfg = replace(base_sim, seed=base_seed + i * 17)
        sim = MarketSimulator(sim_cfg, bars=bars)
        # a fresh ensemble and risk state per run - no leakage between markets
        eng = TradingEngine(cfg, ensemble=default_ensemble())
        _, rep = run_backtest(sim.stream(), cfg, engine=eng)
        returns.append(rep.total_return)
        sharpes.append(rep.sharpe)
        dds.append(rep.max_drawdown)
        wrs.append(rep.win_rate)
        if progress:
            progress(i + 1, runs, rep)
    return MonteCarloResult(runs, returns, sharpes, dds, wrs)
