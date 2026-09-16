"""The trading engine: one bar in, zero or more orders out.

This is the only component that sees everything, and its loop is fixed:

    bar -> indicators -> exit checks -> ensemble -> risk -> broker -> ledger

Order matters. **Exits are evaluated before entries**, on every bar, because a
stop that only gets checked when the strategy happens to have an opinion is not
a stop.  And exits are checked against the bar's *high and low*, not its close -
a stop 2% away is hit by a wick that closes flat, and pretending otherwise is
how a backtest invents money it would never have kept.

The engine is feed-agnostic and broker-agnostic: the same object runs the
backtest, the simulated live demo and (given a keyed broker) a real venue.
That is the whole point of the layering - there is no separate "backtest
version" of the logic that can silently diverge from what trades.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

from .core.indicators import IndicatorSet
from .core.types import (Candle, Direction, ExitReason, Fill, Order, OrderBook,
                         OrderType, Position, Side, Signal, Trade)
from .data.base import MarketEvent
from .execution.broker import Broker, ExecutionConfig, PaperBroker
from .execution.portfolio import Portfolio
from .risk.manager import RiskConfig, RiskDecision, RiskManager
from .strategy.base import StrategyContext
from .strategy.ensemble import EnsembleDecision, StrategyEnsemble, default_ensemble


@dataclass
class EngineConfig:
    symbol: str = "BTC-USD"
    starting_cash: float = 10_000.0
    warmup_bars: int = 60
    allow_shorts: bool = True
    flip_on_reverse: bool = True   # close and reverse when the ensemble flips
    bars_per_year: float = 525_600.0  # 1m bars; used for annualised stats


@dataclass
class LogEntry:
    ts: float
    bar: int
    kind: str      # ENTRY | EXIT | SKIP | HALT | INFO
    message: str
    price: float = 0.0
    pnl: float = 0.0


@dataclass
class EngineState:
    """Everything the dashboards and reports read - never mutated by them."""

    bar: int = 0
    price: float = 0.0
    candle: Optional[Candle] = None
    book: Optional[OrderBook] = None
    decision: Optional[EnsembleDecision] = None
    last_risk: Optional[RiskDecision] = None
    log: List[LogEntry] = field(default_factory=list)
    regime: str = ""

    def push(self, entry: LogEntry, cap: int = 400) -> None:
        self.log.append(entry)
        if len(self.log) > cap:
            del self.log[: len(self.log) - cap]


class TradingEngine:
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        ensemble: Optional[StrategyEnsemble] = None,
        risk: Optional[RiskManager] = None,
        broker: Optional[Broker] = None,
        indicators: Optional[IndicatorSet] = None,
    ) -> None:
        self.cfg = config or EngineConfig()
        self.ensemble = ensemble or default_ensemble()
        self.risk = risk or RiskManager()
        self.broker = broker or PaperBroker()
        self.ind = indicators or IndicatorSet(bars_per_year=self.cfg.bars_per_year)
        self.portfolio = Portfolio(self.cfg.symbol, self.cfg.starting_cash)
        self.state = EngineState()
        #: which strategies voted for the currently-open position (for attribution)
        self._open_contributions: Dict[str, float] = {}
        self.on_event: Optional[Callable[["TradingEngine"], None]] = None
        #: operator controls. Paused stops *new* entries only - an open
        #: position keeps its stop and target, because abandoning risk
        #: management on a live position is never what "pause" should mean.
        self.paused = False
        self._flatten_request = False
        #: set by the runner's watchdog; blocks new entries while data is stale
        self.feed_stale = False
        self.feed_health = "STARTING"

    # ------------------------------------------------------------- controls

    def pause(self) -> None:
        self.paused = True
        self.state.push(LogEntry(self.state.candle.ts if self.state.candle else 0.0,
                                 self.state.bar, "INFO",
                                 "PAUSED by operator - no new entries; "
                                 "open position keeps its stop and target",
                                 self.state.price))

    def resume(self) -> None:
        self.paused = False
        self.state.push(LogEntry(self.state.candle.ts if self.state.candle else 0.0,
                                 self.state.bar, "INFO", "RESUMED by operator", self.state.price))

    def set_feed_health(self, health: str, can_open: bool, note: Optional[str] = None) -> None:
        """Record feed health. Stale data blocks new risk but never fakes an exit.

        There is no honest way to close a position without prices, so a dead
        feed cannot flatten - it can only stop the bot taking on *more* risk
        and make the situation loud.
        """
        self.feed_health = health
        self.feed_stale = not can_open
        if note:
            kind = "HALT" if not can_open else "INFO"
            self.state.push(LogEntry(self.state.candle.ts if self.state.candle else 0.0,
                                     self.state.bar, kind, note, self.state.price))

    def request_flatten(self) -> bool:
        """Ask to close any open position on the next bar.

        Deferred rather than immediate: closing mid-bar would have to invent a
        price, and every other exit in this engine is priced from a real bar.
        Returns whether there was anything to close.
        """
        if not self.portfolio.position.is_open:
            return False
        self._flatten_request = True
        self.state.push(LogEntry(self.state.candle.ts if self.state.candle else 0.0,
                                 self.state.bar, "INFO",
                                 "FLATTEN requested - closing on the next bar", self.state.price))
        return True

    def halt(self, reason: str = "operator kill switch") -> None:
        """Stop trading entirely and flatten. Only a restart resumes."""
        self.risk.halted = True
        self.risk.halt_reason = reason
        self.paused = True
        self._flatten_request = self.portfolio.position.is_open
        self.state.push(LogEntry(self.state.candle.ts if self.state.candle else 0.0,
                                 self.state.bar, "HALT", f"HALTED - {reason}", self.state.price))

    # ------------------------------------------------------------------ loop

    def run(self, feed_events: Iterator[Optional[MarketEvent]], max_bars: Optional[int] = None) -> "TradingEngine":
        for n, event in enumerate(feed_events):
            if event is None:      # heartbeat from a live feed
                continue
            self.process(event)
            if max_bars is not None and n + 1 >= max_bars:
                break
        return self

    def prime(self, candles: List[Candle]) -> int:
        """Warm indicators and strategy state from history, without trading.

        A live feed only yields bars as they close, so starting cold means an
        empty chart that grows one candle a minute and indicators that take an
        hour of wall-clock to warm up - during which the bot cannot trade at
        all. Replaying recent history fixes both: the chart opens full and the
        book is ready on the first live bar.

        No orders, fills or equity points are produced. These bars already
        happened; trading them would invent a position and a P&L history that
        never existed. Strategies carrying their own state (order-flow
        smoothing, momentum returns, the squeeze window) are warmed by
        evaluating them and discarding the result.
        """
        n = 0
        for candle in candles:
            self.ind.update(candle)
            ctx = StrategyContext(self.cfg.symbol, candle, self.ind, None,
                                  self.portfolio.position, self.portfolio.starting_cash)
            self.ensemble.evaluate(ctx)   # discarded: called purely to warm state
            n += 1
        if n:
            last = candles[-1]
            self.state.bar += n
            self.state.candle = last
            self.state.price = last.close
            self.state.push(LogEntry(last.ts, self.state.bar, "INFO",
                                     f"primed {n} historical bars - indicators warm, "
                                     f"last close {last.close:,.2f}", last.close))
        return n

    def process(self, event: MarketEvent) -> None:
        st = self.state
        st.bar += 1
        st.candle = event.candle
        st.book = event.book
        st.price = event.candle.close
        self.ind.update(event.candle)

        if self.portfolio.position.is_open:
            self.portfolio.position.bars_held += 1

        equity = self.portfolio.mark(event.ts, st.price)
        self.risk.on_bar(event.ts, equity, self.portfolio.peak_equity, st.bar)

        # 1. exits first, always
        flattened = False
        if self._flatten_request:
            flattened = bool(self.portfolio.position.is_open)
            if flattened:
                self._close(event, ExitReason.MANUAL)
            self._flatten_request = False
        else:
            self._check_exits(event)

        # 2. is the book warm enough to have an opinion?
        if st.bar <= self.cfg.warmup_bars or not self.ind.warm:
            st.decision = None
            self._notify()
            return

        ctx = StrategyContext(self.cfg.symbol, event.candle, self.ind, event.book,
                              self.portfolio.position, equity)
        decision = self.ensemble.evaluate(ctx)
        st.decision = decision

        # 3. act
        if self.risk.halted:
            if self.portfolio.position.is_open:
                self._close(event, ExitReason.RISK_HALT)
            self._notify()
            return

        if flattened:
            # Re-entering on the very bar the operator asked to be flat defeats
            # the request. One bar of quiet; Pause or Halt is how you stay out.
            st.push(LogEntry(event.ts, st.bar, "SKIP",
                             "flattened this bar - no re-entry until the next", st.price))
            self._notify()
            return

        pos = self.portfolio.position
        if pos.is_open:
            flipped = decision.is_actionable and decision.direction is not pos.direction
            if flipped and self.cfg.flip_on_reverse:
                self._close(event, ExitReason.SIGNAL_FLIP)
                self._try_enter(event, decision)
        elif decision.is_actionable:
            self._try_enter(event, decision)

        self._notify()

    def _notify(self) -> None:
        if self.on_event is not None:
            self.on_event(self)

    # --------------------------------------------------------------- entries

    def _try_enter(self, event: MarketEvent, decision: EnsembleDecision) -> None:
        st = self.state
        if self.paused:
            st.push(LogEntry(event.ts, st.bar, "SKIP", "paused - entry suppressed", st.price))
            return
        if self.feed_stale:
            st.push(LogEntry(event.ts, st.bar, "SKIP",
                             f"feed {self.feed_health} - refusing new risk on stale data", st.price))
            return
        if event.backfill:
            # replayed after an outage: old enough that entering here would be
            # a fill that never could have happened. Exits already ran above.
            st.push(LogEntry(event.ts, st.bar, "SKIP",
                             "backfilled bar - exits only, no new entry", st.price))
            return
        if decision.direction is Direction.SHORT and not self.cfg.allow_shorts:
            st.push(LogEntry(event.ts, st.bar, "SKIP", "short signal but shorts disabled", st.price))
            return

        rd = self.risk.evaluate_entry(decision.direction, decision.confidence, st.price,
                                      self.portfolio.equity(st.price), self.ind, st.bar)
        st.last_risk = rd
        if not rd.approved:
            st.push(LogEntry(event.ts, st.bar, "SKIP", f"{decision.direction.name} rejected - {rd.reason}", st.price))
            return

        side = decision.direction.side
        order = Order(self.cfg.symbol, side, rd.qty, OrderType.MARKET, tag=decision.reason, ts=event.ts)
        fill = self.broker.execute(order, st.price, event.book)
        if fill is None:
            st.push(LogEntry(event.ts, st.bar, "SKIP", "broker rejected order", st.price))
            return

        self.portfolio.apply_fill(fill)
        pos = self.portfolio.position
        pos.stop_price = rd.stop
        pos.take_profit = rd.target
        pos.peak_price = fill.price
        pos.trough_price = fill.price
        self._open_contributions = dict(decision.contributions)

        st.push(LogEntry(
            event.ts, st.bar, "ENTRY",
            f"{side.value} {rd.qty:.6f} @ {fill.price:,.2f} | stop {rd.stop:,.2f} tp {rd.target:,.2f} "
            f"| conf {decision.confidence:.2f} agree {decision.agreement:.0%} | {decision.reason}",
            fill.price,
        ))

    # ---------------------------------------------------------------- exits

    def _check_exits(self, event: MarketEvent) -> None:
        pos = self.portfolio.position
        if not pos.is_open:
            return
        candle = event.candle
        self.risk.update_trailing_stop(pos, candle.close, self.ind)

        long = pos.qty > 0
        # Intrabar resolution: test against high/low, not the close.  When both
        # the stop and the target sit inside one bar we cannot know which came
        # first, so we assume the *stop* - the pessimistic branch.  Assuming the
        # target instead is the single most common way a backtest lies.
        hit_stop = (candle.low <= pos.stop_price) if long else (candle.high >= pos.stop_price)
        hit_target = (candle.high >= pos.take_profit) if long else (candle.low <= pos.take_profit)

        if hit_stop and pos.stop_price > 0:
            reason = ExitReason.TRAILING_STOP if self._is_trailing(pos) else ExitReason.STOP_LOSS
            self._close(event, reason, price=pos.stop_price)
            return
        if hit_target and pos.take_profit > 0:
            self._close(event, ExitReason.TAKE_PROFIT, price=pos.take_profit)
            return
        if pos.bars_held >= self.risk.cfg.max_bars_in_trade:
            self._close(event, ExitReason.TIME_STOP)

    def _is_trailing(self, pos: Position) -> bool:
        """True if the stop has been ratcheted past break-even."""
        if pos.qty > 0:
            return pos.stop_price > pos.entry_price
        return 0 < pos.stop_price < pos.entry_price

    def _close(self, event: MarketEvent, reason: ExitReason, price: Optional[float] = None) -> Optional[Trade]:
        pos = self.portfolio.position
        if not pos.is_open:
            return None
        exit_price = price if price is not None else event.candle.close
        # A stop/target fills *at or through* its level, never better than it.
        exit_price = max(min(exit_price, event.candle.high), event.candle.low)

        side = Side.SELL if pos.qty > 0 else Side.BUY
        order = Order(self.cfg.symbol, side, abs(pos.qty), OrderType.MARKET,
                      reduce_only=True, tag=reason.value, ts=event.ts)
        fill = self.broker.execute(order, exit_price, event.book)
        if fill is None:
            return None

        bars_held = pos.bars_held
        trade = self.portfolio.apply_fill(fill, reason=reason, bars_held=bars_held)
        if trade is None:
            return None

        self.ensemble.attribute(self._open_contributions, trade.net_pnl)
        self._open_contributions = {}
        self.risk.note_trade_closed(trade.is_win, self.portfolio.loss_streak, self.state.bar)

        self.state.push(LogEntry(
            event.ts, self.state.bar, "EXIT",
            f"{reason.value} {side.value} {abs(trade.qty):.6f} @ {fill.price:,.2f} | "
            f"net {trade.net_pnl:+,.2f} ({trade.return_pct:+.2%}) | held {bars_held} bars",
            fill.price, trade.net_pnl,
        ))
        return trade

    def close_all(self, event: MarketEvent) -> None:
        if self.portfolio.position.is_open:
            self._close(event, ExitReason.SHUTDOWN)
