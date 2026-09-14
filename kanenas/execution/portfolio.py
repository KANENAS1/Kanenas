"""Portfolio accounting: cash, one netted position, and the closed-trade ledger.

Every number the dashboard shows and every metric the backtester reports is
derived from here, so this module is deliberately boring and fully auditable:
cash moves only in ``apply_fill``, and equity is the mark-to-market
``cash + qty * price``.  Shorts are modelled as a negative position financed
from cash, which is how a perpetual/margin account actually behaves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.types import Direction, ExitReason, Fill, Position, Side, Trade


@dataclass
class EquityPoint:
    ts: float
    equity: float
    price: float


class Portfolio:
    def __init__(self, symbol: str, starting_cash: float) -> None:
        self.symbol = symbol
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.position = Position(symbol)
        self.trades: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []
        self.fees_paid = 0.0
        self.slippage_paid = 0.0
        self.peak_equity = starting_cash
        self.max_drawdown = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.best_win_streak = 0
        self.worst_loss_streak = 0
        self._pending_tag = ""

    # ------------------------------------------------------------- valuation

    def equity(self, price: float) -> float:
        """Mark-to-market account value.

        ``cash`` already carries the *full notional* of every fill (a buy debits
        price x qty, a short sale credits it), so the open position must be
        marked at ``qty * price`` - not at its unrealised P&L.  Adding P&L here
        instead would double-count the entry notional and report equity that
        does not exist while a position is open; it nets out only once flat,
        which is exactly why that class of bug survives a naive smoke test.
        """
        return self.cash + self.position.qty * price

    def mark(self, ts: float, price: float) -> float:
        eq = self.equity(price)
        self.equity_curve.append(EquityPoint(ts, eq, price))
        if eq > self.peak_equity:
            self.peak_equity = eq
        dd = 0.0 if self.peak_equity <= 0 else (self.peak_equity - eq) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, dd)
        return eq

    @property
    def drawdown(self) -> float:
        if not self.equity_curve or self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity_curve[-1].equity) / self.peak_equity

    @property
    def total_return(self) -> float:
        if not self.equity_curve or self.starting_cash <= 0:
            return 0.0
        return (self.equity_curve[-1].equity - self.starting_cash) / self.starting_cash

    # ----------------------------------------------------------------- fills

    def apply_fill(self, fill: Fill, reason: Optional[ExitReason] = None, bars_held: int = 0) -> Optional[Trade]:
        """Apply a fill, returning a ``Trade`` when it closes/flips a position.

        Handles the four cases explicitly - open, add, reduce/close, flip -
        because getting the flip case wrong is the classic source of phantom
        P&L in a netting book.
        """
        pos = self.position
        signed_qty = fill.qty * fill.order.side.sign
        self.cash -= fill.price * signed_qty   # buying spends cash, selling raises it
        self.cash -= fill.fee
        self.fees_paid += fill.fee
        self.slippage_paid += abs(fill.slippage) * fill.qty
        pos.fees_paid += fill.fee

        trade: Optional[Trade] = None

        if not pos.is_open:
            # --- open a fresh position
            pos.qty = signed_qty
            pos.entry_price = fill.price
            pos.entry_ts = fill.ts
            pos.bars_held = 0
            pos.peak_price = fill.price
            pos.trough_price = fill.price
            pos.tag = fill.order.tag
            return None

        same_side = (pos.qty > 0) == (signed_qty > 0)
        if same_side:
            # --- scale in: volume-weighted average entry
            total = pos.qty + signed_qty
            pos.entry_price = (pos.entry_price * pos.qty + fill.price * signed_qty) / total
            pos.qty = total
            return None

        # --- reducing, closing or flipping
        closing_qty = min(abs(signed_qty), abs(pos.qty))
        direction = pos.direction
        gross = (fill.price - pos.entry_price) * closing_qty * (1 if pos.qty > 0 else -1)
        # fees on this round trip: entry fees pro-rated + this exit's fee
        entry_fee_share = pos.fees_paid - fill.fee
        fee_share = (entry_fee_share * (closing_qty / abs(pos.qty))) + fill.fee
        trade = Trade(
            symbol=self.symbol, direction=direction, qty=closing_qty,
            entry_price=pos.entry_price, exit_price=fill.price,
            entry_ts=pos.entry_ts, exit_ts=fill.ts,
            gross_pnl=gross, fees=fee_share,
            reason=reason or ExitReason.SIGNAL_FLIP, bars_held=bars_held or pos.bars_held,
            tag=pos.tag,
        )
        self._record(trade)

        remaining = pos.qty + signed_qty
        if abs(remaining) < 1e-12:
            self.position = Position(self.symbol)
        else:
            flipped = (remaining > 0) != (pos.qty > 0)
            pos.qty = remaining
            if flipped:
                pos.entry_price = fill.price
                pos.entry_ts = fill.ts
                pos.bars_held = 0
                pos.peak_price = fill.price
                pos.trough_price = fill.price
                pos.fees_paid = 0.0
                pos.tag = fill.order.tag
        return trade

    def _record(self, trade: Trade) -> None:
        self.trades.append(trade)
        if trade.is_win:
            self.win_streak += 1
            self.loss_streak = 0
            self.best_win_streak = max(self.best_win_streak, self.win_streak)
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.worst_loss_streak = max(self.worst_loss_streak, self.loss_streak)

    # --------------------------------------------------------------- metrics

    @property
    def wins(self) -> List[Trade]:
        return [t for t in self.trades if t.is_win]

    @property
    def losses(self) -> List[Trade]:
        return [t for t in self.trades if not t.is_win]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / len(self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        gains = math.fsum(t.net_pnl for t in self.wins)
        pains = abs(math.fsum(t.net_pnl for t in self.losses))
        if pains <= 1e-12:
            return float("inf") if gains > 0 else 0.0
        return gains / pains

    @property
    def expectancy(self) -> float:
        return math.fsum(t.net_pnl for t in self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def realized_pnl(self) -> float:
        return math.fsum(t.net_pnl for t in self.trades)
