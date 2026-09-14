"""Core value objects shared by every layer of the bot.

Everything here is a plain dataclass with no behaviour beyond arithmetic that
belongs to the datum itself.  Keeping the domain model dependency-free is what
lets the same objects flow through the simulator, the backtester and a live
exchange adapter without translation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class Direction(int, Enum):
    """A strategy's view of the market."""

    SHORT = -1
    FLAT = 0
    LONG = 1

    @property
    def side(self) -> Optional[Side]:
        if self is Direction.LONG:
            return Side.BUY
        if self is Direction.SHORT:
            return Side.SELL
        return None


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ExitReason(str, Enum):
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    TRAILING_STOP = "TRAIL"
    TIME_STOP = "TIME"
    SIGNAL_FLIP = "FLIP"
    RISK_HALT = "HALT"
    SHUTDOWN = "EOD"


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar.  ``ts`` is the bar *open* time in epoch seconds."""

    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return self.close - self.open

    @property
    def is_bull(self) -> bool:
        return self.close >= self.open

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBook:
    """Top-of-book snapshot.  ``bids`` descend, ``asks`` ascend."""

    ts: float
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid(self) -> float:
        if not self.bids or not self.asks:
            return self.best_bid or self.best_ask
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.best_ask - self.best_bid

    def imbalance(self, depth: int = 5) -> float:
        """Order-flow imbalance in [-1, 1]; positive means bid-heavy."""
        bid = sum(lvl.size for lvl in self.bids[:depth])
        ask = sum(lvl.size for lvl in self.asks[:depth])
        total = bid + ask
        if total <= 0:
            return 0.0
        return (bid - ask) / total


@dataclass(frozen=True)
class Signal:
    """A single strategy's opinion for one bar."""

    direction: Direction
    confidence: float  # 0.0 .. 1.0
    source: str
    reason: str = ""

    @property
    def score(self) -> float:
        """Signed conviction, the quantity the ensemble actually blends."""
        return float(self.direction.value) * max(0.0, min(1.0, self.confidence))

    @staticmethod
    def flat(source: str, reason: str = "no edge") -> "Signal":
        return Signal(Direction.FLAT, 0.0, source, reason)


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    reduce_only: bool = False
    tag: str = ""
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Fill:
    """What the venue actually gave us, after spread, slippage and fees."""

    order: Order
    price: float
    qty: float
    fee: float
    slippage: float
    ts: float

    @property
    def notional(self) -> float:
        return self.price * self.qty


@dataclass
class Position:
    symbol: str
    qty: float = 0.0  # signed: positive long, negative short
    entry_price: float = 0.0
    entry_ts: float = 0.0
    bars_held: int = 0
    stop_price: float = 0.0
    take_profit: float = 0.0
    peak_price: float = 0.0
    trough_price: float = 0.0
    fees_paid: float = 0.0
    tag: str = ""

    @property
    def is_open(self) -> bool:
        return abs(self.qty) > 1e-12

    @property
    def direction(self) -> Direction:
        if self.qty > 1e-12:
            return Direction.LONG
        if self.qty < -1e-12:
            return Direction.SHORT
        return Direction.FLAT

    def unrealized(self, price: float) -> float:
        if not self.is_open:
            return 0.0
        return (price - self.entry_price) * self.qty

    def unrealized_pct(self, price: float) -> float:
        if not self.is_open or self.entry_price <= 0:
            return 0.0
        return (price - self.entry_price) / self.entry_price * (1 if self.qty > 0 else -1)

    def notional(self, price: float) -> float:
        return abs(self.qty) * price


@dataclass(frozen=True)
class Trade:
    """A closed round-trip, the unit every performance metric is built from."""

    symbol: str
    direction: Direction
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: float
    exit_ts: float
    gross_pnl: float
    fees: float
    reason: ExitReason
    bars_held: int
    tag: str = ""

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def return_pct(self) -> float:
        cost = abs(self.entry_price * self.qty)
        return self.net_pnl / cost if cost > 0 else 0.0
