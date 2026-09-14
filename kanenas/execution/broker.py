"""Brokers: the paper venue the bot trades against, and the live guard rail.

The paper broker exists to make backtests *pessimistic*.  A backtest that fills
at the close price is a fantasy generator - it is the single reason most
published bot equity curves do not survive contact with a real venue.  So every
fill here pays, in order:

1. **the spread** - you cross it, always: buys lift the ask, sells hit the bid;
2. **market impact** - a square-root impact model against available top-of-book
   depth, so size costs you more, and costs you more when the book is thin;
3. **latency drift** - price keeps moving between decision and fill;
4. **taker fees** - in basis points of notional, the venue's cut.

Turn these to zero and the same backtest prints far prettier numbers.  That gap
is the honest measure of how fragile a strategy is.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Protocol

from ..core.types import Fill, Order, OrderBook, OrderType, Side


@dataclass
class ExecutionConfig:
    taker_fee_bps: float = 5.0     # 0.05% - typical spot taker fee
    maker_fee_bps: float = 1.0
    half_spread_bps: float = 1.0   # used when no live book is available
    impact_coefficient: float = 0.35
    latency_bps: float = 0.4       # stdev of adverse drift between send and fill
    reject_probability: float = 0.0
    seed: Optional[int] = 13


class Broker(Protocol):
    def execute(self, order: Order, price: float, book: Optional[OrderBook]) -> Optional[Fill]:
        ...


class PaperBroker:
    """Simulated venue. Deterministic for a given seed so tests are stable."""

    name = "paper"

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        self.cfg = config or ExecutionConfig()
        self.rng = random.Random(self.cfg.seed)
        self.orders_sent = 0
        self.orders_rejected = 0

    def _reference_prices(self, price: float, book: Optional[OrderBook]) -> tuple[float, float, float]:
        """(bid, ask, top-of-book depth) from the real book when we have one."""
        if book is not None and book.bids and book.asks:
            depth = sum(l.size for l in book.bids[:3]) + sum(l.size for l in book.asks[:3])
            return book.best_bid, book.best_ask, max(depth, 1e-9)
        half = price * self.cfg.half_spread_bps / 10_000.0
        return price - half, price + half, 1e9  # unknown book => assume deep

    def execute(self, order: Order, price: float, book: Optional[OrderBook] = None) -> Optional[Fill]:
        self.orders_sent += 1
        if order.qty <= 0:
            return None
        if self.cfg.reject_probability and self.rng.random() < self.cfg.reject_probability:
            self.orders_rejected += 1
            return None

        bid, ask, depth = self._reference_prices(price, book)
        mid = (bid + ask) / 2.0

        # 1. cross the spread
        base = ask if order.side is Side.BUY else bid

        # 2. square-root market impact, scaled by how much of the book we eat
        participation = min(1.0, order.qty / depth) if depth > 0 else 0.0
        impact = self.cfg.impact_coefficient * mid * (participation ** 0.5) / 100.0

        # 3. latency drift - unbiased in direction, but it is *your* cost either way
        drift = self.rng.gauss(0.0, mid * self.cfg.latency_bps / 10_000.0)

        fill_price = base + (impact + drift) * order.side.sign
        fill_price = max(fill_price, 1e-9)

        # 4. fees
        fee_bps = self.cfg.maker_fee_bps if order.type is OrderType.LIMIT else self.cfg.taker_fee_bps
        fee = abs(fill_price * order.qty) * fee_bps / 10_000.0

        return Fill(
            order=order,
            price=fill_price,
            qty=order.qty,
            fee=fee,
            slippage=abs(fill_price - mid),
            ts=order.ts,
        )


class LiveBroker:
    """Placeholder for a keyed exchange connection.

    Deliberately inert.  Wiring real order placement is a handful of signed REST
    calls, but a bot that can lose real money should not acquire that ability as
    a side effect of someone running the demo - so this raises unless it is
    explicitly constructed with credentials and ``i_understand_the_risk=True``.
    """

    name = "live"

    def __init__(self, venue: str, api_key: str = "", api_secret: str = "",
                 i_understand_the_risk: bool = False) -> None:
        if not (api_key and api_secret and i_understand_the_risk):
            raise PermissionError(
                "LiveBroker refuses to arm: pass real credentials and "
                "i_understand_the_risk=True. Paper-trade first; the same engine "
                "drives both, so nothing about your strategy changes."
            )
        self.venue = venue
        self._key = api_key
        self._secret = api_secret

    def execute(self, order: Order, price: float, book: Optional[OrderBook] = None) -> Optional[Fill]:
        raise NotImplementedError(
            "Signed order placement is intentionally not implemented in this repo. "
            "Implement per-venue signing here, and test against the venue testnet first."
        )
