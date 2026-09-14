"""Risk: sizing, stops, and the switches that stop the bot trading at all.

Strategies find edges; this module decides whether an edge is worth risking
money on and how much.  It is the only place in the codebase allowed to size a
position, which means there is exactly one file to audit before trusting the bot
with anything.

Sizing is **volatility-targeted**, not fixed-notional: risk a constant fraction
of equity per trade, with the stop placed at a multiple of ATR, so
``qty = equity * risk_fraction / (atr_mult * ATR)``.  A fixed $-size bets far
more in a violent market than a calm one without you ever asking it to; this
formulation keeps the loss on a stop-out roughly constant in dollars whatever
volatility is doing.

The kill switches exist because the failure mode that actually ends accounts is
not a bad trade, it is a bad *day* compounding: drawdown limit, daily loss
limit, loss-streak cooldown, and an exposure cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..core.indicators import IndicatorSet
from ..core.types import Direction, Position


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.0075     # 0.75% of equity risked per trade
    atr_stop_mult: float = 1.8         # stop distance in ATR
    atr_target_mult: float = 3.2       # take-profit distance in ATR (R:R ~1.8)
    trailing_atr_mult: float = 2.2     # trail distance once in profit
    trail_activate_r: float = 1.0      # start trailing after +1R
    max_position_pct: float = 0.35     # cap notional at 35% of equity (no leverage by default)
    leverage: float = 1.0
    max_drawdown: float = 0.20         # hard stop: halt at -20% from peak
    daily_loss_limit: float = 0.06     # halt for the day at -6%
    loss_streak_cooldown: int = 3      # bars to sit out after N consecutive losses
    cooldown_bars: int = 5
    max_bars_in_trade: int = 240       # time stop
    min_confidence: float = 0.35
    min_trade_notional: float = 10.0
    #: round-trip execution cost estimate in basis points (2 x fee + 2 x half-spread).
    #: Wired from the broker's config so risk and execution cannot disagree.
    round_trip_cost_bps: float = 12.0
    #: a setup must aim for at least this multiple of its own execution cost
    min_edge_over_cost: float = 2.5


@dataclass
class RiskDecision:
    approved: bool
    qty: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    reason: str = ""
    risk_amount: float = 0.0


class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.cfg = config or RiskConfig()
        self.halted = False
        self.halt_reason = ""
        self.cooldown_until_bar = 0
        self.day_key: Optional[str] = None
        self.day_start_equity = 0.0
        self.day_pnl = 0.0
        self.rejections: dict[str, int] = {}

    # ------------------------------------------------------------ bookkeeping

    def _reject(self, reason: str) -> RiskDecision:
        key = reason.split(":")[0].strip()
        self.rejections[key] = self.rejections.get(key, 0) + 1
        return RiskDecision(False, reason=reason)

    def on_bar(self, ts: float, equity: float, peak_equity: float, bar_index: int) -> None:
        """Roll the daily window and evaluate the account-level kill switches."""
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if day != self.day_key:
            self.day_key = day
            self.day_start_equity = equity
            if self.halt_reason.startswith("daily loss"):
                # a new day clears a daily-loss halt, but never a drawdown halt
                self.halted = False
                self.halt_reason = ""
        self.day_pnl = equity - self.day_start_equity

        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            if dd >= self.cfg.max_drawdown and not self.halted:
                self.halted = True
                self.halt_reason = f"max drawdown {dd:.1%} >= {self.cfg.max_drawdown:.0%}"
        if self.day_start_equity > 0:
            day_ret = self.day_pnl / self.day_start_equity
            if day_ret <= -self.cfg.daily_loss_limit and not self.halted:
                self.halted = True
                self.halt_reason = f"daily loss {day_ret:.1%} <= -{self.cfg.daily_loss_limit:.0%}"

    def note_trade_closed(self, is_win: bool, loss_streak: int, bar_index: int) -> None:
        if not is_win and loss_streak >= self.cfg.loss_streak_cooldown:
            self.cooldown_until_bar = bar_index + self.cfg.cooldown_bars

    # ----------------------------------------------------------------- sizing

    def evaluate_entry(
        self,
        direction: Direction,
        confidence: float,
        price: float,
        equity: float,
        ind: IndicatorSet,
        bar_index: int,
    ) -> RiskDecision:
        cfg = self.cfg
        if self.halted:
            return self._reject(f"halted: {self.halt_reason}")
        if direction is Direction.FLAT:
            return self._reject("no direction")
        if bar_index < self.cooldown_until_bar:
            return self._reject(f"cooldown: {self.cooldown_until_bar - bar_index} bars left")
        if confidence < cfg.min_confidence:
            return self._reject(f"confidence {confidence:.2f} < {cfg.min_confidence:.2f}")
        if not ind.atr.ready or ind.atr.value <= 0:
            return self._reject("ATR not ready")
        if equity <= 0 or price <= 0:
            return self._reject("no equity")

        atr = ind.atr.value
        stop_distance = cfg.atr_stop_mult * atr
        if stop_distance <= 0:
            return self._reject("degenerate stop distance")

        # Size so a stop-out costs ~risk_per_trade of equity, scaled by conviction.
        risk_amount = equity * cfg.risk_per_trade * (0.5 + 0.5 * min(1.0, confidence))
        qty = risk_amount / stop_distance

        # Cap 1: notional exposure ceiling
        max_notional = equity * cfg.max_position_pct * cfg.leverage
        if qty * price > max_notional:
            qty = max_notional / price
        # Cap 2: never risk more cash than we hold (no accidental leverage)
        if qty * price > equity * cfg.leverage:
            qty = equity * cfg.leverage / price

        notional = qty * price
        if notional < cfg.min_trade_notional:
            return self._reject(f"notional ${notional:.2f} below minimum")

        # Cost gate.  The stress suite showed the book going from +10% to -10%
        # purely by raising the taker fee from 5bp to 20bp: the strategy was
        # firing setups whose target barely covered the round trip.  A trade is
        # only worth taking if what it is *aiming at* clears what it costs to
        # get in and out by a sensible margin.
        target_distance = cfg.atr_target_mult * atr
        round_trip_cost = price * cfg.round_trip_cost_bps / 10_000.0
        if round_trip_cost > 0 and target_distance < cfg.min_edge_over_cost * round_trip_cost:
            return self._reject(
                f"edge too thin: target {target_distance / price * 10_000:.1f}bp vs cost "
                f"{cfg.round_trip_cost_bps:.1f}bp")

        sign = direction.value
        stop = price - sign * stop_distance
        target = price + sign * target_distance
        if stop <= 0:
            return self._reject("stop below zero")

        return RiskDecision(True, qty=qty, stop=stop, target=target,
                            risk_amount=qty * stop_distance,
                            reason=f"size {qty:.6f} @ risk ${qty * stop_distance:.2f} ({cfg.atr_stop_mult}xATR)")

    # ------------------------------------------------------------------ exits

    def update_trailing_stop(self, pos: Position, price: float, ind: IndicatorSet) -> float:
        """Ratchet the stop once a trade is a winner - never loosen it."""
        cfg = self.cfg
        if not pos.is_open or not ind.atr.ready:
            return pos.stop_price
        atr = ind.atr.value
        initial_risk = cfg.atr_stop_mult * atr
        if initial_risk <= 0:
            return pos.stop_price

        if pos.qty > 0:
            pos.peak_price = max(pos.peak_price or price, price)
            r_multiple = (pos.peak_price - pos.entry_price) / initial_risk
            if r_multiple >= cfg.trail_activate_r:
                candidate = pos.peak_price - cfg.trailing_atr_mult * atr
                pos.stop_price = max(pos.stop_price, candidate)
        else:
            pos.trough_price = min(pos.trough_price or price, price)
            r_multiple = (pos.entry_price - pos.trough_price) / initial_risk
            if r_multiple >= cfg.trail_activate_r:
                candidate = pos.trough_price + cfg.trailing_atr_mult * atr
                pos.stop_price = min(pos.stop_price, candidate) if pos.stop_price else candidate
        return pos.stop_price
