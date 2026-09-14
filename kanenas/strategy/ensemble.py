"""The ensemble: blend independent signals, then let results re-weight them.

Two ideas do the work here.

**Blending.**  Each strategy emits a signed conviction in [-1, 1].  The ensemble
takes a weighted mean and requires the result to clear ``entry_threshold``
*and* to have real agreement behind it.  Agreement matters independently of
magnitude: one strategy screaming at 1.0 while three others lean the other way
is not the same trade as four strategies quietly nodding together, even when
the weighted means match.

**Adaptive weights.**  Every closed trade is attributed back to the strategies
that voted for it, in proportion to their share of that vote.  A rolling window
of attributed P&L becomes a multiplier on each strategy's base weight, bounded
to [w_min, w_max].  So a model that stops working gets quietly demoted instead
of dragging the book down, and one that is printing gets more say - without any
hand-tuning between runs.

The bounds are the important part: unbounded adaptation is just overfitting with
extra steps, and would let one lucky streak hand the whole book to one model.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from ..core.types import Direction, Signal
from .base import Strategy, StrategyContext


@dataclass
class EnsembleDecision:
    """The blended view for one bar, plus the full breakdown for the UI/logs."""

    direction: Direction
    score: float               # weighted mean conviction, signed
    confidence: float          # |score| scaled into 0..1
    agreement: float           # share of participating weight on the winning side
    signals: List[Signal] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.direction is not Direction.FLAT


class StrategyEnsemble:
    def __init__(
        self,
        strategies: List[Strategy],
        entry_threshold: float = 0.28,
        min_agreement: float = 0.55,
        adaptive: bool = True,
        attribution_window: int = 40,
        w_min: float = 0.35,
        w_max: float = 2.0,
    ) -> None:
        if not strategies:
            raise ValueError("ensemble needs at least one strategy")
        self.strategies = strategies
        self.entry_threshold = entry_threshold
        self.min_agreement = min_agreement
        self.adaptive = adaptive
        self.w_min = w_min
        self.w_max = w_max
        self.base_weights: Dict[str, float] = {s.name: s.weight for s in strategies}
        self.perf: Dict[str, Deque[float]] = {s.name: deque(maxlen=attribution_window) for s in strategies}
        self._multiplier: Dict[str, float] = {s.name: 1.0 for s in strategies}

    # ------------------------------------------------------------- weighting

    def weight_of(self, name: str) -> float:
        return self.base_weights.get(name, 1.0) * self._multiplier.get(name, 1.0)

    def _recompute_multipliers(self) -> None:
        """Map each strategy's rolling attributed P&L onto a bounded multiplier."""
        if not self.adaptive:
            return
        stats = {}
        for name, hist in self.perf.items():
            if len(hist) < 5:
                stats[name] = 0.0
                continue
            n = len(hist)
            mean = math.fsum(hist) / n
            var = math.fsum((x - mean) ** 2 for x in hist) / max(1, n - 1)
            # Floor the dispersion before dividing.  A strategy whose trades are
            # all *identically* bad has zero variance, and an unfloored t-stat
            # would read 0.0 - scoring a perfectly consistent loser as neutral.
            # The floor also caps |t| for near-constant series, which keeps one
            # tight winning streak from pinning the multiplier at w_max.
            sd = max(math.sqrt(var), abs(mean) * 0.25, 1e-12)
            # t-like score: profitable *and* consistent beats profitable once
            stats[name] = mean / (sd / math.sqrt(n))

        for name, t in stats.items():
            # squash to (w_min, w_max) with 1.0 at t = 0
            squashed = 2.0 / (1.0 + math.exp(-0.45 * t))  # (0, 2), =1 at t=0
            self._multiplier[name] = max(self.w_min, min(self.w_max, squashed))

    def attribute(self, contributions: Dict[str, float], net_pnl: float) -> None:
        """Credit/debit each strategy for a closed trade it voted for."""
        total = sum(abs(v) for v in contributions.values())
        if total <= 1e-12:
            return
        for name, share in contributions.items():
            if name in self.perf and abs(share) > 1e-12:
                self.perf[name].append(net_pnl * (abs(share) / total))
        self._recompute_multipliers()

    # -------------------------------------------------------------- decision

    def evaluate(self, ctx: StrategyContext) -> EnsembleDecision:
        signals: List[Signal] = []
        weights: Dict[str, float] = {}
        contributions: Dict[str, float] = {}

        weighted_sum = 0.0
        participating = 0.0
        long_w = short_w = 0.0

        for strat in self.strategies:
            sig = strat.evaluate(ctx)
            signals.append(sig)
            w = self.weight_of(strat.name)
            weights[strat.name] = w
            contrib = w * sig.score
            contributions[strat.name] = contrib
            weighted_sum += contrib
            if sig.direction is not Direction.FLAT:
                participating += w
                if sig.direction is Direction.LONG:
                    long_w += w * sig.confidence
                else:
                    short_w += w * sig.confidence

        total_weight = sum(weights.values()) or 1.0
        score = weighted_sum / total_weight

        directional = long_w + short_w
        if directional <= 1e-12:
            return EnsembleDecision(Direction.FLAT, 0.0, 0.0, 0.0, signals, weights, contributions,
                                    "all strategies flat")
        agreement = max(long_w, short_w) / directional

        if abs(score) < self.entry_threshold:
            return EnsembleDecision(Direction.FLAT, score, abs(score), agreement, signals, weights,
                                    contributions, f"score {score:+.2f} < {self.entry_threshold:.2f}")
        if agreement < self.min_agreement:
            return EnsembleDecision(Direction.FLAT, score, abs(score), agreement, signals, weights,
                                    contributions, f"agreement {agreement:.0%} < {self.min_agreement:.0%}")

        direction = Direction.LONG if score > 0 else Direction.SHORT
        confidence = min(1.0, abs(score) / max(self.entry_threshold * 2.2, 1e-9))
        voters = [s.source for s in signals if s.direction is direction]
        return EnsembleDecision(direction, score, confidence, agreement, signals, weights, contributions,
                                f"{len(voters)} agree ({', '.join(voters)}) score {score:+.2f}")

    # ------------------------------------------------------------ reflection

    def snapshot(self) -> Dict[str, dict]:
        out = {}
        for s in self.strategies:
            hist = self.perf[s.name]
            out[s.name] = {
                "base": self.base_weights[s.name],
                "multiplier": self._multiplier[s.name],
                "effective": self.weight_of(s.name),
                "trades": len(hist),
                "attributed_pnl": math.fsum(hist),
            }
        return out


def default_ensemble(**kwargs) -> StrategyEnsemble:
    """The stock five-model book the CLI runs when you do not say otherwise."""
    from .breakout import VolatilityBreakout
    from .mean_reversion import MeanReversion
    from .momentum import RiskAdjustedMomentum
    from .orderflow import OrderFlowPressure
    from .trend import TrendFollow

    return StrategyEnsemble(
        [TrendFollow(), MeanReversion(), VolatilityBreakout(), OrderFlowPressure(), RiskAdjustedMomentum()],
        **kwargs,
    )
