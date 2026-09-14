"""Replay candles from CSV - the bridge between live data and reproducible tests.

Dump a venue's history once (``kanenas fetch``), then backtest against the exact
same bytes forever.  Accepts the common column spellings so a CSV exported from
TradingView, Binance or a pandas ``to_csv`` all load without editing.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from ..core.types import Candle
from .base import MarketEvent

_ALIASES = {
    "ts": ("ts", "time", "timestamp", "date", "datetime", "open_time", "opentime"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "price"),
    "volume": ("volume", "v", "vol", "base_volume"),
}


def _pick(row: dict, field: str) -> Optional[str]:
    lowered = {k.strip().lower(): v for k, v in row.items() if k}
    for alias in _ALIASES[field]:
        if alias in lowered:
            return lowered[alias]
    return None


def _parse_ts(raw: str) -> float:
    raw = raw.strip()
    try:
        val = float(raw)
        # heuristically demote milliseconds / microseconds to seconds
        while val > 4_102_444_800:  # year 2100 in seconds
            val /= 1000.0
        return val
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp: {raw!r}")


def load_csv(path: str | Path) -> List[Candle]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such candle file: {path}")
    out: List[Candle] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append(Candle(
                    _parse_ts(_pick(row, "ts")),
                    float(_pick(row, "open")), float(_pick(row, "high")),
                    float(_pick(row, "low")), float(_pick(row, "close")),
                    float(_pick(row, "volume") or 0.0),
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bad row in {path}: {row} ({exc})") from exc
    if not out:
        raise ValueError(f"{path} contained no candles")
    out.sort(key=lambda c: c.ts)
    return out


def write_csv(path: str | Path, candles: List[Candle]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([f"{c.ts:.0f}", c.open, c.high, c.low, c.close, c.volume])
    return path


class ReplayFeed:
    """Feeds a fixed candle list, optionally throttled to look live."""

    def __init__(self, symbol: str, candles: List[Candle], name: str = "replay") -> None:
        self.symbol = symbol
        self.candles = candles
        self.name = name

    @classmethod
    def from_csv(cls, symbol: str, path: str | Path) -> "ReplayFeed":
        return cls(symbol, load_csv(path), name=f"replay:{Path(path).name}")

    def stream(self) -> Iterator[MarketEvent]:
        for c in self.candles:
            yield MarketEvent(self.symbol, c, None)
