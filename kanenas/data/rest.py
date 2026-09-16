"""Live market data over public REST endpoints - stdlib ``urllib`` only.

Three venues are supported because each blocks a different slice of the world:
Binance, Coinbase Advanced and Kraken.  The adapter normalises all three into
the same ``Candle`` stream, so the engine cannot tell them apart.

Polling REST rather than a websocket is a deliberate trade-off: the bot trades
on *closed bars* (1m and up), so a poll a few seconds after each bar close sees
exactly what a socket would, with none of the reconnect/ordering complexity and
zero dependencies.  ``--interval 1s`` scalping would need a socket; that is what
``WebsocketFeed`` would slot in as, behind the same ``MarketFeed`` protocol.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

from ..core.types import BookLevel, Candle, OrderBook
from .base import MarketEvent

USER_AGENT = "kanenas-bot/1.0 (+https://github.com/kanenas1/kanenas)"


class FeedError(RuntimeError):
    pass


def _get(url: str, timeout: float = 12.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # venue said no - surface the body
        raise FeedError(f"HTTP {exc.code} from {url}: {exc.read()[:200]!r}") from exc
    except urllib.error.URLError as exc:
        raise FeedError(f"cannot reach {url}: {exc.reason}") from exc
    except ssl.SSLError as exc:
        raise FeedError(f"TLS failure for {url}: {exc}") from exc


@dataclass
class VenueSpec:
    """Everything venue-specific, so the feed body stays generic."""

    name: str
    klines_url: Callable[[str, str, int], str]
    parse_klines: Callable[[object], List[Candle]]
    book_url: Callable[[str, int], str]
    parse_book: Callable[[object], tuple]
    interval_map: dict


def _binance_klines(sym: str, interval: str, limit: int) -> str:
    return f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={interval}&limit={limit}"


def _binance_parse(raw: object) -> List[Candle]:
    out = []
    for row in raw:  # [openTime, o, h, l, c, v, closeTime, ...]
        out.append(Candle(float(row[0]) / 1000.0, float(row[1]), float(row[2]),
                          float(row[3]), float(row[4]), float(row[5])))
    return out


def _binance_book(sym: str, depth: int) -> str:
    return f"https://api.binance.com/api/v3/depth?symbol={sym}&limit={max(5, depth)}"


def _binance_parse_book(raw: object) -> tuple:
    bids = tuple(BookLevel(float(p), float(q)) for p, q in raw.get("bids", []))
    asks = tuple(BookLevel(float(p), float(q)) for p, q in raw.get("asks", []))
    return bids, asks


def _coinbase_klines(sym: str, interval: str, limit: int) -> str:
    return f"https://api.exchange.coinbase.com/products/{sym}/candles?granularity={interval}"


def _coinbase_parse(raw: object) -> List[Candle]:
    # [time, low, high, open, close, volume], newest first
    rows = sorted(raw, key=lambda r: r[0])
    return [Candle(float(r[0]), float(r[3]), float(r[2]), float(r[1]), float(r[4]), float(r[5])) for r in rows]


def _coinbase_book(sym: str, depth: int) -> str:
    return f"https://api.exchange.coinbase.com/products/{sym}/book?level=2"


def _coinbase_parse_book(raw: object) -> tuple:
    bids = tuple(BookLevel(float(p), float(q)) for p, q, *_ in raw.get("bids", [])[:20])
    asks = tuple(BookLevel(float(p), float(q)) for p, q, *_ in raw.get("asks", [])[:20])
    return bids, asks


def _kraken_klines(sym: str, interval: str, limit: int) -> str:
    return f"https://api.kraken.com/0/public/OHLC?pair={sym}&interval={interval}"


def _kraken_parse(raw: object) -> List[Candle]:
    result = raw.get("result", {})
    if raw.get("error"):
        raise FeedError(f"kraken error: {raw['error']}")
    series = next((v for k, v in result.items() if k != "last"), [])
    return [Candle(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])) for r in series]


def _kraken_book(sym: str, depth: int) -> str:
    return f"https://api.kraken.com/0/public/Depth?pair={sym}&count={max(5, depth)}"


def _kraken_parse_book(raw: object) -> tuple:
    result = raw.get("result", {})
    book = next(iter(result.values()), {})
    bids = tuple(BookLevel(float(p), float(q)) for p, q, *_ in book.get("bids", []))
    asks = tuple(BookLevel(float(p), float(q)) for p, q, *_ in book.get("asks", []))
    return bids, asks




# ---------------------------------------------------------------- bitstamp

def _bitstamp_klines(sym: str, interval: str, limit: int) -> str:
    return f"https://www.bitstamp.net/api/v2/ohlc/{sym}/?step={interval}&limit={min(1000, max(3, limit))}"


def _bitstamp_parse(raw: object) -> List[Candle]:
    rows = raw.get("data", {}).get("ohlc", [])
    return [Candle(float(r["timestamp"]), float(r["open"]), float(r["high"]),
                   float(r["low"]), float(r["close"]), float(r.get("volume", 0.0))) for r in rows]


def _bitstamp_book(sym: str, depth: int) -> str:
    return f"https://www.bitstamp.net/api/v2/order_book/{sym}/"


def _bitstamp_parse_book(raw: object) -> tuple:
    bids = tuple(BookLevel(float(p), float(q)) for p, q in raw.get("bids", [])[:20])
    asks = tuple(BookLevel(float(p), float(q)) for p, q in raw.get("asks", [])[:20])
    return bids, asks


# --------------------------------------------------------------------- okx

def _okx_klines(sym: str, interval: str, limit: int) -> str:
    return f"https://www.okx.com/api/v5/market/candles?instId={sym}&bar={interval}&limit={min(300, max(3, limit))}"


def _okx_parse(raw: object) -> List[Candle]:
    if str(raw.get("code", "0")) not in ("0", ""):
        raise FeedError(f"okx error: {raw.get('msg')}")
    rows = sorted(raw.get("data", []), key=lambda r: int(r[0]))   # okx returns newest first
    return [Candle(int(r[0]) / 1000.0, float(r[1]), float(r[2]), float(r[3]),
                   float(r[4]), float(r[5])) for r in rows]


def _okx_book(sym: str, depth: int) -> str:
    return f"https://www.okx.com/api/v5/market/books?instId={sym}&sz={max(5, min(40, depth))}"


def _okx_parse_book(raw: object) -> tuple:
    d = (raw.get("data") or [{}])[0]
    bids = tuple(BookLevel(float(r[0]), float(r[1])) for r in d.get("bids", []))
    asks = tuple(BookLevel(float(r[0]), float(r[1])) for r in d.get("asks", []))
    return bids, asks


# ------------------------------------------------------------------- bybit

def _bybit_klines(sym: str, interval: str, limit: int) -> str:
    return (f"https://api.bybit.com/v5/market/kline?category=spot&symbol={sym}"
            f"&interval={interval}&limit={min(1000, max(3, limit))}")


def _bybit_parse(raw: object) -> List[Candle]:
    if raw.get("retCode", 0) not in (0, "0"):
        raise FeedError(f"bybit error: {raw.get('retMsg')}")
    rows = sorted(raw.get("result", {}).get("list", []), key=lambda r: int(r[0]))
    return [Candle(int(r[0]) / 1000.0, float(r[1]), float(r[2]), float(r[3]),
                   float(r[4]), float(r[5])) for r in rows]


def _bybit_book(sym: str, depth: int) -> str:
    return (f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={sym}"
            f"&limit={max(1, min(50, depth))}")


def _bybit_parse_book(raw: object) -> tuple:
    r = raw.get("result", {})
    bids = tuple(BookLevel(float(x[0]), float(x[1])) for x in r.get("b", []))
    asks = tuple(BookLevel(float(x[0]), float(x[1])) for x in r.get("a", []))
    return bids, asks

VENUES = {
    "binance": VenueSpec("binance", _binance_klines, _binance_parse, _binance_book, _binance_parse_book,
                         {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}),
    "coinbase": VenueSpec("coinbase", _coinbase_klines, _coinbase_parse, _coinbase_book, _coinbase_parse_book,
                          {"1m": "60", "5m": "300", "15m": "900", "1h": "3600", "6h": "21600", "1d": "86400"}),
    "kraken": VenueSpec("kraken", _kraken_klines, _kraken_parse, _kraken_book, _kraken_parse_book,
                        {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "1440"}),
    "bitstamp": VenueSpec("bitstamp", _bitstamp_klines, _bitstamp_parse, _bitstamp_book, _bitstamp_parse_book,
                          {"1m": "60", "5m": "300", "15m": "900", "1h": "3600", "4h": "14400", "1d": "86400"}),
    "okx": VenueSpec("okx", _okx_klines, _okx_parse, _okx_book, _okx_parse_book,
                     {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}),
    "bybit": VenueSpec("bybit", _bybit_klines, _bybit_parse, _bybit_book, _bybit_parse_book,
                       {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}),
}

#: The canonical BTC spot pair on each venue. Venues disagree on spelling
#: (Kraken still calls Bitcoin XBT), so "BTC" is resolved per venue rather than
#: making the user look it up.
BTC_SYMBOL = {
    "binance": "BTCUSDT",
    "coinbase": "BTC-USD",
    "kraken": "XBTUSD",
    "bitstamp": "btcusd",
    "okx": "BTC-USDT",
    "bybit": "BTCUSDT",
}

#: Tried in this order. Binance first for depth and rate limits; Coinbase and
#: Kraken next as the most widely reachable; the rest cover regions where the
#: first three are geo-blocked.
VENUE_ORDER = ["binance", "coinbase", "kraken", "bitstamp", "okx", "bybit"]


def resolve_symbol(venue: str, symbol: Optional[str] = None) -> str:
    """Map a friendly name to the venue's own spelling.

    ``"BTC"`` (in any casing) becomes BTCUSDT on Binance and XBTUSD on Kraken.
    Anything else is passed through untouched, so an explicit pair still works.
    """
    if symbol and symbol.strip().upper() not in ("BTC", "BTC-USD", "BTCUSD", "BITCOIN"):
        return symbol
    return BTC_SYMBOL[venue]


def open_live_feed(
    symbol: Optional[str] = "BTC",
    interval: str = "1m",
    venues: Optional[List[str]] = None,
    with_book: bool = True,
) -> "RestFeed":
    """Return the first venue that actually answers, or raise.

    Exchanges are geo-blocked in different places and go down at different
    times, so binding to one venue makes the bot fail for reasons that have
    nothing to do with trading. This probes each in turn with a real request -
    a venue that resolves but returns garbage is not "reachable".

    Raises ``FeedError`` listing every failure rather than silently degrading
    to simulated data: a bot quietly trading a simulation while you believe it
    is on live prices is far worse than one that refuses to start.
    """
    order = venues or VENUE_ORDER
    failures = []
    for venue in order:
        if venue not in VENUES:
            failures.append(f"{venue}: unknown venue")
            continue
        if interval not in VENUES[venue].interval_map:
            failures.append(f"{venue}: no {interval} interval")
            continue
        feed = RestFeed(resolve_symbol(venue, symbol), venue, interval, with_book=with_book)
        try:
            candles = feed.fetch_history(limit=3)
            if not candles or candles[-1].close <= 0:
                raise FeedError("returned no usable candles")
            return feed
        except FeedError as exc:
            failures.append(f"{venue}: {str(exc)[:110]}")
    raise FeedError(
        "No live venue reachable. Tried:\n  " + "\n  ".join(failures) +
        "\n\nCheck your network, or pass --venue to pick one explicitly. "
        "Use --sim to run the built-in simulator instead (clearly labelled, not live data)."
    )

INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "6h": 21600, "1d": 86400}


class RestFeed:
    """Polls closed candles from a public venue and yields them once each."""

    def __init__(
        self,
        symbol: str,
        venue: str = "binance",
        interval: str = "1m",
        limit: int = 500,
        with_book: bool = True,
        poll_slack: float = 2.0,
    ) -> None:
        if venue not in VENUES:
            raise ValueError(f"unknown venue {venue!r}; choose from {sorted(VENUES)}")
        self.spec = VENUES[venue]
        if interval not in self.spec.interval_map:
            raise ValueError(f"{venue} does not support interval {interval!r}")
        self.symbol = symbol
        self.name = f"{venue}:{symbol}:{interval}"
        self.venue = venue
        self.interval = interval
        self.limit = limit
        self.with_book = with_book
        self.poll_slack = poll_slack
        self.bar_seconds = INTERVAL_SECONDS[interval]
        self._last_ts = 0.0

    def fetch_history(self, limit: Optional[int] = None) -> List[Candle]:
        """Historical closed bars - used to warm indicators and to backtest."""
        raw = _get(self.spec.klines_url(self.symbol, self.spec.interval_map[self.interval], limit or self.limit))
        candles = self.spec.parse_klines(raw)
        if not candles:
            raise FeedError(f"{self.name} returned no candles")
        return candles[: (limit or self.limit)] if len(candles) > (limit or self.limit) else candles

    def fetch_book(self, depth: int = 10) -> Optional[OrderBook]:
        if not self.with_book:
            return None
        try:
            bids, asks = self.spec.parse_book(_get(self.spec.book_url(self.symbol, depth)))
            return OrderBook(time.time(), bids, asks)
        except FeedError:
            return None  # book is a nice-to-have; never kill the feed for it

    def stream(self) -> Iterator[MarketEvent]:
        """Yield each newly *closed* bar, polling just after each bar boundary."""
        backoff = 1.0
        while True:
            try:
                candles = self.fetch_history(limit=3)
                backoff = 1.0
            except FeedError as exc:
                # transient venue/network trouble: back off, never spin
                time.sleep(min(backoff, 60.0))
                backoff *= 2
                continue
            closed = candles[-2] if len(candles) >= 2 else candles[-1]
            if closed.ts > self._last_ts:
                self._last_ts = closed.ts
                yield MarketEvent(self.symbol, closed, self.fetch_book())
            now = time.time()
            next_close = (now // self.bar_seconds + 1) * self.bar_seconds
            time.sleep(max(1.0, next_close - now + self.poll_slack))
