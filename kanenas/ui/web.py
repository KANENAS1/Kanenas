"""Browser dashboard served from the standard library.

``http.server`` plus a JSON endpoint, deliberately: adding Flask/FastAPI would
buy nothing here and would cost the "clone it and run it" property that makes
this repo easy to trust.  The engine runs on a worker thread and simply mutates
its own state; the HTTP thread serialises a snapshot whenever the page asks.

Binds to 127.0.0.1 by default.  This endpoint exposes your live position and
P&L and has no authentication, so it should never be put on a public interface.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from ..core.types import Direction
from ..engine import TradingEngine

STATIC = Path(__file__).parent / "static"


def serialise(engine: TradingEngine, mode: str, venue: str, started: float) -> dict:
    """Snapshot everything the page draws.  Read-only; never mutates engine state."""
    p = engine.portfolio
    st = engine.state
    price = st.price
    eq = p.equity(price)
    pos = p.position
    d = st.decision

    candles = [
        {"t": cd.ts, "o": cd.open, "h": cd.high, "l": cd.low, "c": cd.close, "v": cd.volume}
        for cd in list(engine.ind.candles)[-120:]
    ]
    curve = [pt.equity for pt in p.equity_curve][-400:]

    signals = []
    if d is not None:
        for sig in d.signals:
            signals.append({
                "name": sig.source,
                "direction": sig.direction.name,
                "confidence": round(sig.confidence, 4),
                "weight": round(d.weights.get(sig.source, 1.0), 3),
                "reason": sig.reason,
            })

    book = None
    if st.book is not None:
        book = {
            "bids": [{"p": l.price, "s": l.size} for l in st.book.bids[:8]],
            "asks": [{"p": l.price, "s": l.size} for l in st.book.asks[:8]],
            "spread": st.book.spread,
            "imbalance": round(st.book.imbalance(), 4),
        }

    return {
        "ts": time.time(),
        "clock": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "symbol": engine.cfg.symbol,
        "mode": mode,
        "venue": venue,
        "uptime": time.time() - started,
        "bar": st.bar,
        "price": price,
        "candles": candles,
        "equity": eq,
        "equity_curve": curve,
        "starting_cash": p.starting_cash,
        "cash": p.cash,
        "total_return": p.total_return,
        "realized_pnl": p.realized_pnl,
        "fees_paid": p.fees_paid,
        "peak_equity": p.peak_equity,
        "drawdown": p.drawdown,
        "max_drawdown": p.max_drawdown,
        "trades_count": len(p.trades),
        "win_rate": p.win_rate,
        "profit_factor": (None if p.profit_factor == float("inf") else p.profit_factor),
        "expectancy": p.expectancy,
        "win_streak": p.win_streak,
        "loss_streak": p.loss_streak,
        "best_win_streak": p.best_win_streak,
        "worst_loss_streak": p.worst_loss_streak,
        "position": None if not pos.is_open else {
            "side": "LONG" if pos.qty > 0 else "SHORT",
            "qty": abs(pos.qty),
            "entry": pos.entry_price,
            "stop": pos.stop_price,
            "target": pos.take_profit,
            "unrealized": pos.unrealized(price),
            "unrealized_pct": pos.unrealized_pct(price),
            "bars_held": pos.bars_held,
        },
        "ensemble": None if d is None else {
            "direction": d.direction.name,
            "score": round(d.score, 4),
            "confidence": round(d.confidence, 4),
            "agreement": round(d.agreement, 4),
            "reason": d.reason,
        },
        "signals": signals,
        "book": book,
        "attribution": engine.ensemble.snapshot(),
        "halted": engine.risk.halted,
        "halt_reason": engine.risk.halt_reason,
        "risk_rejections": dict(engine.risk.rejections),
        "trades": [
            {"side": t.direction.name, "reason": t.reason.value, "net": t.net_pnl,
             "ret": t.return_pct, "bars": t.bars_held, "entry": t.entry_price,
             "exit": t.exit_price, "ts": t.exit_ts}
            for t in p.trades[-40:][::-1]
        ],
        "log": [
            {"ts": e.ts, "bar": e.bar, "kind": e.kind, "message": e.message,
             "price": e.price, "pnl": e.pnl,
             "time": datetime.fromtimestamp(e.ts, tz=timezone.utc).strftime("%H:%M:%S")}
            for e in engine.state.log[-60:][::-1]
        ],
    }


class DashboardServer:
    def __init__(self, engine: TradingEngine, host: str = "127.0.0.1", port: int = 8787,
                 mode: str = "PAPER", venue: str = "simulator") -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.mode = mode
        self.venue = venue
        self.started = time.time()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    html = (STATIC / "dashboard.html").read_bytes()
                    self._send(html, "text/html; charset=utf-8")
                elif path == "/api/state":
                    payload = serialise(server.engine, server.mode, server.venue, server.started)
                    self._send(json.dumps(payload).encode(), "application/json")
                elif path == "/api/health":
                    self._send(b'{"ok":true}', "application/json")
                else:
                    self._send(b"not found", "text/plain", 404)

            def log_message(self, *args) -> None:
                pass  # the dashboard is the output; access logs would corrupt it

        return Handler

    def start(self, open_browser: bool = False) -> str:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="kanenas-web")
        self._thread.start()
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass
        return self.url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
