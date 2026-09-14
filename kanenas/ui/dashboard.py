"""The live terminal dashboard.

Redraws the whole frame each tick into one string and writes it in a single
``sys.stdout.write``.  Painting panel-by-panel would tear visibly; one write per
frame is effectively atomic at terminal speed, which is why the panes stay
aligned even while the log scrolls.

Everything shown is read from ``TradingEngine`` state - the dashboard owns no
trading state of its own and can be removed entirely without touching a line of
trading logic.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

from ..core.types import Direction
from ..engine import TradingEngine
from . import render as R


class Dashboard:
    def __init__(self, engine: TradingEngine, title: str = "KANENAS TERMINAL",
                 mode: str = "PAPER", venue: str = "simulator", stream=None) -> None:
        self.engine = engine
        self.title = title
        self.mode = mode
        self.venue = venue
        self.out = stream or sys.stdout
        self.started = time.time()
        self.frames = 0
        self._alt_screen = False

    # ------------------------------------------------------------- lifecycle

    def enter(self) -> None:
        if self.out.isatty():
            self.out.write("\x1b[?25l\x1b[?1049h")  # hide cursor, alt screen
            self._alt_screen = True
        self.out.flush()

    def exit(self) -> None:
        if self._alt_screen:
            self.out.write("\x1b[?1049l\x1b[?25h")
            self._alt_screen = False
        self.out.flush()

    # ----------------------------------------------------------------- panes

    def _header(self, width: int) -> str:
        e = self.engine
        p = e.portfolio
        price = e.state.price
        eq = p.equity(price)
        ret = (eq - p.starting_cash) / p.starting_cash if p.starting_cash else 0.0
        clock = datetime.now(timezone.utc).strftime("%H:%M:%S")
        mode_col = R.RED if self.mode == "LIVE" else R.CYAN
        left = (f"{R.c('◆ ' + self.title, R.MAGENTA, True)} {R.c('·', R.DARK)} "
                f"{R.c(e.cfg.symbol, R.WHITE, True)} {R.c('·', R.DARK)} "
                f"{R.c(self.mode, mode_col, True)} {R.c('·', R.DARK)} {R.c(self.venue, R.GREY)}")
        # Listed most-important first, which is also display order: a narrow
        # terminal sheds trailing detail (BAR, then WIN...) and always keeps
        # price and equity, rather than overflowing and shearing the panels
        # below it.
        segments = [
            f"{R.c('PRICE', R.DARK)} {R.c(f'${price:,.2f}', R.WHITE, True)}",
            f"{R.c('EQUITY', R.DARK)} {R.c(f'${eq:,.2f}', R.pnl_colour(ret), True)}",
            f"{R.c('TRADES', R.DARK)} {R.c(str(len(p.trades)), R.WHITE)}",
            f"{R.c('WIN', R.DARK)} {R.c(f'{p.win_rate:.1%}', R.WHITE)}",
            f"{R.c('BAR', R.DARK)} {R.c(str(e.state.bar), R.GREY)}",
        ]
        tail = R.c(clock, R.CYAN)
        chosen: List[str] = []
        for seg in segments:
            sized = "   ".join(chosen + [seg]) + "   " + tail
            if R.visible_len(left) + 2 + R.visible_len(sized) > width:
                break
            chosen.append(seg)
        right = ("   ".join(chosen) + "   " + tail) if chosen else tail
        gap = max(1, width - R.visible_len(left) - R.visible_len(right))
        line = left + " " * gap + right
        return R.truncate(line, width)

    def _wallet(self, width: int) -> List[str]:
        e = self.engine
        p = e.portfolio
        price = e.state.price
        eq = p.equity(price)
        ret = (eq - p.starting_cash) / p.starting_cash if p.starting_cash else 0.0
        pos = p.position
        body = [
            R.c(f"${eq:,.2f}", R.pnl_colour(ret), True),
            f"{R.c(f'{ret:+.2%}', R.pnl_colour(ret))} {R.c('since start', R.DARK)}",
            "",
            f"{R.c('cash     ', R.DARK)}${p.cash:,.2f}",
            f"{R.c('realised ', R.DARK)}{R.c(f'{p.realized_pnl:+,.2f}', R.pnl_colour(p.realized_pnl))}",
            f"{R.c('fees     ', R.DARK)}{R.c(f'-{p.fees_paid:,.2f}', R.RED)}",
            f"{R.c('peak     ', R.DARK)}${p.peak_equity:,.2f}",
            f"{R.c('drawdown ', R.DARK)}{R.c(f'{p.drawdown:.2%}', R.RED if p.drawdown > 0.05 else R.GREY)}",
            "",
        ]
        if pos.is_open:
            u = pos.unrealized(price)
            arrow = "▲ LONG" if pos.qty > 0 else "▼ SHORT"
            body += [
                f"{R.c(arrow, R.GREEN if pos.qty > 0 else R.RED, True)} {R.c(f'{abs(pos.qty):.6f}', R.WHITE)}",
                f"{R.c('entry    ', R.DARK)}{pos.entry_price:,.2f}",
                f"{R.c('u-pnl    ', R.DARK)}{R.c(f'{u:+,.2f} ({pos.unrealized_pct(price):+.2%})', R.pnl_colour(u))}",
                f"{R.c('stop     ', R.DARK)}{R.c(f'{pos.stop_price:,.2f}', R.RED)}",
                f"{R.c('target   ', R.DARK)}{R.c(f'{pos.take_profit:,.2f}', R.GREEN)}",
                f"{R.c('held     ', R.DARK)}{pos.bars_held} bars",
            ]
        else:
            body += [R.c("● FLAT", R.GREY, True), R.c("waiting for setup", R.DARK)]
        return R.panel("WALLET", body, width)

    def _chart(self, width: int) -> List[str]:
        e = self.engine
        inner = width - 2
        candles = list(e.ind.candles)
        height = 12
        axis_w = 11
        chart_w = max(10, inner - axis_w - 12)
        rows = R.candle_chart(candles, chart_w, height, marker_price=e.state.price)
        axis = R.price_axis(list(candles)[-chart_w:], height, axis_w)
        body = [a + " " + r for a, r in zip(axis, rows)]
        last = candles[-1] if candles else None
        if last:
            chg = (last.close - last.open) / last.open if last.open else 0.0
            body.append(R.c(f"  O {last.open:,.2f}  H {last.high:,.2f}  L {last.low:,.2f}  "
                            f"C {last.close:,.2f}  ", R.DARK) + R.c(f"{chg:+.3%}", R.pnl_colour(chg)))
        return R.panel(f"{e.cfg.symbol} · LIVE", body, width)

    def _streak(self, width: int) -> List[str]:
        p = self.engine.portfolio
        streak = p.win_streak if p.win_streak else -p.loss_streak
        col = R.GREEN if streak > 0 else (R.RED if streak < 0 else R.GREY)
        label = "WIN STREAK" if streak > 0 else ("LOSS STREAK" if streak < 0 else "NO STREAK")
        recent = [t.net_pnl for t in p.trades[-24:]]
        body = [
            R.c(f"×{abs(streak)}", col, True) + "  " + R.c(label, col),
            "",
            R.histogram(recent, width - 4),
            R.c("last 24 trades P&L", R.DARK),
            "",
            f"{R.c('best run   ', R.DARK)}{R.c(str(p.best_win_streak), R.GREEN)}",
            f"{R.c('worst run  ', R.DARK)}{R.c(str(p.worst_loss_streak), R.RED)}",
            f"{R.c('win rate   ', R.DARK)}{p.win_rate:.1%}",
            f"{R.c('p-factor   ', R.DARK)}{p.profit_factor:.2f}" if p.profit_factor != float("inf") else
            f"{R.c('p-factor   ', R.DARK)}∞",
            f"{R.c('expectancy ', R.DARK)}{R.c(f'{p.expectancy:+,.2f}', R.pnl_colour(p.expectancy))}",
        ]
        return R.panel("STREAK", body, width)

    def _signals(self, width: int) -> List[str]:
        e = self.engine
        d = e.state.decision
        body: List[str] = []
        if d is None:
            body.append(R.c("warming up indicators…", R.DARK))
        else:
            for sig in d.signals:
                w = d.weights.get(sig.source, 1.0)
                if sig.direction is Direction.LONG:
                    tag, col = "▲ LONG ", R.GREEN
                elif sig.direction is Direction.SHORT:
                    tag, col = "▼ SHORT", R.RED
                else:
                    tag, col = "● flat ", R.DARK
                gauge = R.bar_gauge(sig.confidence, 10, col)
                name = R.c(R.pad(sig.source.upper(), 9), R.WHITE)
                body.append(f"{name}{R.c(tag, col)} {gauge} {R.c(f'w{w:.2f}', R.DARK)}")
                body.append("  " + R.c(R.truncate(sig.reason, width - 6), R.DARK))
            arrow = ("▲ LONG" if d.direction is Direction.LONG else
                     "▼ SHORT" if d.direction is Direction.SHORT else "● FLAT")
            col = R.GREEN if d.direction is Direction.LONG else (R.RED if d.direction is Direction.SHORT else R.GREY)
            body += ["", f"{R.c('ENSEMBLE', R.MAGENTA, True)}  {R.c(arrow, col, True)}  "
                         f"{R.c(f'score {d.score:+.3f}', R.WHITE)}  {R.c(f'agree {d.agreement:.0%}', R.CYAN)}",
                     "  " + R.c(R.truncate(d.reason, width - 6), R.DARK)]
        if e.risk.halted:
            body += ["", R.c(f"⛔ HALTED · {e.risk.halt_reason}", R.RED, True)]
        return R.panel("SIGNAL MATRIX", body, width)

    def _book(self, width: int) -> List[str]:
        book = self.engine.state.book
        body: List[str] = []
        if book is None:
            body.append(R.c("no order book on this feed", R.DARK))
        else:
            depth = 5
            peak = max([l.size for l in book.asks[:depth]] + [l.size for l in book.bids[:depth]] + [1e-9])
            for lvl in reversed(book.asks[:depth]):
                bar = R.bar_gauge(lvl.size / peak, 10, R.RED)
                body.append(f"{R.c(f'{lvl.price:>10,.2f}', R.RED)} {bar} {R.c(f'{lvl.size:6.2f}', R.DARK)}")
            imb = book.imbalance()
            body.append(R.c(f"{'spread':>10} ", R.DARK) + R.c(f"{book.spread:,.2f}", R.YELLOW)
                        + R.c(f"   imb {imb:+.2f}", R.GREEN if imb > 0 else R.RED))
            for lvl in book.bids[:depth]:
                bar = R.bar_gauge(lvl.size / peak, 10, R.GREEN)
                body.append(f"{R.c(f'{lvl.price:>10,.2f}', R.GREEN)} {bar} {R.c(f'{lvl.size:6.2f}', R.DARK)}")
        return R.panel("ORDER BOOK", body, width)

    def _equity(self, width: int) -> List[str]:
        p = self.engine.portfolio
        curve = [pt.equity for pt in p.equity_curve]
        inner = width - 2
        body = R.area_chart(curve, inner, 7, R.CYAN, baseline=p.starting_cash)
        ret = p.total_return
        body.append(R.c(f"{p.starting_cash:,.0f}", R.DARK) + " → "
                    + R.c(f"{curve[-1]:,.2f}" if curve else "-", R.pnl_colour(ret), True)
                    + "  " + R.c(f"{ret:+.2%}", R.pnl_colour(ret)))
        return R.panel("EQUITY", body, width)

    def _trades(self, width: int) -> List[str]:
        p = self.engine.portfolio
        body = [R.c(f"{'side':<5}{'exit':<6}{'net':>10}{'ret':>9}{'bars':>6}", R.DARK)]
        for t in p.trades[-7:][::-1]:
            side = "LONG" if t.direction is Direction.LONG else "SHORT"
            col = R.pnl_colour(t.net_pnl)
            body.append(f"{R.c(f'{side:<5}', R.GREEN if side == 'LONG' else R.RED)}"
                        f"{R.c(f'{t.reason.value:<6}', R.DARK)}"
                        f"{R.c(f'{t.net_pnl:>+10,.2f}', col)}"
                        f"{R.c(f'{t.return_pct:>+9.2%}', col)}"
                        f"{R.c(f'{t.bars_held:>6}', R.DARK)}")
        if not p.trades:
            body.append(R.c("no closed trades yet", R.DARK))
        return R.panel("RECENT TRADES", body, width)

    def _log(self, width: int, rows: int) -> List[str]:
        entries = self.engine.state.log[-rows:]
        kinds = {"ENTRY": R.CYAN, "EXIT": R.MAGENTA, "SKIP": R.DARK, "HALT": R.RED, "INFO": R.GREY}
        body = []
        for e in entries:
            stamp = datetime.fromtimestamp(e.ts, tz=timezone.utc).strftime("%H:%M:%S")
            col = kinds.get(e.kind, R.GREY)
            line = (f"{R.c(stamp, R.DARK)} {R.c(f'#{e.bar:<6}', R.DARK)} "
                    f"{R.c(f'{e.kind:<5}', col)} {e.message}")
            body.append(R.truncate(line, width - 3))
        while len(body) < rows:
            body.append("")
        return R.panel("EXECUTION LOG · LIVE", body, width)

    # ----------------------------------------------------------------- frame

    def frame(self, width: Optional[int] = None) -> str:
        w, _ = R.term_size()
        width = width or w
        width = max(96, min(width, 200))

        left_w = 30
        right_w = 30
        mid_w = width - left_w - right_w - 2

        top = R.hjoin([self._wallet(left_w), self._chart(mid_w), self._streak(right_w)])
        sig_w = (width - 1) // 2
        book_w = width - sig_w - 1
        mid = R.hjoin([self._signals(sig_w), self._book(book_w)])
        eq_w = (width - 1) // 2
        tr_w = width - eq_w - 1
        low = R.hjoin([self._equity(eq_w), self._trades(tr_w)])
        log = self._log(width, 8)

        out = [self._header(width), ""]
        out += top + mid + low + log
        self.frames += 1
        return "\n".join(out)

    def draw(self) -> None:
        text = self.frame()
        if self.out.isatty():
            self.out.write("\x1b[H\x1b[2J" + text + "\n")
        else:
            self.out.write(text + "\n")
        self.out.flush()
