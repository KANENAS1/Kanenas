"""Terminal drawing primitives: colour, sparklines, candles, panels.

Written against raw ANSI rather than a TUI library so the dashboard has no
dependencies and works over plain SSH.  Colour degrades to nothing when stdout
is not a TTY (or ``NO_COLOR`` is set), so piping the dashboard to a file gives
clean text instead of escape soup.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from typing import Iterable, List, Optional, Sequence

# ------------------------------------------------------------------- colour

def _enable_windows_ansi() -> bool:
    """Turn on VT processing so escape codes render instead of printing raw.

    Windows Terminal handles ANSI natively, but the legacy console host - still
    what you get from an old PowerShell or cmd shortcut - prints the escape
    bytes literally unless a process asks for virtual-terminal mode. Without
    this the dashboard renders as pages of garbage on exactly the machines
    least likely to know why.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        return bool(kernel32.SetConsoleMode(
            handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except Exception:
        return False                                  # plain text still works


_ENABLED = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
    and _enable_windows_ansi()
)


def enable_color(flag: Optional[bool] = None) -> bool:
    global _ENABLED
    if flag is not None:
        _ENABLED = flag
    return _ENABLED


def rgb(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"


# The palette always holds real escape sequences.  Gating them at *definition*
# time would freeze the choice at import, so a later enable_color() call could
# never turn colour back on - and a module imported before stdout is known
# would be permanently monochrome.  ``c()`` is the single gate instead.
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

# neon palette, tuned to read on a dark terminal
MAGENTA = rgb(255, 60, 172)
CYAN = rgb(60, 233, 233)
GREEN = rgb(56, 226, 143)
RED = rgb(255, 78, 100)
YELLOW = rgb(255, 199, 74)
WHITE = rgb(232, 236, 244)
GREY = rgb(120, 128, 150)
DARK = rgb(70, 76, 96)
BLUE = rgb(94, 160, 255)


def c(text: str, colour: str, bold: bool = False) -> str:
    if not _ENABLED:
        return text
    return f"{BOLD if bold else ''}{colour}{text}{RESET}"


def pnl_colour(x: float) -> str:
    return GREEN if x > 0 else (RED if x < 0 else GREY)


def visible_len(s: str) -> int:
    """Length of ``s`` ignoring ANSI escape sequences."""
    out, i, n = 0, 0, len(s)
    while i < n:
        if s[i] == "\x1b":
            j = s.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out += 1
        i += 1
    return out


def pad(s: str, width: int, align: str = "left") -> str:
    gap = max(0, width - visible_len(s))
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def truncate(s: str, width: int) -> str:
    if visible_len(s) <= width:
        return s
    if not _ENABLED:
        # nothing to reset when colour is off; appending RESET would inject
        # escape bytes into output the caller asked to keep clean
        return s[:width] if "\x1b" not in s else _truncate_ansi(s, width) 
    return _truncate_ansi(s, width) + RESET


def _truncate_ansi(s: str, width: int) -> str:
    """Cut to ``width`` visible characters, keeping escape sequences intact."""
    out, count, i = [], 0, 0
    while i < len(s) and count < width:
        if s[i] == "\x1b":
            j = s.find("m", i)
            if j == -1:
                break
            out.append(s[i:j + 1])
            i = j + 1
            continue
        out.append(s[i])
        count += 1
        i += 1
    return "".join(out)


def term_size(default=(120, 40)) -> tuple[int, int]:
    try:
        size = shutil.get_terminal_size(default)
        return max(80, size.columns), max(24, size.lines)
    except Exception:
        return default


# --------------------------------------------------------------- sparklines

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], width: int = 40) -> str:
    """Compact unicode line chart of the last ``width`` values."""
    if not values:
        return ""
    data = list(values)[-width:]
    lo, hi = min(data), max(data)
    if hi - lo < 1e-12:
        return _BLOCKS[0] * len(data)
    span = hi - lo
    return "".join(_BLOCKS[min(7, int((v - lo) / span * 7.999))] for v in data)


def area_chart(values: Sequence[float], width: int, height: int,
               colour: str = CYAN, baseline: Optional[float] = None) -> List[str]:
    """Filled line chart, ``height`` rows tall - used for the equity curve."""
    if not values or height < 1 or width < 2:
        return [" " * width for _ in range(max(0, height))]
    data = list(values)
    # resample to exactly `width` columns
    if len(data) > width:
        step = len(data) / width
        data = [data[min(len(data) - 1, int(i * step))] for i in range(width)]
    lo, hi = min(data), max(data)
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    rows = [[" "] * width for _ in range(height)]
    for x, v in enumerate(data):
        level = (v - lo) / (hi - lo)
        top = height - 1 - min(height - 1, int(level * (height - 1) + 0.5))
        for y in range(top, height):
            rows[y][x] = "█" if y > top else "▀"
    base_row = None
    if baseline is not None:
        blevel = (baseline - lo) / (hi - lo)
        base_row = height - 1 - min(height - 1, int(blevel * (height - 1) + 0.5))
    out = []
    for y, row in enumerate(rows):
        line = "".join(row)
        if base_row is not None and y == base_row:
            line = "".join(ch if ch != " " else "·" for ch in line)
        out.append(c(line, colour))
    return out


def candle_chart(candles: Sequence, width: int, height: int,
                 marker_price: Optional[float] = None) -> List[str]:
    """ASCII candlesticks: wick as │, body as █, green up / red down."""
    if not candles or width < 4 or height < 3:
        return [" " * width for _ in range(max(0, height))]
    data = list(candles)[-width:]
    hi = max(x.high for x in data)
    lo = min(x.low for x in data)
    if hi - lo < 1e-12:
        hi, lo = hi + 1.0, lo - 1.0
    span = hi - lo

    def row_of(price: float) -> int:
        frac = (price - lo) / span
        return height - 1 - min(height - 1, max(0, int(frac * (height - 1) + 0.5)))

    grid = [[" "] * len(data) for _ in range(height)]
    colours = [[GREY] * len(data) for _ in range(height)]
    for x, cd in enumerate(data):
        col = GREEN if cd.is_bull else RED
        r_hi, r_lo = row_of(cd.high), row_of(cd.low)
        r_o, r_cl = row_of(cd.open), row_of(cd.close)
        body_top, body_bot = min(r_o, r_cl), max(r_o, r_cl)
        for y in range(r_hi, r_lo + 1):
            grid[y][x] = "│"
            colours[y][x] = col
        for y in range(body_top, body_bot + 1):
            grid[y][x] = "█"
            colours[y][x] = col

    marker_row = row_of(marker_price) if marker_price is not None else None
    out = []
    for y in range(height):
        cells = []
        for x in range(len(data)):
            ch = grid[y][x]
            cells.append(c(ch, colours[y][x]) if ch != " " else " ")
        line = "".join(cells)
        if marker_row is not None and y == marker_row:
            line = pad(line, len(data)) + c(f"◄ {marker_price:,.2f}", YELLOW, True)
        out.append(line)
    return out


def price_axis(candles: Sequence, height: int, width: int = 10) -> List[str]:
    if not candles:
        return [" " * width] * height
    hi = max(x.high for x in candles)
    lo = min(x.low for x in candles)
    out = []
    for y in range(height):
        frac = 1.0 - (y / max(1, height - 1))
        out.append(c(pad(f"{lo + (hi - lo) * frac:,.0f}", width, "right"), DARK))
    return out


def histogram(values: Sequence[float], width: int, colour: str = MAGENTA) -> str:
    """Horizontal bar row - used for the analytics strip."""
    if not values:
        return " " * width
    data = list(values)[-width:]
    peak = max((abs(v) for v in data), default=0.0)
    if peak < 1e-12:
        return "▁" * len(data)
    out = []
    for v in data:
        idx = min(7, int(abs(v) / peak * 7.999))
        out.append(c(_BLOCKS[idx], GREEN if v >= 0 else RED))
    return "".join(out)


# -------------------------------------------------------------------- boxes

def panel(title: str, body: List[str], width: int, colour: str = DARK,
          title_colour: str = MAGENTA) -> List[str]:
    """Draw a titled box exactly ``width`` columns wide, borders included."""
    inner = width - 2
    head = f"┌─ {c(title, title_colour, True)} "
    head_len = visible_len(head)
    head = head + c("─" * max(0, width - head_len - 1), colour) + c("┐", colour)
    lines = [c("┌", colour) + head[1:] if False else head]
    for row in body:
        lines.append(c("│", colour) + pad(truncate(row, inner), inner) + c("│", colour))
    lines.append(c("└" + "─" * inner + "┘", colour))
    return lines


def hjoin(blocks: List[List[str]], gap: int = 1) -> List[str]:
    """Place rendered blocks side by side, padding to the tallest."""
    if not blocks:
        return []
    height = max(len(b) for b in blocks)
    widths = [max((visible_len(r) for r in b), default=0) for b in blocks]
    out = []
    for y in range(height):
        parts = []
        for i, b in enumerate(blocks):
            row = b[y] if y < len(b) else ""
            parts.append(pad(row, widths[i]))
        out.append((" " * gap).join(parts))
    return out


def bar_gauge(value: float, width: int, colour: str = GREEN, track: str = "░") -> str:
    """Horizontal 0..1 gauge."""
    v = max(0.0, min(1.0, value))
    filled = int(round(v * width))
    return c("█" * filled, colour) + c(track * (width - filled), DARK)
