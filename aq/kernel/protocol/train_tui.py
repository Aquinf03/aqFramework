"""Live train dashboard (TTY) — monitor-style panel + step table.

Falls back to plain append when stderr is not a TTY (CI / pipes).
Disable with AQ_TUI=0.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from protocol.term_table import fmt_cell

# ANSI
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_MAGENTA_BG = "\033[48;5;98m\033[97m"  # purple-ish highlight like the inspo
_HIDE = "\033[?25l"
_SHOW = "\033[?25h"


def enabled() -> bool:
    if os.environ.get("AQ_TUI", "1").strip() in ("0", "false", "no", "off"):
        return False
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


class TrainTui:
    def __init__(self) -> None:
        self.info: dict[str, Any] = {}
        self.latest: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.epochs: list[dict[str, Any]] = []
        self._lines = 0
        self._started = False
        self._last_draw = 0.0
        self.max_rows = 12

    def on_start(self, body: dict[str, Any]) -> None:
        for k in ("op", "method", "family", "run_id"):
            if body.get(k) is not None:
                self.info[k] = body[k]
        self._draw(force=True)

    def on_info(self, body: dict[str, Any]) -> None:
        for k, v in body.items():
            if k in ("ts", "event", "run_id", "op", "elapsed_ms") or v is None:
                continue
            self.info[k] = v
        self._draw(force=True)

    def on_step(self, body: dict[str, Any]) -> None:
        self.latest = dict(body)
        self.steps.append(dict(body))
        if len(self.steps) > 200:
            self.steps = self.steps[-200:]
        self._draw()

    def on_epoch(self, body: dict[str, Any]) -> None:
        self.epochs.append(dict(body))
        self.latest = {**self.latest, **body, "event": "epoch"}
        self._draw(force=True)

    def on_end(self, body: dict[str, Any]) -> None:
        self.latest = {**self.latest, **body, "event": "end"}
        self._draw(force=True)
        self._finish()

    def on_error(self, body: dict[str, Any]) -> None:
        self.info["error"] = body.get("error") or "error"
        self._draw(force=True)
        self._finish()

    def _finish(self) -> None:
        if self._started:
            sys.stderr.write(_SHOW)
            sys.stderr.flush()
            self._started = False
            self._lines = 0

    def _bar(self, frac: float, width: int = 18) -> str:
        frac = max(0.0, min(1.0, float(frac)))
        n = int(round(frac * width))
        filled = "|" * n
        empty = "." * (width - n)
        return f"[{_GREEN}{filled}{_RESET}{_DIM}{empty}{_RESET}]"

    def _progress(self, label: str, cur: Any, total: Any, width: int = 16) -> str:
        try:
            c = float(cur)
            t = float(total) if total not in (None, "", 0) else None
        except (TypeError, ValueError):
            return f"{label}  {fmt_cell(cur)}"
        if t and t > 0:
            return f"{label}  {self._bar(c / t, width)}  {fmt_cell(int(c))}/{fmt_cell(int(t))}"
        return f"{label}  {fmt_cell(cur)}"

    def _frame(self) -> list[str]:
        info = self.info
        lat = self.latest
        elapsed = lat.get("elapsed_ms")
        ago = ""
        if elapsed is not None:
            ago = f"elapsed {int(elapsed)}ms"
        lines: list[str] = []
        lines.append(f"  {_DIM}aq train{f'  ·  {ago}' if ago else ''}{_RESET}")

        # top stats — three columns like the monitor
        left = [
            self._progress(
                "epoch",
                lat.get("epoch"),
                info.get("epochs") or lat.get("epochs"),
                14,
            ),
            self._progress(
                "step ",
                lat.get("step"),
                lat.get("steps") or info.get("steps"),
                14,
            ),
        ]
        mid = [
            f"loss   {fmt_cell(lat.get('loss'))}",
            f"acc    {fmt_cell(lat.get('acc'))}",
            f"lr     {fmt_cell(lat.get('lr'))}",
        ]
        right = [
            f"val_loss  {fmt_cell(lat.get('val_loss'))}",
            f"val_acc   {fmt_cell(lat.get('val_acc'))}",
            f"device    {fmt_cell(info.get('device') or '—')}",
        ]
        # pad columns
        def pad_col(rows: list[str], w: int) -> list[str]:
            # strip ansi for width — approximate
            out = []
            for r in rows:
                visible = _visible_len(r)
                out.append(r + " " * max(0, w - visible))
            while len(out) < 3:
                out.append(" " * w)
            return out

        lw, mw, rw = 36, 22, 24
        L, M, R = pad_col(left, lw), pad_col(mid, mw), pad_col(right, rw)
        for i in range(3):
            lines.append(f"  {L[i]} {_DIM}|{_RESET} {M[i]} {_DIM}|{_RESET} {R[i]}")

        # meta strip
        meta_bits = []
        for k in ("method", "family", "arch", "classes", "images", "image_size", "size"):
            if info.get(k) is not None:
                label = "size" if k in ("image_size", "size") else k
                meta_bits.append(f"{label} {fmt_cell(info[k])}")
        if meta_bits:
            lines.append(f"  {_DIM}{'  ·  '.join(meta_bits)}{_RESET}")

        if info.get("error"):
            lines.append(f"  {_BOLD}error  {info['error']}{_RESET}")

        # step table
        headers = ["step", "loss", "lr", "acc", "epoch", "time"]
        rows = self.steps[-self.max_rows :]
        grid = [headers]
        for s in rows:
            step_n = s.get("step")
            total = s.get("steps") or info.get("steps")
            step_label = f"{step_n}/{total}" if total not in (None, "") and step_n is not None else fmt_cell(step_n)
            grid.append(
                [
                    step_label,
                    fmt_cell(s.get("loss")),
                    fmt_cell(s.get("lr")),
                    fmt_cell(s.get("acc")),
                    fmt_cell(s.get("epoch")),
                    f"{int(s['elapsed_ms'])}ms" if s.get("elapsed_ms") is not None else "—",
                ]
            )
        widths = [0] * len(headers)
        for row in grid:
            for i, c in enumerate(row):
                widths[i] = max(widths[i], len(str(c)))

        def fmt_row(cells: list[str], highlight: bool = False) -> str:
            parts = []
            for i, c in enumerate(cells):
                w = widths[i]
                parts.append(c.rjust(w) if i else c.ljust(w))
            body = "  ".join(parts)
            if highlight:
                return f"  {_MAGENTA_BG} {body} {_RESET}"
            return f"  {body}"

        lines.append("")
        lines.append(fmt_row(headers))
        lines.append(f"  {_DIM}{'  '.join('─' * w for w in widths)}{_RESET}")
        for i, row in enumerate(grid[1:]):
            hl = i == len(grid) - 2  # last data row
            lines.append(fmt_row(row, highlight=hl))

        # latest epoch line
        if self.epochs:
            e = self.epochs[-1]
            lines.append(
                f"  {_DIM}epoch {fmt_cell(e.get('epoch'))}  "
                f"loss {fmt_cell(e.get('loss') or e.get('train_loss'))}  "
                f"acc {fmt_cell(e.get('acc') or e.get('train_acc'))}  "
                f"val_acc {fmt_cell(e.get('val_acc'))}{_RESET}"
            )

        if lat.get("event") == "end":
            lines.append(
                f"  {_BOLD}done{_RESET}  "
                f"loss {fmt_cell(lat.get('train_loss') or lat.get('loss'))}  "
                f"acc {fmt_cell(lat.get('train_acc') or lat.get('acc'))}  "
                f"time {fmt_cell(str(int(lat['elapsed_ms'])) + 'ms' if lat.get('elapsed_ms') is not None else None)}"
            )

        return lines

    def _draw(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_draw < 0.05:
            return  # ~20fps cap
        self._last_draw = now
        frame = self._frame()
        out = sys.stderr
        if not self._started:
            out.write(_HIDE)
            self._started = True
        elif self._lines > 0:
            # move cursor up and clear below
            out.write(f"\033[{self._lines}A\033[J")
        text = "\n".join(frame) + "\n"
        out.write(text)
        out.flush()
        self._lines = len(frame)


def _visible_len(s: str) -> int:
    n = 0
    i = 0
    while i < len(s):
        if s[i] == "\033":
            # skip CSI
            i += 1
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
            continue
        n += 1
        i += 1
    return n


_tui: TrainTui | None = None


def get() -> TrainTui | None:
    global _tui
    if not enabled():
        return None
    if _tui is None:
        _tui = TrainTui()
    return _tui


def reset() -> None:
    global _tui
    if _tui is not None:
        _tui._finish()
    _tui = None
