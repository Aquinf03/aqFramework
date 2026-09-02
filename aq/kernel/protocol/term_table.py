"""Small ASCII tables for kernel stderr (train/eval live UI)."""

from __future__ import annotations

import sys
from typing import Any, Sequence


def fmt_cell(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        x = float(v)
        if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
            return f"{x:.4e}"
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s or "0"
    return str(v)


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    indent: str = "  ",
) -> str:
    """Plain aligned table. No box-drawing — stays copy/paste friendly."""
    cols = [list(headers)]
    data = [[fmt_cell(c) for c in row] for row in rows]
    if not data and not headers:
        return ""
    width = len(headers)
    grid = [list(headers)] + data
    widths = [0] * width
    for row in grid:
        for i in range(width):
            widths[i] = max(widths[i], len(row[i]) if i < len(row) else 0)

    def line(cells: Sequence[str]) -> str:
        parts = []
        for i, w in enumerate(widths):
            cell = cells[i] if i < len(cells) else ""
            # right-align numeric-looking cells in body
            parts.append(cell.rjust(w) if i > 0 and _looks_num(cell) else cell.ljust(w))
        return indent + "  ".join(parts)

    out = [line([str(h) for h in headers])]
    out.append(indent + "  ".join("─" * w for w in widths))
    for row in data:
        out.append(line(row))
    return "\n".join(out)


def _looks_num(s: str) -> bool:
    if not s or s == "—":
        return False
    try:
        float(s.replace("ms", "").replace("%", "").replace("e", "e"))
        return True
    except ValueError:
        return s.endswith("ms") or s.endswith("%")


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    indent: str = "  ",
    file=None,
) -> None:
    text = render_table(headers, rows, indent=indent)
    if text:
        print(text, file=file or sys.stderr, flush=True)


def print_kv(rows: Sequence[tuple[str, Any]], *, indent: str = "  ", file=None) -> None:
    """Two-column key/value table."""
    print_table(("key", "value"), [(k, v) for k, v in rows], indent=indent, file=file)
