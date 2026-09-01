"""Matplotlib helpers (Agg backend, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fafafa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "lines.linewidth": 1.8,
        }
    )


def save_fig(path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
