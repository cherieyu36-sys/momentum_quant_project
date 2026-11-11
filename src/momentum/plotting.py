"""Charts. Uses the object-oriented matplotlib API, so no GUI backend is required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter

from .metrics import TRADING_DAYS, drawdown_series


def plot_performance(
    strategy: pd.Series,
    benchmark: pd.Series | None,
    path: str | Path,
    title: str = "Momentum Strategy",
) -> None:
    """Equity curve (log scale) + drawdown + rolling 12-month Sharpe."""
    fig = Figure(figsize=(11, 9))
    axes = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    series = {"Strategy": strategy}
    if benchmark is not None:
        series["Benchmark"] = benchmark
    colors = {"Strategy": "#1f77b4", "Benchmark": "#7f7f7f"}

    for name, r in series.items():
        axes[0].plot((1 + r).cumprod(), label=name, color=colors[name], lw=1.4)
        axes[1].fill_between(r.index, drawdown_series(r), 0, color=colors[name], alpha=0.35, label=name)
        roll = r.rolling(TRADING_DAYS)
        axes[2].plot(roll.mean() / roll.std() * np.sqrt(TRADING_DAYS), color=colors[name], lw=1.0)

    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Equity (log)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left")
    axes[1].set_ylabel("Drawdown")
    axes[2].set_ylabel("Rolling 1Y Sharpe")
    axes[2].axhline(0, color="black", lw=0.6)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_weights(weights: pd.DataFrame, path: str | Path, title: str = "Portfolio Weights") -> None:
    """Month-end holdings: stacked area if long-only, lines if there are shorts."""
    monthly = weights.resample("ME").last()
    monthly = monthly.loc[:, monthly.abs().sum() > 0]
    fig = Figure(figsize=(11, 4.5))
    ax = fig.subplots()
    if (monthly >= -1e-12).all().all():
        ax.stackplot(monthly.index, monthly.T.to_numpy(), labels=monthly.columns, alpha=0.85)
    else:
        for col in monthly.columns:
            ax.plot(monthly.index, monthly[col], label=col, lw=1.0)
        ax.axhline(0, color="black", lw=0.6)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_heatmap(grid: pd.DataFrame, path: str | Path, title: str) -> None:
    """Parameter sweep heatmap."""
    fig = Figure(figsize=(1.3 * grid.shape[1] + 3, 0.8 * grid.shape[0] + 2))
    ax = fig.subplots()
    im = ax.imshow(grid.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [str(c) for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [str(i) for i in grid.index])
    ax.set_xlabel(grid.columns.name or "")
    ax.set_ylabel(grid.index.name or "")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            ax.text(j, i, "-" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
