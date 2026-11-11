"""Performance metrics. All inputs are daily return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

PERCENT_METRICS = {
    "CAGR", "Volatility", "Max Drawdown", "Hit Rate (daily)",
    "Best Month", "Worst Month", "Total Return", "Annual Turnover",
}


def total_return(r: pd.Series) -> float:
    return float((1.0 + r.dropna()).prod() - 1.0)


def annualized_return(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS / len(r)) - 1.0


def annualized_volatility(r: pd.Series) -> float:
    return float(r.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(r: pd.Series, rf: float = 0.0) -> float:
    excess = r.dropna() - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(r: pd.Series, rf: float = 0.0) -> float:
    excess = r.dropna() - rf / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    if not np.isfinite(downside) or downside < 1e-12:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def drawdown_series(r: pd.Series) -> pd.Series:
    equity = (1.0 + r.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def calmar_ratio(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return float("nan") if mdd == 0 else annualized_return(r) / abs(mdd)


def monthly_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r).resample("ME").prod() - 1.0


def summarize(r: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    r = r.dropna()
    monthly = monthly_returns(r)
    active = r[r != 0]
    stats = {
        "Total Return": total_return(r),
        "CAGR": annualized_return(r),
        "Volatility": annualized_volatility(r),
        "Sharpe": sharpe_ratio(r),
        "Sortino": sortino_ratio(r),
        "Max Drawdown": max_drawdown(r),
        "Calmar": calmar_ratio(r),
        "Hit Rate (daily)": float((active > 0).mean()) if len(active) else float("nan"),
        "Best Month": float(monthly.max()),
        "Worst Month": float(monthly.min()),
    }
    if turnover is not None:
        years = len(r) / TRADING_DAYS
        stats["Annual Turnover"] = float(turnover.loc[r.index].sum() / years) if years else float("nan")
    return stats


def format_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Format the metrics table as readable strings (percentages / two decimals)."""
    out = table.astype(object).copy()
    for metric in table.index:
        for col in table.columns:
            v = table.loc[metric, col]
            if pd.isna(v):
                out.loc[metric, col] = "-"
            elif metric in PERCENT_METRICS:
                out.loc[metric, col] = f"{v:.2%}"
            else:
                out.loc[metric, col] = f"{v:.2f}"
    return out
