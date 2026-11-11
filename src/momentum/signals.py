"""Momentum signals and risk estimates. All calculations use only data up to and including the current day."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def momentum_score(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Classic "lookback-skip" momentum: MOM_t = P_{t-skip} / P_{t-lookback} - 1.

    The defaults give 12-1 momentum: the past 12-month return, skipping the latest month
    to avoid short-term reversal.
    """
    if not 0 <= skip < lookback:
        raise ValueError("requires 0 <= skip < lookback")
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def realized_volatility(
    prices: pd.DataFrame, window: int = 63, annualization: int = TRADING_DAYS
) -> pd.DataFrame:
    """Rolling annualized realized volatility."""
    rets = prices.pct_change(fill_method=None)
    return rets.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(annualization)
