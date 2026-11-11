"""Convert momentum signals into target weights on rebalance dates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .signals import momentum_score, realized_volatility


def get_rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Last trading day of each period (day / week / month / quarter)."""
    freq = freq.upper()
    if freq not in {"D", "W", "M", "Q"}:
        raise ValueError(f"Unsupported rebalance frequency: {freq}")
    index = pd.DatetimeIndex(index)
    if freq == "D":
        return index
    last = pd.Series(index, index=index).groupby(index.to_period(freq)).last()
    return pd.DatetimeIndex(last.to_numpy())


def _allocate(names: Iterable[str], vol: pd.Series | None, weighting: str) -> pd.Series:
    """Allocate one unit of capital across the given assets (equal or inverse-volatility weighted)."""
    names = list(names)
    if not names:
        return pd.Series(dtype=float)
    if weighting == "inverse_vol" and vol is not None:
        v = vol.reindex(names)
        if v.notna().all() and (v > 0).all():
            inv = 1.0 / v
            return inv / inv.sum()
    return pd.Series(1.0 / len(names), index=names)


def cross_sectional_weights(
    score: pd.Series,
    vol: pd.Series | None,
    top_n: int,
    long_short: bool = False,
    weighting: str = "equal",
) -> pd.Series:
    """Cross-sectional momentum: long the top_n strongest assets (short the top_n weakest if long-short)."""
    valid = score.dropna()
    n = min(top_n, len(valid) // 2 if long_short else len(valid))
    if n == 0:
        return pd.Series(dtype=float)
    ranked = valid.sort_values(ascending=False, kind="mergesort")
    weights = _allocate(ranked.index[:n], vol, weighting)
    if long_short:
        weights = weights.add(-_allocate(ranked.index[-n:], vol, weighting), fill_value=0.0)
    return weights


def time_series_weights(
    score: pd.Series,
    vol: pd.Series | None,
    long_short: bool = False,
    weighting: str = "equal",
    target_vol: float = 0.10,
) -> pd.Series:
    """Time-series momentum: each asset only looks at the sign of its own past return.

    Long-only mode moves negative-momentum assets to cash; long-short mode shorts them.
    With weighting=inverse_vol, positions are scaled by target_vol / sigma_i
    (Moskowitz, Ooi & Pedersen, 2012).
    """
    valid = score.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    direction = np.sign(valid) if long_short else (valid > 0).astype(float)
    n = len(valid)
    if weighting == "inverse_vol" and vol is not None:
        v = vol.reindex(valid.index)
        size = (target_vol / v).replace([np.inf, -np.inf], np.nan).fillna(0.0) / n
    else:
        size = pd.Series(1.0 / n, index=valid.index)
    return direction * size


def dual_momentum_weights(
    score: pd.Series,
    vol: pd.Series | None,
    top_n: int,
    weighting: str = "equal",
    safe_asset: str | None = None,
    threshold: float = 0.0,
) -> pd.Series:
    """Dual momentum: pick the top_n by relative momentum, then filter by absolute momentum (> threshold).

    Each "slot" gets 1/top_n of capital; unfilled slots go to safe_asset (or cash if not set).
    """
    valid = score.dropna()
    winners = valid[valid > threshold].sort_values(ascending=False, kind="mergesort").index[:top_n]
    slot = 1.0 / top_n
    weights = _allocate(winners, vol, weighting) * (len(winners) * slot)
    leftover = 1.0 - len(winners) * slot
    if safe_asset is not None and safe_asset in valid.index and leftover > 1e-12:
        weights = weights.add(pd.Series({safe_asset: leftover}), fill_value=0.0)
    return weights


def apply_leverage_cap(weights: pd.DataFrame, max_leverage: float) -> pd.DataFrame:
    """If a row's gross weight exceeds max_leverage, scale it down proportionally."""
    gross = weights.abs().sum(axis=1)
    scale = np.where(gross > max_leverage, max_leverage / gross.replace(0.0, np.nan), 1.0)
    return weights.mul(scale, axis=0)


def build_target_weights(prices: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """On each rebalance date (after the close), compute target weights from data available at that time."""
    scores = momentum_score(prices, cfg.lookback, cfg.skip)
    vol = realized_volatility(prices, cfg.vol_window)
    dates = get_rebalance_dates(prices.index, cfg.rebalance)

    rows = []
    for d in dates:
        s, v = scores.loc[d], vol.loc[d]
        if cfg.name == "cross_sectional":
            w = cross_sectional_weights(s, v, cfg.top_n, cfg.long_short, cfg.weighting)
        elif cfg.name == "time_series":
            w = time_series_weights(s, v, cfg.long_short, cfg.weighting, cfg.target_vol)
        elif cfg.name == "dual":
            w = dual_momentum_weights(s, v, cfg.top_n, cfg.weighting, cfg.safe_asset)
        else:  # pragma: no cover - already validated in config
            raise ValueError(f"Unknown strategy: {cfg.name}")
        rows.append(w.reindex(prices.columns).fillna(0.0).to_numpy())

    weights = pd.DataFrame(rows, index=dates, columns=prices.columns, dtype=float)
    return apply_leverage_cap(weights, cfg.max_leverage)
