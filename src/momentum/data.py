"""Data loading: Yahoo Finance / local CSV / synthetic data."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
    source: str = "yahoo",
    csv_path: str | None = None,
    cache_dir: str = "data",
    seed: int = 42,
) -> pd.DataFrame:
    """Return adjusted close prices as a DataFrame (trading days as index, tickers as columns)."""
    tickers = list(tickers)
    source = source.lower()
    if source == "synthetic":
        return generate_synthetic_prices(tickers, start, end or "2024-12-31", seed=seed)
    if source == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required when source=csv")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        missing = set(tickers) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing these tickers: {sorted(missing)}")
        return _clean(df.loc[start:end, tickers])
    if source == "yahoo":
        return _load_yahoo(tickers, start, end, cache_dir)
    raise ValueError(f"Unknown data source: {source}")


def _load_yahoo(tickers: list[str], start: str, end: str | None, cache_dir: str) -> pd.DataFrame:
    key = hashlib.md5(f"{','.join(tickers)}|{start}|{end}".encode()).hexdigest()[:12]
    cache_file = Path(cache_dir) / f"prices_{key}.csv"
    if cache_file.exists():
        return _clean(pd.read_csv(cache_file, index_col=0, parse_dates=True))

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required: pip install yfinance") from exc

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data; check your network connection and tickers")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers)

    empty = [t for t in tickers if close[t].isna().all()]
    if empty:
        warnings.warn(f"No data at all for these tickers: {empty}", stacklevel=2)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(cache_file)
    return _clean(close)


def _clean(prices: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop all-empty rows, and forward-fill short gaps (never back-fill, which leaks future data)."""
    prices = prices.sort_index()
    idx = pd.DatetimeIndex(prices.index)
    prices.index = idx.tz_localize(None) if idx.tz is not None else idx
    prices = prices.dropna(how="all")
    return prices.ffill(limit=5).astype(float)


def generate_synthetic_prices(
    tickers: Sequence[str],
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic prices with persistent drift (trends) for offline demos and unit tests.

    Daily return per asset = base drift + AR(1) persistent drift + common market factor + idiosyncratic noise.
    The AR(1) drift makes trends persistent, so momentum signals work somewhat on synthetic data.
    This only demonstrates the pipeline and says nothing about real-market performance.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, end)
    n, k = len(index), len(tickers)

    phi, shock_sd = 0.998, 4e-5
    drift = np.zeros((n, k))
    shocks = rng.normal(0.0, shock_sd, size=(n, k))
    for t in range(1, n):
        drift[t] = phi * drift[t - 1] + shocks[t]

    idio_vol = rng.uniform(0.006, 0.016, size=k)
    market = rng.normal(0.0, 0.006, size=(n, 1))
    log_rets = 1.5e-4 + drift + market + rng.normal(0.0, 1.0, size=(n, k)) * idio_vol
    log_rets[0] = 0.0

    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    return pd.DataFrame(prices, index=index, columns=list(tickers))
