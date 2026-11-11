import numpy as np
import pandas as pd
import pytest

from momentum.metrics import annualized_return, max_drawdown, sharpe_ratio, summarize


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_max_drawdown():
    assert max_drawdown(_series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_annualized_return_constant():
    r = _series([0.001] * 252)
    assert annualized_return(r) == pytest.approx(1.001 ** 252 - 1)


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(sharpe_ratio(_series([0.0] * 50)))


def test_summarize_keys(prices):
    r = prices["A"].pct_change().dropna()
    stats = summarize(r)
    for key in ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar"]:
        assert key in stats
