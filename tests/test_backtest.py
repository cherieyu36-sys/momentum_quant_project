import numpy as np
import pandas as pd
import pytest

from momentum.backtest import equal_weight_benchmark, run_backtest


def _prices():
    idx = pd.bdate_range("2022-01-03", periods=6)
    return pd.DataFrame({"A": [100, 110, 99, 108.9, 108.9, 119.79],
                         "B": [50, 50, 55, 55, 49.5, 49.5]}, index=idx, dtype=float)


def test_weights_apply_from_next_day():
    prices = _prices()
    tw = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=prices.index[:1])
    res = run_backtest(prices, tw, cost_bps=0.0)
    expected = prices["A"].pct_change().fillna(0.0)
    expected.iloc[0] = 0.0  # position is opened at day 0's close, so it earns nothing that day
    np.testing.assert_allclose(res.returns.to_numpy(), expected.to_numpy())
    assert res.equity.iloc[-1] == pytest.approx(prices["A"].iloc[-1] / prices["A"].iloc[0])


def test_transaction_costs():
    prices = _prices()
    tw = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=prices.index[:1])
    res = run_backtest(prices, tw, cost_bps=10.0)
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.returns.iloc[0] == pytest.approx(-0.001)


def test_weights_drift_like_buy_and_hold():
    prices = _prices()
    tw = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=prices.index[:1])
    res = run_backtest(prices, tw, cost_bps=0.0)
    # Without rebalancing, portfolio equity equals the average of the two assets' buy-and-hold
    bh = 0.5 * prices["A"] / prices["A"].iloc[0] + 0.5 * prices["B"] / prices["B"].iloc[0]
    np.testing.assert_allclose(res.equity.to_numpy(), bh.to_numpy())
    assert res.weights.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(prices)))


def test_unknown_rebalance_date_raises():
    prices = _prices()
    tw = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("1999-01-01")])
    with pytest.raises(ValueError):
        run_backtest(prices, tw)


def test_equal_weight_benchmark_runs(prices):
    bench = equal_weight_benchmark(prices, "M")
    assert bench.notna().all() and len(bench) > 100
