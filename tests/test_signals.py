import numpy as np
import pandas as pd
import pytest

from momentum.signals import momentum_score, realized_volatility


def test_momentum_score_matches_formula():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"X": np.arange(1.0, 301.0)}, index=idx)
    score = momentum_score(prices, lookback=252, skip=21)
    # At position 299: P[278] / P[47] - 1, where price = position + 1
    assert score["X"].iloc[299] == pytest.approx(279.0 / 48.0 - 1.0)
    assert score["X"].iloc[:252].isna().all()


@pytest.mark.parametrize("lookback, skip", [(21, 21), (21, 30), (10, -1)])
def test_invalid_window_raises(lookback, skip):
    prices = pd.DataFrame({"X": [1.0, 2.0]})
    with pytest.raises(ValueError):
        momentum_score(prices, lookback, skip)


def test_realized_vol_is_positive(prices):
    vol = realized_volatility(prices, window=63).dropna()
    assert (vol > 0).all().all()
