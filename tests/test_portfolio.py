import pandas as pd
import pytest

from momentum.config import StrategyConfig
from momentum.portfolio import (
    apply_leverage_cap,
    build_target_weights,
    cross_sectional_weights,
    dual_momentum_weights,
    get_rebalance_dates,
    time_series_weights,
)

SCORE = pd.Series({"A": 0.30, "B": 0.10, "C": -0.05, "D": -0.20, "E": float("nan")})


def test_rebalance_dates_are_month_ends():
    idx = pd.bdate_range("2021-01-01", "2021-04-30")
    dates = get_rebalance_dates(idx, "M")
    assert list(dates.strftime("%Y-%m-%d")) == ["2021-01-29", "2021-02-26", "2021-03-31", "2021-04-30"]


def test_cross_sectional_long_only():
    w = cross_sectional_weights(SCORE, None, top_n=2)
    assert w.to_dict() == pytest.approx({"A": 0.5, "B": 0.5})


def test_cross_sectional_long_short_is_dollar_neutral():
    w = cross_sectional_weights(SCORE, None, top_n=1, long_short=True)
    assert w["A"] == pytest.approx(1.0) and w["D"] == pytest.approx(-1.0)
    assert w.sum() == pytest.approx(0.0)


def test_time_series_long_only_goes_to_cash_on_negative_momentum():
    w = time_series_weights(SCORE, None, long_short=False)
    assert w["A"] == pytest.approx(0.25) and w["C"] == 0.0 and w["D"] == 0.0


def test_dual_momentum_fills_empty_slots_with_safe_asset():
    w = dual_momentum_weights(SCORE, None, top_n=4, safe_asset="C")
    # Only A and B have positive momentum: 1/4 each, the remaining 1/2 goes to C
    assert w.to_dict() == pytest.approx({"A": 0.25, "B": 0.25, "C": 0.5})
    assert w.sum() == pytest.approx(1.0)


def test_inverse_vol_weighting():
    vol = pd.Series({"A": 0.10, "B": 0.30})
    w = cross_sectional_weights(SCORE, vol, top_n=2, weighting="inverse_vol")
    assert w["A"] == pytest.approx(0.75) and w["B"] == pytest.approx(0.25)


def test_leverage_cap():
    w = pd.DataFrame({"A": [1.5, 0.5], "B": [-0.5, 0.2]})
    capped = apply_leverage_cap(w, max_leverage=1.0)
    assert capped.abs().sum(axis=1).tolist() == pytest.approx([1.0, 0.7])


@pytest.mark.parametrize("name", ["cross_sectional", "time_series", "dual"])
def test_no_lookahead(prices, name):
    """Tampering with prices after a date must not change any target weight on or before that date."""
    cfg = StrategyConfig(name=name, lookback=126, skip=21, top_n=2, weighting="inverse_vol")
    base = build_target_weights(prices, cfg)

    cutoff = prices.index[len(prices) // 2]
    tampered = prices.copy()
    tampered.loc[tampered.index > cutoff] *= 3.0
    tampered.loc[tampered.index > cutoff, "A"] *= 0.1
    changed = build_target_weights(tampered, cfg)

    pd.testing.assert_frame_equal(base.loc[:cutoff], changed.loc[:cutoff])
    assert not base.loc[cutoff:].iloc[1:].equals(changed.loc[cutoff:].iloc[1:])
