"""Daily backtesting engine.

Timing convention (avoids look-ahead bias):
  * After the close on day t, target weights are computed from data up to day t and the
    portfolio is rebalanced at day t's close; new weights earn returns from day t+1 onward;
  * between rebalances, weights drift with prices instead of being silently reset to target every day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import get_rebalance_dates


@dataclass
class BacktestResult:
    returns: pd.Series        # daily returns, net of costs
    gross_returns: pd.Series  # daily returns, before costs
    costs: pd.Series          # daily transaction costs (in return terms)
    turnover: pd.Series       # daily turnover (sum of absolute weight changes)
    weights: pd.DataFrame     # actual end-of-day holdings

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    def slice(self, start) -> BacktestResult:
        return BacktestResult(
            returns=self.returns.loc[start:],
            gross_returns=self.gross_returns.loc[start:],
            costs=self.costs.loc[start:],
            turnover=self.turnover.loc[start:],
            weights=self.weights.loc[start:],
        )


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 10.0,
    trim_start: bool = True,
) -> BacktestResult:
    """Backtest a set of target weights defined on rebalance dates.

    Args:
        prices: Adjusted close prices, indexed by trading day.
        target_weights: Target weights on rebalance dates (index must be a subset of prices.index).
            Any unallocated weight is held as cash (zero return); negative weights are shorts.
        cost_bps: Transaction cost in basis points, charged as "turnover x cost".
        trim_start: Start the results at the first rebalance with a position (drops signal warm-up).
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    prices = prices.sort_index()
    tw = target_weights.reindex(columns=prices.columns).fillna(0.0).sort_index()
    missing = tw.index.difference(prices.index)
    if len(missing) > 0:
        raise ValueError(f"Target weight dates not found in price data: {list(missing[:3])}")

    r = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    n_days, n_assets = r.shape
    targets = dict(zip(prices.index.get_indexer(tw.index), tw.to_numpy(), strict=True))
    cost_rate = cost_bps / 1e4

    w = np.zeros(n_assets)
    gross = np.zeros(n_days)
    costs = np.zeros(n_days)
    turnover = np.zeros(n_days)
    held = np.zeros((n_days, n_assets))

    for t in range(n_days):
        # 1) hold yesterday's end-of-day weights and earn today's returns
        port_ret = float(w @ r[t])
        gross[t] = port_ret
        # 2) weights drift with prices (cash earns zero)
        denom = 1.0 + port_ret
        w = w * (1.0 + r[t]) / denom if denom > 0 else np.zeros(n_assets)
        # 3) on rebalance days, trade to target at the close and pay transaction costs
        if t in targets:
            target = targets[t]
            trade = float(np.abs(target - w).sum())
            turnover[t] = trade
            costs[t] = trade * cost_rate * (1.0 + port_ret)
            w = target.copy()
        held[t] = w

    idx = prices.index
    result = BacktestResult(
        returns=pd.Series(gross - costs, index=idx, name="strategy"),
        gross_returns=pd.Series(gross, index=idx, name="gross"),
        costs=pd.Series(costs, index=idx, name="costs"),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
        weights=pd.DataFrame(held, index=idx, columns=prices.columns),
    )
    if trim_start:
        active = tw.index[tw.abs().sum(axis=1) > 0]
        if len(active) > 0:
            result = result.slice(active[0])
    return result


def equal_weight_benchmark(prices: pd.DataFrame, rebalance: str = "M") -> pd.Series:
    """Equal-weight benchmark across all listed assets, rebalanced at the same frequency, no costs."""
    dates = get_rebalance_dates(prices.index, rebalance)
    available = prices.loc[dates].notna().astype(float)
    tw = available.div(available.sum(axis=1), axis=0).fillna(0.0)
    return run_backtest(prices, tw, cost_bps=0.0).returns.rename("benchmark")


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Single-asset buy and hold."""
    return prices.pct_change(fill_method=None).fillna(0.0).rename("benchmark")
