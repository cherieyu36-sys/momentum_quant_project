# 📈 Momentum Quant

[![CI](https://github.com/<your-username>/momentum-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/momentum-quant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A clean, reproducible, **look-ahead-free** momentum strategy backtesting framework. It implements the three most widely studied momentum strategies in quantitative finance: cross-sectional momentum, time-series momentum (trend following), and dual momentum. It also includes a daily backtesting engine, performance analytics, parameter sweeps, and a full unit test suite.

## ✨ Features

- **Three momentum strategies**: Cross-Sectional, Time-Series, and Dual Momentum
- **Rigorous backtesting engine**: signals at day t's close, returns from t+1; weights drift naturally with prices between rebalances; transaction costs charged on turnover
- **Flexible position sizing**: equal or inverse-volatility weighting, long-short, target volatility, leverage cap, safe-haven asset
- **Comprehensive metrics**: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, turnover, and more
- **Parameter sweeps**: lookback × top_n heatmaps to test robustness
- **Multiple data sources**: Yahoo Finance (with local caching), local CSV, synthetic data (runs offline)
- **Engineering**: YAML configs, CLI, pytest suite (including a look-ahead bias test), GitHub Actions CI

## 📁 Project Structure

```
momentum-quant/
├── configs/                  # Strategy configs
│   ├── dual_momentum.yaml    # Dual momentum: cross-asset ETF rotation
│   ├── cross_sectional.yaml  # Cross-sectional momentum: US sector ETFs
│   └── time_series.yaml      # Time-series momentum: vol-weighted trend following
├── src/momentum/
│   ├── config.py             # Config loading and validation
│   ├── data.py               # Data loading (Yahoo / CSV / synthetic)
│   ├── signals.py            # Momentum signals, volatility
│   ├── portfolio.py          # Signals -> target weights
│   ├── backtest.py           # Daily backtest engine, benchmarks
│   ├── metrics.py            # Performance metrics
│   ├── plotting.py           # Charts
│   └── cli.py                # Command-line interface
├── tests/                    # Unit and end-to-end tests
├── .github/workflows/ci.yml  # Continuous integration
└── pyproject.toml
```

## 🚀 Quick Start

```bash
git clone https://github.com/<your-username>/momentum-quant.git
cd momentum-quant

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Run dual momentum on real Yahoo Finance data
momentum run --config configs/dual_momentum.yaml

# Run offline on synthetic data (pipeline demo / no network)
momentum run --config configs/dual_momentum.yaml --source synthetic

# Parameter sweep: Sharpe heatmap over lookback × top_n
momentum sweep --config configs/cross_sectional.yaml --lookbacks 63 126 189 252 --top-ns 1 2 3 4 5
```

You can also use `python -m momentum run ...` instead of `momentum run ...`.

Each run writes the following to `reports/<strategy>_<source>/`:

| File | Contents |
|---|---|
| `summary.csv` | Performance metrics for strategy and benchmark |
| `returns.csv` | Daily returns (gross, costs, net, benchmark) |
| `weights.csv` | Actual daily holdings |
| `performance.png` | Equity curve, drawdown, rolling Sharpe |
| `weights.png` | Holdings over time |
| `config.yaml` | Full snapshot of the config used (for reproducibility) |

## 🧠 Strategies

All strategies share the same momentum signal:

$$\text{MOM}_t = \frac{P_{t-\text{skip}}}{P_{t-\text{lookback}}} - 1$$

The defaults `lookback=252, skip=21` give the classic **12-1 momentum**: the past 12-month return, excluding the most recent month to avoid the short-term reversal effect.

### 1. Cross-Sectional Momentum (`cross_sectional`)
On each rebalance date, assets are ranked by momentum and the strongest `top_n` are held long. With `long_short: true`, the weakest `top_n` are also shorted (100% long and 100% short, then subject to `max_leverage`). This is a **relative** strength strategy: even if every asset is falling, it holds the ones falling the least.

### 2. Time-Series Momentum (`time_series`)
Each asset is compared only with **its own past**: long if momentum is positive, flat (or short with `long_short: true`) if negative. With `inverse_vol` weighting, each position is `target_vol / σᵢ / N`, so high- and low-volatility assets contribute similar risk. Strategies of this kind typically cut drawdowns substantially in major bear markets.

### 3. Dual Momentum (`dual`)
Combines the two: first pick the top `top_n` by relative momentum, then require positive absolute momentum. Each slot gets `1/top_n` of capital; unfilled slots go to `safe_asset` (e.g. intermediate Treasuries, IEF), or to cash if none is set.

## ⚙️ Backtest Engine Design

These are the most common sources of error in momentum backtests, and each one is handled explicitly:

1. **No look-ahead bias**: target weights are computed after day t's close using data up to day t, and new weights earn returns from day t+1. `tests/test_portfolio.py::test_no_lookahead` tampers with prices after a given date and verifies that all weights on or before that date are unchanged.
2. **Weight drift**: there is no silent daily rebalancing between rebalance dates. Holdings drift with prices, and turnover at each rebalance is measured against the drifted weights.
3. **Transaction costs**: cost = turnover × `cost_bps`, deducted on rebalance days.
4. **Data cleaning**: only limited forward-filling; never back-filling, which would leak future prices into the past.
5. **Warm-up trimming**: statistics start at the first rebalance with an actual position, so zero returns during the signal warm-up don't dilute the metrics.

## 🔧 Configuration

| Parameter | Description | Default |
|---|---|---|
| `data.tickers` | Asset universe | — |
| `data.source` | `yahoo` / `csv` / `synthetic` | `yahoo` |
| `data.csv_path` | CSV path (dates as index, tickers as columns) | `null` |
| `strategy.name` | `cross_sectional` / `time_series` / `dual` | `dual` |
| `strategy.lookback` | Momentum lookback window (trading days) | `252` |
| `strategy.skip` | Skip the most recent N trading days | `21` |
| `strategy.rebalance` | Rebalance frequency `D` / `W` / `M` / `Q` | `M` |
| `strategy.top_n` | Number of assets held (cross-sectional / dual) | `3` |
| `strategy.long_short` | Allow short positions | `false` |
| `strategy.weighting` | `equal` / `inverse_vol` | `equal` |
| `strategy.vol_window` | Volatility estimation window | `63` |
| `strategy.target_vol` | Per-asset target volatility (time-series) | `0.10` |
| `strategy.max_leverage` | Gross leverage cap | `1.0` |
| `strategy.safe_asset` | Safe-haven asset (dual momentum) | `null` |
| `backtest.cost_bps` | Transaction cost (basis points) | `10` |
| `backtest.benchmark` | `equal_weight` / a ticker / `null` | `equal_weight` |

## 🧪 Testing

```bash
pytest -q        # unit + end-to-end tests (synthetic data, no network needed)
ruff check .     # linting
```

## ⚠️ Known Limitations

- **Survivorship bias**: the universe consists of ETFs that still exist today, so backtests are inherently optimistic.
- **Zero cash return**: no interest on cash and no borrow cost for shorts.
- **Simple cost model**: linear transaction costs only; no market impact or time-varying bid-ask spreads.
- **Data quality**: Yahoo Finance is an unofficial source; use a paid data vendor for serious research.
- **Synthetic data is for demos only**: synthetic prices have built-in trend persistence, so strategy performance on them says **nothing** about real markets.

## 🛣️ Possible Extensions

- Portfolio-level volatility targeting
- Momentum crash protection (Daniel & Moskowitz, 2016)
- Walk-forward / out-of-sample testing
- Single-stock cross-sectional momentum (requires handling delistings and survivorship bias)
- Cash yield, borrow costs, and slippage models

## 📚 References

- Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). *Time Series Momentum.* Journal of Financial Economics.
- Asness, C., Moskowitz, T., & Pedersen, L. H. (2013). *Value and Momentum Everywhere.* Journal of Finance.
- Antonacci, G. (2014). *Dual Momentum Investing.* McGraw-Hill.
- Daniel, K., & Moskowitz, T. (2016). *Momentum Crashes.* Journal of Financial Economics.

## ⚖️ Disclaimer

This project is for educational and research purposes only and does not constitute investment advice. Past backtest results do not guarantee future performance. Any live trading based on this code is entirely at your own risk.

## 📄 License

[MIT](LICENSE)
