"""Momentum Quant: a clean, reproducible, look-ahead-free momentum strategy backtesting framework."""

from .backtest import BacktestResult, run_backtest
from .config import Config, load_config
from .data import generate_synthetic_prices, load_prices
from .metrics import summarize
from .portfolio import build_target_weights
from .signals import momentum_score, realized_volatility

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "Config",
    "build_target_weights",
    "generate_synthetic_prices",
    "load_config",
    "load_prices",
    "momentum_score",
    "realized_volatility",
    "run_backtest",
    "summarize",
]
