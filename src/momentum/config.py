"""Config loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_STRATEGIES = {"cross_sectional", "time_series", "dual"}
VALID_WEIGHTING = {"equal", "inverse_vol"}
VALID_REBALANCE = {"D", "W", "M", "Q"}
VALID_SOURCES = {"yahoo", "csv", "synthetic"}


@dataclass
class DataConfig:
    tickers: list[str]
    start: str = "2007-01-01"
    end: str | None = None
    source: str = "yahoo"          # yahoo | csv | synthetic
    csv_path: str | None = None    # used when source=csv: dates as index, tickers as columns
    cache_dir: str = "data"
    seed: int = 42                 # random seed when source=synthetic

    def __post_init__(self) -> None:
        self.tickers = [str(t) for t in self.tickers]
        self.start = str(self.start)
        self.end = None if self.end is None else str(self.end)
        if not self.tickers:
            raise ValueError("data.tickers must not be empty")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("data.tickers contains duplicates")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"data.source must be one of {sorted(VALID_SOURCES)}")
        if self.source == "csv" and not self.csv_path:
            raise ValueError("data.csv_path is required when data.source=csv")


@dataclass
class StrategyConfig:
    name: str = "dual"             # cross_sectional | time_series | dual
    lookback: int = 252            # momentum lookback window (trading days)
    skip: int = 21                 # skip the most recent N days to avoid short-term reversal
    rebalance: str = "M"           # D | W | M | Q
    top_n: int = 3                 # number of assets held (cross-sectional / dual)
    long_short: bool = False       # allow short positions
    weighting: str = "equal"       # equal | inverse_vol
    vol_window: int = 63           # volatility estimation window
    target_vol: float = 0.10       # per-asset target volatility (time-series momentum)
    max_leverage: float = 1.0      # gross leverage cap (sum of absolute weights)
    safe_asset: str | None = None  # where unused dual-momentum slots go

    def __post_init__(self) -> None:
        self.rebalance = str(self.rebalance).upper()
        if self.name not in VALID_STRATEGIES:
            raise ValueError(f"strategy.name must be one of {sorted(VALID_STRATEGIES)}")
        if self.weighting not in VALID_WEIGHTING:
            raise ValueError(f"strategy.weighting must be one of {sorted(VALID_WEIGHTING)}")
        if self.rebalance not in VALID_REBALANCE:
            raise ValueError(f"strategy.rebalance must be one of {sorted(VALID_REBALANCE)}")
        if not 0 <= self.skip < self.lookback:
            raise ValueError("requires 0 <= skip < lookback")
        if self.top_n < 1:
            raise ValueError("strategy.top_n must be >= 1")
        if self.vol_window < 2:
            raise ValueError("strategy.vol_window must be >= 2")
        if self.target_vol <= 0 or self.max_leverage <= 0:
            raise ValueError("target_vol and max_leverage must be positive")


@dataclass
class BacktestConfig:
    cost_bps: float = 10.0                  # one-way transaction cost (bps), charged on turnover
    benchmark: str | None = "equal_weight"  # equal_weight | <ticker> | null

    def __post_init__(self) -> None:
        if self.cost_bps < 0:
            raise ValueError("backtest.cost_bps must be non-negative")


@dataclass
class Config:
    data: DataConfig
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output_dir: str = "reports"

    def __post_init__(self) -> None:
        s, tickers = self.strategy, self.data.tickers
        if s.safe_asset is not None and s.safe_asset not in tickers:
            raise ValueError(f"safe_asset '{s.safe_asset}' must be included in data.tickers")
        if s.name == "cross_sectional" and s.long_short and 2 * s.top_n > len(tickers):
            raise ValueError("long-short cross-sectional strategy requires 2 * top_n <= number of assets")
        if s.name in {"cross_sectional", "dual"} and s.top_n > len(tickers):
            raise ValueError("top_n must not exceed the number of assets")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    """Load config from YAML. Unknown keys raise an error so typos are never silently ignored."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "data" not in raw:
        raise ValueError("config file is missing the data section")
    return Config(
        data=DataConfig(**raw["data"]),
        strategy=StrategyConfig(**(raw.get("strategy") or {})),
        backtest=BacktestConfig(**(raw.get("backtest") or {})),
        output_dir=raw.get("output_dir", "reports"),
    )
