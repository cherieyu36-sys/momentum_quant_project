"""Command-line interface:

    momentum run   --config configs/dual_momentum.yaml
    momentum sweep --config configs/cross_sectional.yaml --lookbacks 63 126 252 --top-ns 1 2 3
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from .backtest import BacktestResult, buy_and_hold, equal_weight_benchmark, run_backtest
from .config import Config, load_config
from .data import load_prices
from .metrics import format_summary, summarize
from .plotting import plot_heatmap, plot_performance, plot_weights
from .portfolio import build_target_weights

DEFAULT_CONFIG = "configs/dual_momentum.yaml"


def _load_prices(cfg: Config, tickers: list[str] | None = None) -> pd.DataFrame:
    d = cfg.data
    return load_prices(
        tickers or d.tickers, d.start, d.end, d.source, d.csv_path, d.cache_dir, d.seed
    )


def _benchmark_returns(cfg: Config, prices: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series | None:
    name = cfg.backtest.benchmark
    if name is None:
        return None
    if name == "equal_weight":
        bench = equal_weight_benchmark(prices, cfg.strategy.rebalance)
    elif name in prices.columns:
        bench = buy_and_hold(prices[name])
    elif cfg.data.source == "synthetic":
        # A separately generated synthetic benchmark is unrelated to the strategy's assets
        print(f"Note: benchmark '{name}' is not in the universe under synthetic data; using equal weight.")
        bench = equal_weight_benchmark(prices, cfg.strategy.rebalance)
    else:
        bench = buy_and_hold(_load_prices(cfg, [name])[name])
    return bench.reindex(index).fillna(0.0)


def run_pipeline(cfg: Config, prices: pd.DataFrame | None = None):
    """Full pipeline: data -> signals/weights -> backtest -> benchmark -> metrics."""
    prices = _load_prices(cfg) if prices is None else prices
    weights = build_target_weights(prices, cfg.strategy)
    result = run_backtest(prices, weights, cfg.backtest.cost_bps)
    bench = _benchmark_returns(cfg, prices, result.returns.index)

    summary = pd.DataFrame({"Strategy": summarize(result.returns, result.turnover)})
    if bench is not None:
        summary["Benchmark"] = pd.Series(summarize(bench))
    return result, bench, summary


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.source:
        cfg = replace(cfg, data=replace(cfg.data, source=args.source))
    if args.output_dir:
        cfg = replace(cfg, output_dir=args.output_dir)
    return cfg


def _save_run(
    cfg: Config, result: BacktestResult, bench: pd.Series | None, summary: pd.DataFrame, out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv")
    returns = pd.DataFrame({"strategy": result.returns, "gross": result.gross_returns, "costs": result.costs})
    if bench is not None:
        returns["benchmark"] = bench
    returns.to_csv(out / "returns.csv")
    result.weights.to_csv(out / "weights.csv")
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    title = f"{cfg.strategy.name} momentum ({cfg.data.source} data)"
    plot_performance(result.returns, bench, out / "performance.png", title=title)
    plot_weights(result.weights, out / "weights.png")


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _apply_overrides(load_config(args.config), args)
    result, bench, summary = run_pipeline(cfg)
    out = Path(cfg.output_dir) / (args.name or f"{cfg.strategy.name}_{cfg.data.source}")
    _save_run(cfg, result, bench, summary, out)

    start, end = result.returns.index[0].date(), result.returns.index[-1].date()
    print(f"\nStrategy: {cfg.strategy.name} | Data: {cfg.data.source} | Period: {start} ~ {end}\n")
    print(format_summary(summary).to_string())
    print(f"\nResults saved to: {out.resolve()}")


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = _apply_overrides(load_config(args.config), args)
    if cfg.strategy.name == "time_series":
        print("Note: time_series strategy ignores top_n, so every column of the heatmap will be identical.")
    prices = _load_prices(cfg)
    metric = args.metric

    grid = pd.DataFrame(index=pd.Index(args.lookbacks, name="lookback"),
                        columns=pd.Index(args.top_ns, name="top_n"), dtype=float)
    for lb, n in itertools.product(args.lookbacks, args.top_ns):
        try:
            trial = replace(cfg, strategy=replace(cfg.strategy, lookback=lb, top_n=n))
        except ValueError as exc:
            print(f"Skipping lookback={lb}, top_n={n}: {exc}")
            continue
        _, _, summary = run_pipeline(trial, prices=prices)
        grid.loc[lb, n] = summary.loc[metric, "Strategy"]

    out = Path(cfg.output_dir) / (args.name or f"sweep_{cfg.strategy.name}_{cfg.data.source}")
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / f"sweep_{metric.replace(' ', '_').lower()}.csv")
    plot_heatmap(grid, out / "sweep_heatmap.png", title=f"{metric} by lookback x top_n")

    print(f"\nParameter sweep ({metric}):\n")
    print(grid.round(3).to_string())
    print("\nNote: picking only the best cell is classic overfitting. Robust strategies perform similarly "
          "across neighboring parameters.")
    print(f"Results saved to: {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="momentum", description="Momentum strategy backtesting tool")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=DEFAULT_CONFIG, help="path to YAML config file")
        p.add_argument("--source", choices=["yahoo", "csv", "synthetic"], help="override the data source")
        p.add_argument("--output-dir", help="override the output directory")
        p.add_argument("--name", help="output subdirectory name for this run")

    p_run = sub.add_parser("run", help="run a single backtest")
    common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", help="parameter sweep (lookback x top_n)")
    common(p_sweep)
    p_sweep.add_argument("--lookbacks", type=int, nargs="+", default=[63, 126, 189, 252])
    p_sweep.add_argument("--top-ns", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p_sweep.add_argument(
        "--metric", default="Sharpe", choices=["Sharpe", "CAGR", "Sortino", "Calmar", "Max Drawdown"]
    )
    p_sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
