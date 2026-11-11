from pathlib import Path

import pytest

from momentum.cli import main
from momentum.config import load_config
from momentum.data import generate_synthetic_prices

CONFIGS = sorted(Path(__file__).resolve().parents[1].joinpath("configs").glob("*.yaml"))


def test_synthetic_data_is_reproducible():
    a = generate_synthetic_prices(["X", "Y"], "2020-01-01", "2020-12-31", seed=1)
    b = generate_synthetic_prices(["X", "Y"], "2020-01-01", "2020-12-31", seed=1)
    assert a.equals(b) and (a > 0).all().all()


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_configs_load(config):
    load_config(config)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_end_to_end_run(config, tmp_path):
    main(["run", "--config", str(config), "--source", "synthetic",
          "--output-dir", str(tmp_path), "--name", "t"])
    out = tmp_path / "t"
    for f in ["summary.csv", "returns.csv", "weights.csv", "performance.png", "weights.png", "config.yaml"]:
        assert (out / f).exists(), f
