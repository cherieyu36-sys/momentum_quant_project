import pandas as pd
import pytest

from momentum.data import generate_synthetic_prices


@pytest.fixture
def prices() -> pd.DataFrame:
    return generate_synthetic_prices(["A", "B", "C", "D", "E", "F"], "2015-01-01", "2020-12-31", seed=7)
