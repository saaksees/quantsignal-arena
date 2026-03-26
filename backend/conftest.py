import pytest
import pandas as pd
import numpy as np


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m not slow')"
    )


@pytest.fixture(scope="session")
def sample_ohlcv_session():
    dates = pd.date_range('2020-01-01', periods=252, freq='D', tz='UTC')
    np.random.seed(42)
    prices = 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()
    return pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': 1000000,
        'Adj Close': prices
    }, index=dates)
