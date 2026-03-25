import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtester.signal_base import SignalBase, MomentumSignal

@pytest.fixture
def sample_ohlcv_data():
    dates = pd.date_range(start='2020-01-01', end='2020-03-31', freq='D', tz='UTC')
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * (1 + returns).cumprod()
    data = pd.DataFrame({
        'Open': prices * np.random.uniform(0.98, 1.02, len(dates)),
        'High': prices * np.random.uniform(1.00, 1.05, len(dates)),
        'Low': prices * np.random.uniform(0.95, 1.00, len(dates)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'Adj Close': prices
    }, index=dates)
    return data

class TestSignalBaseAbstract:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            SignalBase()

class TestMomentumSignal:
    def test_momentum_signal_returns_valid_values(self, sample_ohlcv_data):
        signal = MomentumSignal(lookback_period=20)
        result = signal(sample_ohlcv_data)
        valid_values = {-1, 0, 1}
        unique_values = set(result.unique())
        assert unique_values.issubset(valid_values)
    
    def test_output_length_matches_input(self, sample_ohlcv_data):
        signal = MomentumSignal(lookback_period=20)
        result = signal(sample_ohlcv_data)
        assert len(result) == len(sample_ohlcv_data)
    
    def test_output_has_datetime_index(self, sample_ohlcv_data):
        signal = MomentumSignal(lookback_period=20)
        result = signal(sample_ohlcv_data)
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_name_property_returns_class_name(self):
        signal = MomentumSignal(lookback_period=20)
        assert signal.name == 'MomentumSignal'
    
    def test_parameters_property_returns_dict(self):
        signal = MomentumSignal(lookback_period=30)
        params = signal.parameters
        assert isinstance(params, dict)
        assert 'lookback_period' in params
        assert params['lookback_period'] == 30

class TestSignalValidation:
    def test_invalid_signal_values_raise_error(self, sample_ohlcv_data):
        class InvalidSignal(SignalBase):
            def generate_signals(self, ohlcv_data):
                return pd.Series([2] * len(ohlcv_data), index=ohlcv_data.index)
        signal = InvalidSignal()
        with pytest.raises(ValueError, match='invalid values'):
            signal(sample_ohlcv_data)
    
    def test_nan_values_raise_error(self, sample_ohlcv_data):
        class NaNSignal(SignalBase):
            def generate_signals(self, ohlcv_data):
                signals = pd.Series([1] * len(ohlcv_data), index=ohlcv_data.index)
                signals.iloc[10] = np.nan
                return signals
        signal = NaNSignal()
        with pytest.raises(ValueError, match='NaN'):
            signal(sample_ohlcv_data)

class TestParameterValidation:
    def test_invalid_lookback_period_type(self):
        with pytest.raises(ValueError, match='must be an integer'):
            MomentumSignal(lookback_period=20.5)
    
    def test_negative_lookback_period(self):
        with pytest.raises(ValueError, match='must be positive'):
            MomentumSignal(lookback_period=-10)
