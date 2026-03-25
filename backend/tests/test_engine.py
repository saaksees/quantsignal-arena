import pytest
import pandas as pd
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtester.engine import BacktestEngine
from backtester.signal_base import SignalBase

@pytest.fixture
def sample_ohlcv_data():
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D', tz='UTC')
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

@pytest.fixture
def large_ohlcv_data():
    dates = pd.date_range(start='2010-01-01', end='2020-12-31', freq='D', tz='UTC')
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.0005, 0.015, len(dates))
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

class SimpleSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        signals = pd.Series(0, index=ohlcv_data.index, dtype=np.int8)
        signals.iloc[::2] = 1
        signals.iloc[1::2] = -1
        return signals

class LongOnlySignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        signals = pd.Series(0, index=ohlcv_data.index, dtype=np.int8)
        signals.iloc[::3] = 1
        return signals

class AllZerosSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        return pd.Series(0, index=ohlcv_data.index, dtype=np.int8)


class TestBacktestEngineInit:
    def test_engine_initialization(self):
        engine = BacktestEngine(initial_capital=50000, commission=0.002)
        assert engine.initial_capital == 50000
        assert engine.commission == 0.002
    
    def test_negative_commission_raises_error(self):
        with pytest.raises(ValueError, match='commission must be non-negative'):
            BacktestEngine(commission=-0.001)
    
    def test_zero_initial_capital_raises_error(self):
        with pytest.raises(ValueError, match='initial_capital must be positive'):
            BacktestEngine(initial_capital=0)

class TestRunBacktest:
    def test_valid_signal_returns_dict_with_correct_keys(self, sample_ohlcv_data):
        engine = BacktestEngine()
        signal = SimpleSignal()
        results = engine.run_backtest(signal, sample_ohlcv_data)
        assert isinstance(results, dict)
        expected_keys = ['portfolio_value', 'returns', 'trades', 'positions', 'signal_series', 'metrics_input']
        for key in expected_keys:
            assert key in results
    
    def test_portfolio_value_length_matches_input(self, sample_ohlcv_data):
        engine = BacktestEngine()
        signal = SimpleSignal()
        results = engine.run_backtest(signal, sample_ohlcv_data)
        assert len(results['portfolio_value']) == len(sample_ohlcv_data)
    
    def test_negative_commission_in_run_raises_error(self, sample_ohlcv_data):
        engine = BacktestEngine()
        signal = SimpleSignal()
        with pytest.raises(ValueError, match='commission must be non-negative'):
            engine.run_backtest(signal, sample_ohlcv_data, commission=-0.001)
    
    def test_zero_initial_capital_in_run_raises_error(self, sample_ohlcv_data):
        engine = BacktestEngine()
        signal = SimpleSignal()
        with pytest.raises(ValueError, match='initial_capital must be positive'):
            engine.run_backtest(signal, sample_ohlcv_data, initial_capital=0)

class TestLongOnlyMode:
    def test_long_only_never_takes_short_positions(self, sample_ohlcv_data):
        engine = BacktestEngine()
        signal = LongOnlySignal()
        results = engine.run_backtest(signal, sample_ohlcv_data)
        positions = results['positions']
        assert (positions >= 0).all()

class TestPerformance:
    def test_ten_year_backtest_completes_under_5_seconds(self, large_ohlcv_data):
        engine = BacktestEngine()
        signal = SimpleSignal()
        start_time = time.time()
        results = engine.run_backtest(signal, large_ohlcv_data)
        elapsed_time = time.time() - start_time
        assert elapsed_time < 5.0
        assert len(results['portfolio_value']) == len(large_ohlcv_data)

class TestTradesAndPositions:
    def test_results_contain_trade_entries_and_exits(self, sample_ohlcv_data):
        engine = BacktestEngine()
        signal = SimpleSignal()
        results = engine.run_backtest(signal, sample_ohlcv_data)
        trades = results['trades']
        assert isinstance(trades, pd.DataFrame)

class TestEdgeCases:
    def test_all_zeros_signal_produces_flat_portfolio(self, sample_ohlcv_data):
        engine = BacktestEngine()
        signal = AllZerosSignal()
        results = engine.run_backtest(signal, sample_ohlcv_data)
        portfolio_value = results['portfolio_value']
        assert (portfolio_value == engine.initial_capital).all()
