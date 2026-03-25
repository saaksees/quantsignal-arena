"""
Tests for WalkForwardValidator module.

Tests both rolling and anchored window strategies with various configurations.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtester.walk_forward import WalkForwardValidator
from backtester.signal_base import SignalBase


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D', tz='UTC')
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
    }, index=dates)
    
    return data


@pytest.fixture
def validator():
    """Create a WalkForwardValidator instance."""
    return WalkForwardValidator()


class SimpleTestSignal(SignalBase):
    """Simple signal for testing."""
    def generate_signals(self, ohlcv_data):
        signals = pd.Series(0, index=ohlcv_data.index, dtype=np.int8)
        signals.iloc[::2] = 1
        return signals


class TestRollingSplit:
    """Test rolling window split."""
    
    def test_rolling_split_produces_correct_number_of_folds(self, validator, sample_data):
        # 100 days total, train=30, test=20, gap=0
        # Fold 1: train[0:30], test[30:50]
        # Fold 2: train[20:50], test[50:70]
        # Fold 3: train[40:70], test[70:90]
        # Total: 3 folds (last fold would need 90+20=110 days)
        folds = validator.split(sample_data, train_size=30, test_size=20, gap=0, anchored=False)
        assert len(folds) == 3
    
    def test_rolling_split_train_window_slides(self, validator, sample_data):
        folds = validator.split(sample_data, train_size=20, test_size=10, gap=0, anchored=False)
        
        # Check that train windows slide forward
        assert len(folds[0][0]) == 20  # First fold train size
        assert len(folds[1][0]) == 20  # Second fold train size
        
        # Train start should advance by test_size
        train1_start = folds[0][0].index[0]
        train2_start = folds[1][0].index[0]
        assert (train2_start - train1_start).days == 10


class TestAnchoredSplit:
    """Test anchored window split."""
    
    def test_anchored_split_train_always_starts_from_index_zero(self, validator, sample_data):
        folds = validator.split(sample_data, train_size=30, test_size=20, gap=0, anchored=True)
        
        # All folds should start training from index 0
        for fold_idx, (train_df, test_df) in enumerate(folds):
            assert train_df.index[0] == sample_data.index[0]
    
    def test_anchored_split_train_window_grows(self, validator, sample_data):
        folds = validator.split(sample_data, train_size=30, test_size=20, gap=0, anchored=True)
        
        # Train window should grow with each fold
        if len(folds) >= 2:
            assert len(folds[1][0]) > len(folds[0][0])


class TestNoOverlap:
    """Test that test windows never overlap."""
    
    def test_no_test_windows_overlap(self, validator, sample_data):
        folds = validator.split(sample_data, train_size=20, test_size=15, gap=0, anchored=False)
        
        # Check all test windows are non-overlapping
        for i in range(len(folds) - 1):
            test1_end = folds[i][1].index[-1]
            test2_start = folds[i + 1][1].index[0]
            assert test2_start > test1_end


class TestGapPeriod:
    """Test gap period between train and test."""
    
    def test_gap_period_is_respected(self, validator, sample_data):
        gap_days = 5
        folds = validator.split(sample_data, train_size=20, test_size=10, gap=gap_days, anchored=False)
        
        for train_df, test_df in folds:
            train_end = train_df.index[-1]
            test_start = test_df.index[0]
            
            # Calculate actual gap (accounting for weekends/holidays in data)
            actual_gap = (test_start - train_end).days
            
            # Gap should be at least the specified gap_days
            assert actual_gap >= gap_days


class TestPartialWindow:
    """Test that partial final window is excluded."""
    
    def test_partial_final_window_is_excluded(self, validator, sample_data):
        # 100 days, train=30, test=25
        # Fold 1: train[0:30], test[30:55]
        # Fold 2: train[25:55], test[55:80]
        # Fold 3: train[50:80], test[80:105] <- would need 105 days, excluded
        folds = validator.split(sample_data, train_size=30, test_size=25, gap=0, anchored=False)
        
        # Should only have 2 complete folds
        assert len(folds) == 2
        
        # Last test window should end before data ends
        last_test_end = folds[-1][1].index[-1]
        assert last_test_end <= sample_data.index[-1]


class TestValidation:
    """Test parameter validation."""
    
    def test_train_size_zero_raises_error(self, validator, sample_data):
        with pytest.raises(ValueError, match="train_size must be a positive integer"):
            validator.split(sample_data, train_size=0, test_size=10)
    
    def test_negative_train_size_raises_error(self, validator, sample_data):
        with pytest.raises(ValueError, match="train_size must be a positive integer"):
            validator.split(sample_data, train_size=-10, test_size=10)
    
    def test_test_size_zero_raises_error(self, validator, sample_data):
        with pytest.raises(ValueError, match="test_size must be a positive integer"):
            validator.split(sample_data, train_size=10, test_size=0)
    
    def test_negative_gap_raises_error(self, validator, sample_data):
        with pytest.raises(ValueError, match="gap must be a non-negative integer"):
            validator.split(sample_data, train_size=10, test_size=10, gap=-5)
    
    def test_insufficient_data_raises_error(self, validator, sample_data):
        # Request more data than available
        with pytest.raises(ValueError, match="Insufficient data"):
            validator.split(sample_data, train_size=80, test_size=50, gap=0)


class TestRunWalkForward:
    """Test run_walk_forward method."""
    
    def test_run_walk_forward_returns_one_dict_per_fold(self, validator, sample_data):
        signal = SimpleTestSignal()
        results = validator.run_walk_forward(
            signal=signal,
            data=sample_data,
            train_size=20,
            test_size=15,
            gap=0,
            anchored=False
        )
        
        # Should return list of dicts
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
    
    def test_each_metrics_dict_contains_fold_number(self, validator, sample_data):
        signal = SimpleTestSignal()
        results = validator.run_walk_forward(
            signal=signal,
            data=sample_data,
            train_size=20,
            test_size=15,
            gap=0,
            anchored=False
        )
        
        # Check fold_number exists and is sequential
        for idx, result in enumerate(results):
            assert 'fold_number' in result
            assert result['fold_number'] == idx
    
    def test_each_metrics_dict_contains_test_start_and_end(self, validator, sample_data):
        signal = SimpleTestSignal()
        results = validator.run_walk_forward(
            signal=signal,
            data=sample_data,
            train_size=20,
            test_size=15,
            gap=0,
            anchored=False
        )
        
        # Check date fields exist
        for result in results:
            assert 'test_start' in result
            assert 'test_end' in result
            assert 'train_start' in result
            assert 'train_end' in result
            
            # Check dates are in correct order
            assert result['train_start'] <= result['train_end']
            assert result['test_start'] <= result['test_end']
            assert result['train_end'] < result['test_start']
    
    def test_each_metrics_dict_contains_performance_metrics(self, validator, sample_data):
        signal = SimpleTestSignal()
        results = validator.run_walk_forward(
            signal=signal,
            data=sample_data,
            train_size=20,
            test_size=15,
            gap=0,
            anchored=False
        )
        
        # Check metrics exist
        expected_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'win_rate', 'cagr', 'total_return', 'volatility', 'calmar_ratio'
        ]
        
        for result in results:
            for metric in expected_metrics:
                assert metric in result


class TestWindowSizes:
    """Test that window sizes are correct."""
    
    def test_train_and_test_sizes_match_specification(self, validator, sample_data):
        train_size = 25
        test_size = 15
        
        folds = validator.split(sample_data, train_size=train_size, test_size=test_size, gap=0, anchored=False)
        
        for train_df, test_df in folds:
            # Rolling windows should have exact train_size
            assert len(train_df) == train_size
            assert len(test_df) == test_size
