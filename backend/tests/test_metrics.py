"""
Tests for MetricsCalculator module.

Uses hand-calculated expected values for validation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtester.metrics import MetricsCalculator


@pytest.fixture
def calculator():
    """Create a MetricsCalculator instance."""
    return MetricsCalculator(risk_free_rate=0.02)


@pytest.fixture
def flat_returns():
    """Returns series with all zeros."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    return pd.Series(0.0, index=dates)


@pytest.fixture
def positive_returns():
    """Returns series with all positive values."""
    dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
    return pd.Series(0.01, index=dates)


@pytest.fixture
def negative_returns():
    """Returns series with all negative values."""
    dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
    return pd.Series(-0.01, index=dates)


@pytest.fixture
def known_returns():
    """Returns series with known statistical properties."""
    # Create a series with known mean and std
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    # Mean daily return = 0.001 (0.1%), Std = 0.02 (2%)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    return returns


@pytest.fixture
def known_drawdown_returns():
    """Returns series with known drawdown pattern."""
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    # Pattern: +10%, +10%, -20%, -10%, +5%, +5%, +5%, +5%, +5%, +5%
    # This creates a specific drawdown we can calculate
    returns = pd.Series([0.10, 0.10, -0.20, -0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], index=dates)
    return returns


@pytest.fixture
def short_returns():
    """Returns series with less than 30 days."""
    dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
    return pd.Series(0.01, index=dates)


class TestFlatReturns:
    """Test metrics with flat (all-zero) returns."""
    
    def test_flat_returns_sharpe_is_none(self, calculator, flat_returns):
        metrics = calculator.calculate_metrics(flat_returns)
        assert metrics['sharpe_ratio'] is None
    
    def test_flat_returns_max_drawdown_is_zero(self, calculator, flat_returns):
        metrics = calculator.calculate_metrics(flat_returns)
        assert metrics['max_drawdown'] == 0.0


class TestKnownReturns:
    """Test metrics with known return series."""
    
    def test_sharpe_ratio_within_expected_range(self, calculator, known_returns):
        metrics = calculator.calculate_metrics(known_returns)
        # With mean=0.001, std=0.02, rf=0.02/252
        # Sharpe ≈ (0.001 - 0.02/252) / 0.02 * sqrt(252)
        # Expected ≈ 0.73
        assert metrics['sharpe_ratio'] is not None
        assert abs(metrics['sharpe_ratio'] - 0.73) < 0.5  # Allow some variance due to randomness


class TestDrawdown:
    """Test max drawdown calculation."""
    
    def test_known_drawdown_matches_expected(self, calculator, known_drawdown_returns):
        metrics = calculator.calculate_metrics(known_drawdown_returns)
        # Cumulative: 1.1, 1.21, 0.968, 0.8712, 0.91476, 0.96050, 1.00852, 1.05895, 1.11190, 1.16749
        # Running max: 1.1, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21, 1.21
        # Drawdown at index 3: (0.8712 - 1.21) / 1.21 = -0.28
        # Max drawdown should be approximately -0.28
        assert metrics['max_drawdown'] is not None
        assert abs(metrics['max_drawdown'] - (-0.28)) < 0.01


class TestShortReturns:
    """Test metrics with less than 30 days of data."""
    
    def test_short_returns_sharpe_is_none(self, calculator, short_returns):
        metrics = calculator.calculate_metrics(short_returns)
        assert metrics['sharpe_ratio'] is None
    
    def test_short_returns_sortino_is_none(self, calculator, short_returns):
        metrics = calculator.calculate_metrics(short_returns)
        assert metrics['sortino_ratio'] is None


class TestWinRate:
    """Test win rate calculation."""
    
    def test_all_positive_returns_win_rate_is_one(self, calculator, positive_returns):
        metrics = calculator.calculate_metrics(positive_returns)
        assert metrics['win_rate'] == 1.0
    
    def test_all_negative_returns_win_rate_is_zero(self, calculator, negative_returns):
        metrics = calculator.calculate_metrics(negative_returns)
        assert metrics['win_rate'] == 0.0


class TestTotalReturn:
    """Test total return calculation."""
    
    def test_total_return_calculation(self, calculator):
        # Simple case: 3 days with +10%, -5%, +10%
        dates = pd.date_range(start='2020-01-01', periods=3, freq='D')
        returns = pd.Series([0.10, -0.05, 0.10], index=dates)
        metrics = calculator.calculate_metrics(returns)
        # Total: (1.10 * 0.95 * 1.10) - 1 = 1.1495 - 1 = 0.1495
        assert abs(metrics['total_return'] - 0.1495) < 0.0001


class TestCAGR:
    """Test CAGR calculation."""
    
    def test_cagr_calculation(self, calculator):
        # 252 days (1 year) with constant 0.1% daily return
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        returns = pd.Series(0.001, index=dates)
        metrics = calculator.calculate_metrics(returns)
        # CAGR = (1.001^252)^(1/1) - 1 ≈ 0.2872 (28.72%)
        assert metrics['cagr'] is not None
        assert abs(metrics['cagr'] - 0.2872) < 0.01


class TestEmptySeries:
    """Test metrics with empty return series."""
    
    def test_empty_series_returns_dict_with_none_values(self, calculator):
        empty_returns = pd.Series([], dtype=float)
        metrics = calculator.calculate_metrics(empty_returns)
        
        assert isinstance(metrics, dict)
        assert metrics['sharpe_ratio'] is None
        assert metrics['sortino_ratio'] is None
        assert metrics['max_drawdown'] == 0.0
        assert metrics['win_rate'] == 0.0
        assert metrics['cagr'] is None
        assert metrics['total_return'] == 0.0
        assert metrics['volatility'] == 0.0
        assert metrics['calmar_ratio'] is None


class TestCalmarRatio:
    """Test Calmar ratio calculation."""
    
    def test_calmar_ratio_with_valid_inputs(self, calculator, known_drawdown_returns):
        metrics = calculator.calculate_metrics(known_drawdown_returns)
        # Calmar = CAGR / abs(max_drawdown)
        if metrics['cagr'] is not None and metrics['max_drawdown'] != 0:
            expected_calmar = metrics['cagr'] / abs(metrics['max_drawdown'])
            assert abs(metrics['calmar_ratio'] - expected_calmar) < 0.0001
    
    def test_calmar_ratio_is_none_when_no_drawdown(self, calculator, positive_returns):
        metrics = calculator.calculate_metrics(positive_returns)
        # All positive returns means no drawdown (max_drawdown = 0)
        # Calmar should be None when drawdown is zero
        if metrics['max_drawdown'] == 0.0:
            assert metrics['calmar_ratio'] is None


class TestVolatility:
    """Test volatility calculation."""
    
    def test_volatility_is_annualized(self, calculator):
        # Create returns with known daily std = 0.02
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252), index=dates)
        metrics = calculator.calculate_metrics(returns)
        # Annualized vol = 0.02 * sqrt(252) ≈ 0.3175
        assert abs(metrics['volatility'] - 0.02 * np.sqrt(252)) < 0.05
