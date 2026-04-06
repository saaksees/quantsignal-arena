"""
Tests for ReportBuilder module.
"""

import pytest
import os
import pandas as pd
import numpy as np
from datetime import date
from shap_layer.report_builder import ReportBuilder


@pytest.fixture
def synthetic_backtest_results():
    """Generate synthetic backtest results."""
    dates = pd.date_range(end='2024-01-01', periods=100, freq='D')
    portfolio_value = pd.Series(
        100000 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod(),
        index=dates
    )
    
    return {
        'portfolio_value': portfolio_value,
        'returns': portfolio_value.pct_change().dropna(),
        'trades': pd.DataFrame(),
        'positions': pd.Series(),
        'signal_series': pd.Series()
    }


@pytest.fixture
def synthetic_metrics():
    """Generate synthetic metrics."""
    return {
        'sharpe_ratio': 1.5,
        'sortino_ratio': 2.0,
        'max_drawdown': -0.15,
        'cagr': 0.25,
        'win_rate': 0.55,
        'total_return': 0.30,
        'volatility': 0.18
    }


@pytest.fixture
def synthetic_shap_results():
    """Generate synthetic SHAP results."""
    feature_names = ['returns_1d', 'returns_5d', 'returns_20d', 'volatility_20d',
                     'volume_ratio', 'price_momentum', 'rsi_14', 'bb_position']
    
    feature_importance = {
        'returns_20d': 0.15,
        'volatility_20d': 0.12,
        'price_momentum': 0.10,
        'returns_5d': 0.08,
        'rsi_14': 0.07,
        'bb_position': 0.06,
        'returns_1d': 0.05,
        'volume_ratio': 0.04
    }
    
    mean_shap_signed = np.array([0.05, 0.08, 0.15, -0.12, 0.04, 0.10, 0.07, -0.06])
    
    return {
        'shap_values': np.random.randn(100, 8),
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'base_value': 0.5,
        'summary': "Top drivers: returns_20d (positive), volatility_20d (negative), price_momentum (positive)",
        'model': None,
        'mean_shap_signed': mean_shap_signed
    }


@pytest.fixture
def synthetic_drift_results():
    """Generate synthetic drift results."""
    return {
        'signal_psi': 0.08,
        'return_psi': 0.12,
        'drift_detected': False,
        'drift_level': 'none',
        'recommendation': 'signal stable',
        'max_psi': 0.12,
        'reference_period': (pd.Timestamp('2023-01-01'), pd.Timestamp('2023-06-30')),
        'detection_period': (pd.Timestamp('2023-12-01'), pd.Timestamp('2023-12-31'))
    }


@pytest.fixture
def report_builder(tmp_path):
    """Create ReportBuilder with temporary output directory."""
    return ReportBuilder(output_dir=str(tmp_path))


def test_build_returns_string_path(report_builder, synthetic_backtest_results, 
                                   synthetic_metrics, synthetic_shap_results, 
                                   synthetic_drift_results):
    """Test build() returns a string path."""
    result = report_builder.build(
        hypothesis="Test hypothesis",
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results
    )
    
    assert isinstance(result, str)


def test_returned_path_ends_with_pdf(report_builder, synthetic_backtest_results,
                                     synthetic_metrics, synthetic_shap_results,
                                     synthetic_drift_results):
    """Test returned path ends with .pdf."""
    result = report_builder.build(
        hypothesis="Test hypothesis",
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results
    )
    
    assert result.endswith('.pdf')


def test_pdf_file_exists_on_disk(report_builder, synthetic_backtest_results,
                                 synthetic_metrics, synthetic_shap_results,
                                 synthetic_drift_results):
    """Test PDF file actually exists on disk after build()."""
    result = report_builder.build(
        hypothesis="Test hypothesis",
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results
    )
    
    assert os.path.exists(result)
    assert os.path.isfile(result)


def test_output_directory_created_if_not_exists(tmp_path):
    """Test output directory is created if it does not exist."""
    new_dir = os.path.join(str(tmp_path), "new_reports")
    assert not os.path.exists(new_dir)
    
    builder = ReportBuilder(output_dir=new_dir)
    
    assert os.path.exists(new_dir)
    assert os.path.isdir(new_dir)


def test_pdf_filename_contains_date(report_builder, synthetic_backtest_results,
                                    synthetic_metrics, synthetic_shap_results,
                                    synthetic_drift_results):
    """Test PDF filename contains today's date in YYYYMMDD format."""
    result = report_builder.build(
        hypothesis="Test hypothesis",
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results
    )
    
    today_str = date.today().strftime('%Y%m%d')
    assert today_str in result


def test_build_twice_creates_file(report_builder, synthetic_backtest_results,
                                  synthetic_metrics, synthetic_shap_results,
                                  synthetic_drift_results):
    """Test calling build() twice creates a file (idempotent)."""
    result1 = report_builder.build(
        hypothesis="Test hypothesis",
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results
    )
    
    result2 = report_builder.build(
        hypothesis="Test hypothesis 2",
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results
    )
    
    assert os.path.exists(result1)
    assert os.path.exists(result2)
    # Both should have same filename (same date)
    assert os.path.basename(result1) == os.path.basename(result2)


def test_very_long_hypothesis_truncated(report_builder, synthetic_backtest_results,
                                       synthetic_metrics, synthetic_shap_results,
                                       synthetic_drift_results):
    """Test very long hypothesis is truncated and does not crash."""
    long_hypothesis = "A" * 500  # 500 character hypothesis
    
    result = report_builder.build(
        hypothesis=long_hypothesis,
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results
    )
    
    assert os.path.exists(result)


def test_build_with_generated_code(report_builder, synthetic_backtest_results,
                                   synthetic_metrics, synthetic_shap_results,
                                   synthetic_drift_results):
    """Test build() works with generated code."""
    code = "def my_signal():\n    return 1\n" * 30  # 60 lines
    
    result = report_builder.build(
        hypothesis="Test hypothesis",
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results,
        generated_code=code
    )
    
    assert os.path.exists(result)


def test_build_without_generated_code(report_builder, synthetic_backtest_results,
                                      synthetic_metrics, synthetic_shap_results,
                                      synthetic_drift_results):
    """Test build() works without generated code."""
    result = report_builder.build(
        hypothesis="Test hypothesis",
        backtest_results=synthetic_backtest_results,
        metrics=synthetic_metrics,
        shap_results=synthetic_shap_results,
        drift_results=synthetic_drift_results
    )
    
    assert os.path.exists(result)
