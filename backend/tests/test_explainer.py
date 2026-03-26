"""
Tests for SignalExplainer module.
"""

import pytest
import pandas as pd
import numpy as np
from shap_layer.explainer import SignalExplainer
from backtester.signal_base import MomentumSignal


@pytest.fixture
def synthetic_ohlcv_data():
    """Generate 300 days of synthetic OHLCV data."""
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range(end='2024-01-01', periods=n_days, freq='D')
    
    # Generate realistic price data with trend
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% daily return, 2% volatility
    close = initial_price * (1 + returns).cumprod()
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_price = close * (1 + np.random.normal(0, 0.005, n_days))
    
    # Generate volume
    volume = np.random.randint(1_000_000, 10_000_000, n_days)
    
    ohlcv = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return ohlcv


@pytest.fixture
def momentum_signal():
    """Create a MomentumSignal instance for testing."""
    return MomentumSignal(lookback_period=20)


def test_explainer_initialization():
    """Test SignalExplainer can be initialized."""
    explainer = SignalExplainer()
    assert explainer is not None
    assert explainer.backtest_engine is None


def test_build_features_returns_8_columns(synthetic_ohlcv_data):
    """Test _build_features returns DataFrame with exactly 8 columns."""
    explainer = SignalExplainer()
    features = explainer._build_features(synthetic_ohlcv_data)
    
    assert isinstance(features, pd.DataFrame)
    assert len(features.columns) == 8
    
    expected_columns = [
        'returns_1d', 'returns_5d', 'returns_20d', 'volatility_20d',
        'volume_ratio', 'price_momentum', 'rsi_14', 'bb_position'
    ]
    assert list(features.columns) == expected_columns


def test_build_features_drops_nan_rows(synthetic_ohlcv_data):
    """Test _build_features drops NaN rows - output length less than input."""
    explainer = SignalExplainer()
    features = explainer._build_features(synthetic_ohlcv_data)
    
    # Should have fewer rows due to rolling windows (20-day features)
    assert len(features) < len(synthetic_ohlcv_data)
    
    # Should have no NaN values
    assert not features.isna().any().any()


def test_explain_returns_dict_with_required_keys(synthetic_ohlcv_data, momentum_signal):
    """Test explain() returns dict with all required keys."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    assert isinstance(result, dict)
    
    required_keys = ['shap_values', 'feature_names', 'feature_importance', 'base_value', 'summary', 'model']
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"


def test_feature_importance_has_8_keys(synthetic_ohlcv_data, momentum_signal):
    """Test feature_importance dict has exactly 8 keys matching feature names."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    feature_importance = result['feature_importance']
    assert isinstance(feature_importance, dict)
    assert len(feature_importance) == 8
    
    expected_features = [
        'returns_1d', 'returns_5d', 'returns_20d', 'volatility_20d',
        'volume_ratio', 'price_momentum', 'rsi_14', 'bb_position'
    ]
    for feat in expected_features:
        assert feat in feature_importance


def test_shap_values_is_numpy_array(synthetic_ohlcv_data, momentum_signal):
    """Test shap_values is a numpy array."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    shap_values = result['shap_values']
    assert isinstance(shap_values, np.ndarray)
    assert shap_values.ndim == 2  # Should be 2D array (n_samples, n_features)
    assert shap_values.shape[1] == 8  # 8 features


def test_get_top_features_returns_3_items(synthetic_ohlcv_data, momentum_signal):
    """Test get_top_features(result, 3) returns list of exactly 3 items."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    top_features = explainer.get_top_features(result, n=3)
    
    assert isinstance(top_features, list)
    assert len(top_features) == 3


def test_get_top_features_has_required_keys(synthetic_ohlcv_data, momentum_signal):
    """Test each item in get_top_features has keys: name, mean_shap, direction."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    top_features = explainer.get_top_features(result, n=3)
    
    for item in top_features:
        assert isinstance(item, dict)
        assert 'name' in item
        assert 'mean_shap' in item
        assert 'direction' in item


def test_direction_is_positive_or_negative(synthetic_ohlcv_data, momentum_signal):
    """Test direction is either 'positive' or 'negative' - never anything else."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    top_features = explainer.get_top_features(result, n=3)
    
    for item in top_features:
        direction = item['direction']
        assert direction in ['positive', 'negative'], f"Invalid direction: {direction}"


def test_summary_string_not_empty(synthetic_ohlcv_data, momentum_signal):
    """Test summary string is not empty and contains at least one feature name."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    summary = result['summary']
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Check that at least one feature name appears in summary
    feature_names = result['feature_names']
    assert any(feat in summary for feat in feature_names), "Summary should mention at least one feature"


def test_feature_importance_sorted_descending(synthetic_ohlcv_data, momentum_signal):
    """Test feature_importance values are sorted in descending order."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    feature_importance = result['feature_importance']
    importance_values = list(feature_importance.values())
    
    # Check that values are in descending order
    for i in range(len(importance_values) - 1):
        assert importance_values[i] >= importance_values[i + 1], \
            f"Feature importance not sorted: {importance_values[i]} < {importance_values[i + 1]}"


def test_base_value_is_float(synthetic_ohlcv_data, momentum_signal):
    """Test base_value is a float."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    base_value = result['base_value']
    assert isinstance(base_value, float)


def test_model_is_trained(synthetic_ohlcv_data, momentum_signal):
    """Test model is a trained classifier."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    model = result['model']
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')


def test_get_top_features_with_different_n(synthetic_ohlcv_data, momentum_signal):
    """Test get_top_features works with different values of n."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    # Test n=1
    top_1 = explainer.get_top_features(result, n=1)
    assert len(top_1) == 1
    
    # Test n=5
    top_5 = explainer.get_top_features(result, n=5)
    assert len(top_5) == 5
    
    # Test n=8 (all features)
    top_8 = explainer.get_top_features(result, n=8)
    assert len(top_8) == 8


def test_feature_names_match_columns(synthetic_ohlcv_data, momentum_signal):
    """Test feature_names in result match the feature columns."""
    explainer = SignalExplainer()
    result = explainer.explain(momentum_signal, synthetic_ohlcv_data)
    
    feature_names = result['feature_names']
    expected_names = [
        'returns_1d', 'returns_5d', 'returns_20d', 'volatility_20d',
        'volume_ratio', 'price_momentum', 'rsi_14', 'bb_position'
    ]
    
    assert feature_names == expected_names
