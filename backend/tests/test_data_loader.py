"""
Unit tests for DataLoader module.

Tests use mocked yfinance responses to avoid real API calls.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtester.data_loader import DataLoader


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide temporary cache directory for testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def sample_ohlcv_data():
    """Generate synthetic OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D', tz='UTC')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'Adj Close': np.random.uniform(100, 110, len(dates))
    }, index=dates)
    return data


@pytest.fixture
def mock_yfinance_download(sample_ohlcv_data):
    """Mock yfinance.download() to return sample data."""
    with patch('backtester.data_loader.yf.download') as mock_download:
        mock_download.return_value = sample_ohlcv_data
        yield mock_download


class TestDataLoaderInit:
    """Test DataLoader initialization."""
    
    def test_init_creates_cache_directory(self, temp_cache_dir):
        """Test that cache directory is created on initialization."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=True)
        assert Path(temp_cache_dir).exists()
        assert loader.cache_enabled is True
    
    def test_init_with_cache_disabled(self, temp_cache_dir):
        """Test initialization with caching disabled."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        assert loader.cache_enabled is False


class TestLoadData:
    """Test load_data method."""
    
    def test_valid_ticker_returns_correct_dataframe(self, temp_cache_dir, mock_yfinance_download, sample_ohlcv_data):
        """Test that valid ticker returns DataFrame with correct shape and columns."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        result = loader.load_data("AAPL", "2020-01-01", "2020-01-10")
        
        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        
        # Check columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for col in expected_columns:
            assert col in result.columns
        
        # Check index is DatetimeIndex with UTC timezone
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        
        # Verify yfinance was called
        mock_yfinance_download.assert_called_once()
    
    def test_cached_data_returned_without_yfinance_call(self, temp_cache_dir, mock_yfinance_download, sample_ohlcv_data):
        """Test that cached data is returned without calling yfinance."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=True)
        
        # First call - should fetch from yfinance
        result1 = loader.load_data("AAPL", "2020-01-01", "2020-01-10")
        assert mock_yfinance_download.call_count == 1
        
        # Second call - should use cache
        result2 = loader.load_data("AAPL", "2020-01-01", "2020-01-10")
        assert mock_yfinance_download.call_count == 1  # No additional call
        
        # Results should have same shape and values
        assert result1.shape == result2.shape
        assert list(result1.columns) == list(result2.columns)
        # Compare values (allowing for small floating point differences)
        pd.testing.assert_frame_equal(result1, result2, check_freq=False)
    
    def test_invalid_ticker_raises_value_error(self, temp_cache_dir):
        """Test that invalid ticker raises ValueError."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        with patch('backtester.data_loader.yf.download') as mock_download:
            # Mock empty DataFrame for invalid ticker
            mock_download.return_value = pd.DataFrame()
            
            with pytest.raises(ValueError, match="Invalid ticker"):
                loader.load_data("INVALID_TICKER", "2020-01-01", "2020-01-10")
    
    def test_end_date_before_start_date_raises_value_error(self, temp_cache_dir):
        """Test that end_date before start_date raises ValueError."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        with pytest.raises(ValueError, match="end_date must be after start_date"):
            loader.load_data("AAPL", "2020-01-10", "2020-01-01")
    
    def test_invalid_date_format_raises_value_error(self, temp_cache_dir):
        """Test that invalid date format raises ValueError."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        with pytest.raises(ValueError, match="Invalid date format"):
            loader.load_data("AAPL", "invalid-date", "2020-01-10")
    
    def test_network_error_triggers_retry(self, temp_cache_dir):
        """Test that network errors trigger retry logic."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        with patch('backtester.data_loader.yf.download') as mock_download:
            # First two calls fail, third succeeds
            mock_download.side_effect = [
                ConnectionError("Network error"),
                ConnectionError("Network error"),
                pd.DataFrame({
                    'Open': [100],
                    'High': [110],
                    'Low': [90],
                    'Close': [105],
                    'Volume': [1000000],
                    'Adj Close': [105]
                }, index=pd.DatetimeIndex(['2020-01-01']))
            ]
            
            with patch('backtester.data_loader.time.sleep'):  # Mock sleep to speed up test
                result = loader.load_data("AAPL", "2020-01-01", "2020-01-02")
            
            # Should have retried 3 times
            assert mock_download.call_count == 3
            assert not result.empty
    
    def test_all_retries_fail_raises_connection_error(self, temp_cache_dir):
        """Test that all retries failing raises ConnectionError."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        with patch('backtester.data_loader.yf.download') as mock_download:
            # All calls fail
            mock_download.side_effect = ConnectionError("Network error")
            
            with patch('backtester.data_loader.time.sleep'):  # Mock sleep to speed up test
                with pytest.raises(ConnectionError, match="Failed to fetch data"):
                    loader.load_data("AAPL", "2020-01-01", "2020-01-02")
            
            # Should have tried 3 times
            assert mock_download.call_count == 3


class TestDataNormalization:
    """Test data normalization functionality."""
    
    def test_timestamps_normalized_to_utc(self, temp_cache_dir, mock_yfinance_download):
        """Test that all timestamps are normalized to UTC."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        result = loader.load_data("AAPL", "2020-01-01", "2020-01-10")
        
        # Check timezone is UTC
        assert result.index.tz is not None
        assert str(result.index.tz) == 'UTC'
    
    def test_missing_data_forward_filled(self, temp_cache_dir):
        """Test that missing data is forward-filled correctly."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        # Create data with missing values
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        data_with_gaps = pd.DataFrame({
            'Open': [100, np.nan, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'High': [110, np.nan, np.nan, 113, 114, 115, 116, 117, 118, 119],
            'Low': [90, np.nan, np.nan, 93, 94, 95, 96, 97, 98, 99],
            'Close': [105, np.nan, np.nan, 108, 109, 110, 111, 112, 113, 114],
            'Volume': [1000000] * 10,
            'Adj Close': [105, np.nan, np.nan, 108, 109, 110, 111, 112, 113, 114]
        }, index=dates)
        
        with patch('backtester.data_loader.yf.download') as mock_download:
            mock_download.return_value = data_with_gaps
            
            result = loader.load_data("AAPL", "2020-01-01", "2020-01-10")
        
        # Check that NaN values were forward-filled
        assert result['Open'].iloc[1] == 100  # Forward-filled from index 0
        assert result['Open'].iloc[2] == 100  # Forward-filled from index 0
    
    def test_negative_prices_raise_value_error(self, temp_cache_dir):
        """Test that negative prices raise ValueError."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        # Create data with negative prices
        dates = pd.date_range(start='2020-01-01', end='2020-01-05', freq='D')
        data_with_negative = pd.DataFrame({
            'Open': [100, 101, -102, 103, 104],  # Negative price
            'High': [110, 111, 112, 113, 114],
            'Low': [90, 91, 92, 93, 94],
            'Close': [105, 106, 107, 108, 109],
            'Volume': [1000000] * 5,
            'Adj Close': [105, 106, 107, 108, 109]
        }, index=dates)
        
        with patch('backtester.data_loader.yf.download') as mock_download:
            mock_download.return_value = data_with_negative
            
            with pytest.raises(ValueError, match="negative prices"):
                loader.load_data("AAPL", "2020-01-01", "2020-01-05")


class TestCaching:
    """Test caching functionality."""
    
    def test_cache_file_created(self, temp_cache_dir, mock_yfinance_download):
        """Test that cache file is created after loading data."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=True)
        
        loader.load_data("AAPL", "2020-01-01", "2020-01-10")
        
        # Check cache file exists
        cache_file = Path(temp_cache_dir) / "AAPL.parquet"
        assert cache_file.exists()
    
    def test_cache_file_format_is_parquet(self, temp_cache_dir, mock_yfinance_download):
        """Test that cache files are stored in parquet format."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=True)
        
        loader.load_data("AAPL", "2020-01-01", "2020-01-10")
        
        # Check file extension
        cache_file = Path(temp_cache_dir) / "AAPL.parquet"
        assert cache_file.suffix == ".parquet"
        
        # Verify it can be read as parquet
        cached_data = pd.read_parquet(cache_file)
        assert isinstance(cached_data, pd.DataFrame)
    
    def test_partial_cache_triggers_fetch(self, temp_cache_dir, sample_ohlcv_data):
        """Test that partial cache coverage triggers new fetch."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=True)
        
        with patch('backtester.data_loader.yf.download') as mock_download:
            # First call - cache only part of the range
            partial_data = sample_ohlcv_data.iloc[:5]  # Only first 5 days
            mock_download.return_value = partial_data
            loader.load_data("AAPL", "2020-01-01", "2020-01-05")
            
            # Second call - request larger range
            full_data = sample_ohlcv_data  # All 10 days
            mock_download.return_value = full_data
            result = loader.load_data("AAPL", "2020-01-01", "2020-01-10")
            
            # Should have called yfinance twice (cache was incomplete)
            assert mock_download.call_count == 2


class TestLoadMultiple:
    """Test load_multiple method."""
    
    def test_load_multiple_tickers(self, temp_cache_dir, sample_ohlcv_data):
        """Test loading multiple tickers returns dictionary."""
        loader = DataLoader(cache_dir=temp_cache_dir, cache_enabled=False)
        
        with patch('backtester.data_loader.yf.download') as mock_download:
            mock_download.return_value = sample_ohlcv_data
            
            tickers = ["AAPL", "MSFT", "GOOGL"]
            result = loader.load_multiple(tickers, "2020-01-01", "2020-01-10")
        
        # Check result is dictionary
        assert isinstance(result, dict)
        assert len(result) == 3
        
        # Check all tickers are present
        for ticker in tickers:
            assert ticker in result
            assert isinstance(result[ticker], pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
