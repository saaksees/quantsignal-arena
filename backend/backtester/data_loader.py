"""
Data Loader Module for QuantSignal Arena Backtesting Engine.

Handles market data acquisition from yfinance with local caching for performance.
"""

from typing import Optional, List
from datetime import datetime
from pathlib import Path
import logging
import time
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles market data acquisition with local caching.
    
    Attributes:
        cache_dir: Directory path for storing cached parquet files
        cache_enabled: Whether to use local caching
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        cache_enabled: bool = True
    ) -> None:
        """
        Initialize DataLoader with cache configuration.
        
        Args:
            cache_dir: Path to cache directory
            cache_enabled: Enable/disable caching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        
        # Create cache directory if it doesn't exist
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory initialized at: {self.cache_dir}")
    
    def load_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a single ticker.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            
        Returns:
            DataFrame with DatetimeIndex and columns:
            [Open, High, Low, Close, Volume, Adj Close]
            
        Raises:
            ValueError: If ticker is invalid or dates are malformed
            ConnectionError: If yfinance API is unreachable after retries
        """
        # Validate dates
        start_dt, end_dt = self._validate_dates(start_date, end_date)
        
        # Try to load from cache first
        if self.cache_enabled:
            cached_data = self._load_from_cache(ticker, start_dt, end_dt)
            if cached_data is not None:
                logger.info(f"Cache hit for {ticker}")
                return cached_data
            logger.info(f"Cache miss for {ticker}")
        
        # Fetch from yfinance
        data = self._fetch_from_yfinance(ticker, start_date, end_date)
        
        # Normalize data
        data = self._normalize_data(data)
        
        # Save to cache
        if self.cache_enabled:
            self._save_to_cache(ticker, data)
        
        return data
    
    def load_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            
        Returns:
            Dictionary mapping ticker to OHLCV DataFrame
        """
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.load_data(ticker, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to load data for {ticker}: {e}")
                raise
        return result
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get cache file path for a ticker."""
        return self.cache_dir / f"{ticker}.parquet"
    
    def _load_from_cache(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and complete."""
        cache_path = self._get_cache_path(ticker)
        
        if not cache_path.exists():
            return None
        
        try:
            # Load cached data
            cached_data = pd.read_parquet(cache_path)
            
            # Check if cache covers the requested date range
            if cached_data.empty:
                return None
            
            # Ensure start_date and end_date are timezone-aware (UTC)
            if start_date.tzinfo is None:
                start_date = start_date.tz_localize('UTC')
            else:
                start_date = start_date.tz_convert('UTC')
            
            if end_date.tzinfo is None:
                end_date = end_date.tz_localize('UTC')
            else:
                end_date = end_date.tz_convert('UTC')
            
            cache_start = cached_data.index.min()
            cache_end = cached_data.index.max()
            
            # If cache covers the requested range, filter and return
            if cache_start <= start_date and cache_end >= end_date:
                filtered_data = cached_data.loc[start_date:end_date]
                return filtered_data
            
            # Cache is incomplete - need to fetch missing data
            logger.info(f"Cache incomplete for {ticker}, fetching missing data")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {ticker}: {e}")
            return None
    
    def _save_to_cache(
        self,
        ticker: str,
        data: pd.DataFrame
    ) -> None:
        """Save data to cache in parquet format."""
        cache_path = self._get_cache_path(ticker)
        
        try:
            # Check if cache file exists
            if cache_path.exists():
                # Load existing cache
                existing_data = pd.read_parquet(cache_path)
                
                # Combine with new data, removing duplicates
                combined_data = pd.concat([existing_data, data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data = combined_data.sort_index()
                
                # Save combined data
                combined_data.to_parquet(cache_path)
                logger.info(f"Appended data to cache for {ticker}")
            else:
                # Save new cache file
                data.to_parquet(cache_path)
                logger.info(f"Created cache file for {ticker}")
                
        except Exception as e:
            logger.error(f"Failed to save cache for {ticker}: {e}")
    
    def _fetch_from_yfinance(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        retries: int = 3
    ) -> pd.DataFrame:
        """Fetch data from yfinance with retry logic."""
        last_error = None
        
        for attempt in range(retries):
            try:
                logger.info(f"Fetching {ticker} from yfinance (attempt {attempt + 1}/{retries})")
                
                # Download data from yfinance
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    show_errors=False
                )
                
                # Check if data is empty
                if data.empty:
                    raise ValueError(f"Invalid ticker: {ticker}")
                
                return data
                
            except ValueError as e:
                # Don't retry on ValueError (invalid ticker, etc.)
                raise
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {retries} attempts failed for {ticker}")
        
        # All retries failed
        raise ConnectionError(f"Failed to fetch data for {ticker} after {retries} retries: {last_error}")
    
    def _validate_dates(
        self,
        start_date: str,
        end_date: str
    ) -> tuple[datetime, datetime]:
        """Validate and parse date strings."""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got: {start_date}, {end_date}. Error: {e}")
        
        # Validate end_date is after start_date
        if end_dt <= start_dt:
            raise ValueError(f"end_date must be after start_date. Got start={start_date}, end={end_date}")
        
        return start_dt, end_dt
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize OHLCV data:
        - Convert timezone to UTC
        - Forward-fill missing values (max 5 days)
        - Validate no negative prices
        """
        # Convert timezone to UTC
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        else:
            data.index = data.index.tz_convert('UTC')
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_columns:
            if col in data.columns and (data[col] < 0).any():
                raise ValueError(f"OHLCV data contains negative prices in column {col}")
        
        # Forward-fill missing values (max 5 consecutive days)
        data = data.ffill(limit=5)
        
        return data
