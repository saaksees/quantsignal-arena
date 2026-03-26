"""
Tests for CodeExecutor module.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Direct import to avoid circular dependencies
import importlib.util

# Import SignalBase
spec = importlib.util.spec_from_file_location(
    "signal_base",
    Path(__file__).parent.parent / "backtester" / "signal_base.py"
)
signal_base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signal_base_module)
SignalBase = signal_base_module.SignalBase

# Import CodeExecutor
spec2 = importlib.util.spec_from_file_location(
    "code_executor",
    Path(__file__).parent.parent / "agent" / "code_executor.py"
)
code_executor_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(code_executor_module)
CodeExecutor = code_executor_module.CodeExecutor
SecurityError = code_executor_module.SecurityError


VALID_SIGNAL_CODE = """
class TestSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        signals = pd.Series(0, index=ohlcv_data.index, dtype='int8')
        returns = ohlcv_data['Close'].pct_change()
        signals[returns > 0] = 1
        signals[returns < 0] = -1
        return signals
"""


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D', tz='UTC')
    prices = pd.Series(100.0, index=dates)
    for i in range(1, len(prices)):
        prices.iloc[i] = prices.iloc[i-1] * (1 + np.random.uniform(-0.02, 0.02))
    
    return pd.DataFrame({
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': 1000000,
        'Adj Close': prices
    })


class TestCodeExecutor:
    """Test suite for CodeExecutor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.executor = CodeExecutor()
    
    def test_import_os_raises_security_error(self, sample_ohlcv):
        """Test that 'import os' in code raises SecurityError."""
        code = """
import os
class TestSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        return pd.Series(0, index=ohlcv_data.index)
"""
        signal, error = self.executor.execute(code, sample_ohlcv)
        
        assert signal is None
        assert error is not None
        assert "Forbidden import detected: os" in error
    
    def test_import_subprocess_raises_security_error(self, sample_ohlcv):
        """Test that 'import subprocess' raises SecurityError."""
        code = """
import subprocess
class TestSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        return pd.Series(0, index=ohlcv_data.index)
"""
        signal, error = self.executor.execute(code, sample_ohlcv)
        
        assert signal is None
        assert error is not None
        assert "Forbidden import detected: subprocess" in error
    
    def test_from_sys_import_raises_security_error(self, sample_ohlcv):
        """Test that 'from sys import path' raises SecurityError."""
        code = """
from sys import path
class TestSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        return pd.Series(0, index=ohlcv_data.index)
"""
        signal, error = self.executor.execute(code, sample_ohlcv)
        
        assert signal is None
        assert error is not None
        assert "Forbidden import detected: sys" in error
    
    def test_import_requests_raises_security_error(self, sample_ohlcv):
        """Test that 'import requests' raises SecurityError."""
        code = """
import requests
class TestSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        return pd.Series(0, index=ohlcv_data.index)
"""
        signal, error = self.executor.execute(code, sample_ohlcv)
        
        assert signal is None
        assert error is not None
        assert "Forbidden import detected: requests" in error
    
    def test_valid_code_returns_signalbase_instance(self, sample_ohlcv):
        """Test that valid code returns SignalBase subclass instance."""
        signal, error = self.executor.execute(VALID_SIGNAL_CODE, sample_ohlcv)
        
        assert signal is not None
        # Check that it's a SignalBase subclass by checking base class name
        assert any(base.__name__ == 'SignalBase' for base in signal.__class__.__mro__)
        assert error is None
    
    def test_valid_code_returns_none_error(self, sample_ohlcv):
        """Test that valid code returns tuple where second element is None."""
        signal, error = self.executor.execute(VALID_SIGNAL_CODE, sample_ohlcv)
        
        assert error is None
    
    def test_syntax_error_returns_none_and_error_string(self, sample_ohlcv):
        """Test that code with SyntaxError returns (None, string)."""
        code = """
class TestSignal(SignalBase):
    def broken(
"""
        signal, error = self.executor.execute(code, sample_ohlcv)
        
        assert signal is None
        assert error is not None
        assert isinstance(error, str)
        assert "SyntaxError" in error
    
    def test_no_signalbase_subclass_returns_error(self, sample_ohlcv):
        """Test that code with no SignalBase subclass returns (None, string)."""
        code = """
class NotASignal:
    pass
"""
        signal, error = self.executor.execute(code, sample_ohlcv)
        
        assert signal is None
        assert error is not None
        assert "No SignalBase subclass found" in error
    
    def test_wrong_signal_values_returns_error(self, sample_ohlcv):
        """Test that code returning wrong signal values returns (None, string)."""
        code = """
class TestSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        return pd.Series(999, index=ohlcv_data.index)
"""
        signal, error = self.executor.execute(code, sample_ohlcv)
        
        assert signal is None
        assert error is not None
        assert isinstance(error, str)
    
    def test_empty_string_returns_error(self, sample_ohlcv):
        """Test that empty string returns (None, string)."""
        code = ""
        signal, error = self.executor.execute(code, sample_ohlcv)
        
        assert signal is None
        assert error is not None
        assert isinstance(error, str)
