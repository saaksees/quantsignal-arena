"""
Tests for SignalAgent module.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

# Add backend to path
backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import using normal imports now that path is set
from backtester.engine import BacktestEngine
from backtester.metrics import MetricsCalculator

# Direct import for SignalAgent to avoid circular issues
import importlib.util
spec_sa = importlib.util.spec_from_file_location(
    "signal_agent",
    Path(__file__).parent.parent / "agent" / "signal_agent.py"
)
signal_agent_module = importlib.util.module_from_spec(spec_sa)
spec_sa.loader.exec_module(signal_agent_module)
SignalAgent = signal_agent_module.SignalAgent


VALID_SIGNAL_CODE = """
class TestSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        signals = pd.Series(0, index=ohlcv_data.index, dtype='int8')
        returns = ohlcv_data['Close'].pct_change()
        signals[returns > 0] = 1
        signals[returns < 0] = -1
        return signals
"""

BROKEN_SIGNAL_CODE = """
class BrokenSignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        import pandas as pd
        return pd.Series(999, index=ohlcv_data.index)
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


@pytest.fixture
def mock_anthropic_client():
    """Create mock Anthropic client that returns valid signal code."""
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "write_signal"
    mock_block.input = {
        "code": VALID_SIGNAL_CODE,
        "explanation": "Simple momentum signal"
    }
    
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    
    client = MagicMock()
    client.messages.create.return_value = mock_response
    
    return client


@pytest.fixture
def mock_anthropic_client_broken():
    """Create mock Anthropic client that returns broken signal code."""
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "write_signal"
    mock_block.input = {
        "code": BROKEN_SIGNAL_CODE,
        "explanation": "Broken signal"
    }
    
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    
    client = MagicMock()
    client.messages.create.return_value = mock_response
    
    return client


@pytest.fixture
def backtest_engine():
    """Create BacktestEngine instance."""
    return BacktestEngine(initial_capital=100000.0)


@pytest.fixture
def metrics_calculator():
    """Create MetricsCalculator instance."""
    return MetricsCalculator(risk_free_rate=0.02)


class TestSignalAgent:
    """Test suite for SignalAgent class."""
    
    def test_successful_hypothesis_returns_all_required_keys(
        self, mock_anthropic_client, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that successful hypothesis returns dict with all required keys."""
        agent = SignalAgent(mock_anthropic_client, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        required_keys = {
            "hypothesis", "generated_code", "signal_name", "attempts_taken",
            "success", "error", "backtest_results", "metrics"
        }
        assert set(results.keys()) == required_keys
    
    def test_success_true_when_valid_signal_code(
        self, mock_anthropic_client, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that success=True when Claude returns valid signal code."""
        agent = SignalAgent(mock_anthropic_client, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["success"] is True
    
    def test_attempts_taken_one_when_first_succeeds(
        self, mock_anthropic_client, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that attempts_taken=1 when first attempt succeeds."""
        agent = SignalAgent(mock_anthropic_client, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["attempts_taken"] == 1
    
    def test_backtest_results_not_none_on_success(
        self, mock_anthropic_client, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that backtest_results is not None on success."""
        agent = SignalAgent(mock_anthropic_client, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["backtest_results"] is not None
    
    def test_metrics_not_none_on_success(
        self, mock_anthropic_client, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that metrics is not None on success."""
        agent = SignalAgent(mock_anthropic_client, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["metrics"] is not None
    
    def test_hypothesis_preserved_in_results(
        self, mock_anthropic_client, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that hypothesis is preserved in results dict."""
        hypothesis = "Buy when momentum is positive"
        agent = SignalAgent(mock_anthropic_client, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest(hypothesis, sample_ohlcv)
        
        assert results["hypothesis"] == hypothesis
    
    def test_generated_code_not_none_on_success(
        self, mock_anthropic_client, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that generated_code is not None on success."""
        agent = SignalAgent(mock_anthropic_client, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["generated_code"] is not None
    
    def test_success_false_when_all_retries_fail(
        self, mock_anthropic_client_broken, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that success=False when all retries return invalid code."""
        agent = SignalAgent(mock_anthropic_client_broken, backtest_engine, metrics_calculator, max_retries=3)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["success"] is False
    
    def test_attempts_taken_equals_max_retries_when_all_fail(
        self, mock_anthropic_client_broken, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that attempts_taken equals max_retries when all attempts fail."""
        max_retries = 3
        agent = SignalAgent(mock_anthropic_client_broken, backtest_engine, metrics_calculator, max_retries=max_retries)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["attempts_taken"] == max_retries
    
    def test_error_not_none_when_success_false(
        self, mock_anthropic_client_broken, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that error is not None when success=False."""
        agent = SignalAgent(mock_anthropic_client_broken, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["error"] is not None
    
    def test_retry_triggered_when_first_attempt_fails(
        self, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that retry is triggered when first attempt fails."""
        # Create mock that fails first, succeeds second
        broken_block = MagicMock()
        broken_block.type = "tool_use"
        broken_block.name = "write_signal"
        broken_block.input = {"code": BROKEN_SIGNAL_CODE, "explanation": "broken"}
        
        valid_block = MagicMock()
        valid_block.type = "tool_use"
        valid_block.name = "write_signal"
        valid_block.input = {"code": VALID_SIGNAL_CODE, "explanation": "valid"}
        
        broken_response = MagicMock()
        broken_response.content = [broken_block]
        
        valid_response = MagicMock()
        valid_response.content = [valid_block]
        
        client = MagicMock()
        client.messages.create.side_effect = [broken_response, valid_response]
        
        agent = SignalAgent(client, backtest_engine, metrics_calculator, max_retries=3)
        results = agent.generate_and_backtest("test hypothesis", sample_ohlcv)
        
        # Verify client was called twice
        assert client.messages.create.call_count == 2
        assert results["attempts_taken"] == 2
        assert results["success"] is True
    
    def test_signal_name_matches_class_name(
        self, mock_anthropic_client, backtest_engine, metrics_calculator, sample_ohlcv
    ):
        """Test that signal_name matches the class name in generated code."""
        agent = SignalAgent(mock_anthropic_client, backtest_engine, metrics_calculator)
        results = agent.generate_and_backtest("Buy on momentum", sample_ohlcv)
        
        assert results["signal_name"] == "TestSignal"
