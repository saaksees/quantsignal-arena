"""
Tests for Pydantic schemas.
"""
import pytest
from pydantic import ValidationError
from api.models.schemas import (
    GenerateSignalRequest,
    MetricsResponse,
    SignalRunResponse,
)


def test_generate_signal_request_valid():
    """Test: GenerateSignalRequest with valid data creates successfully"""
    req = GenerateSignalRequest(
        hypothesis="Buy when RSI is oversold",
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    assert req.hypothesis == "Buy when RSI is oversold"
    assert req.tickers == ["AAPL", "MSFT"]
    assert req.start_date == "2020-01-01"
    assert req.end_date == "2023-12-31"


def test_hypothesis_too_short():
    """Test: hypothesis shorter than 10 chars raises ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        GenerateSignalRequest(
            hypothesis="short",
            tickers=["AAPL"],
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
    assert "at least 10 characters" in str(exc_info.value)


def test_hypothesis_too_long():
    """Test: hypothesis longer than 500 chars raises ValidationError"""
    long_hypothesis = "a" * 501
    with pytest.raises(ValidationError) as exc_info:
        GenerateSignalRequest(
            hypothesis=long_hypothesis,
            tickers=["AAPL"],
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
    assert "at most 500 characters" in str(exc_info.value)


def test_empty_tickers():
    """Test: empty tickers list raises ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        GenerateSignalRequest(
            hypothesis="Buy when RSI is oversold",
            tickers=[],
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
    assert "at least 1 item" in str(exc_info.value)


def test_too_many_tickers():
    """Test: more than 10 tickers raises ValidationError"""
    tickers = [f"TICK{i}" for i in range(11)]
    with pytest.raises(ValidationError) as exc_info:
        GenerateSignalRequest(
            hypothesis="Buy when RSI is oversold",
            tickers=tickers,
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
    assert "at most 10 items" in str(exc_info.value)


def test_tickers_uppercase():
    """Test: tickers are converted to uppercase automatically"""
    req = GenerateSignalRequest(
        hypothesis="Buy when RSI is oversold",
        tickers=["aapl", "msft", " tsla "],
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    assert req.tickers == ["AAPL", "MSFT", "TSLA"]


def test_invalid_date_format():
    """Test: invalid date format raises ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        GenerateSignalRequest(
            hypothesis="Buy when RSI is oversold",
            tickers=["AAPL"],
            start_date="01/01/2020",
            end_date="2023-12-31"
        )
    assert "String should match pattern" in str(exc_info.value)


def test_end_date_before_start_date():
    """Test: end_date before start_date raises ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        GenerateSignalRequest(
            hypothesis="Buy when RSI is oversold",
            tickers=["AAPL"],
            start_date="2023-12-31",
            end_date="2020-01-01"
        )
    assert "end_date must be after start_date" in str(exc_info.value)


def test_metrics_response_none_fields():
    """Test: MetricsResponse accepts None for all fields"""
    metrics = MetricsResponse()
    assert metrics.sharpe_ratio is None
    assert metrics.sortino_ratio is None
    assert metrics.max_drawdown is None
    assert metrics.cagr is None
    assert metrics.win_rate is None
    assert metrics.total_return is None
    assert metrics.volatility is None


def test_signal_run_response_minimal():
    """Test: SignalRunResponse with minimal fields creates successfully"""
    response = SignalRunResponse(
        run_id="123e4567-e89b-12d3-a456-426614174000",
        hypothesis="Buy when RSI is oversold",
        tickers=["AAPL"],
        success=True
    )
    assert response.run_id == "123e4567-e89b-12d3-a456-426614174000"
    assert response.hypothesis == "Buy when RSI is oversold"
    assert response.tickers == ["AAPL"]
    assert response.success is True
    assert response.metrics is None
    assert response.error is None
