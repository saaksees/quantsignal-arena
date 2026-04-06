"""
Pydantic schemas for API request and response models.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
import uuid


# Request models
class GenerateSignalRequest(BaseModel):
    hypothesis: str = Field(..., min_length=10, max_length=500, description="Plain English trading hypothesis")
    tickers: list[str] = Field(..., min_items=1, max_items=10, description="List of ticker symbols e.g. ['AAPL', 'MSFT']")
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Start date YYYY-MM-DD")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="End date YYYY-MM-DD")
    
    @validator('tickers')
    def tickers_uppercase(cls, v):
        return [t.upper().strip() for t in v]
    
    @validator('end_date')
    def end_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class PaperTradeRequest(BaseModel):
    active: bool = True


# Response models
class MetricsResponse(BaseModel):
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    cagr: Optional[float] = None
    win_rate: Optional[float] = None
    total_return: Optional[float] = None
    volatility: Optional[float] = None
    calmar_ratio: Optional[float] = None


class SHAPResponse(BaseModel):
    summary: str
    top_features: list[dict]
    feature_importance: dict


class DriftResponse(BaseModel):
    drift_level: str
    signal_psi: float
    return_psi: float
    drift_detected: bool
    recommendation: str


class SignalRunResponse(BaseModel):
    run_id: str
    hypothesis: str
    tickers: list[str]
    metrics: Optional[MetricsResponse] = None
    shap_results: Optional[SHAPResponse] = None
    drift_results: Optional[DriftResponse] = None
    report_url: Optional[str] = None
    generated_code: Optional[str] = None
    signal_name: Optional[str] = None
    attempts_taken: Optional[int] = None
    success: bool
    error: Optional[str] = None
    is_paper_trading: bool = False
    created_at: Optional[str] = None


class LeaderboardEntry(BaseModel):
    run_id: str
    hypothesis: str
    signal_name: Optional[str] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    cagr: Optional[float] = None
    win_rate: Optional[float] = None
    is_paper_trading: bool = False
    created_at: Optional[str] = None


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry]
    total: int
    metric: str


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
