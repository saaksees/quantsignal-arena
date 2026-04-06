# Requirements: FastAPI REST API

## Overview

This specification defines a production-ready REST API that exposes the QuantSignal Arena Python engine (Months 1-3) as callable HTTP endpoints. The API provides signal generation, backtesting, leaderboard functionality, and report downloads with Supabase persistence and JWT authentication.

## Functional Requirements

### 1. Signal Generation Endpoint

**1.1** THE API SHALL expose a POST /api/signals/generate endpoint that accepts hypothesis, tickers, start_date, and end_date

**1.2** THE endpoint SHALL validate hypothesis length between 10 and 500 characters

**1.3** THE endpoint SHALL validate tickers list contains 1-10 ticker symbols

**1.4** THE endpoint SHALL validate date format as YYYY-MM-DD using regex pattern

**1.5** THE endpoint SHALL call SignalAgent.generate_and_backtest() with provided parameters

**1.6** THE endpoint SHALL call SignalExplainer.explain() on successful signal generation

**1.7** THE endpoint SHALL call DriftDetector.detect() on successful signal generation

**1.8** THE endpoint SHALL call ReportBuilder.build() to generate PDF tearsheet

**1.9** THE endpoint SHALL save run metadata to Supabase signal_runs table

**1.10** THE endpoint SHALL return run_id, hypothesis, metrics, shap_summary, drift_level, report_url, generated_code, and success status

**1.11** THE endpoint SHALL use Server-Sent Events (SSE) to stream progress updates during execution

**1.12** THE endpoint SHALL associate run with authenticated user_id from JWT

### 2. Signal Retrieval Endpoint

**2.1** THE API SHALL expose a GET /api/signals/{run_id} endpoint

**2.2** THE endpoint SHALL fetch run from Supabase by run_id

**2.3** THE endpoint SHALL return 404 if run_id does not exist

**2.4** THE endpoint SHALL return 403 if run does not belong to authenticated user

**2.5** THE endpoint SHALL return full run details including metrics, shap_results, drift_results, and generated_code

### 3. Report Download Endpoint

**3.1** THE API SHALL expose a GET /api/signals/{run_id}/report endpoint

**3.2** THE endpoint SHALL fetch report_path from Supabase by run_id

**3.3** THE endpoint SHALL return 404 if report file does not exist on disk

**3.4** THE endpoint SHALL return 403 if run does not belong to authenticated user

**3.5** THE endpoint SHALL return PDF as FileResponse with content-type application/pdf

**3.6** THE endpoint SHALL set Content-Disposition header to attachment with filename

### 4. Leaderboard Endpoints

**4.1** THE API SHALL expose a GET /api/leaderboard endpoint

**4.2** THE endpoint SHALL accept query parameter limit (default 20, max 100)

**4.3** THE endpoint SHALL accept query parameter metric (default "sharpe_ratio")

**4.4** THE endpoint SHALL validate metric is one of: sharpe_ratio, sortino_ratio, cagr, total_return

**4.5** THE endpoint SHALL fetch top runs from Supabase ordered by metric descending

**4.6** THE endpoint SHALL filter runs where success=true

**4.7** THE endpoint SHALL return list of runs with run_id, hypothesis, metrics, created_at

**4.8** THE API SHALL expose a GET /api/leaderboard/{run_id} endpoint

**4.9** THE endpoint SHALL return single leaderboard entry with full details

**4.10** THE endpoint SHALL return 404 if run_id does not exist or success=false

### 5. Paper Trading Endpoint

**5.1** THE API SHALL expose a POST /api/signals/{run_id}/paper_trade endpoint

**5.2** THE endpoint SHALL mark signal as active for paper trading

**5.3** THE endpoint SHALL update is_paper_trading=true in Supabase

**5.4** THE endpoint SHALL return 403 if run does not belong to authenticated user

**5.5** THE endpoint SHALL return 400 if run success=false

### 6. Health Check Endpoint

**6.1** THE API SHALL expose a GET /api/health endpoint

**6.2** THE endpoint SHALL return status "ok" and version "1.0.0"

**6.3** THE endpoint SHALL NOT require authentication

**6.4** THE endpoint SHALL return 200 status code

### 7. Authentication Middleware

**7.1** THE API SHALL validate Supabase JWT on every request except /api/health

**7.2** THE middleware SHALL extract user_id from JWT claims

**7.3** THE middleware SHALL attach user_id to request.state

**7.4** THE middleware SHALL return 401 if Authorization header is missing

**7.5** THE middleware SHALL return 401 if JWT token is invalid or expired

**7.6** THE middleware SHALL return 401 if JWT signature verification fails

**7.7** THE middleware SHALL use python-jose library for JWT validation

### 8. Database Integration

**8.1** THE API SHALL use Supabase client for all database operations

**8.2** THE API SHALL create signal_runs table with all required columns

**8.3** THE API SHALL enforce Row Level Security (RLS) policies

**8.4** THE API SHALL create indexes on user_id, sharpe_ratio, and created_at

**8.5** THE API SHALL store generated_code, metrics, shap_summary, drift_level, and report_path

**8.6** THE API SHALL update updated_at timestamp on every update

**8.7** THE API SHALL use UUID for run_id primary key

### 9. Request/Response Schemas

**9.1** THE API SHALL use Pydantic models for request validation

**9.2** THE API SHALL use Pydantic models for response serialization

**9.3** THE API SHALL validate hypothesis min_length=10, max_length=500

**9.4** THE API SHALL validate tickers min_items=1, max_items=10

**9.5** THE API SHALL validate date format using regex pattern ^\d{4}-\d{2}-\d{2}$

**9.6** THE API SHALL return 422 for validation errors with detailed error messages

**9.7** THE API SHALL return MetricsResponse with all metric fields as float | None

**9.8** THE API SHALL return SignalRunResponse with run_id, hypothesis, metrics, shap_summary, drift_level, report_url, generated_code, success, error, created_at

### 10. Error Handling

**10.1** THE API SHALL return 400 for invalid request parameters

**10.2** THE API SHALL return 401 for authentication failures

**10.3** THE API SHALL return 403 for authorization failures

**10.4** THE API SHALL return 404 for resource not found

**10.5** THE API SHALL return 422 for validation errors

**10.6** THE API SHALL return 500 for internal server errors

**10.7** THE API SHALL log all errors with timestamps and context

**10.8** THE API SHALL return error responses with message and detail fields

### 11. CORS Configuration

**11.1** THE API SHALL enable CORS for frontend origins

**11.2** THE API SHALL allow credentials in CORS requests

**11.3** THE API SHALL allow methods: GET, POST, PUT, DELETE, OPTIONS

**11.4** THE API SHALL allow headers: Authorization, Content-Type

### 12. Server-Sent Events (SSE)

**12.1** THE /api/signals/generate endpoint SHALL stream progress updates using SSE

**12.2** THE endpoint SHALL emit events: started, generating_signal, running_backtest, computing_shap, detecting_drift, building_report, completed, error

**12.3** THE endpoint SHALL include progress percentage in each event

**12.4** THE endpoint SHALL close SSE stream on completion or error

**12.5** THE endpoint SHALL set Content-Type to text/event-stream

## Non-Functional Requirements

### 13. Performance

**13.1** THE API SHALL respond to /api/health within 100ms

**13.2** THE API SHALL handle concurrent requests using async/await

**13.3** THE API SHALL use connection pooling for Supabase client

**13.4** THE API SHALL cache Supabase client instance

### 14. Security

**14.1** THE API SHALL use HTTPS in production

**14.2** THE API SHALL validate JWT signature using Supabase public key

**14.3** THE API SHALL enforce RLS policies on all database queries

**14.4** THE API SHALL sanitize user inputs to prevent SQL injection

**14.5** THE API SHALL rate limit requests to prevent abuse

### 15. Testing

**15.1** THE API SHALL have unit tests for all endpoints

**15.2** THE API SHALL mock external dependencies (Supabase, SignalAgent, etc.)

**15.3** THE API SHALL use FastAPI TestClient for integration tests

**15.4** THE API SHALL achieve 90%+ code coverage

**15.5** THE API SHALL test authentication middleware with valid and invalid tokens

**15.6** THE API SHALL test validation errors for all request schemas

**15.7** THE API SHALL test authorization failures for protected endpoints

**15.8** THE API SHALL test SSE streaming for signal generation endpoint

## Summary

**Total Requirements**: 15 categories, 95 individual requirements

**Key Components**:
- 7 REST endpoints (generate, retrieve, report, leaderboard, paper_trade, health)
- JWT authentication middleware
- Supabase database integration
- Pydantic request/response schemas
- Server-Sent Events for progress streaming
- Comprehensive error handling and validation

**Dependencies**:
- Month 1: BacktestEngine, MetricsCalculator
- Month 2: SignalAgent
- Month 3: SignalExplainer, DriftDetector, ReportBuilder
- External: FastAPI, Supabase, python-jose, sse-starlette
