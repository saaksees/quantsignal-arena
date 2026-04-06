"""
API endpoint tests using TestClient with mocked dependencies.
"""
import os
os.environ["TESTING"] = "true"
os.environ["SUPABASE_URL"] = "https://fake.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "fake-key"
os.environ["SUPABASE_JWT_SECRET"] = "fake-secret-32-chars-minimum-ok!!"
os.environ["ANTHROPIC_API_KEY"] = "fake-key"

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.middleware.auth import require_auth

# Override auth dependency
app.dependency_overrides[require_auth] = lambda: {"user_id": "test-user-123"}


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint_returns_ok(client):
    """Test: GET /api/health returns 200 with {"status": "ok", "version": "1.0.0"}"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"


def test_generate_signal_without_auth_returns_401():
    """Test: POST /api/signals/generate without auth header returns 401"""
    # Clear overrides temporarily
    app.dependency_overrides.clear()
    client = TestClient(app)
    
    response = client.post("/api/signals/generate", json={
        "hypothesis": "Buy when RSI is oversold",
        "tickers": ["AAPL"],
        "start_date": "2020-01-01",
        "end_date": "2023-12-31"
    })
    
    assert response.status_code == 401
    
    # Restore override
    app.dependency_overrides[require_auth] = lambda: {"user_id": "test-user-123"}


def test_get_signal_nonexistent_returns_404(client):
    """Test: GET /api/signals/nonexistent-id with auth returns 404"""
    response = client.get("/api/signals/nonexistent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_leaderboard_without_auth_returns_401():
    """Test: GET /api/leaderboard without auth returns 401"""
    # Clear overrides temporarily
    app.dependency_overrides.clear()
    client = TestClient(app)
    
    response = client.get("/api/leaderboard")
    
    assert response.status_code == 401
    
    # Restore override
    app.dependency_overrides[require_auth] = lambda: {"user_id": "test-user-123"}


def test_leaderboard_invalid_metric_returns_400(client):
    """Test: GET /api/leaderboard with invalid metric returns 400"""
    response = client.get("/api/leaderboard?metric=invalid_metric")
    assert response.status_code == 400
    assert "Invalid metric" in response.json()["detail"]


def test_get_report_nonexistent_returns_404(client):
    """Test: GET /api/reports/nonexistent-id with auth returns 404"""
    response = client.get("/api/reports/nonexistent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_paper_trade_nonexistent_returns_404(client):
    """Test: POST /api/signals/fake-id/paper_trade with auth returns 404"""
    response = client.post(
        "/api/signals/fake-id/paper_trade",
        json={"active": True}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
