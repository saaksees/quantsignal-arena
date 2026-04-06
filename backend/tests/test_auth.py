"""
Tests for JWT authentication middleware.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from jose import JWTError
from api.middleware.auth import AuthMiddleware


@pytest.fixture
def auth():
    with patch.dict('os.environ', {'SUPABASE_JWT_SECRET': 'test-secret-key-32-chars-minimum!!'}):
        return AuthMiddleware()


def test_valid_token_returns_user_id(auth):
    """Test: valid token with sub claim returns user_id"""
    with patch('api.middleware.auth.jwt.decode') as mock_decode:
        mock_decode.return_value = {"sub": "user-123", "email": "test@example.com"}
        
        result = auth.verify_token("valid.jwt.token")
        
        assert result["user_id"] == "user-123"
        mock_decode.assert_called_once()


def test_valid_token_returns_full_payload(auth):
    """Test: valid token returns full payload"""
    with patch('api.middleware.auth.jwt.decode') as mock_decode:
        mock_payload = {"sub": "user-123", "email": "test@example.com", "role": "authenticated"}
        mock_decode.return_value = mock_payload
        
        result = auth.verify_token("valid.jwt.token")
        
        assert result["payload"] == mock_payload
        assert result["payload"]["email"] == "test@example.com"
        assert result["payload"]["role"] == "authenticated"


def test_token_missing_sub_raises_401(auth):
    """Test: token missing sub claim raises 401"""
    with patch('api.middleware.auth.jwt.decode') as mock_decode:
        mock_decode.return_value = {"email": "test@example.com"}
        
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token("token.without.sub")
        
        assert exc_info.value.status_code == 401
        assert "missing user id" in exc_info.value.detail


def test_invalid_signature_raises_401(auth):
    """Test: invalid token signature raises 401"""
    with patch('api.middleware.auth.jwt.decode') as mock_decode:
        mock_decode.side_effect = JWTError("Signature verification failed")
        
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token("invalid.signature.token")
        
        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail


def test_expired_token_raises_401(auth):
    """Test: expired token raises 401"""
    with patch('api.middleware.auth.jwt.decode') as mock_decode:
        mock_decode.side_effect = JWTError("Token has expired")
        
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token("expired.jwt.token")
        
        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail


def test_missing_jwt_secret_raises_500():
    """Test: missing JWT secret raises 500"""
    with patch.dict('os.environ', {}, clear=True):
        auth = AuthMiddleware()
        
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token("any.jwt.token")
        
        assert exc_info.value.status_code == 500
        assert "JWT secret not configured" in exc_info.value.detail


def test_malformed_token_raises_401(auth):
    """Test: malformed token string raises 401"""
    with patch('api.middleware.auth.jwt.decode') as mock_decode:
        mock_decode.side_effect = JWTError("Not enough segments")
        
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token("malformed-token")
        
        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail


def test_valid_token_extracts_correct_user_id(auth):
    """Test: valid token extracts correct user_id value"""
    with patch('api.middleware.auth.jwt.decode') as mock_decode:
        expected_user_id = "550e8400-e29b-41d4-a716-446655440000"
        mock_decode.return_value = {"sub": expected_user_id, "aud": "authenticated"}
        
        result = auth.verify_token("valid.jwt.token")
        
        assert result["user_id"] == expected_user_id
        assert isinstance(result["user_id"], str)
