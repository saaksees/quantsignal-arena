"""
JWT authentication middleware.
"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
