# ADR-002: FastAPI API Design Decisions

## Status: Accepted

## Decisions

### SSE over WebSockets for streaming
Signal generation takes 30-60 seconds. SSE (Server-Sent Events) gives real-time progress updates to the frontend without WebSocket complexity. One-directional stream is all we need.

### Lazy Supabase initialization
SupabaseClient connects only when first query runs, not at import time. This lets tests import the app without needing real Supabase credentials.

### MockSupabaseClient via TESTING env var
Routes check TESTING=true to swap real Supabase for in-memory mock. Zero test infrastructure needed — no Docker, no test database.

### Pydantic validators on request models
Ticker uppercase normalization and date ordering enforced at schema level, not in route handlers. Errors returned as 422 before any business logic runs.

## Consequences
- All routes are async — never block the event loop with sync I/O
- Every route except /health requires Bearer token
- Report PDFs stored on local disk — Month 5 moves these to Supabase Storage
