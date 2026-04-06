"""
Leaderboard endpoints.
"""
import os
from fastapi import APIRouter, Depends, Query, HTTPException
from api.middleware.auth import require_auth
from api.models.schemas import LeaderboardResponse, LeaderboardEntry

router = APIRouter()

VALID_METRICS = ["sharpe_ratio", "sortino_ratio", "cagr", "win_rate", "total_return"]


def get_db():
    if os.getenv("TESTING"):
        from api.db.mock_supabase_client import MockSupabaseClient
        return MockSupabaseClient()
    from api.db.supabase_client import SupabaseClient
    return SupabaseClient()


@router.get("")
async def get_leaderboard(
    metric: str = Query("sharpe_ratio", description="Metric to rank by"),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(require_auth)
):
    if metric not in VALID_METRICS:
        raise HTTPException(status_code=400, detail=f"Invalid metric. Choose from: {VALID_METRICS}")
    
    db = get_db()
    entries = db.get_leaderboard(metric=metric, limit=limit)
    total = db.get_leaderboard_count()
    
    return LeaderboardResponse(
        entries=[LeaderboardEntry(**e) for e in entries],
        total=total,
        metric=metric
    )


@router.get("/{run_id}")
async def get_leaderboard_entry(
    run_id: str,
    current_user: dict = Depends(require_auth)
):
    db = get_db()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
