"""
Report download endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
import os
from api.middleware.auth import require_auth

router = APIRouter()


def get_db():
    if os.getenv("TESTING"):
        from api.db.mock_supabase_client import MockSupabaseClient
        return MockSupabaseClient()
    from api.db.supabase_client import SupabaseClient
    return SupabaseClient()


@router.get("/{run_id}")
async def get_report(
    run_id: str,
    current_user: dict = Depends(require_auth)
):
    db = get_db()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    report_path = run.get("report_path")
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=f"quantsignal_report_{run_id[:8]}.pdf"
    )
