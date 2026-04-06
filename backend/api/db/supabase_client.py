"""
Supabase client wrapper for persisting signal runs.
"""
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class SupabaseClient:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")
            if not url or not key:
                raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
            self._client = create_client(url, key)
        return self._client

    def insert_run(self, run_data: dict) -> dict:
        # Insert a new signal run, return the created record with id
        response = self._get_client().table("signal_runs").insert(run_data).execute()
        return response.data[0] if response.data else {}

    def get_run(self, run_id: str) -> dict | None:
        response = self._get_client().table("signal_runs").select("*").eq("id", run_id).execute()
        return response.data[0] if response.data else None

    def update_run(self, run_id: str, updates: dict) -> dict:
        response = self._get_client().table("signal_runs").update(updates).eq("id", run_id).execute()
        return response.data[0] if response.data else {}

    def get_leaderboard(self, metric: str = "sharpe_ratio", limit: int = 20) -> list[dict]:
        response = (
            self._get_client().table("signal_runs")
            .select("id, hypothesis, signal_name, sharpe_ratio, sortino_ratio, max_drawdown, cagr, win_rate, is_paper_trading, created_at")
            .eq("success", True)
            .order(metric, desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    def get_leaderboard_count(self) -> int:
        response = self._get_client().table("signal_runs").select("id", count="exact").eq("success", True).execute()
        return response.count or 0
