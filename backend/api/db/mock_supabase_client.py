"""
In-memory mock Supabase client for testing.
"""
import uuid
from datetime import datetime


class MockSupabaseClient:
    def __init__(self):
        self._runs: dict[str, dict] = {}

    def insert_run(self, run_data: dict) -> dict:
        run_id = str(uuid.uuid4())
        record = {**run_data, "id": run_id, "created_at": datetime.now().isoformat()}
        self._runs[run_id] = record
        return record

    def get_run(self, run_id: str) -> dict | None:
        return self._runs.get(run_id)

    def update_run(self, run_id: str, updates: dict) -> dict:
        if run_id in self._runs:
            self._runs[run_id].update(updates)
            return self._runs[run_id]
        return {}

    def get_leaderboard(self, metric: str = "sharpe_ratio", limit: int = 20) -> list[dict]:
        runs = [r for r in self._runs.values() if r.get("success")]
        sorted_runs = sorted(runs, key=lambda x: x.get(metric) or 0, reverse=True)
        return sorted_runs[:limit]

    def get_leaderboard_count(self) -> int:
        return len([r for r in self._runs.values() if r.get("success")])
