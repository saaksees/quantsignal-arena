"""
Tests for MockSupabaseClient (in-memory testing only).
"""
import pytest
from api.db.mock_supabase_client import MockSupabaseClient


def test_insert_run_returns_record_with_id():
    """Test: insert_run() returns record with auto-generated id"""
    client = MockSupabaseClient()
    run_data = {"hypothesis": "Test hypothesis", "success": True}
    result = client.insert_run(run_data)
    
    assert "id" in result
    assert result["hypothesis"] == "Test hypothesis"
    assert result["success"] is True
    assert "created_at" in result


def test_insert_run_stores_retrievable_record():
    """Test: insert_run() stores record retrievable by get_run()"""
    client = MockSupabaseClient()
    run_data = {"hypothesis": "Test hypothesis", "success": True}
    inserted = client.insert_run(run_data)
    
    retrieved = client.get_run(inserted["id"])
    assert retrieved is not None
    assert retrieved["id"] == inserted["id"]
    assert retrieved["hypothesis"] == "Test hypothesis"


def test_get_run_returns_none_for_nonexistent():
    """Test: get_run() returns None for non-existent id"""
    client = MockSupabaseClient()
    result = client.get_run("nonexistent-id")
    assert result is None


def test_update_run_modifies_existing():
    """Test: update_run() modifies existing record"""
    client = MockSupabaseClient()
    run_data = {"hypothesis": "Original", "success": False}
    inserted = client.insert_run(run_data)
    
    updates = {"success": True, "sharpe_ratio": 1.5}
    updated = client.update_run(inserted["id"], updates)
    
    assert updated["success"] is True
    assert updated["sharpe_ratio"] == 1.5
    assert updated["hypothesis"] == "Original"


def test_update_run_returns_empty_for_nonexistent():
    """Test: update_run() returns empty dict for non-existent id"""
    client = MockSupabaseClient()
    result = client.update_run("nonexistent-id", {"success": True})
    assert result == {}


def test_get_leaderboard_returns_only_successful():
    """Test: get_leaderboard() returns only successful runs"""
    client = MockSupabaseClient()
    client.insert_run({"hypothesis": "Success 1", "success": True, "sharpe_ratio": 2.0})
    client.insert_run({"hypothesis": "Failed", "success": False, "sharpe_ratio": 1.0})
    client.insert_run({"hypothesis": "Success 2", "success": True, "sharpe_ratio": 1.5})
    
    leaderboard = client.get_leaderboard()
    assert len(leaderboard) == 2
    assert all(r["success"] for r in leaderboard)


def test_get_leaderboard_respects_limit():
    """Test: get_leaderboard() respects limit parameter"""
    client = MockSupabaseClient()
    for i in range(5):
        client.insert_run({"hypothesis": f"Run {i}", "success": True, "sharpe_ratio": float(i)})
    
    leaderboard = client.get_leaderboard(limit=3)
    assert len(leaderboard) == 3


def test_get_leaderboard_sorts_by_metric_descending():
    """Test: get_leaderboard() sorts by metric descending"""
    client = MockSupabaseClient()
    client.insert_run({"hypothesis": "Low", "success": True, "sharpe_ratio": 1.0})
    client.insert_run({"hypothesis": "High", "success": True, "sharpe_ratio": 3.0})
    client.insert_run({"hypothesis": "Medium", "success": True, "sharpe_ratio": 2.0})
    
    leaderboard = client.get_leaderboard(metric="sharpe_ratio")
    assert leaderboard[0]["sharpe_ratio"] == 3.0
    assert leaderboard[1]["sharpe_ratio"] == 2.0
    assert leaderboard[2]["sharpe_ratio"] == 1.0


def test_get_leaderboard_count_returns_correct():
    """Test: get_leaderboard_count() returns correct count"""
    client = MockSupabaseClient()
    client.insert_run({"hypothesis": "Success 1", "success": True})
    client.insert_run({"hypothesis": "Failed", "success": False})
    client.insert_run({"hypothesis": "Success 2", "success": True})
    
    count = client.get_leaderboard_count()
    assert count == 2


def test_inserting_multiple_runs_all_retrievable():
    """Test: inserting multiple runs all retrievable independently"""
    client = MockSupabaseClient()
    run1 = client.insert_run({"hypothesis": "Run 1", "success": True})
    run2 = client.insert_run({"hypothesis": "Run 2", "success": False})
    run3 = client.insert_run({"hypothesis": "Run 3", "success": True})
    
    retrieved1 = client.get_run(run1["id"])
    retrieved2 = client.get_run(run2["id"])
    retrieved3 = client.get_run(run3["id"])
    
    assert retrieved1["hypothesis"] == "Run 1"
    assert retrieved2["hypothesis"] == "Run 2"
    assert retrieved3["hypothesis"] == "Run 3"
    assert retrieved1["id"] != retrieved2["id"] != retrieved3["id"]
