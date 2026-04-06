"""
Signal generation endpoint that wires everything together.
"""
import os
import sys
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.middleware.auth import require_auth
from api.models.schemas import GenerateSignalRequest, SignalRunResponse, MetricsResponse, SHAPResponse, DriftResponse, PaperTradeRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def get_db():
    if os.getenv("TESTING"):
        from api.db.mock_supabase_client import MockSupabaseClient
        return MockSupabaseClient()
    from api.db.supabase_client import SupabaseClient
    return SupabaseClient()


@router.post("/generate")
async def generate_signal(
    request: GenerateSignalRequest,
    current_user: dict = Depends(require_auth)
):
    async def event_generator():
        try:
            # Import heavy dependencies only when needed
            import yfinance as yf
            import anthropic
            from agent.signal_agent import SignalAgent
            from backtester.engine import BacktestEngine
            from backtester.metrics import MetricsCalculator
            from shap_layer.explainer import SignalExplainer
            from shap_layer.drift_detector import DriftDetector
            from shap_layer.report_builder import ReportBuilder
            
            # Step 1: Load data
            yield {"data": json.dumps({"step": "loading_data", "message": "Downloading market data..."})}
            
            import pandas as pd
            all_data = {}
            for ticker in request.tickers:
                df = yf.download(ticker, start=request.start_date, end=request.end_date, progress=False)
                if df.empty:
                    yield {"data": json.dumps({"step": "error", "message": f"No data found for {ticker}"})}
                    return
                all_data[ticker] = df
            
            ohlcv_data = all_data[request.tickers[0]]
            
            # Step 2: Generate signal
            yield {"data": json.dumps({"step": "generating_signal", "message": "Claude is writing your signal..."})}
            
            anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            engine = BacktestEngine()
            metrics_calc = MetricsCalculator()
            agent = SignalAgent(anthropic_client, engine, metrics_calc)
            
            agent_results = agent.generate_and_backtest(request.hypothesis, ohlcv_data)
            
            if not agent_results["success"]:
                yield {"data": json.dumps({"step": "error", "message": agent_results.get("error", "Signal generation failed")})}
                return
            
            # Step 3: SHAP explanation
            yield {"data": json.dumps({"step": "explaining", "message": "Analysing what drives this signal..."})}
            
            explainer = SignalExplainer()
            shap_results = explainer.explain(agent_results["signal_instance"], ohlcv_data)
            top_features = explainer.get_top_features(shap_results, n=3)
            
            # Step 4: Drift detection
            yield {"data": json.dumps({"step": "drift_check", "message": "Checking signal stability..."})}
            
            drift_detector = DriftDetector()
            drift_results = drift_detector.detect(agent_results["signal_instance"], ohlcv_data)
            
            # Step 5: Generate report
            yield {"data": json.dumps({"step": "report", "message": "Generating tearsheet PDF..."})}
            
            report_builder = ReportBuilder()
            report_path = report_builder.build(
                hypothesis=request.hypothesis,
                backtest_results=agent_results["backtest_results"],
                metrics=agent_results["metrics"],
                shap_results=shap_results,
                drift_results=drift_results,
                generated_code=agent_results.get("generated_code", "")
            )
            
            # Step 6: Save to Supabase
            yield {"data": json.dumps({"step": "saving", "message": "Saving results..."})}
            
            metrics = agent_results["metrics"] or {}
            db = get_db()
            run_data = {
                "user_id": current_user["user_id"],
                "hypothesis": request.hypothesis,
                "tickers": request.tickers,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "generated_code": agent_results.get("generated_code"),
                "signal_name": agent_results.get("signal_name"),
                "attempts_taken": agent_results.get("attempts_taken"),
                "success": True,
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sortino_ratio": metrics.get("sortino_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "cagr": metrics.get("cagr"),
                "win_rate": metrics.get("win_rate"),
                "total_return": metrics.get("total_return"),
                "volatility": metrics.get("volatility"),
                "shap_summary": shap_results.get("summary"),
                "shap_feature_importance": shap_results.get("feature_importance"),
                "drift_level": drift_results.get("drift_level"),
                "signal_psi": drift_results.get("signal_psi"),
                "return_psi": drift_results.get("return_psi"),
                "report_path": report_path,
            }
            
            saved_run = db.insert_run(run_data)
            run_id = saved_run.get("id")
            
            # Step 7: Done
            yield {"data": json.dumps({
                "step": "complete",
                "run_id": run_id,
                "metrics": metrics,
                "shap_summary": shap_results.get("summary"),
                "drift_level": drift_results.get("drift_level"),
                "report_url": f"/api/reports/{run_id}",
                "generated_code": agent_results.get("generated_code")
            })}
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            yield {"data": json.dumps({"step": "error", "message": str(e)})}
    
    return EventSourceResponse(event_generator())


@router.get("/{run_id}")
async def get_signal_run(
    run_id: str,
    current_user: dict = Depends(require_auth)
):
    db = get_db()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/{run_id}/paper_trade")
async def toggle_paper_trade(
    run_id: str,
    body: PaperTradeRequest,
    current_user: dict = Depends(require_auth)
):
    db = get_db()
    updated = db.update_run(run_id, {"is_paper_trading": body.active})
    if not updated:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "is_paper_trading": body.active}
