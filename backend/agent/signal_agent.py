"""
Signal Agent Module for Claude Signal Agent.

Orchestrates the full workflow from hypothesis to backtest results.
"""

import logging
import time
from typing import Optional
import pandas as pd

# Direct imports to avoid circular dependencies
import sys
from pathlib import Path
import importlib.util

# Import PromptBuilder and CodeExecutor
spec_pb = importlib.util.spec_from_file_location(
    "prompt_builder",
    Path(__file__).parent / "prompt_builder.py"
)
prompt_builder_module = importlib.util.module_from_spec(spec_pb)
spec_pb.loader.exec_module(prompt_builder_module)
PromptBuilder = prompt_builder_module.PromptBuilder

spec_ce = importlib.util.spec_from_file_location(
    "code_executor",
    Path(__file__).parent / "code_executor.py"
)
code_executor_module = importlib.util.module_from_spec(spec_ce)
spec_ce.loader.exec_module(code_executor_module)
CodeExecutor = code_executor_module.CodeExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalAgent:
    """
    Orchestrates hypothesis-to-backtest workflow with Claude API.
    """
    
    def __init__(
        self,
        anthropic_client,
        backtest_engine,
        metrics_calculator,
        max_retries: int = 3
    ):
        """
        Initialize SignalAgent with dependencies.
        
        Args:
            anthropic_client: Anthropic API client instance
            backtest_engine: BacktestEngine instance
            metrics_calculator: MetricsCalculator instance
            max_retries: Maximum code generation attempts (default 3)
        """
        self.client = anthropic_client
        self.engine = backtest_engine
        self.metrics = metrics_calculator
        self.max_retries = max_retries
        self.prompt_builder = PromptBuilder()
        self.code_executor = CodeExecutor()
        
        logger.info(f"SignalAgent initialized with max_retries={max_retries}")
    
    def generate_and_backtest(
        self,
        hypothesis: str,
        ohlcv_data: pd.DataFrame
    ) -> dict:
        """
        Generate signal from hypothesis and run backtest.
        
        Args:
            hypothesis: Plain English trading idea
            ohlcv_data: Historical OHLCV data for backtesting
        
        Returns:
            Results dictionary containing:
            - hypothesis: Original hypothesis text
            - generated_code: Final Python code
            - signal_name: Signal class name
            - attempts_taken: Number of generation attempts
            - success: Boolean indicating overall success
            - error: Error message if failed, None if succeeded
            - backtest_results: Full BacktestEngine output (if success)
            - metrics: Full MetricsCalculator output (if success)
        """
        logger.info(f"Starting generate_and_backtest for hypothesis: {hypothesis[:50]}...")
        
        # Build system prompt once
        system_prompt = self.prompt_builder.build_system_prompt()
        
        # Retry loop
        attempt = 1
        previous_code = None
        last_error = None
        signal_instance = None
        generated_code = None
        
        while attempt <= self.max_retries:
            start_time = time.time()
            logger.info(f"Attempt {attempt}/{self.max_retries}")
            
            try:
                # Step 1: Build user prompt
                if attempt == 1:
                    user_prompt = self.prompt_builder.build_user_prompt(hypothesis)
                else:
                    user_prompt = self.prompt_builder.build_user_prompt(
                        hypothesis, last_error, previous_code
                    )
                
                # Step 2: Call Claude API
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    system=system_prompt,
                    tools=[{
                        "name": "write_signal",
                        "description": "Write a Python trading signal class that inherits from SignalBase",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "description": "Complete Python class code"},
                                "explanation": {"type": "string", "description": "Explanation of the signal logic"}
                            },
                            "required": ["code", "explanation"]
                        }
                    }],
                    messages=[{"role": "user", "content": user_prompt}]
                )
                
                # Step 3: Extract code from tool use response
                code = None
                for block in response.content:
                    if block.type == "tool_use" and block.name == "write_signal":
                        code = block.input["code"]
                        break
                
                if code is None:
                    raise ValueError("No code returned from Claude API")
                
                generated_code = code
                
                # Step 4: Execute code
                signal_instance, error = self.code_executor.execute(code, ohlcv_data)
                
                elapsed = time.time() - start_time
                
                # Step 5: Check if successful
                if signal_instance is not None:
                    logger.info(f"Attempt {attempt} succeeded in {elapsed:.2f}s")
                    break
                else:
                    # Step 6: Handle failure
                    logger.warning(f"Attempt {attempt} failed in {elapsed:.2f}s: {error}")
                    last_error = error
                    previous_code = code
                    attempt += 1
                    
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Attempt {attempt} raised exception in {elapsed:.2f}s: {e}")
                last_error = str(e)
                attempt += 1
        
        # Check if we succeeded
        if signal_instance is None:
            # All attempts failed
            logger.error(f"All {self.max_retries} attempts failed")
            return {
                "hypothesis": hypothesis,
                "generated_code": generated_code,
                "signal_name": None,
                "attempts_taken": self.max_retries,
                "success": False,
                "error": last_error,
                "backtest_results": None,
                "metrics": None
            }
        
        # Success - run backtest
        logger.info("Running backtest on generated signal")
        try:
            backtest_results = self.engine.run_backtest(signal_instance, ohlcv_data)
            metrics_results = self.metrics.calculate_metrics(backtest_results['metrics_input'])
            
            logger.info("Backtest completed successfully")
            
            return {
                "hypothesis": hypothesis,
                "generated_code": generated_code,
                "signal_name": signal_instance.name,
                "attempts_taken": attempt,
                "success": True,
                "error": None,
                "backtest_results": backtest_results,
                "metrics": metrics_results
            }
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {
                "hypothesis": hypothesis,
                "generated_code": generated_code,
                "signal_name": signal_instance.name if signal_instance else None,
                "attempts_taken": attempt,
                "success": False,
                "error": f"Backtest failed: {e}",
                "backtest_results": None,
                "metrics": None
            }
