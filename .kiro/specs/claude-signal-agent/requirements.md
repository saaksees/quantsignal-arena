# Requirements Document

## Introduction

The Claude Signal Agent is an AI-powered trading signal generator that converts plain English investment hypotheses into executable Python trading signals. This Month 2 feature builds on top of the Month 1 Backtesting Engine, enabling users to describe trading ideas in natural language and automatically generate, validate, and backtest SignalBase implementations through Claude's API.

## Glossary

- **Signal_Agent**: The orchestration component that manages the full hypothesis-to-backtest workflow
- **Prompt_Builder**: Component that constructs system and user prompts for Claude API
- **Code_Executor**: Sandboxed execution environment that runs generated signal code safely
- **Hypothesis**: Plain English description of a trading idea (e.g., "momentum in small-cap tech stocks after earnings surprise")
- **SignalBase**: Abstract base class from Month 1 that all trading signals must inherit from
- **BacktestEngine**: Month 1 component that executes vectorized backtests
- **MetricsCalculator**: Month 1 component that computes performance metrics
- **Retry_Loop**: Error feedback mechanism where Claude receives execution errors and attempts fixes
- **Tool_Use**: Claude API feature for structured function calling with defined schemas
- **Sandbox**: Restricted subprocess environment that prevents dangerous imports and enforces timeouts
- **OHLCV_Data**: Open, High, Low, Close, Volume price data for financial instruments
- **SecurityError**: Exception raised when code attempts forbidden imports

## Requirements

### Requirement 1: System Prompt Construction

**User Story:** As a signal agent, I want to provide Claude with clear instructions about its role and the SignalBase interface, so that it generates valid signal implementations.

#### Acceptance Criteria

1. THE Prompt_Builder SHALL generate a system prompt that describes Claude's role as a quantitative signal writer
2. THE Prompt_Builder SHALL include the SignalBase interface specification in the system prompt
3. THE Prompt_Builder SHALL include an example SignalBase subclass implementation in the system prompt
4. THE Prompt_Builder SHALL specify that signals must return pd.Series with only {-1, 0, 1} values
5. THE Prompt_Builder SHALL specify that signals must have a datetime index matching input data
6. THE Prompt_Builder SHALL specify that signals must inherit from SignalBase
7. THE Prompt_Builder SHALL include available pandas and numpy operations in the system prompt

### Requirement 2: User Prompt Construction

**User Story:** As a signal agent, I want to construct user prompts that include the hypothesis and any error feedback, so that Claude can generate or fix signal code.

#### Acceptance Criteria

1. WHEN no error feedback is provided, THE Prompt_Builder SHALL create a user prompt containing only the hypothesis
2. WHEN error feedback is provided, THE Prompt_Builder SHALL include the previous code in the user prompt
3. WHEN error feedback is provided, THE Prompt_Builder SHALL include the exact error message in the user prompt
4. THE Prompt_Builder SHALL format error feedback to clearly distinguish it from the hypothesis
5. THE Prompt_Builder SHALL preserve the original hypothesis in retry prompts

### Requirement 3: Code Extraction from Claude Response

**User Story:** As a signal agent, I want to extract Python code from Claude's response text, so that I can execute the generated signal.

#### Acceptance Criteria

1. THE Prompt_Builder SHALL extract code from markdown code blocks in Claude's response
2. THE Prompt_Builder SHALL return only the Python code without markdown formatting
3. WHEN no code block is found in the response, THE Prompt_Builder SHALL raise a ValueError
4. THE Prompt_Builder SHALL handle multiple code blocks by extracting the first Python code block
5. THE Prompt_Builder SHALL strip leading and trailing whitespace from extracted code

### Requirement 4: Sandboxed Code Execution

**User Story:** As a system operator, I want generated code to run in a restricted environment, so that malicious or buggy code cannot harm the system.

#### Acceptance Criteria

1. THE Code_Executor SHALL run generated code in a subprocess with a 10-second timeout
2. WHEN code execution exceeds 10 seconds, THE Code_Executor SHALL terminate the process and return a timeout error
3. THE Code_Executor SHALL catch SyntaxError and return a descriptive error message
4. THE Code_Executor SHALL catch ImportError and return a descriptive error message
5. THE Code_Executor SHALL catch runtime exceptions and return the exception message
6. THE Code_Executor SHALL catch signal validation errors from SignalBase and return the error message
7. THE Code_Executor SHALL accept OHLCV_Data as input to pass to the signal

### Requirement 5: Security Restrictions

**User Story:** As a system operator, I want to prevent dangerous imports in generated code, so that the system remains secure.

#### Acceptance Criteria

1. WHEN code attempts to import os, THE Code_Executor SHALL raise a SecurityError
2. WHEN code attempts to import sys, THE Code_Executor SHALL raise a SecurityError
3. WHEN code attempts to import subprocess, THE Code_Executor SHALL raise a SecurityError
4. WHEN code attempts to import socket, THE Code_Executor SHALL raise a SecurityError
5. WHEN code attempts to import requests, THE Code_Executor SHALL raise a SecurityError
6. THE Code_Executor SHALL allow imports of pandas, numpy, and backtester modules
7. THE Code_Executor SHALL return SecurityError messages that specify which import was forbidden

### Requirement 6: Successful Code Execution

**User Story:** As a signal agent, I want to receive an instantiated SignalBase object when code executes successfully, so that I can pass it to the BacktestEngine.

#### Acceptance Criteria

1. WHEN code executes without errors, THE Code_Executor SHALL return a tuple (signal_instance, None)
2. THE Code_Executor SHALL instantiate the signal class defined in the generated code
3. THE Code_Executor SHALL validate that the instantiated object is a SignalBase subclass
4. THE Code_Executor SHALL return the signal instance ready for use with BacktestEngine
5. WHEN code executes with errors, THE Code_Executor SHALL return a tuple (None, error_message)

### Requirement 7: Claude API Integration

**User Story:** As a signal agent, I want to call Claude's API with structured tool use, so that I receive well-formatted signal code.

#### Acceptance Criteria

1. THE Signal_Agent SHALL use the claude-sonnet-4-20250514 model
2. THE Signal_Agent SHALL define a tool called write_signal with input schema {code: string, explanation: string}
3. THE Signal_Agent SHALL send the system prompt and user prompt to Claude API
4. THE Signal_Agent SHALL extract the code parameter from Claude's tool use response
5. THE Signal_Agent SHALL handle API errors and return them as execution failures
6. THE Signal_Agent SHALL accept an anthropic_client instance in its constructor

### Requirement 8: Retry Loop with Error Feedback

**User Story:** As a signal agent, I want to send execution errors back to Claude for up to 3 attempts, so that Claude can fix issues in the generated code.

#### Acceptance Criteria

1. THE Signal_Agent SHALL attempt code generation and execution up to 3 times
2. WHEN the first attempt fails, THE Signal_Agent SHALL send the error message to Claude for a second attempt
3. WHEN the second attempt fails, THE Signal_Agent SHALL send the error message to Claude for a third attempt
4. WHEN the third attempt fails, THE Signal_Agent SHALL return success=False in the results
5. WHEN any attempt succeeds, THE Signal_Agent SHALL stop retrying and proceed to backtesting
6. THE Signal_Agent SHALL track the number of attempts taken in the results dictionary

### Requirement 9: Backtest Integration

**User Story:** As a signal agent, I want to automatically run the BacktestEngine on successfully generated signals, so that users receive complete results.

#### Acceptance Criteria

1. WHEN code execution succeeds, THE Signal_Agent SHALL pass the signal instance to BacktestEngine
2. THE Signal_Agent SHALL pass the OHLCV_Data to BacktestEngine.run_backtest
3. THE Signal_Agent SHALL receive backtest results from BacktestEngine
4. THE Signal_Agent SHALL pass backtest returns to MetricsCalculator
5. THE Signal_Agent SHALL receive metrics from MetricsCalculator
6. THE Signal_Agent SHALL accept a BacktestEngine instance in its constructor

### Requirement 10: Results Dictionary Structure

**User Story:** As a user, I want to receive a comprehensive results dictionary, so that I can analyze the generated signal and its performance.

#### Acceptance Criteria

1. THE Signal_Agent SHALL return a dictionary containing the key "hypothesis" with the original hypothesis text
2. THE Signal_Agent SHALL return a dictionary containing the key "generated_code" with the final code
3. THE Signal_Agent SHALL return a dictionary containing the key "signal_name" with the signal class name
4. THE Signal_Agent SHALL return a dictionary containing the key "attempts_taken" with the number of attempts
5. THE Signal_Agent SHALL return a dictionary containing the key "backtest_results" with full BacktestEngine output
6. THE Signal_Agent SHALL return a dictionary containing the key "metrics" with full MetricsCalculator output
7. THE Signal_Agent SHALL return a dictionary containing the key "success" as a boolean
8. WHEN execution fails after max retries, THE Signal_Agent SHALL include the key "error" with the final error message
9. WHEN execution succeeds, THE Signal_Agent SHALL set "error" to None

### Requirement 11: Logging and Observability

**User Story:** As a system operator, I want detailed logs of each attempt, so that I can debug issues and monitor agent performance.

#### Acceptance Criteria

1. THE Signal_Agent SHALL log each attempt with the attempt number
2. THE Signal_Agent SHALL log the time taken for each attempt
3. WHEN an attempt fails, THE Signal_Agent SHALL log the error message
4. WHEN an attempt succeeds, THE Signal_Agent SHALL log the signal name
5. THE Signal_Agent SHALL log the start of the backtest execution
6. THE Signal_Agent SHALL log the completion of the backtest with key metrics
7. THE Signal_Agent SHALL use Python's logging module with INFO level

### Requirement 12: Main Orchestration Method

**User Story:** As a user, I want a single method that handles the entire workflow, so that I can generate and backtest signals with one function call.

#### Acceptance Criteria

1. THE Signal_Agent SHALL expose a generate_and_backtest method that accepts hypothesis and OHLCV_Data
2. THE Signal_Agent SHALL execute the full loop: prompt → Claude API → extract → execute → retry → backtest
3. THE Signal_Agent SHALL return the results dictionary from generate_and_backtest
4. THE Signal_Agent SHALL handle all exceptions and return them in the results dictionary
5. THE Signal_Agent SHALL accept max_retries as a constructor parameter with default value 3

