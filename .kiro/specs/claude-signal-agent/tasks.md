# Implementation Plan: Claude Signal Agent

## Overview

This implementation creates an AI-powered trading signal generator that converts natural language hypotheses into executable Python trading signals. The system consists of three core components: PromptBuilder (constructs prompts for Claude API), CodeExecutor (sandboxed execution environment), and SignalAgent (orchestrates the full workflow with retry logic and backtest integration).

## Tasks

- [ ] 1. Set up project structure and PromptBuilder class
  - Create `backend/agent/prompt_builder.py` with PromptBuilder class
  - Implement `build_system_prompt()` method with SignalBase interface specification
  - Implement `build_user_prompt()` method with hypothesis and error feedback handling
  - Implement `extract_code()` method to parse Claude API responses
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ]* 1.1 Write property test for system prompt completeness
  - **Property 1: System Prompt Completeness**
  - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7**

- [ ]* 1.2 Write property test for hypothesis preservation
  - **Property 2: Hypothesis Preservation in Retry Prompts**
  - **Validates: Requirements 2.5**

- [ ]* 1.3 Write property test for code extraction
  - **Property 3: Code Extraction from Markdown**
  - **Validates: Requirements 3.1, 3.2**

- [ ]* 1.4 Write property test for whitespace stripping
  - **Property 4: Whitespace Stripping in Extraction**
  - **Validates: Requirements 3.5**

- [ ]* 1.5 Write unit tests for PromptBuilder
  - Test system prompt contains required sections
  - Test user prompt with no error feedback
  - Test user prompt with error feedback
  - Test code extraction with no code block raises ValueError
  - Test code extraction with multiple blocks takes first
  - _Requirements: 2.1, 2.2, 2.3, 3.3, 3.4_

- [ ] 2. Implement CodeExecutor with sandboxed execution
  - Create `backend/agent/code_executor.py` with CodeExecutor class
  - Implement security check for forbidden imports (os, sys, subprocess, socket, requests)
  - Implement subprocess execution with 10-second timeout
  - Implement comprehensive error handling (SyntaxError, ImportError, RuntimeError, ValidationError, TimeoutError)
  - Implement signal instantiation and validation
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ]* 2.1 Write property test for error handling completeness
  - **Property 5: Error Handling Completeness**
  - **Validates: Requirements 4.3, 4.4, 4.5, 4.6**

- [ ]* 2.2 Write property test for forbidden import detection
  - **Property 6: Forbidden Import Detection**
  - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.7**

- [ ]* 2.3 Write property test for allowed import acceptance
  - **Property 7: Allowed Import Acceptance**
  - **Validates: Requirements 5.6**

- [ ]* 2.4 Write property test for successful execution return format
  - **Property 8: Successful Execution Return Format**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

- [ ]* 2.5 Write property test for failed execution return format
  - **Property 9: Failed Execution Return Format**
  - **Validates: Requirements 6.5**

- [ ]* 2.6 Write unit tests for CodeExecutor
  - Test timeout enforcement at 10 seconds
  - Test each forbidden import raises SecurityError
  - Test OHLCV data is passed to signal
  - Test allowed imports work correctly
  - _Requirements: 4.2, 4.7, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement SignalAgent orchestration
  - Create `backend/agent/signal_agent.py` with SignalAgent class
  - Implement constructor accepting anthropic_client, backtest_engine, metrics_calculator, max_retries
  - Implement `_call_claude_api()` method with tool_use schema for write_signal
  - Implement `_retry_loop()` method with error feedback and max 3 attempts
  - Implement logging at INFO level for all major steps
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 12.5_

- [ ]* 4.1 Write property test for Claude API response extraction
  - **Property 10: Claude API Response Extraction**
  - **Validates: Requirements 7.4**

- [ ]* 4.2 Write property test for API error handling
  - **Property 11: API Error Handling**
  - **Validates: Requirements 7.5**

- [ ]* 4.3 Write property test for retry loop termination
  - **Property 12: Retry Loop Termination on Success**
  - **Validates: Requirements 8.5, 8.6**

- [ ]* 4.4 Write unit tests for SignalAgent initialization and API
  - Test constructor accepts required dependencies
  - Test uses claude-sonnet-4-20250514 model
  - Test tool schema matches specification
  - Test system and user prompts sent to API
  - Test first attempt failure triggers second attempt
  - Test second attempt failure triggers third attempt
  - Test third attempt failure returns success=False
  - Test logging uses INFO level
  - _Requirements: 7.1, 7.2, 7.3, 7.6, 8.2, 8.3, 8.4, 9.6, 11.7, 12.5_

- [ ] 5. Implement generate_and_backtest workflow
  - Implement `generate_and_backtest()` method in SignalAgent
  - Wire retry loop to backtest execution on success
  - Implement results dictionary construction with all required keys
  - Implement exception handling to catch all errors and return in results
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 12.1, 12.2, 12.3, 12.4_

- [ ]* 5.1 Write property test for backtest integration
  - **Property 13: Backtest Integration on Success**
  - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

- [ ]* 5.2 Write property test for results dictionary completeness
  - **Property 14: Results Dictionary Completeness**
  - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9**

- [ ]* 5.3 Write property test for hypothesis preservation in results
  - **Property 15: Hypothesis Preservation in Results**
  - **Validates: Requirements 10.1**

- [ ]* 5.4 Write property test for error field consistency
  - **Property 16: Error Field Consistency**
  - **Validates: Requirements 10.8, 10.9**

- [ ]* 5.5 Write property test for logging completeness
  - **Property 17: Logging Completeness**
  - **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6**

- [ ]* 5.6 Write property test for exception safety
  - **Property 18: Exception Safety**
  - **Validates: Requirements 12.4**

- [ ]* 5.7 Write property test for workflow completeness
  - **Property 19: Workflow Completeness**
  - **Validates: Requirements 12.2**

- [ ] 6. Create comprehensive test suite
  - Create `backend/tests/test_agent.py` with all unit tests
  - Add integration tests for end-to-end workflow with mocked Claude API
  - Add test for successful signal generation and backtest
  - Add test for failed signal generation after max retries
  - Add test for backtest integration with Month 1 components
  - _Requirements: All requirements validated through comprehensive testing_

- [ ] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties across randomized inputs
- Unit tests validate specific examples and edge cases
- The implementation builds on Month 1 BacktestEngine and MetricsCalculator components
- All generated code must inherit from SignalBase and return signals in {-1, 0, 1}
- Security sandbox prevents dangerous imports (os, sys, subprocess, socket, requests)
- Retry loop provides up to 3 attempts with error feedback to Claude API
