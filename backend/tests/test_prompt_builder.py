"""
Tests for PromptBuilder module.
"""

import pytest
from backend.agent.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """Test suite for PromptBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = PromptBuilder()
    
    def test_build_system_prompt_contains_signalbase(self):
        """Test that system prompt contains 'SignalBase'."""
        prompt = self.builder.build_system_prompt()
        assert "SignalBase" in prompt
    
    def test_build_system_prompt_contains_generate_signals(self):
        """Test that system prompt contains 'generate_signals'."""
        prompt = self.builder.build_system_prompt()
        assert "generate_signals" in prompt
    
    def test_build_system_prompt_contains_valid_values(self):
        """Test that system prompt contains signal value constraints."""
        prompt = self.builder.build_system_prompt()
        # Check for either "{-1, 0, 1}" or "1, -1, 0" or similar variations
        assert ("{-1, 0, 1}" in prompt or 
                "{1, 0, -1}" in prompt or
                "{0, 1, -1}" in prompt or
                ("1" in prompt and "-1" in prompt and "0" in prompt))
    
    def test_build_user_prompt_no_error_contains_hypothesis(self):
        """Test that user prompt with no error contains the hypothesis text."""
        hypothesis = "Buy when momentum is positive"
        prompt = self.builder.build_user_prompt(hypothesis)
        assert hypothesis in prompt
    
    def test_build_user_prompt_with_error_contains_all_parts(self):
        """Test that user prompt with error feedback contains hypothesis, error, and code."""
        hypothesis = "Buy when momentum is positive"
        error = "SyntaxError: invalid syntax"
        code = "class MySignal(SignalBase):\n    pass"
        
        prompt = self.builder.build_user_prompt(hypothesis, error, code)
        
        assert hypothesis in prompt
        assert error in prompt
        assert code in prompt
    
    def test_extract_code_from_markdown_block(self):
        """Test that extract_code correctly extracts code from markdown block."""
        response = """Here's the signal:

```python
class MySignal(SignalBase):
    def generate_signals(self, ohlcv_data):
        return pd.Series(1, index=ohlcv_data.index)
```

This implements the hypothesis."""
        
        code = self.builder.extract_code(response)
        
        assert "class MySignal(SignalBase):" in code
        assert "def generate_signals" in code
        assert "```" not in code
        assert "python" not in code
    
    def test_extract_code_raises_valueerror_when_no_block(self):
        """Test that extract_code raises ValueError when no code block present."""
        response = "This is just text without any code blocks."
        
        with pytest.raises(ValueError) as exc_info:
            self.builder.extract_code(response)
        
        assert "No Python code block found in response" in str(exc_info.value)
    
    def test_extract_code_returns_first_block_when_multiple(self):
        """Test that extract_code returns first block when multiple blocks exist."""
        response = """Here are two options:

```python
class FirstSignal(SignalBase):
    pass
```

Or this one:

```python
class SecondSignal(SignalBase):
    pass
```
"""
        
        code = self.builder.extract_code(response)
        
        assert "FirstSignal" in code
        assert "SecondSignal" not in code
    
    def test_extracted_code_has_no_leading_trailing_whitespace(self):
        """Test that extracted code has no leading or trailing whitespace."""
        response = """```python

class MySignal(SignalBase):
    pass

```"""
        
        code = self.builder.extract_code(response)
        
        # Check no leading whitespace
        assert code[0] != ' ' and code[0] != '\n' and code[0] != '\t'
        # Check no trailing whitespace
        assert code[-1] != ' ' and code[-1] != '\n' and code[-1] != '\t'
    
    def test_extract_code_without_language_identifier(self):
        """Test that extract_code works with code blocks without 'python' identifier."""
        response = """```
class MySignal(SignalBase):
    pass
```"""
        
        code = self.builder.extract_code(response)
        
        assert "class MySignal(SignalBase):" in code
