"""
Prompt Builder Module for Claude Signal Agent.

Constructs system and user prompts for Claude API and extracts code from responses.
"""

import re
from typing import Optional


class PromptBuilder:
    """
    Constructs prompts for Claude API and extracts code from responses.
    """
    
    def build_system_prompt(self) -> str:
        """
        Build system prompt with SignalBase interface and instructions.
        
        Returns:
            System prompt string containing role description, SignalBase interface,
            example implementation, and rules.
        """
        return """You are a quantitative signal writer for a trading system. Your role is to write Python trading signals as SignalBase subclasses.

**SignalBase Interface:**

All signals must inherit from SignalBase and implement the generate_signals method:

```python
from backtester.signal_base import SignalBase
import pandas as pd
import numpy as np

class YourSignal(SignalBase):
    def generate_signals(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        # Your signal logic here
        pass
```

**Method Signature:**
- Input: `ohlcv_data` is a DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'] and a DatetimeIndex
- Output: Must return a pd.Series with:
  - DatetimeIndex matching the input data
  - Values must be only {-1, 0, 1}:
    - 1 = Long position
    - 0 = Neutral (no position)
    - -1 = Short position
  - No NaN values allowed

**Example Implementation:**

```python
from backtester.signal_base import SignalBase
import pandas as pd
import numpy as np

class MomentumSignal(SignalBase):
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        super().__init__(lookback_period=lookback_period)
    
    def generate_signals(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        # Calculate N-day returns
        returns = ohlcv_data['Close'].pct_change(periods=self.lookback_period)
        
        # Generate signals
        signals = pd.Series(0, index=ohlcv_data.index, dtype=np.int8)
        signals[returns > 0] = 1
        signals[returns < 0] = -1
        
        # First N days will have NaN returns, set to neutral
        signals.iloc[:self.lookback_period] = 0
        
        return signals
```

**Rules:**
1. Must inherit from SignalBase
2. Must return pd.Series with DatetimeIndex matching input data
3. Signal values must be only {-1, 0, 1}
4. No NaN values in output
5. Available libraries: pandas, numpy only - no other imports allowed

Write clean, efficient signal code that implements the given hypothesis."""
    
    def build_user_prompt(
        self,
        hypothesis: str,
        error_feedback: Optional[str] = None,
        previous_code: Optional[str] = None
    ) -> str:
        """
        Build user prompt with hypothesis and optional error feedback.
        
        Args:
            hypothesis: Plain English trading idea
            error_feedback: Optional error message from previous attempt
            previous_code: Optional code from previous attempt
        
        Returns:
            User prompt string
        """
        if error_feedback is None or previous_code is None:
            # First attempt - just the hypothesis
            return f"Generate a SignalBase implementation for this hypothesis:\n\n{hypothesis}"
        
        # Retry attempt - include previous code and error
        return f"""Generate a SignalBase implementation for this hypothesis:

{hypothesis}

**Previous attempt failed with this error:**

{error_feedback}

**Previous code:**

```python
{previous_code}
```

Please fix the error and generate corrected code."""
    
    def extract_code(self, response_text: str) -> str:
        """
        Extract Python code from Claude's response text.
        
        Args:
            response_text: Claude API response text
        
        Returns:
            Extracted Python code string (stripped of markdown)
        
        Raises:
            ValueError: If no code block found in response
        """
        # Match markdown code blocks with optional language identifier
        pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if not matches:
            raise ValueError("No Python code block found in response")
        
        # Return first code block, stripped of whitespace
        return matches[0].strip()
