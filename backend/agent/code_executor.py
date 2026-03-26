"""
Code Executor Module for Claude Signal Agent.

Executes generated signal code in a sandboxed environment with security restrictions.
"""

import re
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Direct import to avoid circular dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "signal_base",
    Path(__file__).parent.parent / "backtester" / "signal_base.py"
)
signal_base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signal_base_module)
SignalBase = signal_base_module.SignalBase


class SecurityError(Exception):
    """Raised when code attempts forbidden imports."""
    pass


class CodeExecutor:
    """
    Executes generated signal code in a sandboxed environment.
    """
    
    FORBIDDEN_IMPORTS = {'os', 'sys', 'subprocess', 'socket', 'requests'}
    TIMEOUT_SECONDS = 10
    
    def execute(
        self,
        code: str,
        ohlcv_data: pd.DataFrame
    ) -> tuple[Optional[SignalBase], Optional[str]]:
        """
        Execute code in sandboxed environment and return signal instance.
        
        Args:
            code: Python code string to execute
            ohlcv_data: OHLCV DataFrame to pass to signal
        
        Returns:
            Tuple of (signal_instance, error_message):
            - Success: (SignalBase instance, None)
            - Failure: (None, error message string)
        """
        # Step 1: Security scan
        try:
            self._check_security(code)
        except SecurityError as e:
            return None, str(e)
        
        try:
            # Step 2: Execute with restricted globals
            allowed_globals = {
                "__builtins__": {
                    "__build_class__": __build_class__,
                    "__name__": __name__,
                    "isinstance": isinstance,
                    "issubclass": issubclass,
                    "hasattr": hasattr,
                    "getattr": getattr,
                    "len": len,
                    "range": range,
                    "list": list,
                    "dict": dict,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "None": None,
                    "True": True,
                    "False": False,
                    "print": print,
                    "Exception": Exception,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "NotImplementedError": NotImplementedError,
                },
                "pd": pd,
                "np": np,
                "SignalBase": SignalBase
            }
            local_namespace = {}
            
            exec(code, allowed_globals, local_namespace)
            
            # Step 3: Find the generated class
            signal_class = None
            for name, obj in local_namespace.items():
                if isinstance(obj, type) and obj.__name__ != "SignalBase" and any(
                    c.__name__ == "SignalBase" for c in obj.__mro__
                ):
                    signal_class = obj
                    break
            
            if signal_class is None:
                return None, "No SignalBase subclass found in generated code"
            
            # Step 4: Instantiate and validate
            signal_instance = signal_class()
            
            # Check by class name and MRO instead of isinstance
            base_class_names = [c.__name__ for c in type(signal_instance).__mro__]
            if "SignalBase" not in base_class_names:
                return None, "Generated class is not a SignalBase subclass"
            
            # Validate by calling generate_signals once - this will trigger SignalBase validation
            # Use the __call__ method which includes validation
            signal_instance(ohlcv_data)
            
            return signal_instance, None
            
        except SyntaxError as e:
            return None, f"SyntaxError: {e}"
        except ImportError as e:
            return None, f"ImportError: {e}"
        except NameError as e:
            return None, f"NameError: {e}"
        except TypeError as e:
            return None, f"TypeError: {e}"
        except Exception as e:
            return None, str(e)
    
    def _check_security(self, code: str) -> None:
        """
        Check code for forbidden imports.
        
        Args:
            code: Python code string
        
        Raises:
            SecurityError: If forbidden import found
        """
        for module in self.FORBIDDEN_IMPORTS:
            # Pattern matches: import os, from os, import os.path, from os.path
            pattern = rf'\b(?:import|from)\s+{module}\b'
            if re.search(pattern, code):
                raise SecurityError(f"Forbidden import detected: {module}")
