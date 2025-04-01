"""
Master Control Program (MCP) for code execution.
"""
import ast
import re
import time
import os
import inspect
import traceback
import asyncio
import logfire
from typing import Dict, Callable, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
import torch

from pydantic import BaseModel, Field

from .models import ExecutionResult
from audio_editor import audio_tools


# Configure Logfire for debugging
logfire.configure()


class CodeParsingResult(BaseModel):
    """Result of parsing code."""
    tool_name: str
    kwargs: Dict[str, Union[str, int, float, bool, List[Union[str, int, float, bool]]]] = Field(default_factory=dict)
    is_valid: bool = True
    error_message: str = ""


class MCPCodeExecutor:
    """
    Master Control Program for executing audio processing code.
    Provides a safe environment for executing generated code with proper context.
    """
    def __init__(self, workspace_dir: Path):
        """Initialize the MCP with workspace directory."""
        self.workspace_dir = Path(workspace_dir)
        self.audio_dir = workspace_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        # Keep a mapping of function names to their full signatures
        self.function_signatures = {}
        
        # Get all audio tools and their signatures
        for name, func in inspect.getmembers(audio_tools):
            if inspect.isfunction(func) and name.isupper() and not name.startswith("_"):
                try:
                    sig = str(inspect.signature(func))
                    self.function_signatures[name] = sig
                except ValueError:
                    pass
    
    def _create_execution_context(self) -> dict:
        """Create the execution context with all necessary globals."""
        return {
            'audio_tools': audio_tools,
            'np': np,
            'torch': torch,
            'os': os,
            'Path': Path,
            'logfire': logfire,
            'traceback': traceback,
            'workspace_dir': str(self.workspace_dir),
            'audio_dir': str(self.audio_dir),
            'SAMPLE_RATE': audio_tools.SAMPLE_RATE,
            'AUDIO_QA': getattr(audio_tools, 'AUDIO_QA', None),
            'AUDIO_DIFF': getattr(audio_tools, 'AUDIO_DIFF', None),
            'AUDIO_GENERATE': getattr(audio_tools, 'AUDIO_GENERATE', None),
            **{name: func for name, func in inspect.getmembers(audio_tools) 
               if inspect.isfunction(func) and name.isupper() and not name.startswith("_")}
        }

    def _validate_code(self, code: str) -> Tuple[bool, str, Optional[str]]:
        """Validate the code and extract function name."""
        with logfire.span("validate_code") as span:
            span.set_attribute("code", code)
            
            # Remove any leading/trailing whitespace
            code = code.strip()
            
            if ";" in code or "\n" in code.strip():
                return False, "Only single function calls are allowed.", None
            
            # Look for any UPPERCASE function name
            func_name_match = re.search(r'([A-Z][A-Z_]+)\(', code)
            if not func_name_match:
                return False, "Could not identify function name in code.", None
            
            func_name = func_name_match.group(1)
            
            # If there's content before the function name, try to extract just the function call
            func_start = code.find(func_name)
            if func_start > 0:
                logfire.warning(f"Found potential invalid prefix before function name: '{code[:func_start]}'")
                # Check if what comes after could be a valid function call
                remainder = code[func_start:]
                if "(" in remainder and remainder.count("(") == remainder.count(")"):
                    # This looks like it could be a valid function call with a bad prefix
                    logfire.info(f"Extracted potential function call from code: {remainder}")
                    code = remainder
                    span.set_attribute("cleaned_code", code)
            
            if not hasattr(audio_tools, func_name):
                return False, f"Function '{func_name}' not found in audio_tools.", None
            
            return True, "", func_name

    async def execute_code(self, code: str, description: str) -> ExecutionResult:
        """
        Execute a line of audio processing code.
        
        Args:
            code: Python code to execute (typically a single line function call)
            description: Description of the processing step
            
        Returns:
            ExecutionResult object with execution status and results
        """
        with logfire.span("execute_code") as span:
            start_time = time.time()
            span.set_attribute("original_code", code)
            span.set_attribute("description", description)

            # Validate code
            is_valid, error_msg, func_name = self._validate_code(code)
            if not is_valid:
                return ExecutionResult(
                    status="FAILURE",
                    error_message=error_msg,
                    duration=0.0
                )

            # Fix paths in the code
            code = self._fix_paths_in_code(code)
            span.set_attribute("processed_code", code)

            # Create execution context
            exec_globals = self._create_execution_context()
            
            # Create simpler wrapped code
            wrapped_code = f"""
def _execute():
    logfire.info("Executing function: {func_name}")
    # Escape quotes in the code for debug logging
    debug_code = '''{code}'''
    logfire.debug(f"Full code: {{debug_code}}")
    
    try:
        result = {code}
        
        # Handle different result types
        if isinstance(result, str):
            if os.path.exists(result):
                return {{"status": "SUCCESS", "output_path": result, "result": result}}
            elif result.startswith("Error:"):
                return {{"status": "FAILURE", "error_message": result}}
            else:
                return {{"status": "SUCCESS", "result": result}}
        elif isinstance(result, dict):
            return {{"status": "SUCCESS", **result}}
        else:
            return {{"status": "SUCCESS", "result": result}}
    except Exception as e:
        logfire.error(f"Error in {func_name}: {{str(e)}}", exc_info=True)
        return {{"status": "FAILURE", "error_message": str(e), "error_details": traceback.format_exc()}}

result = _execute()
"""
            span.set_attribute("wrapped_code", wrapped_code)
            
            try:
                # Execute the code
                exec_locals = {}
                exec(wrapped_code, exec_globals, exec_locals)
                result = exec_locals.get('result', {})
                
                duration = time.time() - start_time
                
                if result.get('status') == 'SUCCESS':
                    output = result.get('result')
                    output_path = result.get('output_path')
                    
                    # Log success details
                    logfire.info(f"Successfully executed {func_name}", 
                               extra={"output": str(output)[:100], "output_path": output_path})
                    
                    return ExecutionResult(
                        status="SUCCESS",
                        output=output,
                        output_path=output_path,
                        duration=duration
                    )
                else:
                    error_msg = result.get('error_message', 'Unknown error')
                    error_details = result.get('error_details', '')
                    
                    # Log error details
                    logfire.error(f"Failed to execute {func_name}: {error_msg}",
                                extra={"error_details": error_details})
                    
                    return ExecutionResult(
                        status="FAILURE",
                        error_message=f"{error_msg}\n\n{error_details}",
                        duration=duration
                    )
                    
            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"Error executing code: {str(e)}"
                logfire.error(error_msg, exc_info=True,
                            extra={"code": code, "wrapped_code": wrapped_code})
                return ExecutionResult(
                    status="FAILURE",
                    error_message=error_msg,
                    duration=duration
                )
    
    def _fix_paths_in_code(self, code: str) -> str:
        """Fix file paths in the code to use the workspace directory."""
        with logfire.span("fix_paths_in_code"):
            # Helper function to fix individual paths
            def _fix_path(match):
                path = match.group(1)
                if path.endswith('.wav'):
                    # Audio files should be in the audio subdirectory
                    return f'"{os.path.join(str(self.audio_dir), os.path.basename(path))}"'
                else:
                    # Other files go in workspace root
                    return f'"{os.path.join(str(self.workspace_dir), os.path.basename(path))}"'

            # Fix paths in keyword arguments
            pattern = r'(\w+)=(["\'])(.*?)(["\'])'
            def _fix_kwarg(match):
                kwarg_name = match.group(1)
                quote = match.group(2)
                path = match.group(3)
                if path.endswith('.wav'):
                    # Audio files should be in the audio subdirectory
                    return f'{kwarg_name}={quote}{os.path.join(str(self.audio_dir), os.path.basename(path))}{quote}'
                else:
                    # Other files go in workspace root
                    return f'{kwarg_name}={quote}{os.path.join(str(self.workspace_dir), os.path.basename(path))}{quote}'

            # Fix paths in lists
            def _fix_list_paths(match):
                paths = match.group(1).split(',')
                fixed_paths = []
                for path in paths:
                    path = path.strip().strip('"\'')
                    if path.endswith('.wav'):
                        # Audio files should be in the audio subdirectory
                        fixed_paths.append(f'"{os.path.join(str(self.audio_dir), os.path.basename(path))}"')
                    else:
                        # Other files go in workspace root
                        fixed_paths.append(f'"{os.path.join(str(self.workspace_dir), os.path.basename(path))}"')
                return '[' + ', '.join(fixed_paths) + ']'

            # First fix keyword arguments
            code = re.sub(pattern, _fix_kwarg, code)
            
            # Then fix any remaining paths in lists
            code = re.sub(r'\[((?:"[^"]+\.(?:wav|txt|json)"(?:\s*,\s*)?)+)\]', _fix_list_paths, code)
            code = re.sub(r"\[((?:'[^']+\.(?:wav|txt|json)'(?:\s*,\s*)?)+)\]", _fix_list_paths, code)
            
            logfire.debug(f"Fixed paths in code: {code}")
            return code 