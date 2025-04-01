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
    
    async def execute_code(self, code: str, description: str) -> ExecutionResult:
        """
        Execute a line of audio processing code.
        
        Args:
            code: Python code to execute (typically a single line function call)
            description: Description of the processing step
            
        Returns:
            ExecutionResult object with execution status and results
        """
        with logfire.span("execute_code"):
            start_time = time.time()
            
            # Check if the code contains only one function call (simple validation)
            if ";" in code or "\n" in code.strip():
                return ExecutionResult(
                    status="FAILURE",
                    error_message="Only single function calls are allowed.",
                    duration=0.0
                )
            
            # Extract function name
            func_name_match = re.match(r'([A-Z_]+)\(', code.strip())
            if not func_name_match:
                return ExecutionResult(
                    status="FAILURE",
                    error_message="Could not identify function name in code.",
                    duration=0.0
                )
            
            func_name = func_name_match.group(1)
            
            # Check if function exists in audio_tools
            if not hasattr(audio_tools, func_name):
                return ExecutionResult(
                    status="FAILURE",
                    error_message=f"Function '{func_name}' not found in audio_tools.",
                    duration=0.0
                )
            
            # Create context for execution
            exec_globals = {
                'audio_tools': audio_tools,
                'np': np,
                'torch': torch,
                'os': os,
                'Path': Path,
                'logfire': logfire,
                'workspace_dir': str(self.workspace_dir),
                'audio_dir': str(self.audio_dir),
                'SAMPLE_RATE': audio_tools.SAMPLE_RATE,
                'AUDIO_QA': audio_tools.AUDIO_QA if hasattr(audio_tools, 'AUDIO_QA') else None,
                'AUDIO_DIFF': audio_tools.AUDIO_DIFF if hasattr(audio_tools, 'AUDIO_DIFF') else None,
                'AUDIO_GENERATE': audio_tools.AUDIO_GENERATE if hasattr(audio_tools, 'AUDIO_GENERATE') else None,
            }
            
            # Add all upper-case functions from audio_tools
            for name, func in inspect.getmembers(audio_tools):
                if inspect.isfunction(func) and name.isupper() and not name.startswith("_"):
                    exec_globals[name] = func
            
            # Fix paths in the code
            code = self._fix_paths_in_code(code)
            
            # Wrap the code to capture the output path
            wrapped_code = f"""
def _execute_wrapped():
    try:
        # Log the execution
        logfire.info(f"Executing: {code}")
        
        # Execute the function call
        result = {code}
        
        # Return the result with additional metadata
        if isinstance(result, str) and os.path.exists(result):
            return {{
                "status": "SUCCESS",
                "output_path": result,
                "result": result,
                "message": f"Successfully executed {func_name}. Output file: {{result}}"
            }}
        elif isinstance(result, dict) and "output_path" in result:
            return {{
                "status": "SUCCESS",
                "output_path": result["output_path"],
                "result": result,
                "message": f"Successfully executed {func_name}. Output file: {{result['output_path']}}"
            }}
        elif isinstance(result, str) and result.startswith("Error:"):
            return {{
                "status": "FAILURE",
                "error_message": result,
                "result": None,
                "message": f"Error executing {func_name}: {{result}}"
            }}
        elif func_name in ["AUDIO_QA", "AUDIO_DIFF"]:
            # Special handling for analysis functions that return text
            return {{
                "status": "SUCCESS",
                "output_path": None,  # No output file
                "result": result,
                "message": f"Successfully executed {func_name}. Analysis result: {{result[:100]}}..."
            }}
        else:
            return {{
                "status": "SUCCESS",
                "output_path": str(result) if result is not None else None,
                "result": result,
                "message": f"Successfully executed {func_name}. Result: {{result}}"
            }}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {{
            "status": "FAILURE",
            "error_message": str(e),
            "error_details": error_details,
            "result": None,
            "message": f"Error executing {func_name}: {{str(e)}}"
        }}

execution_result = _execute_wrapped()
"""
            
            # Execute the code
            try:
                exec_locals = {}
                exec(wrapped_code, exec_globals, exec_locals)
                result = exec_locals.get('execution_result', {})
                
                # Extract results
                status = result.get('status', 'FAILURE')
                output_path = result.get('output_path', None)
                message = result.get('message', 'No message provided')
                error_message = result.get('error_message', None)
                
                duration = time.time() - start_time
                
                if status == 'SUCCESS':
                    # For analysis functions, return the actual text in the output
                    if func_name in ["AUDIO_QA", "AUDIO_DIFF"]:
                        actual_result = result.get('result', '')
                        return ExecutionResult(
                            status=status,
                            output=actual_result,  # Return the actual analysis text
                            output_path=None,  # No output file for analysis functions
                            error_message=None,
                            duration=duration
                        )
                    else:
                        # For processing functions, return the output path
                        return ExecutionResult(
                            status=status,
                            output=message,
                            output_path=output_path,
                            error_message=None,
                            duration=duration
                        )
                else:
                    error_details = result.get('error_details', '')
                    formatted_error = f"{error_message}\n\n{error_details}" if error_details else error_message
                    return ExecutionResult(
                        status=status,
                        output=None,
                        error_message=formatted_error or "Unknown error occurred",
                        duration=duration
                    )
                    
            except Exception as e:
                duration = time.time() - start_time
                logfire.error(f"Error executing code: {e}", exc_info=True)
                return ExecutionResult(
                    status="FAILURE",
                    error_message=f"Error executing code: {str(e)}",
                    duration=duration
                )
    
    def _fix_paths_in_code(self, code: str) -> str:
        """Fix file paths in the code to use the audio directory."""
        # Match parameters that look like file paths
        path_pattern = r'(["\'])([^"\'\(\)]+\.wav)(["\'])'
        
        def _fix_path(match):
            quote = match.group(1)
            path = match.group(2)
            
            # If it's already an absolute path, leave it as is
            if os.path.isabs(path):
                return match.group(0)
                
            # Otherwise, consider it relative to the audio directory
            return f'{quote}{path}{quote}'
            
        # Replace paths in the code
        fixed_code = re.sub(path_pattern, _fix_path, code)
        
        # Special handling for AUDIO_QA, AUDIO_DIFF, and AUDIO_GENERATE
        if "AUDIO_QA(" in fixed_code or "AUDIO_DIFF(" in fixed_code or "AUDIO_GENERATE(" in fixed_code:
            # These functions need full paths, so modify any wav_path or similar arguments
            # to prepend the audio directory
            wav_path_pattern = r'(wav_path\s*=\s*)(["\'])([^"\'\(\)]+\.wav)(["\'])'
            file_pattern = r'(file\s*=\s*)(["\'])([^"\'\(\)]+\.wav)(["\'])'
            filename_pattern = r'(filename\s*=\s*)(["\'])([^"\'\(\)]+\.wav)(["\'])'
            
            def _fix_full_path(match):
                prefix = match.group(1)
                quote = match.group(2)
                path = match.group(3)
                quote_end = match.group(4)
                
                # If it's already an absolute path, leave it as is
                if os.path.isabs(path):
                    return match.group(0)
                    
                # Otherwise, prepend the audio directory
                return f'{prefix}{quote}' + str(self.audio_dir / path) + f'{quote_end}'
                
            # Replace paths in the code for these specific functions
            fixed_code = re.sub(wav_path_pattern, _fix_full_path, fixed_code)
            fixed_code = re.sub(file_pattern, _fix_full_path, fixed_code)
            fixed_code = re.sub(filename_pattern, _fix_full_path, fixed_code)
            
            # Handle lists of paths for AUDIO_DIFF
            if "AUDIO_DIFF(" in fixed_code:
                list_pattern = r'(\[)(["\'][^"\'\(\)]+\.wav["\'](?:\s*,\s*["\'][^"\'\(\)]+\.wav["\'])+)(\])'
                
                def _fix_list_paths(match):
                    prefix = match.group(1)
                    paths_str = match.group(2)
                    suffix = match.group(3)
                    
                    # Split by commas and handle each path
                    path_parts = paths_str.split(',')
                    fixed_parts = []
                    
                    for part in path_parts:
                        # Extract the path from quotes
                        path_match = re.search(r'(["\'])([^"\'\(\)]+\.wav)(["\'])', part.strip())
                        if path_match:
                            quote = path_match.group(1)
                            path = path_match.group(2)
                            
                            # If it's already an absolute path, leave it as is
                            if os.path.isabs(path):
                                fixed_parts.append(part.strip())
                            else:
                                # Otherwise, prepend the audio directory
                                fixed_parts.append(f'{quote}' + str(self.audio_dir / path) + f'{quote}')
                        else:
                            fixed_parts.append(part.strip())
                            
                    return prefix + ', '.join(fixed_parts) + suffix
                    
                fixed_code = re.sub(list_pattern, _fix_list_paths, fixed_code)
        
        return fixed_code 