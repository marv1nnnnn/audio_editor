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
from typing import Dict, Any, Callable, Optional, List, Tuple

from .models import ExecutionResult
import audio_tools


class MCPCodeExecutor:
    """Master Control Program for executing Python code in a controlled environment."""
    
    def __init__(self, workspace_dir: str):
        """Initialize the MCP Python executor.
        
        Args:
            workspace_dir: Directory to use as working directory for execution
        """
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.tools = self._gather_tools()
        logfire.info(f"MCP initialized with {len(self.tools)} audio tools in workspace {self.workspace_dir}")
    
    def _gather_tools(self) -> Dict[str, Callable]:
        """Gather all available tools from the audio_tools module."""
        tools = {}
        for name, func in inspect.getmembers(audio_tools):
            # Tools are uppercase functions
            if inspect.isfunction(func) and name.isupper() and not name.startswith("_"):
                tools[name] = func
        return tools
    
    def _prepare_execution_environment(self) -> Dict[str, Any]:
        """Prepare the execution environment with available tools."""
        # Set up a clean environment with just the audio tools
        environment = self.tools.copy()
        
        # Add basic modules/functions needed for execution
        environment['os'] = os
        
        return environment
    
    def _parse_code(self, code_string: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse the code to extract the tool name and arguments.
        
        Args:
            code_string: String containing the Python code to execute
            
        Returns:
            Tuple of (tool_name, tool_args)
            
        Raises:
            SyntaxError: If the code has syntax errors
            ValueError: If the code doesn't contain a valid tool call
        """
        try:
            # Parse the code into an AST
            parsed = ast.parse(code_string.strip())
            
            # We expect a simple tool call like TOOL_NAME(arg1="value", arg2=123)
            if not parsed.body or not isinstance(parsed.body[0], ast.Expr):
                raise ValueError("Code must contain a single tool call expression")
            
            expr = parsed.body[0].value
            if not isinstance(expr, ast.Call):
                raise ValueError("Expression must be a function call")
            
            # Extract function name
            if isinstance(expr.func, ast.Name):
                func_name = expr.func.id
            else:
                raise ValueError("Function call must use a simple name")
            
            # Extract arguments
            kwargs = {}
            for kw in expr.keywords:
                # For string literals
                if isinstance(kw.value, ast.Constant):
                    kwargs[kw.arg] = kw.value.value
                # For numeric literals, booleans, etc.
                elif isinstance(kw.value, (ast.Num, ast.NameConstant)):
                    kwargs[kw.arg] = kw.value.n if hasattr(kw.value, 'n') else kw.value.value
                # For lists
                elif isinstance(kw.value, ast.List):
                    items = []
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant):
                            items.append(elt.value)
                        else:
                            # Simplified handling - could be extended for more complex cases
                            items.append(None)
                    kwargs[kw.arg] = items
                else:
                    # Skip arguments we can't directly interpret
                    pass
            
            return func_name, kwargs
            
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in code: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing code: {str(e)}")
    
    async def execute_code(self, code_string: str) -> ExecutionResult:
        """
        Execute the provided Python code in a controlled environment.
        
        Args:
            code_string: String containing the Python code to execute
            
        Returns:
            ExecutionResult with status, output, and error message if any
        """
        start_time = time.time()
        
        try:
            with logfire.span("mcp_execute_code", code=code_string):
                # Parse the code to extract tool name and arguments
                func_name, kwargs = self._parse_code(code_string)
                
                # Check if the tool exists
                if func_name not in self.tools:
                    return ExecutionResult(
                        status="FAILURE",
                        error_message=f"Tool '{func_name}' not found",
                        duration=time.time() - start_time
                    )
                
                # Get the tool function
                tool_func = self.tools[func_name]
                
                # Execute the tool function
                logfire.info(f"Executing {func_name} with args: {kwargs}")
                result = tool_func(**kwargs)
                
                # Process the result
                output_path = None
                output_paths = None
                
                if isinstance(result, str) and os.path.exists(result):
                    # Tool returned a single file path
                    output_path = os.path.abspath(result)
                elif isinstance(result, list) and result and all(isinstance(item, str) and os.path.exists(item) for item in result):
                    # Tool returned multiple file paths
                    output_paths = [os.path.abspath(p) for p in result]
                    output_path = output_paths[0]  # Use first as primary
                
                return ExecutionResult(
                    status="SUCCESS",
                    output=str(result),
                    duration=time.time() - start_time,
                    output_path=output_path,
                    output_paths=output_paths
                )
                
        except Exception as e:
            logfire.error(f"Error executing code: {str(e)}", exc_info=True)
            return ExecutionResult(
                status="FAILURE",
                error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                duration=time.time() - start_time
            ) 