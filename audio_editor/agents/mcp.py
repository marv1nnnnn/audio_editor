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
    """Master Control Program for executing Python code in a controlled environment."""
    
    def __init__(self, workspace_dir: str):
        """Initialize the MCP Python executor.
        
        Args:
            workspace_dir: Directory to use as working directory for execution
        """
        with logfire.span("mcp_init"):
            self.workspace_dir = os.path.abspath(workspace_dir)
            self.tools = self._gather_tools()
            logfire.info(f"MCP initialized with {len(self.tools)} audio tools in workspace {self.workspace_dir}")
    
    def _gather_tools(self) -> Dict[str, Callable]:
        """Gather all available tools from the audio_tools module."""
        with logfire.span("gather_tools"):
            tools = {}
            for name, func in inspect.getmembers(audio_tools):
                # Tools are uppercase functions
                if inspect.isfunction(func) and name.isupper() and not name.startswith("_"):
                    tools[name] = func
                    logfire.debug(f"Added tool: {name}")
            return tools
    
    def _prepare_execution_environment(self) -> Dict[str, Union[Callable, object]]:
        """Prepare the execution environment with available tools."""
        with logfire.span("prepare_execution_environment"):
            # Set up a clean environment with just the audio tools
            environment = self.tools.copy()
            
            # Add basic modules/functions needed for execution
            environment['os'] = os
            
            return environment
    
    def _parse_code(self, code_string: str) -> CodeParsingResult:
        """
        Parse the code to extract the tool name and arguments.
        
        Args:
            code_string: String containing the Python code to execute
            
        Returns:
            CodeParsingResult with tool_name, kwargs, and validation info
        """
        with logfire.span("parse_code", code=code_string):
            try:
                # Parse the code into an AST
                parsed = ast.parse(code_string.strip())
                
                # We expect a simple tool call like TOOL_NAME(arg1="value", arg2=123)
                if not parsed.body or not isinstance(parsed.body[0], ast.Expr):
                    return CodeParsingResult(
                        tool_name="",
                        kwargs={},
                        is_valid=False,
                        error_message="Code must contain a single tool call expression"
                    )
                
                expr = parsed.body[0].value
                if not isinstance(expr, ast.Call):
                    return CodeParsingResult(
                        tool_name="",
                        kwargs={},
                        is_valid=False,
                        error_message="Expression must be a function call"
                    )
                
                # Extract function name
                if isinstance(expr.func, ast.Name):
                    func_name = expr.func.id
                else:
                    return CodeParsingResult(
                        tool_name="",
                        kwargs={},
                        is_valid=False,
                        error_message="Function call must use a simple name"
                    )
                
                # Extract arguments
                kwargs = {}
                
                # Handle positional arguments first
                if expr.args:
                    # Get the function signature to map positional args to parameter names
                    if func_name in self.tools:
                        sig = inspect.signature(self.tools[func_name])
                        param_names = list(sig.parameters.keys())
                        
                        # Map positional args to parameter names
                        for i, arg in enumerate(expr.args):
                            if i < len(param_names):
                                param_name = param_names[i]
                                if isinstance(arg, ast.Constant):
                                    kwargs[param_name] = arg.value
                                elif isinstance(arg, (ast.Num, ast.NameConstant)):
                                    kwargs[param_name] = arg.n if hasattr(arg, 'n') else arg.value
                                elif isinstance(arg, ast.List):
                                    items = []
                                    for elt in arg.elts:
                                        if isinstance(elt, ast.Constant):
                                            items.append(elt.value)
                                        else:
                                            items.append(None)
                                    kwargs[param_name] = items
                                else:
                                    logfire.warning(f"Unsupported argument type: {type(arg)} for parameter {param_name}")
                    else:
                        logfire.warning(f"Tool {func_name} not found, cannot map positional arguments")
                
                # Handle keyword arguments
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
                                items.append(None)
                        kwargs[kw.arg] = items
                    else:
                        # Log unsupported argument types
                        logfire.warning(f"Unsupported argument type: {type(kw.value)} for argument {kw.arg}")
                
                return CodeParsingResult(
                    tool_name=func_name,
                    kwargs=kwargs,
                    is_valid=True
                )
                
            except SyntaxError as e:
                logfire.error(f"Syntax error in code: {str(e)}")
                return CodeParsingResult(
                    tool_name="",
                    kwargs={},
                    is_valid=False,
                    error_message=f"Syntax error in code: {str(e)}"
                )
            except Exception as e:
                logfire.error(f"Error parsing code: {str(e)}")
                return CodeParsingResult(
                    tool_name="",
                    kwargs={},
                    is_valid=False,
                    error_message=f"Error parsing code: {str(e)}"
                )
    
    async def execute_code(self, code_string: str, description: str = "Code execution") -> ExecutionResult:
        """
        Execute the provided Python code in a controlled environment.
        
        Args:
            code_string: String containing the Python code to execute
            description: Description of the code being executed (for logging)
            
        Returns:
            ExecutionResult with status, output, and error message if any
        """
        with logfire.span("execute_code", description=description):
            start_time = time.time()
            
            try:
                # Parse the code to extract tool name and arguments
                logfire.info(f"Parsing code: {code_string}")
                parsing_result = self._parse_code(code_string)
                
                if not parsing_result.is_valid:
                    logfire.error(f"Invalid code: {parsing_result.error_message}")
                    return ExecutionResult(
                        status="FAILURE",
                        error_message=parsing_result.error_message,
                        duration=time.time() - start_time
                    )
                
                func_name = parsing_result.tool_name
                kwargs = parsing_result.kwargs
                
                # Check if the tool exists
                if func_name not in self.tools:
                    logfire.error(f"Tool not found: {func_name}")
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
                logfire.info(f"Execution completed with result: {result}")
                
                # Process the result
                output_path = None
                output_paths = None
                
                if isinstance(result, str) and os.path.exists(result):
                    # Tool returned a single file path
                    output_path = os.path.abspath(result)
                    logfire.debug(f"Result is a file path: {output_path}")
                elif isinstance(result, list) and result and all(isinstance(item, str) and os.path.exists(item) for item in result):
                    # Tool returned multiple file paths
                    output_paths = [os.path.abspath(p) for p in result]
                    output_path = output_paths[0]  # Use first as primary
                    logfire.debug(f"Result is multiple file paths. Primary: {output_path}")
                
                return ExecutionResult(
                    status="SUCCESS",
                    output=str(result),
                    duration=time.time() - start_time,
                    output_path=output_path,
                    output_paths=output_paths
                )
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                logfire.error(f"Error executing code: {error_msg}")
                return ExecutionResult(
                    status="FAILURE",
                    error_message=error_msg,
                    duration=time.time() - start_time
                ) 