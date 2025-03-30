"""
Error Analyzer for the audio processing multi-agent system.
This module provides error analysis capabilities to interpret execution errors.
"""
import logfire
import traceback
import re
from typing import List, Optional, Dict, Any

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field

from .models import ErrorAnalysisResult
from .dependencies import ErrorAnalysisDependencies


class ErrorAnalysisResponse(BaseModel):
    """Response from the error analyzer."""
    error_type: str
    root_cause: str
    fix_suggestions: List[str] = []
    code_fixes: Optional[str] = None
    requires_replanning: bool = False
    confidence: float
    
    model_config = {}


# Initialize Error Analyzer Agent
error_analyzer_agent = Agent(
    'gemini-2.0-pro',  # Using a more powerful model for better error analysis
    deps_type=ErrorAnalysisDependencies,
    result_type=ErrorAnalysisResponse,
    system_prompt=(
        "You are an expert error analyzer for audio processing code. "
        "Your task is to diagnose execution errors, identify root causes, "
        "and propose fixes or determine if replanning is needed."
    )
)


@error_analyzer_agent.system_prompt
def add_error_context(ctx: RunContext[ErrorAnalysisDependencies]) -> str:
    """Add the error context to the system prompt."""
    execution_result = ctx.deps.execution_result
    plan = ctx.deps.plan
    step_index = ctx.deps.plan_step_index
    
    if 0 <= step_index < len(plan.steps):
        current_step = plan.steps[step_index]
        
        return f"""
Task Description: {plan.task_description}

Current Step ({step_index + 1}/{len(plan.steps)}):
Tool: {current_step.tool_name}
Description: {current_step.description}

Error Message:
{execution_result.error_message}

Code That Generated the Error:
```python
{ctx.deps.generated_code}
```
"""
    else:
        return "Error: Invalid step index."


@error_analyzer_agent.tool
def analyze_error(
    ctx: RunContext[ErrorAnalysisDependencies]
) -> ErrorAnalysisResponse:
    """
    Analyze the execution error to identify the root cause and suggest fixes.
    
    Returns:
        ErrorAnalysisResponse containing the analysis results
    """
    with logfire.span("analyze_error"):
        execution_result = ctx.deps.execution_result
        generated_code = ctx.deps.generated_code
        
        if not execution_result or not execution_result.error_message:
            raise ModelRetry("No error message provided for analysis.")
            
        if not generated_code:
            raise ModelRetry("No code provided for analysis.")
            
        # Preprocess the error message to extract key information
        error_type, error_details = _extract_error_info(execution_result.error_message)
            
        # This is where the model will analyze the error and generate suggestions
        # The model should consider:
        # - The type of error (syntax, runtime, logic, etc.)
        # - The specific line or component that caused the error
        # - Potential fixes based on similar past errors
        # - Whether the error is fixable through code changes or requires replanning
            
        return ErrorAnalysisResponse(
            error_type=error_type,
            root_cause="",  # The model will determine this
            fix_suggestions=[],  # The model will generate these
            code_fixes=None,  # The model may provide fixed code
            requires_replanning=False,  # The model will determine this
            confidence=0.0  # The model will provide a confidence score
        )


@error_analyzer_agent.tool
def generate_code_fix(
    ctx: RunContext[ErrorAnalysisDependencies],
    error_type: str,
    root_cause: str
) -> str:
    """
    Generate fixed code based on the identified error and root cause.
    
    Args:
        error_type: Type of error (e.g., 'SyntaxError', 'TypeError')
        root_cause: Root cause of the error
        
    Returns:
        Fixed version of the code
    """
    with logfire.span("generate_code_fix", error_type=error_type):
        generated_code = ctx.deps.generated_code
        
        if not generated_code:
            raise ModelRetry("No code provided to fix.")
            
        # This is where the model will generate fixed code based on the error analysis
            
        return generated_code  # The model will modify this with fixes


@error_analyzer_agent.tool
def determine_if_replanning_needed(
    ctx: RunContext[ErrorAnalysisDependencies],
    error_type: str,
    fix_attempts: int
) -> bool:
    """
    Determine if the error requires replanning or just code fixes.
    
    Args:
        error_type: Type of error encountered
        fix_attempts: Number of unsuccessful fix attempts
        
    Returns:
        Boolean indicating whether replanning is needed
    """
    with logfire.span("determine_if_replanning_needed", fix_attempts=fix_attempts):
        # Consider both the error type and the number of failed fix attempts
        # Certain errors (like missing files, invalid operations) may require replanning
        # Persistent failures despite fixes may also indicate plan issues
        
        # Error types that typically require replanning
        replanning_error_types = [
            "FileNotFoundError",
            "PermissionError",
            "ValueError",  # If related to fundamental incompatibilities
            "ResourceError",
            "UnsupportedOperationError"
        ]
        
        # Simple heuristic: if error type suggests replanning OR too many failures
        if error_type in replanning_error_types or fix_attempts >= 2:
            return True
            
        return False


def _extract_error_info(error_message: str) -> tuple:
    """Extract the error type and details from an error message."""
    # Try to identify the error type
    error_type_match = re.search(r'((?:[A-Z][a-z]*)+Error|Exception)', error_message)
    error_type = error_type_match.group(0) if error_type_match else "UnknownError"
    
    # The details are everything after the error type
    if error_type_match:
        error_details = error_message[error_type_match.end():].strip()
    else:
        error_details = error_message
        
    return error_type, error_details


@error_analyzer_agent.tool
def categorize_error(
    ctx: RunContext[ErrorAnalysisDependencies],
    error_message: str
) -> Dict[str, Any]:
    """
    Categorize an error message into different types and extract relevant information.
    
    Args:
        error_message: The error message to categorize
        
    Returns:
        Dictionary containing categorized error information
    """
    with logfire.span("categorize_error"):
        result = {
            "error_type": "Unknown",
            "is_syntax_error": False,
            "is_runtime_error": False,
            "is_logic_error": False,
            "is_io_error": False,
            "is_value_error": False,
            "is_type_error": False,
            "related_line": None,
            "function_name": None,
            "variable_name": None,
            "file_path": None
        }
        
        # Check for syntax errors
        if "SyntaxError" in error_message:
            result["error_type"] = "SyntaxError"
            result["is_syntax_error"] = True
            
            # Try to extract line information
            line_match = re.search(r'line (\d+)', error_message)
            if line_match:
                result["related_line"] = int(line_match.group(1))
                
        # Check for common runtime errors
        elif "TypeError" in error_message:
            result["error_type"] = "TypeError"
            result["is_runtime_error"] = True
            result["is_type_error"] = True
            
            # Try to extract function/variable information
            func_match = re.search(r"'([^']+)' object", error_message)
            if func_match:
                result["variable_name"] = func_match.group(1)
                
        elif "ValueError" in error_message:
            result["error_type"] = "ValueError"
            result["is_runtime_error"] = True
            result["is_value_error"] = True
            
        elif "FileNotFoundError" in error_message or "IOError" in error_message:
            result["error_type"] = "IOError"
            result["is_runtime_error"] = True
            result["is_io_error"] = True
            
            # Try to extract file path
            file_match = re.search(r"'([^']+)'", error_message)
            if file_match:
                result["file_path"] = file_match.group(1)
                
        elif "IndexError" in error_message or "KeyError" in error_message:
            result["error_type"] = "AccessError"
            result["is_runtime_error"] = True
            
        # Try to extract line number from traceback
        if "line" in error_message and "related_line" not in result:
            line_match = re.search(r'line (\d+)', error_message)
            if line_match:
                result["related_line"] = int(line_match.group(1))
                
        # Try to extract function name from traceback
        func_match = re.search(r'in ([A-Za-z0-9_]+)', error_message)
        if func_match:
            result["function_name"] = func_match.group(1)
            
        return result 