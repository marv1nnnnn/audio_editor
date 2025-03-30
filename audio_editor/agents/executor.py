"""
Execution Agent for the audio processing multi-agent system.
"""
import logfire
import ast
import inspect
from typing import Dict, Any, Optional

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field

from .models import ExecutorOutput, AudioPlan, PlanStep, StepStatus
from .dependencies import ExecutorDependencies


class CodeGenerationResponse(BaseModel):
    """Response from the code generator."""
    generated_code: str
    expected_output_path: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {"additionalProperties": True}
    }


# Initialize Execution Agent
executor_agent = Agent(
    "gemini-2.0-flash",
    deps_type=ExecutorDependencies,
    result_type=CodeGenerationResponse,
    system_prompt=(
        "You are an expert audio processing code generator. "
        "Your task is to generate Python code to process audio files based on the provided plan step. "
        "Focus on writing clean, efficient code that uses the available audio processing tools."
    )
)


@executor_agent.system_prompt
def add_tool_definitions(ctx: RunContext[ExecutorDependencies]) -> str:
    """Add the available tools to the system prompt."""
    tool_definitions = ctx.deps.tool_definitions
    tools_str = "\n\n".join([
        f"Tool: {tool.name}\n"
        f"Signature: {tool.signature}\n"
        f"Description: {tool.docstring}"
        for tool in tool_definitions
    ])
    
    return f"""
Available Tools:
{tools_str}
"""


@executor_agent.system_prompt
def add_plan_context(ctx: RunContext[ExecutorDependencies]) -> str:
    """Add the current plan context to the system prompt."""
    plan = ctx.deps.plan
    step_index = ctx.deps.plan_step_index
    
    if 0 <= step_index < len(plan.steps):
        current_step = plan.steps[step_index]
        
        return f"""
Current Task: {plan.task_description}

Current Step ({step_index + 1}/{len(plan.steps)}):
Tool: {current_step.tool_name}
Description: {current_step.description}
Status: {current_step.status}

Current Audio Path: {plan.current_audio_path}
"""
    else:
        return "Error: Invalid step index."


@executor_agent.tool
def generate_code_for_step(
    ctx: RunContext[ExecutorDependencies]
) -> CodeGenerationResponse:
    """
    Generate Python code to execute the current plan step.
    
    Returns:
        CodeGenerationResponse containing the generated code
    """
    with logfire.span("generate_code_for_step"):
        plan = ctx.deps.plan
        step_index = ctx.deps.plan_step_index
        
        if step_index >= len(plan.steps):
            raise ModelRetry("Invalid step index. Please generate code for a valid step in the plan.")
        
        current_step = plan.steps[step_index]
        
        # Generate the code based on the tool and its parameters
        context = {
            "tool_name": current_step.tool_name,
            "tool_args": current_step.tool_args,
            "description": current_step.description,
            "audio_path": plan.current_audio_path
        }
        
        # Find the tool signature to help with code generation
        tool_signature = None
        for tool_def in ctx.deps.tool_definitions:
            if tool_def.name == current_step.tool_name:
                tool_signature = tool_def.signature
                break
        
        return CodeGenerationResponse(
            generated_code=f"{current_step.tool_name}{tool_signature if tool_signature else '(...)'}"
        )


@executor_agent.tool
def refine_code_after_error(
    ctx: RunContext[ExecutorDependencies],
    code: str,
    error_message: str
) -> CodeGenerationResponse:
    """
    Refine the generated code after an execution error.
    
    Args:
        code: The code that failed to execute
        error_message: The error message from the failed execution
        
    Returns:
        Updated code to fix the error
    """
    with logfire.span("refine_code_after_error"):
        # Get the current retry count from dependencies
        execution_result = ctx.deps.execution_result
        retry_limit = ctx.deps.retry_limit
        
        # Track retry count
        retry_count = 1
        if execution_result and hasattr(execution_result, "retry_count"):
            retry_count = execution_result.retry_count + 1
            
        # Check if we've reached the retry limit
        persistent_failure = retry_count >= retry_limit
        
        if persistent_failure:
            return CodeGenerationResponse(
                generated_code=code,
                persistent_failure=True,
                retry_count=retry_count
            )
            
        # Generate refined code based on the error
        return CodeGenerationResponse(
            generated_code=code,  # This would be refined by the model based on error
            persistent_failure=False,
            retry_count=retry_count
        )


@executor_agent.tool
def validate_code(ctx: RunContext[ExecutorDependencies], code: str) -> str:
    """
    Validate the generated code to ensure it uses valid tools and parameters.
    
    Args:
        code: The code to validate
        
    Returns:
        Error message if invalid, or the code if valid
    """
    with logfire.span("validate_code"):
        try:
            # Parse the code to check for syntax errors
            ast.parse(code)
            
            # Additional validation could be added here
            
            return code
        except SyntaxError as e:
            raise ModelRetry(f"The generated code has syntax errors: {str(e)}")
        except Exception as e:
            raise ModelRetry(f"The generated code is invalid: {str(e)}") 