"""
Critique Agent for the audio processing multi-agent system.
This agent reviews plans and code for quality, efficiency, and correctness.
"""
import logfire
from typing import List, Optional
import os

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field

from .models import CritiqueResult
from .dependencies import CritiqueAgentDependencies


class CritiqueResponse(BaseModel):
    """Response from the critique agent."""
    is_approved: bool
    suggestions: List[str] = []
    improved_version: Optional[str] = None
    reasoning: str
    critique_type: str


# Initialize Critique Agent - using a more powerful model for deep analysis
critique_agent = Agent(
    'gemini-2.0-flash',  # Using a more powerful model for deep analysis
    deps_type=CritiqueAgentDependencies,
    result_type=CritiqueResponse,
    system_prompt=(
        "You are an expert reviewer for audio processing plans and code. "
        "Your task is to analyze plans and code for quality, correctness, and efficiency. "
        "Provide detailed feedback and suggestions for improvement."
    )
)


@critique_agent.system_prompt
def add_tool_definitions(ctx: RunContext[CritiqueAgentDependencies]) -> str:
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


@critique_agent.system_prompt
def add_critique_context(ctx: RunContext[CritiqueAgentDependencies]) -> str:
    """Add the context based on what's being critiqued."""
    critique_type = ctx.deps.critique_type
    
    if critique_type == "plan" and ctx.deps.plan:
        plan = ctx.deps.plan
        return f"""
Critique Type: Audio Processing Plan
Task Description: {ctx.deps.task_description or plan.task_description}

The plan consists of {len(plan.steps)} steps.
Your job is to assess whether this plan is:
1. Complete - Does it accomplish the task?
2. Efficient - Are there redundant or unnecessary steps?
3. Logical - Is the sequence of steps appropriate?
4. Robust - Does it handle potential errors or edge cases?
"""
    elif critique_type == "code" and ctx.deps.generated_code and ctx.deps.plan:
        plan = ctx.deps.plan
        step_index = ctx.deps.plan_step_index
        step = plan.steps[step_index] if 0 <= step_index < len(plan.steps) else None
        
        return f"""
Critique Type: Generated Python Code
Task Description: {ctx.deps.task_description or plan.task_description}
Step Description: {step.description if step else "Unknown step"}

Your job is to assess whether this code:
1. Correctly implements the intended functionality
2. Uses the appropriate tools and parameters
3. Handles potential errors gracefully
4. Is efficient and follows best practices
5. Will achieve the desired audio processing result
"""
    else:
        return "Error: Invalid critique context. Please provide a plan or code to critique."


@critique_agent.tool
def critique_plan(
    ctx: RunContext[CritiqueAgentDependencies]
) -> CritiqueResponse:
    """
    Critique an audio processing plan for quality, completeness, and efficiency.
    
    Returns:
        CritiqueResponse containing assessment and suggestions
    """
    with logfire.span("critique_plan"):
        plan = ctx.deps.plan
        
        if not plan:
            logfire.error("No plan provided for critique")
            raise ModelRetry("No plan provided for critique. Please provide a plan to evaluate.")
        
        # Log plan details for debugging
        logfire.debug(f"Critiquing plan:")
        logfire.debug(f"  - Task: {plan.task_description}")
        logfire.debug(f"  - Current audio path: {plan.current_audio_path}")
        logfire.debug(f"  - Number of steps: {len(plan.steps)}")
        for i, step in enumerate(plan.steps):
            logfire.debug(f"  - Step {i+1}: {step.description}")
            logfire.debug(f"    Tool: {step.tool_name}")
            logfire.debug(f"    Status: {step.status}")
            logfire.debug(f"    Args: {step.tool_args}")
        
        # Verify audio file exists
        if not os.path.exists(plan.current_audio_path):
            logfire.error(f"Audio file not found: {plan.current_audio_path}")
            logfire.debug("Working directory contents:")
            for file in os.listdir(os.path.dirname(plan.current_audio_path)):
                logfire.debug(f"  - {file}")
            raise FileNotFoundError(f"Audio file not found: {plan.current_audio_path}")
            
        # This is where the model will generate its critique based on the plan
        # The system will evaluate the plan for:
        # - Missing steps
        # - Inefficient orderings
        # - Potential errors or edge cases
        # - Overall quality and effectiveness
            
        return CritiqueResponse(
            is_approved=True,  # This will be determined by the model's assessment
            suggestions=[],  # The model will populate this
            improved_version=None,  # The model may provide an improved plan if needed
            reasoning="",  # The model will provide reasoning
            critique_type="plan"
        )


@critique_agent.tool
def critique_code(
    ctx: RunContext[CritiqueAgentDependencies]
) -> CritiqueResponse:
    """
    Critique generated Python code for correctness, efficiency, and best practices.
    
    Returns:
        CritiqueResponse containing assessment and suggestions
    """
    with logfire.span("critique_code"):
        code = ctx.deps.generated_code
        plan = ctx.deps.plan
        step_index = ctx.deps.plan_step_index
        
        if not code:
            logfire.error("No code provided for critique")
            raise ModelRetry("No code provided for critique. Please provide code to evaluate.")
            
        if not plan or step_index is None or step_index < 0 or step_index >= len(plan.steps):
            logfire.error(f"Invalid plan or step index: plan={bool(plan)}, step_index={step_index}")
            raise ModelRetry("Invalid plan or step index for code critique.")
        
        # Log code details for debugging
        logfire.debug(f"Critiquing code for step {step_index + 1}:")
        logfire.debug(f"  - Task: {plan.task_description}")
        logfire.debug(f"  - Step description: {plan.steps[step_index].description}")
        logfire.debug(f"  - Tool: {plan.steps[step_index].tool_name}")
        logfire.debug(f"  - Code length: {len(code)} characters")
        logfire.debug(f"  - Code preview (first 200 chars):\n{code[:200]}...")
        
        # This is where the model will generate its critique based on the code
        # The system will evaluate the code for:
        # - Correctness (does it properly implement the step)
        # - Efficiency (any performance concerns)
        # - Error handling (does it handle exceptions appropriately)
        # - Best practices (follows Python conventions)
            
        return CritiqueResponse(
            is_approved=True,  # This will be determined by the model's assessment
            suggestions=[],  # The model will populate this
            improved_version=None,  # The model may provide improved code if needed
            reasoning="",  # The model will provide reasoning
            critique_type="code"
        )


@critique_agent.tool
def suggest_improvements(
    ctx: RunContext[CritiqueAgentDependencies],
    critique_type: str,
    current_content: str,
    issues: List[str]
) -> str:
    """
    Generate an improved version based on identified issues.
    
    Args:
        critique_type: Either "plan" or "code"
        current_content: The current plan or code
        issues: List of identified issues
        
    Returns:
        Improved version of the plan or code
    """
    with logfire.span("suggest_improvements", critique_type=critique_type):
        if critique_type not in ["plan", "code"]:
            logfire.error(f"Invalid critique type: {critique_type}")
            raise ModelRetry("Invalid critique type. Must be either 'plan' or 'code'.")
            
        # Log improvement request details
        logfire.debug(f"Suggesting improvements:")
        logfire.debug(f"  - Type: {critique_type}")
        logfire.debug(f"  - Issues: {issues}")
        logfire.debug(f"  - Content length: {len(current_content)} characters")
        logfire.debug(f"  - Content preview (first 200 chars):\n{current_content[:200]}...")
            
        # This is where the model will generate an improved version
        # based on the issues identified in the critique
            
        return current_content  # The model will modify this with improvements 