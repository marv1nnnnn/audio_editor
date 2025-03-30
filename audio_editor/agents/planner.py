"""
Planner Agent for the audio processing multi-agent system.
"""
import logfire
from typing import Optional, List, Dict, Any, TypeVar, Union
import json

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field, ConfigDict

from .models import AudioPlan, StepStatus, PlanStep, ExecutionResult, ErrorAnalysisResult
from .dependencies import PlannerDependencies


class PlannerResponse(BaseModel):
    """Response from the planner agent."""
    updated_plan: AudioPlan = Field(
        description="The updated audio processing plan"
    )
    replanning_needed: bool = Field(
        default=False,
        description="Whether replanning is needed"
    )
    checkpoint_index: Optional[int] = Field(
        default=None,
        description="Index of the last checkpoint"
    )
    
    model_config = ConfigDict(extra='forbid')
    # Remove json_schema_extra, Pydantic generates schema from fields


# Initialize Planner Agent
planner_agent = Agent(
    "gemini-2.0-flash",  # Using Gemini model
    deps_type=PlannerDependencies,
    result_type=PlannerResponse,
    system_prompt=(
        "You are an expert audio processing planner. "
        "Your task is to create and maintain a detailed plan for audio processing tasks. "
        "You should break down complex tasks into simple steps that can be executed with Python code. "
        "For each step, provide a clear description and the expected output."
    )
)


@planner_agent.system_prompt
def add_tool_definitions(ctx: RunContext[PlannerDependencies]) -> str:
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


@planner_agent.system_prompt
def add_task_description(ctx: RunContext[PlannerDependencies]) -> str:
    """Add the task description to the system prompt."""
    return f"""
Task Description:
{ctx.deps.task_description}
"""


@planner_agent.tool
def generate_initial_plan(
    ctx: RunContext[PlannerDependencies], 
    task_description: str,
    current_audio_path: str
) -> PlannerResponse:
    """
    Generate the initial step-by-step plan based on the task description.
    
    Args:
        task_description: Description of the audio processing task
        current_audio_path: Path to the input audio file
        
    Returns:
        A PlannerResponse containing the initial plan
    """
    with logfire.span("generate_initial_plan", task=task_description):
        # Create an initial plan with an AUDIO_QA first step
        steps = []
        
        # Always start with audio analysis
        analysis_step = PlanStep(
            description=f"Analyze audio to understand its properties for the task: {task_description}",
            status=StepStatus.PENDING,
            tool_name="AUDIO_QA",
            tool_args=json.dumps({
                "wav_path": current_audio_path,
                "task": f"Analyze this audio to understand its properties for the task: {task_description}"
            })
        )
        steps.append(analysis_step)
        
        initial_plan = AudioPlan(
            task_description=task_description,
            steps=steps,
            current_audio_path=current_audio_path,
            completed_step_indices=[],
            is_complete=False,
            checkpoint_indices=[]
        )
        
        return PlannerResponse(
            updated_plan=initial_plan,
            replanning_needed=False,
            checkpoint_index=None
        )


@planner_agent.tool
def update_plan_after_execution(
    ctx: RunContext[PlannerDependencies],
    plan: AudioPlan,
    step_index: int,
    execution_result: ExecutionResult
) -> PlannerResponse:
    """
    Update the plan based on the execution result of a step.
    
    Args:
        plan: Current audio processing plan
        step_index: Index of the step that was executed
        execution_result: Result of the step execution
        
    Returns:
        An updated plan with the next step to execute
    """
    with logfire.span("update_plan_after_execution", step_index=step_index):
        updated_plan = plan.model_copy(deep=True)  # Make a deep copy
        
        # Update the status of the current step
        if execution_result.status == "SUCCESS":
            updated_plan.steps[step_index].status = StepStatus.DONE
            updated_plan.completed_step_indices.append(step_index)
            
            # Set as checkpoint if this is a major step
            checkpoint_index = step_index
        else:
            updated_plan.steps[step_index].status = StepStatus.FAILED
            checkpoint_index = None
            
        # Find the next step to execute
        next_step_index = None
        for i, step in enumerate(updated_plan.steps):
            if step.status == StepStatus.PENDING and i not in updated_plan.completed_step_indices:
                next_step_index = i
                break
                
        # Check if the plan is complete
        is_complete = all(step.status != StepStatus.PENDING for step in updated_plan.steps)
        updated_plan.is_complete = is_complete
        
        return PlannerResponse(
            updated_plan=updated_plan,
            replanning_needed=False,
            checkpoint_index=checkpoint_index
        )


@planner_agent.tool
def replan_from_checkpoint(
    ctx: RunContext[PlannerDependencies],
    plan: AudioPlan,
    checkpoint_index: int
) -> PlannerResponse:
    """
    Generate a revised plan starting from the checkpoint.
    
    Args:
        plan: Current audio processing plan
        checkpoint_index: Index of the last successful checkpoint
        
    Returns:
        A revised plan with new steps starting from the checkpoint
    """
    with logfire.span("replan_from_checkpoint", checkpoint_index=checkpoint_index):
        updated_plan = plan.model_copy(deep=True)  # Make a deep copy
        
        # Preserve all steps up to and including the checkpoint
        preserved_steps = updated_plan.steps[:checkpoint_index + 1]
        
        # Update the plan
        updated_plan.steps = preserved_steps
        updated_plan.is_complete = False
        
        return PlannerResponse(
            updated_plan=updated_plan,
            replanning_needed=False,
            checkpoint_index=None
        ) 