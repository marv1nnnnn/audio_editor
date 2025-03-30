"""
Planner Agent for the audio processing multi-agent system.
"""
import logfire
from typing import Optional, List, Dict, Any, TypeVar, Union

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field

from .models import AudioPlan, PlannerOutput, StepStatus, PlanStep, ExecutionResult, ErrorAnalysisResult
from .dependencies import PlannerDependencies


class PlannerResponse(BaseModel):
    """Response from the planner agent."""
    updated_plan: AudioPlan
    replanning_needed: bool = False
    checkpoint_index: Optional[int] = None
    
    model_config = {
        "json_schema_extra": {"additionalProperties": True}
    }


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
) -> AudioPlan:
    """
    Generate the initial step-by-step plan based on the task description.
    
    Args:
        task_description: Description of the audio processing task
        current_audio_path: Path to the input audio file
        
    Returns:
        An AudioPlan with a sequence of PlanStep objects
    """
    with logfire.span("generate_initial_plan", task=task_description):
        # Create an initial plan with an AUDIO_QA first step
        steps = []
        
        # Always start with audio analysis
        analysis_step = PlanStep(
            description=f"Analyze audio to understand its properties for the task: {task_description}",
            status=StepStatus.PENDING,
            tool_name="AUDIO_QA",
            tool_args={
                "wav_path": current_audio_path,
                "task": f"Analyze this audio to understand its properties for the task: {task_description}"
            }
        )
        steps.append(analysis_step)
        
        return AudioPlan(
            task_description=task_description,
            steps=steps,
            current_audio_path=current_audio_path
        )


@planner_agent.tool
def update_plan_after_execution(
    ctx: RunContext[PlannerDependencies],
    plan: AudioPlan,
    step_index: int,
    execution_result: dict
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
        updated_plan = plan.model_copy(deep=True)
        
        # Update the status of the current step
        if execution_result["status"] == "SUCCESS":
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
        
        return PlannerResponse(
            updated_plan=updated_plan,
            next_step_index=next_step_index,
            is_complete=is_complete,
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
        updated_plan = plan.model_copy(deep=True)
        
        # Preserve all steps up to and including the checkpoint
        preserved_steps = updated_plan.steps[:checkpoint_index + 1]
        
        # Create a new AudioPlan with only the preserved steps
        updated_plan.steps = preserved_steps
        
        # Find the next step to execute (if any preserved steps are still pending)
        next_step_index = None
        for i, step in enumerate(updated_plan.steps):
            if step.status == StepStatus.PENDING and i not in updated_plan.completed_step_indices:
                next_step_index = i
                break
                
        return PlannerResponse(
            updated_plan=updated_plan,
            next_step_index=next_step_index,
            is_complete=False,
            replanning_needed=False,
            checkpoint_index=None
        ) 