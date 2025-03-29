"""
Pydantic models for the audio processing multi-agent system.
"""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "PENDING"
    DONE = "DONE"
    FAILED = "FAILED"


class PlanStep(BaseModel):
    """A step in the audio processing plan."""
    description: str = Field(..., description="Description of the step")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Status of the step")
    code: Optional[str] = Field(None, description="Python code to execute this step")
    tool_name: str = Field(..., description="Name of the tool to use")
    tool_args: dict = Field(default_factory=dict, description="Arguments for the tool")


class AudioPlan(BaseModel):
    """Plan for processing an audio file."""
    task_description: str = Field(..., description="Description of the task to perform")
    steps: List[PlanStep] = Field(default_factory=list, description="List of steps in the plan")
    current_audio_path: Path = Field(..., description="Path to the current audio file")
    completed_step_indices: List[int] = Field(default_factory=list, description="Indices of completed steps")
    is_complete: bool = Field(default=False, description="Whether the plan is complete")
    checkpoint_indices: List[int] = Field(default_factory=list, description="Indices of steps marked as checkpoints")


class ExecutionResult(BaseModel):
    """Result of executing a step in the plan."""
    status: str = Field(..., description="SUCCESS or FAILURE")
    output: str = Field(default="", description="Standard output from execution")
    error_message: str = Field(default="", description="Error message if execution failed")
    output_path: Optional[str] = Field(None, description="Path to output file if any")
    output_paths: Optional[List[str]] = Field(None, description="Paths to output files if multiple")
    duration: float = Field(..., description="Execution time in seconds")


class AudioInput(BaseModel):
    """Representation of audio input with transcript."""
    transcript: str = Field(..., description="Transcript of the audio")
    timestamp: float = Field(..., description="Timestamp of the audio")


class ToolDefinition(BaseModel):
    """Definition of a tool available to the agents."""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    signature: str = Field(..., description="Signature of the tool")
    docstring: str = Field(..., description="Docstring of the tool")


class PlannerOutput(BaseModel):
    """Output from the Planner Agent."""
    updated_plan: AudioPlan = Field(..., description="Updated plan with step statuses")
    next_step_index: Optional[int] = Field(None, description="Index of the next step to execute")
    is_complete: bool = Field(default=False, description="Whether the task is complete")
    replanning_needed: bool = Field(default=False, description="Whether replanning is needed")
    checkpoint_index: Optional[int] = Field(None, description="Index to set as checkpoint if successful")


class ExecutorOutput(BaseModel):
    """Output from the Executor Agent."""
    step_index: int = Field(..., description="Index of the step being executed")
    generated_code: str = Field(..., description="Generated Python code for execution")
    persistent_failure: bool = Field(default=False, description="Whether the step has persistently failed")
    retry_count: int = Field(default=0, description="Number of retries attempted") 