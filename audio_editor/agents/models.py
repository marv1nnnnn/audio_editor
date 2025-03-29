"""
Pydantic models for the audio processing multi-agent system.
"""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

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

# New models for the optimized workflow

class CritiqueResult(BaseModel):
    """Result of a critique of a plan or code."""
    is_approved: bool = Field(..., description="Whether the plan or code is approved")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    improved_version: Optional[str] = Field(None, description="Improved version if not approved")
    reasoning: str = Field(..., description="Reasoning behind the critique")
    critique_type: str = Field(..., description="Type of critique (plan or code)")


class QAResult(BaseModel):
    """Result of QA verification of execution output."""
    meets_requirements: bool = Field(..., description="Whether the output meets requirements")
    issues: List[str] = Field(default_factory=list, description="Issues with the output")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Quantitative metrics if applicable")
    reasoning: str = Field(..., description="Reasoning behind the QA assessment")


class ErrorAnalysisResult(BaseModel):
    """Result of analyzing an error trace."""
    error_type: str = Field(..., description="Type of error")
    root_cause: str = Field(..., description="Root cause of the error")
    fix_suggestions: List[str] = Field(default_factory=list, description="Suggestions to fix the error")
    code_fixes: Optional[str] = Field(None, description="Suggested code fixes")
    requires_replanning: bool = Field(default=False, description="Whether replanning is needed")
    confidence: float = Field(..., description="Confidence in the analysis (0-1)")


class UserFeedbackRequest(BaseModel):
    """Request for feedback from the user."""
    query: str = Field(..., description="Question to ask the user")
    context: str = Field(..., description="Context for the question")
    options: Optional[List[str]] = Field(None, description="Options for the user to choose from")
    severity: str = Field(default="info", description="Severity of the request (info, warning, error)")
    request_type: str = Field(..., description="Type of request (clarification, confirmation, choice)")


class UserFeedbackResponse(BaseModel):
    """Response from the user to a feedback request."""
    response: str = Field(..., description="User's response")
    timestamp: float = Field(..., description="Timestamp of the response") 