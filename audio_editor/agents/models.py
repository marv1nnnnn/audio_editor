"""
Pydantic models for the audio processing multi-agent system.
"""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "PENDING"
    DONE = "DONE"
    FAILED = "FAILED"


class PlanStep(BaseModel):
    """A step in the audio processing plan."""
    description: str
    status: StepStatus = StepStatus.PENDING
    code: Optional[str] = None
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    
    # Explicitly forbid extra properties to try and satisfy Gemini's schema requirements
    model_config = ConfigDict(extra='forbid')


class AudioPlan(BaseModel):
    """Plan for processing an audio file."""
    task_description: str
    steps: List[PlanStep] = []
    current_audio_path: Path
    completed_step_indices: List[int] = []
    is_complete: bool = False
    checkpoint_indices: List[int] = []


class ExecutionResult(BaseModel):
    """Result of executing a step in the plan."""
    status: str
    output: str = ""
    error_message: str = ""
    output_path: Optional[str] = None
    output_paths: Optional[List[str]] = None
    duration: float


class AudioInput(BaseModel):
    """Representation of audio input with transcript."""
    transcript: str
    timestamp: float


class ToolDefinition(BaseModel):
    """Definition of a tool available to the agents."""
    name: str
    description: str
    signature: str
    docstring: str


class PlannerOutput(BaseModel):
    """Output from the Planner Agent."""
    updated_plan: AudioPlan
    next_step_index: Optional[int] = None
    is_complete: bool = False
    replanning_needed: bool = False
    checkpoint_index: Optional[int] = None


class ExecutorOutput(BaseModel):
    """Output from the Executor Agent."""
    step_index: int
    generated_code: str
    persistent_failure: bool = False
    retry_count: int = 0

# New models for the optimized workflow

class CritiqueResult(BaseModel):
    """Result of a critique of a plan or code."""
    is_approved: bool
    suggestions: List[str] = []
    improved_version: Optional[str] = None
    reasoning: str
    critique_type: str


class QAResult(BaseModel):
    """Result of QA verification of execution output."""
    meets_requirements: bool
    issues: List[str] = []
    suggestions: List[str] = []
    metrics: Dict[str, Any] = {}
    reasoning: str


class ErrorAnalysisResult(BaseModel):
    """Result of analyzing an error trace."""
    error_type: str
    root_cause: str
    fix_suggestions: List[str] = []
    code_fixes: Optional[str] = None
    requires_replanning: bool = False
    confidence: float


class UserFeedbackRequest(BaseModel):
    """Request for feedback from the user."""
    query: str
    context: str
    options: Optional[List[str]] = None
    severity: str = "info"
    request_type: str


class UserFeedbackResponse(BaseModel):
    """Response from the user to a feedback request."""
    response: str
    timestamp: float 