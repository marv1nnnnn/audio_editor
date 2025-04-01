"""
Pydantic models for the audio processing multi-agent system.
"""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

# Add import for Pydantic AI's audio input
from pydantic_ai import AudioUrl, BinaryContent


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
    tool_args: str = Field(default="{}")  # Store as JSON string
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={"additionalProperties": True}  # Allow additional properties for Gemini
    )


class AudioPlan(BaseModel):
    """Plan for processing an audio file."""
    task_description: str
    steps: List[PlanStep] = Field(default_factory=list)
    current_audio_path: str  # Use str instead of Path
    completed_step_indices: List[int] = Field(default_factory=list)
    is_complete: bool = False
    checkpoint_indices: List[int] = Field(default_factory=list)

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={"additionalProperties": True}  # Allow additional properties for Gemini
    )


class ExecutionResult(BaseModel):
    """Result of executing a step in the plan."""
    status: str
    output: str = ""
    error_message: str = ""
    output_path: Optional[str] = None
    output_paths: Optional[List[str]] = None
    duration: float = 0.0
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={"additionalProperties": True}  # Allow additional properties for Gemini
    )


class AudioInput(BaseModel):
    """Representation of audio input with transcript."""
    transcript: str
    timestamp: float
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={"additionalProperties": True}  # Allow additional properties for Gemini
    )


class AudioContent(BaseModel):
    """Representation of actual audio content for direct model access."""
    file_path: Path = Field(..., description="Path to the audio file")
    content: Optional[Union[AudioUrl, BinaryContent]] = Field(
        default=None, 
        description="Audio content for direct model access"
    )
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True,  # Allow AudioUrl and BinaryContent types
        json_schema_extra={"additionalProperties": True}  # Allow additional properties for Gemini
    )


class ToolDefinition(BaseModel):
    """Definition of a tool available to the agents."""
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Short description of the tool")
    signature: str = Field(description="Function signature of the tool")
    docstring: str = Field(description="Full documentation of the tool")
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={"additionalProperties": True}  # Allow additional properties for Gemini
    )


class ExecutorOutput(BaseModel):
    """Output from the executor agent."""
    step_index: int
    generated_code: str
    persistent_failure: bool = False
    retry_count: int = 0
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={"additionalProperties": True}  # Allow additional properties for Gemini
    )


# New models for the optimized workflow

class CritiqueResult(BaseModel):
    """Result from the critique agent."""
    is_approved: bool
    suggestions: List[str] = []
    improved_version: Optional[str] = None
    reasoning: str
    critique_type: str


# Define a specific class for metrics to avoid using Dict with additionalProperties
class QAMetrics(BaseModel):
    """Metrics for quality assessment."""
    clarity_score: Optional[float] = None
    noise_level: Optional[float] = None
    quality_rating: Optional[float] = None
    processing_time_ms: Optional[int] = None
    sample_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    file_size_kb: Optional[int] = None


class QAResult(BaseModel):
    """Result of QA verification of execution output."""
    meets_requirements: bool
    issues: List[str] = []
    suggestions: List[str] = []
    metrics: QAMetrics = Field(default_factory=QAMetrics)  # Use the dedicated metrics class
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


class StepInfo(BaseModel):
    """Information about a processing step."""
    id: str = Field(..., description="Unique identifier for the step (e.g., step_1)")
    title: str = Field(..., description="Brief title for the step")
    description: str = Field(..., description="Detailed description of what the step should accomplish")
    status: str = Field(default="PENDING", description="Current status of the step")
    input_audio: str = Field(..., description="Path to input audio file or variable reference")
    output_audio: str = Field(..., description="Path where output audio will be saved")
    code: Optional[str] = Field(default=None, description="Generated Python code for the step")
    execution_results: Optional[str] = Field(default=None, description="Results from executing the step")
    timestamp_start: Optional[str] = Field(default=None, description="When step execution started")
    timestamp_end: Optional[str] = Field(default=None, description="When step execution finished/failed")


class PlanResult(BaseModel):
    """Result from the planner agent."""
    prd: str = Field(..., description="The Product Requirements Document")
    steps: List[StepInfo] = Field(..., description="The list of processing steps")


class CodeGenerationResult(BaseModel):
    """Result from the code generation agent."""
    code: str = Field(..., description="Generated Python code for the step")
    explanation: Optional[str] = Field(default=None, description="Explanation of the generated code")


class ProcessingResult(BaseModel):
    """Final result of the audio processing."""
    output_path: str = Field(..., description="Path to the final processed audio")
    status: str = Field(..., description="Overall processing status")
    steps_completed: int = Field(..., description="Number of completed steps")
    qa_result: Optional[QAResult] = Field(default=None, description="Results of quality assessment") 