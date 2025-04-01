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
    id: str
    title: str
    description: str
    input_audio: str
    output_audio: str
    step_type: str = Field(default="processing", description="Type of step: processing, validation, reference")
    
    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
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
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class ExecutionResult(BaseModel):
    """Result of executing a step in the plan."""
    status: str
    output: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    duration: float = 0.0
    quality_validation: Optional[str] = None
    
    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class AudioInput(BaseModel):
    """Representation of audio input with transcript."""
    transcript: str
    timestamp: float
    
    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class AudioContent(BaseModel):
    """Audio content for sending to the model."""
    file_path: Path
    content: Optional[BinaryContent] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Allow AudioUrl and BinaryContent types
    )


class ToolDefinition(BaseModel):
    """Definition of a tool available to the agents."""
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Short description of the tool")
    signature: str = Field(description="Function signature of the tool")
    docstring: str = Field(description="Full documentation of the tool")
    
    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class ExecutorOutput(BaseModel):
    """Output from the executor agent."""
    step_index: int
    generated_code: str
    persistent_failure: bool = False
    retry_count: int = 0
    
    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


# New models for the optimized workflow

class CritiqueResult(BaseModel):
    """Result from the critique agent."""
    is_approved: bool
    suggestions: List[str] = []
    improved_version: Optional[str] = None
    reasoning: str
    critique_type: str

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class QAMetrics(BaseModel):
    """Metrics for quality assessment."""
    clarity_score: Optional[float] = None
    noise_level: Optional[float] = None
    quality_rating: Optional[float] = None
    processing_time_ms: Optional[int] = None
    sample_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    file_size_kb: Optional[int] = None

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class QAResult(BaseModel):
    """Result of QA verification of execution output."""
    meets_requirements: bool
    issues: List[str] = []
    suggestions: List[str] = []
    metrics: QAMetrics = Field(default_factory=QAMetrics)  # Use the dedicated metrics class
    reasoning: str

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class ErrorAnalysisResult(BaseModel):
    """Result of analyzing an error trace."""
    error_type: str
    root_cause: str
    fix_suggestions: List[str] = []
    code_fixes: Optional[str] = None
    requires_replanning: bool = False
    confidence: float

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class UserFeedbackRequest(BaseModel):
    """Request for feedback from the user."""
    query: str
    context: str
    options: Optional[List[str]] = None
    severity: str = "info"
    request_type: str

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class UserFeedbackResponse(BaseModel):
    """Response from the user to a feedback request."""
    response: str
    timestamp: float

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class StepInfo(BaseModel):
    """Information about a processing step."""
    id: str
    title: str
    description: str
    status: str
    input_audio: str
    output_audio: str
    code: Optional[str] = None
    execution_results: Optional[str] = None
    step_type: str = Field(default="processing", description="Type of step: processing, validation, reference")
    quality_validation: Optional[str] = None

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class PlanResult(BaseModel):
    """Result of the planning process."""
    prd: str
    steps: List[PlanStep]
    quality_checkpoints: List[str] = Field(default_factory=list, description="Points where quality validation should occur")
    reference_audio: Optional[str] = None

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class CodeGenerationResult(BaseModel):
    """Result of code generation for a step."""
    code: str
    explanation: Optional[str] = None
    references: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    )


class ProcessingResult(BaseModel):
    """Result of the audio processing workflow."""
    success: bool
    output_path: str
    steps_completed: int
    quality_assessment: Optional[str] = None
    improvement_summary: Optional[str] = None

    model_config = ConfigDict(
        extra='forbid'  # Disable additional properties for Gemini compatibility
    ) 