"""
Dependencies for the audio processing multi-agent system.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Any

import inspect
from audio_editor import audio_tools

from .models import (
    AudioPlan, AudioInput, ExecutionResult, ToolDefinition,
    CritiqueResult, QAResult, ErrorAnalysisResult,
    UserFeedbackRequest, UserFeedbackResponse
)


@dataclass
class AudioProcessingContext:
    """Context for audio processing, shared between agents."""
    workspace_dir: Path
    available_tools: Dict[str, Callable]
    model_name: str = "gemini-2.0-flash"
    user_feedback_handler: Optional[Callable[[UserFeedbackRequest], UserFeedbackResponse]] = None
    
    @classmethod
    def create(cls, workspace_dir: Path, model_name: str = "gemini-2.0-flash") -> "AudioProcessingContext":
        """Create a new context with tools from audio_tools."""
        tools = {}
        for name, func in inspect.getmembers(audio_tools):
            # Tools are uppercase functions
            if inspect.isfunction(func) and name.isupper() and not name.startswith("_"):
                tools[name] = func
                
        return cls(
            workspace_dir=workspace_dir,
            available_tools=tools,
            model_name=model_name
        )


@dataclass
class PlannerDependencies:
    """Dependencies for the Planner Agent."""
    context: AudioProcessingContext
    task_description: str
    tool_definitions: List[ToolDefinition]
    audio_input: AudioInput
    current_plan: Optional[AudioPlan] = None
    execution_result: Optional[ExecutionResult] = None
    critique_result: Optional[CritiqueResult] = None
    user_feedback: Optional[UserFeedbackResponse] = None


@dataclass
class ExecutorDependencies:
    """Dependencies for the Executor Agent."""
    context: AudioProcessingContext
    plan: AudioPlan
    plan_step_index: int
    tool_definitions: List[ToolDefinition]
    execution_result: Optional[ExecutionResult] = None
    error_analysis: Optional[ErrorAnalysisResult] = None
    critique_result: Optional[CritiqueResult] = None
    retry_limit: int = 2
    user_feedback: Optional[UserFeedbackResponse] = None


@dataclass
class CritiqueAgentDependencies:
    """Dependencies for the Critique Agent."""
    context: AudioProcessingContext
    tool_definitions: List[ToolDefinition]
    plan: Optional[AudioPlan] = None
    plan_step_index: Optional[int] = None
    generated_code: Optional[str] = None
    critique_type: str = "plan"  # Either "plan" or "code"
    task_description: Optional[str] = None


@dataclass
class QAAgentDependencies:
    """Dependencies for the QA Agent."""
    context: AudioProcessingContext
    task_description: str
    plan: AudioPlan
    execution_result: ExecutionResult
    original_audio_path: Path
    processed_audio_path: Path
    tool_definitions: List[ToolDefinition]


@dataclass
class ErrorAnalysisDependencies:
    """Dependencies for error analysis."""
    context: AudioProcessingContext
    execution_result: ExecutionResult
    plan: AudioPlan
    plan_step_index: int
    generated_code: str
    tool_definitions: List[ToolDefinition] 