"""
Dependencies for the audio processing multi-agent system.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable

import inspect
import audio_tools

from .models import AudioPlan, AudioInput, ExecutionResult, ToolDefinition


@dataclass
class AudioProcessingContext:
    """Context for audio processing, shared between agents."""
    workspace_dir: Path
    available_tools: Dict[str, Callable]
    model_name: str = "gemini-2.0-flash"
    
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


@dataclass
class ExecutorDependencies:
    """Dependencies for the Executor Agent."""
    context: AudioProcessingContext
    tool_definitions: List[ToolDefinition]
    plan_step_index: int
    plan: AudioPlan
    execution_result: Optional[ExecutionResult] = None
    retry_limit: int = 2 