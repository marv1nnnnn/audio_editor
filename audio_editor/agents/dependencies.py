"""
Dependencies for the audio processing multi-agent system.
"""
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import inspect
from pydantic import BaseModel, Field, ConfigDict
from audio_editor import audio_tools
import logfire

from .models import (
    AudioPlan, AudioInput, ExecutionResult, ToolDefinition,
    CritiqueResult, QAResult, ErrorAnalysisResult,
    UserFeedbackRequest, UserFeedbackResponse
)


# Configure Logfire for debugging
logfire.configure()


class AudioProcessingContext(BaseModel):
    """Context for audio processing, shared between agents."""
    workspace_dir: Path  # Can use Path directly with Python 3.11+
    available_tools: Dict[str, Callable] = Field(  # Store actual function objects
        description="Available audio processing tools",
        exclude=True  # Exclude from schema since functions can't be serialized
    )
    model_name: str = "gemini-2.0-flash"
    user_feedback_handler: Optional[Callable[[UserFeedbackRequest], UserFeedbackResponse]] = Field(
        default=None,
        description="Handler for user feedback requests",
        exclude=True  # Exclude from schema since functions can't be serialized
    )
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Allow Callable types
    )
    
    @classmethod
    def create(cls, workspace_dir: Path, model_name: str = "gemini-2.0-flash") -> "AudioProcessingContext":
        """Create a new context with tools from audio_tools."""
        with logfire.span("create_audio_processing_context"):
            logfire.info(f"Creating AudioProcessingContext with model: {model_name}")
            tools = {}
            
            # Collect available tools from audio_tools
            for name, func in inspect.getmembers(audio_tools):
                # Tools are uppercase functions
                if inspect.isfunction(func) and name.isupper() and not name.startswith("_"):
                    tools[name] = func  # Store the actual function object
                    logfire.debug(f"Added tool: {name}")
                    
            # Create the context instance
            context = cls(
                workspace_dir=workspace_dir,
                available_tools=tools,
                model_name=model_name
            )
            
            logfire.info(f"Created AudioProcessingContext with {len(tools)} tools")
            return context

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get structured definitions of all available tools."""
        with logfire.span("get_tool_definitions"):
            definitions = []
            
            for name, func in self.available_tools.items():
                doc = inspect.getdoc(func) or "No description available."
                first_line_doc = doc.splitlines()[0].strip()
                
                try:
                    sig = str(inspect.signature(func))
                    sig = sig.replace("NoneType", "None")
                except ValueError:
                    sig = "(...)"  # Fallback
                    
                definitions.append(ToolDefinition(
                    name=name,
                    description=first_line_doc,
                    signature=sig,
                    docstring=doc
                ))
                
            return definitions

    def get_tool_signatures(self) -> Dict[str, str]:
        """Get the signatures of available tools as strings."""
        with logfire.span("get_tool_signatures"):
            return {
                name: str(inspect.signature(func))
                for name, func in self.available_tools.items()
            }


class SerializableAudioProcessingContext(BaseModel):
    """Serializable subset of AudioProcessingContext, excluding callables."""
    workspace_dir: Path
    model_name: str = "gemini-2.0-flash"
    
    model_config = ConfigDict(extra='forbid')


class PlannerDependencies(BaseModel):
    """Dependencies for the Planner Agent."""
    context: SerializableAudioProcessingContext = Field(
        description="Audio processing context"
    )
    task_description: str = Field(description="Description of the audio processing task")
    tool_definitions: List[ToolDefinition] = Field(description="List of available tools")
    audio_input: AudioInput = Field(description="Audio input information")
    current_plan: Optional[AudioPlan] = Field(
        default=None,
        description="Current audio plan"
    )
    execution_result: Optional[ExecutionResult] = Field(
        default=None,
        description="Result of execution"
    )
    critique_result: Optional[CritiqueResult] = Field(
        default=None,
        description="Result of critique"
    )
    user_feedback: Optional[UserFeedbackResponse] = Field(
        default=None,
        description="User feedback"
    )
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Allow ToolDefinition objects
    )

    @classmethod
    def from_models(
        cls,
        context: AudioProcessingContext,
        task_description: str,
        tool_definitions: List[ToolDefinition],
        audio_input: AudioInput,
        current_plan: Optional[AudioPlan] = None,
        execution_result: Optional[ExecutionResult] = None,
        critique_result: Optional[CritiqueResult] = None,
        user_feedback: Optional[UserFeedbackResponse] = None
    ) -> "PlannerDependencies":
        """Create PlannerDependencies from model instances."""
        # Create serializable context with only serializable fields
        serializable_context = SerializableAudioProcessingContext(
            workspace_dir=context.workspace_dir,
            model_name=context.model_name
        )
        
        return cls(
            context=serializable_context,
            task_description=task_description,
            tool_definitions=tool_definitions,  # Pass ToolDefinition objects directly
            audio_input=audio_input,            # Pass AudioInput object directly
            current_plan=current_plan,          # Pass AudioPlan object directly
            execution_result=execution_result,  # Pass ExecutionResult object directly
            critique_result=critique_result,    # Pass CritiqueResult object directly
            user_feedback=user_feedback         # Pass UserFeedbackResponse object directly
        )


class ExecutorDependencies(BaseModel):
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
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Allow complex types
    )


class CritiqueAgentDependencies(BaseModel):
    """Dependencies for the Critique Agent."""
    context: AudioProcessingContext
    tool_definitions: List[ToolDefinition]
    plan: Optional[AudioPlan] = None
    plan_step_index: Optional[int] = None
    generated_code: Optional[str] = None
    critique_type: str = "plan"  # Either "plan" or "code"
    task_description: Optional[str] = None
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Allow complex types
    )


class QAAgentDependencies(BaseModel):
    """Dependencies for the QA Agent."""
    context: AudioProcessingContext
    task_description: str
    plan: AudioPlan
    execution_result: ExecutionResult
    original_audio_path: Path
    processed_audio_path: Path
    tool_definitions: List[ToolDefinition]
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Allow complex types
    )


class ErrorAnalysisDependencies(BaseModel):
    """Dependencies for error analysis."""
    context: AudioProcessingContext
    execution_result: ExecutionResult
    plan: AudioPlan
    plan_step_index: int
    generated_code: str
    tool_definitions: List[ToolDefinition]
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Allow complex types
    ) 