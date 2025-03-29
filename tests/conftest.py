"""
Pytest configuration and shared fixtures.
"""
import os
import pytest
from pathlib import Path
import tempfile
import shutil
import torch
import torchaudio

from audio_editor.agents.models import (
    AudioPlan, PlanStep, ExecutionResult, StepStatus,
    AudioInput, ToolDefinition
)
from audio_editor.agents.dependencies import (
    AudioProcessingContext, PlannerDependencies,
    ExecutorDependencies
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_audio_file(temp_workspace):
    """Create a sample audio file for testing."""
    # Create a simple sine wave
    sample_rate = 44100
    duration = 2.0  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
    
    # Save to temp file
    output_path = temp_workspace / "test_audio.wav"
    torchaudio.save(str(output_path), waveform, sample_rate)
    
    return output_path


@pytest.fixture
def audio_processing_context(temp_workspace):
    """Create a test AudioProcessingContext."""
    return AudioProcessingContext.create(
        workspace_dir=temp_workspace,
        model_name="test-model"
    )


@pytest.fixture
def sample_tool_definitions():
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="TEST_TOOL",
            description="A test tool",
            signature="(input_path: str, param: float = 1.0) -> str",
            docstring="Test tool for unit testing."
        )
    ]


@pytest.fixture
def sample_audio_plan(sample_audio_file):
    """Create a sample AudioPlan for testing."""
    return AudioPlan(
        task_description="Test audio processing task",
        steps=[
            PlanStep(
                description="Test step 1",
                status=StepStatus.PENDING,
                tool_name="TEST_TOOL",
                tool_args={"input_path": str(sample_audio_file)}
            )
        ],
        current_audio_path=sample_audio_file,
        completed_step_indices=[],
        is_complete=False,
        checkpoint_indices=[]
    )


@pytest.fixture
def sample_audio_input():
    """Create a sample AudioInput for testing."""
    return AudioInput(
        transcript="This is a test audio file",
        timestamp=1234567890.0
    )


@pytest.fixture
def planner_dependencies(
    audio_processing_context,
    sample_tool_definitions,
    sample_audio_input,
    sample_audio_plan
):
    """Create sample PlannerDependencies for testing."""
    return PlannerDependencies(
        context=audio_processing_context,
        task_description="Test task",
        tool_definitions=sample_tool_definitions,
        audio_input=sample_audio_input,
        current_plan=sample_audio_plan
    )


@pytest.fixture
def executor_dependencies(
    audio_processing_context,
    sample_tool_definitions,
    sample_audio_plan
):
    """Create sample ExecutorDependencies for testing."""
    return ExecutorDependencies(
        context=audio_processing_context,
        tool_definitions=sample_tool_definitions,
        plan_step_index=0,
        plan=sample_audio_plan,
        retry_limit=2
    ) 