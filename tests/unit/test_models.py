"""
Unit tests for Pydantic models.
"""
import pytest
from pathlib import Path
from pydantic import ValidationError

from audio_editor.agents.models import (
    AudioPlan, PlanStep, ExecutionResult, StepStatus,
    AudioInput, ToolDefinition
)


def test_plan_step_creation():
    """Test PlanStep model creation and validation."""
    # Test valid creation
    step = PlanStep(
        description="Test step",
        tool_name="TEST_TOOL",
        tool_args={"param": 1.0}
    )
    assert step.description == "Test step"
    assert step.tool_name == "TEST_TOOL"
    assert step.status == StepStatus.PENDING
    assert step.tool_args == {"param": 1.0}
    assert step.code is None

    # Test invalid creation
    with pytest.raises(ValidationError):
        PlanStep(
            description="",  # Empty description
            tool_name="TEST_TOOL"
        )


def test_audio_plan_creation(sample_audio_file):
    """Test AudioPlan model creation and validation."""
    # Test valid creation
    plan = AudioPlan(
        task_description="Test task",
        steps=[
            PlanStep(
                description="Step 1",
                tool_name="TEST_TOOL",
                tool_args={}
            )
        ],
        current_audio_path=sample_audio_file
    )
    assert plan.task_description == "Test task"
    assert len(plan.steps) == 1
    assert isinstance(plan.current_audio_path, Path)
    assert not plan.is_complete
    assert not plan.completed_step_indices
    assert not plan.checkpoint_indices

    # Test invalid creation
    with pytest.raises(ValidationError):
        AudioPlan(
            task_description="",  # Empty task description
            current_audio_path=sample_audio_file
        )


def test_execution_result_creation():
    """Test ExecutionResult model creation and validation."""
    # Test successful result
    result = ExecutionResult(
        status="SUCCESS",
        output="Test output",
        duration=1.5,
        output_path="/path/to/output.wav"
    )
    assert result.status == "SUCCESS"
    assert result.output == "Test output"
    assert result.error_message == ""
    assert result.duration == 1.5
    assert result.output_path == "/path/to/output.wav"

    # Test failed result
    result = ExecutionResult(
        status="FAILURE",
        error_message="Test error",
        duration=0.5
    )
    assert result.status == "FAILURE"
    assert result.error_message == "Test error"
    assert result.output == ""
    assert result.duration == 0.5
    assert result.output_path is None

    # Test invalid creation
    with pytest.raises(ValidationError):
        ExecutionResult(
            status="INVALID",  # Invalid status
            duration=1.0
        )


def test_audio_input_creation():
    """Test AudioInput model creation and validation."""
    # Test valid creation
    audio_input = AudioInput(
        transcript="Test transcript",
        timestamp=1234567890.0
    )
    assert audio_input.transcript == "Test transcript"
    assert audio_input.timestamp == 1234567890.0

    # Test invalid creation
    with pytest.raises(ValidationError):
        AudioInput(
            transcript="",  # Empty transcript
            timestamp=1234567890.0
        )


def test_tool_definition_creation():
    """Test ToolDefinition model creation and validation."""
    # Test valid creation
    tool_def = ToolDefinition(
        name="TEST_TOOL",
        description="Test tool",
        signature="(param: float = 1.0) -> str",
        docstring="Test tool docstring"
    )
    assert tool_def.name == "TEST_TOOL"
    assert tool_def.description == "Test tool"
    assert tool_def.signature == "(param: float = 1.0) -> str"
    assert tool_def.docstring == "Test tool docstring"

    # Test invalid creation
    with pytest.raises(ValidationError):
        ToolDefinition(
            name="",  # Empty name
            description="Test tool",
            signature="()",
            docstring=""
        ) 