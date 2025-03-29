"""
Unit tests for dependency classes.
"""
import pytest
from pathlib import Path

from audio_editor.agents.dependencies import (
    AudioProcessingContext, PlannerDependencies, ExecutorDependencies
)


def test_audio_processing_context_creation(temp_workspace):
    """Test AudioProcessingContext creation and tool collection."""
    # Test context creation
    context = AudioProcessingContext.create(
        workspace_dir=temp_workspace,
        model_name="test-model"
    )
    
    assert isinstance(context.workspace_dir, Path)
    assert context.workspace_dir == temp_workspace
    assert context.model_name == "test-model"
    assert isinstance(context.available_tools, dict)
    assert len(context.available_tools) > 0  # Should have some tools


def test_planner_dependencies_creation(
    audio_processing_context,
    sample_tool_definitions,
    sample_audio_input,
    sample_audio_plan
):
    """Test PlannerDependencies creation and validation."""
    # Test creation with all fields
    deps = PlannerDependencies(
        context=audio_processing_context,
        task_description="Test task",
        tool_definitions=sample_tool_definitions,
        audio_input=sample_audio_input,
        current_plan=sample_audio_plan,
        execution_result=None
    )
    
    assert deps.context == audio_processing_context
    assert deps.task_description == "Test task"
    assert deps.tool_definitions == sample_tool_definitions
    assert deps.audio_input == sample_audio_input
    assert deps.current_plan == sample_audio_plan
    assert deps.execution_result is None

    # Test creation with minimal fields
    deps = PlannerDependencies(
        context=audio_processing_context,
        task_description="Test task",
        tool_definitions=sample_tool_definitions,
        audio_input=sample_audio_input
    )
    
    assert deps.context == audio_processing_context
    assert deps.current_plan is None
    assert deps.execution_result is None


def test_executor_dependencies_creation(
    audio_processing_context,
    sample_tool_definitions,
    sample_audio_plan
):
    """Test ExecutorDependencies creation and validation."""
    # Test creation with all fields
    deps = ExecutorDependencies(
        context=audio_processing_context,
        tool_definitions=sample_tool_definitions,
        plan_step_index=0,
        plan=sample_audio_plan,
        execution_result=None,
        retry_limit=3
    )
    
    assert deps.context == audio_processing_context
    assert deps.tool_definitions == sample_tool_definitions
    assert deps.plan_step_index == 0
    assert deps.plan == sample_audio_plan
    assert deps.execution_result is None
    assert deps.retry_limit == 3

    # Test creation with default retry limit
    deps = ExecutorDependencies(
        context=audio_processing_context,
        tool_definitions=sample_tool_definitions,
        plan_step_index=0,
        plan=sample_audio_plan
    )
    
    assert deps.retry_limit == 2  # Default value


def test_audio_processing_context_tool_loading(temp_workspace):
    """Test that AudioProcessingContext loads tools correctly."""
    context = AudioProcessingContext.create(
        workspace_dir=temp_workspace
    )
    
    # Check that we have the expected tools
    tools = context.available_tools
    assert isinstance(tools, dict)
    
    # Tools should be uppercase functions
    for name in tools:
        assert name.isupper()
        assert callable(tools[name])


def test_dependencies_immutability(
    audio_processing_context,
    sample_tool_definitions,
    sample_audio_input,
    sample_audio_plan
):
    """Test that dependency objects maintain immutability."""
    # Create initial dependencies
    planner_deps = PlannerDependencies(
        context=audio_processing_context,
        task_description="Test task",
        tool_definitions=sample_tool_definitions,
        audio_input=sample_audio_input,
        current_plan=sample_audio_plan
    )
    
    executor_deps = ExecutorDependencies(
        context=audio_processing_context,
        tool_definitions=sample_tool_definitions,
        plan_step_index=0,
        plan=sample_audio_plan
    )
    
    # Try to modify attributes (should raise AttributeError)
    with pytest.raises(AttributeError):
        planner_deps.task_description = "New task"
    
    with pytest.raises(AttributeError):
        executor_deps.plan_step_index = 1 