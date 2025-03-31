"""
Unit tests for dependency classes.
"""
import pytest
from pathlib import Path

from audio_editor.agents.dependencies import AudioProcessingContext


def test_audio_processing_context_creation(temp_workspace):
    """Test AudioProcessingContext creation and tool collection."""
    # Test context creation
    context = AudioProcessingContext.create(
        workspace_dir=temp_workspace,
        model_name="test-model"
    )
    
    assert isinstance(context.workspace_dir, str)
    assert context.workspace_dir == str(temp_workspace)
    assert context.model_name == "test-model"
    assert isinstance(context.available_tools, dict)
    assert len(context.available_tools) > 0  # Should have some tools


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


def test_audio_processing_context_get_tool_signatures(temp_workspace):
    """Test that AudioProcessingContext can get tool signatures."""
    context = AudioProcessingContext.create(
        workspace_dir=temp_workspace
    )
    
    # Get tool signatures
    signatures = context.get_tool_signatures()
    
    # Check that we have signatures
    assert isinstance(signatures, dict)
    assert len(signatures) > 0
    
    # Check that signatures are strings
    for name, sig in signatures.items():
        assert isinstance(name, str)
        assert isinstance(sig, str) 