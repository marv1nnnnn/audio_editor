"""
Unit tests for Pydantic models.
"""
import pytest
from pydantic import ValidationError

from audio_editor.agents.models import (
    ExecutionResult, UserFeedbackRequest, UserFeedbackResponse
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


def test_user_feedback_request_creation():
    """Test UserFeedbackRequest model creation and validation."""
    # Test valid creation
    feedback_request = UserFeedbackRequest(
        query="Test query",
        context="Test context",
        request_type="clarification"
    )
    assert feedback_request.query == "Test query"
    assert feedback_request.context == "Test context"
    assert feedback_request.request_type == "clarification"
    assert feedback_request.severity == "info"  # Default
    assert feedback_request.options is None

    # Test with options
    feedback_request = UserFeedbackRequest(
        query="Test query",
        context="Test context",
        request_type="choice",
        options=["Option 1", "Option 2"],
        severity="warning"
    )
    assert feedback_request.options == ["Option 1", "Option 2"]
    assert feedback_request.severity == "warning"


def test_user_feedback_response_creation():
    """Test UserFeedbackResponse model creation and validation."""
    # Test valid creation
    feedback_response = UserFeedbackResponse(
        response="Test response",
        timestamp=1234567890.0
    )
    assert feedback_response.response == "Test response"
    assert feedback_response.timestamp == 1234567890.0 