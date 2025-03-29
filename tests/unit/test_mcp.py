"""
Unit tests for the Master Control Program (MCP).
"""
import pytest
import os
from pathlib import Path

from audio_editor.agents.mcp import MCPCodeExecutor
from audio_editor.agents.models import ExecutionResult


def test_mcp_initialization(temp_workspace):
    """Test MCP initialization and tool gathering."""
    mcp = MCPCodeExecutor(str(temp_workspace))
    
    assert os.path.abspath(mcp.workspace_dir) == os.path.abspath(str(temp_workspace))
    assert isinstance(mcp.tools, dict)
    assert len(mcp.tools) > 0  # Should have some tools


def test_mcp_code_parsing():
    """Test MCP code parsing functionality."""
    mcp = MCPCodeExecutor("/tmp")
    
    # Test valid code parsing
    func_name, kwargs = mcp._parse_code("TEST_TOOL(input_path='test.wav', param=1.0)")
    assert func_name == "TEST_TOOL"
    assert kwargs == {"input_path": "test.wav", "param": 1.0}
    
    # Test invalid code
    with pytest.raises(SyntaxError):
        mcp._parse_code("invalid python code")
    
    with pytest.raises(ValueError):
        mcp._parse_code("print('not a tool call')")


@pytest.mark.asyncio
async def test_mcp_code_execution(temp_workspace, sample_audio_file):
    """Test MCP code execution with a real audio file."""
    mcp = MCPCodeExecutor(str(temp_workspace))
    
    # Test successful execution
    code = f"LEN(wav_path='{sample_audio_file}')"
    result = await mcp.execute_code(code)
    
    assert isinstance(result, ExecutionResult)
    assert result.status == "SUCCESS"
    assert float(result.output) > 0  # Should return audio length
    assert result.error_message == ""
    
    # Test execution with invalid tool
    code = "INVALID_TOOL(param=1.0)"
    result = await mcp.execute_code(code)
    
    assert result.status == "FAILURE"
    assert "Tool 'INVALID_TOOL' not found" in result.error_message


@pytest.mark.asyncio
async def test_mcp_error_handling(temp_workspace):
    """Test MCP error handling during execution."""
    mcp = MCPCodeExecutor(str(temp_workspace))
    
    # Test execution with invalid parameters
    code = "LEN(wav_path='nonexistent.wav')"
    result = await mcp.execute_code(code)
    
    assert result.status == "FAILURE"
    assert result.error_message != ""
    assert result.duration > 0
    
    # Test execution with syntax error
    code = "LEN(wav_path=)"  # Invalid syntax
    result = await mcp.execute_code(code)
    
    assert result.status == "FAILURE"
    assert "SyntaxError" in result.error_message


@pytest.mark.asyncio
async def test_mcp_output_handling(temp_workspace, sample_audio_file):
    """Test MCP handling of tool outputs."""
    mcp = MCPCodeExecutor(str(temp_workspace))
    
    # Test tool returning a file path
    output_path = temp_workspace / "output.wav"
    code = f"COPY(wav_path='{sample_audio_file}', output_path='{output_path}')"
    result = await mcp.execute_code(code)
    
    assert result.status == "SUCCESS"
    assert result.output_path is not None
    assert os.path.exists(result.output_path)
    
    # Test tool returning multiple file paths
    output_paths = [
        temp_workspace / "output1.wav",
        temp_workspace / "output2.wav"
    ]
    code = f"SPLIT(wav_path='{sample_audio_file}', output_paths={[str(p) for p in output_paths]})"
    result = await mcp.execute_code(code)
    
    assert result.status == "SUCCESS"
    assert result.output_paths is not None
    assert len(result.output_paths) == 2
    for path in result.output_paths:
        assert os.path.exists(path) 