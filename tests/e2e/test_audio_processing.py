"""
End-to-end tests for the audio processing system.
"""
import pytest
import os
import asyncio
from pathlib import Path

from audio_editor.agents.main import process_audio_file
from audio_editor.agents.coordinator import AudioProcessingCoordinator


@pytest.mark.asyncio
async def test_basic_audio_processing(temp_workspace, sample_audio_file):
    """Test basic audio processing workflow."""
    # Process audio file with a simple task
    result_path = await process_audio_file(
        task="Get the length of the audio and create a copy",
        input_file=str(sample_audio_file),
        working_dir=str(temp_workspace)
    )
    
    assert os.path.exists(result_path)
    assert Path(result_path).suffix == ".wav"


@pytest.mark.asyncio
async def test_complex_processing_chain(temp_workspace, sample_audio_file):
    """Test a complex chain of audio processing steps."""
    # Process audio with multiple transformations
    result_path = await process_audio_file(
        task=(
            "First get the audio length, then create a copy with increased volume, "
            "and finally normalize the audio loudness"
        ),
        input_file=str(sample_audio_file),
        working_dir=str(temp_workspace)
    )
    
    assert os.path.exists(result_path)
    assert Path(result_path).suffix == ".wav"
    
    # Verify the file is different from input
    assert os.path.getsize(result_path) != os.path.getsize(sample_audio_file)


@pytest.mark.asyncio
async def test_error_recovery_workflow(temp_workspace, sample_audio_file):
    """Test workflow with error recovery."""
    coordinator = AudioProcessingCoordinator(str(temp_workspace))
    
    # Create a task that will partially succeed but then fail
    result_path = await coordinator.process_audio(
        task_description=(
            "First copy the audio file, then try to use a non-existent tool, "
            "and finally get the audio length"
        ),
        audio_file_path=str(sample_audio_file)
    )
    
    # Despite the error in the middle, we should still get a result
    assert os.path.exists(result_path)
    assert Path(result_path).suffix == ".wav"


@pytest.mark.asyncio
async def test_audio_processing_with_transcript(temp_workspace, sample_audio_file):
    """Test audio processing with transcript information."""
    result_path = await process_audio_file(
        task="Analyze the audio content and create a processed version",
        input_file=str(sample_audio_file),
        working_dir=str(temp_workspace),
        transcript="This is a test audio file containing a 440 Hz sine wave"
    )
    
    assert os.path.exists(result_path)
    assert Path(result_path).suffix == ".wav"


@pytest.mark.asyncio
async def test_custom_output_location(temp_workspace, sample_audio_file):
    """Test specifying custom output location."""
    output_path = temp_workspace / "custom_output.wav"
    
    result_path = await process_audio_file(
        task="Copy the audio file",
        input_file=str(sample_audio_file),
        output_file=str(output_path),
        working_dir=str(temp_workspace)
    )
    
    assert result_path == str(output_path)
    assert os.path.exists(output_path)


@pytest.mark.asyncio
async def test_concurrent_processing(temp_workspace, sample_audio_file):
    """Test running multiple audio processing tasks concurrently."""
    tasks = [
        process_audio_file(
            task=f"Create copy {i} of the audio file",
            input_file=str(sample_audio_file),
            output_file=str(temp_workspace / f"output_{i}.wav"),
            working_dir=str(temp_workspace / f"work_{i}")
        )
        for i in range(3)
    ]
    
    # Run tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Verify all tasks completed successfully
    for result_path in results:
        assert os.path.exists(result_path)
        assert Path(result_path).suffix == ".wav"
    
    # Verify files are different (have unique paths)
    assert len(set(results)) == len(results) 