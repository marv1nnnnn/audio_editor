"""
Main entry point for the multi-agent audio processor.
"""
import os
import asyncio
import argparse
import logfire
from pathlib import Path

from .coordinator import AudioProcessingCoordinator


async def process_audio_file(
    task: str,
    input_file: str,
    output_file: str = None,
    model_name: str = "gemini-2.0-flash",
    working_dir: str = None,
    transcript: str = ""
) -> str:
    """
    Process an audio file using the multi-agent system.
    
    Args:
        task: Description of the processing task
        input_file: Path to the input audio file
        output_file: Path to save the output file (optional)
        model_name: Name of the LLM model to use
        working_dir: Directory for intermediate files
        transcript: Transcript of the audio file (if available)
        
    Returns:
        Path to the output audio file
    """
    # Set up working directory
    if not working_dir:
        working_dir = os.path.join(os.getcwd(), "audio_editor_work")
    
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize the coordinator
    coordinator = AudioProcessingCoordinator(working_dir, model_name)
    
    # Process the audio
    result_path = await coordinator.process_audio(task, input_file, transcript)
    
    # Copy to output location if specified
    if output_file:
        import shutil
        shutil.copy(result_path, output_file)
        logfire.info(f"Copied final result to {output_file}")
        return output_file
    
    return result_path


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Multi-agent audio processor")
    
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="Description of the audio processing task"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input audio file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the output audio file (default: auto-generated)"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.0-flash",
        help="LLM model to use (default: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--working-dir", "-w",
        default=None,
        help="Working directory for intermediate files"
    )
    parser.add_argument(
        "--transcript",
        default="",
        help="Transcript of the audio file (if available)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logfire.configure()
    
    # Process the audio file
    try:
        result_path = asyncio.run(process_audio_file(
            task=args.task,
            input_file=args.input,
            output_file=args.output,
            model_name=args.model,
            working_dir=args.working_dir,
            transcript=args.transcript
        ))
        
        print(f"Audio processing completed successfully. Result: {result_path}")
        return 0
        
    except Exception as e:
        logfire.error(f"Audio processing failed: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 