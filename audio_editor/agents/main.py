"""
Main entry point for the multi-agent audio processor.
"""
import os
import asyncio
import argparse
import logfire
from pathlib import Path
import inspect
import importlib.util
from audio_editor.agents.coordinator import AudioProcessingCoordinator


async def process_audio_file(
    task: str,
    input_file: str,
    output_file: str = None,
    model_name: str = "gemini-2.0-flash",
    working_dir: str = None,
    transcript: str = "",
    interactive: bool = True,
    enable_error_analyzer: bool = True,
    enable_qa: bool = True,
    enable_audio_content: bool = True
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
        interactive: Whether to enable interactive user feedback
        enable_error_analyzer: Whether to enable the Error Analyzer
        enable_qa: Whether to enable the QA Agent
        enable_audio_content: Whether to enable direct audio content for models
        
    Returns:
        Path to the output audio file
    """
    # Configure Logfire
    logfire.configure()
    
    # Set up working directory
    if not working_dir:
        working_dir = os.path.join(os.getcwd(), "audio_editor_work")
    
    os.makedirs(working_dir, exist_ok=True)
    
    # Check if pydantic_ai is installed and supports audio input
    pydantic_ai_supports_audio = False
    try:
        if importlib.util.find_spec("pydantic_ai") is not None:
            from pydantic_ai import BinaryContent, AudioUrl
            pydantic_ai_supports_audio = True
            logfire.info("Pydantic AI with audio support is available")
        else:
            logfire.warning("Pydantic AI not found, disabling audio content support")
            enable_audio_content = False
    except (ImportError, AttributeError):
        logfire.warning("Pydantic AI doesn't support audio input, disabling audio content support")
        enable_audio_content = False
    
    # Initialize the coordinator with new options
    coordinator = AudioProcessingCoordinator(
        working_dir=working_dir, 
        model_name=model_name,
        interactive=interactive,
        enable_error_analyzer=enable_error_analyzer,
        enable_qa=enable_qa,
        enable_audio_content=enable_audio_content and pydantic_ai_supports_audio
    )
    
    # Log the task information
    logfire.info(f"Processing audio task: {task}")
    logfire.info(f"Input file: {input_file}")
    logfire.info(f"Model: {model_name}")
    logfire.info(f"Working directory: {working_dir}")
    logfire.info(f"Audio content enabled: {enable_audio_content and pydantic_ai_supports_audio}")
    
    # Process the audio using the appropriate method
    result_path = await coordinator.run_workflow(task, input_file, transcript)
    
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
    
    # Add new command-line arguments for the enhanced features
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive user feedback (use default responses)"
    )
    parser.add_argument(
        "--disable-error-analyzer",
        action="store_true",
        help="Disable the Error Analyzer for fixing failed steps"
    )
    parser.add_argument(
        "--disable-qa",
        action="store_true",
        help="Disable the QA Agent for output quality verification"
    )
    parser.add_argument(
        "--disable-audio-content",
        action="store_true",
        help="Disable direct audio content for models (use file paths only)"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Set logging level (default: info)"
    )
    parser.add_argument(
        "--workflow-file",
        default=None,
        help="Path to an existing workflow file to resume processing (optional)"
    )
    
    args = parser.parse_args()
    
    # Configure logging with the specified level
    logfire.configure()
    
    # Process the audio file with new options
    try:
        result_path = asyncio.run(process_audio_file(
            task=args.task,
            input_file=args.input,
            output_file=args.output,
            model_name=args.model,
            working_dir=args.working_dir,
            transcript=args.transcript,
            interactive=not args.non_interactive,
            enable_error_analyzer=not args.disable_error_analyzer,
            enable_qa=not args.disable_qa,
            enable_audio_content=not args.disable_audio_content
        ))
        
        print(f"Audio processing completed successfully. Result: {result_path}")
        
        # Print the location of the workflow Markdown file
        if args.working_dir:
            docs_dir = os.path.join(args.working_dir, "docs")
        else:
            docs_dir = os.path.join(os.getcwd(), "audio_editor_work", "docs")
            
        if os.path.exists(docs_dir):
            workflow_files = [f for f in os.listdir(docs_dir) if f.startswith("workflow_") and f.endswith(".md")]
            if workflow_files:
                latest_workflow = max(workflow_files, key=lambda f: os.path.getmtime(os.path.join(docs_dir, f)))
                print(f"Workflow file: {os.path.join(docs_dir, latest_workflow)}")
        
        return 0
        
    except Exception as e:
        logfire.error(f"Audio processing failed: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 