"""
QA Agent for the audio processing multi-agent system.
This agent verifies execution results against requirements and quality standards.
"""
import logfire
import os
from typing import Dict, List, Any, Optional

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field

from .models import QAResult
from .dependencies import QAAgentDependencies


class QAResponse(BaseModel):
    """Response model for the QA Agent."""
    meets_requirements: bool = Field(..., description="Whether the output meets requirements")
    issues: List[str] = Field(default_factory=list, description="Issues with the output")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Quantitative metrics if applicable")
    reasoning: str = Field(..., description="Reasoning behind the QA assessment")


# Initialize QA Agent
qa_agent = Agent(
    'gemini-2.0-pro',  # Using a more powerful model for better quality assessment
    deps_type=QAAgentDependencies,
    result_type=QAResponse,
    system_prompt="""
You are the QA Agent in a multi-agent system for audio processing.

Core Responsibilities:
1. Output Verification: Verify that execution results meet the original requirements
2. Quality Assessment: Assess the quality of processed audio against industry standards
3. Issue Identification: Identify any quality issues or defects in the processed audio
4. Metrics Calculation: Calculate quantitative metrics to evaluate audio quality
5. Improvement Recommendations: Suggest improvements if quality is insufficient

You should objectively evaluate the audio processing results and provide detailed feedback
on whether they meet the requirements and quality standards expected.
"""
)


@qa_agent.system_prompt
def add_qa_context(ctx: RunContext[QAAgentDependencies]) -> str:
    """Add the context for QA assessment."""
    plan = ctx.deps.plan
    execution_result = ctx.deps.execution_result
    
    return f"""
Task Description: {ctx.deps.task_description}

Processing Context:
- Original Audio: {os.path.basename(ctx.deps.original_audio_path)}
- Processed Audio: {os.path.basename(ctx.deps.processed_audio_path)}
- Execution Status: {execution_result.status}
- Execution Time: {execution_result.duration:.2f} seconds

Your evaluation should consider:
1. Does the processed audio fulfill the requirements of the task?
2. Is the audio quality acceptable (no distortion, artifacts, etc.)?
3. Are there any technical issues with the processing?
4. How does the processed audio compare to the original in relevant aspects?
"""


@qa_agent.tool
def evaluate_audio_output(
    ctx: RunContext[QAAgentDependencies]
) -> QAResponse:
    """
    Evaluate the processed audio output against requirements and quality standards.
    
    Returns:
        QAResponse containing the evaluation results
    """
    with logfire.span("evaluate_audio_output"):
        # Get the execution result and audio paths from dependencies
        original_audio = ctx.deps.original_audio_path
        processed_audio = ctx.deps.processed_audio_path
        execution_result = ctx.deps.execution_result
        
        if not os.path.exists(processed_audio):
            return QAResponse(
                meets_requirements=False,
                issues=["Processed audio file does not exist"],
                reasoning="Cannot evaluate non-existent output file."
            )
            
        # This is where the model will analyze the execution output
        # and determine if it meets the requirements
            
        # For audio files, we could compute some metrics here
        # such as duration, frequency analysis, etc.
        metrics = {}
        
        try:
            import torchaudio
            original_info = torchaudio.info(original_audio)
            processed_info = torchaudio.info(processed_audio)
            
            metrics["original_duration"] = original_info.num_frames / original_info.sample_rate
            metrics["processed_duration"] = processed_info.num_frames / processed_info.sample_rate
            metrics["original_sample_rate"] = original_info.sample_rate
            metrics["processed_sample_rate"] = processed_info.sample_rate
            metrics["duration_change_pct"] = ((metrics["processed_duration"] / metrics["original_duration"]) - 1) * 100
        except Exception as e:
            logfire.warning(f"Could not compute audio metrics: {str(e)}")
            
        return QAResponse(
            meets_requirements=True,  # This will be determined by the model's assessment
            issues=[],  # The model will populate this if there are issues
            suggestions=[],  # The model will provide suggestions if needed
            metrics=metrics,
            reasoning=""  # The model will provide reasoning
        )


@qa_agent.tool
def analyze_audio_properties(
    ctx: RunContext[QAAgentDependencies],
    audio_path: str,
    properties_to_analyze: List[str]
) -> Dict[str, Any]:
    """
    Analyze specific properties of an audio file.
    
    Args:
        audio_path: Path to the audio file to analyze
        properties_to_analyze: List of properties to analyze (e.g. 'duration', 'frequency_spectrum')
        
    Returns:
        Dictionary of property names and their values
    """
    with logfire.span("analyze_audio_properties"):
        if not os.path.exists(audio_path):
            raise ModelRetry(f"Audio file does not exist: {audio_path}")
            
        results = {}
        
        try:
            import torchaudio
            import torch
            import numpy as np
            
            # Load the audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Analyze the requested properties
            for prop in properties_to_analyze:
                if prop == "duration":
                    duration = waveform.shape[1] / sample_rate
                    results["duration"] = duration
                    
                elif prop == "sample_rate":
                    results["sample_rate"] = sample_rate
                    
                elif prop == "num_channels":
                    results["num_channels"] = waveform.shape[0]
                    
                elif prop == "max_amplitude":
                    results["max_amplitude"] = float(torch.max(torch.abs(waveform)).item())
                    
                elif prop == "rms_level":
                    results["rms_level"] = float(torch.sqrt(torch.mean(waveform ** 2)).item())
                    
                elif prop == "frequency_spectrum":
                    # Compute FFT for the first 10 seconds or full duration if shorter
                    max_samples = min(10 * sample_rate, waveform.shape[1])
                    fft = torch.fft.rfft(waveform[:, :max_samples], dim=1)
                    magnitude = torch.abs(fft)
                    
                    # Calculate average magnitude across channels
                    avg_magnitude = torch.mean(magnitude, dim=0).numpy()
                    
                    # Get the frequencies corresponding to the FFT bins
                    freqs = np.fft.rfftfreq(max_samples, 1/sample_rate)
                    
                    # Store only a representative summary (max magnitude and its frequency)
                    max_idx = np.argmax(avg_magnitude)
                    results["peak_frequency"] = float(freqs[max_idx])
                    results["peak_magnitude"] = float(avg_magnitude[max_idx])
                    
                    # Calculate energy in different frequency bands
                    bands = [(20, 200), (200, 2000), (2000, 20000)]
                    for i, (low, high) in enumerate(bands):
                        mask = (freqs >= low) & (freqs <= high)
                        if np.any(mask):
                            band_energy = np.sum(avg_magnitude[mask])
                            results[f"energy_band_{low}_{high}"] = float(band_energy)
        
        except Exception as e:
            logfire.error(f"Error analyzing audio properties: {str(e)}")
            results["error"] = str(e)
            
        return results


@qa_agent.tool
def compare_audio_files(
    ctx: RunContext[QAAgentDependencies],
    original_path: str,
    processed_path: str
) -> Dict[str, Any]:
    """
    Compare properties between the original and processed audio files.
    
    Args:
        original_path: Path to the original audio file
        processed_path: Path to the processed audio file
        
    Returns:
        Dictionary of comparison results
    """
    with logfire.span("compare_audio_files"):
        if not os.path.exists(original_path):
            raise ModelRetry(f"Original audio file does not exist: {original_path}")
            
        if not os.path.exists(processed_path):
            raise ModelRetry(f"Processed audio file does not exist: {processed_path}")
            
        comparison = {}
        
        try:
            # Analyze both files with the same properties
            properties = ["duration", "sample_rate", "num_channels", "max_amplitude", "rms_level"]
            
            original_props = ctx.call_tool(
                "analyze_audio_properties",
                {"audio_path": original_path, "properties_to_analyze": properties}
            )
            
            processed_props = ctx.call_tool(
                "analyze_audio_properties",
                {"audio_path": processed_path, "properties_to_analyze": properties}
            )
            
            # Compare the properties
            for prop in properties:
                if prop in original_props and prop in processed_props:
                    original_val = original_props[prop]
                    processed_val = processed_props[prop]
                    
                    comparison[f"{prop}_original"] = original_val
                    comparison[f"{prop}_processed"] = processed_val
                    
                    if isinstance(original_val, (int, float)) and isinstance(processed_val, (int, float)):
                        if original_val != 0:
                            change_pct = ((processed_val / original_val) - 1) * 100
                            comparison[f"{prop}_change_pct"] = change_pct
            
        except Exception as e:
            logfire.error(f"Error comparing audio files: {str(e)}")
            comparison["error"] = str(e)
            
        return comparison 