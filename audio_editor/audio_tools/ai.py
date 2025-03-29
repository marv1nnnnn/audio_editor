"""
AI-powered audio analysis and generation functions.
"""
import os
import torch
import diffusers
import logfire
import numpy as np
from google import genai

from .config import SAMPLE_RATE
from .io import WRITE_AUDIO

# Lazy-loaded AudioLDM pipeline
_audio_gen_pipe = None

def AUDIO_QA(wav_path: str, task: str) -> str:
    """Analyze audio and answer questions using Gemini AI.
    
    Args:
        wav_path (str): Path to input audio file.
        task (str): Question or task for the AI to address.
        
    Returns:
        str: Text response from Gemini AI.
        
    Raises:
        FileNotFoundError: If audio file doesn't exist.
        ValueError: If no task is provided.
    """
    if not wav_path or not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    if not task:
        raise ValueError("A task/question is required")
    
    # Enhance generic analysis requests to be more specific
    if any(phrase in task.lower() for phrase in ["sound good", "does it sound", "how does it sound"]):
        task = f"""Critically analyze this audio file and provide a detailed assessment.
Please be specific about:
1. Audio quality (clarity, artifacts, noise)
2. Frequency balance (bass, mids, highs)
3. Overall coherence and professional quality
4. Issues that need fixing
5. Specific improvement recommendations

Original question: {task}"""
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        
        # Upload audio and generate analysis
        audio_file = client.files.upload(file=wav_path)
        response = client.models.generate_content(
            model='gemini-2.0-flash-thinking-exp',
            contents=[task, audio_file]
        )
        
        if not response.candidates:
            return "No response generated from Gemini."
            
        return response.text.strip()
    except Exception as e:
        logfire.error(f"AUDIO_QA failed: {e}")
        return f"Error analyzing audio: {e}"

def AUDIO_DIFF(wav_paths: list[str], task: str = None) -> str:
    """Compare multiple audio files and analyze differences using Gemini.
    
    Args:
        wav_paths (list[str]): List of paths to audio files to compare.
        task (str, optional): Specific comparison instructions. Defaults to None.
        
    Returns:
        str: Text comparison from Gemini AI.
        
    Raises:
        ValueError: If fewer than 2 audio files are provided.
        FileNotFoundError: If any audio file doesn't exist.
    """
    if not wav_paths or len(wav_paths) < 2:
        raise ValueError("At least 2 audio files required for comparison")
        
    # Validate all files exist
    for path in wav_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Default comparison task if none provided
    if not task:
        task = f"""Compare these {len(wav_paths)} audio files and analyze differences:
1. Key sonic differences
2. Audio quality comparison
3. Frequency balance differences 
4. Processing differences
5. Which file is better and why
6. Technical differences (volume, clarity, noise)"""
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        
        # Upload all audio files
        audio_files = [client.files.upload(file=path) for path in wav_paths]
        
        # Generate comparison analysis
        response = client.models.generate_content(
            model='gemini-2.0-flash-thinking-exp',
            contents=[task, *audio_files]
        )
        
        if not response.candidates:
            return "No response generated from Gemini."
            
        return response.text.strip()
    except Exception as e:
        logfire.error(f"AUDIO_DIFF failed: {e}")
        return f"Error comparing audio files: {e}"

def AUDIO_GENERATE(text: str, filename: str, audio_length_in_s: float = 5.0, guidance_scale: float = 3.0, num_inference_steps: int = 100, sr: int = SAMPLE_RATE) -> str:
    """Generate audio from text description using AudioLDM.
    
    Args:
        text (str): Text description of the audio to generate.
        filename (str): Path to save the generated audio file.
        audio_length_in_s (float, optional): Length of generated audio in seconds. Defaults to 5.0.
        guidance_scale (float, optional): How closely to follow the prompt. Defaults to 3.0.
        num_inference_steps (int, optional): Number of diffusion steps. Defaults to 100.
        sr (int, optional): Sample rate (ignored, uses fixed SAMPLE_RATE). Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to the generated audio file.
        
    Raises:
        RuntimeError: If AudioLDM model fails to load.
    """
    # AudioLDM only works with the built-in sample rate
    if sr != SAMPLE_RATE:
        logfire.warning(f"AudioLDM fixed at {SAMPLE_RATE}Hz. Requested {sr}Hz ignored.")
    
    # Ensure output filename has .wav extension
    if not filename.lower().endswith(".wav"):
        filename += ".wav"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Load model if needed
    global _audio_gen_pipe
    if _audio_gen_pipe is None:
        try:
            _audio_gen_pipe = diffusers.AudioLDMPipeline.from_pretrained(
                "cvssp/audioldm-s-full-v2", 
                torch_dtype=torch.float16
            )
            _audio_gen_pipe = _audio_gen_pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load AudioLDM model: {e}")
    
    # Generate audio
    audio_waveform = _audio_gen_pipe(
        prompt=text,
        audio_length_in_s=audio_length_in_s,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).audios[0]
    
    # Convert to numpy if needed
    if not isinstance(audio_waveform, np.ndarray) and hasattr(audio_waveform, "numpy"):
        audio_waveform = audio_waveform.numpy()
    
    # Save the generated audio
    WRITE_AUDIO(audio_waveform, filename, SAMPLE_RATE)
    return filename
