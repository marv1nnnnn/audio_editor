"""
Basic audio manipulation functions.
"""
import os
import torchaudio
import numpy as np
from pathlib import Path

from .config import SAMPLE_RATE
from .utils import get_output_path
from .io import READ_AUDIO_NUMPY, WRITE_AUDIO

def LEN(wav_path: str) -> float:
    """Return audio duration in seconds.
    
    Args:
        wav_path (str): Path to the audio file.
        
    Returns:
        float: Duration of the audio in seconds.
    """
    info = torchaudio.info(wav_path)
    return info.num_frames / info.sample_rate

def CLIP(wav_path: str, offset: float, onset: float = 0, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Extract section of audio between onset and offset.
    
    Args:
        wav_path (str): Path to input audio file.
        offset (float): End time in seconds.
        onset (float, optional): Start time in seconds. Defaults to 0.
        out_wav (str, optional): Path to output WAV file. Defaults to auto-generated name.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output WAV file containing the clipped audio.
        
    Raises:
        ValueError: If onset is greater than or equal to offset.
    """
    out_wav = out_wav or get_output_path(wav_path, "clip")
    
    wav_data = READ_AUDIO_NUMPY(wav_path, sr=sr)
    duration_secs = len(wav_data) / sr
    
    # Validate parameters
    onset_sample = int(max(0, onset) * sr)
    offset_sample = int(min(offset, duration_secs) * sr)
    
    if onset_sample >= offset_sample:
        raise ValueError(f"Onset ({onset}s) must be less than offset ({offset}s)")
    
    # Extract segment and write
    clipped_wav = wav_data[onset_sample:offset_sample]
    WRITE_AUDIO(clipped_wav, out_wav, sr)
    return out_wav

def SPLIT(wav_path: str, break_points: list[float], out_wav_prefix: str = None, sr: int = SAMPLE_RATE) -> list[str]:
    """Split audio at specified breakpoints.
    
    Args:
        wav_path (str): Path to input audio file.
        break_points (list[float]): List of timestamps in seconds where to split audio.
        out_wav_prefix (str, optional): Prefix for output files. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        list[str]: List of paths to all output segments.
    """
    out_wav_prefix = out_wav_prefix or f"{Path(wav_path).stem}_split_{get_output_path('', '', '')}"
    target_dir = os.path.dirname(wav_path) or '.'
    
    wav_data = READ_AUDIO_NUMPY(wav_path, sr=sr)
    duration_samples = len(wav_data)
    duration_secs = duration_samples / sr
    
    # Validate and sort breakpoints
    break_points = sorted([p for p in break_points if 0 < p < duration_secs])
    
    # Split and save segments
    results = []
    last_offset = 0
    
    # Process breakpoints
    for i, bp_sec in enumerate(break_points):
        onset = last_offset
        offset = min(int(bp_sec * sr), duration_samples)
        
        if onset >= offset:  # Skip zero-length segments
            continue
            
        # Save segment
        segment = wav_data[onset:offset]
        out_path = os.path.join(target_dir, f"{out_wav_prefix}_{i}.wav")
        WRITE_AUDIO(segment, out_path, sr)
        results.append(out_path)
        last_offset = offset
    
    # Add final segment if needed
    if last_offset < duration_samples:
        segment = wav_data[last_offset:]
        out_path = os.path.join(target_dir, f"{out_wav_prefix}_{len(break_points)}.wav")
        WRITE_AUDIO(segment, out_path, sr)
        results.append(out_path)
    
    return results

def MIX(wavs: list[str], offsets: list[float], out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Mix multiple audio files with specified offsets.
    
    Args:
        wavs (list[str]): List of paths to input audio files.
        offsets (list[float]): List of onset times in seconds for each file.
        out_wav (str, optional): Path to output mixed file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output mixed WAV file.
        
    Raises:
        ValueError: If number of audio files and offsets don't match.
    """
    if not wavs or not offsets or len(wavs) != len(offsets):
        raise ValueError("Must provide equal number of audio files and offset positions")
    
    out_wav = out_wav or f"mixed_{get_output_path('', '', '')}"
    
    # Calculate final length and prepare mix template
    max_length = 0
    audio_segments = []
    
    # Load all audio and determine length
    for wav_path, onset_sec in zip(wavs, offsets):
        wav_data = READ_AUDIO_NUMPY(wav_path, sr=sr)
        onset_sample = int(onset_sec * sr)
        end_sample = onset_sample + len(wav_data)
        max_length = max(max_length, end_sample)
        audio_segments.append({'data': wav_data, 'onset': onset_sample})
    
    # Mix audio
    mix = np.zeros(max_length, dtype=np.float32)
    for item in audio_segments:
        data, onset = item['data'], item['onset']
        segment_len = min(len(data), max_length - onset)
        if segment_len > 0:
            mix[onset:onset + segment_len] += data[:segment_len]
    
    WRITE_AUDIO(mix, out_wav, sr)
    return out_wav

def CAT(wavs: list[str], out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Concatenate multiple audio files.
    
    Args:
        wavs (list[str]): List of paths to input audio files.
        out_wav (str, optional): Path to output concatenated file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output concatenated WAV file.
        
    Raises:
        ValueError: If no valid audio files are provided.
    """
    if not wavs:
        raise ValueError("Must provide at least one audio file")
    
    out_wav = out_wav or f"concatenated_{get_output_path('', '', '')}"
    
    # Load and concatenate
    segments = [READ_AUDIO_NUMPY(wav, sr=sr) for wav in wavs]
    if not segments:
        raise ValueError("No valid audio segments found")
    
    concatenated = np.concatenate(segments)
    WRITE_AUDIO(concatenated, out_wav, sr)
    return out_wav
