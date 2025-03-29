"""
Audio file input/output functions.
"""
import os
import numpy as np
import torchaudio
import logfire
from scipy.io.wavfile import write

from .config import SAMPLE_RATE

def READ_AUDIO_NUMPY(wav_path: str, sr: int = SAMPLE_RATE) -> np.array:
    """Read audio file into numpy array.
    
    Args:
        wav_path (str): Path to input audio file.
        sr (int, optional): Target sample rate for reading. Defaults to SAMPLE_RATE.
        
    Returns:
        np.array: Audio data as numpy array (mono, float32).
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != sr:
        logfire.info(f"Resampling {wav_path} from {sample_rate}Hz to {sr}Hz")
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=sr)
    
    # Return first channel as mono
    return waveform[0].numpy()

def WRITE_AUDIO(wav: np.array, name: str, sr: int = SAMPLE_RATE):
    """Write numpy array to WAV file.
    
    Args:
        wav (np.array): Audio data as numpy array.
        name (str): Path to output WAV file.
        sr (int, optional): Sample rate to write at. Defaults to SAMPLE_RATE.
        
    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If output filename is empty.
    """
    if not isinstance(wav, np.ndarray):
        raise TypeError(f"Input must be a numpy array, got {type(wav)}")
    if not name:
        raise ValueError("Output filename cannot be empty")

    # Create directory if needed
    os.makedirs(os.path.dirname(name) or '.', exist_ok=True)
    
    # Handle multichannel audio
    if len(wav.shape) > 1:
        wav = wav[0]  # Take first channel
    
    # Clean data (handle NaNs, Infs, and clipping)
    if np.isnan(wav).any() or np.isinf(wav).any():
        wav = np.nan_to_num(wav)
    
    # Handle clipping
    max_val = np.max(np.abs(wav)) if wav.size > 0 else 0
    if max_val > 1.0:
        wav = wav * (0.99 / max_val)
    
    # Ensure within valid range and convert to int16
    wav = np.clip(wav, -1.0, 1.0)
    scaled_wav = np.round(wav * 32767).astype(np.int16)
    
    # Write to file
    write(name, sr, scaled_wav)
