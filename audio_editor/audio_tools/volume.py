"""
Volume processing and normalization functions.
"""
import torchaudio
import numpy as np
import pyloudnorm as pyln

from .config import SAMPLE_RATE
from .utils import get_output_path
from .io import READ_AUDIO_NUMPY, WRITE_AUDIO

def LOUDNESS_NORM(wav_path: str, volume: float = -23.0, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Normalize to target LUFS loudness level.
    
    Args:
        wav_path (str): Path to input audio file.
        volume (float, optional): Target loudness in LUFS. Defaults to -23.0 (broadcast standard).
        out_wav (str, optional): Path to output normalized file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output normalized WAV file.
    """
    out_wav = out_wav or get_output_path(wav_path, "lnorm")
    
    wav_data = READ_AUDIO_NUMPY(wav_path, sr=sr)
    
    # Skip processing if audio is silent
    if np.max(np.abs(wav_data)) < 1e-6:
        WRITE_AUDIO(wav_data, out_wav, sr)
        return out_wav
    
    # Apply loudness normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav_data)
    normalized = pyln.normalize.loudness(wav_data, loudness, volume)
    
    WRITE_AUDIO(normalized, out_wav, sr)
    return out_wav

def INC_VOL(wav_path: str, gain_db: float, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Increase volume by gain_db decibels.
    
    Args:
        wav_path (str): Path to input audio file.
        gain_db (float): Amount to increase volume in dB (must be positive).
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output WAV file.
        
    Raises:
        ValueError: If gain_db is not positive.
    """
    if gain_db <= 0:
        raise ValueError("gain_db must be positive")
    
    out_wav = out_wav or get_output_path(wav_path, "volinc")
    return _apply_gain(wav_path, gain_db, out_wav, sr)

def DEC_VOL(wav_path: str, gain_db: float, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Decrease volume by gain_db decibels.
    
    Args:
        wav_path (str): Path to input audio file.
        gain_db (float): Amount to decrease volume in dB (must be positive).
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output WAV file.
        
    Raises:
        ValueError: If gain_db is not positive.
    """
    if gain_db <= 0:
        raise ValueError("gain_db must be positive")
    
    out_wav = out_wav or get_output_path(wav_path, "voldec")
    return _apply_gain(wav_path, -gain_db, out_wav, sr)

def _apply_gain(wav_path: str, gain_db: float, out_wav: str, sr: int) -> str:
    """Apply gain in dB to audio file.
    
    Args:
        wav_path (str): Path to input audio file.
        gain_db (float): Amount of gain to apply in dB (positive or negative).
        out_wav (str): Path to output file.
        sr (int): Sample rate.
        
    Returns:
        str: Path to output WAV file.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=sr)
    
    # Apply gain and save
    waveform = torchaudio.functional.gain(waveform, gain_db=gain_db)
    WRITE_AUDIO(waveform[0].numpy(), out_wav, sr)
    return out_wav
