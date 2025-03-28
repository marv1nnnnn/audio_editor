import os
import random
import string
import numpy as np
import torchaudio
import pyloudnorm as pyln
from scipy.io.wavfile import write
import diffusers
import torch
import inspect
import logfire
from pathlib import Path
from google import generativeai, genai
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift,
    Gain, LowPassFilter, HighPassFilter, BandPassFilter, Normalize,
    RoomSimulator, PolarityInversion, ClippingDistortion,
    AddColorNoise, AddBackgroundNoise, AddShortNoises, Mp3Compression,
    LowShelfFilter, HighShelfFilter, Trim, PeakingFilter
)

# --- Configuration ---
SAMPLE_RATE = 16000

# --- Utility Functions ---
def generate_random_series(n=9):
    """Generate a random string for filename uniqueness.
    
    Args:
        n (int, optional): Length of the random string. Defaults to 9.
        
    Returns:
        str: Random string of uppercase letters and digits.
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

def get_output_path(wav_path, suffix, ext=".wav"):
    """Generate output path with a suffix and random string.
    
    Args:
        wav_path (str, optional): Path to input audio file to base output name on.
        suffix (str): Descriptive suffix to add to filename.
        ext (str, optional): File extension. Defaults to ".wav".
        
    Returns:
        str: Generated output path.
    """
    if wav_path:
        return f"{Path(wav_path).stem}_{suffix}_{generate_random_series()}{ext}"
    return f"{suffix}_{generate_random_series()}{ext}"

# --- File I/O ---
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

# --- Audio Information & Basic Manipulation ---
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
    out_wav_prefix = out_wav_prefix or f"{Path(wav_path).stem}_split_{generate_random_series()}"
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

# --- Audio Combining Functions ---
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
    
    out_wav = out_wav or f"mixed_{generate_random_series()}.wav"
    
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
    
    out_wav = out_wav or f"concatenated_{generate_random_series()}.wav"
    
    # Load and concatenate
    segments = [READ_AUDIO_NUMPY(wav, sr=sr) for wav in wavs]
    if not segments:
        raise ValueError("No valid audio segments found")
    
    concatenated = np.concatenate(segments)
    WRITE_AUDIO(concatenated, out_wav, sr)
    return out_wav

# --- Volume Processing ---
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

# --- Audio Effect Processing ---
def _apply_effect(transform_class, wav_path: str, out_wav: str = None, sr: int = SAMPLE_RATE, **kwargs) -> str:
    """Generic function to apply an audiomentations effect.
    
    Args:
        transform_class: Audiomentations transform class to instantiate.
        wav_path (str): Path to input audio file.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        **kwargs: Parameters to pass to the transform.
        
    Returns:
        str: Path to output WAV file.
    """
    transform_name = transform_class.__name__.lower()
    out_wav = out_wav or get_output_path(wav_path, transform_name)
    
    # Read audio
    wav_data = READ_AUDIO_NUMPY(wav_path, sr=sr)
    
    # Ensure effect always applies
    kwargs.setdefault('p', 1.0)
    
    # Keep only valid parameters for the transform
    sig = inspect.signature(transform_class.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    # Apply transform and save
    transform = transform_class(**valid_kwargs)
    processed = transform(samples=wav_data, sample_rate=sr)
    WRITE_AUDIO(processed, out_wav, sr)
    return out_wav

# --- Define audio effects using the common _apply_effect function ---
def ADD_NOISE(wav_path: str, min_amplitude: float = 0.001, max_amplitude: float = 0.015, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Add gaussian noise.
    
    Args:
        wav_path (str): Path to input audio file.
        min_amplitude (float, optional): Minimum noise amplitude. Defaults to 0.001.
        max_amplitude (float, optional): Maximum noise amplitude. Defaults to 0.015.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output WAV file with added noise.
    """
    return _apply_effect(AddGaussianNoise, wav_path, out_wav, sr, min_amplitude=min_amplitude, max_amplitude=max_amplitude)

def TIME_STRETCH(wav_path: str, min_rate: float = 0.8, max_rate: float = 1.25, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Time stretch audio without changing pitch.
    
    Args:
        wav_path (str): Path to input audio file.
        min_rate (float, optional): Minimum stretch rate (0.5 = half speed). Defaults to 0.8.
        max_rate (float, optional): Maximum stretch rate (2.0 = double speed). Defaults to 1.25.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output time-stretched WAV file.
    """
    return _apply_effect(TimeStretch, wav_path, out_wav, sr, min_rate=min_rate, max_rate=max_rate)

def PITCH_SHIFT(wav_path: str, min_semitones: int = -4, max_semitones: int = 4, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Shift audio pitch.
    
    Args:
        wav_path (str): Path to input audio file.
        min_semitones (int, optional): Minimum semitones to shift. Defaults to -4.
        max_semitones (int, optional): Maximum semitones to shift. Defaults to 4.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output pitch-shifted WAV file.
    """
    return _apply_effect(PitchShift, wav_path, out_wav, sr, min_semitones=min_semitones, max_semitones=max_semitones)

def APPLY_LOWPASS(wav_path: str, min_cutoff_freq: float = 1500.0, max_cutoff_freq: float = 7500.0, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Apply low-pass filter.
    
    Args:
        wav_path (str): Path to input audio file.
        min_cutoff_freq (float, optional): Minimum cutoff frequency. Defaults to 1500.0.
        max_cutoff_freq (float, optional): Maximum cutoff frequency. Defaults to 7500.0.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output filtered WAV file.
    """
    return _apply_effect(LowPassFilter, wav_path, out_wav, sr, min_cutoff_freq=min_cutoff_freq, max_cutoff_freq=max_cutoff_freq)

def APPLY_HIGHPASS(wav_path: str, min_cutoff_freq: float = 20.0, max_cutoff_freq: float = 400.0, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Apply high-pass filter.
    
    Args:
        wav_path (str): Path to input audio file.
        min_cutoff_freq (float, optional): Minimum cutoff frequency. Defaults to 20.0.
        max_cutoff_freq (float, optional): Maximum cutoff frequency. Defaults to 400.0.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output filtered WAV file.
    """
    return _apply_effect(HighPassFilter, wav_path, out_wav, sr, min_cutoff_freq=min_cutoff_freq, max_cutoff_freq=max_cutoff_freq)

def APPLY_BANDPASS(wav_path: str, min_center_freq: float = 200.0, max_center_freq: float = 4000.0, min_bandwidth_fraction: float = 0.5, max_bandwidth_fraction: float = 1.9, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Apply band-pass filter.
    
    Args:
        wav_path (str): Path to input audio file.
        min_center_freq (float, optional): Minimum center frequency. Defaults to 200.0.
        max_center_freq (float, optional): Maximum center frequency. Defaults to 4000.0.
        min_bandwidth_fraction (float, optional): Minimum bandwidth fraction. Defaults to 0.5.
        max_bandwidth_fraction (float, optional): Maximum bandwidth fraction. Defaults to 1.9.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output filtered WAV file.
    """
    return _apply_effect(BandPassFilter, wav_path, out_wav, sr, 
                         min_center_freq=min_center_freq, max_center_freq=max_center_freq,
                         min_bandwidth_fraction=min_bandwidth_fraction, max_bandwidth_fraction=max_bandwidth_fraction)

def AUDIO_CLIP_DISTORT(wav_path: str, min_percentile_threshold: int = 0, max_percentile_threshold: int = 40, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Apply clipping distortion.
    
    Args:
        wav_path (str): Path to input audio file.
        min_percentile_threshold (int, optional): Minimum percentile for clipping. Defaults to 0.
        max_percentile_threshold (int, optional): Maximum percentile for clipping. Defaults to 40.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output distorted WAV file.
    """
    return _apply_effect(ClippingDistortion, wav_path, out_wav, sr, 
                         min_percentile_threshold=min_percentile_threshold, max_percentile_threshold=max_percentile_threshold)

def APPLY_SHIFT(wav_path: str, min_fraction: float = -0.25, max_fraction: float = 0.25, rollover: bool = True, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Shift audio in time.
    
    Args:
        wav_path (str): Path to input audio file.
        min_fraction (float, optional): Minimum shift fraction. Defaults to -0.25.
        max_fraction (float, optional): Maximum shift fraction. Defaults to 0.25.
        rollover (bool, optional): Whether to roll over audio that gets shifted. Defaults to True.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output shifted WAV file.
    """
    return _apply_effect(Shift, wav_path, out_wav, sr, min_fraction=min_fraction, max_fraction=max_fraction, rollover=rollover)

def NORMALIZE_AUDIO(wav_path: str, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Normalize audio using RMS.
    
    Args:
        wav_path (str): Path to input audio file.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output normalized WAV file.
    """
    return _apply_effect(Normalize, wav_path, out_wav, sr)

def INVERT_POLARITY(wav_path: str, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Invert audio polarity.
    
    Args:
        wav_path (str): Path to input audio file.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output inverted WAV file.
    """
    return _apply_effect(PolarityInversion, wav_path, out_wav, sr)

def ADD_COLOR_NOISE(wav_path: str, min_snr_in_db: float = 3.0, max_snr_in_db: float = 30.0, noise_color: str = "pink", out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Add colored noise.
    
    Args:
        wav_path (str): Path to input audio file.
        min_snr_in_db (float, optional): Minimum signal-to-noise ratio in dB. Defaults to 3.0.
        max_snr_in_db (float, optional): Maximum signal-to-noise ratio in dB. Defaults to 30.0.
        noise_color (str, optional): Color of noise. Defaults to "pink".
            Options: "white", "pink", "blue", "brown", "violet".
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output WAV file with added colored noise.
        
    Raises:
        ValueError: If noise_color is not a valid option.
    """
    valid_colors = ["white", "pink", "blue", "brown", "violet"]
    if noise_color not in valid_colors:
        raise ValueError(f"Invalid noise_color '{noise_color}'. Must be one of {valid_colors}")
    return _apply_effect(AddColorNoise, wav_path, out_wav, sr, 
                         min_snr_in_db=min_snr_in_db, max_snr_in_db=max_snr_in_db, 
                         noise_colors=[noise_color])

def APPLY_MP3_COMPRESSION(wav_path: str, min_bitrate: int = 32, max_bitrate: int = 128, backend: str = "lameenc", out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Apply MP3 compression artifacts.
    
    Args:
        wav_path (str): Path to input audio file.
        min_bitrate (int, optional): Minimum bitrate in kbps. Defaults to 32.
        max_bitrate (int, optional): Maximum bitrate in kbps. Defaults to 128.
        backend (str, optional): Backend to use. Defaults to "lameenc".
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output compressed WAV file.
    """
    return _apply_effect(Mp3Compression, wav_path, out_wav, sr, min_bitrate=min_bitrate, max_bitrate=max_bitrate, backend=backend)

def TRIM_SILENCE(wav_path: str, top_db: float = 20.0, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Trim leading/trailing silence.
    
    Args:
        wav_path (str): Path to input audio file.
        top_db (float, optional): Threshold in decibels. Defaults to 20.0.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output trimmed WAV file.
    """
    return _apply_effect(Trim, wav_path, out_wav, sr, top_db=top_db)

def APPLY_ROOM_SIMULATOR(wav_path: str, out_wav: str = None, sr: int = SAMPLE_RATE) -> str:
    """Apply room reverberation effect.
    
    Args:
        wav_path (str): Path to input audio file.
        out_wav (str, optional): Path to output file. Defaults to auto-generated.
        sr (int, optional): Sample rate. Defaults to SAMPLE_RATE.
        
    Returns:
        str: Path to output WAV file with room simulation.
    """
    return _apply_effect(RoomSimulator, wav_path, out_wav, sr)

# --- AI Analysis Tools ---
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

# --- AI Generation ---
# Lazy-loaded AudioLDM pipeline
_audio_gen_pipe = None

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
