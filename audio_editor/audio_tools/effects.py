"""
Audio effects and signal processing functions.
"""
import inspect
from audiomentations import (
    AddGaussianNoise, TimeStretch, PitchShift, Shift,
    LowPassFilter, HighPassFilter, BandPassFilter, Normalize,
    RoomSimulator, PolarityInversion, ClippingDistortion,
    AddColorNoise, Mp3Compression, Trim
)

from .config import SAMPLE_RATE
from .utils import get_output_path
from .io import READ_AUDIO_NUMPY, WRITE_AUDIO

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
