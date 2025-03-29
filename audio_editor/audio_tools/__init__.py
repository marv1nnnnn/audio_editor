"""
Audio Tools package for audio processing.

This package provides a collection of functions for audio processing, including:
- File I/O
- Audio manipulation
- Effects processing
- Volume adjustment
- AI-powered analysis and generation
"""

# Re-export all functions from modules
from .config import SAMPLE_RATE
from .utils import generate_random_series, get_output_path
from .io import READ_AUDIO_NUMPY, WRITE_AUDIO
from .manipulation import LEN, CLIP, SPLIT, MIX, CAT
from .volume import LOUDNESS_NORM, INC_VOL, DEC_VOL
from .effects import (
    ADD_NOISE, TIME_STRETCH, PITCH_SHIFT, 
    APPLY_LOWPASS, APPLY_HIGHPASS, APPLY_BANDPASS,
    AUDIO_CLIP_DISTORT, APPLY_SHIFT, NORMALIZE_AUDIO,
    INVERT_POLARITY, ADD_COLOR_NOISE, APPLY_MP3_COMPRESSION,
    TRIM_SILENCE, APPLY_ROOM_SIMULATOR
)
from .ai import AUDIO_QA, AUDIO_DIFF, AUDIO_GENERATE

__all__ = [
    # Config
    'SAMPLE_RATE',
    
    # Utils
    'generate_random_series', 'get_output_path',
    
    # I/O
    'READ_AUDIO_NUMPY', 'WRITE_AUDIO',
    
    # Manipulation
    'LEN', 'CLIP', 'SPLIT', 'MIX', 'CAT',
    
    # Volume
    'LOUDNESS_NORM', 'INC_VOL', 'DEC_VOL',
    
    # Effects
    'ADD_NOISE', 'TIME_STRETCH', 'PITCH_SHIFT',
    'APPLY_LOWPASS', 'APPLY_HIGHPASS', 'APPLY_BANDPASS',
    'AUDIO_CLIP_DISTORT', 'APPLY_SHIFT', 'NORMALIZE_AUDIO',
    'INVERT_POLARITY', 'ADD_COLOR_NOISE', 'APPLY_MP3_COMPRESSION',
    'TRIM_SILENCE', 'APPLY_ROOM_SIMULATOR',
    
    # AI
    'AUDIO_QA', 'AUDIO_DIFF', 'AUDIO_GENERATE'
]
