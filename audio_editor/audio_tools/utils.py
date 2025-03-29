"""
Utility functions for audio processing.
"""
import os
import random
import string
from pathlib import Path
import inspect

from .config import SAMPLE_RATE

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
