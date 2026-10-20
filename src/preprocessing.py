"""
Signal Preprocessing Module for ECG Signals.
Contains functions for filtering raw ECG signals, removing baseline wander,
attenuating high-frequency noise, segmenting heartbeats, and normalizing waveforms.
"""

import numpy as np
from scipy.signal import butter, filtfilt
import config

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 3):
    """
    Design a digital Butterworth bandpass filter.
    
    Args:
        lowcut: Lower cutoff frequency in Hz (removes baseline wander like respiration)
        highcut: Higher cutoff frequency in Hz (removes muscle noise and powerline interference)
        fs: Sampling frequency of the ECG signal (360 Hz for MIT-BIH)
        order: Filter order (3 is standard, gives steep roll-off without instability)
        
    Returns:
        b, a: Numerator (b) and denominator (a) filter polynomials.
    """
    # Nyquist frequency is half of the sampling frequency
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure normalized frequencies stay within valid (0, 1) interval
    low = max(0.001, min(low, 0.99))
    high = max(0.001, min(high, 0.99))
    
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal: np.ndarray, 
                          fs: float = config.SAMPLING_RATE, 
                          lowcut: float = config.FILTER_LOWCUT, 
                          highcut: float = config.FILTER_HIGHCUT, 
                          order: int = config.FILTER_ORDER) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to ECG signal.
    Uses 'filtfilt' (forward-backward filtering) to ensure ZERO phase distortion.
    
    Why filtering is needed in ECG:
    1. Removes baseline wander caused by patient respiration and motion (< 0.5 Hz).
    2. Removes high-frequency electromyogram (EMG/muscle artifact) and 50/60 Hz powerline noise (> 45 Hz).
    
    Args:
        signal: 1D NumPy array containing raw voltage values.
        fs: Sampling rate (default 360 Hz).
        lowcut: Lower frequency bound (default 0.5 Hz).
        highcut: Upper frequency bound (default 45.0 Hz).
        order: Filter order (default 3).
        
    Returns:
        Filtered 1D ECG signal as NumPy array.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # filtfilt processes forward and backward to prevent phase shifts in R-peaks
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def normalize_segment(segment: np.ndarray) -> np.ndarray:
    """
    Normalize an individual heartbeat segment using Z-score standardization.
    Formula: z = (x - mean) / (std + epsilon)
    
    Why Z-score normalization is used:
    - Sets baseline mean to 0 and standard deviation to 1.
    - Prevents amplitude differences caused by electrode placement, skin impedance,
      or patient physiology from biasing the neural network.
      
    Args:
        segment: 1D array of single heartbeat voltage values.
        
    Returns:
        Normalized 1D array with zero mean and unit variance.
    """
    std = np.std(segment)
    if std < 1e-8:
        # Avoid division by zero if segment is completely flat
        return segment - np.mean(segment)
    return (segment - np.mean(segment)) / std

def extract_beat_segment(signal: np.ndarray, 
                         r_peak_idx: int, 
                         window_before: int = config.WINDOW_BEFORE, 
                         window_after: int = config.WINDOW_AFTER) -> np.ndarray:
    """
    Extracts a fixed-length window around an annotated R-peak location.
    
    Boundary handling:
    If the beat is too close to the beginning or end of the recording,
    it returns None so that incomplete truncated beats are safely ignored.
    
    Args:
        signal: 1D filtered ECG signal array.
        r_peak_idx: Sample index of the annotated R-peak.
        window_before: Number of samples before R-peak (100 samples ~ 277 ms).
        window_after: Number of samples after R-peak (156 samples ~ 433 ms).
        
    Returns:
        1D segment array of length 256, or None if boundary is exceeded.
    """
    start_idx = r_peak_idx - window_before
    end_idx = r_peak_idx + window_after
    
    # Boundary check: ensure segment does not exceed signal boundaries
    if start_idx < 0 or end_idx > len(signal):
        return None
        
    raw_segment = signal[start_idx:end_idx]
    normalized_segment = normalize_segment(raw_segment)
    return normalized_segment
