"""
Configuration file for ECG Arrhythmia Signal Classification Project.
Contains dataset paths, preprocessing parameters, model architecture parameters,
and training settings. Designed to be beginner-friendly, clean, and easily modifiable.
"""

import os
from pathlib import Path

# Prevent OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset locations (Auto-detects mit-bih-arrhythmia-database-1.0.0 or data/raw)
LOCAL_MITDB_DIR = BASE_DIR / "mit-bih-arrhythmia-database-1.0.0"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

def get_raw_data_dir() -> Path:
    """Returns the valid directory containing the MIT-BIH dataset files."""
    if LOCAL_MITDB_DIR.exists() and (LOCAL_MITDB_DIR / "100.hea").exists():
        return LOCAL_MITDB_DIR
    elif RAW_DATA_DIR.exists() and (RAW_DATA_DIR / "100.hea").exists():
        return RAW_DATA_DIR
    return LOCAL_MITDB_DIR

DATA_DIR = get_raw_data_dir()
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Saved artifact paths
MODEL_PATH = MODELS_DIR / "ecg_1dcnn.pth"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# Signal processing parameters
SAMPLING_RATE = 360          # MIT-BIH sampling rate is 360 Hz
FILTER_LOWCUT = 0.5          # Remove low-frequency baseline wander (< 0.5 Hz)
FILTER_HIGHCUT = 45.0        # Remove high-frequency noise & 50/60 Hz powerline (> 45 Hz)
FILTER_ORDER = 3             # 3rd-order Butterworth bandpass filter

# Heartbeat window parameters (around R-peak annotation)
# Total segment length = 100 + 156 = 256 samples (~711 milliseconds at 360 Hz)
WINDOW_BEFORE = 100          # 100 samples before R-peak (~277 ms)
WINDOW_AFTER = 156           # 156 samples after R-peak (~433 ms)
SEGMENT_LENGTH = WINDOW_BEFORE + WINDOW_AFTER  # 256 samples

# AAMI Heartbeat Classification Mapping (Standard 4-class problem)
# Maps clinical MIT-BIH annotation symbols into 4 major classes:
# N: Normal / bundle branch block beat
# S: Supraventricular ectopic beat
# V: Ventricular ectopic beat
# F: Fusion beat
AAMI_MAPPING = {
    # Normal and bundle branch blocks
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    # Supraventricular ectopic beats
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # Ventricular ectopic beats
    'V': 2, 'E': 2, '!': 2,
    # Fusion beats
    'F': 3
}

# Human-readable class names and descriptions
CLASS_NAMES = [
    'Normal (N)',
    'Supraventricular (S)',
    'Ventricular (V)',
    'Fusion (F)'
]

CLASS_SHORT_NAMES = ['N', 'S', 'V', 'F']

CLASS_DESCRIPTIONS = {
    'Normal (N)': 'Normal sinus rhythm, bundle branch blocks (LBBB/RBBB), or nodal escape beats.',
    'Supraventricular (S)': 'Premature beats triggered in atria or AV node (Atrial Premature Beats).',
    'Ventricular (V)': 'Premature beats originating in ventricles (Premature Ventricular Contractions).',
    'Fusion (F)': 'Hybrid beat occurring when ventricular and normal impulses collide.'
}

# Patient / Recording Level Split (STRICTLY PREVENTS DATA LEAKAGE)
# Patient recordings are partitioned independently across train, val, and test splits.
# All 4 AAMI classes are represented in train, val, and test splits.
TRAIN_RECORDS = ['100', '106', '118', '201', '208', '209', '210']
VAL_RECORDS = ['119', '223']
TEST_RECORDS = ['200', '213', '214', '222']

ALL_RECORDS = TRAIN_RECORDS + VAL_RECORDS + TEST_RECORDS

# Model and training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10                  # 10 epochs for optimal convergence on CPU
RANDOM_SEED = 42
DEVICE = 'cpu'               # Kept strictly CPU-compatible for academic reproducibility
