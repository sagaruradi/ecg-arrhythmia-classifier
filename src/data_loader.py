"""
Data Loader Module for MIT-BIH Arrhythmia Database.
Loads ECG signals (.dat, .hea) and annotations (.atr) using WFDB,
segments heartbeats, maps them according to the AAMI standard,
and partitions them by patient recording to prevent data leakage.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import wfdb
from collections import Counter

import config
from src.preprocessing import apply_bandpass_filter, extract_beat_segment

def get_available_records(data_dir: Optional[Path] = None) -> List[str]:
    """
    Scans the data directory and returns a sorted list of record IDs that have
    both .hea (header) and .atr (annotation) files available.
    """
    if data_dir is None:
        data_dir = config.get_raw_data_dir()
        
    records = []
    if not data_dir.exists():
        return records
        
    for file in os.listdir(data_dir):
        if file.endswith(".hea"):
            record_id = file.replace(".hea", "")
            # Check if matching .atr exists
            if os.path.exists(os.path.join(data_dir, f"{record_id}.atr")):
                records.append(record_id)
                
    return sorted(records)

def load_record_signal(record_name: str, data_dir: Optional[Path] = None) -> Tuple[np.ndarray, int, str]:
    """
    Loads raw ECG signal and sampling frequency for a single record.
    Prioritizes the standard modified limb lead II ('MLII') if present.
    
    Args:
        record_name: Record ID string (e.g. '100', '208').
        data_dir: Path to directory containing MIT-BIH files.
        
    Returns:
        signal (1D np.ndarray), fs (int), lead_name (str)
    """
    if data_dir is None:
        data_dir = config.get_raw_data_dir()
        
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    
    # Select lead MLII if available, else first channel
    channel_idx = 0
    if "MLII" in record.sig_name:
        channel_idx = record.sig_name.index("MLII")
    elif "ML2" in record.sig_name:
        channel_idx = record.sig_name.index("ML2")
        
    lead_name = record.sig_name[channel_idx]
    signal = record.p_signal[:, channel_idx]
    fs = record.fs
    
    return signal, fs, lead_name

def load_record_annotations(record_name: str, data_dir: Optional[Path] = None):
    """
    Reads expert physician heartbeat annotations from .atr file.
    
    Returns:
        wfdb.Annotation object containing sample indices and symbols.
    """
    if data_dir is None:
        data_dir = config.get_raw_data_dir()
        
    record_path = os.path.join(data_dir, record_name)
    annotation = wfdb.rdann(record_path, 'atr')
    return annotation

def process_record(record_name: str, data_dir: Optional[Path] = None) -> Dict:
    """
    Processes one record:
    1. Loads raw signal and annotations.
    2. Filters signal using Butterworth bandpass filter.
    3. Segments valid heartbeats around R-peaks (length 256).
    4. Maps annotation symbols into 4 AAMI classes (N, S, V, F).
    5. Normalizes each heartbeat segment.
    
    Returns:
        Dictionary with segments (N, 256), labels (N,), r_peaks (N,), symbols (N,)
    """
    signal, fs, lead_name = load_record_signal(record_name, data_dir)
    annotation = load_record_annotations(record_name, data_dir)
    
    # Step 1: Bandpass filter signal
    filtered_signal = apply_bandpass_filter(signal, fs=fs)
    
    segments = []
    labels = []
    r_peaks = []
    symbols = []
    ignored_symbols = Counter()
    
    for r_idx, symbol in zip(annotation.sample, annotation.symbol):
        # Only process beats present in our documented AAMI standard mapping
        if symbol in config.AAMI_MAPPING:
            seg = extract_beat_segment(filtered_signal, r_idx)
            if seg is not None:
                segments.append(seg)
                labels.append(config.AAMI_MAPPING[symbol])
                r_peaks.append(r_idx)
                symbols.append(symbol)
        else:
            ignored_symbols[symbol] += 1
            
    return {
        'record': record_name,
        'segments': np.array(segments, dtype=np.float32),
        'labels': np.array(labels, dtype=np.int64),
        'r_peaks': np.array(r_peaks, dtype=np.int64),
        'symbols': symbols,
        'ignored_symbols': dict(ignored_symbols)
    }

def prepare_dataset(train_records: List[str] = config.TRAIN_RECORDS,
                    val_records: List[str] = config.VAL_RECORDS,
                    test_records: List[str] = config.TEST_RECORDS,
                    cache: bool = True,
                    force_rebuild: bool = False) -> Tuple[Dict, Dict, Dict]:
    """
    Extracts, segments, and partitions data strictly by recording to prevent data leakage.
    Saves processed arrays to data/processed for ultra-fast reloading.
    
    Args:
        train_records: List of record IDs for training.
        val_records: List of record IDs for validation.
        test_records: List of record IDs for testing.
        cache: Whether to save and load cached npz files.
        force_rebuild: If True, forces reprocessing even if cache exists.
        
    Returns:
        train_data, val_data, test_data dictionaries.
    """
    processed_dir = config.PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_cache = processed_dir / "train_data.npz"
    val_cache = processed_dir / "val_data.npz"
    test_cache = processed_dir / "test_data.npz"
    
    # Check if cache exists
    if cache and not force_rebuild and train_cache.exists() and val_cache.exists() and test_cache.exists():
        print("[INFO] Loading preprocessed data from cache...")
        train_data = dict(np.load(train_cache, allow_pickle=True))
        val_data = dict(np.load(val_cache, allow_pickle=True))
        test_data = dict(np.load(test_cache, allow_pickle=True))
        print(f"[INFO] Loaded Train: {len(train_data['labels'])} beats, "
              f"Val: {len(val_data['labels'])} beats, "
              f"Test: {len(test_data['labels'])} beats.")
        return train_data, val_data, test_data

    raw_dir = config.get_raw_data_dir()
    print(f"[INFO] Preparing dataset from: {raw_dir}")
    print(f"[INFO] Train records: {train_records}")
    print(f"[INFO] Validation records: {val_records}")
    print(f"[INFO] Test records: {test_records}")
    
    def process_record_group(record_list: List[str], group_name: str) -> Dict:
        all_segments = []
        all_labels = []
        all_record_ids = []
        
        for rec in record_list:
            rec_hea = raw_dir / f"{rec}.hea"
            if not rec_hea.exists():
                print(f"[WARNING] Record {rec} not found at {raw_dir}. Skipping.")
                continue
                
            print(f"  -> Processing {group_name} Record {rec}...")
            res = process_record(rec, raw_dir)
            n_beats = len(res['labels'])
            print(f"     Extracted {n_beats} valid beats. (Ignored non-beat markers: {res['ignored_symbols']})")
            
            if n_beats > 0:
                all_segments.append(res['segments'])
                all_labels.append(res['labels'])
                all_record_ids.extend([rec] * n_beats)
                
        if all_segments:
            final_segments = np.vstack(all_segments)
            final_labels = np.concatenate(all_labels)
        else:
            final_segments = np.empty((0, config.SEGMENT_LENGTH), dtype=np.float32)
            final_labels = np.empty((0,), dtype=np.int64)
            
        return {
            'segments': final_segments,
            'labels': final_labels,
            'record_ids': np.array(all_record_ids)
        }
        
    train_data = process_record_group(train_records, "TRAIN")
    val_data = process_record_group(val_records, "VAL")
    test_data = process_record_group(test_records, "TEST")
    
    if cache:
        print("[INFO] Saving cached datasets to data/processed/...")
        np.savez_compressed(train_cache, **train_data)
        np.savez_compressed(val_cache, **val_data)
        np.savez_compressed(test_cache, **test_data)
        print("[INFO] Cached files successfully saved.")
        
    return train_data, val_data, test_data

if __name__ == "__main__":
    prepare_dataset(force_rebuild=True)
