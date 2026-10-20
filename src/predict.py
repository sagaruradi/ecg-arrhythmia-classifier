"""
Inference and Single-Beat Prediction Module.
Loads the saved 1D-CNN model and classifies individual heartbeat segments,
providing predicted class, confidence percentage, and class probability distribution.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch

import config
from src.model import ECG1DCNN
from src.preprocessing import normalize_segment

_MODEL_CACHE = None

def get_model(model_path: Path = config.MODEL_PATH) -> ECG1DCNN:
    """
    Loads and caches the trained PyTorch model for repeated fast inference.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
        
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run 'python main.py --train' first."
        )
        
    model = ECG1DCNN(in_channels=1, num_classes=len(config.CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    _MODEL_CACHE = model
    return model

def predict_heartbeat(segment: np.ndarray, model_path: Path = config.MODEL_PATH) -> Dict:
    """
    Predicts the arrhythmia category for a single 256-sample ECG heartbeat window.
    
    Args:
        segment: 1D NumPy array of length 256 (or close, will be normalized/checked).
        model_path: Path to model weights.
        
    Returns:
        dict: {
            'class_id': int,
            'class_name': str,
            'confidence': float,
            'confidence_percent': float,
            'probabilities': dict of class_name -> float,
            'description': str
        }
    """
    # Validate segment length
    if len(segment) != config.SEGMENT_LENGTH:
        raise ValueError(f"Input segment length must be {config.SEGMENT_LENGTH}, got {len(segment)}.")
        
    # Ensure segment is normalized
    std = np.std(segment)
    mean = np.mean(segment)
    if std > 1.5 or abs(mean) > 0.5:
        # Segment is likely unnormalized, apply Z-score
        segment = normalize_segment(segment)
        
    model = get_model(model_path)
    
    # Format input tensor: (Batch=1, Channel=1, Length=256)
    input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        
    pred_class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_class_id])
    class_name = config.CLASS_NAMES[pred_class_id]
    
    prob_dict = {
        config.CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(config.CLASS_NAMES))
    }
    
    return {
        'class_id': pred_class_id,
        'class_name': class_name,
        'confidence': round(confidence, 4),
        'confidence_percent': round(confidence * 100, 2),
        'probabilities': prob_dict,
        'description': config.CLASS_DESCRIPTIONS.get(class_name, "")
    }

def get_sample_test_beat(index: Optional[int] = None) -> Tuple[np.ndarray, int, str]:
    """
    Convenience function to fetch a test heartbeat segment along with its true label.
    Useful for quick CLI testing and the Streamlit app.
    """
    test_cache = config.PROCESSED_DATA_DIR / "test_data.npz"
    if not test_cache.exists():
        raise FileNotFoundError("Processed test data not found. Run 'python main.py --prepare' first.")
        
    data = np.load(test_cache, allow_pickle=True)
    segments = data['segments']
    labels = data['labels']
    record_ids = data['record_ids']
    
    if index is None or index >= len(labels) or index < 0:
        index = np.random.randint(0, len(labels))
        
    segment = segments[index]
    true_label = int(labels[index])
    record_id = str(record_ids[index])
    
    return segment, true_label, record_id

if __name__ == "__main__":
    # Test inference on a sample
    try:
        seg, true_lbl, rec = get_sample_test_beat(0)
        res = predict_heartbeat(seg)
        print(f"Sample from Record {rec}: True Label = {config.CLASS_NAMES[true_lbl]}")
        print(f"Prediction: {res['class_name']} ({res['confidence_percent']}%)")
        print(f"Probabilities: {res['probabilities']}")
    except Exception as e:
        print(f"Prediction test note: {e}")
