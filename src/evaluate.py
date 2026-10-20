"""
Evaluation and Metrics Generation Module.
Evaluates the trained 1D-CNN on held-out patient test records,
computes genuine classification metrics (Accuracy, Precision, Recall, F1),
generates dark-theme compatible confusion matrix and waveform plots, and saves results.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)

import config
from src.data_loader import prepare_dataset, load_record_signal, load_record_annotations
from src.preprocessing import apply_bandpass_filter
from src.model import ECG1DCNN

def evaluate_model(model_path: Path = config.MODEL_PATH) -> dict:
    """
    Evaluates the trained model on the unseen test dataset.
    
    Args:
        model_path: Path to saved PyTorch model checkpoint.
        
    Returns:
        metrics: Dictionary containing accuracy, precision, recall, f1, and confusion matrix.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at '{model_path}'. "
            "Please train the model first by running: python main.py --train"
        )
        
    print("=" * 65)
    print("      ECG 1D-CNN ARRHYTHMIA CLASSIFICATION - EVALUATION")
    print("=" * 65)
    
    # 1. Load Preprocessed Data
    train_data, val_data, test_data = prepare_dataset(
        train_records=config.TRAIN_RECORDS,
        val_records=config.VAL_RECORDS,
        test_records=config.TEST_RECORDS
    )
    X_test, y_test = test_data['segments'], test_data['labels']
    
    print(f"\n[EVALUATION DATASET (HELD-OUT PATIENT RECORDS)]")
    print(f"  Test Patient Records: {config.TEST_RECORDS}")
    print(f"  Total Test Heartbeats: {len(y_test):,} beats")
    
    # 2. Load Model
    model = ECG1DCNN(in_channels=1, num_classes=len(config.CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 3. Predict on Test Set
    test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        logits = model(test_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()
        y_pred = np.argmax(probabilities, axis=1)
        
    # 4. Compute Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    
    # Macro averages (unweighted average across all classes)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    # Weighted averages (accounts for class imbalance)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(config.CLASS_NAMES)), zero_division=0
    )
    
    cm = confusion_matrix(y_test, y_pred, labels=range(len(config.CLASS_NAMES)))
    
    # Text classification report
    report_text = classification_report(
        y_test, y_pred, 
        target_names=config.CLASS_NAMES,
        labels=range(len(config.CLASS_NAMES)),
        zero_division=0
    )
    
    print("\n" + "=" * 65)
    print("                 DETAILED CLASSIFICATION REPORT")
    print("=" * 65)
    print(report_text)
    
    print("=" * 65)
    print("         SUMMARY PERFORMANCE METRICS")
    print("=" * 65)
    print(f"* Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"* Macro F1-Score:   {f1_macro * 100:.2f}% (Average across all 4 classes)")
    print(f"* Weighted F1-Score: {f1_weighted * 100:.2f}% (Balanced by class size)")
    print("=" * 65)
    
    # 5. Save Metrics to JSON
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_summary = {
        'test_records': config.TEST_RECORDS,
        'total_test_samples': int(len(y_test)),
        'accuracy': round(accuracy, 4),
        'accuracy_percent': round(accuracy * 100, 2),
        'macro_precision': round(float(p_macro), 4),
        'macro_recall': round(float(r_macro), 4),
        'macro_f1': round(float(f1_macro), 4),
        'weighted_precision': round(float(p_weighted), 4),
        'weighted_recall': round(float(r_weighted), 4),
        'weighted_f1': round(float(f1_weighted), 4),
        'per_class_metrics': {
            config.CLASS_NAMES[i]: {
                'precision': round(float(p_per_class[i]), 4),
                'recall': round(float(r_per_class[i]), 4),
                'f1_score': round(float(f1_per_class[i]), 4),
                'support': int(support_per_class[i])
            }
            for i in range(len(config.CLASS_NAMES))
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open(config.METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\n[SAVED] Metrics JSON saved to: {config.METRICS_PATH}")
    
    # 6. Generate and Save Visualizations
    plot_confusion_matrix(cm, config.CLASS_NAMES)
    plot_class_distribution(train_data, val_data, test_data)
    plot_sample_ecg_filtering(record_name=config.TEST_RECORDS[0])
    
    return metrics_summary

def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """Generates and saves a clean, dark-theme compatible confusion matrix heatmap."""
    save_path = config.RESULTS_DIR / "confusion_matrix.png"
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')
    
    # Use cyan/blue colormap
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = plt.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    
    # Axis labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9, color='white')
    ax.set_yticklabels(class_names, fontsize=9, color='white')
    ax.tick_params(colors='white')
    
    ax.set_title("ECG Arrhythmia Confusion Matrix", fontsize=12, fontweight='bold', color='white', pad=15)
    ax.set_ylabel("True Clinical Annotation", fontsize=10, fontweight='bold', color='white')
    ax.set_xlabel("Predicted 1D-CNN Class", fontsize=10, fontweight='bold', color='white')
    
    # Annotate numbers inside each matrix cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "#00b4d8"
            ax.text(j, i, f"{val:,}",
                    ha="center", va="center", color=color,
                    fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[SAVED] Confusion matrix image saved to: {save_path}")

def plot_class_distribution(train_data: dict, val_data: dict, test_data: dict):
    """Plots the distribution of heartbeat classes across Train, Val, and Test splits."""
    save_path = config.RESULTS_DIR / "class_distribution.png"
    
    train_counts = np.bincount(train_data['labels'], minlength=4)
    val_counts = np.bincount(val_data['labels'], minlength=4)
    test_counts = np.bincount(test_data['labels'], minlength=4)
    
    x = np.arange(len(config.CLASS_NAMES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')
    
    ax.bar(x - width, train_counts, width, label='Train (7 Patients)', color='#00b4d8')
    ax.bar(x, val_counts, width, label='Validation (2 Patients)', color='#2a9d8f')
    ax.bar(x + width, test_counts, width, label='Test (4 Patients)', color='#e63946')
    
    ax.set_title('Heartbeat Class Distribution Across Patient Splits', fontsize=12, fontweight='bold', color='white', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(config.CLASS_NAMES, fontsize=9, color='white')
    ax.set_ylabel('Heartbeat Count (log scale)', fontsize=10, color='white')
    ax.tick_params(colors='white')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls=":", alpha=0.25, color='gray')
    ax.legend(fontsize=9, facecolor='#161b22', edgecolor='gray', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[SAVED] Class distribution graph saved to: {save_path}")

def plot_sample_ecg_filtering(record_name: str = '200'):
    """
    Plots an example raw vs. bandpass filtered ECG waveform with annotated R-peaks.
    Visual proof of preprocessing for reports and demonstrations.
    """
    save_path = config.RESULTS_DIR / "sample_ecg_filtered.png"
    
    raw_dir = config.get_raw_data_dir()
    signal, fs, lead = load_record_signal(record_name, raw_dir)
    ann = load_record_annotations(record_name, raw_dir)
    filtered = apply_bandpass_filter(signal, fs=fs)
    
    start_sec = 10
    duration_sec = 5
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)
    
    t = np.linspace(0, duration_sec, end_sample - start_sample)
    raw_seg = signal[start_sample:end_sample]
    filt_seg = filtered[start_sample:end_sample]
    
    peaks_in_window = [
        (s - start_sample, sym) for s, sym in zip(ann.sample, ann.symbol)
        if start_sample <= s < end_sample
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=150)
    fig.patch.set_facecolor('#0e1117')
    
    # Raw ECG
    ax1.set_facecolor('#161b22')
    ax1.plot(t, raw_seg, color='#888888', linewidth=1.2)
    ax1.set_title(f'Record {record_name} (Lead {lead}): Raw ECG Signal with Noise & Drift', fontsize=11, fontweight='bold', color='white')
    ax1.set_ylabel('Voltage (mV)', fontsize=9, color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, linestyle=':', alpha=0.3, color='gray')
    
    # Filtered ECG
    ax2.set_facecolor('#161b22')
    ax2.plot(t, filt_seg, color='#00b4d8', linewidth=1.4, label='Filtered ECG (0.5 - 45 Hz)')
    for idx_rel, sym in peaks_in_window:
        t_peak = idx_rel / fs
        y_val = filt_seg[idx_rel]
        marker_color = "#2a9d8f" if sym in ['N', 'L', 'R'] else "#e63946"
        ax2.plot(t_peak, y_val, 'o', color=marker_color, markersize=6)
        ax2.annotate(f"{sym}", xy=(t_peak, y_val), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color=marker_color)
                     
    ax2.set_title('Butterworth Bandpass Filtered ECG with Doctor R-Peak Annotations', fontsize=11, fontweight='bold', color='white')
    ax2.set_xlabel('Time (seconds)', fontsize=9, color='white')
    ax2.set_ylabel('Voltage (mV)', fontsize=9, color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, linestyle=':', alpha=0.3, color='gray')
    ax2.legend(loc='upper right', facecolor='#161b22', edgecolor='gray', labelcolor='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[SAVED] Sample ECG filtering graph saved to: {save_path}")

if __name__ == "__main__":
    evaluate_model()
