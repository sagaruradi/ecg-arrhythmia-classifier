"""
Training Pipeline for ECG Arrhythmia 1D-CNN.
Handles data loading, calibrated class loss weighting,
validation checkpoint selection, and artifact saving.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

import config
from src.data_loader import prepare_dataset
from src.model import ECG1DCNN

def train_model(epochs: int = 7, 
                batch_size: int = config.BATCH_SIZE, 
                lr: float = config.LEARNING_RATE,
                force_rebuild_data: bool = False) -> dict:
    """
    Trains the 1D-CNN model on the MIT-BIH Arrhythmia train split.
    
    Key Engineering Techniques Applied:
    1. Calibrated Class Loss Weighting: Gives appropriate emphasis to minority
       arrhythmia categories (Supraventricular and Fusion) without destabilizing
       the dominant Normal baseline.
    2. Weighted-F1 Validation Checkpoint Selection: Selects the model checkpoint that maximizes
       overall clinical balance across classes rather than misleading raw accuracy.
    3. Strict Patient-Level Splitting: Ensures zero patient overlap across splits.
    """
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print("=" * 65)
    print("      ECG 1D-CNN ARRHYTHMIA CLASSIFICATION - TRAINING")
    print("=" * 65)
    
    # 1. Load Data
    train_data, val_data, test_data = prepare_dataset(
        train_records=config.TRAIN_RECORDS,
        val_records=config.VAL_RECORDS,
        test_records=config.TEST_RECORDS,
        cache=True,
        force_rebuild=force_rebuild_data
    )
    
    X_train, y_train = train_data['segments'], train_data['labels']
    X_val, y_val = val_data['segments'], val_data['labels']
    
    print(f"\n[DATASET SUMMARY (PATIENT-INDEPENDENT SPLITS)]")
    print(f"  Training Patient Records:   {config.TRAIN_RECORDS} -> {len(y_train):,} beats")
    print(f"  Validation Patient Records: {config.VAL_RECORDS} -> {len(y_val):,} beats")
    print(f"  Test Patient Records:       {config.TEST_RECORDS} -> {len(test_data['labels']):,} beats")
    
    # Class distribution in training set
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print("\n[CLASS DISTRIBUTION (TRAINING)]")
    for cls_idx, count in zip(unique_classes, counts):
        pct = (count / len(y_train)) * 100
        print(f"  Class {cls_idx} ({config.CLASS_NAMES[cls_idx]}): {count:,} beats ({pct:.1f}%)")
        
    # 2. Calibrated Class Loss Weights
    # Balances gradient pressure across the 4 categories
    loss_weights = torch.tensor([1.0, 1.5, 1.1, 1.8], dtype=torch.float32)
    print(f"\n[CALIBRATED CLASS LOSS WEIGHTS]: {loss_weights.tolist()}")
    
    # 3. Create PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Initialize Model, Loss Function, and Optimizer
    model = ECG1DCNN(in_channels=1, num_classes=len(config.CLASS_NAMES))
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 5. Training Loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_weighted_f1': []
    }
    
    best_f1 = 0.0
    best_val_acc = 0.0
    best_state_dict = None
    
    print(f"\n[TRAINING PROGRESS - {epochs} EPOCHS (MONITORING VALIDATION F1)]")
    print("-" * 75)
    
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
        epoch_train_loss = running_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100
        
        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_running_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
        epoch_val_loss = val_running_loss / len(val_targets)
        epoch_val_acc = accuracy_score(val_targets, val_preds) * 100
        epoch_val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0) * 100
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_weighted_f1'].append(epoch_val_f1)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% "
              f"- Val F1: {epoch_val_f1:.2f}%")
              
        # Model Selection
        if epoch_val_f1 > best_f1:
            best_f1 = epoch_val_f1
            best_val_acc = epoch_val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(best_state_dict, config.MODEL_PATH)
            
    print("-" * 75)
    print(f"[SUCCESS] Best Validation F1: {best_f1:.2f}% (Val Accuracy: {best_val_acc:.2f}%)")
    print(f"[SAVED] Trained model checkpoint saved to: {config.MODEL_PATH}")
    
    # 6. Save Class Mapping Metadata
    class_mapping_data = {
        'class_names': config.CLASS_NAMES,
        'class_short_names': config.CLASS_SHORT_NAMES,
        'aami_mapping': config.AAMI_MAPPING,
        'train_records': config.TRAIN_RECORDS,
        'val_records': config.VAL_RECORDS,
        'test_records': config.TEST_RECORDS,
        'sampling_rate': config.SAMPLING_RATE,
        'segment_length': config.SEGMENT_LENGTH,
        'best_val_f1': best_f1,
        'best_val_acc': best_val_acc
    }
    with open(config.CLASS_MAPPING_PATH, 'w') as f:
        json.dump(class_mapping_data, f, indent=4)
    print(f"[SAVED] Class metadata saved to: {config.CLASS_MAPPING_PATH}")
    
    # 7. Generate Training Curve Plots
    plot_training_curves(history)
    
    return history

def plot_training_curves(history: dict):
    """Generates and saves professional dark-theme compatible Loss and Accuracy curves."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.RESULTS_DIR / "training_curves.png"
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor('#0e1117')
    
    # Loss subplot
    ax1.set_facecolor('#161b22')
    ax1.plot(epochs_range, history['train_loss'], 'o-', color='#e63946', linewidth=2, label='Training Loss')
    ax1.plot(epochs_range, history['val_loss'], 's--', color='#00b4d8', linewidth=2, label='Validation Loss')
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=12, fontweight='bold', color='white', pad=10)
    ax1.set_xlabel('Epoch', fontsize=10, color='white')
    ax1.set_ylabel('Loss', fontsize=10, color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, linestyle=':', alpha=0.3, color='gray')
    ax1.legend(fontsize=9, facecolor='#161b22', edgecolor='gray', labelcolor='white')
    
    # Accuracy & F1 subplot
    ax2.set_facecolor('#161b22')
    ax2.plot(epochs_range, history['train_acc'], 'o-', color='#2a9d8f', linewidth=2, label='Training Acc (%)')
    ax2.plot(epochs_range, history['val_acc'], 's--', color='#e76f51', linewidth=2, label='Val Acc (%)')
    ax2.plot(epochs_range, history['val_weighted_f1'], '^-.', color='#00b4d8', linewidth=2, label='Val F1 (%)')
    ax2.set_title('Classification Performance vs. Epochs', fontsize=12, fontweight='bold', color='white', pad=10)
    ax2.set_xlabel('Epoch', fontsize=10, color='white')
    ax2.set_ylabel('Score (%)', fontsize=10, color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, linestyle=':', alpha=0.3, color='gray')
    ax2.legend(fontsize=9, facecolor='#161b22', edgecolor='gray', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[SAVED] Training curves graph saved to: {save_path}")

if __name__ == "__main__":
    train_model(epochs=7)
