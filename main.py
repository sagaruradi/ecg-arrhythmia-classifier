"""
Main Command-Line Interface (CLI) for ECG Arrhythmia Classification Project.
Provides clean, straightforward commands for:
  - Data preparation and segmentation (--prepare)
  - Model training (--train)
  - Evaluation and metrics generation (--evaluate)
  - Running the complete workflow in one shot (--all)
"""

import argparse
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.data_loader import prepare_dataset, get_available_records
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import get_sample_test_beat, predict_heartbeat

def run_prepare(force: bool = True):
    print("\n>>> STEP 1: PREPARING AND SEGMENTING ECG DATASET")
    raw_dir = config.get_raw_data_dir()
    print(f"Dataset location: {raw_dir}")
    available = get_available_records(raw_dir)
    print(f"Discovered {len(available)} available recordings with annotations.")
    train_data, val_data, test_data = prepare_dataset(force_rebuild=force)
    print("\n[DONE] Data preparation complete!")
    print(f"  Training heartbeats:   {len(train_data['labels']):,}")
    print(f"  Validation heartbeats: {len(val_data['labels']):,}")
    print(f"  Testing heartbeats:    {len(test_data['labels']):,}")

def run_train(epochs: int = config.EPOCHS):
    print(f"\n>>> STEP 2: TRAINING 1D-CNN MODEL ({epochs} EPOCHS)")
    history = train_model(epochs=epochs)
    print("\n[DONE] Model training complete!")

def run_evaluate():
    print("\n>>> STEP 3: EVALUATING MODEL ON UNSEEN TEST PATIENTS")
    metrics = evaluate_model()
    print("\n[DONE] Evaluation complete!")
    print(f"Test Accuracy: {metrics['accuracy_percent']}%")

def run_quick_predict():
    print("\n>>> SAMPLE HEARTBEAT PREDICTION DEMO")
    try:
        seg, true_lbl, rec = get_sample_test_beat()
        res = predict_heartbeat(seg)
        print(f"Source Patient Record: {rec}")
        print(f"Ground Truth Label:   {config.CLASS_NAMES[true_lbl]}")
        print(f"Model Prediction:     {res['class_name']}")
        print(f"Model Confidence:     {res['confidence_percent']}%")
        print("Class Probabilities:")
        for cls_name, prob in res['probabilities'].items():
            print(f"  - {cls_name}: {prob:.1f}%")
    except Exception as e:
        print(f"[ERROR] Could not run prediction demo: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Real-Time ECG Arrhythmia Signal Classification Using 1D-CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --prepare     # Prepare, filter, and segment raw ECG records
  python main.py --train       # Train the 1D-CNN model on segmented beats
  python main.py --evaluate    # Evaluate on held-out test records & save plots
  python main.py --all         # Run entire pipeline (prepare -> train -> evaluate)
  python main.py --predict     # Run single sample prediction demo
        """
    )
    
    parser.add_argument("--prepare", action="store_true", help="Prepare, filter, segment, and cache ECG data")
    parser.add_argument("--train", action="store_true", help="Train the 1D-CNN model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model on test records")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline: prepare -> train -> evaluate")
    parser.add_argument("--predict", action="store_true", help="Run a quick single heartbeat prediction test")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs (default: 5)")
    
    args = parser.parse_args()
    
    # If no argument is provided, show help message
    if not (args.prepare or args.train or args.evaluate or args.all or args.predict):
        parser.print_help()
        sys.exit(0)
        
    if args.all:
        run_prepare(force=True)
        run_train(epochs=args.epochs)
        run_evaluate()
        run_quick_predict()
        print("\n" + "=" * 65)
        print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
        print("Launch the Streamlit interactive dashboard with:")
        print("  streamlit run app.py")
        print("=" * 65)
        return

    if args.prepare:
        run_prepare(force=True)
        
    if args.train:
        run_train(epochs=args.epochs)
        
    if args.evaluate:
        run_evaluate()
        
    if args.predict:
        run_quick_predict()

if __name__ == "__main__":
    main()
