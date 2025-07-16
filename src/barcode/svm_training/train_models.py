

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barcode.svm_preprocessing.pipeline_05.pipeline_runner import SimpleRunner
from barcode.svm_preprocessing.examples.basic_usage import create_sample_data

# Define constants
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "svm_training_run"

def train_and_save_models(processed_data_dir: Path):
    """
    Trains an SVM model for each anomaly type and saves it to disk.
    """
    print("\n🚀 Starting SVM model training for all anomaly types...")
    MODELS_DIR.mkdir(exist_ok=True)

    anomaly_types = [d.name for d in processed_data_dir.iterdir() if d.is_dir()]

    for anomaly_type in anomaly_types:
        print(f"\n🔧 Training model for: {anomaly_type}")
        anomaly_data_dir = processed_data_dir / anomaly_type

        try:
            X_train = np.load(anomaly_data_dir / "X_train.npy")
            y_train = np.load(anomaly_data_dir / "y_train.npy")
            X_test = np.load(anomaly_data_dir / "X_test.npy")
            y_test = np.load(anomaly_data_dir / "y_test.npy")

            if len(X_train) == 0:
                print("   - ⚠️ Warning: No training data available. Skipping model training.")
                continue

            svm_model = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
            svm_model.fit(X_train, y_train)

            if len(X_test) > 0:
                y_pred = svm_model.predict(X_test)
                print("   - Classification Report:")
                print(classification_report(y_test, y_pred, zero_division=0))

            model_path = MODELS_DIR / f"{anomaly_type}_svm_model.joblib"
            joblib.dump(svm_model, model_path)
            print(f"   - ✅ Model saved to: {model_path}")

            # Also save the scaler
            scaler_path = anomaly_data_dir / "feature_scaler.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                scaler_save_path = MODELS_DIR / f"{anomaly_type}_scaler.joblib"
                joblib.dump(scaler, scaler_save_path)
                print(f"   - ✅ Scaler saved to: {scaler_save_path}")

        except FileNotFoundError:
            print(f"   - ❌ Error: Preprocessed data not found for {anomaly_type}. Skipping.")
        except Exception as e:
            print(f"   - ❌ An unexpected error occurred during training for {anomaly_type}: {e}")

def main():
    """
    Main function to run the entire training pipeline.
    """
    print("🎓 Starting the SVM Model Training Pipeline")
    sample_df = create_sample_data()
    runner = SimpleRunner(output_dir=str(DATA_DIR))
    runner.process_data(sample_df)
    train_and_save_models(DATA_DIR)
    print("\n🎉 SVM model training pipeline finished successfully!")

if __name__ == "__main__":
    main()

