
"""
Train SVM Models for Barcode Anomaly Detection

This script orchestrates the complete process of training SVM models for all
five barcode anomaly types. It uses the svm_preprocessing library to handle
data preparation and then trains and saves the models for later use in prediction.

Workflow:
1.  Generate realistic sample barcode scan data.
2.  Run the complete SVM preprocessing pipeline using SimpleRunner.
    - This cleans data, extracts features, generates labels, normalizes
      features, and creates train/test splits for all 5 anomaly types.
3.  Load the preprocessed training data for each anomaly type.
4.  Train a Support Vector Machine (SVC) model for each anomaly type.
    - Uses a balanced class weight to handle imbalanced data.
    - Enables probability estimates for confidence scores.
5.  Save the trained models and feature scalers to the `models/` directory.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Add project root to the Python path
from barcode.svm_preprocessing.examples.basic_usage import create_sample_data

# Define constants
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "svm_training_run"


def train_and_save_models(processed_data_dir: Path):
    """
    Trains an SVM model for each anomaly type and saves it to disk.

    Args:
        processed_data_dir (Path): The directory containing the preprocessed
                                   train/test data from the SimpleRunner.
    """
    print("\n🚀 Starting SVM model training for all anomaly types...")
    print("=" * 60)

    # Ensure the main models directory exists
    MODELS_DIR.mkdir(exist_ok=True)

    anomaly_types = [d.name for d in processed_data_dir.iterdir() if d.is_dir()]

    for anomaly_type in anomaly_types:
        print(f"\n🔧 Training model for: {anomaly_type}")
        anomaly_data_dir = processed_data_dir / anomaly_type

        try:
            # Load the preprocessed training data
            X_train = np.load(anomaly_data_dir / "X_train.npy")
            y_train = np.load(anomaly_data_dir / "y_train.npy")
            X_test = np.load(anomaly_data_dir / "X_test.npy")
            y_test = np.load(anomaly_data_dir / "y_test.npy")

            print(f"   - Loaded training data: {X_train.shape[0]} samples")
            print(f"   - Loaded testing data: {X_test.shape[0]} samples")

            if len(X_train) == 0:
                print("   - ⚠️ Warning: No training data available. Skipping model training.")
                continue

            # Initialize and train the SVM model
            # - C=1.0: Standard regularization parameter.
            # - kernel='rbf': A good default kernel for non-linear problems.
            # - probability=True: Enables prediction confidence scores.
            # - class_weight='balanced': Automatically adjusts weights inversely
            #   proportional to class frequencies in the input data. Essential for
            #   imbalanced anomaly detection datasets.
            svm_model = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
            print("   - Training SVM model...")
            svm_model.fit(X_train, y_train)
            print("   - ✅ Model training complete.")

            # Evaluate the model on the test set
            if len(X_test) > 0:
                print("   - Evaluating model performance...")
                y_pred = svm_model.predict(X_test)
                report = classification_report(y_test, y_pred, zero_division=0)
                print("   - Classification Report:")
                print(report)

            # Save the trained model
            model_path = MODELS_DIR / f"{anomaly_type}_svm_model.joblib"
            joblib.dump(svm_model, model_path)
            print(f"   - ✅ Model saved to: {model_path}")

        except FileNotFoundError:
            print(f"   - ❌ Error: Preprocessed data not found for {anomaly_type}. Skipping.")
        except Exception as e:
            print(f"   - ❌ An unexpected error occurred during training for {anomaly_type}: {e}")


def main():
    """
    Main function to run the entire training pipeline.
    """
    print("Starting the SVM Model Training Pipeline")
    print("=" * 60)

    # 1. Generate sample data
    print("\nStep 1: Generating sample data...")
    sample_df = create_sample_data()
    print(f"   - ✅ Generated {len(sample_df)} sample scan events.")

    # 2. Run the preprocessing pipeline
    print("\nStep 2: Running the data preprocessing pipeline...")
    # The SimpleRunner will handle everything: cleaning, feature extraction,
    # normalization, and saving the train/test splits to disk.
    runner = SimpleRunner(output_dir=str(DATA_DIR))
    results = runner.process_data(sample_df)
    print("   - ✅ Data preprocessing complete.")
    print(f"   - Preprocessed data saved to: {DATA_DIR}")

    # 3. Train and save the SVM models
    train_and_save_models(DATA_DIR)

    print("\n🎉 SVM model training pipeline finished successfully!")
    print(f"   - All models are saved in: {MODELS_DIR}")
    print("   - Ready to be used for prediction.")


if __name__ == "__main__":
    main()
