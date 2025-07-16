

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.barcode.svm_preprocessing.pipeline_runner import SimpleRunner

# Define constants
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "svm_training_run"

def create_sample_data() -> pd.DataFrame:
    """
    Create realistic sample barcode scan data for demonstration.
    """
    sample_data = []
    for product_id in range(1, 11):
        epc_code = f"001.8804823.{product_id:07d}.123456.20240101.{product_id:09d}"
        num_events = np.random.randint(2, 7)
        for event_num in range(num_events):
            base_time = pd.Timestamp('2024-01-01 09:00:00')
            time_offset = pd.Timedelta(days=event_num, hours=np.random.randint(0, 8))
            event_time = base_time + time_offset
            if event_num == 0:
                event_type = 'Aggregation'
                location = '화성공장'
            elif event_num < num_events - 1:
                event_type = 'WMS_Inbound'
                location = f'물류센터_{event_num}'
            else:
                event_type = 'POS_Retail'
                location = '마트_A점'
            sample_data.append({
                'epc_code': epc_code,
                'event_time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': event_type,
                'reader_location': location,
                'business_step': event_type.split('_')[0]
            })
    duplicate_epc = "001.8804823.9999999.123456.20240101.999999999"
    for location in ['서울물류센터', '부산물류센터']:
        sample_data.append({
            'epc_code': duplicate_epc,
            'event_time': '2024-01-06 14:30:00',
            'event_type': 'WMS_Inbound',
            'reader_location': location,
            'business_step': 'WMS'
        })
    jump_epc = "001.8804823.8888888.123456.20240101.888888888"
    sample_data.extend([
        {
            'epc_code': jump_epc,
            'event_time': '2024-01-01 09:00:00',
            'event_type': 'Aggregation',
            'reader_location': '화성공장',
            'business_step': 'Aggregation'
        },
        {
            'epc_code': jump_epc,
            'event_time': '2024-06-01 15:00:00',
            'event_type': 'POS_Retail',
            'reader_location': '마트_B점',
            'business_step': 'POS'
        }
    ])
    df = pd.DataFrame(sample_data)
    return df

def train_and_save_models(processed_data_dir: Path):
    """
    Trains an SVM model for each anomaly type and saves it to disk.
    """
    print("\nStarting SVM model training for all anomaly types...")
    MODELS_DIR.mkdir(exist_ok=True)
    anomaly_types = [d.name for d in processed_data_dir.iterdir() if d.is_dir()]
    for anomaly_type in anomaly_types:
        print(f"\nTraining model for: {anomaly_type}")
        anomaly_data_dir = processed_data_dir / anomaly_type
        try:
            X_train = np.load(anomaly_data_dir / "X_train.npy")
            y_train = np.load(anomaly_data_dir / "y_train.npy")
            X_test = np.load(anomaly_data_dir / "X_test.npy")
            y_test = np.load(anomaly_data_dir / "y_test.npy")
            if len(X_train) == 0:
                print("Warning: No training data available. Skipping model training.")
                continue
            svm_model = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
            svm_model.fit(X_train, y_train)
            if len(X_test) > 0:
                y_pred = svm_model.predict(X_test)
                report = classification_report(y_test, y_pred, zero_division=0)
                print("Classification Report:")
                print(report)
            model_path = MODELS_DIR / f"{anomaly_type}_svm_model.joblib"
            joblib.dump(svm_model, model_path)
            print(f"Model saved to: {model_path}")
        except FileNotFoundError:
            print(f"Error: Preprocessed data not found for {anomaly_type}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred during training for {anomaly_type}: {e}")

def main():
    """
    Main function to run the entire training pipeline.
    """
    print("Starting the SVM Model Training Pipeline")
    sample_df = create_sample_data()
    runner = SimpleRunner(output_dir=str(DATA_DIR))
    results = runner.process_data(sample_df)
    train_and_save_models(DATA_DIR)
    print("\nSVM model training pipeline finished successfully!")

if __name__ == "__main__":
    main()

