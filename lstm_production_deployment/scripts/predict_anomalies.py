#!/usr/bin/env python3
"""
LSTM Anomaly Prediction Script - Simple Interface for PM
Author: Data Science Team
Date: 2025-07-21

This script uses the trained LSTM model to detect anomalies in new barcode data.
Just run: python predict_anomalies.py

No Python knowledge required - just run and get your anomaly detection results!
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from lstm_model import OptimizedLSTMAnomalyDetector
from lstm_inferencer import ProductionLSTMInferencer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def print_header():
    """Print welcome message for PM"""
    print("=" * 60)
    print("🔍 LSTM ANOMALY DETECTION - PREDICTION MODE")
    print("=" * 60)
    print("Welcome! This script will analyze your barcode data for anomalies.")
    print("The AI model will identify 5 types of problems:")
    print("1. Fake EPCs (counterfeit products)")
    print("2. Duplicate EPCs (supply chain errors)")
    print("3. Location Errors (misplaced items)")
    print("4. Event Order Errors (process violations)")
    print("5. Jump Anomalies (suspicious movements)")
    print("=" * 60)

def check_model_availability():
    """Check if trained model is available"""
    possible_model_paths = [
        'trained_lstm_model.pth',
        '../trained_lstm_model.pth',
        '../models/trained_lstm_model.pth',
        '../../models/trained_lstm_model.pth'
    ]
    
    for path in possible_model_paths:
        if os.path.exists(path):
            return path
    
    return None

def check_prediction_data():
    """Check for data to analyze"""
    possible_data_paths = [
        'new_data.csv',
        '../data/new_data.csv', 
        '../../data/new_data.csv',
        'sample_training_data.csv',  # Fallback to sample data
        '../sample_training_data.csv'
    ]
    
    for path in possible_data_paths:
        if os.path.exists(path):
            return path
    
    return None

def create_sample_prediction_data():
    """Create sample data for demonstration"""
    logger.info("Creating sample data for anomaly detection demonstration...")
    
    # Create sample data with some intentional anomalies
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'epc': [f'EPC_{i:06d}' for i in np.random.randint(0, 500, n_samples)],
        'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1min'),
        'location_id': np.random.choice(['WAREHOUSE_A', 'WAREHOUSE_B', 'STORE_1', 'STORE_2'], n_samples),
        'latitude': np.random.uniform(40.0, 45.0, n_samples),
        'longitude': np.random.uniform(-74.0, -70.0, n_samples),
        'temperature': np.random.normal(20, 5, n_samples),
        'humidity': np.random.normal(50, 10, n_samples),
        'signal_strength': np.random.normal(-60, 10, n_samples)
    })
    
    # Add some intentional patterns that might indicate anomalies
    # Simulate fake EPCs with unusual signal patterns
    fake_indices = np.random.choice(n_samples, 50, replace=False)
    sample_data.loc[fake_indices, 'signal_strength'] = np.random.normal(-80, 5, 50)  # Weaker signal
    
    # Simulate location errors with GPS outliers
    location_error_indices = np.random.choice(n_samples, 30, replace=False)
    sample_data.loc[location_error_indices, 'latitude'] = np.random.uniform(35.0, 39.0, 30)  # Out of normal range
    
    sample_data.to_csv('sample_prediction_data.csv', index=False)
    logger.info("Sample prediction data created: sample_prediction_data.csv")
    
    return 'sample_prediction_data.csv'

def load_model(model_path):
    """Load the trained model"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model with saved configuration
        model_config = checkpoint['model_config']
        model = OptimizedLSTMAnomalyDetector(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Training date: {checkpoint.get('training_date', 'Unknown')}")
        
        return model, model_config
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def predict_anomalies(model, data, model_config):
    """Run anomaly detection on the data"""
    
    # Prepare features (same as training)
    feature_columns = [col for col in data.columns if col not in 
                      ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']]
    
    # Simple feature preparation
    features = data[feature_columns].select_dtypes(include=[np.number]).fillna(0).values
    
    # Ensure feature count matches model expectations
    if features.shape[1] != model_config['input_size']:
        logger.warning(f"Feature count mismatch. Expected {model_config['input_size']}, got {features.shape[1]}")
        # Pad or truncate features to match
        if features.shape[1] < model_config['input_size']:
            padding = np.zeros((features.shape[0], model_config['input_size'] - features.shape[1]))
            features = np.hstack([features, padding])
        else:
            features = features[:, :model_config['input_size']]
    
    # Convert to tensor and predict
    X_tensor = torch.FloatTensor(features)
    
    with torch.no_grad():
        predictions = torch.sigmoid(model(X_tensor)).numpy()
    
    return predictions

def generate_report(data, predictions, output_file='anomaly_report.csv'):
    """Generate human-readable anomaly report"""
    
    anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
    anomaly_names = [
        'Fake EPC (Counterfeit)',
        'Duplicate EPC (Supply Chain Error)', 
        'Location Error (Misplaced Item)',
        'Event Order Error (Process Violation)',
        'Jump Anomaly (Suspicious Movement)'
    ]
    
    # Create results dataframe
    results = data.copy()
    
    # Add prediction scores
    for i, (anomaly_type, anomaly_name) in enumerate(zip(anomaly_types, anomaly_names)):
        results[f'{anomaly_type}_score'] = predictions[:, i]
        results[f'{anomaly_type}_detected'] = predictions[:, i] > 0.5
    
    # Add overall anomaly flag
    results['any_anomaly_detected'] = (predictions > 0.5).any(axis=1)
    results['max_anomaly_score'] = predictions.max(axis=1)
    results['anomaly_count'] = (predictions > 0.5).sum(axis=1)
    
    # Sort by most suspicious first
    results = results.sort_values('max_anomaly_score', ascending=False)
    
    # Save full results
    results.to_csv(output_file, index=False)
    
    # Create summary
    total_items = len(results)
    anomalous_items = results['any_anomaly_detected'].sum()
    anomaly_rate = anomalous_items / total_items * 100
    
    # Per-anomaly type summary
    anomaly_summary = {}
    for i, (anomaly_type, anomaly_name) in enumerate(zip(anomaly_types, anomaly_names)):
        detected_count = results[f'{anomaly_type}_detected'].sum()
        avg_score = results[f'{anomaly_type}_score'].mean()
        anomaly_summary[anomaly_name] = {
            'count': int(detected_count),
            'percentage': detected_count / total_items * 100,
            'avg_confidence': avg_score
        }
    
    return results, anomaly_summary, anomaly_rate

def print_summary(anomaly_summary, anomaly_rate, total_items):
    """Print business-friendly summary"""
    
    print("\n" + "=" * 60)
    print("📊 ANOMALY DETECTION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total items analyzed: {total_items:,}")
    print(f"Items with anomalies: {anomaly_rate:.1f}%")
    print("\nDetailed breakdown:")
    print("-" * 40)
    
    for anomaly_name, stats in anomaly_summary.items():
        print(f"{anomaly_name}:")
        print(f"  • Detected: {stats['count']} items ({stats['percentage']:.1f}%)")
        print(f"  • Avg confidence: {stats['avg_confidence']:.1%}")
        print()
    
    print("=" * 60)
    print("📋 BUSINESS RECOMMENDATIONS:")
    
    if anomaly_rate > 10:
        print("🔴 HIGH ALERT: Anomaly rate >10% indicates serious supply chain issues")
        print("   → Immediate investigation required")
        print("   → Review supplier processes and data quality")
    elif anomaly_rate > 5:
        print("🟡 MEDIUM ALERT: Elevated anomaly rate requires attention")
        print("   → Schedule review of flagged items")
        print("   → Implement additional quality controls")
    elif anomaly_rate > 1:
        print("🟢 LOW ALERT: Normal anomaly levels detected")
        print("   → Review flagged items during regular audits")
        print("   → Continue monitoring")
    else:
        print("✅ EXCELLENT: Very low anomaly rate detected")
        print("   → Systems operating normally")
        print("   → Maintain current processes")
    
    print("\n📄 Detailed results saved to: anomaly_report.csv")
    print("=" * 60)

def main():
    """Main prediction function"""
    try:
        print_header()
        
        # Step 1: Check for trained model
        logger.info("Step 1/5: Checking for trained model...")
        model_path = check_model_availability()
        
        if model_path is None:
            print("\n❌ ERROR: No trained model found!")
            print("Please run 'python train_model.py' first to train the model.")
            return False
        
        logger.info(f"Found trained model: {model_path}")
        
        # Step 2: Load model
        logger.info("Step 2/5: Loading AI model...")
        model, model_config = load_model(model_path)
        
        if model is None:
            print("\n❌ ERROR: Failed to load model!")
            return False
        
        # Step 3: Check for prediction data
        logger.info("Step 3/5: Checking for data to analyze...")
        data_path = check_prediction_data()
        
        if data_path is None:
            data_path = create_sample_prediction_data()
        else:
            logger.info(f"Found data to analyze: {data_path}")
        
        # Step 4: Load and predict
        logger.info("Step 4/5: Analyzing data for anomalies...")
        df = pd.read_csv(data_path)
        logger.info(f"Analyzing {len(df):,} items...")
        
        predictions = predict_anomalies(model, df, model_config)
        logger.info("Anomaly detection completed!")
        
        # Step 5: Generate report
        logger.info("Step 5/5: Generating business report...")
        results, anomaly_summary, anomaly_rate = generate_report(df, predictions)
        
        # Print summary for PM
        print_summary(anomaly_summary, anomaly_rate, len(df))
        
        return True
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        print("\n❌ PREDICTION FAILED")
        print("Please check the error log above and contact the data science team.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)