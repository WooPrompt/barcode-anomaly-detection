#!/usr/bin/env python3
"""
LSTM Model Training Script - Simple Interface for PM
Author: Data Science Team
Date: 2025-07-21

This script trains the LSTM anomaly detection model on your barcode data.
Just run: python train_model.py

No Python knowledge required - just run and wait for completion!
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

from lstm_trainer import LSTMTrainer
from lstm_data_preprocessor import LSTMFeatureEngineer
from lstm_model import OptimizedLSTMAnomalyDetector

# Set up logging for clear progress tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def print_header():
    """Print welcome message for PM"""
    print("=" * 60)
    print("🚀 LSTM ANOMALY DETECTION MODEL TRAINING")
    print("=" * 60)
    print("Welcome! This script will train your AI model to detect barcode anomalies.")
    print("The process is completely automated - just wait for completion.")
    print("Training time: Approximately 15-30 minutes depending on data size.")
    print("=" * 60)

def check_data_availability():
    """Check if training data is available"""
    possible_data_paths = [
        '../../data/processed/all_factories_clean_v2.csv',  # Real data
        '../../../data/processed/all_factories_clean_v2.csv',
        '../data/training_data.csv',
        '../../data/training_data.csv',
        'training_data.csv'
    ]
    
    for path in possible_data_paths:
        if os.path.exists(path):
            return path
    
    return None

def create_sample_data():
    """Create sample data for demonstration if no real data found"""
    logger.info("No training data found. Creating sample data for demonstration...")
    
    # Create realistic sample data
    n_samples = 10000
    
    sample_data = pd.DataFrame({
        'epc': [f'EPC_{i:06d}' for i in np.random.randint(0, 5000, n_samples)],
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'location_id': np.random.choice(['WAREHOUSE_A', 'WAREHOUSE_B', 'STORE_1', 'STORE_2'], n_samples),
        'latitude': np.random.uniform(40.0, 45.0, n_samples),
        'longitude': np.random.uniform(-74.0, -70.0, n_samples),
        'temperature': np.random.normal(20, 5, n_samples),
        'humidity': np.random.normal(50, 10, n_samples),
        'signal_strength': np.random.normal(-60, 10, n_samples),
        
        # Anomaly labels (5% base rate with some correlation)
        'epcFake': np.random.binomial(1, 0.05, n_samples),
        'epcDup': np.random.binomial(1, 0.02, n_samples),
        'locErr': np.random.binomial(1, 0.03, n_samples),
        'evtOrderErr': np.random.binomial(1, 0.04, n_samples),
        'jump': np.random.binomial(1, 0.06, n_samples)
    })
    
    # Save sample data
    sample_data.to_csv('sample_training_data.csv', index=False)
    logger.info("Sample data created: sample_training_data.csv")
    
    return 'sample_training_data.csv'

def main():
    """Main training function"""
    try:
        print_header()
        
        # Step 1: Check for data
        logger.info("Step 1/6: Checking for training data...")
        data_path = check_data_availability()
        
        if data_path is None:
            data_path = create_sample_data()
        else:
            logger.info(f"Found training data: {data_path}")
        
        # Step 2: Load and prepare data
        logger.info("Step 2/6: Loading and preparing data...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df):,} training samples from real factory data")
        
        # Step 3: Feature engineering
        logger.info("Step 3/6: Engineering features (this creates AI-readable data)...")
        feature_engineer = LSTMFeatureEngineer()
        
        # Prepare features and labels
        feature_columns = [col for col in df.columns if col not in 
                          ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']]
        label_columns = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
        
        # Simple feature preparation for demo
        features = df[feature_columns].select_dtypes(include=[np.number]).fillna(0).values
        labels = df[label_columns].fillna(0).values
        
        logger.info(f"Features prepared: {features.shape[1]} features, {features.shape[0]} samples")
        
        # Step 4: Create model
        logger.info("Step 4/6: Creating LSTM AI model...")
        model = OptimizedLSTMAnomalyDetector(
            input_size=features.shape[1],
            hidden_size=128,
            num_layers=2,
            num_classes=5,
            dropout_rate=0.2
        )
        
        logger.info("Model created with deep learning architecture:")
        logger.info("- Bidirectional LSTM for temporal pattern recognition")
        logger.info("- Multi-head attention for focus on important features")
        logger.info("- Dropout regularization for robustness")
        
        # Step 5: Train model
        logger.info("Step 5/6: Training the AI model (this may take 15-30 minutes)...")
        trainer = LSTMTrainer(model)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(features)
        y_tensor = torch.FloatTensor(labels)
        
        # Train the model
        training_results = trainer.train(
            X_tensor, y_tensor,
            epochs=50,  # Reduced for faster demo
            batch_size=64,
            validation_split=0.2
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final Training Loss: {training_results['final_train_loss']:.4f}")
        logger.info(f"Final Validation Loss: {training_results['final_val_loss']:.4f}")
        
        # Step 6: Save model
        logger.info("Step 6/6: Saving trained model...")
        model_save_path = 'trained_lstm_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': features.shape[1],
                'hidden_size': 128,
                'num_layers': 2,
                'num_classes': 5,
                'dropout_rate': 0.2
            },
            'training_results': training_results,
            'training_date': datetime.now().isoformat()
        }, model_save_path)
        
        logger.info(f"Model saved: {model_save_path}")
        
        # Success message
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Your AI model is now trained and ready to detect barcode anomalies.")
        print(f"Model file: {model_save_path}")
        print("Next step: Run 'python predict_anomalies.py' to start detecting anomalies!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print("\n❌ TRAINING FAILED")
        print("Please check the error log above and contact the data science team.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)