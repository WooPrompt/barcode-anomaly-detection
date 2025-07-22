#!/usr/bin/env python3
"""
Step 2: Train Production LSTM Model
Uses prepared data to train a production-ready LSTM model
"""

import torch
import torch.nn as nn
import numpy as np
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add source paths
sys.path.append('src')
sys.path.append('lstm_academic_implementation/src')

from production_lstm_model import ProductionLSTM, LSTMTrainer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_weighted_sampler(labels):
    """Create weighted sampler for class imbalance"""
    
    # Calculate class weights
    class_weights = []
    for i in range(labels.shape[1]):
        pos_count = labels[:, i].sum()
        neg_count = len(labels) - pos_count
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0
        class_weights.append(weight)
    
    # Calculate sample weights
    sample_weights = np.ones(len(labels))
    for i, label in enumerate(labels):
        if label.sum() > 0:  # If any anomaly present
            max_weight = max([class_weights[j] for j, val in enumerate(label) if val == 1])
            sample_weights[i] = max_weight
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_lstm_model():
    """Train LSTM model with prepared data"""
    
    print("Starting LSTM model training")
    
    # Step 1: Check for prepared data
    data_dir = Path('lstm_academic_implementation')
    required_files = [
        'train_sequences.npy',
        'train_labels.npy',
        'test_sequences.npy', 
        'test_labels.npy'
    ]
    
    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"ERROR: Missing file: {file}")
            print("Please run step1_prepare_data_with_eda.py first!")
            return False
    
    # Step 2: Load prepared data
    print("Loading prepared training data...")
    train_sequences = np.load(data_dir / 'train_sequences.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    test_sequences = np.load(data_dir / 'test_sequences.npy')
    test_labels = np.load(data_dir / 'test_labels.npy')
    
    print(f"Training data shape: {train_sequences.shape}")
    print(f"Testing data shape: {test_sequences.shape}")
    print(f"Labels shape: {train_labels.shape}")
    
    # Data validation
    if len(train_sequences) == 0:
        print("ERROR: No training data found!")
        return False
    
    # Step 3: Analyze class distribution
    print("Analyzing class distribution...")
    anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
    
    for i, anomaly_type in enumerate(anomaly_types):
        if i < train_labels.shape[1]:
            pos_count = train_labels[:, i].sum()
            pos_rate = pos_count / len(train_labels)
            print(f"   {anomaly_type}: {pos_count:,} positive ({pos_rate:.3%})")
    
    # Step 4: Configure training parameters (GTX 1650 optimized)
    TRAINING_CONFIG = {
        'batch_size': 64,   # GTX 1650 4GB VRAM optimized (was 128)
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'max_epochs': 50,
        'patience': 10,
        'hidden_size': 128,  # Increased for more capacity
        'num_layers': 3,
        'dropout': 0.3,
        'attention_heads': 8
    }
    
    print(f"Training configuration: {TRAINING_CONFIG}")
    
    # Step 5: Create model
    print("Creating LSTM model...")
    input_size = train_sequences.shape[2]
    
    model = ProductionLSTM(
        input_size=input_size,
        hidden_size=TRAINING_CONFIG['hidden_size'],
        num_layers=TRAINING_CONFIG['num_layers'],
        num_classes=train_labels.shape[1],
        dropout=TRAINING_CONFIG['dropout'],
        attention_heads=TRAINING_CONFIG['attention_heads']
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Step 6: Setup device and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        
        # Dynamic batch size optimization for different GPUs
        if gpu_memory < 6:  # GTX 1650, GTX 1060, etc.
            TRAINING_CONFIG['batch_size'] = 64
            print(f"   Optimized batch size for {gpu_memory:.0f}GB VRAM: {TRAINING_CONFIG['batch_size']}")
        elif gpu_memory < 12:  # RTX 3070, RTX 2080, etc.
            TRAINING_CONFIG['batch_size'] = 128
        else:  # RTX 3090, RTX 4090, etc.
            TRAINING_CONFIG['batch_size'] = 256
    else:
        # CPU optimization
        TRAINING_CONFIG['batch_size'] = 32
        print(f"   CPU mode: reduced batch size to {TRAINING_CONFIG['batch_size']}")
    
    # Step 7: Setup trainer
    trainer = LSTMTrainer(model, device=device)
    trainer.setup_training(
        learning_rate=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        max_epochs=TRAINING_CONFIG['max_epochs']
    )
    
    # Step 8: Create data loaders
    print("Creating data loaders...")
    
    # Convert to tensors
    train_sequences_tensor = torch.FloatTensor(train_sequences)
    train_labels_tensor = torch.FloatTensor(train_labels)
    test_sequences_tensor = torch.FloatTensor(test_sequences)
    test_labels_tensor = torch.FloatTensor(test_labels)
    
    # Create datasets
    train_dataset = TensorDataset(train_sequences_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_sequences_tensor, test_labels_tensor)
    
    # Create weighted sampler for class imbalance
    weighted_sampler = create_weighted_sampler(train_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        sampler=weighted_sampler,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda'
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Step 9: Start training
    print("Starting LSTM training...")
    estimated_time = len(train_loader) * TRAINING_CONFIG['max_epochs'] * 0.5 / 60
    print(f"Estimated training time: {estimated_time:.0f} minutes")
    
    start_time = datetime.now()
    
    try:
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=TRAINING_CONFIG['max_epochs'],
            patience=TRAINING_CONFIG['patience']
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds() / 60
        
        print("Training completed successfully!")
        print(f"Actual training time: {training_duration:.1f} minutes")
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        return False
    
    # Step 10: Save trained model
    print("Saving trained model...")
    
    # Save full model
    model_path = data_dir / 'trained_lstm_model.pt'
    torch.save(model.state_dict(), model_path)
    
    # Create quantized version for production
    try:
        from production_lstm_model import quantize_model
        quantized_path = data_dir / 'trained_lstm_quantized.pt'
        quantized_model = quantize_model(model, str(quantized_path))
        print("Quantized model created for production deployment")
    except Exception as e:
        print(f"WARNING: Quantization failed: {e}")
    
    # Step 11: Save training results
    training_summary = {
        'training_config': TRAINING_CONFIG,
        'model_config': {
            'input_size': input_size,
            'hidden_size': TRAINING_CONFIG['hidden_size'],
            'num_layers': TRAINING_CONFIG['num_layers'],
            'num_classes': train_labels.shape[1],
            'total_parameters': param_count
        },
        'training_results': training_results,
        'training_time_minutes': training_duration,
        'device_used': str(device),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(data_dir / 'training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    
    # Step 12: Print results
    print("Training Results:")
    print(f"   Best Validation AUC: {training_results['best_val_auc']:.4f}")
    print(f"   Total epochs: {training_results['total_epochs']}")
    print(f"   Training time: {training_duration:.1f} minutes")
    
    # Performance assessment
    if training_results['best_val_auc'] >= 0.85:
        print("EXCELLENT! Production-ready performance!")
    elif training_results['best_val_auc'] >= 0.75:
        print("GOOD! Suitable for production with monitoring!")
    elif training_results['best_val_auc'] >= 0.65:
        print("MODERATE! Consider more training or data!")
    else:
        print("POOR! Needs significant improvement!")
    
    print("\nFiles created:")
    print("   trained_lstm_model.pt (full model)")
    print("   trained_lstm_quantized.pt (production model)")
    print("   training_summary.json (training report)")
    
    return True

if __name__ == "__main__":
    success = train_lstm_model()
    if success:
        print("\nSUCCESS: Ready for Step 3: FastAPI Integration")
        print("Run: python lstm_academic_implementation/step3_integrate_fastapi.py")
    else:
        print("\nERROR: Training failed. Please check the errors above.")