# -*- coding: utf-8 -*-
"""
LSTM Trainer with Focal Loss and Advanced Training Strategies
Author: Vector Space Engineering Team - MLE & ML Scientist
Date: 2025-07-21

Academic Foundation: Implements stratified training with cost-sensitive learning
as per accelerated timeline requirements. Includes comprehensive evaluation
framework for professor-level academic defense.

Key Features:
- Stratified K-fold validation for robust performance estimation
- Focal loss with class weighting for imbalanced anomaly detection
- Learning rate scheduling with warm restarts
- Early stopping with patience-based convergence detection
- Comprehensive metric tracking (AUC, AUPR, F-beta, business metrics)
- Gradient clipping and regularization for stable training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, f1_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import time
from pathlib import Path
import warnings
from collections import defaultdict
import copy

from .lstm_model import OptimizedLSTMAnomalyDetector, FocalLoss, create_optimized_model
from .lstm_data_preprocessor import LSTMDataPreprocessor

logger = logging.getLogger(__name__)

class CostSensitiveMetrics:
    """
    ML Scientist Role: Business-aligned evaluation metrics for supply chain anomaly detection
    
    Academic Justification:
    - Cost-weighted F-beta emphasizes business impact over statistical accuracy
    - False negative costs (missed fraud) typically 10x higher than false positives
    - Multi-label evaluation handles simultaneous anomaly types
    """
    
    def __init__(self, false_negative_costs: Dict[str, float] = None, false_positive_costs: Dict[str, float] = None):
        self.anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
        
        # Default business costs (can be customized based on domain expertise)
        self.fn_costs = false_negative_costs or {
            'epcFake': 100.0,      # Counterfeit products - highest cost
            'epcDup': 30.0,        # Duplicate scanning - medium cost
            'locErr': 50.0,        # Location errors - high cost
            'evtOrderErr': 40.0,   # Event order errors - high cost
            'jump': 80.0           # Impossible jumps - very high cost
        }
        
        self.fp_costs = false_positive_costs or {
            'epcFake': 10.0,       # False alerts for legitimate products
            'epcDup': 5.0,         # False duplicate alerts
            'locErr': 8.0,         # False location alerts
            'evtOrderErr': 6.0,    # False event order alerts
            'jump': 12.0           # False jump alerts
        }
    
    def calculate_cost_weighted_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate business cost-weighted evaluation metrics
        
        Args:
            y_true: [num_samples, 5] true binary labels
            y_pred: [num_samples, 5] predicted probabilities
            threshold: Decision threshold for binary classification
            
        Returns:
            Dictionary of cost-weighted metrics
        """
        metrics = {}
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        total_cost = 0.0
        total_samples = len(y_true)
        
        for i, anomaly_type in enumerate(self.anomaly_types):
            true_labels = y_true[:, i]
            pred_labels = y_pred_binary[:, i]
            pred_probs = y_pred[:, i]
            
            # Confusion matrix elements
            tn = np.sum((true_labels == 0) & (pred_labels == 0))
            tp = np.sum((true_labels == 1) & (pred_labels == 1))
            fn = np.sum((true_labels == 1) & (pred_labels == 0))
            fp = np.sum((true_labels == 0) & (pred_labels == 1))
            
            # Business costs
            cost_fn = fn * self.fn_costs[anomaly_type]
            cost_fp = fp * self.fp_costs[anomaly_type]
            total_anomaly_cost = cost_fn + cost_fp
            
            # Standard metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # AUC metrics
            if len(np.unique(true_labels)) > 1:
                auc = roc_auc_score(true_labels, pred_probs)
                aupr = average_precision_score(true_labels, pred_probs)
            else:
                auc = 0.5
                aupr = np.mean(true_labels)
            
            # Cost-weighted F-beta (beta=2 emphasizes recall due to high FN costs)
            beta = 2.0
            cost_weight = self.fn_costs[anomaly_type] / self.fp_costs[anomaly_type]
            weighted_precision = precision
            weighted_recall = recall * cost_weight
            
            f_beta = (1 + beta**2) * (weighted_precision * weighted_recall) / \
                    (beta**2 * weighted_precision + weighted_recall) if \
                    (beta**2 * weighted_precision + weighted_recall) > 0 else 0.0
            
            metrics[f'{anomaly_type}_precision'] = precision
            metrics[f'{anomaly_type}_recall'] = recall
            metrics[f'{anomaly_type}_f1'] = f1
            metrics[f'{anomaly_type}_auc'] = auc
            metrics[f'{anomaly_type}_aupr'] = aupr
            metrics[f'{anomaly_type}_f_beta'] = f_beta
            metrics[f'{anomaly_type}_cost'] = total_anomaly_cost
            
            total_cost += total_anomaly_cost
        
        # Overall metrics
        metrics['total_cost'] = total_cost
        metrics['cost_per_sample'] = total_cost / total_samples
        metrics['macro_auc'] = np.mean([metrics[f'{t}_auc'] for t in self.anomaly_types])
        metrics['macro_aupr'] = np.mean([metrics[f'{t}_aupr'] for t in self.anomaly_types])
        metrics['macro_f_beta'] = np.mean([metrics[f'{t}_f_beta'] for t in self.anomaly_types])
        
        return metrics

class AdvancedTrainingCallbacks:
    """
    MLOps Role: Training monitoring and control callbacks
    
    Features:
    - Early stopping with patience-based convergence detection
    - Learning rate scheduling with warm restarts
    - Model checkpointing with best performance tracking
    - Training progress visualization
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 save_dir: str = "models/checkpoints"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_score = -np.inf
        self.patience_counter = 0
        self.should_stop = False
        self.training_history = defaultdict(list)
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: nn.Module):
        """Called at the end of each training epoch"""
        
        # Track training history
        for key, value in metrics.items():
            self.training_history[key].append(value)
        
        # Check for improvement (using validation AUC as primary metric)
        current_score = metrics.get('val_macro_auc', 0.0)
        
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.patience_counter = 0
            
            # Save best model checkpoint
            checkpoint_path = self.save_dir / f"best_model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': self.best_score,
                'metrics': metrics
            }, checkpoint_path)
            
            logger.info(f"New best model saved: {checkpoint_path} (AUC: {current_score:.4f})")
            
        else:
            self.patience_counter += 1
            
        # Early stopping check
        if self.patience_counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {epoch} epochs (patience: {self.patience})")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history for analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', alpha=0.7)
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss', alpha=0.7)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # AUC curves
        axes[0, 1].plot(self.training_history['train_macro_auc'], label='Train AUC', alpha=0.7)
        axes[0, 1].plot(self.training_history['val_macro_auc'], label='Val AUC', alpha=0.7)
        axes[0, 1].set_title('AUC Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        
        # F-beta curves
        axes[1, 0].plot(self.training_history['train_macro_f_beta'], label='Train F-beta', alpha=0.7)
        axes[1, 0].plot(self.training_history['val_macro_f_beta'], label='Val F-beta', alpha=0.7)
        axes[1, 0].set_title('Cost-Weighted F-beta Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F-beta')
        axes[1, 0].legend()
        
        # Cost curves
        axes[1, 1].plot(self.training_history['train_cost_per_sample'], label='Train Cost', alpha=0.7)
        axes[1, 1].plot(self.training_history['val_cost_per_sample'], label='Val Cost', alpha=0.7)
        axes[1, 1].set_title('Business Cost per Sample')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Cost per Sample')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class LSTMTrainer:
    """
    Unified LSTM Training Pipeline
    
    Team Coordination:
    - ML Scientist: Statistical validation and evaluation metrics
    - MLE: Model training and optimization
    - MLOps: Training infrastructure and monitoring
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 focal_loss_alpha: float = 0.7,
                 focal_loss_gamma: float = 2.0,
                 gradient_clip_norm: float = 1.0,
                 mixed_precision: bool = True,
                 save_dir: str = "models/lstm_checkpoints"):
        
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
        
        # Training configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        
        # Loss function with class weighting for imbalanced data
        # Weights based on business impact: [epcFake, epcDup, locErr, evtOrderErr, jump]
        pos_weights = torch.FloatTensor([10.0, 3.0, 5.0, 4.0, 8.0]).to(self.device)
        self.criterion = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma, pos_weight=pos_weights)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay,
                                   betas=(0.9, 0.999))
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=learning_rate,
            steps_per_epoch=1,  # Will be updated during training
            epochs=1,  # Will be updated during training
            pct_start=0.3,
            div_factor=25,
            final_div_factor=10000
        )
        
        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = GradScaler()
        
        # Evaluation metrics
        self.metrics_calculator = CostSensitiveMetrics()
        self.callbacks = None
        
        # Training state
        self.training_history = defaultdict(list)
        self.current_epoch = 0
        
    def _update_scheduler_params(self, steps_per_epoch: int, epochs: int):
        """Update scheduler parameters based on actual training configuration"""
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=10000
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        epoch_losses = []
        all_predictions = []
        all_labels = []
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Collect predictions and labels for metrics
            epoch_losses.append(loss.item())
            all_predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
        
        # Calculate epoch metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        train_metrics = self.metrics_calculator.calculate_cost_weighted_metrics(all_labels, all_predictions)
        train_metrics['train_loss'] = np.mean(epoch_losses)
        
        return train_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        epoch_losses = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(sequences)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)
                
                epoch_losses.append(loss.item())
                all_predictions.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate validation metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        val_metrics = self.metrics_calculator.calculate_cost_weighted_metrics(all_labels, all_predictions)
        val_metrics['val_loss'] = np.mean(epoch_losses)
        
        return val_metrics
    
    def fit(self, 
            train_loader: DataLoader, 
            val_loader: DataLoader,
            epochs: int = 50,
            patience: int = 10,
            save_best: bool = True) -> Dict[str, Any]:
        """
        Complete training pipeline with early stopping and checkpointing
        
        Returns:
            Training history and final metrics
        """
        
        logger.info(f"Starting LSTM training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        
        # Update scheduler parameters
        self._update_scheduler_params(len(train_loader), epochs)
        
        # Initialize callbacks
        self.callbacks = AdvancedTrainingCallbacks(patience=patience, save_dir=str(self.save_dir))
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Combine metrics
            epoch_metrics = {}
            for key, value in train_metrics.items():
                epoch_metrics[f'train_{key}'] = value
            for key, value in val_metrics.items():
                epoch_metrics[f'val_{key}'] = value
            
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
            epoch_metrics['epoch_time'] = time.time() - epoch_start
            
            # Update training history
            for key, value in epoch_metrics.items():
                self.training_history[key].append(value)
            
            # Callbacks
            self.callbacks.on_epoch_end(epoch, epoch_metrics, self.model)
            
            # Logging
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val AUC: {val_metrics['val_macro_auc']:.4f} | "
                    f"Val F-beta: {val_metrics['val_macro_f_beta']:.4f} | "
                    f"Time: {epoch_metrics['epoch_time']:.1f}s"
                )
            
            # Early stopping check
            if self.callbacks.should_stop:
                logger.info(f"Training stopped early at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Generate training visualization
        fig = self.callbacks.plot_training_history(
            save_path=str(self.save_dir / "training_history.png")
        )
        
        return {
            'training_history': dict(self.training_history),
            'best_score': self.callbacks.best_score,
            'total_training_time': total_time,
            'final_epoch': self.current_epoch,
            'model_save_dir': str(self.save_dir)
        }
    
    def stratified_cross_validation(self, 
                                  dataset: torch.utils.data.Dataset,
                                  k_folds: int = 5,
                                  epochs_per_fold: int = 30) -> Dict[str, Any]:
        """
        ML Scientist Role: Stratified K-fold cross-validation for robust model evaluation
        
        Academic Justification:
        - Provides unbiased performance estimation
        - Ensures all data used for both training and validation
        - Stratification maintains class balance across folds
        """
        
        logger.info(f"Starting {k_folds}-fold stratified cross-validation")
        
        # Extract labels for stratification (using 'has_any_anomaly' as stratification key)
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            # Create binary stratification label (any anomaly vs none)
            has_anomaly = int(torch.any(label > 0.5))
            labels.append(has_anomaly)
        
        labels = np.array(labels)
        
        # Stratified K-fold split
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels)):
            logger.info(f"Training fold {fold + 1}/{k_folds}")
            
            # Create fold datasets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)
            
            # Reset model for each fold
            self.model.apply(self._reset_weights)
            
            # Train fold
            fold_results = self.fit(train_loader, val_loader, epochs=epochs_per_fold, patience=7)
            
            # Extract final validation metrics
            final_metrics = {}
            for key, values in fold_results['training_history'].items():
                if key.startswith('val_') and values:
                    final_metrics[key] = values[-1]
            
            cv_results['fold_metrics'].append(final_metrics)
            logger.info(f"Fold {fold + 1} completed - Val AUC: {final_metrics.get('val_macro_auc', 0.0):.4f}")
        
        # Calculate mean and std across folds
        metric_names = set()
        for fold_metrics in cv_results['fold_metrics']:
            metric_names.update(fold_metrics.keys())
        
        for metric_name in metric_names:
            values = [fold_metrics.get(metric_name, 0.0) for fold_metrics in cv_results['fold_metrics']]
            cv_results['mean_metrics'][metric_name] = np.mean(values)
            cv_results['std_metrics'][metric_name] = np.std(values)
        
        logger.info("Cross-validation completed")
        logger.info(f"Mean validation AUC: {cv_results['mean_metrics'].get('val_macro_auc', 0.0):.4f} "
                   f"± {cv_results['std_metrics'].get('val_macro_auc', 0.0):.4f}")
        
        return cv_results
    
    def _reset_weights(self, module):
        """Reset model weights for cross-validation"""
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    
    def save_model(self, filepath: str, include_optimizer: bool = False):
        """Save trained model"""
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_projection.in_features,
                'd_model': self.model.d_model,
                'num_anomaly_types': self.model.num_anomaly_types
            },
            'training_history': dict(self.training_history)
        }
        
        if include_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, load_optimizer: bool = False):
        """Load trained model"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {filepath}")

def create_trainer_from_config(model_config: Dict, trainer_config: Dict) -> LSTMTrainer:
    """
    Factory function to create trainer with configuration
    
    Args:
        model_config: Model architecture parameters
        trainer_config: Training hyperparameters
        
    Returns:
        Configured LSTMTrainer instance
    """
    
    # Create model
    model = create_optimized_model(**model_config)
    
    # Create trainer
    trainer = LSTMTrainer(model, **trainer_config)
    
    return trainer

if __name__ == "__main__":
    # Example training pipeline
    from lstm_data_preprocessor import LSTMDataPreprocessor
    
    # Initialize preprocessor
    preprocessor = LSTMDataPreprocessor(sequence_length=15, stratified_ratio=0.2)
    
    # Prepare data (using stratified sampling for acceleration)
    data_paths = ["data/raw/icn.csv", "data/raw/kum.csv", "data/raw/ygs.csv", "data/raw/hws.csv"]
    
    try:
        train_loader, val_loader, metadata = preprocessor.prepare_lstm_data(
            data_paths, use_stratified_subset=True
        )
        
        # Model configuration
        model_config = {
            'input_size': metadata['feature_count'],
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 2,
            'lstm_hidden': 64,
            'dropout': 0.2
        }
        
        # Trainer configuration
        trainer_config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'focal_loss_alpha': 0.7,
            'focal_loss_gamma': 2.0,
            'mixed_precision': True
        }
        
        # Create trainer
        trainer = create_trainer_from_config(model_config, trainer_config)
        
        # Train model
        results = trainer.fit(train_loader, val_loader, epochs=50, patience=10)
        
        # Save trained model
        trainer.save_model("models/lstm_trained_model.pt", include_optimizer=True)
        
        print("Training completed successfully!")
        print(f"Best validation AUC: {results['best_score']:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()