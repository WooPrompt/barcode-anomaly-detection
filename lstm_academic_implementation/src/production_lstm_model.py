#!/usr/bin/env python3
"""
Production LSTM Model - Academic Implementation
Based on: Claude_Final_LSTM_Implementation_Plan_0721_1150.md
Updated: 2025-07-22 with Critical Fixes Integration

Author: ML Engineering Team
Date: 2025-07-22

Academic Features:
- Bidirectional LSTM with multi-head attention
- Multi-label focal loss for class imbalance
- Integrated Gradients for explainability
- Production optimizations with quantization
- Integration with critical fixes for production readiness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label focal loss for handling class imbalance
    
    Based on academic plan: Addresses rare anomaly types while maintaining
    multi-label compatibility with 5-dimensional output vector.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate multi-label focal loss
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: True labels [batch_size, num_classes]
            
        Returns:
            Focal loss value
        """
        
        # Multi-label focal loss calculation
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Focal weight calculation
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ProductionLSTM(nn.Module):
    """
    Production-ready LSTM with bidirectional processing and attention
    
    Based on academic plan: Optimized architecture for temporal anomaly detection
    with multi-head attention mechanism and production constraints.
    """
    
    def __init__(self, 
                 input_size: int = 11,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 num_classes: int = 5,
                 dropout: float = 0.2,
                 attention_heads: int = 8):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.attention_heads = attention_heads
        
        # Bidirectional LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
            nn.Sigmoid()  # Multi-label probabilities
        )
        
        # Initialize weights with academic best practices
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization"""
        
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Bias initialization
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (LSTM best practice)
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            x: Input sequences [batch_size, seq_length, input_size]
            attention_mask: Optional attention mask [batch_size, seq_length]
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        
        batch_size, seq_length, _ = x.shape
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention over sequence
        if attention_mask is not None:
            # Convert boolean mask to float mask for attention
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask
        )
        
        # Layer normalization
        normalized = self.layer_norm(attn_out)
        
        # Use last time step for classification (sequence-to-one)
        if attention_mask is not None:
            # Use last valid position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
            final_representation = normalized[torch.arange(batch_size), seq_lengths]
        else:
            final_representation = normalized[:, -1, :]
        
        # Multi-label prediction
        predictions = self.classifier(final_representation)
        
        return predictions, attn_weights
    
    def get_attention_patterns(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention patterns for explainability analysis
        
        Args:
            x: Input sequences [batch_size, seq_length, input_size]
            
        Returns:
            Dictionary containing attention analysis
        """
        
        self.eval()
        with torch.no_grad():
            predictions, attention_weights = self.forward(x)
            
            # Analyze temporal attention patterns
            temporal_attention = attention_weights.mean(dim=1)  # Average over heads
            
            # Attention focus analysis
            early_focus = temporal_attention[:, :5].mean(dim=1)    # First 5 positions
            middle_focus = temporal_attention[:, 5:10].mean(dim=1) # Middle 5 positions
            late_focus = temporal_attention[:, 10:].mean(dim=1)    # Last 5+ positions
            
            patterns = {
                'raw_attention': attention_weights,
                'temporal_attention': temporal_attention,
                'early_focus': early_focus,
                'middle_focus': middle_focus,
                'late_focus': late_focus,
                'attention_entropy': self._calculate_attention_entropy(temporal_attention)
            }
            
            return patterns
    
    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of attention distribution"""
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attention_probs = attention_weights + eps
        
        # Normalize to ensure valid probability distribution
        attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = -(attention_probs * torch.log(attention_probs)).sum(dim=-1)
        
        return entropy

class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainability for LSTM model
    
    Based on academic plan: Model-agnostic explanations to avoid
    post-hoc rationalization bias common in attention-based explanations.
    """
    
    def __init__(self, model: ProductionLSTM):
        self.model = model
        
    def integrated_gradients(self, 
                           input_sequence: torch.Tensor,
                           target_class: int,
                           baseline: Optional[torch.Tensor] = None,
                           steps: int = 50) -> torch.Tensor:
        """
        Generate Integrated Gradients explanations
        
        Args:
            input_sequence: Input sequence [1, seq_length, input_size]
            target_class: Target class index for explanation
            baseline: Baseline input (default: zeros)
            steps: Number of integration steps
            
        Returns:
            Integrated gradients [seq_length, input_size]
        """
        
        if baseline is None:
            baseline = torch.zeros_like(input_sequence)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).view(-1, 1, 1)
        alphas = alphas.to(input_sequence.device)
        
        # Interpolate between baseline and input
        interpolated_inputs = baseline + alphas * (input_sequence - baseline)
        interpolated_inputs = interpolated_inputs.view(steps, *input_sequence.shape[1:])
        
        # Calculate gradients for each interpolated input
        gradients = []
        
        for i in range(steps):
            interpolated = interpolated_inputs[i:i+1]  # Add batch dimension
            interpolated.requires_grad_(True)
            
            # Forward pass
            predictions, _ = self.model(interpolated)
            
            # Calculate gradient w.r.t target class
            target_output = predictions[0, target_class]
            grad = torch.autograd.grad(target_output, interpolated, create_graph=False)[0]
            
            gradients.append(grad.squeeze(0))  # Remove batch dimension
        
        # Integrate gradients (Riemann sum approximation)
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (input_sequence.squeeze(0) - baseline.squeeze(0)) * avg_gradients
        
        return integrated_gradients
    
    def explain_prediction(self, 
                         input_sequence: torch.Tensor,
                         feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            input_sequence: Input sequence [1, seq_length, input_size]
            feature_names: Names of input features
            
        Returns:
            Explanation dictionary with feature attributions
        """
        
        self.model.eval()
        
        # Get model prediction
        with torch.no_grad():
            predictions, attention_weights = self.model(input_sequence)
        
        predictions_np = predictions.cpu().numpy()[0]
        anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
        
        # Generate explanations for each predicted anomaly
        explanations = {}
        
        for i, (anomaly_type, pred_score) in enumerate(zip(anomaly_types, predictions_np)):
            if pred_score > 0.1:  # Only explain significant predictions
                
                # Calculate Integrated Gradients
                ig_attributions = self.integrated_gradients(input_sequence, target_class=i)
                
                # Aggregate attributions by feature across time steps
                feature_attributions = {}
                
                for j, feature_name in enumerate(feature_names):
                    if j < ig_attributions.shape[1]:
                        # Sum attributions across time steps for this feature
                        total_attribution = ig_attributions[:, j].sum().item()
                        feature_attributions[feature_name] = total_attribution
                
                # Temporal attribution analysis
                temporal_attributions = ig_attributions.abs().sum(dim=1).cpu().numpy()
                
                explanations[anomaly_type] = {
                    'prediction_score': float(pred_score),
                    'feature_attributions': feature_attributions,
                    'temporal_attributions': temporal_attributions.tolist(),
                    'top_contributing_features': sorted(
                        feature_attributions.items(), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )[:5]
                }
        
        return explanations

class LSTMTrainer:
    """
    Academic-grade LSTM trainer with comprehensive metrics and validation
    
    Features:
    - Mixed precision training for efficiency
    - Comprehensive evaluation metrics
    - Learning rate scheduling
    - Early stopping with patience
    """
    
    def __init__(self, 
                 model: ProductionLSTM,
                 device: torch.device = None):
        
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training components
        self.criterion = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
    def setup_training(self, 
                      learning_rate: float = 1e-3,
                      weight_decay: float = 1e-4,
                      max_epochs: int = 50):
        """Setup optimizer and learning rate scheduler"""
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # We'll set up OneCycleLR after knowing the number of training steps
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device).float()
            labels = labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions, _ = self.model(sequences)
                    loss = self.criterion(predictions, labels)
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions, _ = self.model(sequences)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate model performance"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device).float()
                labels = labels.to(self.device).float()
                
                predictions, _ = self.model(sequences)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all predictions and labels
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(all_predictions, all_labels)
        metrics['val_loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        from sklearn.metrics import roc_auc_score, average_precision_score, hamming_loss
        
        metrics = {}
        
        try:
            # Multi-label AUC metrics
            metrics['macro_auc'] = roc_auc_score(labels, predictions, average='macro')
            metrics['micro_auc'] = roc_auc_score(labels, predictions, average='micro')
            
            # Average Precision
            metrics['macro_ap'] = average_precision_score(labels, predictions, average='macro')
            metrics['micro_ap'] = average_precision_score(labels, predictions, average='micro')
            
            # Hamming Loss
            binary_predictions = (predictions > 0.5).astype(int)
            metrics['hamming_loss'] = hamming_loss(labels, binary_predictions)
            
            # Per-class metrics
            anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
            
            for i, anomaly_type in enumerate(anomaly_types):
                if i < labels.shape[1]:
                    if len(np.unique(labels[:, i])) > 1:  # Only if both classes present
                        auc = roc_auc_score(labels[:, i], predictions[:, i])
                        metrics[f'{anomaly_type}_auc'] = auc
            
        except Exception as e:
            logger.warning(f"Metric calculation failed: {e}")
            metrics['macro_auc'] = 0.0
        
        return metrics
    
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              epochs: int = 50,
              patience: int = 10) -> Dict[str, Any]:
        """
        Complete training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training results and final metrics
        """
        
        # Setup scheduler after knowing training steps
        total_steps = len(train_loader) * epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate * 10,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"Starting training for {epochs} epochs with patience {patience}")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['val_loss']
            val_auc = val_metrics.get('macro_auc', 0.0)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_metrics'].append(val_metrics)
            
            # Early stopping check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = self.validate(val_loader)
        
        training_results = {
            'best_val_auc': best_val_auc,
            'final_metrics': final_metrics,
            'total_epochs': epoch + 1,
            'training_history': self.training_history,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        logger.info(f"Training complete: Best Val AUC: {best_val_auc:.4f}")
        
        return training_results

def quantize_model(model: ProductionLSTM, output_path: str) -> torch.jit.ScriptModule:
    """
    Quantize LSTM model for production deployment
    
    Based on academic plan: INT8 quantization for 4x speedup while
    maintaining accuracy for production inference.
    """
    
    logger.info("Quantizing LSTM model for production deployment")
    
    model.eval()
    
    # Dynamic quantization (better accuracy than static for LSTMs)
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.LSTM}, 
        dtype=torch.qint8
    )
    
    # TorchScript compilation for additional optimization
    scripted_model = torch.jit.script(quantized_model)
    scripted_model.save(output_path)
    
    logger.info(f"Quantized model saved: {output_path}")
    
    return scripted_model

if __name__ == "__main__":
    # Example usage and testing
    
    try:
        print("Testing Production LSTM Model...")
        
        # Model configuration
        input_size = 11
        hidden_size = 64
        num_classes = 5
        sequence_length = 15
        batch_size = 32
        
        # Create model
        model = ProductionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes
        )
        
        # Test forward pass
        sample_input = torch.randn(batch_size, sequence_length, input_size)
        predictions, attention_weights = model(sample_input)
        
        print(f"Model output shape: {predictions.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        
        # Test focal loss
        focal_loss = MultiLabelFocalLoss()
        sample_labels = torch.randint(0, 2, (batch_size, num_classes)).float()
        loss = focal_loss(predictions, sample_labels)
        print(f"Focal loss: {loss.item():.4f}")
        
        # Test Integrated Gradients
        explainer = IntegratedGradientsExplainer(model)
        single_input = sample_input[:1]  # Single sample
        
        feature_names = [
            'time_gap_log', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'location_changed', 'business_step_regression', 'location_entropy',
            'time_entropy', 'scan_progress', 'is_business_hours'
        ]
        
        explanations = explainer.explain_prediction(single_input, feature_names)
        print(f"Generated explanations for {len(explanations)} anomaly types")
        
        # Test attention patterns
        patterns = model.get_attention_patterns(single_input)
        print(f"Attention pattern analysis complete")
        
        print("✅ Production LSTM Model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()