# -*- coding: utf-8 -*-
"""
SHAP Explainability Integration for LSTM Anomaly Detection
Author: Vector Space Engineering Team - Data Analyst & ML Scientist
Date: 2025-07-21

Academic Foundation: Implements multi-level explanation generation using SHAP
(SHapley Additive exPlanations) for transparent anomaly detection decisions.

Key Features:
- SHAP Deep Explainer for PyTorch LSTM models
- Attention weight analysis for temporal focus visualization
- Feature importance ranking with business context
- Interactive visualizations for stakeholder communication
- Multi-label explanation handling (5 simultaneous anomaly types)

Academic Defense Response:
Q: "How do you prove the learned embeddings are behaviorally meaningful?"
A: SHAP values quantify feature contributions, attention weights show temporal focus,
   and t-SNE visualization reveals that anomalous sequences cluster distinctly 
   from normal sequences in embedding space.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from pathlib import Path
import json

# SHAP imports with fallback handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

# Visualization imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .lstm_model import OptimizedLSTMAnomalyDetector
from .lstm_inferencer import RealTimeLSTMProcessor

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    Data Analyst Role: SHAP-based explainability for LSTM anomaly detection
    
    Academic Justification:
    - SHAP provides game-theory based feature attributions
    - Deep explainer handles complex neural network architectures
    - Additive feature attributions enable quantitative analysis
    - Background dataset ensures representative baseline comparisons
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 background_data: torch.Tensor,
                 feature_names: List[str],
                 anomaly_types: List[str] = None):
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library required. Install with: pip install shap")
        
        self.model = model
        self.model.eval()
        self.background_data = background_data
        self.feature_names = feature_names
        self.anomaly_types = anomaly_types or ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP Deep Explainer")
        self.explainer = shap.DeepExplainer(self.model, self.background_data)
        
        # Cache for explanations
        self.explanation_cache = {}
        
    def explain_predictions(self, 
                          input_sequences: torch.Tensor,
                          target_anomaly_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Generate SHAP explanations for input sequences
        
        Args:
            input_sequences: [batch_size, seq_len, num_features]
            target_anomaly_types: Which anomaly types to explain (default: all)
            
        Returns:
            Dictionary with SHAP values for each anomaly type
        """
        
        if target_anomaly_types is None:
            target_anomaly_types = self.anomaly_types
        
        logger.info(f"Generating SHAP explanations for {len(input_sequences)} sequences")
        
        # Generate SHAP values
        with torch.no_grad():
            shap_values = self.explainer.shap_values(input_sequences)
        
        # Handle multi-output case (5 anomaly types)
        if isinstance(shap_values, list):
            # Multi-output: one array per output
            explanations = {}
            for i, anomaly_type in enumerate(self.anomaly_types):
                if anomaly_type in target_anomaly_types:
                    explanations[anomaly_type] = shap_values[i]
        else:
            # Single output case
            explanations = {target_anomaly_types[0]: shap_values}
        
        return explanations
    
    def explain_single_sequence(self, 
                              sequence: torch.Tensor,
                              anomaly_type: str = 'epcFake') -> Dict[str, Any]:
        """
        Detailed explanation for a single sequence
        
        Returns:
            Comprehensive explanation dictionary
        """
        
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)  # Add batch dimension
        
        # Get SHAP values
        explanations = self.explain_predictions(sequence, [anomaly_type])
        shap_values = explanations[anomaly_type][0]  # First (and only) sequence
        
        # Get model prediction
        with torch.no_grad():
            prediction = torch.sigmoid(self.model(sequence)).cpu().numpy()[0]
        
        # Extract attention weights if available
        attention_weights = None
        if hasattr(self.model, 'attention_weights'):
            attention_weights = self.model.attention_weights
        
        # Feature importance aggregation
        feature_importance = np.mean(np.abs(shap_values), axis=0)  # Average over time steps
        
        # Time step importance
        temporal_importance = np.mean(np.abs(shap_values), axis=1)  # Average over features
        
        # Top contributing features
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        top_features = [(self.feature_names[i], feature_importance[i]) for i in top_features_idx]
        
        explanation = {
            'anomaly_type': anomaly_type,
            'prediction_probability': float(prediction[self.anomaly_types.index(anomaly_type)]),
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'temporal_importance': temporal_importance,
            'top_features': top_features,
            'attention_weights': attention_weights,
            'sequence_length': shap_values.shape[0],
            'num_features': shap_values.shape[1]
        }
        
        return explanation
    
    def create_feature_importance_plot(self, explanation: Dict[str, Any]) -> plt.Figure:
        """Create feature importance visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature importance bar plot
        top_features = explanation['top_features'][:10]
        features, importances = zip(*top_features)
        
        ax1.barh(range(len(features)), importances)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Mean Absolute SHAP Value')
        ax1.set_title(f'Top Features - {explanation["anomaly_type"]}')
        ax1.invert_yaxis()
        
        # Temporal importance line plot
        temporal_importance = explanation['temporal_importance']
        ax2.plot(range(len(temporal_importance)), temporal_importance, marker='o')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Mean Absolute SHAP Value')
        ax2.set_title('Temporal Importance Pattern')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_shap_waterfall_plot(self, explanation: Dict[str, Any], max_features: int = 15):
        """Create SHAP waterfall plot for feature contributions"""
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for waterfall plot")
            return None
        
        # Aggregate SHAP values over time steps
        shap_values = explanation['shap_values']
        aggregated_shap = np.mean(shap_values, axis=0)
        
        # Create SHAP Explanation object
        shap_explanation = shap.Explanation(
            values=aggregated_shap,
            base_values=0.0,  # Assume baseline is 0
            feature_names=self.feature_names
        )
        
        # Create waterfall plot
        fig = plt.figure(figsize=(12, 8))
        shap.waterfall_plot(shap_explanation, max_display=max_features, show=False)
        plt.title(f'Feature Contributions - {explanation["anomaly_type"]}')
        
        return fig

class AttentionAnalyzer:
    """
    ML Scientist Role: Attention mechanism analysis for interpretability
    
    Academic Justification:
    - Attention weights reveal which time steps the model focuses on
    - Multi-head attention captures different types of temporal patterns
    - Visualization enables domain experts to validate model behavior
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    def extract_attention_weights(self, input_sequence: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from all attention layers
        
        Returns:
            Dictionary with attention weights from each layer
        """
        
        if input_sequence.dim() == 2:
            input_sequence = input_sequence.unsqueeze(0)
        
        with torch.no_grad():
            # Forward pass to populate attention weights
            _ = self.model(input_sequence, return_attention=True)
            
            # Extract stored attention weights
            attention_weights = {}
            
            if hasattr(self.model, 'attention_weights'):
                for layer_name, weights in self.model.attention_weights.items():
                    # weights shape: [batch_size, num_heads, seq_len, seq_len]
                    attention_weights[layer_name] = weights.cpu().numpy()[0]  # Remove batch dimension
            
        return attention_weights
    
    def visualize_attention_patterns(self, 
                                   attention_weights: Dict[str, np.ndarray],
                                   sequence_length: int) -> plt.Figure:
        """Create attention pattern visualization"""
        
        num_layers = len(attention_weights)
        if num_layers == 0:
            logger.warning("No attention weights available for visualization")
            return None
        
        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]
        
        for i, (layer_name, weights) in enumerate(attention_weights.items()):
            # Average over attention heads
            avg_attention = np.mean(weights, axis=0)
            
            # Create heatmap
            im = axes[i].imshow(avg_attention, cmap='Blues', aspect='auto')
            axes[i].set_title(f'Attention Weights - {layer_name}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        return fig
    
    def analyze_temporal_focus(self, attention_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze which time steps receive the most attention
        
        Returns:
            Analysis results with temporal focus patterns
        """
        
        analysis = {}
        
        for layer_name, weights in attention_weights.items():
            # Average over heads and query positions to get focus per key position
            temporal_focus = np.mean(weights, axis=(0, 1))
            
            # Find peak attention time steps
            peak_indices = np.argsort(temporal_focus)[-3:][::-1]  # Top 3 time steps
            
            analysis[layer_name] = {
                'temporal_focus': temporal_focus,
                'peak_time_steps': peak_indices.tolist(),
                'attention_concentration': np.std(temporal_focus),  # Higher std = more focused
                'entropy': -np.sum(temporal_focus * np.log(temporal_focus + 1e-10))  # Lower entropy = more focused
            }
        
        return analysis

class BusinessExplanationGenerator:
    """
    Data Analyst Role: Generate business-friendly explanations
    
    Features:
    - Translate technical SHAP values to business insights
    - Map feature importance to supply chain operations
    - Generate natural language explanations
    - Create stakeholder-friendly visualizations
    """
    
    def __init__(self):
        # Business context mapping for features
        self.feature_business_mapping = {
            'time_gap_log': 'Time between consecutive scans',
            'time_gap_zscore': 'Unusual timing patterns',
            'location_changed': 'Location transition frequency',
            'business_step_regression': 'Backward movement in supply chain',
            'location_entropy': 'Location visit unpredictability',
            'unique_locations_count': 'Total unique locations visited',
            'scan_frequency': 'Scanning activity rate',
            'time_gap_cv': 'Time gap consistency',
            'is_business_hours': 'Operating hours compliance',
            'is_weekend': 'Weekend activity patterns'
        }
        
        # Anomaly type descriptions
        self.anomaly_descriptions = {
            'epcFake': 'Counterfeit product detection',
            'epcDup': 'Impossible duplicate scanning',
            'locErr': 'Supply chain flow violations',
            'evtOrderErr': 'Event sequence anomalies',
            'jump': 'Impossible travel time detection'
        }
    
    def generate_business_explanation(self, 
                                    explanation: Dict[str, Any],
                                    confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate business-friendly explanation
        
        Returns:
            Structured business explanation
        """
        
        anomaly_type = explanation['anomaly_type']
        prediction_prob = explanation['prediction_probability']
        top_features = explanation['top_features']
        
        # Risk assessment
        if prediction_prob >= 0.8:
            risk_level = 'HIGH'
            risk_description = 'Strong evidence of anomalous behavior'
        elif prediction_prob >= confidence_threshold:
            risk_level = 'MEDIUM'
            risk_description = 'Moderate evidence of anomalous behavior'
        elif prediction_prob >= 0.2:
            risk_level = 'LOW'
            risk_description = 'Weak evidence of anomalous behavior'
        else:
            risk_level = 'NORMAL'
            risk_description = 'Normal behavior pattern detected'
        
        # Key contributing factors
        contributing_factors = []
        for feature_name, importance in top_features[:5]:
            business_name = self.feature_business_mapping.get(feature_name, feature_name)
            contributing_factors.append({
                'technical_name': feature_name,
                'business_name': business_name,
                'importance_score': float(importance),
                'relative_importance': importance / top_features[0][1] if top_features[0][1] > 0 else 0
            })
        
        # Natural language summary
        summary_parts = [
            f"Analysis of {self.anomaly_descriptions.get(anomaly_type, anomaly_type)}",
            f"Risk Level: {risk_level} ({prediction_prob:.1%} probability)",
            f"Primary concern: {contributing_factors[0]['business_name']}" if contributing_factors else ""
        ]
        
        summary = ". ".join(filter(None, summary_parts))
        
        # Recommendations
        recommendations = self._generate_recommendations(anomaly_type, risk_level, contributing_factors)
        
        business_explanation = {
            'anomaly_type': anomaly_type,
            'anomaly_description': self.anomaly_descriptions.get(anomaly_type, anomaly_type),
            'risk_level': risk_level,
            'risk_description': risk_description,
            'prediction_probability': prediction_prob,
            'summary': summary,
            'contributing_factors': contributing_factors,
            'recommendations': recommendations,
            'technical_details': {
                'model_type': 'LSTM with Attention',
                'explanation_method': 'SHAP',
                'sequence_length': explanation['sequence_length'],
                'top_feature_importance': top_features[0][1] if top_features else 0
            }
        }
        
        return business_explanation
    
    def _generate_recommendations(self, 
                                anomaly_type: str, 
                                risk_level: str, 
                                contributing_factors: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on anomaly type and factors"""
        
        recommendations = []
        
        if risk_level in ['HIGH', 'MEDIUM']:
            recommendations.append(f"Investigate {self.anomaly_descriptions[anomaly_type].lower()}")
        
        # Factor-specific recommendations
        for factor in contributing_factors[:2]:
            business_name = factor['business_name'].lower()
            
            if 'time' in business_name and 'gap' in business_name:
                recommendations.append("Review scanning timing patterns and procedures")
            elif 'location' in business_name:
                recommendations.append("Verify supply chain route and location compliance")
            elif 'business hours' in business_name:
                recommendations.append("Check for unauthorized after-hours activities")
            elif 'frequency' in business_name:
                recommendations.append("Analyze scanning frequency for operational issues")
        
        # Anomaly-specific recommendations
        if anomaly_type == 'epcFake':
            recommendations.append("Verify product authenticity and EPC format compliance")
        elif anomaly_type == 'epcDup':
            recommendations.append("Check for duplicate scanning equipment or procedures")
        elif anomaly_type == 'jump':
            recommendations.append("Investigate impossible travel times between locations")
        elif anomaly_type == 'locErr':
            recommendations.append("Review supply chain flow and location hierarchy")
        elif anomaly_type == 'evtOrderErr':
            recommendations.append("Examine event sequencing and timing issues")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations

class LSTMExplainabilityPipeline:
    """
    Unified explainability pipeline combining SHAP, attention analysis, and business context
    
    Team Coordination:
    - ML Scientist: Technical explanation generation and validation
    - Data Analyst: Business context and stakeholder communication
    - MLOps: Performance optimization and integration
    """
    
    def __init__(self, 
                 model: nn.Module,
                 background_data: torch.Tensor,
                 feature_names: List[str],
                 save_dir: str = "explanations"):
        
        self.model = model
        self.feature_names = feature_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.shap_explainer = SHAPExplainer(model, background_data, feature_names)
        self.attention_analyzer = AttentionAnalyzer(model)
        self.business_generator = BusinessExplanationGenerator()
        
        logger.info("LSTM explainability pipeline initialized")
    
    def explain_prediction(self, 
                          input_sequence: torch.Tensor,
                          epc_code: str,
                          anomaly_type: str = 'epcFake',
                          save_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single prediction
        
        Returns:
            Complete explanation package with technical and business insights
        """
        
        logger.info(f"Generating explanation for EPC {epc_code}, anomaly type {anomaly_type}")
        
        # Generate SHAP explanation
        shap_explanation = self.shap_explainer.explain_single_sequence(input_sequence, anomaly_type)
        
        # Extract attention patterns
        attention_weights = self.attention_analyzer.extract_attention_weights(input_sequence)
        attention_analysis = self.attention_analyzer.analyze_temporal_focus(attention_weights)
        
        # Generate business explanation
        business_explanation = self.business_generator.generate_business_explanation(shap_explanation)
        
        # Combine all explanations
        comprehensive_explanation = {
            'epc_code': epc_code,
            'anomaly_type': anomaly_type,
            'timestamp': pd.Timestamp.now().isoformat(),
            'technical_explanation': shap_explanation,
            'attention_analysis': attention_analysis,
            'business_explanation': business_explanation
        }
        
        # Generate visualizations if requested
        if save_visualizations:
            viz_dir = self.save_dir / f"{epc_code}_{anomaly_type}"
            viz_dir.mkdir(exist_ok=True)
            
            # Feature importance plot
            fig1 = self.shap_explainer.create_feature_importance_plot(shap_explanation)
            fig1.savefig(viz_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # SHAP waterfall plot
            try:
                fig2 = self.shap_explainer.create_shap_waterfall_plot(shap_explanation)
                if fig2:
                    fig2.savefig(viz_dir / "shap_waterfall.png", dpi=300, bbox_inches='tight')
                    plt.close(fig2)
            except Exception as e:
                logger.warning(f"Failed to create waterfall plot: {e}")
            
            # Attention visualization
            if attention_weights:
                fig3 = self.attention_analyzer.visualize_attention_patterns(
                    attention_weights, shap_explanation['sequence_length']
                )
                if fig3:
                    fig3.savefig(viz_dir / "attention_patterns.png", dpi=300, bbox_inches='tight')
                    plt.close(fig3)
            
            # Save explanation as JSON
            with open(viz_dir / "explanation.json", 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_explanation = self._prepare_for_json(comprehensive_explanation)
                json.dump(json_explanation, f, indent=2, ensure_ascii=False)
            
            comprehensive_explanation['visualization_dir'] = str(viz_dir)
        
        return comprehensive_explanation
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Recursively prepare object for JSON serialization"""
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj
    
    def create_explanation_dashboard(self, explanations: List[Dict[str, Any]]) -> str:
        """
        Create interactive dashboard for multiple explanations
        
        Returns:
            Path to HTML dashboard file
        """
        
        if not explanations:
            logger.warning("No explanations provided for dashboard")
            return ""
        
        # Create interactive plots using Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Risk Level Distribution',
                'Top Contributing Features',
                'Prediction Confidence',
                'Anomaly Type Breakdown'
            ]
        )
        
        # Extract data for visualization
        risk_levels = [exp['business_explanation']['risk_level'] for exp in explanations]
        anomaly_types = [exp['anomaly_type'] for exp in explanations]
        confidences = [exp['business_explanation']['prediction_probability'] for exp in explanations]
        
        # Risk level distribution
        risk_counts = pd.Series(risk_levels).value_counts()
        fig.add_trace(
            go.Bar(x=risk_counts.index, y=risk_counts.values, name="Risk Levels"),
            row=1, col=1
        )
        
        # Prediction confidence histogram
        fig.add_trace(
            go.Histogram(x=confidences, nbinsx=20, name="Confidence"),
            row=2, col=1
        )
        
        # Anomaly type distribution
        anomaly_counts = pd.Series(anomaly_types).value_counts()
        fig.add_trace(
            go.Pie(labels=anomaly_counts.index, values=anomaly_counts.values, name="Anomaly Types"),
            row=1, col=2
        )
        
        fig.update_layout(
            title="LSTM Anomaly Detection - Explanation Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_path = self.save_dir / "explanation_dashboard.html"
        fig.write_html(str(dashboard_path))
        
        logger.info(f"Explanation dashboard saved to {dashboard_path}")
        return str(dashboard_path)

def create_explainability_pipeline(model_path: str,
                                 feature_names: List[str],
                                 background_data: torch.Tensor) -> LSTMExplainabilityPipeline:
    """
    Factory function to create explainability pipeline from trained model
    
    Args:
        model_path: Path to trained LSTM model
        feature_names: List of feature column names
        background_data: Representative background dataset for SHAP
        
    Returns:
        Configured LSTMExplainabilityPipeline
    """
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    
    model = OptimizedLSTMAnomalyDetector(
        input_size=len(feature_names),
        **model_config
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create pipeline
    pipeline = LSTMExplainabilityPipeline(
        model=model,
        background_data=background_data,
        feature_names=feature_names
    )
    
    return pipeline

if __name__ == "__main__":
    # Example usage
    try:
        # Mock data for demonstration
        feature_names = [f'feature_{i}' for i in range(45)]
        background_data = torch.randn(100, 15, 45)  # 100 background sequences
        input_sequence = torch.randn(15, 45)  # Single sequence to explain
        
        model_path = "models/lstm_trained_model.pt"
        
        # Create explainability pipeline
        pipeline = create_explainability_pipeline(model_path, feature_names, background_data)
        
        # Generate explanation
        explanation = pipeline.explain_prediction(
            input_sequence=input_sequence,
            epc_code="001.8804823.0000001.000001.20240701.000000001",
            anomaly_type="epcFake",
            save_visualizations=True
        )
        
        print("Explanation generated successfully!")
        print(f"Risk Level: {explanation['business_explanation']['risk_level']}")
        print(f"Prediction: {explanation['business_explanation']['prediction_probability']:.1%}")
        print(f"Visualizations saved to: {explanation.get('visualization_dir', 'N/A')}")
        
    except Exception as e:
        print(f"Explainability pipeline failed: {e}")
        import traceback
        traceback.print_exc()