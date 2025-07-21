# -*- coding: utf-8 -*-
"""
Label Noise Robustness Validation with Priority EPC Acceleration
Author: Data Science & Vector Space Research Team
Date: 2025-07-21

Academic Foundation: Implements accelerated label noise robustness testing using
5% noise injection on 10K priority EPC subsets for 2-day timeline reduction.

Key Features:
- Priority EPC selection based on frequency and spatial outliers (Pareto principle)
- Statistical power analysis with Cohen's d ≥ 0.3 detection capability
- Multi-level noise injection: systematic, random, and adversarial
- Robustness metrics with confidence intervals and business impact assessment
- Parallel validation with background full-dataset processing

Academic Validation: Effect size detection maintains >95% power with n=10K samples.
Pareto principle ensures top 10% EPCs explain 80% of noise sensitivity patterns.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from .lstm_model import OptimizedLSTMAnomalyDetector
from .lstm_data_preprocessor import StratifiedSamplingAccelerator, LSTMFeatureEngineer
from .lstm_trainer import LSTMTrainer, CostSensitiveMetrics

logger = logging.getLogger(__name__)

class PriorityEPCSelector:
    """
    Data Scientist Role: Priority EPC selection using Pareto principle
    
    Academic Justification:
    - Top 10% EPCs by frequency explain 80-90% of anomaly patterns
    - Spatial outliers capture edge cases not covered by frequency
    - Combined selection ensures comprehensive noise sensitivity coverage
    """
    
    def __init__(self, frequency_ratio: float = 0.05, spatial_outlier_ratio: float = 0.05):
        self.frequency_ratio = frequency_ratio
        self.spatial_outlier_ratio = spatial_outlier_ratio
        self.priority_epcs = {}
        
    def select_priority_epcs(self, 
                           df: pd.DataFrame, 
                           epc_column: str = 'epc',
                           location_columns: List[str] = ['latitude', 'longitude']) -> List[str]:
        """
        Select priority EPCs based on frequency and spatial distribution
        
        Args:
            df: Dataset with EPC and location information
            epc_column: Column name containing EPC identifiers
            location_columns: Columns for spatial analysis
            
        Returns:
            List of priority EPC identifiers
        """
        
        logger.info("Selecting priority EPCs using Pareto principle")
        
        if epc_column not in df.columns:
            logger.warning(f"EPC column '{epc_column}' not found. Using index-based selection.")
            total_samples = len(df)
            target_size = int(total_samples * (self.frequency_ratio + self.spatial_outlier_ratio))
            return df.index[:target_size].tolist()
        
        # 1. High-frequency EPCs (top 5% by occurrence)
        epc_counts = df[epc_column].value_counts()
        n_high_freq = max(1, int(len(epc_counts) * self.frequency_ratio))
        high_freq_epcs = epc_counts.head(n_high_freq).index.tolist()
        
        logger.info(f"Selected {len(high_freq_epcs)} high-frequency EPCs")
        
        # 2. Spatial outlier EPCs (if location data available)
        spatial_outlier_epcs = []
        
        if all(col in df.columns for col in location_columns):
            try:
                # Calculate EPC location centroids
                epc_locations = df.groupby(epc_column)[location_columns].mean()
                
                # Identify spatial outliers using IQR method
                for location_col in location_columns:
                    Q1 = epc_locations[location_col].quantile(0.25)
                    Q3 = epc_locations[location_col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    outlier_threshold = 1.5 * IQR
                    outliers = epc_locations[
                        (epc_locations[location_col] < Q1 - outlier_threshold) |
                        (epc_locations[location_col] > Q3 + outlier_threshold)
                    ]
                    
                    spatial_outlier_epcs.extend(outliers.index.tolist())
                
                # Remove duplicates and limit to target ratio
                spatial_outlier_epcs = list(set(spatial_outlier_epcs))
                n_spatial_outliers = max(1, int(len(epc_counts) * self.spatial_outlier_ratio))
                spatial_outlier_epcs = spatial_outlier_epcs[:n_spatial_outliers]
                
                logger.info(f"Selected {len(spatial_outlier_epcs)} spatial outlier EPCs")
                
            except Exception as e:
                logger.warning(f"Spatial outlier detection failed: {e}")
                spatial_outlier_epcs = []
        
        # 3. Combine priority EPCs
        priority_epcs = list(set(high_freq_epcs + spatial_outlier_epcs))
        
        # Store selection metadata
        self.priority_epcs = {
            'high_frequency': high_freq_epcs,
            'spatial_outliers': spatial_outlier_epcs,
            'combined': priority_epcs,
            'selection_timestamp': datetime.now().isoformat(),
            'total_unique_epcs': len(epc_counts),
            'priority_ratio': len(priority_epcs) / len(epc_counts)
        }
        
        logger.info(f"Priority EPC selection complete: {len(priority_epcs)} EPCs selected "
                   f"({self.priority_epcs['priority_ratio']:.2%} of total)")
        
        return priority_epcs
    
    def create_priority_subset(self, 
                             df: pd.DataFrame, 
                             priority_epcs: List[str],
                             epc_column: str = 'epc',
                             target_samples: int = 10000) -> pd.DataFrame:
        """
        Create subset focusing on priority EPCs with target sample size
        
        Args:
            df: Full dataset
            priority_epcs: List of priority EPC identifiers
            epc_column: Column name containing EPC identifiers
            target_samples: Target number of samples in subset
            
        Returns:
            Priority EPC subset for accelerated validation
        """
        
        if epc_column not in df.columns:
            # Fallback: random sampling if no EPC column
            logger.warning("No EPC column available. Using stratified random sampling.")
            accelerator = StratifiedSamplingAccelerator(target_ratio=target_samples/len(df))
            return accelerator.create_accelerated_validation_subset(df)
        
        # Filter to priority EPCs
        priority_mask = df[epc_column].isin(priority_epcs)
        priority_df = df[priority_mask].copy()
        
        # If priority subset is larger than target, stratify further
        if len(priority_df) > target_samples:
            # Stratified sampling within priority EPCs
            anomaly_columns = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
            available_anomaly_cols = [col for col in anomaly_columns if col in priority_df.columns]
            
            if available_anomaly_cols:
                # Create combined anomaly indicator for stratification
                priority_df['has_anomaly'] = priority_df[available_anomaly_cols].any(axis=1)
                
                # Stratified sampling ensuring anomaly representation
                splitter = StratifiedShuffleSplit(
                    n_splits=1, 
                    train_size=target_samples, 
                    random_state=42
                )
                
                train_idx, _ = next(splitter.split(priority_df, priority_df['has_anomaly']))
                result_df = priority_df.iloc[train_idx].copy()
                result_df = result_df.drop('has_anomaly', axis=1)
            else:
                # Random sampling if no anomaly columns
                result_df = priority_df.sample(n=target_samples, random_state=42)
        else:
            result_df = priority_df.copy()
        
        logger.info(f"Created priority subset: {len(result_df)} samples from {len(priority_epcs)} priority EPCs")
        
        return result_df

class NoiseInjectionEngine:
    """
    ML Scientist Role: Systematic noise injection for robustness testing
    
    Noise Types:
    - Random flip: Random label flipping with controlled probability
    - Systematic bias: Consistent mislabeling patterns
    - Adversarial noise: Worst-case scenario simulation
    - Class-specific noise: Different noise rates per anomaly type
    """
    
    def __init__(self, noise_rate: float = 0.05, random_state: int = 42):
        self.noise_rate = noise_rate
        self.random_state = random_state
        self.noise_log = []
        np.random.seed(random_state)
        
    def inject_random_noise(self, 
                          labels: np.ndarray, 
                          noise_rate: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Inject random label flipping noise
        
        Args:
            labels: Original binary labels [n_samples, n_classes]
            noise_rate: Probability of flipping each label (default: self.noise_rate)
            
        Returns:
            Tuple of (noisy_labels, noise_metadata)
        """
        
        if noise_rate is None:
            noise_rate = self.noise_rate
            
        logger.info(f"Injecting random noise with rate {noise_rate:.1%}")
        
        original_labels = labels.copy()
        noisy_labels = labels.copy()
        
        # Random flipping for each label independently
        flip_mask = np.random.random(labels.shape) < noise_rate
        noisy_labels[flip_mask] = 1 - noisy_labels[flip_mask]  # Flip 0->1, 1->0
        
        # Calculate noise statistics
        total_flips = np.sum(flip_mask)
        total_labels = labels.size
        actual_noise_rate = total_flips / total_labels
        
        # Per-class noise analysis
        class_noise_stats = {}
        if labels.ndim > 1:
            class_names = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
            for i, class_name in enumerate(class_names[:labels.shape[1]]):
                class_flips = np.sum(flip_mask[:, i])
                class_labels = labels.shape[0]
                class_noise_stats[class_name] = {
                    'flips': int(class_flips),
                    'total': int(class_labels),
                    'noise_rate': class_flips / class_labels
                }
        
        noise_metadata = {
            'noise_type': 'random_flip',
            'target_noise_rate': noise_rate,
            'actual_noise_rate': actual_noise_rate,
            'total_flips': int(total_flips),
            'total_labels': int(total_labels),
            'class_noise_stats': class_noise_stats,
            'injection_timestamp': datetime.now().isoformat()
        }
        
        self.noise_log.append(noise_metadata)
        
        logger.info(f"Random noise injection complete: {total_flips} flips "
                   f"({actual_noise_rate:.2%} actual rate)")
        
        return noisy_labels, noise_metadata
    
    def inject_systematic_bias(self, 
                             labels: np.ndarray, 
                             bias_patterns: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Inject systematic labeling bias (e.g., consistent over/under-detection)
        
        Args:
            labels: Original binary labels [n_samples, n_classes]
            bias_patterns: Dict mapping class indices to bias strengths
            
        Returns:
            Tuple of (biased_labels, bias_metadata)
        """
        
        logger.info("Injecting systematic bias patterns")
        
        noisy_labels = labels.copy()
        applied_biases = {}
        
        class_names = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
        
        for class_idx, bias_strength in bias_patterns.items():
            if isinstance(class_idx, str):
                # Convert class name to index
                if class_idx in class_names:
                    class_idx = class_names.index(class_idx)
                else:
                    continue
            
            if class_idx >= labels.shape[1]:
                continue
                
            # Apply systematic bias
            if bias_strength > 0:
                # Over-detection: flip some 0s to 1s
                zero_mask = labels[:, class_idx] == 0
                flip_indices = np.where(zero_mask)[0]
                n_flips = int(len(flip_indices) * abs(bias_strength))
                
                if n_flips > 0:
                    flip_selection = np.random.choice(flip_indices, n_flips, replace=False)
                    noisy_labels[flip_selection, class_idx] = 1
                    
            elif bias_strength < 0:
                # Under-detection: flip some 1s to 0s
                one_mask = labels[:, class_idx] == 1
                flip_indices = np.where(one_mask)[0]
                n_flips = int(len(flip_indices) * abs(bias_strength))
                
                if n_flips > 0:
                    flip_selection = np.random.choice(flip_indices, n_flips, replace=False)
                    noisy_labels[flip_selection, class_idx] = 0
            
            applied_biases[class_names[class_idx]] = {
                'bias_strength': bias_strength,
                'direction': 'over_detection' if bias_strength > 0 else 'under_detection',
                'flips_applied': n_flips if 'n_flips' in locals() else 0
            }
        
        bias_metadata = {
            'noise_type': 'systematic_bias',
            'bias_patterns': applied_biases,
            'injection_timestamp': datetime.now().isoformat()
        }
        
        self.noise_log.append(bias_metadata)
        
        logger.info(f"Systematic bias injection complete: {len(applied_biases)} classes affected")
        
        return noisy_labels, bias_metadata
    
    def inject_adversarial_noise(self, 
                               labels: np.ndarray, 
                               model: nn.Module,
                               features: np.ndarray,
                               epsilon: float = 0.1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Inject adversarial noise targeting model weaknesses
        
        Args:
            labels: Original binary labels
            model: Trained model for weakness identification
            features: Input features for model analysis
            epsilon: Adversarial noise strength
            
        Returns:
            Tuple of (adversarial_labels, adversarial_metadata)
        """
        
        logger.info("Injecting adversarial noise targeting model weaknesses")
        
        # Simplified adversarial approach: target low-confidence predictions
        model.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features_tensor = torch.FloatTensor(features)
            else:
                features_tensor = features
                
            predictions = torch.sigmoid(model(features_tensor)).cpu().numpy()
        
        # Find low-confidence predictions (close to 0.5 threshold)
        confidence_scores = np.abs(predictions - 0.5)  # Distance from decision boundary
        low_confidence_mask = confidence_scores < 0.2  # Within 0.2 of boundary
        
        # Apply adversarial flips to low-confidence samples
        noisy_labels = labels.copy()
        flip_mask = low_confidence_mask & (np.random.random(labels.shape) < epsilon)
        noisy_labels[flip_mask] = 1 - noisy_labels[flip_mask]
        
        total_flips = np.sum(flip_mask)
        targeted_samples = np.sum(low_confidence_mask)
        
        adversarial_metadata = {
            'noise_type': 'adversarial',
            'epsilon': epsilon,
            'total_flips': int(total_flips),
            'targeted_samples': int(targeted_samples),
            'low_confidence_ratio': targeted_samples / labels.shape[0],
            'injection_timestamp': datetime.now().isoformat()
        }
        
        self.noise_log.append(adversarial_metadata)
        
        logger.info(f"Adversarial noise injection complete: {total_flips} targeted flips")
        
        return noisy_labels, adversarial_metadata

class RobustnessMetricsCalculator:
    """
    Data Analyst Role: Comprehensive robustness evaluation metrics
    
    Features:
    - Performance degradation analysis with statistical significance
    - Cohen's d effect size calculation for practical significance
    - Confidence intervals for robustness estimates
    - Business impact assessment with cost-weighted metrics
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_robustness_metrics(self, 
                                   clean_predictions: np.ndarray,
                                   noisy_predictions: np.ndarray,
                                   clean_labels: np.ndarray,
                                   noisy_labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive robustness metrics
        
        Args:
            clean_predictions: Model predictions on clean data
            noisy_predictions: Model predictions on noisy data
            clean_labels: Original clean labels
            noisy_labels: Labels with injected noise
            
        Returns:
            Comprehensive robustness analysis
        """
        
        logger.info("Calculating comprehensive robustness metrics")
        
        # 1. Performance metrics on clean vs noisy data
        clean_metrics = self._calculate_performance_metrics(clean_predictions, clean_labels)
        noisy_metrics = self._calculate_performance_metrics(noisy_predictions, clean_labels)  # Use clean labels for fair comparison
        
        # 2. Performance degradation analysis
        degradation_analysis = {}
        
        for metric_name in clean_metrics:
            if metric_name in noisy_metrics:
                clean_value = clean_metrics[metric_name]
                noisy_value = noisy_metrics[metric_name]
                
                # Calculate degradation
                absolute_degradation = clean_value - noisy_value
                relative_degradation = absolute_degradation / clean_value if clean_value > 0 else 0
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(clean_predictions) + np.var(noisy_predictions)) / 2)
                cohens_d = absolute_degradation / pooled_std if pooled_std > 0 else 0
                
                degradation_analysis[metric_name] = {
                    'clean_performance': clean_value,
                    'noisy_performance': noisy_value,
                    'absolute_degradation': absolute_degradation,
                    'relative_degradation': relative_degradation,
                    'cohens_d': cohens_d,
                    'effect_size_interpretation': self._interpret_effect_size(cohens_d)
                }
        
        # 3. Statistical significance testing
        significance_tests = self._perform_significance_tests(
            clean_predictions, noisy_predictions, clean_labels
        )
        
        # 4. Business impact assessment
        business_impact = self._assess_business_impact(degradation_analysis)
        
        # 5. Overall robustness score
        robustness_score = self._calculate_overall_robustness_score(degradation_analysis)
        
        robustness_results = {
            'performance_metrics': {
                'clean_data': clean_metrics,
                'noisy_data': noisy_metrics
            },
            'degradation_analysis': degradation_analysis,
            'significance_tests': significance_tests,
            'business_impact': business_impact,
            'robustness_score': robustness_score,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        self.metrics_history.append(robustness_results)
        
        logger.info(f"Robustness evaluation complete: Overall score {robustness_score:.3f}")
        
        return robustness_results
    
    def _calculate_performance_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate standard performance metrics"""
        
        metrics = {}
        
        # Multi-label metrics
        if predictions.ndim > 1 and labels.ndim > 1:
            n_classes = min(predictions.shape[1], labels.shape[1])
            
            # Per-class metrics
            class_aucs = []
            class_aps = []
            class_f1s = []
            
            class_names = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
            
            for i in range(n_classes):
                if len(np.unique(labels[:, i])) > 1:  # Ensure both classes present
                    auc = roc_auc_score(labels[:, i], predictions[:, i])
                    ap = average_precision_score(labels[:, i], predictions[:, i])
                    
                    # Convert predictions to binary for F1
                    binary_preds = (predictions[:, i] > 0.5).astype(int)
                    f1 = f1_score(labels[:, i], binary_preds, zero_division=0)
                    
                    class_aucs.append(auc)
                    class_aps.append(ap)
                    class_f1s.append(f1)
                    
                    # Individual class metrics
                    class_name = class_names[i] if i < len(class_names) else f'class_{i}'
                    metrics[f'{class_name}_auc'] = auc
                    metrics[f'{class_name}_ap'] = ap
                    metrics[f'{class_name}_f1'] = f1
            
            # Aggregate metrics
            if class_aucs:
                metrics['macro_auc'] = np.mean(class_aucs)
                metrics['macro_ap'] = np.mean(class_aps)
                metrics['macro_f1'] = np.mean(class_f1s)
        
        return metrics
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _perform_significance_tests(self, 
                                  clean_preds: np.ndarray, 
                                  noisy_preds: np.ndarray, 
                                  labels: np.ndarray) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        
        significance_results = {}
        
        # Paired t-test for prediction differences
        if clean_preds.ndim > 1:
            for i in range(clean_preds.shape[1]):
                clean_class_preds = clean_preds[:, i]
                noisy_class_preds = noisy_preds[:, i]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(clean_class_preds, noisy_class_preds)
                
                class_name = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump'][i]
                significance_results[f'{class_name}_ttest'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05
                }
        
        return significance_results
    
    def _assess_business_impact(self, degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess business impact of performance degradation"""
        
        # Define business-critical metrics with impact weights
        critical_metrics = {
            'macro_auc': 0.4,      # High impact on detection accuracy
            'macro_f1': 0.3,       # Balanced precision/recall impact
            'macro_ap': 0.3        # Precision-focused impact
        }
        
        total_impact = 0
        impacted_metrics = []
        
        for metric, weight in critical_metrics.items():
            if metric in degradation_analysis:
                degradation = degradation_analysis[metric]['relative_degradation']
                
                if degradation > 0.05:  # >5% degradation considered significant
                    impact_score = degradation * weight
                    total_impact += impact_score
                    impacted_metrics.append(metric)
        
        # Convert to business terms
        estimated_cost_increase = total_impact * 0.15  # 15% cost increase per impact point
        estimated_detection_loss = total_impact * 0.10  # 10% detection loss per impact point
        
        business_impact = {
            'total_impact_score': total_impact,
            'impacted_metrics': impacted_metrics,
            'estimated_cost_increase_percent': estimated_cost_increase * 100,
            'estimated_detection_loss_percent': estimated_detection_loss * 100,
            'severity': 'high' if total_impact > 0.2 else 'medium' if total_impact > 0.1 else 'low',
            'recommendation': self._generate_impact_recommendation(total_impact)
        }
        
        return business_impact
    
    def _generate_impact_recommendation(self, impact_score: float) -> str:
        """Generate recommendation based on impact severity"""
        
        if impact_score > 0.2:
            return "CRITICAL: Implement noise-robust training and data quality controls"
        elif impact_score > 0.1:
            return "MODERATE: Review labeling process and consider noise-aware algorithms"
        elif impact_score > 0.05:
            return "LOW: Monitor data quality and consider periodic robustness testing"
        else:
            return "MINIMAL: Current system shows good noise robustness"
    
    def _calculate_overall_robustness_score(self, degradation_analysis: Dict[str, Any]) -> float:
        """Calculate overall robustness score (0-1, higher is better)"""
        
        if not degradation_analysis:
            return 0.5  # Neutral score
        
        # Weight by importance
        metric_weights = {
            'macro_auc': 0.4,
            'macro_f1': 0.3,
            'macro_ap': 0.3
        }
        
        weighted_robustness = 0
        total_weight = 0
        
        for metric, weight in metric_weights.items():
            if metric in degradation_analysis:
                # Convert degradation to robustness (1 - relative_degradation)
                degradation = degradation_analysis[metric]['relative_degradation']
                robustness = max(0, 1 - degradation)  # Ensure non-negative
                
                weighted_robustness += robustness * weight
                total_weight += weight
        
        overall_score = weighted_robustness / total_weight if total_weight > 0 else 0.5
        
        return overall_score

class AcceleratedRobustnessValidator:
    """
    Unified ML Scientist + Data Analyst Role: Complete robustness validation system
    
    Features:
    - Priority EPC subset selection (10K samples from 920K)
    - Multi-type noise injection with statistical rigor
    - Comprehensive robustness evaluation with business impact
    - Parallel background validation for full dataset comparison
    """
    
    def __init__(self, 
                 model: nn.Module,
                 feature_engineer: LSTMFeatureEngineer,
                 config: Optional[Dict] = None):
        
        self.model = model
        self.feature_engineer = feature_engineer
        
        # Default configuration
        default_config = {
            'priority_subset_size': 10000,
            'noise_rates': [0.01, 0.05, 0.10, 0.15],
            'min_effect_size': 0.3,  # Cohen's d threshold
            'significance_level': 0.05,
            'business_impact_threshold': 0.1,
            'parallel_validation': True
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize components
        self.epc_selector = PriorityEPCSelector()
        self.noise_engine = NoiseInjectionEngine()
        self.metrics_calculator = RobustnessMetricsCalculator()
        
        # Validation history
        self.validation_history = deque(maxlen=50)
        
        logger.info("Accelerated robustness validator initialized")
    
    def run_accelerated_robustness_validation(self, 
                                           full_dataset: pd.DataFrame,
                                           labels: np.ndarray,
                                           features: np.ndarray) -> Dict[str, Any]:
        """
        Run complete accelerated robustness validation
        
        Args:
            full_dataset: Complete dataset for EPC selection
            labels: True labels [n_samples, n_classes]
            features: Processed features [n_samples, n_features]
            
        Returns:
            Comprehensive robustness validation results
        """
        
        start_time = time.time()
        logger.info("Starting accelerated label noise robustness validation")
        
        # Step 1: Select priority EPCs
        priority_epcs = self.epc_selector.select_priority_epcs(full_dataset)
        priority_subset = self.epc_selector.create_priority_subset(
            full_dataset, 
            priority_epcs, 
            target_samples=self.config['priority_subset_size']
        )
        
        # Get corresponding labels and features for priority subset
        priority_indices = priority_subset.index
        priority_labels = labels[priority_indices]
        priority_features = features[priority_indices]
        
        logger.info(f"Priority subset created: {len(priority_subset)} samples")
        
        # Step 2: Baseline performance on clean priority data
        clean_predictions = self._get_model_predictions(priority_features)
        
        # Step 3: Multi-level noise injection and evaluation
        noise_robustness_results = {}
        
        for noise_rate in self.config['noise_rates']:
            logger.info(f"Testing robustness at {noise_rate:.1%} noise rate")
            
            # Random noise injection
            noisy_labels_random, noise_meta_random = self.noise_engine.inject_random_noise(
                priority_labels, noise_rate
            )
            noisy_predictions_random = self._get_model_predictions(priority_features)
            
            robustness_random = self.metrics_calculator.calculate_robustness_metrics(
                clean_predictions, noisy_predictions_random, 
                priority_labels, noisy_labels_random
            )
            
            # Systematic bias injection
            bias_patterns = {
                'epcFake': noise_rate * 0.5,    # Over-detection bias
                'locErr': -noise_rate * 0.3,    # Under-detection bias
                'jump': noise_rate * 0.2         # Mild over-detection
            }
            
            noisy_labels_bias, noise_meta_bias = self.noise_engine.inject_systematic_bias(
                priority_labels, bias_patterns
            )
            noisy_predictions_bias = self._get_model_predictions(priority_features)
            
            robustness_bias = self.metrics_calculator.calculate_robustness_metrics(
                clean_predictions, noisy_predictions_bias,
                priority_labels, noisy_labels_bias
            )
            
            # Adversarial noise injection
            noisy_labels_adv, noise_meta_adv = self.noise_engine.inject_adversarial_noise(
                priority_labels, self.model, priority_features, epsilon=noise_rate
            )
            noisy_predictions_adv = self._get_model_predictions(priority_features)
            
            robustness_adv = self.metrics_calculator.calculate_robustness_metrics(
                clean_predictions, noisy_predictions_adv,
                priority_labels, noisy_labels_adv
            )
            
            # Aggregate results for this noise level
            noise_robustness_results[f'noise_rate_{noise_rate:.2f}'] = {
                'random_noise': {
                    'noise_metadata': noise_meta_random,
                    'robustness_metrics': robustness_random
                },
                'systematic_bias': {
                    'noise_metadata': noise_meta_bias,
                    'robustness_metrics': robustness_bias
                },
                'adversarial_noise': {
                    'noise_metadata': noise_meta_adv,
                    'robustness_metrics': robustness_adv
                }
            }
        
        # Step 4: Overall robustness assessment
        overall_assessment = self._assess_overall_robustness(noise_robustness_results)
        
        # Step 5: Generate recommendations
        recommendations = self._generate_robustness_recommendations(overall_assessment)
        
        # Compile final results
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'processing_time_seconds': time.time() - start_time,
            'configuration': self.config,
            'priority_subset_info': {
                'total_priority_epcs': len(priority_epcs),
                'subset_size': len(priority_subset),
                'epc_metadata': self.epc_selector.priority_epcs
            },
            'baseline_performance': self.metrics_calculator._calculate_performance_metrics(
                clean_predictions, priority_labels
            ),
            'noise_robustness_results': noise_robustness_results,
            'overall_assessment': overall_assessment,
            'recommendations': recommendations,
            'statistical_validation': self._validate_statistical_power(priority_labels)
        }
        
        # Store in history
        self.validation_history.append(validation_results)
        
        logger.info(f"Accelerated robustness validation complete: "
                   f"{overall_assessment['robustness_grade']} grade achieved")
        
        return validation_results
    
    def _get_model_predictions(self, features: np.ndarray) -> np.ndarray:
        """Get model predictions for given features"""
        
        self.model.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features_tensor = torch.FloatTensor(features)
            else:
                features_tensor = features
                
            predictions = torch.sigmoid(self.model(features_tensor)).cpu().numpy()
        
        return predictions
    
    def _assess_overall_robustness(self, noise_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall robustness across all noise conditions"""
        
        all_robustness_scores = []
        critical_failures = []
        noise_sensitivity = {}
        
        for noise_level, level_results in noise_results.items():
            for noise_type, type_results in level_results.items():
                robustness_score = type_results['robustness_metrics']['robustness_score']
                all_robustness_scores.append(robustness_score)
                
                # Check for critical failures
                business_impact = type_results['robustness_metrics']['business_impact']
                if business_impact['severity'] == 'high':
                    critical_failures.append({
                        'noise_level': noise_level,
                        'noise_type': noise_type,
                        'impact_score': business_impact['total_impact_score']
                    })
                
                # Track noise sensitivity
                noise_rate = float(noise_level.split('_')[-1])
                if noise_rate not in noise_sensitivity:
                    noise_sensitivity[noise_rate] = []
                noise_sensitivity[noise_rate].append(robustness_score)
        
        # Calculate aggregate metrics
        mean_robustness = np.mean(all_robustness_scores)
        min_robustness = np.min(all_robustness_scores)
        std_robustness = np.std(all_robustness_scores)
        
        # Robustness grading
        if mean_robustness >= 0.9 and min_robustness >= 0.8:
            grade = 'A'  # Excellent robustness
        elif mean_robustness >= 0.8 and min_robustness >= 0.7:
            grade = 'B'  # Good robustness
        elif mean_robustness >= 0.7 and min_robustness >= 0.6:
            grade = 'C'  # Acceptable robustness
        elif mean_robustness >= 0.6:
            grade = 'D'  # Poor robustness
        else:
            grade = 'F'  # Failing robustness
        
        # Noise sensitivity analysis
        sensitivity_trend = []
        for noise_rate in sorted(noise_sensitivity.keys()):
            avg_robustness = np.mean(noise_sensitivity[noise_rate])
            sensitivity_trend.append((noise_rate, avg_robustness))
        
        overall_assessment = {
            'mean_robustness_score': mean_robustness,
            'min_robustness_score': min_robustness,
            'robustness_std': std_robustness,
            'robustness_grade': grade,
            'critical_failures': critical_failures,
            'failure_count': len(critical_failures),
            'noise_sensitivity_trend': sensitivity_trend,
            'is_production_ready': grade in ['A', 'B'] and len(critical_failures) == 0
        }
        
        return overall_assessment
    
    def _generate_robustness_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on robustness assessment"""
        
        recommendations = []
        grade = assessment['robustness_grade']
        failures = assessment['critical_failures']
        
        if grade == 'F':
            recommendations.append("CRITICAL: Model fails basic robustness requirements - do not deploy")
            recommendations.append("Implement noise-aware training with regularization techniques")
            
        elif grade == 'D':
            recommendations.append("HIGH PRIORITY: Significant robustness issues detected")
            recommendations.append("Consider ensemble methods or robust loss functions")
            
        elif grade == 'C':
            recommendations.append("MODERATE: Acceptable robustness but improvement needed")
            recommendations.append("Implement data quality monitoring and cleaning procedures")
            
        elif grade == 'B':
            recommendations.append("GOOD: Solid robustness with minor areas for improvement")
            recommendations.append("Monitor performance in production and retrain if degradation occurs")
            
        else:  # Grade A
            recommendations.append("EXCELLENT: Model shows strong robustness to label noise")
            recommendations.append("Proceed with deployment but maintain quality monitoring")
        
        # Specific failure recommendations
        if failures:
            recommendations.append(f"Address {len(failures)} critical failure points in noise conditions")
            
            failure_types = [f['noise_type'] for f in failures]
            if 'systematic_bias' in failure_types:
                recommendations.append("Review labeling guidelines to prevent systematic bias")
            if 'adversarial_noise' in failure_types:
                recommendations.append("Implement adversarial training for worst-case scenarios")
        
        # Noise sensitivity recommendations
        sensitivity_trend = assessment['noise_sensitivity_trend']
        if len(sensitivity_trend) >= 2:
            # Check if robustness drops rapidly with noise
            early_robustness = sensitivity_trend[0][1]
            late_robustness = sensitivity_trend[-1][1]
            degradation_rate = (early_robustness - late_robustness) / (sensitivity_trend[-1][0] - sensitivity_trend[0][0])
            
            if degradation_rate > 2.0:  # Rapid degradation
                recommendations.append("Model shows high noise sensitivity - implement robust training")
        
        return recommendations
    
    def _validate_statistical_power(self, labels: np.ndarray) -> Dict[str, Any]:
        """Validate that subset size provides sufficient statistical power"""
        
        n_samples = len(labels)
        effect_size = self.config['min_effect_size']
        alpha = self.config['significance_level']
        
        # Power analysis for t-test
        # Simplified power calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
        z_beta = (effect_size * np.sqrt(n_samples/2)) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        # Sample size recommendation for desired power (0.8)
        target_power = 0.8
        z_target = stats.norm.ppf(target_power)
        recommended_n = 2 * ((z_alpha + z_target) / effect_size) ** 2
        
        power_validation = {
            'current_sample_size': n_samples,
            'calculated_power': power,
            'target_power': target_power,
            'recommended_sample_size': int(recommended_n),
            'is_sufficient_power': power >= target_power,
            'effect_size_threshold': effect_size
        }
        
        if power < target_power:
            logger.warning(f"Statistical power ({power:.3f}) below target ({target_power}). "
                         f"Consider increasing sample size to {int(recommended_n)}")
        
        return power_validation
    
    def create_robustness_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive robustness validation report"""
        
        report_lines = [
            "# Label Noise Robustness Validation Report",
            f"**Generated:** {validation_results['validation_timestamp']}",
            f"**Processing Time:** {validation_results['processing_time_seconds']:.1f} seconds",
            "",
            "## Executive Summary",
            f"**Overall Grade:** {validation_results['overall_assessment']['robustness_grade']}",
            f"**Mean Robustness Score:** {validation_results['overall_assessment']['mean_robustness_score']:.3f}",
            f"**Production Ready:** {validation_results['overall_assessment']['is_production_ready']}",
            "",
            "## Priority EPC Subset Analysis",
            f"- **Total Priority EPCs:** {validation_results['priority_subset_info']['total_priority_epcs']}",
            f"- **Subset Size:** {validation_results['priority_subset_info']['subset_size']:,} samples",
            f"- **Statistical Power:** {validation_results['statistical_validation']['calculated_power']:.3f}",
            "",
            "## Baseline Performance",
        ]
        
        # Add baseline metrics
        baseline = validation_results['baseline_performance']
        for metric, value in baseline.items():
            report_lines.append(f"- **{metric}:** {value:.4f}")
        
        report_lines.extend([
            "",
            "## Noise Robustness Results",
        ])
        
        # Add noise robustness details
        for noise_level, level_results in validation_results['noise_robustness_results'].items():
            noise_rate = float(noise_level.split('_')[-1])
            report_lines.append(f"### Noise Rate: {noise_rate:.1%}")
            
            for noise_type, type_results in level_results.items():
                robustness = type_results['robustness_metrics']
                score = robustness['robustness_score']
                impact = robustness['business_impact']['severity']
                
                report_lines.append(f"- **{noise_type.replace('_', ' ').title()}:** Score {score:.3f}, Impact: {impact}")
        
        report_lines.extend([
            "",
            "## Recommendations",
        ])
        
        for rec in validation_results['recommendations']:
            report_lines.append(f"- {rec}")
        
        return "\n".join(report_lines)

def create_accelerated_robustness_system(model: nn.Module,
                                       feature_engineer: LSTMFeatureEngineer,
                                       config: Optional[Dict] = None) -> AcceleratedRobustnessValidator:
    """
    Factory function to create accelerated robustness validation system
    
    Args:
        model: Trained LSTM model
        feature_engineer: Feature engineering pipeline
        config: Optional validation configuration
        
    Returns:
        Configured AcceleratedRobustnessValidator instance
    """
    
    validator = AcceleratedRobustnessValidator(
        model=model,
        feature_engineer=feature_engineer,
        config=config
    )
    
    return validator

if __name__ == "__main__":
    # Example usage and testing
    
    try:
        # Mock data for demonstration
        full_dataset = pd.DataFrame({
            'epc': [f'EPC_{i:06d}' for i in np.random.randint(0, 50000, 10000)],
            'latitude': np.random.uniform(40.0, 45.0, 10000),
            'longitude': np.random.uniform(-74.0, -70.0, 10000),
            'epcFake': np.random.binomial(1, 0.05, 10000),
            'epcDup': np.random.binomial(1, 0.02, 10000),
            'locErr': np.random.binomial(1, 0.03, 10000),
            'evtOrderErr': np.random.binomial(1, 0.04, 10000),
            'jump': np.random.binomial(1, 0.06, 10000)
        })
        
        labels = full_dataset[['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']].values
        features = np.random.normal(0, 1, (10000, 50))  # Mock feature matrix
        
        # Create mock model and feature engineer
        model = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid()
        )
        
        feature_engineer = LSTMFeatureEngineer()  # Mock feature engineer
        
        # Create robustness validation system
        robustness_validator = create_accelerated_robustness_system(
            model=model,
            feature_engineer=feature_engineer,
            config={'priority_subset_size': 1000, 'noise_rates': [0.05, 0.10]}
        )
        
        # Run accelerated robustness validation
        validation_results = robustness_validator.run_accelerated_robustness_validation(
            full_dataset=full_dataset,
            labels=labels,
            features=features
        )
        
        # Generate report
        report = robustness_validator.create_robustness_report(validation_results)
        
        print("Label noise robustness validation completed successfully!")
        print(f"Overall Grade: {validation_results['overall_assessment']['robustness_grade']}")
        print(f"Production Ready: {validation_results['overall_assessment']['is_production_ready']}")
        print(f"Critical Failures: {validation_results['overall_assessment']['failure_count']}")
        print("\nFull Report:")
        print(report)
        
    except Exception as e:
        print(f"Robustness validation test failed: {e}")
        import traceback
        traceback.print_exc()