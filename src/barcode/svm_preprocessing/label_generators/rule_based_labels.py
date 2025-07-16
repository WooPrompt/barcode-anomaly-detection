"""
Rule-based Label Generator for SVM Training
Reuses scoring functions from multi_anomaly_detector.py lines 98-161, 163-191, 193-214, 249-279, 314-343
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_anomaly_detector import (
    calculate_epc_fake_score, calculate_duplicate_score, calculate_time_jump_score,
    calculate_event_order_score, calculate_location_error_score
)
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class RuleBasedLabelGenerator:
    """Generate training labels using existing rule-based scoring functions"""
    
    def __init__(self, thresholds: Dict[str, int] = None):
        # Default thresholds for binary classification
        self.thresholds = thresholds or {
            'epcFake': 50,
            'epcDup': 50, 
            'evtOrderErr': 50,
            'locErr': 50,
            'jump': 50
        }
        
        # Map anomaly types to their scoring functions
        self.scoring_functions = {
            'epcFake': self._score_epc_fake,
            'epcDup': self._score_epc_dup,
            'evtOrderErr': self._score_event_order,
            'locErr': self._score_location_error,
            'jump': self._score_time_jump
        }
    
    def generate_labels(self, epc_groups: Dict[str, pd.DataFrame], 
                       anomaly_type: str) -> Tuple[List[int], List[float], List[str]]:
        """
        Generate binary labels and confidence scores for an anomaly type
        
        Returns:
            Tuple of (labels, scores, epc_codes)
        """
        if anomaly_type not in self.scoring_functions:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        labels = []
        scores = []
        epc_codes = []
        
        scoring_func = self.scoring_functions[anomaly_type]
        threshold = self.thresholds[anomaly_type]
        
        for epc_code, epc_group in epc_groups.items():
            # Get score using appropriate function
            score = scoring_func(epc_code, epc_group)
            
            # Convert to binary label
            label = 1 if score >= threshold else 0
            
            labels.append(label)
            scores.append(score)
            epc_codes.append(epc_code)
        
        return labels, scores, epc_codes
    
    def generate_all_labels(self, epc_groups: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Generate labels for all anomaly types"""
        all_labels = {}
        
        for anomaly_type in self.scoring_functions.keys():
            labels, scores, epc_codes = self.generate_labels(epc_groups, anomaly_type)
            
            all_labels[anomaly_type] = {
                'labels': labels,
                'scores': scores,
                'epc_codes': epc_codes,
                'label_distribution': self._analyze_label_distribution(labels, scores),
                'threshold_used': self.thresholds[anomaly_type]
            }
        
        return all_labels
    
    def _score_epc_fake(self, epc_code: str, epc_group: pd.DataFrame) -> float:
        """REUSE: lines 98-161 (calculate_epc_fake_score)"""
        return float(calculate_epc_fake_score(epc_code))
    
    def _score_epc_dup(self, epc_code: str, epc_group: pd.DataFrame) -> float:
        """REUSE: lines 163-191 (calculate_duplicate_score)"""
        return float(calculate_duplicate_score(epc_group))
    
    def _score_event_order(self, epc_code: str, epc_group: pd.DataFrame) -> float:
        """REUSE: lines 249-279 (calculate_event_order_score)"""
        return float(calculate_event_order_score(epc_group))
    
    def _score_location_error(self, epc_code: str, epc_group: pd.DataFrame) -> float:
        """REUSE: lines 314-343 (calculate_location_error_score)"""
        return float(calculate_location_error_score(epc_group))
    
    def _score_time_jump(self, epc_code: str, epc_group: pd.DataFrame) -> float:
        """REUSE: lines 193-214 (calculate_time_jump_score)"""
        return float(calculate_time_jump_score(epc_group))
    
    def _analyze_label_distribution(self, labels: List[int], scores: List[float]) -> Dict[str, Any]:
        """Analyze distribution of generated labels"""
        labels_array = np.array(labels)
        scores_array = np.array(scores)
        
        unique, counts = np.unique(labels_array, return_counts=True)
        distribution = dict(zip(unique.astype(int), counts.astype(int)))
        
        positive_count = distribution.get(1, 0)
        negative_count = distribution.get(0, 0)
        total_count = len(labels)
        
        # Score statistics for positive and negative cases
        positive_scores = scores_array[labels_array == 1] if positive_count > 0 else []
        negative_scores = scores_array[labels_array == 0] if negative_count > 0 else []
        
        analysis = {
            'total_samples': total_count,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_count / total_count if total_count > 0 else 0.0,
            'imbalance_ratio': negative_count / positive_count if positive_count > 0 else float('inf'),
            'score_stats': {
                'overall': {
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'min': float(np.min(scores_array)),
                    'max': float(np.max(scores_array)),
                    'median': float(np.median(scores_array))
                }
            }
        }
        
        if len(positive_scores) > 0:
            analysis['score_stats']['positive'] = {
                'mean': float(np.mean(positive_scores)),
                'std': float(np.std(positive_scores)),
                'min': float(np.min(positive_scores)),
                'max': float(np.max(positive_scores))
            }
        
        if len(negative_scores) > 0:
            analysis['score_stats']['negative'] = {
                'mean': float(np.mean(negative_scores)),
                'std': float(np.std(negative_scores)),
                'min': float(np.min(negative_scores)),
                'max': float(np.max(negative_scores))
            }
        
        return analysis
    
    def optimize_thresholds(self, epc_groups: Dict[str, pd.DataFrame], 
                          target_positive_ratio: float = 0.1) -> Dict[str, int]:
        """Optimize thresholds to achieve target positive ratio"""
        optimized_thresholds = {}
        
        for anomaly_type in self.scoring_functions.keys():
            scores = []
            scoring_func = self.scoring_functions[anomaly_type]
            
            # Collect all scores for this anomaly type
            for epc_code, epc_group in epc_groups.items():
                score = scoring_func(epc_code, epc_group)
                scores.append(score)
            
            scores_array = np.array(scores)
            
            # Find threshold that gives closest to target positive ratio
            threshold_percentile = (1.0 - target_positive_ratio) * 100
            optimal_threshold = np.percentile(scores_array, threshold_percentile)
            
            # Round to nearest integer and ensure reasonable bounds
            optimal_threshold = max(0, min(100, int(round(optimal_threshold))))
            
            optimized_thresholds[anomaly_type] = optimal_threshold
        
        return optimized_thresholds
    
    def analyze_borderline_cases(self, epc_groups: Dict[str, pd.DataFrame], 
                                anomaly_type: str, window: int = 5) -> List[Dict[str, Any]]:
        """Find borderline cases near the threshold for analysis"""
        if anomaly_type not in self.scoring_functions:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        scoring_func = self.scoring_functions[anomaly_type]
        threshold = self.thresholds[anomaly_type]
        
        borderline_cases = []
        
        for epc_code, epc_group in epc_groups.items():
            score = scoring_func(epc_code, epc_group)
            
            # Check if score is within window of threshold
            if abs(score - threshold) <= window:
                borderline_cases.append({
                    'epc_code': epc_code,
                    'score': score,
                    'distance_from_threshold': abs(score - threshold),
                    'predicted_label': 1 if score >= threshold else 0,
                    'confidence': 'low'  # All borderline cases have low confidence
                })
        
        # Sort by distance from threshold (most uncertain first)
        borderline_cases.sort(key=lambda x: x['distance_from_threshold'])
        
        return borderline_cases
    
    def update_thresholds(self, new_thresholds: Dict[str, int]):
        """Update thresholds for label generation"""
        self.thresholds.update(new_thresholds)
    
    def get_threshold_analysis(self, epc_groups: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Analyze the effect of different thresholds"""
        analysis = {}
        
        for anomaly_type in self.scoring_functions.keys():
            scores = []
            scoring_func = self.scoring_functions[anomaly_type]
            
            for epc_code, epc_group in epc_groups.items():
                score = scoring_func(epc_code, epc_group)
                scores.append(score)
            
            scores_array = np.array(scores)
            
            # Test different thresholds
            test_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            threshold_results = {}
            
            for threshold in test_thresholds:
                labels = (scores_array >= threshold).astype(int)
                positive_ratio = np.mean(labels)
                
                threshold_results[threshold] = {
                    'positive_ratio': float(positive_ratio),
                    'positive_count': int(np.sum(labels)),
                    'negative_count': int(len(labels) - np.sum(labels))
                }
            
            analysis[anomaly_type] = {
                'score_distribution': {
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'percentiles': {
                        '10': float(np.percentile(scores_array, 10)),
                        '25': float(np.percentile(scores_array, 25)),
                        '50': float(np.percentile(scores_array, 50)),
                        '75': float(np.percentile(scores_array, 75)),
                        '90': float(np.percentile(scores_array, 90))
                    }
                },
                'threshold_analysis': threshold_results,
                'current_threshold': self.thresholds[anomaly_type]
            }
        
        return analysis