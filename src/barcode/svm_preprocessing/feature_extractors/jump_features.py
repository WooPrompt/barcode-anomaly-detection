"""
Time Jump Feature Extractor for SVM Training
Reuses logic from multi_anomaly_detector.py lines 193-214
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta


class JumpFeatureExtractor:
    """Time jump/impossible travel detection features - Fixed 10 dimensions"""
    
    FEATURE_DIMENSIONS = 10
    
    def __init__(self):
        self.feature_names = [
            'sequence_length', 'total_time_span_hours', 'avg_time_between_events',
            'max_time_gap_hours', 'min_time_gap_seconds', 'time_gap_variance',
            'impossible_jumps_count', 'negative_time_count', 'zero_time_gaps',
            'time_regularity_score'
        ]
        
        # Thresholds for impossible travel (in hours)
        self.max_reasonable_gap = 168  # 1 week
        self.min_reasonable_gap = 0.001  # ~3.6 seconds
    
    def extract_features(self, epc_group: pd.DataFrame) -> List[float]:
        """
        Extract time jump features for an EPC sequence
        REUSE: Logic adapted from lines 193-214 (calculate_time_jump_score)
        
        Returns fixed 10-dimensional feature vector
        """
        if epc_group.empty:
            return [0.0] * self.FEATURE_DIMENSIONS
        
        features = []
        
        # Sort by event time
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        # Convert event_time to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(epc_group['event_time']):
            epc_group['event_time'] = pd.to_datetime(epc_group['event_time'])
        
        # Basic temporal features (6 dimensions)
        sequence_length = len(epc_group)
        
        if sequence_length > 1:
            # Calculate time differences
            time_diffs_seconds = epc_group['event_time'].diff().dt.total_seconds().dropna()
            time_diffs_hours = time_diffs_seconds / 3600
            
            total_time_span = (epc_group['event_time'].max() - epc_group['event_time'].min()).total_seconds() / 3600
            avg_time_between_events = time_diffs_hours.mean()
            max_time_gap_hours = time_diffs_hours.max()
            min_time_gap_seconds = time_diffs_seconds.min()
            time_gap_variance = time_diffs_hours.var()
            
            # Normalize features
            total_time_span_normalized = min(total_time_span / (24 * 7), 1.0)  # Normalize by week
            avg_time_normalized = min(avg_time_between_events / 24, 1.0)  # Normalize by day
            max_gap_normalized = min(max_time_gap_hours / self.max_reasonable_gap, 1.0)
            min_gap_normalized = max(min_time_gap_seconds / 3600, 0.0)  # Convert to hours
            variance_normalized = min(time_gap_variance / (24**2), 1.0) if not np.isnan(time_gap_variance) else 0.0
            
            features.extend([
                float(sequence_length),
                float(total_time_span_normalized),
                float(avg_time_normalized),
                float(max_gap_normalized),
                float(min_gap_normalized),
                float(variance_normalized)
            ])
            
            # Anomaly detection features (4 dimensions)
            impossible_jumps = sum(1 for gap in time_diffs_hours if gap > self.max_reasonable_gap)
            negative_time = sum(1 for gap in time_diffs_seconds if gap < 0)
            zero_time_gaps = sum(1 for gap in time_diffs_seconds if gap == 0)
            
            # Calculate time regularity score
            time_regularity = self._calculate_time_regularity(time_diffs_hours.tolist())
            
            features.extend([
                float(impossible_jumps),
                float(negative_time),
                float(zero_time_gaps),
                float(time_regularity)
            ])
            
        else:
            # Single event case
            features.extend([
                float(sequence_length),  # 1
                0.0,  # total_time_span_hours
                0.0,  # avg_time_between_events
                0.0,  # max_time_gap_hours
                0.0,  # min_time_gap_seconds
                0.0,  # time_gap_variance
                0.0,  # impossible_jumps_count
                0.0,  # negative_time_count
                0.0,  # zero_time_gaps
                1.0   # time_regularity_score (perfect for single event)
            ])
        
        # Ensure exactly 10 dimensions
        assert len(features) == self.FEATURE_DIMENSIONS, f"Expected {self.FEATURE_DIMENSIONS} features, got {len(features)}"
        
        return features
    
    def _calculate_time_regularity(self, time_gaps: List[float]) -> float:
        """Calculate regularity of time gaps between events"""
        if len(time_gaps) < 2:
            return 1.0
        
        # Remove outliers for regularity calculation
        q25, q75 = np.percentile(time_gaps, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        filtered_gaps = [gap for gap in time_gaps if lower_bound <= gap <= upper_bound]
        
        if len(filtered_gaps) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower = more regular)
        mean_gap = np.mean(filtered_gaps)
        std_gap = np.std(filtered_gaps)
        
        if mean_gap == 0:
            return 0.0
        
        cv = std_gap / mean_gap
        # Convert to regularity score (0 = irregular, 1 = perfectly regular)
        regularity = 1.0 / (1.0 + cv)
        
        return regularity
    
    def extract_batch_features(self, epc_groups: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Extract features for multiple EPC groups"""
        feature_matrix = []
        
        for epc_code, epc_group in epc_groups.items():
            features = self.extract_features(epc_group)
            feature_matrix.append(features)
        
        return np.array(feature_matrix)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        return self.feature_names.copy()
    
    def analyze_time_anomalies(self, epc_group: pd.DataFrame) -> Dict[str, Any]:
        """Detailed analysis of time-based anomalies"""
        if epc_group.empty or len(epc_group) < 2:
            return {'anomalies': [], 'summary': {'total_gaps': 0}}
        
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        # Convert event_time to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(epc_group['event_time']):
            epc_group['event_time'] = pd.to_datetime(epc_group['event_time'])
        
        anomalies = []
        time_diffs_seconds = epc_group['event_time'].diff().dt.total_seconds().dropna()
        
        for i, time_diff in enumerate(time_diffs_seconds, 1): time_diff_hours = time_diff / 3600
