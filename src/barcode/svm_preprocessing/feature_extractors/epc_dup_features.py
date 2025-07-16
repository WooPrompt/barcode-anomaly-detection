"""
EPC Duplicate Feature Extractor for SVM Training
Reuses logic from multi_anomaly_detector.py lines 163-191
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime


class EPCDupFeatureExtractor:
    """EPC duplicate detection features - Fixed 8 dimensions"""
    
    FEATURE_DIMENSIONS = 8
    
    def __init__(self):
        self.feature_names = [
            'event_count', 'unique_locations_count', 'unique_event_types_count',
            'time_span_hours', 'avg_time_between_events', 'max_time_gap',
            'location_repetition_ratio', 'event_type_repetition_ratio'
        ]
    
    def extract_features(self, epc_group: pd.DataFrame) -> List[float]:
        """
        Extract duplicate detection features for an EPC sequence
        REUSE: Logic adapted from lines 163-191 (calculate_duplicate_score)
        
        Returns fixed 8-dimensional feature vector
        """
        if epc_group.empty:
            return [0.0] * self.FEATURE_DIMENSIONS
        
        features = []
        
        # Sort by event time
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        # Basic counting features (3 dimensions)
        event_count = len(epc_group)
        unique_locations = epc_group['reader_location'].nunique()
        unique_event_types = epc_group['event_type'].nunique()
        
        features.extend([
            float(event_count),
            float(unique_locations),
            float(unique_event_types)
        ])
        
        # Time-based features (3 dimensions)
        if event_count > 1:
            # Convert event_time to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(epc_group['event_time']):
                epc_group['event_time'] = pd.to_datetime(epc_group['event_time'])
            
            time_diffs = epc_group['event_time'].diff().dt.total_seconds() / 3600  # hours
            time_diffs = time_diffs.dropna()
            
            time_span_hours = (epc_group['event_time'].max() - epc_group['event_time'].min()).total_seconds() / 3600
            avg_time_between_events = time_diffs.mean() if len(time_diffs) > 0 else 0.0
            max_time_gap = time_diffs.max() if len(time_diffs) > 0 else 0.0
            
            features.extend([
                float(time_span_hours),
                float(avg_time_between_events),
                float(max_time_gap)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Repetition ratio features (2 dimensions)
        location_repetition_ratio = 1.0 - (unique_locations / event_count) if event_count > 0 else 0.0
        event_type_repetition_ratio = 1.0 - (unique_event_types / event_count) if event_count > 0 else 0.0
        
        features.extend([
            float(location_repetition_ratio),
            float(event_type_repetition_ratio)
        ])
        
        # Normalize time features to reasonable ranges
        features[3] = min(features[3] / 168.0, 1.0)  # Normalize by week (168 hours)
        features[4] = min(features[4] / 24.0, 1.0)   # Normalize by day (24 hours)
        features[5] = min(features[5] / 48.0, 1.0)   # Normalize by 2 days (48 hours)
        
        # Ensure exactly 8 dimensions
        assert len(features) == self.FEATURE_DIMENSIONS, f"Expected {self.FEATURE_DIMENSIONS} features, got {len(features)}"
        
        return features
    
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
    
    def calculate_duplicate_indicators(self, epc_group: pd.DataFrame) -> Dict[str, Any]:
        """Calculate additional duplicate indicators for analysis"""
        if epc_group.empty:
            return {}
        
        # Sort by event time
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        # Check for exact duplicates
        duplicate_rows = epc_group.duplicated(subset=['reader_location', 'event_type'], keep=False)
        exact_duplicates = duplicate_rows.sum()
        
        # Check for temporal anomalies (same location within short time)
        temporal_duplicates = 0
        if len(epc_group) > 1:
            if not pd.api.types.is_datetime64_any_dtype(epc_group['event_time']):
                epc_group['event_time'] = pd.to_datetime(epc_group['event_time'])
            
            for i in range(1, len(epc_group)):
                time_diff = (epc_group.iloc[i]['event_time'] - epc_group.iloc[i-1]['event_time']).total_seconds()
                if (time_diff < 60 and  # Within 1 minute
                    epc_group.iloc[i]['reader_location'] == epc_group.iloc[i-1]['reader_location']):
                    temporal_duplicates += 1
        
        return {
            'exact_duplicates': exact_duplicates,
            'temporal_duplicates': temporal_duplicates,
            'total_events': len(epc_group),
            'duplicate_ratio': exact_duplicates / len(epc_group) if len(epc_group) > 0 else 0.0
        }