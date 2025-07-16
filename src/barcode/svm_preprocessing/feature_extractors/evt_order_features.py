"""
Event Order Feature Extractor for SVM Training
Reuses logic from multi_anomaly_detector.py lines 216-247, 249-279
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_anomaly_detector import classify_event_type
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class EventOrderFeatureExtractor:
    """Event order violation detection features - Fixed 12 dimensions"""
    
    FEATURE_DIMENSIONS = 12
    
    def __init__(self):
        self.feature_names = [
            'sequence_length', 'unique_event_types', 'transition_count',
            'backward_transitions', 'temporal_disorder_count', 'max_time_gap_hours',
            'production_ratio', 'logistics_ratio', 'retail_ratio',
            'event_type_entropy', 'sequence_regularity', 'violation_density'
        ]
        
        # Expected event order (from supply chain flow)
        self.expected_order = {
            'production': 1,
            'logistics': 2, 
            'retail': 3
        }
    
    def extract_features(self, epc_group: pd.DataFrame) -> List[float]:
        """
        Extract event order features for an EPC sequence
        REUSE: Logic adapted from lines 216-247 (classify_event_type), 249-279 (calculate_event_order_score)
        
        Returns fixed 12-dimensional feature vector
        """
        if epc_group.empty:
            return [0.0] * self.FEATURE_DIMENSIONS
        
        features = []
        
        # Sort by event time
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        # Basic sequence features (3 dimensions)
        sequence_length = len(epc_group)
        unique_event_types = epc_group['event_type'].nunique()
        
        # REUSE: classify_event_type logic (lines 216-247)
        event_categories = []
        for _, row in epc_group.iterrows():
            category = classify_event_type(row['event_type'])
            event_categories.append(category)
        
        transition_count = len(set(zip(event_categories[:-1], event_categories[1:]))) if len(event_categories) > 1 else 0
        
        features.extend([
            float(sequence_length),
            float(unique_event_types),
            float(transition_count)
        ])
        
        # Order violation features (3 dimensions)
        backward_transitions = 0
        temporal_disorder_count = 0
        max_time_gap_hours = 0.0
        
        if len(epc_group) > 1:
            # Convert event_time to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(epc_group['event_time']):
                epc_group['event_time'] = pd.to_datetime(epc_group['event_time'])
            
            for i in range(1, len(event_categories)):
                curr_category = event_categories[i]
                prev_category = event_categories[i-1]
                
                # Check for backward transitions in supply chain
                if (curr_category in self.expected_order and 
                    prev_category in self.expected_order):
                    if self.expected_order[curr_category] < self.expected_order[prev_category]:
                        backward_transitions += 1
                
                # Check for temporal disorder
                time_diff = (epc_group.iloc[i]['event_time'] - epc_group.iloc[i-1]['event_time']).total_seconds()
                if time_diff < 0:  # Time going backward
                    temporal_disorder_count += 1
                
                # Track maximum time gap
                time_gap_hours = abs(time_diff) / 3600
                max_time_gap_hours = max(max_time_gap_hours, time_gap_hours)
        
        features.extend([
            float(backward_transitions),
            float(temporal_disorder_count),
            float(min(max_time_gap_hours / 168.0, 1.0))  # Normalize by week
        ])
        
        # Category distribution features (3 dimensions)
        category_counts = {}
        for category in event_categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        total_events = len(event_categories)
        production_ratio = category_counts.get('production', 0) / total_events if total_events > 0 else 0.0
        logistics_ratio = category_counts.get('logistics', 0) / total_events if total_events > 0 else 0.0
        retail_ratio = category_counts.get('retail', 0) / total_events if total_events > 0 else 0.0
        
        features.extend([
            float(production_ratio),
            float(logistics_ratio),
            float(retail_ratio)
        ])
        
        # Advanced pattern features (3 dimensions)
        event_type_entropy = self._calculate_entropy(event_categories)
        sequence_regularity = self._calculate_regularity(event_categories)
        violation_density = (backward_transitions + temporal_disorder_count) / sequence_length if sequence_length > 0 else 0.0
        
        features.extend([
            float(event_type_entropy),
            float(sequence_regularity),
            float(violation_density)
        ])
        
        # Ensure exactly 12 dimensions
        assert len(features) == self.FEATURE_DIMENSIONS, f"Expected {self.FEATURE_DIMENSIONS} features, got {len(features)}"
        
        return features
    
    def _calculate_entropy(self, sequence: List[str]) -> float:
        """Calculate Shannon entropy of event sequence"""
        if not sequence:
            return 0.0
        
        from collections import Counter
        counts = Counter(sequence)
        total = len(sequence)
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _calculate_regularity(self, sequence: List[str]) -> float:
        """Calculate sequence regularity (0 = irregular, 1 = perfectly regular)"""
        if len(sequence) < 2:
            return 1.0
        
        # Check for repeating patterns
        pattern_scores = []
        
        for pattern_length in range(1, min(5, len(sequence) // 2 + 1)):
            pattern_matches = 0
            for i in range(len(sequence) - pattern_length):
                pattern = sequence[i:i+pattern_length]
                if i + 2*pattern_length <= len(sequence):
                    next_pattern = sequence[i+pattern_length:i+2*pattern_length]
                    if pattern == next_pattern:
                        pattern_matches += 1
            
            if len(sequence) - pattern_length > 0:
                pattern_score = pattern_matches / (len(sequence) - pattern_length)
                pattern_scores.append(pattern_score)
        
        return max(pattern_scores) if pattern_scores else 0.0
    
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
    
    def analyze_order_violations(self, epc_group: pd.DataFrame) -> Dict[str, Any]:
        """Detailed analysis of order violations"""
        if epc_group.empty:
            return {}
        
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        violations = []
        for i in range(1, len(epc_group)):
            curr_event = classify_event_type(epc_group.iloc[i]['event_type'])
            prev_event = classify_event_type(epc_group.iloc[i-1]['event_type'])
            
            if (curr_event in self.expected_order and 
                prev_event in self.expected_order and
                self.expected_order[curr_event] < self.expected_order[prev_event]):
                
                violations.append({
                    'position': i,
                    'from_category': prev_event,
                    'to_category': curr_event,
                    'event_type': epc_group.iloc[i]['event_type'],
                    'location': epc_group.iloc[i]['reader_location']
                })
        
        return {
            'violations': violations,
            'violation_count': len(violations),
            'violation_positions': [v['position'] for v in violations],
            'most_common_violation': self._get_most_common_violation_type(violations)
        }
    
    def _get_most_common_violation_type(self, violations: List[Dict]) -> str:
        """Find the most common type of order violation"""
        if not violations:
            return "none"
        
        violation_types = [f"{v['from_category']}->{v['to_category']}" for v in violations]
        from collections import Counter
        most_common = Counter(violation_types).most_common(1)
        
        return most_common[0][0] if most_common else "unknown"