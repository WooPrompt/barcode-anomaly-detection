"""
Location Error Feature Extractor for SVM Training
Reuses logic from multi_anomaly_detector.py lines 293-312, 314-343
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_anomaly_detector import get_location_hierarchy_level
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class LocationErrorFeatureExtractor:
    """Location hierarchy violation detection features - Fixed 15 dimensions"""
    
    FEATURE_DIMENSIONS = 15
    
    def __init__(self):
        self.feature_names = [
            'sequence_length', 'unique_locations', 'hierarchy_violations',
            'unknown_locations', 'reverse_transitions', 'max_hierarchy_jump',
            'level0_ratio', 'level1_ratio', 'level2_ratio', 'level3_ratio',
            'location_entropy', 'hierarchy_consistency', 'geographic_scatter',
            'transition_regularity', 'location_anomaly_density'
        ]
        
        # Location hierarchy levels (from supply chain)
        self.hierarchy_order = {
            0: 'country',
            1: 'region', 
            2: 'facility',
            3: 'local'
        }
    
    def extract_features(self, epc_group: pd.DataFrame) -> List[float]:
        """
        Extract location hierarchy features for an EPC sequence
        REUSE: Logic adapted from lines 293-312 (get_location_hierarchy_level), 314-343 (calculate_location_error_score)
        
        Returns fixed 15-dimensional feature vector
        """
        if epc_group.empty:
            return [0.0] * self.FEATURE_DIMENSIONS
        
        features = []
        
        # Sort by event time
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        # Basic location features (5 dimensions)
        sequence_length = len(epc_group)
        unique_locations = epc_group['reader_location'].nunique()
        
        # REUSE: get_location_hierarchy_level logic (lines 293-312)
        hierarchy_levels = []
        unknown_locations = 0
        
        for _, row in epc_group.iterrows():
            level = get_location_hierarchy_level(row['reader_location'])
            if level == -1:  # Unknown location
                unknown_locations += 1
                hierarchy_levels.append(-1)
            else:
                hierarchy_levels.append(level)
        
        # Count hierarchy violations and reverse transitions
        hierarchy_violations = 0
        reverse_transitions = 0
        max_hierarchy_jump = 0
        
        for i in range(1, len(hierarchy_levels)):
            curr_level = hierarchy_levels[i]
            prev_level = hierarchy_levels[i-1]
            
            if curr_level != -1 and prev_level != -1:
                level_diff = curr_level - prev_level
                
                # Track maximum hierarchy jump
                max_hierarchy_jump = max(max_hierarchy_jump, abs(level_diff))
                
                # Check for reverse transitions (higher to lower level inappropriately)
                if level_diff < -1:  # Skipping levels going backward
                    reverse_transitions += 1
                    hierarchy_violations += 1
                elif level_diff > 2:  # Skipping too many levels forward
                    hierarchy_violations += 1
        
        features.extend([
            float(sequence_length),
            float(unique_locations),
            float(hierarchy_violations),
            float(unknown_locations),
            float(reverse_transitions)
        ])
        
        # Hierarchy jump feature (1 dimension)
        max_hierarchy_jump_normalized = min(max_hierarchy_jump / 3.0, 1.0)  # Normalize by max possible jump
        features.append(float(max_hierarchy_jump_normalized))
        
        # Level distribution features (4 dimensions)
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        valid_levels = [level for level in hierarchy_levels if level != -1]
        
        for level in valid_levels:
            if level in level_counts:
                level_counts[level] += 1
        
        total_valid = len(valid_levels)
        if total_valid > 0:
            level0_ratio = level_counts[0] / total_valid
            level1_ratio = level_counts[1] / total_valid
            level2_ratio = level_counts[2] / total_valid
            level3_ratio = level_counts[3] / total_valid
        else:
            level0_ratio = level1_ratio = level2_ratio = level3_ratio = 0.0
        
        features.extend([
            float(level0_ratio),
            float(level1_ratio),
            float(level2_ratio),
            float(level3_ratio)
        ])
        
        # Advanced pattern features (5 dimensions)
        location_entropy = self._calculate_location_entropy(epc_group['reader_location'].tolist())
        hierarchy_consistency = self._calculate_hierarchy_consistency(hierarchy_levels)
        geographic_scatter = self._calculate_geographic_scatter(epc_group)
        transition_regularity = self._calculate_transition_regularity(hierarchy_levels)
        location_anomaly_density = (hierarchy_violations + unknown_locations) / sequence_length if sequence_length > 0 else 0.0
        
        features.extend([
            float(location_entropy),
            float(hierarchy_consistency),
            float(geographic_scatter),
            float(transition_regularity),
            float(location_anomaly_density)
        ])
        
        # Ensure exactly 15 dimensions
        assert len(features) == self.FEATURE_DIMENSIONS, f"Expected {self.FEATURE_DIMENSIONS} features, got {len(features)}"
        
        return features
    
    def _calculate_location_entropy(self, locations: List[str]) -> float:
        """Calculate Shannon entropy of location sequence"""
        if not locations:
            return 0.0
        
        from collections import Counter
        counts = Counter(locations)
        total = len(locations)
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_hierarchy_consistency(self, hierarchy_levels: List[int]) -> float:
        """Calculate consistency of hierarchy progression"""
        if len(hierarchy_levels) < 2:
            return 1.0
        
        valid_levels = [level for level in hierarchy_levels if level != -1]
        if len(valid_levels) < 2:
            return 0.0
        
        # Check for smooth transitions (no big jumps)
        smooth_transitions = 0
        total_transitions = len(valid_levels) - 1
        
        for i in range(1, len(valid_levels)):
            level_diff = abs(valid_levels[i] - valid_levels[i-1])
            if level_diff <= 1:  # Smooth transition
                smooth_transitions += 1
        
        return smooth_transitions / total_transitions if total_transitions > 0 else 1.0
    
    def _calculate_geographic_scatter(self, epc_group: pd.DataFrame) -> float:
        """Calculate geographic scatter of locations"""
        unique_locations = epc_group['reader_location'].nunique()
        total_events = len(epc_group)
        
        # Higher scatter = more unique locations relative to events
        scatter = unique_locations / total_events if total_events > 0 else 0.0
        return min(scatter, 1.0)
    
    def _calculate_transition_regularity(self, hierarchy_levels: List[int]) -> float:
        """Calculate regularity of hierarchy transitions"""
        if len(hierarchy_levels) < 3:
            return 1.0
        
        valid_levels = [level for level in hierarchy_levels if level != -1]
        if len(valid_levels) < 3:
            return 0.0
        
        # Look for consistent transition patterns
        transitions = []
        for i in range(1, len(valid_levels)):
            transition = valid_levels[i] - valid_levels[i-1]
            transitions.append(transition)
        
        # Calculate variance of transitions (lower = more regular)
        if len(transitions) > 1:
            transition_variance = np.var(transitions)
            # Convert to regularity score (higher = more regular)
            regularity = 1.0 / (1.0 + transition_variance)
        else:
            regularity = 1.0
        
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
    
    def analyze_location_violations(self, epc_group: pd.DataFrame) -> Dict[str, Any]:
        """Detailed analysis of location hierarchy violations"""
        if epc_group.empty:
            return {}
        
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        violations = []
        location_path = []
        
        for i, row in epc_group.iterrows():
            location = row['reader_location']
            level = get_location_hierarchy_level(location)
            
            location_info = {
                'position': i,
                'location': location,
                'hierarchy_level': level,
                'is_unknown': level == -1
            }
            
            if i > 0:
                prev_level = location_path[-1]['hierarchy_level']
                if level != -1 and prev_level != -1:
                    level_jump = level - prev_level
                    if abs(level_jump) > 2:  # Significant jump
                        violations.append({
                            'position': i,
                            'from_location': location_path[-1]['location'],
                            'to_location': location,
                            'from_level': prev_level,
                            'to_level': level,
                            'level_jump': level_jump,
                            'violation_type': 'big_jump'
                        })
                    elif level_jump < -1:  # Inappropriate reverse
                        violations.append({
                            'position': i,
                            'from_location': location_path[-1]['location'],
                            'to_location': location,
                            'from_level': prev_level,
                            'to_level': level,
                            'level_jump': level_jump,
                            'violation_type': 'reverse_jump'
                        })
            
            location_path.append(location_info)
        
        return {
            'violations': violations,
            'location_path': location_path,
            'violation_count': len(violations),
            'unknown_location_count': sum(1 for info in location_path if info['is_unknown']),
            'violation_types': list(set(v['violation_type'] for v in violations))
        }