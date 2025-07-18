"""
Event-Level Location Error Feature Extractor for SVM Training
Refactored to produce a feature vector for each event in a sequence.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from multi_anomaly_detector import get_location_hierarchy_level

class LocationErrorFeatureExtractorEventLevel:
    """
    Extracts event-level features for location hierarchy violation detection.
    For each event, it analyzes its location relative to the previous and next events.
    """
    
    FEATURE_DIMENSIONS = 10
    
    def __init__(self):
        self.feature_names = [
            'current_location_level',
            'prev_location_level',
            'next_location_level',
            'level_change_from_prev',
            'level_change_to_next',
            'is_reverse_transition',
            'is_big_jump',
            'is_first_event',
            'is_last_event',
            'sequence_length'
        ]

    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        return self.feature_names.copy()

    def extract_features_per_event(self, epc_group: pd.DataFrame) -> List[List[float]]:
        """
        Extracts location-based features for each event in an EPC sequence.

        Args:
            epc_group: A DataFrame containing all events for a single EPC code.

        Returns:
            A list of feature vectors, one for each event in the epc_group.
        """
        if epc_group.empty:
            return []

        # Ensure data is sorted by time
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        sequence_length = len(epc_group)

        # Get hierarchy levels for all events in the sequence
        hierarchy_levels = [get_location_hierarchy_level(loc) for loc in epc_group['reader_location']]

        event_features_list = []
        for i in range(sequence_length):
            current_level = float(hierarchy_levels[i])
            prev_level = float(hierarchy_levels[i-1]) if i > 0 else -2.0  # Use -2 to signify no previous event
            next_level = float(hierarchy_levels[i+1]) if i < sequence_length - 1 else -2.0 # Use -2 to signify no next event

            level_change_from_prev = current_level - prev_level if i > 0 and current_level != -1 and prev_level != -1 else 0.0
            level_change_to_next = next_level - current_level if i < sequence_length - 1 and current_level != -1 and next_level != -1 else 0.0

            # Anomaly flags
            is_reverse = 1.0 if level_change_from_prev < -1 else 0.0
            is_big_jump = 1.0 if abs(level_change_from_prev) > 2 else 0.0

            is_first = 1.0 if i == 0 else 0.0
            is_last = 1.0 if i == sequence_length - 1 else 0.0

            features = [
                current_level,
                prev_level,
                next_level,
                level_change_from_prev,
                level_change_to_next,
                is_reverse,
                is_big_jump,
                is_first,
                is_last,
                float(sequence_length)
            ]
            
            event_features_list.append(features)
            
        return event_features_list
