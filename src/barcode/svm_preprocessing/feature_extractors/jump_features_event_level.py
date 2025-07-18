"""
Event-Level Time Jump Feature Extractor for SVM Training
Refactored to produce a feature vector for each event in a sequence.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import List, Dict, Any

class JumpFeatureExtractorEventLevel:
    """
    Extracts event-level features for time jump/impossible travel detection.
    For each event, it looks at the time difference to the previous and next events.
    """
    
    FEATURE_DIMENSIONS = 8
    
    def __init__(self):
        self.feature_names = [
            'time_diff_from_prev_hours',
            'time_diff_to_next_hours',
            'normalized_event_position',
            'sequence_length',
            'total_time_span_hours',
            'avg_time_gap_in_sequence_hours',
            'is_first_event',
            'is_last_event'
        ]

    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        return self.feature_names.copy()

    def extract_features_per_event(self, epc_group: pd.DataFrame) -> List[List[float]]:
        """
        Extracts time-based features for each event in an EPC sequence.

        Args:
            epc_group: A DataFrame containing all events for a single EPC code.

        Returns:
            A list of feature vectors, one for each event in the epc_group.
        """
        if epc_group.empty:
            return []

        # Ensure data is sorted by time
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        # Convert event_time to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(epc_group['event_time']):
            epc_group['event_time'] = pd.to_datetime(epc_group['event_time'])

        sequence_length = len(epc_group)
        if sequence_length == 1:
            # Handle single-event sequences
            return [[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0]]

        # Calculate time differences between consecutive events
        time_diffs_hours = epc_group['event_time'].diff().dt.total_seconds().fillna(0) / 3600
        
        total_time_span_hours = (epc_group['event_time'].iloc[-1] - epc_group['event_time'].iloc[0]).total_seconds() / 3600
        avg_time_gap_in_sequence_hours = time_diffs_hours[1:].mean() if sequence_length > 1 else 0

        event_features_list = []
        for i in range(sequence_length):
            is_first = 1.0 if i == 0 else 0.0
            is_last = 1.0 if i == sequence_length - 1 else 0.0

            time_from_prev = time_diffs_hours.iloc[i] if i > 0 else 0.0
            time_to_next = time_diffs_hours.iloc[i + 1] if i < sequence_length - 1 else 0.0
            
            # Normalize event position (0 for first, 1 for last)
            normalized_pos = i / (sequence_length - 1) if sequence_length > 1 else 0.5

            features = [
                time_from_prev,
                time_to_next,
                normalized_pos,
                float(sequence_length),
                total_time_span_hours,
                avg_time_gap_in_sequence_hours,
                is_first,
                is_last
            ]
            
            event_features_list.append(features)
            
        return event_features_list
