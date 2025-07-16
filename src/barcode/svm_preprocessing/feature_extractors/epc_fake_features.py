"""
EPC Fake Feature Extractor for SVM Training
Reuses functions from multi_anomaly_detector.py lines 49-73, 75-96, 98-161
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_anomaly_detector import validate_epc_parts, validate_manufacture_date
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class EPCFakeFeatureExtractor:
    """EPC format validation features - Fixed 10 dimensions"""
    
    FEATURE_DIMENSIONS = 10
    
    def __init__(self):
        self.feature_names = [
            'structure_valid', 'header_valid', 'company_valid', 
            'product_valid', 'lot_valid', 'serial_valid',
            'date_valid', 'date_future_error', 'date_old_error',
            'epc_length_normalized'
        ]
    
    def extract_features(self, epc_code: str) -> List[float]:
        """
        Extract EPC format features for a single EPC code
        REUSE: lines 49-73 (validate_epc_parts), 75-96 (validate_manufacture_date)
        
        Returns fixed 10-dimensional feature vector
        """
        features = []
        
        # Parse EPC parts
        parts = epc_code.split('.')
        
        # REUSE: validate_epc_parts (lines 49-73)
        validations = validate_epc_parts(parts)
        
        # Basic validation features (6 dimensions)
        features.extend([
            float(validations['structure']),
            float(validations['header']),
            float(validations['company']),
            float(validations['product']),
            float(validations['lot']),
            float(validations['serial'])
        ])
        
        # Date validation features (3 dimensions)
        if len(parts) >= 5:
            # REUSE: validate_manufacture_date (lines 75-96)
            date_valid, error_type = validate_manufacture_date(parts[4])
            features.extend([
                float(date_valid),
                float(error_type == 'future_date'),
                float(error_type == 'too_old')
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # EPC length feature (1 dimension)
        # Normalize length to 0-1 range (assuming max reasonable length is 100)
        epc_length_normalized = min(len(epc_code) / 100.0, 1.0)
        features.append(epc_length_normalized)
        
        # Ensure exactly 10 dimensions
        assert len(features) == self.FEATURE_DIMENSIONS, f"Expected {self.FEATURE_DIMENSIONS} features, got {len(features)}"
        
        return features
    
    def extract_batch_features(self, epc_codes: List[str]) -> np.ndarray:
        """Extract features for multiple EPC codes"""
        feature_matrix = []
        
        for epc_code in epc_codes:
            features = self.extract_features(epc_code)
            feature_matrix.append(features)
        
        return np.array(feature_matrix)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        return self.feature_names.copy()
    
    def analyze_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate simple correlation-based feature importance"""
        importances = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if np.std(features[:, i]) > 0:  # Avoid division by zero
                correlation = np.corrcoef(features[:, i], labels)[0, 1]
                importances[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importances[feature_name] = 0.0
        
        return importances