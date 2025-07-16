"""
02_features - Feature Extraction Components

This module contains feature extractors for each type of anomaly detection.
Each extractor converts raw EPC data into fixed-length numerical vectors
suitable for SVM training.

🎯 Learning Path:
1. Start with base_feature_extractor.py - understand common patterns
2. Try epc_fake_extractor.py - simplest case (single EPC analysis)
3. Explore other extractors - understand sequence analysis
4. Learn about fixed dimensions - critical for SVM compatibility

🔧 What each extractor does:
- Takes raw data (EPC codes, event sequences, locations, times)
- Converts to fixed-length numerical features
- Returns exactly the same number of features every time
- Preserves important patterns while standardizing format

🧠 Key Concepts:
- Fixed dimensions: SVM needs consistent input size
- Feature engineering: Extract meaningful patterns from raw data
- Temporal analysis: Understanding event sequences over time
- Statistical features: Mean, std, entropy, ratios, counts

Components:
- BaseFeatureExtractor: Common patterns and utilities
- EPCFakeExtractor: EPC format validation features (10 dimensions)
- EPCDupExtractor: Duplicate scan detection features (8 dimensions)
- EventOrderExtractor: Event sequence analysis features (12 dimensions)
- LocationErrorExtractor: Location hierarchy features (15 dimensions)
- TimeJumpExtractor: Time anomaly detection features (10 dimensions)
"""

from .base_feature_extractor import BaseFeatureExtractor
from .epc_fake_extractor import EPCFakeExtractor
from .epc_dup_extractor import EPCDupExtractor
from .event_order_extractor import EventOrderExtractor
from .location_error_extractor import LocationErrorExtractor
from .time_jump_extractor import TimeJumpExtractor

__all__ = [
    'BaseFeatureExtractor',
    'EPCFakeExtractor', 
    'EPCDupExtractor',
    'EventOrderExtractor',
    'LocationErrorExtractor',
    'TimeJumpExtractor'
]