"""
01_core - Core Components for SVM Preprocessing

This module contains the fundamental building blocks that all other components use.
Start here to understand the basic data flow and processing concepts.

Components:
- BasePreprocessor: Cleans and groups raw barcode scan data
- SequenceProcessor: Handles variable-length sequences intelligently  
- FeatureNormalizer: Scales features for optimal SVM performance

Learning Path:
1. Read BasePreprocessor to understand data cleaning
2. Explore SequenceProcessor for sequence handling
3. Study FeatureNormalizer for SVM optimization
"""

from .base_preprocessor import BasePreprocessor
from .sequence_processor import SequenceProcessor  
from .feature_normalizer import FeatureNormalizer

__all__ = ['BasePreprocessor', 'SequenceProcessor', 'FeatureNormalizer']