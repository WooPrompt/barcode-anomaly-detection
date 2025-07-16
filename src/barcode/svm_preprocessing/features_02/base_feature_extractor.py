"""
Base Feature Extractor - Common Patterns for All Extractors

This is the FOUNDATION for all feature extraction. Every specific extractor
(EPC fake, duplicates, etc.) inherits from this base class to ensure
consistency and reusability.

🎓 Learning Goals:
- Understand the feature extraction contract (input → fixed output)
- Learn common utility functions used by all extractors
- See how to ensure fixed dimensions for SVM compatibility
- Understand the difference between raw features and SVM-ready features

🔧 What this provides:
- Fixed dimension enforcement (critical for SVM)
- Common statistical calculations (entropy, ratios, etc.)
- Sequence processing utilities
- Feature validation and debugging tools

🧠 Key Insight:
All feature extractors must return exactly the same number of features
every time, regardless of input complexity. This base class ensures that.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod
import sys
import os

# Import sequence processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from svm_preprocessing.sequence_processor import SequenceProcessor


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    This class defines the contract that all feature extractors must follow:
    1. Must have a FEATURE_DIMENSIONS constant (exact number of outputs)
    2. Must implement extract_features() method
    3. Must return exactly FEATURE_DIMENSIONS float values
    4. Must handle edge cases gracefully (empty data, invalid input)
    
    Example:
        >>> class MyExtractor(BaseFeatureExtractor):
        ...     FEATURE_DIMENSIONS = 5
        ...     def extract_features(self, data):
        ...         # Your feature logic here
        ...         return [1.0, 2.0, 3.0, 4.0, 5.0]  # Exactly 5 features
    """
    
    # Subclasses MUST define this
    FEATURE_DIMENSIONS: int = None
    
    def __init__(self):
        """Initialize the feature extractor with common utilities."""
        if self.FEATURE_DIMENSIONS is None:
            raise ValueError(f"{self.__class__.__name__} must define FEATURE_DIMENSIONS")
        
        self.sequence_processor = SequenceProcessor()
        print(f"🔧 {self.__class__.__name__} initialized")
        print(f"   📏 Fixed dimensions: {self.FEATURE_DIMENSIONS}")
    
    @abstractmethod
    def extract_features(self, data: Any) -> List[float]:
        """
        Extract features from input data.
        
        This method MUST be implemented by all subclasses.
        It MUST return exactly FEATURE_DIMENSIONS float values.
        
        Args:
            data: Input data (format varies by extractor type)
            
        Returns:
            List of exactly FEATURE_DIMENSIONS float values
        """
        pass
    
    def ensure_fixed_dimensions(self, features: List[float]) -> List[float]:
        """
        Ensure feature vector has exactly the required dimensions.
        
        This is critical for SVM - all training samples must have
        the same number of features, otherwise training fails.
        
        Args:
            features: Raw feature list (any length)
            
        Returns:
            Feature list with exactly FEATURE_DIMENSIONS elements
            
        Example:
            >>> extractor.FEATURE_DIMENSIONS = 5
            >>> raw_features = [1.0, 2.0, 3.0]  # Too short
            >>> fixed = extractor.ensure_fixed_dimensions(raw_features)
            >>> print(len(fixed))  # 5 (padded with zeros)
        """
        if len(features) == self.FEATURE_DIMENSIONS:
            return features
        
        elif len(features) < self.FEATURE_DIMENSIONS:
            # Pad with zeros if too few features
            padding_needed = self.FEATURE_DIMENSIONS - len(features)
            padded_features = features + [0.0] * padding_needed
            
            print(f"   ⚠️ Padded {padding_needed} features with zeros")
            return padded_features
        
        else:
            # Truncate if too many features
            truncated_features = features[:self.FEATURE_DIMENSIONS]
            removed_count = len(features) - self.FEATURE_DIMENSIONS
            
            print(f"   ⚠️ Truncated {removed_count} features")
            return truncated_features
    
    def calculate_entropy(self, sequence: List[Union[str, int, float]]) -> float:
        """
        Calculate Shannon entropy of a sequence.
        
        Entropy measures how "random" or "unpredictable" a sequence is.
        Higher entropy = more random, Lower entropy = more predictable.
        
        Args:
            sequence: List of values (strings, numbers, etc.)
            
        Returns:
            Entropy value (0.0 = completely predictable, higher = more random)
            
        Example:
            >>> # Very predictable sequence
            >>> entropy1 = extractor.calculate_entropy(['A', 'A', 'A', 'A'])
            >>> print(entropy1)  # 0.0 (completely predictable)
            >>>
            >>> # Random sequence  
            >>> entropy2 = extractor.calculate_entropy(['A', 'B', 'C', 'D'])
            >>> print(entropy2)  # 2.0 (maximum entropy for 4 items)
        """
        if not sequence:
            return 0.0
        
        # Count occurrences of each unique value
        from collections import Counter
        counts = Counter(sequence)
        total = len(sequence)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def calculate_sequence_statistics(self, sequence: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a numerical sequence.
        
        This provides a rich set of statistical features that often
        reveal anomalous patterns in the data.
        
        Args:
            sequence: List of numerical values
            
        Returns:
            Dictionary with statistical measures
            
        Example:
            >>> stats = extractor.calculate_sequence_statistics([1, 2, 3, 4, 5])
            >>> print(stats['mean'])     # 3.0
            >>> print(stats['std'])      # ~1.58
            >>> print(stats['range'])    # 4.0
        """
        if not sequence:
            return {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'range': 0.0, 'median': 0.0, 'q25': 0.0, 'q75': 0.0
            }
        
        sequence_array = np.array(sequence)
        
        stats = {
            'mean': float(np.mean(sequence_array)),
            'std': float(np.std(sequence_array)),
            'min': float(np.min(sequence_array)),
            'max': float(np.max(sequence_array)),
            'range': float(np.max(sequence_array) - np.min(sequence_array)),
            'median': float(np.median(sequence_array))
        }
        
        # Add quartiles if we have enough data
        if len(sequence) >= 4:
            stats['q25'] = float(np.percentile(sequence_array, 25))
            stats['q75'] = float(np.percentile(sequence_array, 75))
        else:
            stats['q25'] = stats['median']
            stats['q75'] = stats['median']
        
        return stats
    
    def calculate_ratio_features(self, numerator_count: int, total_count: int) -> float:
        """
        Calculate ratio features with safe division.
        
        Ratios are common in anomaly detection (success rate, error rate, etc.)
        This ensures we handle division by zero gracefully.
        
        Args:
            numerator_count: Count of specific events
            total_count: Total count of all events
            
        Returns:
            Ratio as float (0.0 to 1.0)
            
        Example:
            >>> ratio = extractor.calculate_ratio_features(3, 10)
            >>> print(ratio)  # 0.3
            >>>
            >>> safe_ratio = extractor.calculate_ratio_features(5, 0)
            >>> print(safe_ratio)  # 0.0 (safe division by zero)
        """
        if total_count == 0:
            return 0.0
        return float(numerator_count) / float(total_count)
    
    def detect_outliers_zscore(self, sequence: List[float], threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.
        
        Z-score measures how many standard deviations away from the mean
        each value is. Values with |z-score| > threshold are outliers.
        
        Args:
            sequence: Numerical sequence to analyze
            threshold: Z-score threshold for outlier detection (default: 2.0)
            
        Returns:
            Dictionary with outlier analysis
            
        Example:
            >>> values = [1, 2, 3, 4, 100]  # 100 is clearly an outlier
            >>> outliers = extractor.detect_outliers_zscore(values)
            >>> print(outliers['outlier_count'])  # 1
            >>> print(outliers['outlier_ratio'])  # 0.2 (1 out of 5)
        """
        if len(sequence) < 3:  # Need at least 3 points for meaningful z-score
            return {
                'outlier_count': 0,
                'outlier_ratio': 0.0,
                'outlier_indices': [],
                'max_zscore': 0.0
            }
        
        sequence_array = np.array(sequence)
        mean = np.mean(sequence_array)
        std = np.std(sequence_array)
        
        if std == 0:  # All values are the same
            return {
                'outlier_count': 0,
                'outlier_ratio': 0.0,
                'outlier_indices': [],
                'max_zscore': 0.0
            }
        
        # Calculate z-scores
        z_scores = np.abs((sequence_array - mean) / std)
        outlier_mask = z_scores > threshold
        
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_count = len(outlier_indices)
        outlier_ratio = outlier_count / len(sequence)
        max_zscore = float(np.max(z_scores))
        
        return {
            'outlier_count': outlier_count,
            'outlier_ratio': outlier_ratio,
            'outlier_indices': outlier_indices,
            'max_zscore': max_zscore
        }
    
    def process_time_sequence(self, timestamps: List[str], 
                            target_length: Optional[int] = None) -> List[float]:
        """
        Process timestamp sequences into numerical features.
        
        Converts timestamps into time differences (hours between events)
        and processes them to fixed length for SVM compatibility.
        
        Args:
            timestamps: List of timestamp strings
            target_length: Desired sequence length (uses FEATURE_DIMENSIONS//2 if None)
            
        Returns:
            List of time differences in hours
            
        Example:
            >>> timestamps = ['2024-01-01 10:00:00', '2024-01-01 11:30:00']
            >>> time_features = extractor.process_time_sequence(timestamps)
            >>> print(time_features)  # [1.5] (1.5 hours difference)
        """
        if len(timestamps) < 2:
            target_len = target_length or max(1, self.FEATURE_DIMENSIONS // 2)
            return [0.0] * target_len
        
        # Convert to datetime and calculate differences
        try:
            datetime_objects = pd.to_datetime(timestamps)
            time_diffs = datetime_objects.diff().dt.total_seconds() / 3600  # Convert to hours
            time_diffs = time_diffs.dropna().tolist()  # Remove first NaN value
        except Exception as e:
            print(f"   ⚠️ Error processing timestamps: {e}")
            target_len = target_length or max(1, self.FEATURE_DIMENSIONS // 2)
            return [0.0] * target_len
        
        # Process to fixed length if specified
        if target_length:
            time_diffs = self.sequence_processor.process_sequence(
                time_diffs, target_length, 'temporal'
            )
        
        return time_diffs
    
    def validate_features(self, features: List[float]) -> bool:
        """
        Validate that extracted features are suitable for SVM.
        
        Checks for common problems that would break SVM training:
        - Wrong number of features
        - NaN or infinite values
        - All zeros (uninformative features)
        
        Args:
            features: Feature list to validate
            
        Returns:
            True if features are valid, False otherwise
            
        Example:
            >>> good_features = [1.0, 2.0, 3.0, 4.0, 5.0]
            >>> valid = extractor.validate_features(good_features)
            >>> print(valid)  # True
            >>>
            >>> bad_features = [1.0, float('nan'), 3.0]
            >>> valid = extractor.validate_features(bad_features)
            >>> print(valid)  # False
        """
        # Check dimension count
        if len(features) != self.FEATURE_DIMENSIONS:
            print(f"   ❌ Wrong feature count: {len(features)} != {self.FEATURE_DIMENSIONS}")
            return False
        
        # Check for invalid values
        for i, value in enumerate(features):
            if not isinstance(value, (int, float)):
                print(f"   ❌ Feature {i} is not numeric: {type(value)}")
                return False
            
            if np.isnan(value) or np.isinf(value):
                print(f"   ❌ Feature {i} is NaN or infinite: {value}")
                return False
        
        # Check for all-zero features (might indicate a problem)
        if all(f == 0.0 for f in features):
            print(f"   ⚠️ Warning: All features are zero (might be uninformative)")
        
        return True
    
    def get_feature_names(self) -> List[str]:
        """
        Get descriptive names for each feature dimension.
        
        Subclasses should override this to provide meaningful names.
        This helps with model interpretation and debugging.
        
        Returns:
            List of feature names
        """
        return [f"feature_{i}" for i in range(self.FEATURE_DIMENSIONS)]
    
    def extract_and_validate(self, data: Any) -> List[float]:
        """
        Extract features and validate the result.
        
        This is the main entry point that combines feature extraction
        with validation and dimension enforcement.
        
        Args:
            data: Input data for feature extraction
            
        Returns:
            Validated feature vector with exactly FEATURE_DIMENSIONS elements
            
        Example:
            >>> # Safe extraction that handles all edge cases
            >>> features = extractor.extract_and_validate(input_data)
            >>> print(len(features))  # Always equals FEATURE_DIMENSIONS
        """
        try:
            # Extract raw features
            raw_features = self.extract_features(data)
            
            # Ensure fixed dimensions
            fixed_features = self.ensure_fixed_dimensions(raw_features)
            
            # Validate result
            if not self.validate_features(fixed_features):
                print(f"   ⚠️ Feature validation failed, using zeros")
                return [0.0] * self.FEATURE_DIMENSIONS
            
            return fixed_features
            
        except Exception as e:
            print(f"   ❌ Feature extraction failed: {e}")
            return [0.0] * self.FEATURE_DIMENSIONS
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about this feature extractor.
        
        Returns metadata about the extractor for documentation and debugging.
        
        Returns:
            Dictionary with extractor information
        """
        return {
            'extractor_name': self.__class__.__name__,
            'feature_dimensions': self.FEATURE_DIMENSIONS,
            'feature_names': self.get_feature_names(),
            'description': self.__doc__.split('\\n')[0] if self.__doc__ else 'No description'
        }


# Example implementation for learning
class ExampleFeatureExtractor(BaseFeatureExtractor):
    """
    Example feature extractor for learning purposes.
    
    This shows how to implement a real feature extractor by inheriting
    from BaseFeatureExtractor and implementing the required methods.
    """
    
    FEATURE_DIMENSIONS = 3
    
    def extract_features(self, data: str) -> List[float]:
        """Extract simple features from a string."""
        if not data:
            return [0.0, 0.0, 0.0]
        
        features = [
            float(len(data)),                    # Feature 1: String length
            float(data.count('.')),              # Feature 2: Number of dots
            self.calculate_entropy(list(data))   # Feature 3: Character entropy
        ]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get descriptive names for our features."""
        return ['string_length', 'dot_count', 'character_entropy']


# Testing and examples
if __name__ == "__main__":
    """
    Example usage of BaseFeatureExtractor.
    
    This shows how to create and use feature extractors.
    Run this file directly to see it in action!
    """
    
    print("🎓 BaseFeatureExtractor Example")
    print("=" * 50)
    
    # Create example extractor
    extractor = ExampleFeatureExtractor()
    
    # Test with various inputs
    test_cases = [
        "001.8804823.1234567",           # Normal EPC part
        "invalid.format",                # Simple string
        "",                              # Empty string
        "a.b.c.d.e.f.g.h.i.j.k.l.m"    # Long string with many dots
    ]
    
    print(f"\\n🧪 Testing feature extraction:")
    for i, test_input in enumerate(test_cases):
        print(f"\\n   Test {i+1}: '{test_input}'")
        
        # Extract features
        features = extractor.extract_and_validate(test_input)
        feature_names = extractor.get_feature_names()
        
        # Display results
        for name, value in zip(feature_names, features):
            print(f"      {name}: {value:.3f}")
    
    # Test utility functions
    print(f"\\n🔧 Testing utility functions:")
    
    # Test entropy calculation
    test_sequences = [
        ['A', 'A', 'A', 'A'],              # Low entropy (predictable)
        ['A', 'B', 'C', 'D'],              # High entropy (random)
        [1, 2, 1, 2, 1, 2],                # Medium entropy (pattern)
    ]
    
    for seq in test_sequences:
        entropy = extractor.calculate_entropy(seq)
        print(f"   Entropy of {seq}: {entropy:.3f}")
    
    # Test outlier detection
    test_values = [1, 2, 3, 4, 5, 100]  # 100 is clearly an outlier
    outliers = extractor.detect_outliers_zscore(test_values)
    print(f"   Outliers in {test_values}: {outliers['outlier_count']} found")
    
    # Test sequence statistics
    stats = extractor.calculate_sequence_statistics([1, 2, 3, 4, 5])
    print(f"   Statistics: mean={stats['mean']:.1f}, std={stats['std']:.1f}")
    
    # Show extractor info
    info = extractor.get_feature_info()
    print(f"\\n📋 Extractor Info:")
    print(f"   Name: {info['extractor_name']}")
    print(f"   Dimensions: {info['feature_dimensions']}")
    print(f"   Features: {info['feature_names']}")
    
    print("\\n✅ Example complete!")
    print("Next step: Learn specific extractors in epc_fake_extractor.py")