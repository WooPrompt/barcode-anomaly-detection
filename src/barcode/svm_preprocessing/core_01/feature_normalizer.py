"""
Feature Normalizer - Essential for SVM Success

This is the THIRD core component to understand. SVM is extremely sensitive
to feature scaling - without proper normalization, SVM will focus only on
the largest values and ignore all other features completely.

🎓 Learning Goals:
- Understand why feature scaling is critical for SVM
- Learn the difference between StandardScaler and RobustScaler
- See how to save/load scalers for consistent prediction

🚨 Real Problem Example:
Without normalization:
- EPC features: [1.0, 0.0, 1.0, 47.0, 0.000234, 1672531200, 156.8, 0.85]
- SVM sees: "1672531200 is the most important feature! Ignore everything else!"
- Result: Model only considers timestamps, ignores actual anomaly patterns

With normalization:
- Normalized: [0.23, -1.41, 0.23, 0.15, -0.89, 1.34, 0.67, 0.12]
- SVM sees: "All features are equally important"
- Result: Model learns real anomaly patterns
"""

import numpy as np
import joblib
import os
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler


class FeatureNormalizer:
    """
    Normalize features for optimal SVM performance.
    
    This class solves the critical problem where large feature values
    (like timestamps or counts) dominate small feature values (like ratios).
    SVM uses euclidean distance, so scale differences destroy performance.
    
    Example:
        >>> normalizer = FeatureNormalizer()
        >>> # Raw features with mixed scales
        >>> raw_features = np.array([[1.0, 47.0, 1672531200]])
        >>> # Normalize for training
        >>> normalized = normalizer.fit_transform(raw_features, 'epcFake')
        >>> print(normalized)  # All values now in similar range
    """
    
    def __init__(self, method: str = 'robust'):
        """
        Initialize the feature normalizer.
        
        Args:
            method: Normalization method to use
                   'robust' - Uses median and IQR (recommended for anomaly data)
                   'standard' - Uses mean and std (classic approach)
        
        Why 'robust' is better for anomaly detection:
        - Robust to outliers (important for anomaly data)
        - Uses median instead of mean (less affected by extreme values)
        - Uses IQR instead of std (more stable with skewed data)
        """
        self.method = method
        self.scalers = {}  # Store fitted scalers for each anomaly type
        
        print(f"🔧 FeatureNormalizer initialized with '{method}' method")
        if method == 'robust':
            print("   📊 Using RobustScaler (recommended for anomaly detection)")
        else:
            print("   📊 Using StandardScaler (classic normalization)")
    
    def fit_transform_features(self, X: np.ndarray, anomaly_type: str) -> np.ndarray:
        """
        Fit scaler on training data and transform features.
        
        This is used during training phase. The scaler learns the statistics
        of your training data and saves them for future use.
        
        Args:
            X: Feature matrix (samples × features)
            anomaly_type: Type of anomaly (e.g., 'epcFake', 'epcDup')
            
        Returns:
            Normalized feature matrix
            
        Example:
            >>> # Training phase
            >>> raw_features = np.array([[1.0, 100.0], [2.0, 200.0]])
            >>> normalized = normalizer.fit_transform_features(raw_features, 'epcFake')
            >>> print(normalized)  # Values normalized to reasonable range
        """
        print(f"📏 Fitting and transforming features for {anomaly_type}")
        print(f"   📊 Input shape: {X.shape}")
        
        # Show scale differences before normalization
        feature_ranges = []
        for i in range(X.shape[1]):
            col_min, col_max = X[:, i].min(), X[:, i].max()
            feature_ranges.append(col_max - col_min)
        
        max_range = max(feature_ranges) if feature_ranges else 1
        min_range = min(feature_ranges) if feature_ranges else 1
        scale_ratio = max_range / min_range if min_range > 0 else 1
        
        print(f"   ⚖️ Scale difference: {scale_ratio:.1f}x (max range / min range)")
        if scale_ratio > 1000:
            print(f"   🚨 WARNING: Extreme scale differences detected!")
            print(f"      This will severely hurt SVM performance without normalization")
        
        # Create and fit scaler
        if self.method == 'robust':
            scaler = RobustScaler()
            print(f"   📈 Using RobustScaler (median and IQR)")
        else:
            scaler = StandardScaler()
            print(f"   📈 Using StandardScaler (mean and std)")
        
        # Fit and transform
        X_scaled = scaler.fit_transform(X)
        
        # Store scaler for prediction time
        self.scalers[anomaly_type] = scaler
        
        # Show improvement
        new_ranges = []
        for i in range(X_scaled.shape[1]):
            col_min, col_max = X_scaled[:, i].min(), X_scaled[:, i].max()
            new_ranges.append(col_max - col_min)
        
        new_max_range = max(new_ranges) if new_ranges else 1
        new_min_range = min(new_ranges) if new_ranges else 1
        new_scale_ratio = new_max_range / new_min_range if new_min_range > 0 else 1
        
        print(f"   ✅ After normalization: {new_scale_ratio:.1f}x scale difference")
        print(f"   📊 Output shape: {X_scaled.shape}")
        
        return X_scaled
    
    def transform_features(self, X: np.ndarray, anomaly_type: str) -> np.ndarray:
        """
        Transform features using previously fitted scaler.
        
        This is used during prediction phase. Uses the same statistics
        learned during training to normalize new data consistently.
        
        Args:
            X: Feature matrix to transform
            anomaly_type: Type of anomaly (must match training)
            
        Returns:
            Normalized feature matrix
            
        Example:
            >>> # Prediction phase
            >>> new_features = np.array([[1.5, 150.0]])
            >>> normalized = normalizer.transform_features(new_features, 'epcFake')
        """
        if anomaly_type not in self.scalers:
            raise ValueError(
                f"No fitted scaler found for '{anomaly_type}'. "
                f"Available types: {list(self.scalers.keys())}. "
                f"Call fit_transform_features() first during training."
            )
        
        scaler = self.scalers[anomaly_type]
        X_scaled = scaler.transform(X)
        
        print(f"🔄 Transformed {X.shape[0]} samples for {anomaly_type}")
        
        return X_scaled
    
    def save_scalers(self, output_dir: str) -> Dict[str, str]:
        """
        Save fitted scalers to disk for later use.
        
        This is critical for production deployment - you need the exact
        same normalization during prediction as during training.
        
        Args:
            output_dir: Directory to save scaler files
            
        Returns:
            Dictionary mapping anomaly types to saved file paths
            
        Example:
            >>> file_paths = normalizer.save_scalers('models/scalers/')
            >>> print(file_paths)  # {'epcFake': 'models/scalers/epcFake_scaler.joblib', ...}
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        print(f"💾 Saving {len(self.scalers)} fitted scalers to {output_dir}")
        
        for anomaly_type, scaler in self.scalers.items():
            filename = f"{anomaly_type}_scaler.joblib"
            filepath = os.path.join(output_dir, filename)
            
            joblib.dump(scaler, filepath)
            saved_files[anomaly_type] = filepath
            
            print(f"   ✅ Saved {anomaly_type} scaler: {filename}")
        
        return saved_files
    
    def load_scalers(self, model_dir: str) -> Dict[str, bool]:
        """
        Load previously saved scalers from disk.
        
        This is used when starting a prediction service - load the scalers
        that were fitted during training.
        
        Args:
            model_dir: Directory containing saved scaler files
            
        Returns:
            Dictionary showing which scalers were successfully loaded
            
        Example:
            >>> success = normalizer.load_scalers('models/scalers/')
            >>> print(success)  # {'epcFake': True, 'epcDup': False, ...}
        """
        load_results = {}
        expected_types = ['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump']
        
        print(f"📂 Loading scalers from {model_dir}")
        
        for anomaly_type in expected_types:
            filename = f"{anomaly_type}_scaler.joblib"
            filepath = os.path.join(model_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    scaler = joblib.load(filepath)
                    self.scalers[anomaly_type] = scaler
                    load_results[anomaly_type] = True
                    print(f"   ✅ Loaded {anomaly_type} scaler")
                except Exception as e:
                    load_results[anomaly_type] = False
                    print(f"   ❌ Failed to load {anomaly_type} scaler: {e}")
            else:
                load_results[anomaly_type] = False
                print(f"   ⚠️  {anomaly_type} scaler not found: {filename}")
        
        loaded_count = sum(load_results.values())
        print(f"📊 Successfully loaded {loaded_count}/{len(expected_types)} scalers")
        
        return load_results
    
    def get_scaler_info(self, anomaly_type: str) -> Optional[Dict]:
        """
        Get information about a fitted scaler.
        
        This is useful for debugging and understanding what the scaler learned.
        
        Args:
            anomaly_type: Type of anomaly to inspect
            
        Returns:
            Dictionary with scaler information, or None if not found
            
        Example:
            >>> info = normalizer.get_scaler_info('epcFake')
            >>> print(f"Center values: {info['center']}")
            >>> print(f"Scale values: {info['scale']}")
        """
        if anomaly_type not in self.scalers:
            return None
        
        scaler = self.scalers[anomaly_type]
        
        if isinstance(scaler, RobustScaler):
            return {
                'type': 'RobustScaler',
                'center': scaler.center_.tolist() if hasattr(scaler, 'center_') else None,
                'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                'method': 'median_and_iqr'
            }
        elif isinstance(scaler, StandardScaler):
            return {
                'type': 'StandardScaler',
                'center': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                'method': 'mean_and_std'
            }
        else:
            return {
                'type': type(scaler).__name__,
                'method': 'unknown'
            }
    
    def demonstrate_normalization_effect(self) -> None:
        """
        Show examples of normalization effects on different feature types.
        
        This is a learning function that demonstrates why normalization
        is crucial for SVM performance.
        """
        print("🎓 Feature Normalization Demonstration")
        print("=" * 50)
        
        # Create example features with very different scales
        print("📊 Creating features with mixed scales...")
        
        # Simulate real barcode anomaly features
        raw_features = np.array([
            # Sample 1: Normal EPC
            [1.0, 1.0, 1.0, 47.0, 0.000234, 1672531200, 156.8, 0.85],
            # Sample 2: Anomalous EPC  
            [0.0, 0.0, 0.0, 45.0, 0.000198, 1672531150, 154.2, 0.82],
            # Sample 3: Another normal EPC
            [1.0, 1.0, 1.0, 48.0, 0.000267, 1672531300, 158.1, 0.87]
        ])
        
        feature_names = [
            'structure_valid', 'header_valid', 'company_valid', 'epc_length',
            'entropy', 'timestamp', 'char_count', 'digit_ratio'
        ]
        
        print("\\n📈 Original features (mixed scales):")
        for i, name in enumerate(feature_names):
            col = raw_features[:, i]
            print(f"   {name:15}: {col[0]:12.6f} | Range: {col.min():.6f} - {col.max():.6f}")
        
        # Calculate SVM distance before normalization
        sample1, sample2 = raw_features[0], raw_features[1]
        distance_before = np.sqrt(np.sum((sample1 - sample2) ** 2))
        
        # Show which features dominate
        feature_contributions = (sample1 - sample2) ** 2
        dominant_feature = np.argmax(feature_contributions)
        contribution_ratio = feature_contributions[dominant_feature] / np.sum(feature_contributions)
        
        print(f"\\n🎯 SVM distance analysis (before normalization):")
        print(f"   Total distance: {distance_before:.2f}")
        print(f"   Dominant feature: {feature_names[dominant_feature]} ({contribution_ratio:.1%} of total)")
        print(f"   🚨 Problem: '{feature_names[dominant_feature]}' dominates everything else!")
        
        # Apply normalization
        print(f"\\n🔄 Applying normalization...")
        normalized_features = self.fit_transform_features(raw_features, 'demo')
        
        print(f"\\n📉 Normalized features (balanced scales):")
        for i, name in enumerate(feature_names):
            col = normalized_features[:, i]
            print(f"   {name:15}: {col[0]:8.3f} | Range: {col.min():.3f} - {col.max():.3f}")
        
        # Calculate SVM distance after normalization
        norm_sample1, norm_sample2 = normalized_features[0], normalized_features[1]
        distance_after = np.sqrt(np.sum((norm_sample1 - norm_sample2) ** 2))
        
        # Show balanced contributions
        norm_contributions = (norm_sample1 - norm_sample2) ** 2
        contribution_balance = np.std(norm_contributions) / np.mean(norm_contributions)
        
        print(f"\\n✅ SVM distance analysis (after normalization):")
        print(f"   Total distance: {distance_after:.2f}")
        print(f"   Contribution balance: {contribution_balance:.3f} (lower = more balanced)")
        print(f"   🎉 Success: All features now contribute meaningfully!")
        
        # Clean up demo scaler
        if 'demo' in self.scalers:
            del self.scalers['demo']
        
        print(f"\\n💡 Key Takeaway:")
        print(f"   Without normalization: {feature_names[dominant_feature]} = {contribution_ratio:.0%} of decision")
        print(f"   With normalization: Balanced contributions from all features")
        print(f"   Result: SVM can learn real anomaly patterns instead of just scale differences")


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of FeatureNormalizer.
    
    This shows how to use the normalizer for training and prediction.
    Run this file directly to see it in action!
    """
    
    print("🎓 FeatureNormalizer Example")
    print("=" * 50)
    
    # Initialize normalizer
    normalizer = FeatureNormalizer(method='robust')
    
    # Run demonstration
    normalizer.demonstrate_normalization_effect()
    
    # Test with realistic EPC features
    print("\\n🧪 Real EPC Feature Example:")
    print("-" * 30)
    
    # Create realistic EPC features with scale problems
    epc_features = np.array([
        # Normal EPCs
        [1, 1, 1, 47, 0.85, 1672531200, 8, 156],
        [1, 1, 1, 48, 0.87, 1672531300, 8, 158],
        [1, 1, 1, 46, 0.83, 1672531100, 8, 154],
        # Anomalous EPCs  
        [0, 0, 0, 12, 0.23, 1672530000, 3, 45],
        [0, 1, 0, 25, 0.45, 1672532000, 5, 89]
    ])
    
    print(f"📊 Input features shape: {epc_features.shape}")
    print(f"📊 Feature ranges before normalization:")
    for i in range(epc_features.shape[1]):
        col = epc_features[:, i]
        print(f"   Feature {i}: {col.min():.1f} - {col.max():.1f} (range: {col.max()-col.min():.1f})")
    
    # Normalize for training
    normalized = normalizer.fit_transform_features(epc_features, 'epcFake')
    
    print(f"\\n📊 Feature ranges after normalization:")
    for i in range(normalized.shape[1]):
        col = normalized[:, i]
        print(f"   Feature {i}: {col.min():.3f} - {col.max():.3f} (range: {col.max()-col.min():.3f})")
    
    # Test prediction with new data
    print(f"\\n🔮 Testing prediction normalization:")
    new_epc = np.array([[1, 1, 1, 47, 0.85, 1672531250, 8, 157]])
    normalized_new = normalizer.transform_features(new_epc, 'epcFake')
    print(f"   New EPC normalized: {normalized_new[0]}")
    
    # Test scaler info
    info = normalizer.get_scaler_info('epcFake')
    print(f"\\n📋 Scaler info: {info['type']} using {info['method']}")
    
    print("\\n✅ Example complete!")
    print("Next step: Learn about Feature Extractors in ../02_features/")