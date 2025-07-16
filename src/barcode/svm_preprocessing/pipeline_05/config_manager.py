"""
Configuration Manager - Easy Customization for All Settings

This manages all configuration settings for SVM preprocessing in a 
beginner-friendly way. You can customize thresholds, feature dimensions,
and processing options without digging into code.

🎯 What this does:
- Centralize all configuration settings
- Provide easy ways to customize behavior
- Handle validation of settings
- Support different environments (dev, prod, etc.)

🚀 Example Usage:
    >>> config = ConfigManager()
    >>> config.set_threshold('epcFake', 75)  # More strict
    >>> config.set_feature_dimensions('epcDup', 12)  # More features
    >>> pipeline = PreprocessingPipeline(config=config)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    epc_fake_dimensions: int = 10
    epc_dup_dimensions: int = 8
    evt_order_dimensions: int = 12
    loc_err_dimensions: int = 15
    jump_dimensions: int = 10
    
    # Feature processing options
    enable_normalization: bool = True
    normalization_method: str = 'robust'  # 'robust' or 'standard'
    handle_outliers: bool = True
    outlier_threshold: float = 3.0


@dataclass
class LabelConfig:
    """Configuration for label generation"""
    epc_fake_threshold: int = 50
    epc_dup_threshold: int = 40
    evt_order_threshold: int = 60
    loc_err_threshold: int = 55
    jump_threshold: int = 45
    
    # Label processing options
    use_soft_labels: bool = False
    label_smoothing: float = 0.0


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    
    # Data cleaning
    remove_duplicates: bool = True
    handle_missing_values: bool = True
    validate_epc_format: bool = True
    
    # Sequence processing
    sequence_truncation_threshold: float = 0.03
    min_sequence_length: int = 1
    max_sequence_length: int = 50
    
    # Time handling
    time_format: str = '%Y-%m-%d %H:%M:%S'
    timezone_aware: bool = False


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    
    # Memory management
    max_memory_usage: float = 0.8
    enable_batch_processing: bool = False
    batch_size: int = 1000
    
    # Processing options
    parallel_processing: bool = False
    num_workers: int = 4
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    # Class imbalance handling
    handle_imbalance: bool = True
    imbalance_method: str = 'smote'  # 'smote', 'undersample', 'oversample'


class ConfigManager:
    """
    Centralized configuration management for SVM preprocessing.
    
    This class makes it easy to customize all aspects of the preprocessing
    pipeline without needing to modify code. Perfect for beginners who
    want to experiment with different settings.
    
    Features:
    - Easy-to-use methods for common customizations
    - Configuration validation
    - Save/load configurations to/from files
    - Different presets for different use cases
    - Environment-specific configurations
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file to load
        """
        
        # Initialize with default configurations
        self.features = FeatureConfig()
        self.labels = LabelConfig()
        self.preprocessing = PreprocessingConfig()
        self.performance = PerformanceConfig()
        
        # Meta configuration
        self.config_version = "1.0"
        self.created_by = "SVM Preprocessing Pipeline"
        self.description = "Default configuration"
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Setup logging
        self.logger = logging.getLogger('config_manager')
    
    def set_threshold(self, anomaly_type: str, threshold: int) -> None:
        """
        Set the threshold for a specific anomaly type.
        
        Higher thresholds = more strict detection (fewer positives)
        Lower thresholds = more lenient detection (more positives)
        
        Args:
            anomaly_type: Type of anomaly ('epcFake', 'epcDup', etc.)
            threshold: Threshold value (0-100)
            
        Example:
            >>> config.set_threshold('epcFake', 75)  # More strict
            >>> config.set_threshold('epcDup', 30)   # More lenient
        """
        
        if not 0 <= threshold <= 100:
            raise ValueError("Threshold must be between 0 and 100")
        
        threshold_map = {
            'epcFake': 'epc_fake_threshold',
            'epcDup': 'epc_dup_threshold',
            'evtOrderErr': 'evt_order_threshold',
            'locErr': 'loc_err_threshold',
            'jump': 'jump_threshold'
        }
        
        if anomaly_type not in threshold_map:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        setattr(self.labels, threshold_map[anomaly_type], threshold)
        
        self.logger.info(f"Set {anomaly_type} threshold to {threshold}")
    
    def set_feature_dimensions(self, anomaly_type: str, dimensions: int) -> None:
        """
        Set the number of features for a specific anomaly type.
        
        More features = more information but also more complexity
        Fewer features = simpler but might miss important patterns
        
        Args:
            anomaly_type: Type of anomaly
            dimensions: Number of features to extract
            
        Example:
            >>> config.set_feature_dimensions('epcDup', 12)  # More features
        """
        
        if dimensions < 1:
            raise ValueError("Feature dimensions must be positive")
        
        dimension_map = {
            'epcFake': 'epc_fake_dimensions',
            'epcDup': 'epc_dup_dimensions',
            'evtOrderErr': 'evt_order_dimensions',
            'locErr': 'loc_err_dimensions',
            'jump': 'jump_dimensions'
        }
        
        if anomaly_type not in dimension_map:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        setattr(self.features, dimension_map[anomaly_type], dimensions)
        
        self.logger.info(f"Set {anomaly_type} feature dimensions to {dimensions}")
    
    def set_normalization_method(self, method: str) -> None:
        """
        Set the feature normalization method.
        
        'robust': Less sensitive to outliers (recommended for anomaly detection)
        'standard': Standard scaling (may be affected by outliers)
        
        Args:
            method: Normalization method ('robust' or 'standard')
        """
        
        if method not in ['robust', 'standard']:
            raise ValueError("Method must be 'robust' or 'standard'")
        
        self.features.normalization_method = method
        self.logger.info(f"Set normalization method to {method}")
    
    def set_test_size(self, test_size: float) -> None:
        """
        Set the proportion of data to use for testing.
        
        Args:
            test_size: Fraction for testing (0.1 = 10%, 0.2 = 20%, etc.)
        """
        
        if not 0.05 <= test_size <= 0.5:
            raise ValueError("Test size must be between 0.05 and 0.5")
        
        self.performance.test_size = test_size
        self.logger.info(f"Set test size to {test_size:.1%}")
    
    def enable_batch_processing(self, batch_size: int = 1000) -> None:
        """
        Enable batch processing for large datasets.
        
        Args:
            batch_size: Number of items to process in each batch
        """
        
        if batch_size < 100:
            raise ValueError("Batch size should be at least 100")
        
        self.performance.enable_batch_processing = True
        self.performance.batch_size = batch_size
        
        self.logger.info(f"Enabled batch processing with size {batch_size}")
    
    def disable_batch_processing(self) -> None:
        """Disable batch processing for smaller datasets."""
        
        self.performance.enable_batch_processing = False
        self.logger.info("Disabled batch processing")
    
    def set_class_imbalance_handling(self, method: str, enabled: bool = True) -> None:
        """
        Configure how to handle class imbalance.
        
        Args:
            method: Method to use ('smote', 'undersample', 'oversample')
            enabled: Whether to enable imbalance handling
        """
        
        valid_methods = ['smote', 'undersample', 'oversample']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of: {valid_methods}")
        
        self.performance.handle_imbalance = enabled
        self.performance.imbalance_method = method
        
        self.logger.info(f"Set imbalance handling: {method} (enabled: {enabled})")
    
    def use_preset(self, preset_name: str) -> None:
        """
        Load a predefined configuration preset.
        
        Available presets:
        - 'strict': High thresholds, fewer false positives
        - 'lenient': Low thresholds, catch more potential issues
        - 'balanced': Balanced settings (default)
        - 'fast': Optimized for speed
        - 'accurate': Optimized for accuracy
        
        Args:
            preset_name: Name of the preset to load
        """
        
        presets = {
            'strict': self._get_strict_preset,
            'lenient': self._get_lenient_preset,
            'balanced': self._get_balanced_preset,
            'fast': self._get_fast_preset,
            'accurate': self._get_accurate_preset
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        preset_config = presets[preset_name]()
        
        # Apply preset
        self.features = preset_config['features']
        self.labels = preset_config['labels']
        self.preprocessing = preset_config['preprocessing']
        self.performance = preset_config['performance']
        
        self.description = f"Preset: {preset_name}"
        
        self.logger.info(f"Applied preset: {preset_name}")
    
    def _get_strict_preset(self) -> Dict[str, Any]:
        """Strict preset - fewer false positives"""
        return {
            'features': FeatureConfig(normalization_method='robust', handle_outliers=True),
            'labels': LabelConfig(
                epc_fake_threshold=70,
                epc_dup_threshold=60,
                evt_order_threshold=75,
                loc_err_threshold=70,
                jump_threshold=65
            ),
            'preprocessing': PreprocessingConfig(validate_epc_format=True),
            'performance': PerformanceConfig(handle_imbalance=True, test_size=0.25)
        }
    
    def _get_lenient_preset(self) -> Dict[str, Any]:
        """Lenient preset - catch more potential issues"""
        return {
            'features': FeatureConfig(normalization_method='robust'),
            'labels': LabelConfig(
                epc_fake_threshold=30,
                epc_dup_threshold=25,
                evt_order_threshold=35,
                loc_err_threshold=30,
                jump_threshold=25
            ),
            'preprocessing': PreprocessingConfig(),
            'performance': PerformanceConfig(handle_imbalance=True)
        }
    
    def _get_balanced_preset(self) -> Dict[str, Any]:
        """Balanced preset - default settings"""
        return {
            'features': FeatureConfig(),
            'labels': LabelConfig(),
            'preprocessing': PreprocessingConfig(),
            'performance': PerformanceConfig()
        }
    
    def _get_fast_preset(self) -> Dict[str, Any]:
        """Fast preset - optimized for speed"""
        return {
            'features': FeatureConfig(
                epc_fake_dimensions=8,
                epc_dup_dimensions=6,
                evt_order_dimensions=10,
                loc_err_dimensions=12,
                jump_dimensions=8,
                normalization_method='standard'
            ),
            'labels': LabelConfig(),
            'preprocessing': PreprocessingConfig(validate_epc_format=False),
            'performance': PerformanceConfig(
                enable_batch_processing=True,
                parallel_processing=True,
                handle_imbalance=False
            )
        }
    
    def _get_accurate_preset(self) -> Dict[str, Any]:
        """Accurate preset - optimized for accuracy"""
        return {
            'features': FeatureConfig(
                epc_fake_dimensions=12,
                epc_dup_dimensions=10,
                evt_order_dimensions=15,
                loc_err_dimensions=18,
                jump_dimensions=12,
                normalization_method='robust',
                handle_outliers=True
            ),
            'labels': LabelConfig(use_soft_labels=True, label_smoothing=0.1),
            'preprocessing': PreprocessingConfig(
                validate_epc_format=True,
                sequence_truncation_threshold=0.01
            ),
            'performance': PerformanceConfig(
                test_size=0.15,
                validation_size=0.15,
                handle_imbalance=True,
                imbalance_method='smote'
            )
        }
    
    def get_thresholds(self) -> Dict[str, int]:
        """Get all anomaly type thresholds as a dictionary."""
        return {
            'epcFake': self.labels.epc_fake_threshold,
            'epcDup': self.labels.epc_dup_threshold,
            'evtOrderErr': self.labels.evt_order_threshold,
            'locErr': self.labels.loc_err_threshold,
            'jump': self.labels.jump_threshold
        }
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get all feature dimensions as a dictionary."""
        return {
            'epcFake': self.features.epc_fake_dimensions,
            'epcDup': self.features.epc_dup_dimensions,
            'evtOrderErr': self.features.evt_order_dimensions,
            'locErr': self.features.loc_err_dimensions,
            'jump': self.features.jump_dimensions
        }
    
    def get_test_size(self) -> float:
        """Get the test size fraction."""
        return self.performance.test_size
    
    def get_handle_imbalance(self) -> bool:
        """Get whether class imbalance handling is enabled."""
        return self.performance.handle_imbalance
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path where to save the configuration
        """
        
        config_dict = {
            'meta': {
                'config_version': self.config_version,
                'created_by': self.created_by,
                'description': self.description,
                'created_at': datetime.now().isoformat()
            },
            'features': asdict(self.features),
            'labels': asdict(self.labels),
            'preprocessing': asdict(self.preprocessing),
            'performance': asdict(self.performance)
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration saved to {file_path}")
    
    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the configuration file
        """
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Load configurations
        if 'features' in config_dict:
            self.features = FeatureConfig(**config_dict['features'])
        
        if 'labels' in config_dict:
            self.labels = LabelConfig(**config_dict['labels'])
        
        if 'preprocessing' in config_dict:
            self.preprocessing = PreprocessingConfig(**config_dict['preprocessing'])
        
        if 'performance' in config_dict:
            self.performance = PerformanceConfig(**config_dict['performance'])
        
        # Load meta information
        if 'meta' in config_dict:
            meta = config_dict['meta']
            self.config_version = meta.get('config_version', self.config_version)
            self.created_by = meta.get('created_by', self.created_by)
            self.description = meta.get('description', self.description)
        
        self.logger.info(f"Configuration loaded from {file_path}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration and return any issues.
        
        Returns:
            Dictionary with validation results
        """
        
        issues = []
        warnings = []
        
        # Validate thresholds
        thresholds = self.get_thresholds()
        for anomaly_type, threshold in thresholds.items():
            if not 0 <= threshold <= 100:
                issues.append(f"Invalid threshold for {anomaly_type}: {threshold}")
        
        # Validate feature dimensions
        dimensions = self.get_feature_dimensions()
        for anomaly_type, dim in dimensions.items():
            if dim < 1:
                issues.append(f"Invalid feature dimensions for {anomaly_type}: {dim}")
            elif dim > 50:
                warnings.append(f"Very high feature dimensions for {anomaly_type}: {dim}")
        
        # Validate test size
        if not 0.05 <= self.performance.test_size <= 0.5:
            issues.append(f"Invalid test size: {self.performance.test_size}")
        
        # Validate batch size
        if self.performance.enable_batch_processing and self.performance.batch_size < 100:
            warnings.append(f"Small batch size may be inefficient: {self.performance.batch_size}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'score': max(0, 100 - len(issues) * 25 - len(warnings) * 5)
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        
        summary = f"""
Configuration Summary:
{'-' * 50}
Description: {self.description}

Anomaly Detection Thresholds:
  EPC Fake:     {self.labels.epc_fake_threshold}
  EPC Duplicate: {self.labels.epc_dup_threshold}
  Event Order:  {self.labels.evt_order_threshold}
  Location Error: {self.labels.loc_err_threshold}
  Time Jump:    {self.labels.jump_threshold}

Feature Dimensions:
  EPC Fake:     {self.features.epc_fake_dimensions}
  EPC Duplicate: {self.features.epc_dup_dimensions}
  Event Order:  {self.features.evt_order_dimensions}
  Location Error: {self.features.loc_err_dimensions}
  Time Jump:    {self.features.jump_dimensions}

Processing Settings:
  Normalization: {self.features.normalization_method}
  Test Size:    {self.performance.test_size:.1%}
  Batch Processing: {self.performance.enable_batch_processing}
  Handle Imbalance: {self.performance.handle_imbalance}
"""
        
        return summary.strip()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the configuration manager.
    Run this file directly to see it in action!
    """
    
    print("🎓 Configuration Manager Examples")
    print("=" * 50)
    
    # Create default configuration
    config = ConfigManager()
    print("\n📊 Default Configuration:")
    print(config.get_summary())
    
    # Customize settings
    print("\n🔧 Customizing Settings...")
    config.set_threshold('epcFake', 75)  # More strict
    config.set_threshold('epcDup', 30)   # More lenient
    config.set_feature_dimensions('epcDup', 12)  # More features
    config.set_normalization_method('standard')
    config.enable_batch_processing(1500)
    
    print("✅ Settings customized")
    
    # Try different presets
    print("\n🎨 Testing Presets...")
    
    for preset in ['strict', 'lenient', 'fast', 'accurate']:
        config.use_preset(preset)
        print(f"\n{preset.title()} Preset:")
        print(f"  EPC Fake threshold: {config.labels.epc_fake_threshold}")
        print(f"  Feature dimensions: {config.get_feature_dimensions()['epcDup']}")
        print(f"  Batch processing: {config.performance.enable_batch_processing}")
    
    # Save and load configuration
    print("\n💾 Testing Save/Load...")
    config.use_preset('balanced')
    config.save_to_file('temp_config.json')
    
    # Load it back
    new_config = ConfigManager('temp_config.json')
    print("✅ Configuration saved and loaded successfully")
    
    # Validate configuration
    validation = config.validate_configuration()
    print(f"\n🔍 Validation Score: {validation['score']}/100")
    if validation['issues']:
        print("Issues:", validation['issues'])
    if validation['warnings']:
        print("Warnings:", validation['warnings'])
    
    # Cleanup
    if os.path.exists('temp_config.json'):
        os.remove('temp_config.json')
    
    print("\n✅ Configuration manager examples completed!")