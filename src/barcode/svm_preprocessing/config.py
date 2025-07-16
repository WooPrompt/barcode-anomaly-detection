"""
Configuration Management for SVM Preprocessing Pipeline
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    epc_fake_dimensions: int = 10
    epc_dup_dimensions: int = 8
    evt_order_dimensions: int = 12
    loc_err_dimensions: int = 15
    jump_dimensions: int = 10
    
    # Runtime optimization flags
    runtime_safe: bool = False
    exclude_heavy_features: bool = False
    max_sequence_length: int = 10


@dataclass
class LabelConfig:
    """Configuration for label generation"""
    epc_fake_threshold: int = 50
    epc_dup_threshold: int = 50
    evt_order_threshold: int = 50
    loc_err_threshold: int = 50
    jump_threshold: int = 50
    
    # Optimization settings
    auto_optimize_thresholds: bool = False
    target_positive_ratios: Dict[str, float] = None
    
    def __post_init__(self):
        if self.target_positive_ratios is None:
            self.target_positive_ratios = {
                'epcFake': 0.05,
                'epcDup': 0.02,
                'evtOrderErr': 0.10,
                'locErr': 0.08,
                'jump': 0.03
            }


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    # Normalization settings
    normalization_method: str = 'robust'  # 'robust' or 'standard'
    apply_normalization: bool = True
    
    # Class imbalance handling
    imbalance_strategy: str = 'smote'  # 'smote', 'weighted', 'threshold_tuning'
    min_samples_per_class: int = 50
    fallback_strategy: str = 'weighted'
    
    # Train/test split
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    # Memory management
    enable_batch_processing: bool = False
    batch_size: int = 10000
    
    # Runtime optimization
    runtime_safe_features: bool = False
    parallel_processing: bool = False
    max_workers: int = 4
    
    # Logging
    enable_logging: bool = True
    log_level: str = 'INFO'
    log_file: str = 'svm_preprocessing.log'


@dataclass
class SVMConfig:
    """Complete SVM preprocessing configuration"""
    features: FeatureConfig
    labels: LabelConfig
    preprocessing: PreprocessingConfig
    performance: PerformanceConfig
    
    # Output settings
    output_dir: str = 'data/svm_training'
    save_intermediate_results: bool = True
    
    # Pipeline settings
    anomaly_types: list = None
    
    def __post_init__(self):
        if self.anomaly_types is None:
            self.anomaly_types = ['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump']


class SVMConfigManager:
    """Manager for SVM preprocessing configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'svm_config.json'
        self._config = None
    
    def load_config(self, config_path: Optional[str] = None) -> SVMConfig:
        """Load configuration from file"""
        path = config_path or self.config_path
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            # Convert dict to config objects
            config = self._dict_to_config(config_dict)
        else:
            # Create default config
            config = self.get_default_config()
            # Save default config for future use
            self.save_config(config, path)
        
        self._config = config
        return config
    
    def save_config(self, config: SVMConfig, config_path: Optional[str] = None):
        """Save configuration to file"""
        path = config_path or self.config_path
        
        config_dict = self._config_to_dict(config)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_default_config(self) -> SVMConfig:
        """Get default configuration"""
        return SVMConfig(
            features=FeatureConfig(),
            labels=LabelConfig(),
            preprocessing=PreprocessingConfig(),
            performance=PerformanceConfig()
        )
    
    def get_runtime_safe_config(self) -> SVMConfig:
        """Get configuration optimized for runtime performance"""
        config = self.get_default_config()
        
        # Enable runtime optimizations
        config.features.runtime_safe = True
        config.features.exclude_heavy_features = True
        config.performance.enable_batch_processing = True
        config.performance.runtime_safe_features = True
        config.preprocessing.imbalance_strategy = 'weighted'  # Faster than SMOTE
        
        return config
    
    def get_development_config(self) -> SVMConfig:
        """Get configuration optimized for development/experimentation"""
        config = self.get_default_config()
        
        # Enable all features for experimentation
        config.features.runtime_safe = False
        config.labels.auto_optimize_thresholds = True
        config.performance.enable_logging = True
        config.save_intermediate_results = True
        
        return config
    
    def get_production_config(self) -> SVMConfig:
        """Get configuration optimized for production"""
        config = self.get_default_config()
        
        # Production optimizations
        config.features.runtime_safe = True
        config.performance.enable_batch_processing = True
        config.performance.parallel_processing = True
        config.preprocessing.imbalance_strategy = 'weighted'
        config.performance.log_level = 'WARNING'
        
        return config
    
    def update_config(self, **kwargs) -> SVMConfig:
        """Update current configuration with new values"""
        if self._config is None:
            self._config = self.load_config()
        
        # Update nested configuration
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested updates (e.g., 'features.runtime_safe')
                parts = key.split('.')
                obj = self._config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self._config, key, value)
        
        return self._config
    
    def _config_to_dict(self, config: SVMConfig) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        return {
            'features': asdict(config.features),
            'labels': asdict(config.labels),
            'preprocessing': asdict(config.preprocessing),
            'performance': asdict(config.performance),
            'output_dir': config.output_dir,
            'save_intermediate_results': config.save_intermediate_results,
            'anomaly_types': config.anomaly_types
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SVMConfig:
        """Convert dictionary to config object"""
        return SVMConfig(
            features=FeatureConfig(**config_dict.get('features', {})),
            labels=LabelConfig(**config_dict.get('labels', {})),
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            performance=PerformanceConfig(**config_dict.get('performance', {})),
            output_dir=config_dict.get('output_dir', 'data/svm_training'),
            save_intermediate_results=config_dict.get('save_intermediate_results', True),
            anomaly_types=config_dict.get('anomaly_types', ['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump'])
        )
    
    def validate_config(self, config: SVMConfig) -> Dict[str, Any]:
        """Validate configuration settings"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validate feature dimensions
        total_features = (
            config.features.epc_fake_dimensions +
            config.features.epc_dup_dimensions +
            config.features.evt_order_dimensions +
            config.features.loc_err_dimensions +
            config.features.jump_dimensions
        )
        
        if total_features > 1000:
            validation_results['warnings'].append(
                f"High total feature count ({total_features}). Consider reducing dimensions for better performance."
            )
        
        # Validate thresholds
        thresholds = [
            config.labels.epc_fake_threshold,
            config.labels.epc_dup_threshold,
            config.labels.evt_order_threshold,
            config.labels.loc_err_threshold,
            config.labels.jump_threshold
        ]
        
        for i, threshold in enumerate(thresholds):
            if not 0 <= threshold <= 100:
                validation_results['errors'].append(
                    f"Threshold {i} ({threshold}) must be between 0 and 100"
                )
                validation_results['valid'] = False
        
        # Validate test size
        if not 0.1 <= config.preprocessing.test_size <= 0.5:
            validation_results['warnings'].append(
                f"Test size ({config.preprocessing.test_size}) is outside recommended range (0.1-0.5)"
            )
        
        # Validate batch size
        if config.performance.enable_batch_processing and config.performance.batch_size < 100:
            validation_results['warnings'].append(
                f"Small batch size ({config.performance.batch_size}) may reduce efficiency"
            )
        
        # Validate output directory
        if not os.path.exists(os.path.dirname(config.output_dir or '.')):
            validation_results['warnings'].append(
                f"Output directory parent ({os.path.dirname(config.output_dir)}) does not exist"
            )
        
        return validation_results
    
    def get_config_summary(self, config: Optional[SVMConfig] = None) -> Dict[str, Any]:
        """Get summary of configuration settings"""
        if config is None:
            config = self._config or self.load_config()
        
        return {
            'feature_dimensions': {
                'epcFake': config.features.epc_fake_dimensions,
                'epcDup': config.features.epc_dup_dimensions,
                'evtOrderErr': config.features.evt_order_dimensions,
                'locErr': config.features.loc_err_dimensions,
                'jump': config.features.jump_dimensions,
                'total': (config.features.epc_fake_dimensions +
                         config.features.epc_dup_dimensions +
                         config.features.evt_order_dimensions +
                         config.features.loc_err_dimensions +
                         config.features.jump_dimensions)
            },
            'label_thresholds': {
                'epcFake': config.labels.epc_fake_threshold,
                'epcDup': config.labels.epc_dup_threshold,
                'evtOrderErr': config.labels.evt_order_threshold,
                'locErr': config.labels.loc_err_threshold,
                'jump': config.labels.jump_threshold
            },
            'optimization_settings': {
                'runtime_safe': config.features.runtime_safe,
                'batch_processing': config.performance.enable_batch_processing,
                'normalization': config.preprocessing.normalization_method,
                'imbalance_strategy': config.preprocessing.imbalance_strategy
            },
            'anomaly_types': config.anomaly_types,
            'output_directory': config.output_dir
        }


# Create global config manager instance
config_manager = SVMConfigManager()

# Convenience functions
def load_config(config_path: Optional[str] = None) -> SVMConfig:
    """Load SVM configuration"""
    return config_manager.load_config(config_path)

def save_config(config: SVMConfig, config_path: Optional[str] = None):
    """Save SVM configuration"""
    return config_manager.save_config(config, config_path)

def get_default_config() -> SVMConfig:
    """Get default SVM configuration"""
    return config_manager.get_default_config()

def get_runtime_safe_config() -> SVMConfig:
    """Get runtime-optimized configuration"""
    return config_manager.get_runtime_safe_config()

def get_production_config() -> SVMConfig:
    """Get production-optimized configuration"""
    return config_manager.get_production_config()