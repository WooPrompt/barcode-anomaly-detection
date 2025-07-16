"""
Preprocessing Pipeline - Full-Featured Pipeline with All Options

This is the COMPLETE preprocessing pipeline for advanced users who want
full control over every aspect of the SVM preprocessing workflow.

🎯 When to use this:
- You understand the basics and want full customization
- You need to fine-tune specific aspects of preprocessing
- You want to integrate into larger systems
- You need advanced features like batch processing

🚀 Example Usage:
    >>> pipeline = PreprocessingPipeline(
    ...     config=custom_config,
    ...     enable_logging=True,
    ...     batch_processing=True
    ... )
    >>> results = pipeline.process_data(data, save_data=True)
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime

# Import our components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svm_preprocessing.base_preprocessor import BasePreprocessor
from svm_preprocessing.feature_extractors.epc_fake_features import EPCFakeFeatureExtractor
from svm_preprocessing.feature_extractors.epc_dup_features import EPCDupFeatureExtractor
from svm_preprocessing.feature_extractors.evt_order_features import EventOrderFeatureExtractor
from svm_preprocessing.feature_extractors.loc_err_features import LocationErrorFeatureExtractor
from svm_preprocessing.feature_extractors.jump_features import JumpFeatureExtractor
from svm_preprocessing.label_generators.rule_based_labels import RuleBasedLabelGenerator
from svm_preprocessing.data_manager import SVMDataManager
from svm_preprocessing.pipeline_05.config_manager import ConfigManager


class PreprocessingPipeline:
    """
    Full-featured preprocessing pipeline with complete customization options.
    
    This is the most powerful interface - suitable for advanced users who
    need full control over the preprocessing workflow.
    
    Features:
    - Complete configuration system
    - Advanced logging and monitoring
    - Batch processing for large datasets
    - Memory-efficient operations
    - Detailed error handling and recovery
    - Performance optimization
    - Progress tracking
    - Data validation at each step
    """
    
    def __init__(self, 
                 config: Optional[ConfigManager] = None,
                 output_dir: str = "data/svm_training",
                 enable_logging: bool = True,
                 log_level: str = "INFO",
                 batch_processing: bool = False,
                 max_memory_usage: float = 0.8,
                 progress_callback: Optional[callable] = None):
        """
        Initialize the full preprocessing pipeline.
        
        Args:
            config: Configuration manager with all settings
            output_dir: Directory for output files
            enable_logging: Whether to enable detailed logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            batch_processing: Enable batch processing for large datasets
            max_memory_usage: Maximum memory usage ratio (0.0-1.0)
            progress_callback: Function to call with progress updates
        """
        
        # Configuration
        self.config = config or ConfigManager()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing options
        self.batch_processing = batch_processing
        self.max_memory_usage = max_memory_usage
        self.progress_callback = progress_callback
        
        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging(log_level)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'start_time': None,
            'end_time': None,
            'processing_time': 0,
            'memory_peak': 0,
            'data_size': 0,
            'steps_completed': []
        }
        
        if self.enable_logging:
            self.logger.info("PreprocessingPipeline initialized")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info(f"Batch processing: {self.batch_processing}")
    
    def _setup_logging(self, log_level: str):
        """Setup detailed logging system"""
        
        self.logger = logging.getLogger('svm_preprocessing')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / 'preprocessing.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _initialize_components(self):
        """Initialize all preprocessing components"""
        
        # Core components
        self.preprocessor = BasePreprocessor()
        self.data_manager = SVMDataManager(str(self.output_dir))
        
        # Feature extractors
        self.extractors = {
            'epcFake': EPCFakeFeatureExtractor(),
            'epcDup': EPCDupFeatureExtractor(),
            'evtOrderErr': EventOrderFeatureExtractor(),
            'locErr': LocationErrorFeatureExtractor(),
            'jump': JumpFeatureExtractor()
        }
        
        # Label generator with custom thresholds
        self.label_generator = RuleBasedLabelGenerator(
            thresholds=self.config.get_thresholds()
        )
        
        if self.enable_logging:
            self.logger.info(f"Initialized {len(self.extractors)} feature extractors")
    
    def process_data(self, 
                    data: Union[pd.DataFrame, str, Path],
                    anomaly_types: Optional[List[str]] = None,
                    save_data: bool = True,
                    validate_input: bool = True,
                    batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process barcode scan data through the complete SVM preprocessing pipeline.
        
        This method provides full control over every aspect of preprocessing:
        1. Input validation and data quality assessment
        2. Data cleaning and standardization
        3. EPC grouping and temporal sorting
        4. Feature extraction for specified anomaly types
        5. Label generation using rule-based scoring
        6. Feature normalization and scaling
        7. Data splitting and class balance handling
        8. Output generation and validation
        
        Args:
            data: Input data (DataFrame, CSV path, or Path object)
            anomaly_types: List of anomaly types to process (default: all)
            save_data: Whether to save processed data to disk
            validate_input: Whether to validate input data quality
            batch_size: Batch size for processing (enables batch mode if set)
            
        Returns:
            Dictionary with complete results for each anomaly type
        """
        
        self.performance_stats['start_time'] = datetime.now()
        
        if self.enable_logging:
            self.logger.info("Starting complete SVM preprocessing pipeline")
        
        # Step 1: Load and validate input data
        df = self._load_and_validate_data(data, validate_input)
        self.performance_stats['data_size'] = len(df)
        
        if self.progress_callback:
            self.progress_callback("Data loading completed", 10)
        
        # Step 2: Data preprocessing
        clean_df, epc_groups = self._preprocess_data(df)
        
        if self.progress_callback:
            self.progress_callback("Data preprocessing completed", 30)
        
        # Step 3: Determine processing strategy
        if batch_size or (self.batch_processing and len(epc_groups) > 10000):
            return self._process_with_batching(epc_groups, anomaly_types, save_data, batch_size)
        else:
            return self._process_standard(epc_groups, anomaly_types, save_data)
    
    def _load_and_validate_data(self, data: Union[pd.DataFrame, str, Path], validate: bool) -> pd.DataFrame:
        """Load data and perform comprehensive validation"""
        
        if self.enable_logging:
            self.logger.info("Loading and validating input data")
        
        # Load data
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.json']:
                df = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
                
            if self.enable_logging:
                self.logger.info(f"Loaded data from {data_path}")
        else:
            df = data.copy()
            if self.enable_logging:
                self.logger.info("Using provided DataFrame")
        
        # Basic validation
        if len(df) == 0:
            if self.enable_logging:
                self.logger.warning("Input data is empty")
            return df
        
        # Detailed validation if requested
        if validate:
            validation_results = self._validate_data_quality(df)
            
            if self.enable_logging:
                self.logger.info(f"Data validation score: {validation_results['score']:.2f}/100")
                if validation_results['issues']:
                    for issue in validation_results['issues']:
                        self.logger.warning(f"Data quality issue: {issue}")
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        
        required_columns = ['epc_code', 'event_time', 'event_type', 'reader_location']
        issues = []
        score = 100
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            score -= 30
        
        # Check data completeness
        for col in required_columns:
            if col in df.columns:
                null_ratio = df[col].isnull().sum() / len(df)
                if null_ratio > 0.1:  # More than 10% null
                    issues.append(f"High null ratio in {col}: {null_ratio:.1%}")
                    score -= 15
        
        # Check EPC format
        if 'epc_code' in df.columns:
            valid_epc_count = df['epc_code'].str.count(r'\.').eq(5).sum()
            valid_ratio = valid_epc_count / len(df)
            if valid_ratio < 0.8:  # Less than 80% valid
                issues.append(f"Low EPC format validity: {valid_ratio:.1%}")
                score -= 20
        
        # Check time format
        if 'event_time' in df.columns:
            try:
                pd.to_datetime(df['event_time'].head(100))
            except:
                issues.append("Invalid time format detected")
                score -= 15
        
        return {
            'score': max(0, score),
            'issues': issues,
            'recommendations': self._generate_recommendations(issues)
        }
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate data quality improvement recommendations"""
        
        recommendations = []
        
        for issue in issues:
            if "Missing required columns" in issue:
                recommendations.append("Ensure input data contains all required columns")
            elif "High null ratio" in issue:
                recommendations.append("Clean data to reduce null values before processing")
            elif "EPC format validity" in issue:
                recommendations.append("Validate and fix EPC code formats")
            elif "Invalid time format" in issue:
                recommendations.append("Standardize timestamp format to YYYY-MM-DD HH:MM:SS")
        
        return recommendations
    
    def _preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Comprehensive data preprocessing"""
        
        if self.enable_logging:
            self.logger.info(f"Preprocessing {len(df)} scan events")
        
        # Clean data
        clean_df = self.preprocessor.clean_data(df)
        
        if self.enable_logging:
            self.logger.info(f"Cleaned to {len(clean_df)} events ({len(clean_df)/len(df):.1%} retained)")
        
        # Group by EPC
        epc_groups = self.preprocessor.group_by_epc(clean_df)
        
        if self.enable_logging:
            self.logger.info(f"Grouped into {len(epc_groups)} unique products")
        
        # Data quality analysis
        quality_report = self.preprocessor.analyze_data_quality(epc_groups)
        
        if self.enable_logging:
            self.logger.info(f"Data quality score: {quality_report['overall_score']:.1f}/100")
            if quality_report['recommendations']:
                for rec in quality_report['recommendations']:
                    self.logger.info(f"Recommendation: {rec}")
        
        self.performance_stats['steps_completed'].append('preprocessing')
        return clean_df, epc_groups
    
    def _process_standard(self, epc_groups: Dict[str, pd.DataFrame], 
                         anomaly_types: Optional[List[str]], 
                         save_data: bool) -> Dict[str, Any]:
        """Standard processing for manageable dataset sizes"""
        
        anomaly_types = anomaly_types or list(self.extractors.keys())
        
        if self.enable_logging:
            self.logger.info(f"Processing {len(anomaly_types)} anomaly types")
        
        results = {}
        
        for i, anomaly_type in enumerate(anomaly_types):
            if self.enable_logging:
                self.logger.info(f"Processing {anomaly_type} ({i+1}/{len(anomaly_types)})")
            
            try:
                result = self._process_single_anomaly_type(epc_groups, anomaly_type, save_data)
                results[anomaly_type] = result
                
                if self.progress_callback:
                    progress = 30 + (60 * (i + 1) / len(anomaly_types))
                    self.progress_callback(f"Completed {anomaly_type}", progress)
                
            except Exception as e:
                if self.enable_logging:
                    self.logger.error(f"Error processing {anomaly_type}: {e}")
                results[anomaly_type] = {'error': str(e)}
        
        # Generate summary
        results['_summary'] = self._generate_summary(results, epc_groups)
        
        self.performance_stats['end_time'] = datetime.now()
        self.performance_stats['processing_time'] = (
            self.performance_stats['end_time'] - self.performance_stats['start_time']
        ).total_seconds()
        
        if self.progress_callback:
            self.progress_callback("Processing completed", 100)
        
        return results
    
    def _process_single_anomaly_type(self, epc_groups: Dict[str, pd.DataFrame], 
                                   anomaly_type: str, save_data: bool) -> Dict[str, Any]:
        """Process a single anomaly type with full pipeline"""
        
        extractor = self.extractors[anomaly_type]
        
        # Extract features
        all_features = []
        all_labels = []
        all_scores = []
        all_epc_codes = []
        
        for epc_code, epc_group in epc_groups.items():
            # Features
            if anomaly_type == 'epcFake':
                features = extractor.extract_features(epc_code)
            else:
                features = extractor.extract_features(epc_group)
            
            # Labels
            labels, scores, epc_codes = self.label_generator.generate_labels(
                {epc_code: epc_group}, anomaly_type
            )
            
            all_features.append(features)
            all_labels.extend(labels)
            all_scores.extend(scores)
            all_epc_codes.extend(epc_codes)
        
        # Convert to arrays
        feature_array = np.array(all_features)
        
        # Save data if requested
        metadata = None
        if save_data:
            metadata = self.data_manager.save_training_data(
                features=feature_array,
                labels=all_labels,
                scores=all_scores,
                epc_codes=all_epc_codes,
                anomaly_type=anomaly_type,
                feature_names=extractor.get_feature_names(),
                test_size=self.config.get_test_size(),
                apply_normalization=True,
                handle_imbalance=self.config.get_handle_imbalance()
            )
        
        # Generate result
        return {
            'features': feature_array,
            'labels': all_labels,
            'scores': all_scores,
            'epc_codes': all_epc_codes,
            'feature_names': extractor.get_feature_names(),
            'summary': {
                'total_samples': len(all_labels),
                'positive_samples': sum(all_labels),
                'negative_samples': len(all_labels) - sum(all_labels),
                'positive_ratio': sum(all_labels) / len(all_labels) if all_labels else 0.0,
                'feature_dimensions': feature_array.shape[1],
                'score_range': (min(all_scores), max(all_scores)) if all_scores else (0, 0)
            },
            'metadata': metadata
        }
    
    def _process_with_batching(self, epc_groups: Dict[str, pd.DataFrame], 
                              anomaly_types: Optional[List[str]], 
                              save_data: bool,
                              batch_size: Optional[int]) -> Dict[str, Any]:
        """Batch processing for large datasets"""
        
        if self.enable_logging:
            self.logger.info(f"Using batch processing for {len(epc_groups)} products")
        
        batch_size = batch_size or 1000
        epc_items = list(epc_groups.items())
        
        # Process in batches
        anomaly_types = anomaly_types or list(self.extractors.keys())
        results = {}
        
        for anomaly_type in anomaly_types:
            if self.enable_logging:
                self.logger.info(f"Batch processing {anomaly_type}")
            
            all_features = []
            all_labels = []
            all_scores = []
            all_epc_codes = []
            
            # Process in batches
            for i in range(0, len(epc_items), batch_size):
                batch_items = epc_items[i:i + batch_size]
                batch_groups = dict(batch_items)
                
                if self.enable_logging:
                    self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(epc_items)-1)//batch_size + 1}")
                
                # Process batch
                batch_result = self._process_single_anomaly_type(batch_groups, anomaly_type, False)
                
                # Accumulate results
                all_features.extend(batch_result['features'])
                all_labels.extend(batch_result['labels'])
                all_scores.extend(batch_result['scores'])
                all_epc_codes.extend(batch_result['epc_codes'])
            
            # Save final result
            feature_array = np.array(all_features)
            
            metadata = None
            if save_data:
                metadata = self.data_manager.save_training_data(
                    features=feature_array,
                    labels=all_labels,
                    scores=all_scores,
                    epc_codes=all_epc_codes,
                    anomaly_type=anomaly_type,
                    feature_names=self.extractors[anomaly_type].get_feature_names()
                )
            
            results[anomaly_type] = {
                'features': feature_array,
                'labels': all_labels,
                'scores': all_scores,
                'epc_codes': all_epc_codes,
                'feature_names': self.extractors[anomaly_type].get_feature_names(),
                'summary': {
                    'total_samples': len(all_labels),
                    'positive_samples': sum(all_labels),
                    'positive_ratio': sum(all_labels) / len(all_labels) if all_labels else 0.0,
                    'feature_dimensions': feature_array.shape[1]
                },
                'metadata': metadata
            }
        
        # Generate summary
        results['_summary'] = self._generate_summary(results, epc_groups)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any], epc_groups: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive processing summary"""
        
        successful_types = [k for k, v in results.items() if k != '_summary' and 'error' not in v]
        failed_types = [k for k, v in results.items() if k != '_summary' and 'error' in v]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'performance': self.performance_stats,
            'data_summary': {
                'total_products_processed': len(epc_groups),
                'total_anomaly_types_requested': len(self.extractors),
                'successful_types': successful_types,
                'failed_types': failed_types
            },
            'configuration': {
                'batch_processing': self.batch_processing,
                'max_memory_usage': self.max_memory_usage,
                'output_directory': str(self.output_dir)
            }
        }
        
        if successful_types:
            total_samples = sum(results[t]['summary']['total_samples'] for t in successful_types)
            total_positive = sum(results[t]['summary']['positive_samples'] for t in successful_types)
            
            summary['aggregate_statistics'] = {
                'total_samples_across_types': total_samples,
                'total_positive_across_types': total_positive,
                'average_positive_ratio': np.mean([results[t]['summary']['positive_ratio'] for t in successful_types]),
                'feature_dimensions_by_type': {t: results[t]['summary']['feature_dimensions'] for t in successful_types}
            }
        
        return summary
    
    def analyze_preprocessing_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze preprocessing results and provide recommendations.
        
        This method provides comprehensive analysis of the preprocessing results,
        including data quality assessment, feature distribution analysis, and
        recommendations for model training.
        """
        
        if self.enable_logging:
            self.logger.info("Analyzing preprocessing results")
        
        analysis = {
            'data_quality': {},
            'feature_analysis': {},
            'recommendations': [],
            'warnings': []
        }
        
        successful_types = [k for k, v in results.items() if k != '_summary' and 'error' not in v]
        
        for anomaly_type in successful_types:
            result = results[anomaly_type]
            
            # Data quality analysis
            positive_ratio = result['summary']['positive_ratio']
            total_samples = result['summary']['total_samples']
            
            quality_score = 100
            issues = []
            
            if total_samples < 100:
                quality_score -= 30
                issues.append("Very small dataset - consider collecting more data")
            elif total_samples < 500:
                quality_score -= 15
                issues.append("Small dataset - results may be less reliable")
            
            if positive_ratio < 0.05:
                quality_score -= 25
                issues.append("Very few positive samples - severe class imbalance")
            elif positive_ratio < 0.1:
                quality_score -= 15
                issues.append("Few positive samples - class imbalance detected")
            
            if positive_ratio > 0.8:
                quality_score -= 20
                issues.append("Too many positive samples - may indicate labeling issues")
            
            analysis['data_quality'][anomaly_type] = {
                'score': max(0, quality_score),
                'issues': issues,
                'sample_count': total_samples,
                'positive_ratio': positive_ratio
            }
            
            # Feature analysis
            features = result['features']
            
            # Check for constant features
            feature_vars = np.var(features, axis=0)
            constant_features = np.sum(feature_vars < 1e-8)
            
            # Check for highly correlated features
            corr_matrix = np.corrcoef(features.T)
            high_corr = np.sum(np.abs(corr_matrix) > 0.95) - features.shape[1]  # Subtract diagonal
            
            analysis['feature_analysis'][anomaly_type] = {
                'feature_dimensions': features.shape[1],
                'constant_features': int(constant_features),
                'highly_correlated_pairs': int(high_corr / 2),  # Divide by 2 for symmetry
                'feature_variance_range': (float(np.min(feature_vars)), float(np.max(feature_vars)))
            }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_analysis_recommendations(analysis)
        
        return analysis
    
    def _generate_analysis_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        
        recommendations = []
        
        for anomaly_type, quality in analysis['data_quality'].items():
            if quality['score'] < 70:
                recommendations.append(f"{anomaly_type}: Consider improving data quality (score: {quality['score']})")
            
            if quality['positive_ratio'] < 0.1:
                recommendations.append(f"{anomaly_type}: Use techniques for handling class imbalance")
            
            if quality['sample_count'] < 500:
                recommendations.append(f"{anomaly_type}: Collect more training data if possible")
        
        for anomaly_type, features in analysis['feature_analysis'].items():
            if features['constant_features'] > 0:
                recommendations.append(f"{anomaly_type}: Remove {features['constant_features']} constant features")
            
            if features['highly_correlated_pairs'] > 5:
                recommendations.append(f"{anomaly_type}: Consider feature selection to reduce correlation")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the full preprocessing pipeline.
    Run this file directly to see it in action!
    """
    
    print("🎓 Full Preprocessing Pipeline Example")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame([
        {
            'epc_code': '001.8804823.1234567.123456.20240101.123456789',
            'event_time': '2024-01-01 10:00:00',
            'event_type': 'production',
            'reader_location': 'Factory_A',
            'business_step': 'Factory'
        },
        {
            'epc_code': '001.8804823.1234567.123456.20240101.123456789',
            'event_time': '2024-01-02 14:00:00',
            'event_type': 'logistics',
            'reader_location': 'Warehouse_B',
            'business_step': 'WMS'
        },
        {
            'epc_code': 'INVALID.FORMAT.EPC',
            'event_time': '2024-01-03 09:00:00',
            'event_type': 'retail',
            'reader_location': 'Store_C',
            'business_step': 'Retail'
        }
    ])
    
    # Progress callback for demonstration
    def progress_callback(message: str, progress: float):
        print(f"Progress: {progress:5.1f}% - {message}")
    
    # Create pipeline with full configuration
    pipeline = PreprocessingPipeline(
        output_dir="temp_full_pipeline",
        enable_logging=True,
        log_level="INFO",
        progress_callback=progress_callback
    )
    
    # Process data
    results = pipeline.process_data(
        sample_data,
        anomaly_types=['epcFake', 'epcDup'],
        save_data=True,
        validate_input=True
    )
    
    # Analyze results
    analysis = pipeline.analyze_preprocessing_results(results)
    
    print("\n📊 Analysis Results:")
    for anomaly_type, quality in analysis['data_quality'].items():
        print(f"  {anomaly_type}: Quality score {quality['score']}/100")
    
    print("\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
    
    # Cleanup
    import shutil
    if os.path.exists("temp_full_pipeline"):
        shutil.rmtree("temp_full_pipeline")
    
    print("\n✅ Full pipeline example completed!")