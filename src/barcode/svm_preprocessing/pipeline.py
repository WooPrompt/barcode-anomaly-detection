"""
Main SVM Preprocessing Pipeline
Orchestrates the complete preprocessing workflow from raw data to SVM training data
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .base_preprocessor import BasePreprocessor
from .feature_extractors.epc_fake_features import EPCFakeFeatureExtractor
from .feature_extractors.epc_dup_features import EPCDupFeatureExtractor
from .feature_extractors.evt_order_features import EventOrderFeatureExtractor
from .feature_extractors.loc_err_features import LocationErrorFeatureExtractor
from .feature_extractors.jump_features import JumpFeatureExtractor
from .feature_extractors.loc_err_features_event_level import LocationErrorFeatureExtractorEventLevel
from .feature_extractors.jump_features_event_level import JumpFeatureExtractorEventLevel
from .label_generators.rule_based_labels import RuleBasedLabelGenerator
from .data_manager import SVMDataManager


class SVMPreprocessingPipeline:
    """Complete SVM preprocessing pipeline"""
    
    def __init__(self, output_dir: str = "data/svm_training",
                 label_thresholds: Dict[str, int] = None,
                 enable_logging: bool = True):
        
        # Initialize components
        self.base_preprocessor = BasePreprocessor()
        self.data_manager = SVMDataManager(output_dir)
        self.label_generator = RuleBasedLabelGenerator(label_thresholds)
        
        # Initialize feature extractors
        self.feature_extractors = {
            'epcFake': EPCFakeFeatureExtractor(),
            'epcDup': EPCDupFeatureExtractor(),
            'evtOrderErr': EventOrderFeatureExtractor(),
            'locErr': LocationErrorFeatureExtractor(),
            'jump': JumpFeatureExtractor(),
            'locErr_event': LocationErrorFeatureExtractorEventLevel(),
            'jump_event': JumpFeatureExtractorEventLevel()
        }
        
        # Setup logging
        if enable_logging:
            self._setup_logging()
        
        self.logger = logging.getLogger(__name__) if enable_logging else None
    
    def _setup_logging(self):
        """Setup logging for pipeline operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('svm_preprocessing.log'),
                logging.StreamHandler()
            ]
        )
    
    def process_data(self, raw_df: pd.DataFrame, 
                    anomaly_types: List[str] = None,
                    save_data: bool = True,
                    batch_size: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Complete preprocessing pipeline from raw data to SVM training data
        
        Args:
            raw_df: Raw barcode scan data
            anomaly_types: List of anomaly types to process (default: all)
            save_data: Whether to save processed data to disk
            batch_size: Batch size for memory-efficient processing
        
        Returns:
            Dictionary with processed data for each anomaly type
        """
        
        if self.logger:
            self.logger.info(f"Starting SVM preprocessing pipeline with {len(raw_df)} raw records")
        
        # Default to all anomaly types
        if anomaly_types is None:
            anomaly_types = list(self.feature_extractors.keys())
        
        # Phase 1: Base preprocessing
        if self.logger:
            self.logger.info("Phase 1: Base preprocessing - cleaning and grouping data")
        
        df_clean = self.base_preprocessor.preprocess(raw_df)
        if self.logger:
            self.logger.info(f"Cleaned data: {len(df_clean)} records")
        
        epc_groups = self.base_preprocessor.get_epc_groups(df_clean)
        if self.logger:
            self.logger.info(f"Grouped into {len(epc_groups)} EPC sequences")
        
        # Analyze sequence characteristics
        sequence_stats = self.base_preprocessor.analyze_dataset_sequences(df_clean)
        if self.logger:
            self.logger.info(f"Sequence statistics: {sequence_stats}")
        
        # Process each anomaly type
        results = {}
        
        for anomaly_type in anomaly_types:
            if self.logger:
                self.logger.info(f"Processing anomaly type: {anomaly_type}")
            
            try:
                result = self._process_single_anomaly_type(
                    epc_groups, anomaly_type, save_data, batch_size
                )
                results[anomaly_type] = result
                
                if self.logger:
                    self.logger.info(f"Completed {anomaly_type}: {result['summary']}")
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to process {anomaly_type}: {str(e)}")
                results[anomaly_type] = {'error': str(e)}
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(results, sequence_stats)
        results['_summary'] = overall_summary
        
        if self.logger:
            self.logger.info("SVM preprocessing pipeline completed")
            self.logger.info(f"Overall summary: {overall_summary}")
        
        return results
    
    def _process_single_anomaly_type(self, epc_groups: Dict[str, pd.DataFrame],
                                   anomaly_type: str, save_data: bool,
                                   batch_size: Optional[int]) -> Dict[str, Any]:
        """Process a single anomaly type through the complete pipeline"""
        
        # Phase 2: Feature extraction
        extractor = self.feature_extractors[anomaly_type]
        
        if batch_size and len(epc_groups) > batch_size:
            # Memory-efficient batch processing
            features = self._extract_features_batched(
                epc_groups, extractor, anomaly_type, batch_size
            )
        else:
            # Standard processing
            features = self._extract_features_standard(epc_groups, extractor, anomaly_type)
        
        # Phase 3: Label generation
        labels, scores, epc_codes = self.label_generator.generate_labels(
            epc_groups, anomaly_type
        )
        
        # Validate data consistency
        if len(features) != len(labels):
            raise ValueError(f"Feature count ({len(features)}) != label count ({len(labels)})")
        
        # Phase 4: Data analysis and saving
        feature_array = np.array(features)
        result = {
            'features': feature_array,
            'labels': labels,
            'scores': scores,
            'epc_codes': epc_codes,
            'feature_names': extractor.get_feature_names(),
            'summary': {
                'total_samples': len(labels),
                'positive_samples': sum(labels),
                'negative_samples': len(labels) - sum(labels),
                'positive_ratio': sum(labels) / len(labels) if len(labels) > 0 else 0.0,
                'feature_dimensions': feature_array.shape[1],
                'score_range': (min(scores), max(scores)) if scores else (0, 0)
            }
        }
        
        # Save to disk if requested
        if save_data:
            metadata = self.data_manager.save_training_data(
                features=feature_array,
                labels=labels,
                scores=scores,
                epc_codes=epc_codes,
                anomaly_type=anomaly_type,
                feature_names=extractor.get_feature_names()
            )
            result['saved_metadata'] = metadata
        
        return result
    
    def _extract_features_standard(self, epc_groups: Dict[str, pd.DataFrame],
                                 extractor, anomaly_type: str) -> List[List[float]]:
        """Standard feature extraction for all EPC groups"""
        features = []
        
        for epc_code, epc_group in epc_groups.items():
            if "event" in anomaly_type:
                event_features = extractor.extract_features_per_event(epc_group)
                features.extend(event_features)
            elif anomaly_type == 'epcFake':
                # EPC fake only needs the EPC code
                feature_vector = extractor.extract_features(epc_code)
                features.append(feature_vector)
            else:
                # Other anomaly types need the full EPC group
                feature_vector = extractor.extract_features(epc_group)
                features.append(feature_vector)

        return features
    
    def _extract_features_batched(self, epc_groups: Dict[str, pd.DataFrame],
                                extractor, anomaly_type: str, batch_size: int) -> List[List[float]]:
        """Memory-efficient batched feature extraction"""
        features = []
        epc_items = list(epc_groups.items())
        
        for i in range(0, len(epc_items), batch_size):
            batch_items = epc_items[i:i + batch_size]
            batch_features = []
            
            for epc_code, epc_group in batch_items:
                if "event" in anomaly_type:
                    event_features = extractor.extract_features_per_event(epc_group)
                    batch_features.extend(event_features)
                elif anomaly_type == 'epcFake':
                    feature_vector = extractor.extract_features(epc_code)
                    batch_features.append(feature_vector)
                else:
                    # Other anomaly types need the full EPC group
                    feature_vector = extractor.extract_features(epc_group)
                    batch_features.append(feature_vector)

            features.extend(batch_features)
            
            if self.logger:
                self.logger.debug(f"Processed batch {i//batch_size + 1}, features: {len(batch_features)}")
        
        return features
    
    def _generate_overall_summary(self, results: Dict[str, Any], 
                                sequence_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of preprocessing results"""
        successful_types = [k for k, v in results.items() if 'error' not in v]
        failed_types = [k for k, v in results.items() if 'error' in v]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_anomaly_types': len(results),
            'successful_types': successful_types,
            'failed_types': failed_types,
            'sequence_statistics': sequence_stats
        }
        
        if successful_types:
            # Aggregate statistics across successful types
            total_samples = sum(results[t]['summary']['total_samples'] for t in successful_types)
            total_positive = sum(results[t]['summary']['positive_samples'] for t in successful_types)
            
            summary['aggregate_statistics'] = {
                'total_samples_across_types': total_samples,
                'total_positive_across_types': total_positive,
                'avg_positive_ratio': np.mean([results[t]['summary']['positive_ratio'] for t in successful_types]),
                'feature_dimensions_by_type': {t: results[t]['summary']['feature_dimensions'] for t in successful_types}
            }
        
        return summary
    
    def analyze_preprocessing_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed analysis of preprocessing results"""
        analysis = {
            'data_quality': {},
            'class_balance': {},
            'feature_analysis': {},
            'recommendations': []
        }
        
        for anomaly_type, result in results.items():
            if anomaly_type.startswith('_') or 'error' in result:
                continue
            
            # Data quality analysis
            positive_ratio = result['summary']['positive_ratio']
            sample_count = result['summary']['total_samples']
            
            quality_score = 'good'
            if positive_ratio < 0.01 or positive_ratio > 0.99:
                quality_score = 'poor_balance'
            elif sample_count < 100:
                quality_score = 'low_samples'
            elif positive_ratio < 0.05 or positive_ratio > 0.95:
                quality_score = 'imbalanced'
            
            analysis['data_quality'][anomaly_type] = {
                'quality_score': quality_score,
                'sample_count': sample_count,
                'positive_ratio': positive_ratio,
                'imbalance_ratio': (1 - positive_ratio) / positive_ratio if positive_ratio > 0 else float('inf')
            }
            
            # Feature analysis
            if 'features' in result:
                feature_array = result['features']
                analysis['feature_analysis'][anomaly_type] = {
                    'feature_count': feature_array.shape[1],
                    'feature_variance': np.var(feature_array, axis=0).tolist(),
                    'feature_means': np.mean(feature_array, axis=0).tolist(),
                    'zero_variance_features': int(np.sum(np.var(feature_array, axis=0) == 0))
                }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        for anomaly_type, quality in analysis['data_quality'].items():
            if quality['quality_score'] == 'poor_balance':
                recommendations.append(
                    f"{anomaly_type}: Severe class imbalance (ratio: {quality['positive_ratio']:.4f}). "
                    "Consider threshold adjustment or synthetic data generation."
                )
            elif quality['quality_score'] == 'low_samples':
                recommendations.append(
                    f"{anomaly_type}: Low sample count ({quality['sample_count']}). "
                    "Consider collecting more data or using transfer learning."
                )
            elif quality['quality_score'] == 'imbalanced':
                recommendations.append(
                    f"{anomaly_type}: Class imbalance detected. "
                    "Consider using SMOTE or class weighting in SVM training."
                )
        
        for anomaly_type, features in analysis['feature_analysis'].items():
            if features['zero_variance_features'] > 0:
                recommendations.append(
                    f"{anomaly_type}: {features['zero_variance_features']} features have zero variance. "
                    "Consider feature selection or data cleaning."
                )
        
        return recommendations
    
    def optimize_thresholds(self, raw_df: pd.DataFrame, 
                          target_positive_ratios: Dict[str, float] = None) -> Dict[str, int]:
        """Optimize label thresholds to achieve target positive ratios"""
        if target_positive_ratios is None:
            target_positive_ratios = {
                'epcFake': 0.05,    # 5% anomaly rate
                'epcDup': 0.02,     # 2% anomaly rate
                'evtOrderErr': 0.10, # 10% anomaly rate
                'locErr': 0.08,     # 8% anomaly rate
                'jump': 0.03        # 3% anomaly rate
            }
        
        # Get clean EPC groups
        df_clean = self.base_preprocessor.preprocess(raw_df)
        epc_groups = self.base_preprocessor.get_epc_groups(df_clean)
        
        # Optimize each anomaly type
        optimized_thresholds = {}
        
        for anomaly_type, target_ratio in target_positive_ratios.items():
            if anomaly_type in self.feature_extractors:
                # Temporarily set target ratio for optimization
                original_threshold = self.label_generator.thresholds.get(anomaly_type, 50)
                
                # Use label generator's optimization
                threshold_suggestions = self.label_generator.optimize_thresholds(
                    epc_groups, target_ratio
                )
                
                optimized_thresholds[anomaly_type] = threshold_suggestions.get(
                    anomaly_type, original_threshold
                )
        
        if self.logger:
            self.logger.info(f"Optimized thresholds: {optimized_thresholds}")
        
        return optimized_thresholds
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration"""
        return {
            'feature_extractors': {
                name: {
                    'class': extractor.__class__.__name__,
                    'feature_dimensions': extractor.FEATURE_DIMENSIONS,
                    'feature_names': extractor.get_feature_names()
                }
                for name, extractor in self.feature_extractors.items()
            },
            'label_thresholds': self.label_generator.thresholds,
            'data_manager_config': {
                'output_dir': self.data_manager.output_dir,
                'normalization_method': self.data_manager.feature_normalizer.method
            }
        }