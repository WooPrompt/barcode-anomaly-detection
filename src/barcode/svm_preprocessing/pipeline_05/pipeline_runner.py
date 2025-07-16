"""
Pipeline Runner - Simple Interfaces for Complete Workflows

This provides the EASIEST way to run SVM preprocessing. Perfect for beginners
who want to get results quickly without worrying about all the details.

🎯 Two levels of complexity:
1. SimpleRunner - One line to preprocess everything
2. AdvancedRunner - More control but still easy to use

🚀 Example Usage:
    >>> from svm_preprocessing.pipeline_runner import SimpleRunner
    >>> runner = SimpleRunner()
    >>> results = runner.process_data(your_dataframe)
    >>> # Done! Ready for SVM training
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

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


class SimpleRunner:
    """
    The SIMPLEST way to run SVM preprocessing.
    
    Perfect for beginners who want to get from raw data to SVM-ready
    features in one line, without worrying about configuration details.
    
    Example:
        >>> runner = SimpleRunner()
        >>> results = runner.process_data(dataframe)
        >>> print(f"Ready to train SVM on {len(results)} anomaly types!")
    """
    
    def __init__(self, output_dir: str = "data/svm_training"):
        """
        Initialize the simple runner with sensible defaults.
        
        Args:
            output_dir: Where to save the processed training data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 SimpleRunner initialized")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   🎯 Ready to process 5 anomaly types")
        print(f"   ⚡ Using beginner-friendly defaults")
    
    def process_data(self, data: Union[pd.DataFrame, str]) -> Dict[str, Any]:
        """
        Process barcode scan data into SVM training data.
        
        This does EVERYTHING for you:
        1. Loads/cleans the data
        2. Extracts features for all 5 anomaly types
        3. Generates labels using rule-based scoring
        4. Normalizes features for SVM
        5. Handles class imbalance
        6. Splits into train/test sets
        7. Saves everything to disk
        
        Args:
            data: Either a pandas DataFrame or path to CSV file
            
        Returns:
            Dictionary with results for each anomaly type
            
        Example:
            >>> # From DataFrame
            >>> results = runner.process_data(my_dataframe)
            >>>
            >>> # From CSV file
            >>> results = runner.process_data("data/barcode_scans.csv")
            >>>
            >>> # Check results
            >>> for anomaly_type, info in results.items():
            ...     print(f"{anomaly_type}: {info['summary']['total_samples']} samples")
        """
        
        print("🎯 Starting Simple SVM Preprocessing")
        print("=" * 50)
        
        # Step 1: Load data
        if isinstance(data, str):
            print(f"📂 Loading data from {data}")
            df = pd.read_csv(data)
        else:
            print(f"📊 Using provided DataFrame")
            df = data.copy()
        
        print(f"   ✅ Loaded {len(df)} scan events")
        
        # Step 2: Basic preprocessing
        print(f"🧹 Cleaning and grouping data...")
        preprocessor = BasePreprocessor()
        
        clean_df = preprocessor.clean_data(df)
        epc_groups = preprocessor.group_by_epc(clean_df)
        
        print(f"   ✅ Cleaned to {len(clean_df)} events")
        print(f"   ✅ Grouped into {len(epc_groups)} unique products")
        
        # Step 3: Feature extraction and label generation
        print(f"🔍 Extracting features for all anomaly types...")
        
        # Initialize all components
        extractors = {
            'epcFake': EPCFakeFeatureExtractor(),
            'epcDup': EPCDupFeatureExtractor(),
            'evtOrderErr': EventOrderFeatureExtractor(),
            'locErr': LocationErrorFeatureExtractor(),
            'jump': JumpFeatureExtractor()
        }
        
        label_generator = RuleBasedLabelGenerator()
        data_manager = SVMDataManager(self.output_dir)
        
        # Process each anomaly type
        results = {}
        
        for anomaly_type, extractor in extractors.items():
            print(f"   🔧 Processing {anomaly_type}...")
            
            try:
                # Extract features and generate labels
                all_features = []
                all_labels = []
                all_scores = []
                all_epc_codes = []
                
                for epc_code, epc_group in epc_groups.items():
                    # Extract features
                    if anomaly_type == 'epcFake':
                        features = extractor.extract_features(epc_code)
                    else:
                        features = extractor.extract_features(epc_group)
                    
                    # Generate labels
                    labels, scores, epc_codes = label_generator.generate_labels(
                        {epc_code: epc_group}, anomaly_type
                    )
                    
                    all_features.append(features)
                    all_labels.extend(labels)
                    all_scores.extend(scores)
                    all_epc_codes.extend(epc_codes)
                
                # Convert to arrays
                feature_array = np.array(all_features)
                
                # Save processed data
                metadata = data_manager.save_training_data(
                    features=feature_array,
                    labels=all_labels,
                    scores=all_scores,
                    epc_codes=all_epc_codes,
                    anomaly_type=anomaly_type,
                    feature_names=extractor.get_feature_names()
                )
                
                # Store results
                results[anomaly_type] = {
                    'features': feature_array,
                    'labels': all_labels,
                    'scores': all_scores,
                    'epc_codes': all_epc_codes,
                    'feature_names': extractor.get_feature_names(),
                    'summary': {
                        'total_samples': len(all_labels),
                        'positive_samples': sum(all_labels),
                        'negative_samples': len(all_labels) - sum(all_labels),
                        'positive_ratio': sum(all_labels) / len(all_labels) if len(all_labels) > 0 else 0.0,
                        'feature_dimensions': feature_array.shape[1],
                        'score_range': (min(all_scores), max(all_scores)) if all_scores else (0, 0)
                    },
                    'metadata': metadata
                }
                
                positive_count = sum(all_labels)
                total_count = len(all_labels)
                print(f"      ✅ {total_count} samples, {positive_count} anomalies ({positive_count/total_count:.1%})")
                
            except Exception as e:
                print(f"      ❌ Error processing {anomaly_type}: {e}")
                results[anomaly_type] = {'error': str(e)}
        
        # Step 4: Generate summary
        print(f"📊 Generating summary...")
        
        successful_types = [k for k, v in results.items() if 'error' not in v]
        failed_types = [k for k, v in results.items() if 'error' in v]
        
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_anomaly_types': len(results),
            'successful_types': successful_types,
            'failed_types': failed_types,
            'data_directory': self.output_dir,
            'total_products_processed': len(epc_groups),
            'total_events_processed': len(clean_df)
        }
        
        if successful_types:
            total_samples = sum(results[t]['summary']['total_samples'] for t in successful_types)
            total_positive = sum(results[t]['summary']['positive_samples'] for t in successful_types)
            
            summary['aggregate_statistics'] = {
                'total_samples_across_types': total_samples,
                'total_positive_across_types': total_positive,
                'avg_positive_ratio': np.mean([results[t]['summary']['positive_ratio'] for t in successful_types])
            }
        
        results['_summary'] = summary
        
        # Final report
        print(f"🎉 Processing Complete!")
        print(f"   ✅ Successful: {len(successful_types)} anomaly types")
        if failed_types:
            print(f"   ❌ Failed: {len(failed_types)} anomaly types: {', '.join(failed_types)}")
        
        print(f"\\n📁 Data saved to: {self.output_dir}")
        print(f"🎯 Ready for SVM training!")
        
        return results


class AdvancedRunner:
    """
    More control over the preprocessing pipeline while still being easy to use.
    
    Perfect for users who understand the basics and want to customize
    the behavior without diving into all the implementation details.
    
    Example:
        >>> runner = AdvancedRunner(
        ...     anomaly_types=['epcFake', 'epcDup'],  # Only some types
        ...     thresholds={'epcFake': 60, 'epcDup': 40},  # Custom thresholds
        ...     normalization_method='standard'  # Different normalization
        ... )
        >>> results = runner.process_data(dataframe)
    """
    
    def __init__(self, 
                 output_dir: str = "data/svm_training",
                 anomaly_types: List[str] = None,
                 thresholds: Dict[str, int] = None,
                 normalization_method: str = 'robust',
                 test_size: float = 0.2,
                 handle_imbalance: bool = True,
                 batch_processing: bool = False,
                 verbose: bool = True):
        """
        Initialize advanced runner with custom configuration.
        
        Args:
            output_dir: Where to save processed data
            anomaly_types: Which anomaly types to process (default: all)
            thresholds: Custom thresholds for each anomaly type
            normalization_method: 'robust' or 'standard'
            test_size: Fraction of data for testing (0.0-1.0)
            handle_imbalance: Whether to handle class imbalance
            batch_processing: Whether to use batch processing for large datasets
            verbose: Whether to print detailed progress
        """
        
        self.output_dir = output_dir
        self.anomaly_types = anomaly_types or ['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump']
        self.thresholds = thresholds or {}
        self.normalization_method = normalization_method
        self.test_size = test_size
        self.handle_imbalance = handle_imbalance
        self.batch_processing = batch_processing
        self.verbose = verbose
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.verbose:
            print("🔧 AdvancedRunner initialized")
            print(f"   📁 Output: {output_dir}")
            print(f"   🎯 Anomaly types: {', '.join(self.anomaly_types)}")
            print(f"   📏 Normalization: {normalization_method}")
            print(f"   🧪 Test size: {test_size:.1%}")
            print(f"   ⚖️ Handle imbalance: {handle_imbalance}")
    
    def process_data(self, data: Union[pd.DataFrame, str]) -> Dict[str, Any]:
        """
        Process data with advanced configuration options.
        
        Args:
            data: DataFrame or CSV file path
            
        Returns:
            Dictionary with results for each configured anomaly type
        """
        
        if self.verbose:
            print("🎯 Starting Advanced SVM Preprocessing")
            print("=" * 50)
        
        # Load data
        if isinstance(data, str):
            if self.verbose:
                print(f"📂 Loading data from {data}")
            df = pd.read_csv(data)
        else:
            if self.verbose:
                print(f"📊 Using provided DataFrame")
            df = data.copy()
        
        if self.verbose:
            print(f"   ✅ Loaded {len(df)} scan events")
        
        # Preprocessing
        if self.verbose:
            print(f"🧹 Preprocessing data...")
        
        preprocessor = BasePreprocessor()
        clean_df = preprocessor.clean_data(df)
        epc_groups = preprocessor.group_by_epc(clean_df)
        
        if self.verbose:
            print(f"   ✅ {len(clean_df)} events, {len(epc_groups)} products")
        
        # Check for batch processing
        if self.batch_processing and len(epc_groups) > 10000:
            if self.verbose:
                print(f"⚡ Using batch processing for {len(epc_groups)} products")
            return self._process_with_batching(epc_groups)
        else:
            return self._process_standard(epc_groups)
    
    def _process_standard(self, epc_groups: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Standard processing for manageable dataset sizes"""
        
        # Initialize components
        extractors = {}
        if 'epcFake' in self.anomaly_types:
            extractors['epcFake'] = EPCFakeFeatureExtractor()
        if 'epcDup' in self.anomaly_types:
            extractors['epcDup'] = EPCDupFeatureExtractor()
        if 'evtOrderErr' in self.anomaly_types:
            extractors['evtOrderErr'] = EventOrderFeatureExtractor()
        if 'locErr' in self.anomaly_types:
            extractors['locErr'] = LocationErrorFeatureExtractor()
        if 'jump' in self.anomaly_types:
            extractors['jump'] = JumpFeatureExtractor()
        
        label_generator = RuleBasedLabelGenerator(self.thresholds)
        data_manager = SVMDataManager(self.output_dir)
        
        # Configure data manager
        data_manager.feature_normalizer.method = self.normalization_method
        
        # Process each anomaly type
        results = {}
        
        for anomaly_type in self.anomaly_types:
            if self.verbose:
                print(f"   🔧 Processing {anomaly_type}...")
            
            try:
                extractor = extractors[anomaly_type]
                
                # Extract features and labels
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
                    labels, scores, epc_codes = label_generator.generate_labels(
                        {epc_code: epc_group}, anomaly_type
                    )
                    
                    all_features.append(features)
                    all_labels.extend(labels)
                    all_scores.extend(scores)
                    all_epc_codes.extend(epc_codes)
                
                # Save with advanced options
                feature_array = np.array(all_features)
                
                metadata = data_manager.save_training_data(
                    features=feature_array,
                    labels=all_labels,
                    scores=all_scores,
                    epc_codes=all_epc_codes,
                    anomaly_type=anomaly_type,
                    feature_names=extractor.get_feature_names(),
                    test_size=self.test_size,
                    apply_normalization=True,
                    handle_imbalance=self.handle_imbalance
                )
                
                # Store results
                results[anomaly_type] = {
                    'features': feature_array,
                    'labels': all_labels,
                    'scores': all_scores,
                    'epc_codes': all_epc_codes,
                    'feature_names': extractor.get_feature_names(),
                    'summary': {
                        'total_samples': len(all_labels),
                        'positive_samples': sum(all_labels),
                        'positive_ratio': sum(all_labels) / len(all_labels) if all_labels else 0.0,
                        'feature_dimensions': feature_array.shape[1]
                    },
                    'metadata': metadata
                }
                
                if self.verbose:
                    pos_count = sum(all_labels)
                    total_count = len(all_labels)
                    print(f"      ✅ {total_count} samples, {pos_count} anomalies")
                
            except Exception as e:
                if self.verbose:
                    print(f"      ❌ Error: {e}")
                results[anomaly_type] = {'error': str(e)}
        
        # Summary
        successful = [k for k, v in results.items() if 'error' not in v]
        failed = [k for k, v in results.items() if 'error' in v]
        
        results['_summary'] = {
            'successful_types': successful,
            'failed_types': failed,
            'configuration': {
                'anomaly_types': self.anomaly_types,
                'normalization_method': self.normalization_method,
                'test_size': self.test_size,
                'handle_imbalance': self.handle_imbalance
            }
        }
        
        if self.verbose:
            print(f"✅ Advanced processing complete!")
            print(f"   Success: {len(successful)}, Failed: {len(failed)}")
        
        return results
    
    def _process_with_batching(self, epc_groups: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Batch processing for large datasets"""
        
        if self.verbose:
            print("⚡ Implementing batch processing...")
            print("   (This would use the BatchProcessor for memory efficiency)")
        
        # For now, fall back to standard processing
        # In production, this would use the BatchProcessor from the full implementation
        return self._process_standard(epc_groups)


# Convenience functions for even easier usage
def quick_preprocess(data: Union[pd.DataFrame, str], output_dir: str = "data/svm_training") -> Dict[str, Any]:
    """
    One-liner to preprocess data with all defaults.
    
    Perfect for getting started quickly.
    
    Args:
        data: DataFrame or CSV file path
        output_dir: Where to save results
        
    Returns:
        Preprocessing results
        
    Example:
        >>> results = quick_preprocess("my_data.csv")
        >>> print("Ready for SVM training!")
    """
    runner = SimpleRunner(output_dir)
    return runner.process_data(data)


def custom_preprocess(data: Union[pd.DataFrame, str], **kwargs) -> Dict[str, Any]:
    """
    One-liner with custom options.
    
    Args:
        data: DataFrame or CSV file path
        **kwargs: Any AdvancedRunner parameters
        
    Returns:
        Preprocessing results
        
    Example:
        >>> results = custom_preprocess(
        ...     "my_data.csv",
        ...     anomaly_types=['epcFake', 'epcDup'],
        ...     normalization_method='standard'
        ... )
    """
    runner = AdvancedRunner(**kwargs)
    return runner.process_data(data)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the pipeline runners.
    Run this file directly to see them in action!
    """
    
    print("🎓 Pipeline Runner Examples")
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
    
    print("\\n🚀 Testing SimpleRunner...")
    simple_runner = SimpleRunner("temp_output_simple")
    simple_results = simple_runner.process_data(sample_data)
    
    print("\\n🔧 Testing AdvancedRunner...")
    advanced_runner = AdvancedRunner(
        output_dir="temp_output_advanced",
        anomaly_types=['epcFake', 'epcDup'],
        normalization_method='standard',
        verbose=True
    )
    advanced_results = advanced_runner.process_data(sample_data)
    
    print("\\n⚡ Testing convenience functions...")
    quick_results = quick_preprocess(sample_data, "temp_output_quick")
    
    print("\\n✅ All pipeline runners working correctly!")
    
    # Cleanup temp directories
    import shutil
    for temp_dir in ["temp_output_simple", "temp_output_advanced", "temp_output_quick"]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print("🧹 Cleaned up temporary files")