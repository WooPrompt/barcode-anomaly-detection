"""
Test Suite for SVM Preprocessing Pipeline
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime, timedelta

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from barcode.svm_preprocessing.pipeline import SVMPreprocessingPipeline
from barcode.svm_preprocessing.base_preprocessor import BasePreprocessor, SequenceProcessor
from barcode.svm_preprocessing.feature_extractors.epc_fake_features import EPCFakeFeatureExtractor
from barcode.svm_preprocessing.feature_extractors.epc_dup_features import EPCDupFeatureExtractor
from barcode.svm_preprocessing.label_generators.rule_based_labels import RuleBasedLabelGenerator
from barcode.svm_preprocessing.data_manager import SVMDataManager
from barcode.svm_preprocessing.config import get_default_config


class TestSVMPreprocessing:
    """Test class for SVM preprocessing components"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample barcode scan data for testing"""
        
        # Create realistic test data
        data = []
        base_time = datetime.now() - timedelta(days=30)
        
        # Generate test EPCs and events
        for i in range(100):
            epc_code = f"001.8804823.1234567.123456.20240101.{i:09d}"
            
            # Generate sequence of events for each EPC
            num_events = np.random.randint(1, 8)  # 1-7 events per EPC
            
            for j in range(num_events):
                event_time = base_time + timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                data.append({
                    'epc_code': epc_code,
                    'event_time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_type': np.random.choice(['production', 'logistics', 'retail']),
                    'reader_location': f"Location_{np.random.randint(1, 20)}",
                    'additional_data': f"data_{j}"
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_base_preprocessor(self, sample_data):
        """Test base preprocessor functionality"""
        preprocessor = BasePreprocessor()
        
        # Test data preprocessing
        df_clean = preprocessor.preprocess(sample_data)
        assert len(df_clean) > 0
        assert 'event_time' in df_clean.columns
        
        # Test EPC grouping
        epc_groups = preprocessor.get_epc_groups(df_clean)
        assert len(epc_groups) > 0
        
        # Verify each group is sorted by time
        for epc_code, group in epc_groups.items():
            assert len(group) > 0
            if len(group) > 1:
                times = pd.to_datetime(group['event_time'])
                assert times.is_monotonic_increasing
    
    def test_sequence_processor(self):
        """Test intelligent sequence processing"""
        processor = SequenceProcessor()
        
        # Test padding
        short_sequence = [1.0, 2.0, 3.0]
        padded = processor.process_sequence(short_sequence, 5, 'temporal')
        assert len(padded) == 5
        assert padded[:3] == short_sequence
        assert padded[3] == padded[4] == 3.0  # Last value padding
        
        # Test truncation
        long_sequence = list(range(20))
        truncated = processor.process_sequence(long_sequence, 10, 'general')
        assert len(truncated) == 10
        
        # Test interpolation for large truncation
        very_long_sequence = list(range(100))
        interpolated = processor.process_sequence(very_long_sequence, 10, 'general')
        assert len(interpolated) == 10
        assert interpolated[0] == 0
        assert interpolated[-1] == 99
    
    def test_epc_fake_feature_extractor(self):
        """Test EPC fake feature extraction"""
        extractor = EPCFakeFeatureExtractor()
        
        # Test valid EPC
        valid_epc = "001.8804823.1234567.123456.20240101.123456789"
        features = extractor.extract_features(valid_epc)
        
        assert len(features) == extractor.FEATURE_DIMENSIONS
        assert all(isinstance(f, (int, float)) for f in features)
        
        # Most features should be 1.0 for valid EPC
        assert features[0] == 1.0  # structure_valid
        assert features[1] == 1.0  # header_valid
        assert features[2] == 1.0  # company_valid
        
        # Test invalid EPC
        invalid_epc = "invalid.epc.format"
        invalid_features = extractor.extract_features(invalid_epc)
        
        assert len(invalid_features) == extractor.FEATURE_DIMENSIONS
        assert invalid_features[0] == 0.0  # structure_valid should be False
    
    def test_epc_dup_feature_extractor(self, sample_data):
        """Test EPC duplicate feature extraction"""
        extractor = EPCDupFeatureExtractor()
        preprocessor = BasePreprocessor()
        
        df_clean = preprocessor.preprocess(sample_data)
        epc_groups = preprocessor.get_epc_groups(df_clean)
        
        # Test with first EPC group
        epc_code, epc_group = next(iter(epc_groups.items()))
        features = extractor.extract_features(epc_group)
        
        assert len(features) == extractor.FEATURE_DIMENSIONS
        assert all(isinstance(f, (int, float)) for f in features)
        
        # Event count should match group size
        assert features[0] == len(epc_group)
        
        # Ratios should be between 0 and 1
        assert 0 <= features[6] <= 1  # location_repetition_ratio
        assert 0 <= features[7] <= 1  # event_type_repetition_ratio
    
    def test_label_generator(self, sample_data):
        """Test rule-based label generation"""
        label_generator = RuleBasedLabelGenerator()
        preprocessor = BasePreprocessor()
        
        df_clean = preprocessor.preprocess(sample_data)
        epc_groups = preprocessor.get_epc_groups(df_clean)
        
        # Test label generation for each anomaly type
        for anomaly_type in ['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump']:
            labels, scores, epc_codes = label_generator.generate_labels(epc_groups, anomaly_type)
            
            assert len(labels) == len(scores) == len(epc_codes)
            assert len(labels) == len(epc_groups)
            
            # All labels should be 0 or 1
            assert all(label in [0, 1] for label in labels)
            
            # All scores should be between 0 and 100
            assert all(0 <= score <= 100 for score in scores)
            
            # EPC codes should match
            assert set(epc_codes) == set(epc_groups.keys())
    
    def test_data_manager(self, sample_data, temp_output_dir):
        """Test data manager functionality"""
        data_manager = SVMDataManager(temp_output_dir)
        
        # Create dummy features and labels
        features = np.random.rand(50, 10)
        labels = [0, 1] * 25
        scores = [float(i) for i in range(50)]
        epc_codes = [f"EPC_{i}" for i in range(50)]
        feature_names = [f"feature_{i}" for i in range(10)]
        
        # Test saving data
        metadata = data_manager.save_training_data(
            features=features,
            labels=labels,
            scores=scores,
            epc_codes=epc_codes,
            anomaly_type='test_anomaly',
            feature_names=feature_names
        )
        
        assert metadata['anomaly_type'] == 'test_anomaly'
        assert metadata['total_samples'] == 50
        assert metadata['feature_dimensions'] == 10
        
        # Test loading data
        loaded_data = data_manager.load_training_data('test_anomaly')
        
        assert loaded_data['X_train'].shape[1] == 10
        assert len(loaded_data['y_train']) > 0
        assert len(loaded_data['epc_mapping']['train']) == len(loaded_data['y_train'])
        assert loaded_data['feature_names'] == feature_names
        
        # Test data integrity validation
        validation = data_manager.validate_data_integrity('test_anomaly')
        assert validation['valid'] == True
        assert len(validation['errors']) == 0
    
    def test_full_pipeline(self, sample_data, temp_output_dir):
        """Test complete preprocessing pipeline"""
        pipeline = SVMPreprocessingPipeline(
            output_dir=temp_output_dir,
            enable_logging=False
        )
        
        # Test processing with minimal data
        results = pipeline.process_data(
            sample_data,
            anomaly_types=['epcFake', 'epcDup'],  # Test subset of anomaly types
            save_data=True
        )
        
        assert '_summary' in results
        assert 'epcFake' in results
        assert 'epcDup' in results
        
        # Check individual results
        for anomaly_type in ['epcFake', 'epcDup']:
            result = results[anomaly_type]
            assert 'features' in result
            assert 'labels' in result
            assert 'scores' in result
            assert 'epc_codes' in result
            assert 'summary' in result
            
            # Verify feature dimensions
            if anomaly_type == 'epcFake':
                assert result['features'].shape[1] == 10
            elif anomaly_type == 'epcDup':
                assert result['features'].shape[1] == 8
        
        # Test analysis
        analysis = pipeline.analyze_preprocessing_results(results)
        assert 'data_quality' in analysis
        assert 'recommendations' in analysis
    
    def test_config_integration(self):
        """Test configuration system integration"""
        config = get_default_config()
        
        # Test config structure
        assert hasattr(config, 'features')
        assert hasattr(config, 'labels')
        assert hasattr(config, 'preprocessing')
        assert hasattr(config, 'performance')
        
        # Test feature dimensions
        assert config.features.epc_fake_dimensions == 10
        assert config.features.epc_dup_dimensions == 8
        assert config.features.evt_order_dimensions == 12
        assert config.features.loc_err_dimensions == 15
        assert config.features.jump_dimensions == 10
        
        # Test thresholds
        assert 0 <= config.labels.epc_fake_threshold <= 100
        assert 0 <= config.labels.epc_dup_threshold <= 100
    
    def test_error_handling(self):
        """Test error handling in various components"""
        
        # Test with empty data
        empty_df = pd.DataFrame()
        preprocessor = BasePreprocessor()
        
        # Should handle empty dataframe gracefully
        df_clean = preprocessor.preprocess(empty_df)
        epc_groups = preprocessor.get_epc_groups(df_clean)
        assert len(epc_groups) == 0
        
        # Test feature extractor with invalid input
        extractor = EPCFakeFeatureExtractor()
        features = extractor.extract_features("")  # Empty EPC
        assert len(features) == extractor.FEATURE_DIMENSIONS
        
        # Test label generator with empty groups
        label_generator = RuleBasedLabelGenerator()
        labels, scores, epc_codes = label_generator.generate_labels({}, 'epcFake')
        assert len(labels) == len(scores) == len(epc_codes) == 0
    
    def test_memory_efficiency(self, temp_output_dir):
        """Test memory-efficient processing features"""
        
        # Create larger dataset for memory testing
        large_data = []
        for i in range(1000):  # 1000 EPCs
            epc_code = f"001.8804823.1234567.123456.20240101.{i:09d}"
            for j in range(3):  # 3 events each
                large_data.append({
                    'epc_code': epc_code,
                    'event_time': f"2024-01-{(j+1):02d} 10:00:00",
                    'event_type': 'production',
                    'reader_location': f"Location_{j+1}"
                })
        
        large_df = pd.DataFrame(large_data)
        
        # Test with batch processing
        pipeline = SVMPreprocessingPipeline(
            output_dir=temp_output_dir,
            enable_logging=False
        )
        
        results = pipeline.process_data(
            large_df,
            anomaly_types=['epcFake'],  # Single type for speed
            save_data=False,
            batch_size=500  # Enable batch processing
        )
        
        assert 'epcFake' in results
        assert results['epcFake']['summary']['total_samples'] == 1000


if __name__ == '__main__':
    # Run tests if called directly
    pytest.main([__file__, '-v'])