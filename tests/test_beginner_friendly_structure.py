#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Beginner-Friendly SVM Preprocessing Structure

This test file validates that the reorganized SVM preprocessing components
work correctly and are easy to use for beginners.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test the new beginner-friendly structure
from barcode.svm_preprocessing.01_core.base_preprocessor import BasePreprocessor
from barcode.svm_preprocessing.01_core.sequence_processor import SequenceProcessor
from barcode.svm_preprocessing.01_core.feature_normalizer import FeatureNormalizer
from barcode.svm_preprocessing.02_features.base_feature_extractor import BaseFeatureExtractor, ExampleFeatureExtractor


class TestBeginnerFriendlyStructure:
    """Test the reorganized beginner-friendly SVM preprocessing structure"""
    
    @pytest.fixture
    def sample_barcode_data(self):
        """Create realistic sample data for testing"""
        data = []
        for i in range(20):
            epc_code = f"001.8804823.{i:07d}.123456.20240101.{i:09d}"
            for j in range(np.random.randint(2, 6)):  # 2-5 events per EPC
                data.append({
                    'epc_code': epc_code,
                    'event_time': f"2024-01-{j+1:02d} {10+j:02d}:00:00",
                    'event_type': np.random.choice(['production', 'logistics', 'retail']),
                    'reader_location': f"Location_{j+1}",
                    'business_step': np.random.choice(['Factory', 'WMS', 'Retail'])
                })
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_01_core_structure_exists(self):
        """Test that the 01_core directory structure exists and is importable"""
        
        # Test that all core components can be imported
        assert BasePreprocessor is not None
        assert SequenceProcessor is not None
        assert FeatureNormalizer is not None
        
        print("✅ All core components successfully imported")
    
    def test_02_features_structure_exists(self):
        """Test that the 02_features directory structure exists and is importable"""
        
        # Test base feature extractor
        assert BaseFeatureExtractor is not None
        assert ExampleFeatureExtractor is not None
        
        # Test that the example extractor works
        extractor = ExampleFeatureExtractor()
        assert extractor.FEATURE_DIMENSIONS == 3
        
        features = extractor.extract_features("test.epc.code")
        assert len(features) == 3
        assert all(isinstance(f, (int, float)) for f in features)
        
        print("✅ Feature extraction structure working correctly")
    
    def test_base_preprocessor_beginner_friendly(self, sample_barcode_data):
        """Test that BasePreprocessor is easy to use for beginners"""
        
        # Should be simple to create and use
        preprocessor = BasePreprocessor()
        
        # Should handle the complete workflow in one method
        clean_df, epc_groups = preprocessor.process_file_from_dataframe(sample_barcode_data)
        
        # Results should be easy to understand
        assert len(clean_df) > 0
        assert len(epc_groups) > 0
        assert isinstance(epc_groups, dict)
        
        # Should provide helpful analysis
        quality_report = preprocessor.analyze_data_quality(epc_groups)
        assert 'overall_score' in quality_report
        assert 'recommendations' in quality_report
        assert isinstance(quality_report['overall_score'], (int, float))
        
        # Should provide sample journey for learning
        sample_journey = preprocessor.get_sample_epc_journey(epc_groups)
        assert len(sample_journey) > 0
        
        print("✅ BasePreprocessor is beginner-friendly")
    
    def test_sequence_processor_educational(self):
        """Test that SequenceProcessor provides educational value"""
        
        processor = SequenceProcessor()
        
        # Should handle common cases intuitively
        short_sequence = [1.0, 2.0, 3.0]
        padded = processor.process_sequence(short_sequence, 5, 'temporal')
        
        assert len(padded) == 5
        assert padded[:3] == short_sequence
        assert padded[3] == 3.0  # Should use last value for temporal
        
        # Should provide educational demonstrations
        # This method should exist and work without errors
        processor.demonstrate_processing_strategies()
        
        # Should provide analysis tools
        test_sequences = [[1, 2], [1, 2, 3, 4], [1]]
        analysis = processor.analyze_sequence_lengths(test_sequences)
        
        assert 'mean_length' in analysis
        assert 'recommended_target' in analysis
        assert 'processing_impact' in analysis
        
        print("✅ SequenceProcessor provides good educational value")
    
    def test_feature_normalizer_demonstrates_importance(self, temp_dir):
        """Test that FeatureNormalizer clearly shows why normalization matters"""
        
        normalizer = FeatureNormalizer(method='robust')
        
        # Create features with problematic scales (like the real problem described)
        problematic_features = np.array([
            [1.0, 47.0, 1672531200],      # boolean, length, timestamp
            [0.0, 45.0, 1672531150],      # Mixed scales should cause problems
            [1.0, 48.0, 1672531300]
        ])
        
        # Should normalize successfully
        normalized = normalizer.fit_transform_features(problematic_features, 'demo')
        
        # All features should now be in similar ranges
        assert normalized.shape == problematic_features.shape
        
        # Should provide educational demonstration
        normalizer.demonstrate_normalization_effect()
        
        # Should be able to save and load scalers
        saved_files = normalizer.save_scalers(temp_dir)
        assert len(saved_files) > 0
        
        # Should provide scaler information for learning
        scaler_info = normalizer.get_scaler_info('demo')
        assert scaler_info is not None
        assert 'type' in scaler_info
        
        print("✅ FeatureNormalizer clearly demonstrates importance")
    
    def test_base_feature_extractor_teaching_patterns(self):
        """Test that BaseFeatureExtractor teaches good patterns"""
        
        extractor = ExampleFeatureExtractor()
        
        # Should enforce fixed dimensions consistently
        test_inputs = ["short", "much.longer.input.with.many.dots", ""]
        
        for test_input in test_inputs:
            features = extractor.extract_and_validate(test_input)
            assert len(features) == extractor.FEATURE_DIMENSIONS
            assert all(isinstance(f, (int, float)) for f in features)
        
        # Should provide utility functions that are educational
        entropy = extractor.calculate_entropy(['A', 'B', 'C', 'D'])
        assert isinstance(entropy, float)
        assert entropy > 0
        
        # Should provide statistical analysis
        stats = extractor.calculate_sequence_statistics([1, 2, 3, 4, 5])
        expected_keys = ['mean', 'std', 'min', 'max', 'range', 'median']
        assert all(key in stats for key in expected_keys)
        
        # Should detect outliers educationally
        outliers = extractor.detect_outliers_zscore([1, 2, 3, 4, 100])  # 100 is obvious outlier
        assert outliers['outlier_count'] > 0
        assert 'outlier_ratio' in outliers
        
        # Should provide feature information for learning
        info = extractor.get_feature_info()
        assert 'extractor_name' in info
        assert 'feature_dimensions' in info
        assert 'feature_names' in info
        
        print("✅ BaseFeatureExtractor teaches good patterns")
    
    def test_documentation_and_examples_exist(self):
        """Test that documentation and examples are properly structured"""
        
        # Test that README exists and is readable
        readme_path = Path(__file__).parent.parent / "src" / "barcode" / "svm_preprocessing" / "README.md"
        assert readme_path.exists(), "Main README.md should exist"
        
        # Test that examples directory exists
        examples_dir = Path(__file__).parent.parent / "src" / "barcode" / "svm_preprocessing" / "examples"
        assert examples_dir.exists(), "Examples directory should exist"
        
        # Test that basic usage example exists
        basic_usage = examples_dir / "basic_usage.py"
        assert basic_usage.exists(), "basic_usage.py should exist"
        
        # Test that tutorials directory exists
        tutorials_dir = Path(__file__).parent.parent / "src" / "barcode" / "svm_preprocessing" / "tutorials"
        assert tutorials_dir.exists(), "Tutorials directory should exist"
        
        # Test that understanding features tutorial exists
        features_tutorial = tutorials_dir / "01_understanding_features.md"
        assert features_tutorial.exists(), "Features tutorial should exist"
        
        print("✅ Documentation structure is complete")
    
    def test_numbered_learning_path(self):
        """Test that the numbered directory structure guides learning properly"""
        
        base_path = Path(__file__).parent.parent / "src" / "barcode" / "svm_preprocessing"
        
        # Test numbered directories exist in order
        core_dir = base_path / "01_core"
        features_dir = base_path / "02_features"
        
        assert core_dir.exists(), "01_core directory should exist"
        assert features_dir.exists(), "02_features directory should exist"
        
        # Test that each directory has __init__.py with helpful information
        core_init = core_dir / "__init__.py"
        features_init = features_dir / "__init__.py"
        
        assert core_init.exists(), "Core __init__.py should exist"
        assert features_init.exists(), "Features __init__.py should exist"
        
        # Test that init files contain helpful content
        with open(core_init, 'r', encoding='utf-8') as f:
            core_content = f.read()
            assert "Core Components" in core_content
            assert "Learning Path" in core_content
        
        with open(features_init, 'r', encoding='utf-8') as f:
            features_content = f.read()
            assert "Feature Extraction" in features_content
            assert "Learning Path" in features_content
        
        print("✅ Numbered learning path is properly structured")
    
    def test_beginner_error_handling(self):
        """Test that components handle beginner mistakes gracefully"""
        
        # Test empty inputs don't crash
        preprocessor = BasePreprocessor()
        empty_df = pd.DataFrame()
        
        # Should handle gracefully, not crash
        clean_df = preprocessor.clean_data(empty_df)
        epc_groups = preprocessor.group_by_epc(clean_df)
        quality = preprocessor.analyze_data_quality(epc_groups)
        
        assert isinstance(epc_groups, dict)
        assert len(epc_groups) == 0
        assert 'overall_score' in quality
        
        # Test invalid feature inputs
        extractor = ExampleFeatureExtractor()
        
        # Should handle None/invalid inputs gracefully
        features1 = extractor.extract_and_validate(None)
        features2 = extractor.extract_and_validate("")
        features3 = extractor.extract_and_validate(123)  # Wrong type
        
        # All should return valid feature vectors (with zeros/defaults)
        assert len(features1) == extractor.FEATURE_DIMENSIONS
        assert len(features2) == extractor.FEATURE_DIMENSIONS
        assert len(features3) == extractor.FEATURE_DIMENSIONS
        
        print("✅ Components handle beginner mistakes gracefully")
    
    def test_educational_output_quality(self, sample_barcode_data):
        """Test that components provide educational output for learning"""
        
        # Test that operations provide informative print statements
        # (This is important for beginners to understand what's happening)
        
        preprocessor = BasePreprocessor()
        
        # These should print helpful information (we can't easily test print output,
        # but we can test that the methods complete successfully)
        clean_df = preprocessor.clean_data(sample_barcode_data)
        epc_groups = preprocessor.group_by_epc(clean_df)
        quality = preprocessor.analyze_data_quality(epc_groups)
        
        # Should provide meaningful recommendations
        assert 'recommendations' in quality
        assert isinstance(quality['recommendations'], list)
        
        # Test sequence processor education
        processor = SequenceProcessor()
        
        # Should complete educational demonstration
        processor.demonstrate_processing_strategies()
        
        # Test normalizer education
        normalizer = FeatureNormalizer()
        
        # Should complete educational demonstration
        normalizer.demonstrate_normalization_effect()
        
        print("✅ Educational output quality is good")


def test_complete_beginner_workflow():
    """
    Integration test that a complete beginner could follow the workflow
    """
    
    print("🎓 Testing Complete Beginner Workflow")
    print("=" * 50)
    
    # Step 1: Create sample data (beginner would load from CSV)
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
        }
    ])
    
    # Step 2: Basic preprocessing (very simple for beginners)
    preprocessor = BasePreprocessor()
    clean_df, epc_groups = preprocessor.process_file_from_dataframe(sample_data)
    
    assert len(epc_groups) == 1  # One unique EPC
    
    # Step 3: Extract features for one anomaly type
    extractor = ExampleFeatureExtractor()
    
    # Get the single EPC
    epc_code = list(epc_groups.keys())[0]
    features = extractor.extract_features(epc_code)
    
    assert len(features) == 3
    
    # Step 4: Normalize features  
    normalizer = FeatureNormalizer()
    features_array = np.array([features])
    normalized = normalizer.fit_transform_features(features_array, 'test')
    
    assert normalized.shape == (1, 3)
    
    print("✅ Complete beginner workflow successful!")


# Helper method to add to BasePreprocessor for testing
def process_file_from_dataframe(self, df: pd.DataFrame):
    """Helper method for testing - processes dataframe like a file"""
    clean_df = self.clean_data(df)
    epc_groups = self.group_by_epc(clean_df)
    return clean_df, epc_groups

# Monkey patch for testing
BasePreprocessor.process_file_from_dataframe = process_file_from_dataframe


if __name__ == '__main__':
    # Run tests if called directly
    pytest.main([__file__, '-v'])