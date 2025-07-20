#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV-based SVM Training Pipeline

Trains SVM models using large CSV datasets from data/raw/
Handles memory-efficient processing and periodic retraining.

Author: Data Analysis Team
Date: 2025-07-17
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc="Processing", **kwargs):
        print(f"{desc}...")
        return iterable
import logging

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from csv_data_processor import CSVDataProcessor, process_all_csv_files
from svm_anomaly_detector import SVMAnomalyDetector
from multi_anomaly_detector import detect_anomalies_backend_format

# Create log directory if it doesn't exist
log_dir = 'data/training_logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'svm_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import preprocessing pipeline
try:
    from svm_preprocessing.pipeline import SVMPreprocessingPipeline
    from svm_preprocessing.data_manager import SVMDataManager
    PREPROCESSING_AVAILABLE = True
    logger.info("Advanced preprocessing pipeline available for CSV training")
except ImportError:
    PREPROCESSING_AVAILABLE = False
    logger.warning("Advanced preprocessing not available, using basic processing")

class SVMCSVTrainer:
    """SVM model trainer for large CSV datasets"""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models/svm_models",
                 preprocessing_dir: str = "data/svm_training", event_level_anomalies: List[str] = None):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.preprocessing_dir = preprocessing_dir
        self.processor = CSVDataProcessor(data_dir)
        self.detector = SVMAnomalyDetector(model_dir, preprocessing_dir)
        self.event_level_anomalies = event_level_anomalies or []
        
        # Initialize preprocessing pipeline if available
        if PREPROCESSING_AVAILABLE:
            self.preprocessing_pipeline = SVMPreprocessingPipeline(
                output_dir=preprocessing_dir,
                enable_logging=True
            )
            self.data_manager = SVMDataManager(preprocessing_dir)
            logger.info("Advanced preprocessing pipeline initialized for CSV training")
        else:
            self.preprocessing_pipeline = None
            self.data_manager = None
            logger.warning("Using basic CSV processing")
        
        # Training configuration (will be updated based on GPU availability)
        self.chunk_size = 15000  # Will be optimized automatically
        self.batch_size = 5      # Process 5 chunks at once
        self.max_training_samples = 1000000  # Use more data for better training
        
        # Create directories
        os.makedirs("data/training_logs", exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(preprocessing_dir, exist_ok=True)
        
    def analyze_csv_data(self) -> Dict[str, Any]:
        """Analyze CSV files before training"""
        logger.info("Analyzing CSV data...")
        
        analysis = process_all_csv_files(self.data_dir, self.chunk_size)
        
        # Log analysis results
        stats = analysis['file_stats']
        memory_est = analysis['memory_estimates']
        
        logger.info(f"Found {stats['total_files']} CSV files")
        logger.info(f"Estimated total rows: {stats['estimated_total_rows']:,}")
        logger.info(f"Estimated memory usage: {memory_est['total_memory_mb']:.1f} MB")
        logger.info(f"Recommended chunk size: {memory_est['recommended_chunk_size']:,}")
        
        # Update chunk size based on GPU availability and memory
        recommended_size = memory_est['recommended_chunk_size']
        if hasattr(self.detector, 'use_gpu') and self.detector.use_gpu:
            # Use larger chunks for GPU
            self.chunk_size = max(recommended_size, self.detector.chunk_size)
            logger.info(f"GPU mode: Updated chunk size to {self.chunk_size:,}")
        else:
            # Use recommended size for CPU
            self.chunk_size = recommended_size
            logger.info(f"CPU mode: Updated chunk size to {self.chunk_size:,}")
        
        return analysis

    def train_models_from_csv(self, train_ratio: float = 0.8) -> Dict[str, Any]:
        """Train SVM models from CSV data using advanced preprocessing"""
        start_time = time.time()
        logger.info("Starting SVM model training from CSV data")
        
        if PREPROCESSING_AVAILABLE and self.preprocessing_pipeline:
            return self._train_models_with_preprocessing(train_ratio, start_time)
        else:
            return self._train_models_basic(train_ratio, start_time)

    def _train_models_with_preprocessing(self, train_ratio: float, start_time: float) -> Dict[str, Any]:
        """Train models using advanced preprocessing pipeline with proper train/eval split"""
        logger.info("Using advanced preprocessing pipeline with tt.txt compliant data splitting")
        
        # Step 1: Analyze CSV data
        analysis = self.analyze_csv_data()
        csv_files = analysis['csv_files']
        
        # Step 2: Split data BEFORE preprocessing (tt.txt requirement)
        logger.info("Splitting data into train/eval sets BEFORE preprocessing (tt.txt requirement)")
        train_data, eval_data = self._split_csv_data_for_evaluation(csv_files, train_ratio)
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training data: {len(train_data)} files")
        logger.info(f"  Evaluation data: {len(eval_data)} files (saved for later evaluation)")
        
        # Save evaluation data for later use
        self._save_evaluation_data(eval_data)
        
        # Step 3: Process ONLY training data through preprocessing pipeline
        all_training_data = {}
        total_chunks = 0
        processed_chunks = 0
        
        anomaly_types_to_process = list(self.detector.feature_extractors.keys()) + self.event_level_anomalies

        for chunk_data, chunk_name in tqdm(
            self.processor.process_csv_for_training(train_data, self.chunk_size),
            desc="Processing CSV chunks with advanced preprocessing"
        ):
            total_chunks += 1
            
            try:
                # Convert chunk to DataFrame
                df = pd.DataFrame(chunk_data['data'])
                
                # Ensure required field mappings exist and validate
                if 'location_id' in df.columns:
                    if 'scan_location' not in df.columns:
                        df['scan_location'] = df['location_id'].astype(str)
                    if 'reader_location' not in df.columns:
                        df['reader_location'] = df['location_id'].astype(str)
                    
                    # Validate reader_location field was created successfully
                    assert 'reader_location' in df.columns, "reader_location field not created"
                    assert df['reader_location'].notna().all(), "reader_location contains NaN values"
                    logger.info(f"reader_location field validated: {df['reader_location'].nunique()} unique locations")
                
                # Process through preprocessing pipeline
                preprocessing_results = self.preprocessing_pipeline.process_data(
                    raw_df=df,
                    save_data=True,  # Save training data for later use
                    batch_size=self.chunk_size,
                    anomaly_types=anomaly_types_to_process
                )
                
                # Accumulate results for each anomaly type
                for anomaly_type, result in preprocessing_results.items():
                    if anomaly_type.startswith('_'):  # Skip summary
                        continue
                    
                    if 'error' in result:
                        logger.warning(f"Error in {anomaly_type} for chunk {chunk_name}: {result['error']}")
                        continue
                    
                    if anomaly_type not in all_training_data:
                        all_training_data[anomaly_type] = {
                            'features': [],
                            'labels': [],
                            'scores': [],
                            'epc_codes': []
                        }
                    
                    # Accumulate data
                    if 'features' in result and 'labels' in result:
                        all_training_data[anomaly_type]['features'].extend(result['features'].tolist())
                        all_training_data[anomaly_type]['labels'].extend(result['labels'])
                        all_training_data[anomaly_type]['scores'].extend(result.get('scores', []))
                        all_training_data[anomaly_type]['epc_codes'].extend(result.get('epc_codes', []))
                
                processed_chunks += 1
                
                # Log progress every 10 chunks
                if processed_chunks % 10 == 0:
                    logger.info(f"Processed {processed_chunks}/{total_chunks} chunks with preprocessing")
                    self._log_advanced_training_progress(all_training_data)
                
                # Check data size limit
                current_size = sum(len(data['features']) for data in all_training_data.values())
                if current_size > self.max_training_samples:
                    logger.info(f"Reached max training samples ({self.max_training_samples:,}), stopping")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_name} with preprocessing: {e}")
                continue
        
        # Step 4: Train SVM models with preprocessed data
        logger.info("Training SVM models with preprocessed features...")
        training_results = self.detector.train_models(all_training_data)
        
        # Log final preprocessing statistics
        if 'preprocessing_results' in locals() and preprocessing_results and '_summary' in preprocessing_results:
            summary = preprocessing_results['_summary']
            logger.info(f"Preprocessing summary: {summary}")
        
        return self._finalize_training_results(training_results, train_data, analysis, start_time)
    
    def _train_models_basic(self, train_ratio: float, start_time: float) -> Dict[str, Any]:
        """Train models using basic feature extraction (fallback)"""
        logger.warning("Using basic feature extraction for CSV training")
        
        # Step 1: Analyze CSV data
        analysis = self.analyze_csv_data()
        csv_files = analysis['csv_files']
        
        # Step 2: Use all CSV files for training (no train/validation split)
        train_files = csv_files  # Use all files
        logger.info(f"Using all {len(train_files)} CSV files for training: {[os.path.basename(f) for f in train_files]}")
        
        # Step 3: Extract training features
        training_data = self.extract_training_features_from_csv(train_files)
        
        # Step 4: Train SVM models
        logger.info("Training SVM models...")
        training_results = self.detector.train_models(training_data)
        
        # Step 5: Training completed (no validation needed)
        validation_results = {}
        
        # Step 6: Save training metadata
        training_metadata = {
            'training_start': datetime.fromtimestamp(start_time).isoformat(),
            'training_end': datetime.now().isoformat(),
            'training_duration_minutes': (time.time() - start_time) / 60,
            'csv_files_used': [os.path.basename(f) for f in train_files],
            'validation_files': [],  # No validation split
            'chunk_size': self.chunk_size,
            'training_results': training_results,
            'validation_results': validation_results,
            'file_analysis': analysis['file_stats']
        }
        
        self._save_training_metadata(training_metadata)
        
        logger.info(f"Training completed in {training_metadata['training_duration_minutes']:.1f} minutes")
        return training_metadata

    def extract_training_features_from_csv(self, csv_files: List[str]) -> Dict[str, Dict]:
        """Extract features and labels from CSV files using rule-based detection"""
        logger.info(f"Extracting training features from {len(csv_files)} CSV files")
        
        # Initialize training data collectors
        training_data = {
            'epcFake': {'features': [], 'labels': []},
            'epcDup': {'features': [], 'labels': []},
            'locErr': {'features': [], 'labels': []},
            'evtOrderErr': {'features': [], 'labels': []},
            'jump': {'features': [], 'labels': []}
        }
        
        total_chunks = 0
        processed_chunks = 0
        
        # Process CSV files in chunks
        for chunk_data, chunk_name in tqdm(
            self.processor.process_csv_for_training(csv_files, self.chunk_size),
            desc="Processing CSV chunks"
        ):
            total_chunks += 1
            
            try:
                # Convert chunk to JSON for rule-based detection
                json_str = json.dumps(chunk_data)
                
                # Get rule-based detection results for labels
                rule_results = detect_anomalies_backend_format(json_str)
                rule_dict = json.loads(rule_results)
                
                # Convert to DataFrame for feature extraction
                df = pd.DataFrame(chunk_data['data'])
                if 'location_id' in df.columns:
                    df['reader_location'] = df['location_id'].astype(str)
                    
                    # Validate reader_location field was created successfully
                    assert 'reader_location' in df.columns, "reader_location field not created"
                    assert df['reader_location'].notna().all(), "reader_location contains NaN values"
                
                # Extract features using SVM detector
                self.detector._extract_training_features(df, rule_dict, training_data)
                
                processed_chunks += 1
                
                # Log progress every 10 chunks
                if processed_chunks % 10 == 0:
                    logger.info(f"Processed {processed_chunks}/{total_chunks} chunks")
                    self._log_training_progress(training_data)
                
                # Allow more training data for better model performance
                current_size = self._check_training_data_size(training_data)
                if current_size > self.max_training_samples:
                    logger.info(f"Reached max training samples ({self.max_training_samples:,}), stopping data collection")
                    logger.info(f"Current training data size: {current_size:,} samples")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_name}: {e}")
                continue
        
        # Final statistics
        logger.info(f"Feature extraction completed: {processed_chunks}/{total_chunks} chunks processed")
        self._log_final_training_stats(training_data)
        
        return training_data

    def _finalize_training_results(self, training_results: Dict, train_files: List, 
                                 analysis: Dict, start_time: float) -> Dict[str, Any]:
        """Finalize and save training results"""
        validation_results = {}
        
        # Save training metadata
        training_metadata = {
            'training_start': datetime.fromtimestamp(start_time).isoformat(),
            'training_end': datetime.now().isoformat(),
            'training_duration_minutes': (time.time() - start_time) / 60,
            'csv_files_used': [os.path.basename(f) for f in train_files],
            'validation_files': [],  # No validation split
            'chunk_size': self.chunk_size,
            'training_results': training_results,
            'validation_results': validation_results,
            'file_analysis': analysis['file_stats'],
            'preprocessing_used': PREPROCESSING_AVAILABLE
        }
        
        self._save_training_metadata(training_metadata)
        
        logger.info(f"Training completed in {training_metadata['training_duration_minutes']:.1f} minutes")
        return {
            'training_results': training_results,
            'validation_results': validation_results,
            'metadata': training_metadata
        }
    
    def _check_training_data_size(self, training_data: Dict) -> int:
        """Check current size of training data"""
        total_size = 0
        for data in training_data.values():
            total_size += len(data['labels'])
        return total_size
    
    def _split_csv_data_for_evaluation(self, csv_files: List[str], train_ratio: float) -> Tuple[List[str], List[str]]:
        """Split CSV files into training and evaluation sets (tt.txt requirement)"""
        import random
        
        logger.info(f"Splitting {len(csv_files)} CSV files with ratio {train_ratio:.2f}")
        
        # Create a copy of the file list to shuffle
        files_copy = csv_files.copy()
        random.shuffle(files_copy)
        
        # Calculate split point
        train_count = int(len(files_copy) * train_ratio)
        
        # Split the files
        train_files = files_copy[:train_count]
        eval_files = files_copy[train_count:]
        
        logger.info(f"Data split results:")
        logger.info(f"  Training files ({len(train_files)}): {[os.path.basename(f) for f in train_files]}")
        logger.info(f"  Evaluation files ({len(eval_files)}): {[os.path.basename(f) for f in eval_files]}")
        
        return train_files, eval_files
    
    def _save_evaluation_data(self, eval_files: List[str]):
        """Save evaluation data filenames for later use (tt.txt requirement)"""
        eval_metadata = {
            "evaluation_files": [os.path.basename(f) for f in eval_files],
            "evaluation_file_paths": eval_files,
            "saved_at": datetime.now().isoformat(),
            "purpose": "tt.txt compliant evaluation - data not used in training"
        }
        
        eval_metadata_path = os.path.join(self.model_dir, "evaluation_data_metadata.json")
        with open(eval_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(eval_metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation data metadata saved: {eval_metadata_path}")
        logger.info(f"Evaluation files reserved for testing: {len(eval_files)} files")
    
    def _save_training_metadata(self, metadata: Dict):
        """Save training metadata to file"""
        metadata_path = os.path.join(self.model_dir, "csv_training_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training metadata saved: {metadata_path}")
    
    def retrain_models(self) -> Dict[str, Any]:
        """Retrain models with fresh data (for periodic retraining)"""
        logger.info("Starting periodic model retraining")
        
        # Load previous training metadata if exists
        prev_metadata_path = os.path.join(self.model_dir, "csv_training_metadata.json")
        if os.path.exists(prev_metadata_path):
            with open(prev_metadata_path, 'r', encoding='utf-8') as f:
                prev_metadata = json.load(f)
            logger.info(f"Previous training: {prev_metadata.get('training_end', 'Unknown')}")
        
        # Train with current data
        return self.train_models_from_csv()