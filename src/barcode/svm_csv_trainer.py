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
                 preprocessing_dir: str = "data/svm_training"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.preprocessing_dir = preprocessing_dir
        self.processor = CSVDataProcessor(data_dir)
        self.detector = SVMAnomalyDetector(model_dir, preprocessing_dir)
        
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
            logger.info(f"🎮 GPU mode: Updated chunk size to {self.chunk_size:,}")
        else:
            # Use recommended size for CPU
            self.chunk_size = recommended_size
            logger.info(f"💻 CPU mode: Updated chunk size to {self.chunk_size:,}")
        
        return analysis
    
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
        logger.info(f"  Training data: {len(train_data)} chunks")
        logger.info(f"  Evaluation data: {len(eval_data)} chunks (saved for later evaluation)")
        
        # Save evaluation data for later use
        self._save_evaluation_data(eval_data)
        
        # Step 3: Process ONLY training data through preprocessing pipeline
        all_training_data = {}
        total_chunks = 0
        processed_chunks = 0
        
        for chunk_data, chunk_name in tqdm(
            self.processor.process_csv_for_training(train_data, self.chunk_size),
            desc="Processing CSV chunks with advanced preprocessing"
        ):
            total_chunks += 1
            
            try:
                # Convert chunk to DataFrame
                df = pd.DataFrame(chunk_data['data'])
                
                # Add scan_location mapping if location_id exists
                if 'location_id' in df.columns and 'scan_location' not in df.columns:
                    df['scan_location'] = df['location_id'].astype(str)
                
                # Process through preprocessing pipeline
                preprocessing_results = self.preprocessing_pipeline.process_data(
                    raw_df=df,
                    save_data=True,  # Save training data for later use
                    batch_size=self.chunk_size
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
        if preprocessing_results and '_summary' in preprocessing_results:
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
    
    def _validate_models(self, val_files: List[str]) -> Dict[str, Any]:
        """Validate trained models on validation data"""
        logger.info(f"Validating models on {len(val_files)} validation files")
        
        validation_results = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_prediction_time': 0,
            'anomaly_detection_rates': {}
        }
        
        prediction_times = []
        total_anomalies = {'epcFake': 0, 'epcDup': 0, 'locErr': 0, 'evtOrderErr': 0, 'jump': 0}
        
        # Sample validation data (don't process all validation files to save time)
        sample_size = min(3, len(val_files))
        sample_files = val_files[:sample_size]
        
        for chunk_data, chunk_name in self.processor.process_csv_for_training(sample_files, self.chunk_size):
            try:
                # Predict using SVM models
                start_time = time.time()
                json_str = json.dumps(chunk_data)
                result_json = self.detector.predict_anomalies(json_str)
                prediction_time = time.time() - start_time
                
                prediction_times.append(prediction_time)
                validation_results['total_predictions'] += 1
                validation_results['successful_predictions'] += 1
                
                # Count detected anomalies
                result = json.loads(result_json)
                for epc_stat in result.get('epcAnomalyStats', []):
                    total_anomalies['epcFake'] += epc_stat['epcFakeCount']
                    total_anomalies['epcDup'] += epc_stat['epcDupCount']
                    total_anomalies['locErr'] += epc_stat['locErrCount']
                    total_anomalies['evtOrderErr'] += epc_stat['evtOrderErrCount']
                    total_anomalies['jump'] += epc_stat['jumpCount']
                
                # Limit validation to prevent long runtime
                if validation_results['total_predictions'] >= 10:
                    break
                    
            except Exception as e:
                logger.error(f"Validation error on {chunk_name}: {e}")
                validation_results['total_predictions'] += 1
                continue
        
        # Calculate validation metrics
        if prediction_times:
            validation_results['average_prediction_time'] = np.mean(prediction_times)
            validation_results['max_prediction_time'] = np.max(prediction_times)
        
        validation_results['anomaly_detection_rates'] = total_anomalies
        
        logger.info(f"Validation completed: {validation_results['successful_predictions']}/{validation_results['total_predictions']} successful")
        logger.info(f"Average prediction time: {validation_results.get('average_prediction_time', 0):.3f} seconds")
        
        return validation_results
    
    def _log_training_progress(self, training_data: Dict):
        """Log current training data statistics"""
        for anomaly_type, data in training_data.items():
            normal_count = sum(1 for label in data['labels'] if label == 0)
            anomaly_count = sum(1 for label in data['labels'] if label == 1)
            logger.debug(f"{anomaly_type}: {normal_count} normal, {anomaly_count} anomaly samples")
    
    def _log_final_training_stats(self, training_data: Dict):
        """Log final training data statistics"""
        logger.info("Final training data statistics:")
        total_samples = 0
        for anomaly_type, data in training_data.items():
            normal_count = sum(1 for label in data['labels'] if label == 0)
            anomaly_count = sum(1 for label in data['labels'] if label == 1)
            total_count = len(data['labels'])
            total_samples += total_count
            
            logger.info(f"  {anomaly_type}: {total_count} total ({normal_count} normal, {anomaly_count} anomaly)")
        
        logger.info(f"Total training samples across all models: {total_samples}")
    
    def _log_advanced_training_progress(self, training_data: Dict):
        """Log progress of advanced preprocessing training"""
        logger.info("Advanced preprocessing progress:")
        total_samples = 0
        
        for anomaly_type, data in training_data.items():
            if not data['features']:
                continue
                
            feature_count = len(data['features'][0]) if data['features'] else 0
            positive_count = sum(data['labels'])
            total_count = len(data['labels'])
            positive_ratio = positive_count / total_count if total_count > 0 else 0
            total_samples += total_count
            
            logger.info(f"  {anomaly_type}: {total_count} samples, {feature_count} features, "
                       f"{positive_ratio:.3f} positive ratio")
            
            # Log feature statistics for debugging
            if data['features']:
                features_array = np.array(data['features'])
                feature_means = np.mean(features_array, axis=0)
                feature_stds = np.std(features_array, axis=0)
                zero_variance_features = np.sum(feature_stds == 0)
                
                logger.debug(f"  {anomaly_type} feature stats: "
                           f"mean_range=[{np.min(feature_means):.3f}, {np.max(feature_means):.3f}], "
                           f"std_range=[{np.min(feature_stds):.3f}, {np.max(feature_stds):.3f}], "
                           f"zero_variance={zero_variance_features}")
        
        logger.info(f"Total training samples across all models: {total_samples}")
    
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
    
    def evaluate_svm_models(self) -> Dict[str, Any]:
        """Evaluate SVM models using reserved evaluation data (tt.txt compliant)"""
        logger.info("Starting SVM model evaluation with tt.txt compliant data splitting")
        
        # Load evaluation data metadata
        eval_metadata_path = os.path.join(self.model_dir, "evaluation_data_metadata.json")
        if not os.path.exists(eval_metadata_path):
            raise RuntimeError("No evaluation data found. Please train models first to generate evaluation data split.")
        
        with open(eval_metadata_path, 'r', encoding='utf-8') as f:
            eval_metadata = json.load(f)
        
        eval_files = eval_metadata["evaluation_file_paths"]
        logger.info(f"Using {len(eval_files)} evaluation files: {[os.path.basename(f) for f in eval_files]}")
        
        # Initialize evaluation results
        evaluation_results = {
            'epcFake': {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0},
            'epcDup': {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0},
            'locErr': {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0},
            'evtOrderErr': {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0},
            'jump': {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0}
        }
        
        total_samples = 0
        processed_samples = 0
        
        # Process evaluation data chunks
        for chunk_data, chunk_name in tqdm(
            self.processor.process_csv_for_training(eval_files, self.chunk_size),
            desc="Evaluating SVM models on reserved test data"
        ):
            try:
                # Convert chunk to JSON for rule-based truth labels
                json_str = json.dumps(chunk_data)
                
                # Get ground truth from rule-based detection
                rule_results = detect_anomalies_backend_format(json_str)
                ground_truth = json.loads(rule_results)
                
                # Get SVM predictions 
                svm_results = self.detector.predict_anomalies(json_str)
                svm_predictions = json.loads(svm_results)
                
                # Compare SVM predictions with rule-based ground truth
                self._compare_svm_vs_ground_truth(ground_truth, svm_predictions, evaluation_results)
                
                total_samples += len(chunk_data.get('data', []))
                processed_samples += 1
                
                # Limit evaluation to prevent long runtime
                if processed_samples >= 5:  # Limit to 5 chunks for faster evaluation
                    logger.info(f"Evaluated {processed_samples} chunks, stopping for performance")
                    break
                    
            except Exception as e:
                logger.error(f"Error evaluating chunk {chunk_name}: {e}")
                continue
        
        # Calculate evaluation metrics
        final_metrics = self._calculate_evaluation_metrics(evaluation_results, total_samples)
        
        # Save evaluation results
        self._save_evaluation_results(final_metrics, eval_metadata)
        
        logger.info(f"SVM evaluation completed: {processed_samples} chunks, {total_samples} samples")
        return final_metrics
    
    def _compare_svm_vs_ground_truth(self, ground_truth: Dict, svm_predictions: Dict, results: Dict):
        """Compare SVM predictions against rule-based ground truth"""
        # Extract event histories for comparison
        gt_events = {event['eventId']: event for event in ground_truth.get('EventHistory', [])}
        svm_events = {event['eventId']: event for event in svm_predictions.get('EventHistory', [])}
        
        # Compare each anomaly type
        for event_id in gt_events.keys():
            if event_id not in svm_events:
                continue
                
            gt_event = gt_events[event_id]
            svm_event = svm_events[event_id]
            
            # Check each anomaly type
            for anomaly_type in ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']:
                gt_anomaly = gt_event.get(anomaly_type, False)
                svm_anomaly = svm_event.get(anomaly_type, False)
                
                # Update confusion matrix
                if gt_anomaly and svm_anomaly:
                    results[anomaly_type]['true_positives'] += 1
                elif not gt_anomaly and not svm_anomaly:
                    results[anomaly_type]['true_negatives'] += 1
                elif not gt_anomaly and svm_anomaly:
                    results[anomaly_type]['false_positives'] += 1
                elif gt_anomaly and not svm_anomaly:
                    results[anomaly_type]['false_negatives'] += 1
    
    def _calculate_evaluation_metrics(self, results: Dict, total_samples: int) -> Dict[str, Any]:
        """Calculate precision, recall, F1-score for each anomaly type"""
        metrics = {}
        
        for anomaly_type, confusion_matrix in results.items():
            tp = confusion_matrix['true_positives']
            fp = confusion_matrix['false_positives']
            tn = confusion_matrix['true_negatives']
            fn = confusion_matrix['false_negatives']
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            metrics[anomaly_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'confusion_matrix': confusion_matrix,
                'support': tp + fn  # Number of actual anomalies
            }
        
        # Add overall metrics
        metrics['_summary'] = {
            'total_samples_evaluated': total_samples,
            'evaluation_method': 'svm_vs_rule_based_ground_truth',
            'evaluation_date': datetime.now().isoformat(),
            'compliance': 'tt.txt_train_eval_separation'
        }
        
        return metrics
    
    def _save_evaluation_results(self, metrics: Dict, eval_metadata: Dict):
        """Save evaluation results to file"""
        evaluation_report = {
            'evaluation_metadata': eval_metadata,
            'evaluation_metrics': metrics,
            'model_performance': {}
        }
        
        # Extract key performance indicators
        for anomaly_type, metric in metrics.items():
            if anomaly_type.startswith('_'):
                continue
            evaluation_report['model_performance'][f'{anomaly_type}_svm'] = {
                'f1_score': metric['f1_score'],
                'precision': metric['precision'],
                'recall': metric['recall'],
                'accuracy': metric['accuracy']
            }
        
        # Save results
        results_path = os.path.join(self.model_dir, "svm_evaluation_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved: {results_path}")
        
        # Log summary to console
        logger.info("SVM Model Evaluation Summary (tt.txt compliant):")
        for anomaly_type, metric in metrics.items():
            if anomaly_type.startswith('_'):
                continue
            logger.info(f"  {anomaly_type}: F1={metric['f1_score']:.3f}, Precision={metric['precision']:.3f}, Recall={metric['recall']:.3f}")
    
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


def main():
    """Main training script"""
    print("SVM CSV Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SVMCSVTrainer()
    
    try:
        # Train models
        results = trainer.train_models_from_csv()
        
        print("\n" + "=" * 50)
        print("✅ Training Completed Successfully!")
        print(f"Duration: {results['training_duration_minutes']:.1f} minutes")
        print(f"Models trained: {len(results['training_results'])}")
        
        # Print model performance
        print("\nModel Performance:")
        for model_name, metrics in results['training_results'].items():
            print(f"  {model_name}:")
            print(f"    - Accuracy: {metrics['accuracy']:.3f}")
            print(f"    - Training samples: {metrics['normal_samples']}")
            print(f"    - Features: {metrics['feature_dimensions']}")
        
        print(f"\nModels saved to: models/svm_models/")
        print("Ready for API usage! 🎉")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    main()