"""
Batch Processor for Memory-Efficient SVM Preprocessing
Handles large datasets that don't fit in memory
"""

import os
import gc
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Generator, Optional, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

from .base_preprocessor import BasePreprocessor
from .feature_extractors.epc_fake_features import EPCFakeFeatureExtractor
from .feature_extractors.epc_dup_features import EPCDupFeatureExtractor
from .feature_extractors.evt_order_features import EventOrderFeatureExtractor
from .feature_extractors.loc_err_features import LocationErrorFeatureExtractor
from .feature_extractors.jump_features import JumpFeatureExtractor
from .label_generators.rule_based_labels import RuleBasedLabelGenerator


class BatchProcessor:
    """Memory-efficient batch processing for large datasets"""
    
    def __init__(self, batch_size: int = 10000, 
                 temp_dir: Optional[str] = None,
                 max_memory_usage: float = 0.8,
                 parallel_processing: bool = False,
                 max_workers: int = 4):
        
        self.batch_size = batch_size
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_memory_usage = max_memory_usage  # Max 80% of available memory
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        
        self.logger = logging.getLogger(__name__)
        
        # Track temporary files for cleanup
        self.temp_files = []
        
        # Memory monitoring
        self.process = psutil.Process()
    
    def process_large_dataset(self, df: pd.DataFrame, 
                            anomaly_types: List[str],
                            base_preprocessor: BasePreprocessor,
                            feature_extractors: Dict[str, Any],
                            label_generator: RuleBasedLabelGenerator) -> Dict[str, Dict[str, Any]]:
        """
        Process large dataset in memory-efficient batches
        
        Returns processed data for each anomaly type
        """
        
        self.logger.info(f"Starting batch processing for {len(df)} records")
        self.logger.info(f"Batch size: {self.batch_size}, Temp dir: {self.temp_dir}")
        
        # Clean and group data
        df_clean = base_preprocessor.preprocess(df)
        epc_groups = base_preprocessor.get_epc_groups(df_clean)
        
        total_epcs = len(epc_groups)
        self.logger.info(f"Processing {total_epcs} EPC groups")
        
        # Process in batches
        batch_results = {}
        for anomaly_type in anomaly_types:
            self.logger.info(f"Processing anomaly type: {anomaly_type}")
            
            batch_results[anomaly_type] = self._process_anomaly_type_batched(
                epc_groups, anomaly_type, feature_extractors[anomaly_type], label_generator
            )
        
        # Cleanup temporary files
        self._cleanup_temp_files()
        
        return batch_results
    
    def _process_anomaly_type_batched(self, epc_groups: Dict[str, pd.DataFrame],
                                    anomaly_type: str, 
                                    feature_extractor: Any,
                                    label_generator: RuleBasedLabelGenerator) -> Dict[str, Any]:
        """Process single anomaly type in batches"""
        
        epc_items = list(epc_groups.items())
        total_batches = (len(epc_items) + self.batch_size - 1) // self.batch_size
        
        batch_files = {
            'features': [],
            'labels': [],
            'scores': [],
            'epc_codes': []
        }
        
        # Process each batch
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(epc_items))
            
            batch_epc_groups = dict(epc_items[start_idx:end_idx])
            
            self.logger.debug(f"Processing batch {batch_idx + 1}/{total_batches} "
                            f"({len(batch_epc_groups)} EPCs)")
            
            # Monitor memory usage
            self._check_memory_usage()
            
            # Process batch
            if self.parallel_processing and len(batch_epc_groups) > 100:
                batch_result = self._process_batch_parallel(
                    batch_epc_groups, anomaly_type, feature_extractor, label_generator
                )
            else:
                batch_result = self._process_batch_sequential(
                    batch_epc_groups, anomaly_type, feature_extractor, label_generator
                )
            
            # Save batch to temporary files
            batch_file_paths = self._save_batch_to_temp(batch_result, anomaly_type, batch_idx)
            
            for key, path in batch_file_paths.items():
                batch_files[key].append(path)
            
            # Force garbage collection
            del batch_result, batch_epc_groups
            gc.collect()
        
        # Merge all batches
        final_result = self._merge_batches(batch_files, anomaly_type)
        
        return final_result
    
    def _process_batch_sequential(self, batch_epc_groups: Dict[str, pd.DataFrame],
                                anomaly_type: str, feature_extractor: Any,
                                label_generator: RuleBasedLabelGenerator) -> Dict[str, List]:
        """Process batch sequentially"""
        
        features = []
        labels = []
        scores = []
        epc_codes = []
        
        for epc_code, epc_group in batch_epc_groups.items():
            # Extract features
            if anomaly_type == 'epcFake':
                feature_vector = feature_extractor.extract_features(epc_code)
            else:
                feature_vector = feature_extractor.extract_features(epc_group)
            
            features.append(feature_vector)
            epc_codes.append(epc_code)
        
        # Generate labels for this batch
        batch_labels, batch_scores, _ = label_generator.generate_labels(
            batch_epc_groups, anomaly_type
        )
        
        labels.extend(batch_labels)
        scores.extend(batch_scores)
        
        return {
            'features': features,
            'labels': labels,
            'scores': scores,
            'epc_codes': epc_codes
        }
    
    def _process_batch_parallel(self, batch_epc_groups: Dict[str, pd.DataFrame],
                              anomaly_type: str, feature_extractor: Any,
                              label_generator: RuleBasedLabelGenerator) -> Dict[str, List]:
        """Process batch using parallel processing"""
        
        # Split batch into smaller chunks for parallel processing
        epc_items = list(batch_epc_groups.items())
        chunk_size = max(1, len(epc_items) // self.max_workers)
        chunks = [epc_items[i:i + chunk_size] for i in range(0, len(epc_items), chunk_size)]
        
        all_features = []
        all_labels = []
        all_scores = []
        all_epc_codes = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._process_chunk, 
                    dict(chunk), anomaly_type, feature_extractor, label_generator
                ): chunk_idx
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    all_features.extend(chunk_result['features'])
                    all_labels.extend(chunk_result['labels'])
                    all_scores.extend(chunk_result['scores'])
                    all_epc_codes.extend(chunk_result['epc_codes'])
                    
                except Exception as exc:
                    self.logger.error(f"Chunk {chunk_idx} generated exception: {exc}")
                    # Continue with other chunks
        
        return {
            'features': all_features,
            'labels': all_labels,
            'scores': all_scores,
            'epc_codes': all_epc_codes
        }
    
    @staticmethod
    def _process_chunk(chunk_epc_groups: Dict[str, pd.DataFrame],
                      anomaly_type: str, feature_extractor: Any,
                      label_generator: RuleBasedLabelGenerator) -> Dict[str, List]:
        """Process a single chunk (used in parallel processing)"""
        
        features = []
        epc_codes = []
        
        for epc_code, epc_group in chunk_epc_groups.items():
            if anomaly_type == 'epcFake':
                feature_vector = feature_extractor.extract_features(epc_code)
            else:
                feature_vector = feature_extractor.extract_features(epc_group)
            
            features.append(feature_vector)
            epc_codes.append(epc_code)
        
        # Generate labels
        labels, scores, _ = label_generator.generate_labels(chunk_epc_groups, anomaly_type)
        
        return {
            'features': features,
            'labels': labels,
            'scores': scores,
            'epc_codes': epc_codes
        }
    
    def _save_batch_to_temp(self, batch_result: Dict[str, List],
                          anomaly_type: str, batch_idx: int) -> Dict[str, str]:
        """Save batch result to temporary files"""
        
        batch_files = {}
        
        for data_type, data in batch_result.items():
            if data_type == 'features':
                # Convert to numpy array for efficient storage
                data_array = np.array(data)
            else:
                data_array = np.array(data)
            
            # Create temporary file
            temp_file = os.path.join(
                self.temp_dir, 
                f"svm_batch_{anomaly_type}_{data_type}_{batch_idx}.npy"
            )
            
            np.save(temp_file, data_array)
            batch_files[data_type] = temp_file
            self.temp_files.append(temp_file)
        
        return batch_files
    
    def _merge_batches(self, batch_files: Dict[str, List[str]], 
                      anomaly_type: str) -> Dict[str, Any]:
        """Merge all batch files into final result"""
        
        self.logger.info(f"Merging {len(batch_files['features'])} batches for {anomaly_type}")
        
        merged_data = {}
        
        for data_type, file_paths in batch_files.items():
            if not file_paths:
                merged_data[data_type] = []
                continue
            
            # Load and concatenate all batch files
            all_data = []
            for file_path in file_paths:
                batch_data = np.load(file_path)
                all_data.append(batch_data)
            
            if data_type == 'features':
                # Features need to be stacked as 2D array
                merged_array = np.vstack(all_data)
            else:
                # Labels, scores, epc_codes can be concatenated
                merged_array = np.concatenate(all_data)
            
            merged_data[data_type] = merged_array
        
        # Generate summary
        total_samples = len(merged_data.get('labels', []))
        positive_samples = int(np.sum(merged_data.get('labels', [])))
        
        result = {
            'features': merged_data['features'],
            'labels': merged_data['labels'].tolist(),
            'scores': merged_data['scores'].tolist(),
            'epc_codes': merged_data['epc_codes'].tolist(),
            'summary': {
                'total_samples': total_samples,
                'positive_samples': positive_samples,
                'negative_samples': total_samples - positive_samples,
                'positive_ratio': positive_samples / total_samples if total_samples > 0 else 0.0,
                'feature_dimensions': merged_data['features'].shape[1] if len(merged_data['features']) > 0 else 0,
                'processing_method': 'batch'
            }
        }
        
        return result
    
    def _check_memory_usage(self):
        """Monitor memory usage and warn if approaching limits"""
        memory_info = self.process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        
        # Get system memory
        system_memory = psutil.virtual_memory()
        memory_percent = memory_info.rss / system_memory.total
        
        if memory_percent > self.max_memory_usage:
            self.logger.warning(
                f"High memory usage: {memory_usage_mb:.1f}MB "
                f"({memory_percent*100:.1f}% of system memory)"
            )
            
            # Force garbage collection
            gc.collect()
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        cleaned_count = 0
        
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        self.logger.info(f"Cleaned up {cleaned_count} temporary files")
        self.temp_files.clear()
    
    def estimate_memory_requirements(self, num_epcs: int, 
                                   feature_dimensions: Dict[str, int]) -> Dict[str, float]:
        """Estimate memory requirements for processing"""
        
        # Estimate memory per EPC (in MB)
        base_memory_per_epc = 0.001  # Base overhead
        
        memory_estimates = {}
        
        for anomaly_type, dimensions in feature_dimensions.items():
            # Features: float64 * dimensions * num_epcs
            feature_memory = (8 * dimensions * num_epcs) / (1024 * 1024)
            
            # Labels and scores: int32 + float64 * num_epcs  
            label_memory = (4 + 8) * num_epcs / (1024 * 1024)
            
            # EPC codes: assume average 50 chars per EPC code
            epc_memory = (50 * num_epcs) / (1024 * 1024)
            
            total_memory = feature_memory + label_memory + epc_memory + (base_memory_per_epc * num_epcs)
            
            memory_estimates[anomaly_type] = {
                'feature_memory_mb': feature_memory,
                'label_memory_mb': label_memory,
                'epc_memory_mb': epc_memory,
                'total_memory_mb': total_memory
            }
        
        # Overall estimate
        total_all_types = sum(est['total_memory_mb'] for est in memory_estimates.values())
        
        memory_estimates['overall'] = {
            'total_memory_all_types_mb': total_all_types,
            'recommended_batch_size': self._recommend_batch_size(num_epcs, total_all_types),
            'system_memory_mb': psutil.virtual_memory().total / (1024 * 1024),
            'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
        
        return memory_estimates
    
    def _recommend_batch_size(self, num_epcs: int, estimated_memory_mb: float) -> int:
        """Recommend optimal batch size based on memory constraints"""
        
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        target_memory_mb = available_memory_mb * self.max_memory_usage
        
        if estimated_memory_mb <= target_memory_mb:
            # Can process all at once
            return num_epcs
        
        # Calculate batch size to fit in target memory
        ratio = target_memory_mb / estimated_memory_mb
        recommended_batch_size = max(100, int(num_epcs * ratio))
        
        return min(recommended_batch_size, num_epcs)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing"""
        return {
            'batch_size': self.batch_size,
            'temp_dir': self.temp_dir,
            'max_memory_usage': self.max_memory_usage,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'temp_files_count': len(self.temp_files),
            'current_memory_usage_mb': self.process.memory_info().rss / (1024 * 1024),
            'system_memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'system_memory_available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }